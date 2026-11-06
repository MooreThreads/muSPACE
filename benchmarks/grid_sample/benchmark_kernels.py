#!/usr/bin/env python3
"""Measure grid sampling device kernels separately from event-timed calls."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("musa", "cuda"), required=True)
    parser.add_argument("--input-shape", nargs="+", type=int, required=True)
    parser.add_argument("--grid-shape", nargs="+", type=int, required=True)
    parser.add_argument("--direction", choices=("forward", "backward"), required=True)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-calls", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats, args.profile_calls) <= 0:
        parser.error("measurement counts must be positive")
    if len(args.input_shape) not in (4, 5) or len(args.grid_shape) != len(args.input_shape):
        parser.error("input and grid must both be 4-D or both be 5-D")
    if min(*args.input_shape, *args.grid_shape) <= 0:
        parser.error("benchmark shapes must be nonempty")
    if (
        args.grid_shape[0] != args.input_shape[0]
        or args.grid_shape[-1] != len(args.input_shape) - 2
    ):
        parser.error("grid batch or coordinate dimension does not match input")
    return args


def main() -> None:
    args = parse_args()
    if args.device == "musa":
        import torch_musa  # noqa: F401

        from mu_space import grid_sample

        activity = torch.profiler.ProfilerActivity.PrivateUse1

        def forward(value, grid):
            return grid_sample(value, grid, backend="musa")

    else:
        activity = torch.profiler.ProfilerActivity.CUDA

        def forward(value, grid):
            return torch.nn.functional.grid_sample(value, grid, align_corners=False)

    device = getattr(torch, args.device)
    generator = torch.Generator().manual_seed(args.seed)
    value = torch.randn(args.input_shape, generator=generator).to(args.device).requires_grad_()
    grid = (
        (torch.rand(args.grid_shape, generator=generator) * 2 - 1).to(args.device).requires_grad_()
    )
    if args.direction == "backward":
        output = forward(value, grid)
        grad_output = torch.ones_like(output)

        def call():
            return torch.autograd.grad(output, (value, grid), grad_output, retain_graph=True)

    else:

        def call():
            return forward(value, grid)

    for _ in range(args.warmup):
        call()
    device.synchronize()
    event_times = []
    for _ in range(args.repeats):
        start, end = device.Event(enable_timing=True), device.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            call()
        end.record()
        device.synchronize()
        event_times.append(start.elapsed_time(end) / args.iterations)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, activity]) as prof:
        for _ in range(args.profile_calls):
            call()
        device.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace = args.output.with_suffix(".trace.json")
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    # Runtime launches are CPU events; only device kernel events count here.
    kernels = [event for event in events if event.get("cat") == "kernel"]
    sampling = [
        event
        for event in kernels
        if (
            "sample_" in event.get("name", "") and "kernel" in event.get("name", "")
            if args.device == "musa"
            else "grid_sampler" in event.get("name", "") and "kernel" in event.get("name", "")
        )
    ]
    if len(sampling) != args.profile_calls:
        raise RuntimeError(
            f"expected {args.profile_calls} sampling kernels, found {len(sampling)}; "
            "check device profiler availability and kernel names"
        )
    samples = [event["dur"] for event in sampling]
    auxiliary = [
        {"name": event["name"], "duration_us": event["dur"]}
        for event in events
        if event.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy") and event not in sampling
    ]
    record = {
        "operator": "grid_sample",
        "scope": "sampling_kernel",
        "device": args.device,
        "dtype": "float32",
        "input_shape": args.input_shape,
        "grid_shape": args.grid_shape,
        "direction": args.direction,
        "seed": args.seed,
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": False,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "profile_calls": args.profile_calls,
        "sampling_kernel_median_us": statistics.median(samples),
        "sampling_kernel_samples_us": samples,
        "event_call_median_ms": statistics.median(event_times),
        "event_call_repeats_ms": event_times,
        "auxiliary_events": auxiliary,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
