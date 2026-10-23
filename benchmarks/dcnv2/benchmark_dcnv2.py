#!/usr/bin/env python3
"""Benchmark modulated deformable convolution (DCNv2) on the measured device.

Two native paths are measured:

``pack``
    The production path: offset convolution through muDNN, the fused NHWC deformable
    gather, and the main GEMM on channel-last columns.
``general``
    The general NCHW path: offset convolution and sigmoid in PyTorch, then
    ``mu_space.ops.modulated_deform_conv2d``.

Alongside latency each record reports the arithmetic intensity of the operator and the
bandwidth it implies.  ``effective_gbps`` comes from a traffic model of the implemented
dataflow, documented in :func:`traffic_terms`: it charges each tensor once per pass that
touches it and ignores cache reuse in the four-corner gather, so it is a lower bound on
real DRAM traffic.  Use hardware counters when a measured bandwidth is required.

Measurement protocol, record format and table rules live in ``docs/benchmarking.md``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from benchmarks._harness import (
    add_common_args,
    device_module,
    make_record,
    measure,
    print_summary,
    resolve_output,
)
from dcnv2_bevformer_case import (
    BEVFORMER_CASES,
    CASES,
    FORMAT as SYNTHETIC_FORMAT,
    make_case,
)
from mu_space.ops import modulated_deform_conv2d, modulated_deform_conv2d_pack

DTYPES = {"float32": torch.float32, "float16": torch.float16}
PATHS = ("pack", "general")
INPUT_NAMES = ("input", "offset_weight", "offset_bias", "weight", "bias")
FORWARD_SCOPE = "operator"
BACKWARD_SCOPE = "forward-backward"


def load_case(path: Path) -> dict:
    """Load a captured input artifact.

    Captured artifacts are a local investigation path only: they are never committed and
    they never produce a published row, which always comes from the fixed-seed case.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") not in {"modulated_deform_conv_v1", SYNTHETIC_FORMAT}:
        raise ValueError(f"unsupported DCNv2 benchmark artifact: {path.name}")
    missing = set(INPUT_NAMES).difference(payload.get("inputs", {}))
    if missing:
        raise ValueError(f"{path.name} is missing inputs: {sorted(missing)}")
    return payload


def move_inputs(
    payload: dict, dtype: torch.dtype, channels_last: bool, device: str
) -> tuple[torch.Tensor, ...]:
    tensors = []
    for name in INPUT_NAMES:
        tensor = payload["inputs"][name].to(device=device, dtype=dtype)
        if name == "input" and channels_last:
            tensor = tensor.to(memory_format=torch.channels_last)
        tensors.append(tensor.requires_grad_(True))
    return tuple(tensors)


def run_once(path: str, tensors: tuple[torch.Tensor, ...], scalars: dict) -> torch.Tensor:
    x, offset_weight, offset_bias, weight, bias = tensors
    stride, padding, dilation = scalars["stride"], scalars["padding"], scalars["dilation"]
    if path == "pack":
        return modulated_deform_conv2d_pack(
            x, offset_weight, offset_bias, weight, bias, stride, padding, dilation
        )
    raw = F.conv2d(x, offset_weight, offset_bias, stride, padding, dilation)
    # First 2 * dg * kH * kW channels are the offsets, the rest the pre-sigmoid mask.
    offset_channels = 2 * scalars["deform_groups"] * scalars["kernel_size"] ** 2
    return modulated_deform_conv2d(
        x,
        raw[:, :offset_channels],
        torch.sigmoid(raw[:, offset_channels:]),
        weight,
        bias,
        stride,
        padding,
        dilation,
        backend="musa",
    )


def make_runner(
    path: str, tensors: tuple[torch.Tensor, ...], scalars: dict, include_backward: bool
) -> Callable[[], object]:
    def run() -> object:
        output = run_once(path, tensors, scalars)
        if include_backward:
            output.sum().backward()
            for tensor in tensors:
                tensor.grad = None
        return output

    return run


def gemm_flops(scalars: dict, include_backward: bool) -> float:
    """Multiply-accumulate FLOPs of the two convolutions in the operator."""
    taps = scalars["kernel_size"] ** 2
    positions = scalars["batch"] * scalars["height"] * scalars["width"]
    main = 2.0 * positions * scalars["channels_out"] * scalars["channels"] * taps
    offset = 2.0 * positions * 3 * taps * scalars["channels"] * taps
    forward = main + offset
    # Backward runs a data gradient and a weight gradient for both convolutions.
    return forward * 3.0 if include_backward else forward


def traffic_terms(scalars: dict, path: str, include_backward: bool) -> dict[str, float]:
    """Compulsory HBM traffic of the implemented dataflow, in elements.

    Each tensor is counted once per pass that touches it, and an atomic accumulation
    counts as a read plus a write. Cache behaviour is not modelled, so the four-corner
    gather is charged for reading the input once rather than four times.

    ``x``       input feature map, ``B*H*W*C``
    ``y``       raw offset-conv output, ``B*Ho*Wo*27``
    ``columns`` deformable column matrix, ``B*Ho*Wo*9*C`` -- the dominant term
    ``out``     operator output, ``B*Ho*Wo*Cout``

    The backward model assumes the default ``MUSPACE_DCN_RECOMPUTE_COLUMNS=1``.
    """
    taps = scalars["kernel_size"] ** 2
    positions = scalars["batch"] * scalars["height"] * scalars["width"]
    terms = {
        "x": float(positions * scalars["channels"]),
        "y": float(positions * 3 * taps),
        "columns": float(positions * taps * scalars["channels"]),
        "out": float(positions * scalars["channels_out"]),
    }
    if path == "pack":
        # forward: conv reads x writes y; gather reads x and y, writes columns;
        # GEMM reads columns, writes out.
        weights = {"x": 2, "y": 2, "columns": 2, "out": 1}
        if include_backward:
            # re-gather, dCol GEMM, dW GEMM, col2im (atomic grad_input), offset-conv
            # backward for data and weights, and the grad_input sum.
            weights = {"x": 11, "y": 7, "columns": 6, "out": 3}
    else:
        # The general path additionally materialises offset and mask from y and
        # transposes the output out of the channel-major GEMM layout.
        weights = {"x": 2, "y": 4, "columns": 2, "out": 3}
        if include_backward:
            weights = {"x": 10, "y": 10, "columns": 6, "out": 7}
    return {name: terms[name] * weight for name, weight in weights.items()}


def relative_difference(tensors: tuple[torch.Tensor, ...], scalars: dict) -> float:
    """Largest relative difference between the two native paths."""
    pack = run_once("pack", tensors, scalars).detach().float()
    general = run_once("general", tensors, scalars).detach().float()
    return ((pack - general).abs().max() / general.abs().max().clamp_min(1e-12)).item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, dtypes=tuple(DTYPES))
    parser.add_argument(
        "--case",
        choices=tuple(CASES),
        action="append",
        help="repeatable; defaults to the published BEVFormer image-backbone shapes",
    )
    parser.add_argument("--path", choices=PATHS, action="append", help="repeatable")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--include-backward", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="compare the two native paths before timing"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="captured input artifact to replay locally; published rows use the fixed-seed case",
    )
    return parser.parse_args()


def record_path(base: Path, case: str, path: str, total: int) -> Path:
    """Return the record path of one measurement.

    ``--output`` names the record directly when the run publishes a single row; a run that
    selects several cases or paths writes one record per row next to it, without repeating a
    case name the record path already carries.
    """
    if total == 1:
        return base
    stem = base.stem if base.stem.endswith(case) else f"{base.stem}-{case}"
    return base.with_name(f"{stem}-{path}{base.suffix}")


def main() -> int:
    args = parse_args()
    module = device_module(args.device)
    dtype = DTYPES[args.dtype]
    paths = args.path or list(PATHS)
    if args.input_file is None:
        payloads = [(name, make_case(name, args.seed)) for name in (args.case or BEVFORMER_CASES)]
    else:
        payload = load_case(args.input_file)
        payloads = [(str(payload.get("case") or args.input_file.stem), payload)]
    scope = BACKWARD_SCOPE if args.include_backward else FORWARD_SCOPE
    output = resolve_output(args, "dcnv2")
    total = len(payloads) * len(paths)
    for case, payload in payloads:
        scalars = payload["scalars"]
        tensors = move_inputs(payload, dtype, args.channels_last, args.device)
        difference = relative_difference(tensors, scalars) if args.check else None
        for path in paths:
            module.reset_peak_memory_stats()
            measurement = measure(
                make_runner(path, tensors, scalars, args.include_backward),
                device=args.device,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            traffic = traffic_terms(scalars, path, args.include_backward)
            traffic_bytes = sum(traffic.values()) * torch.empty((), dtype=dtype).element_size()
            flops = gemm_flops(scalars, args.include_backward)
            record = make_record(
                operator="modulated_deform_conv2d",
                case=case,
                dtype=args.dtype,
                scope=scope,
                device=args.device,
                seed=int(payload.get("seed", args.seed)),
                measurement=measurement,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
                derived={
                    "path": path,
                    "channels_last": bool(args.channels_last),
                    "include_backward": bool(args.include_backward),
                    "fp32_gemm_tf32": os.environ.get("MUSPACE_DCN_FP32_TF32", "0"),
                    "input_shape": list(tensors[0].shape),
                    "tflops": flops / (measurement.p50_ms * 1e-3) / 1e12,
                    "effective_gbps": traffic_bytes / (measurement.p50_ms * 1e-3) / 1e9,
                    "modeled_traffic_gib": traffic_bytes / 2**30,
                    "arithmetic_intensity": flops / traffic_bytes,
                    "peak_memory_gib": module.max_memory_allocated() / 2**30,
                    "pack_vs_general_max_relative_difference": difference,
                    "input_file": args.input_file.name if args.input_file else None,
                },
            )
            record.save(record_path(output, case, path, total))
            print_summary(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
