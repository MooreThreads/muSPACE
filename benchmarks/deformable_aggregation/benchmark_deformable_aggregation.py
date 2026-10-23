#!/usr/bin/env python3
"""Benchmark SparseDrive deformable aggregation (opt) on MUSA.

The published protocol and the record format live in ``benchmarks/_harness.py`` and
``docs/benchmarking.md``.  Forward and backward are separate entry points, so they are
measured and recorded as separate cases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch

# Allow `python benchmarks/deformable_aggregation/benchmark_*.py` without PYTHONPATH
# manipulation beyond the repo root for `mu_space`, while keeping the harness and the
# local case module importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import add_common_args, make_record, measure, print_summary, resolve_output
from deformable_aggregation_case import (
    BACKWARD_CASE,
    CASES,
    DEFAULT_CASE,
    DUMP_FORMAT,
    FORMAT,
    FORWARD_CASE,
    TENSOR_NAMES,
    make_case,
)

OPERATOR = "deformable_aggregation"
SCOPE = "kernel-only"
IMPLEMENTATION = "mu_space_opt"
ENTRY_POINTS = {
    FORWARD_CASE: "deformable_aggregation_forward_opt",
    BACKWARD_CASE: "deformable_aggregation_backward_opt",
}


def load_case(path: Path) -> dict:
    """Load a captured artifact for local investigation."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") not in {FORMAT, DUMP_FORMAT}:
        raise ValueError(f"Unsupported deformable aggregation benchmark artifact: {path}")
    for group, extra in (("forward_inputs", set()), ("backward_inputs", {"grad_output"})):
        missing = set(TENSOR_NAMES).union(extra).difference(payload.get(group, {}))
        if missing:
            raise ValueError(f"{path} is missing {group}: {sorted(missing)}")
    return payload


def to_device(payload: dict, device: str) -> dict[str, dict[str, torch.Tensor]]:
    """Move the case tensors to ``device`` without touching dtype or layout."""
    return {
        group: {name: tensor.to(device) for name, tensor in payload[group].items()}
        for group in ("forward_inputs", "backward_inputs")
    }


def input_group(case: str) -> str:
    return "forward_inputs" if case == FORWARD_CASE else "backward_inputs"


def build_launch(native, case: str, inputs: dict[str, torch.Tensor]) -> Callable[[], None]:
    """Return the timed launcher: the native entry point, with buffers preallocated."""
    args = tuple(inputs[name] for name in TENSOR_NAMES)
    if case == FORWARD_CASE:
        output = args[0].new_zeros((args[0].size(0), args[3].size(1), args[0].size(2)))

        def launch() -> None:
            native.deformable_aggregation_forward_opt(*args, output)

        return launch

    grad_output = inputs["grad_output"]
    grad_feat = torch.zeros_like(args[0])
    grad_loc = torch.zeros_like(args[3])
    grad_weights = torch.zeros_like(args[4])

    def launch() -> None:
        native.deformable_aggregation_backward_opt(
            *args, grad_output, grad_feat, grad_loc, grad_weights
        )

    return launch


def describe(case: str, inputs: dict[str, torch.Tensor], payload: dict, capture: bool) -> dict:
    """Return the case properties recorded next to the latency."""
    args = tuple(inputs[name] for name in TENSOR_NAMES)
    derived = {
        "implementation": IMPLEMENTATION,
        "entry_point": ENTRY_POINTS[case],
        "captured_input": capture,
        "mc_ms_feat_shape": list(args[0].shape),
        "spatial_shape": list(args[1].shape),
        "scale_start_index_shape": list(args[2].shape),
        "sampling_location_shape": list(args[3].shape),
        "weights_shape": list(args[4].shape),
    }
    active_frac = (payload.get("scalars") or {}).get("active_frac")
    if active_frac is not None:
        derived["active_frac"] = active_frac
    if case == BACKWARD_CASE:
        derived["grad_output_shape"] = list(inputs["grad_output"].shape)
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="captured forward/backward artifact for local investigation; never published",
    )
    add_common_args(parser, cases=CASES, default_case=DEFAULT_CASE, dtypes=("float32",))
    args = parser.parse_args()

    payload = make_case(args.seed) if args.input_file is None else load_case(args.input_file)
    inputs = to_device(payload, args.device)[input_group(args.case)]

    from mu_space.runtime.native import native_module_proxy

    launch = build_launch(native_module_proxy(), args.case, inputs)
    measurement = measure(
        launch,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    seed = payload.get("seed")
    record = make_record(
        operator=OPERATOR,
        case=args.case,
        dtype=args.dtype,
        scope=SCOPE,
        device=args.device,
        seed=args.seed if seed is None else int(seed),
        measurement=measurement,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        derived=describe(args.case, inputs, payload, args.input_file is not None),
    )
    record.save(resolve_output(args, OPERATOR))
    print_summary(record)


if __name__ == "__main__":
    main()
