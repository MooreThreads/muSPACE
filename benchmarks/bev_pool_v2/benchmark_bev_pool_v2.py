#!/usr/bin/env python3
"""Benchmark the native BEVPoolV2 forward-and-backward path on MUSA.

The published protocol and the record format live in ``benchmarks/_harness.py`` and
``docs/benchmarking.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch

# Allow `python benchmarks/bev_pool_v2/benchmark_*.py` without PYTHONPATH manipulation
# beyond the repo root for `mu_space`, while keeping the harness and the local case
# module importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import add_common_args, make_record, measure, print_summary, resolve_output
from bev_pool_v2_case import CASES, DEFAULT_CASE, FORMAT, make_case

OPERATOR = "bev_pool_v2"
SCOPE = "forward-backward"
IMPLEMENTATION = "mu_space_native"


def load_case(path: Path) -> dict:
    """Load a captured case artifact for local investigation."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"Unsupported BEVPoolV2 benchmark artifact: {path}")
    required = {
        "depth",
        "feat",
        "ranks_depth",
        "ranks_feat",
        "ranks_bev",
        "interval_starts",
        "interval_lengths",
        "grad_output",
    }
    missing = required.difference(payload.get("backward_inputs", {}))
    if missing:
        raise ValueError(f"{path} is missing backward_inputs: {sorted(missing)}")
    return payload


def to_device(payload: dict, device: str) -> dict[str, torch.Tensor]:
    """Move the case tensors to ``device`` without touching dtype or layout."""
    return {name: tensor.to(device) for name, tensor in payload["backward_inputs"].items()}


def make_runner(payload: dict, tensors: dict[str, torch.Tensor]) -> Callable[[], torch.Tensor]:
    from mu_space import bev_pool_v2

    shape = tuple(payload["scalars"]["bev_feat_shape"])

    def run() -> torch.Tensor:
        depth = tensors["depth"].detach().requires_grad_(True)
        feat = tensors["feat"].detach().requires_grad_(True)
        output = bev_pool_v2(
            depth,
            feat,
            tensors["ranks_depth"],
            tensors["ranks_feat"],
            tensors["ranks_bev"],
            shape,
            tensors["interval_starts"],
            tensors["interval_lengths"],
            backend="musa",
        )
        output.backward(tensors["grad_output"])
        return output

    return run


def describe(payload: dict, tensors: dict[str, torch.Tensor], capture: bool) -> dict:
    """Return the case properties recorded next to the latency."""
    return {
        "implementation": IMPLEMENTATION,
        "captured_input": capture,
        "depth_shape": list(tensors["depth"].shape),
        "feat_shape": list(tensors["feat"].shape),
        "bev_feat_shape": list(payload["scalars"]["bev_feat_shape"]),
        "points": int(payload["scalars"]["points"]),
        "unique_ranks_depth": int(tensors["ranks_depth"].unique().numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="captured case artifact for local investigation; never published",
    )
    add_common_args(parser, cases=CASES, default_case=DEFAULT_CASE, dtypes=("float32",))
    args = parser.parse_args()

    payload = make_case(args.seed) if args.input_file is None else load_case(args.input_file)
    tensors = to_device(payload, args.device)
    measurement = measure(
        make_runner(payload, tensors),
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
        derived=describe(payload, tensors, args.input_file is not None),
    )
    record.save(resolve_output(args, OPERATOR))
    print_summary(record)


if __name__ == "__main__":
    main()
