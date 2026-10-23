#!/usr/bin/env python3
"""Benchmark the muSPACE BEVFusion sigmoid focal loss on MUSA and on the reference device.

The operator's reference implementation is a plain PyTorch expression, so the same runner and the
same case run on both devices: ``--device musa`` measures the native kernel and ``--device cuda``
measures the reference expression.  The published protocol and the record format live in
``benchmarks/_harness.py`` and ``docs/benchmarking.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch

# Allow `python benchmarks/sigmoid_focal_loss/benchmark_sigmoid_focal_loss.py` without PYTHONPATH
# manipulation beyond the repository root for `mu_space`, while keeping the harness and the local
# case module importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (
    add_common_args,
    device_module,
    make_record,
    measure,
    print_summary,
    resolve_output,
)
from sigmoid_focal_loss_case import DEFAULT_CASE, FORWARD, case_ids, make_named_case

OPERATOR = "sigmoid_focal_loss"
#: One direction of the operator boundary per row: the forward expression, or that expression plus
#: the backward pass it triggers.
SCOPE = "operator"
#: The native backend is selected by `auto` on a MUSA tensor and the PyTorch reference expression by
#: `auto` on any other device, so one call site covers both measured devices.
BACKEND = "auto"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        cases=case_ids(),
        default_case=DEFAULT_CASE,
        dtypes=("float32", "float16"),
        devices=("musa", "cuda"),
    )
    return parser.parse_args(argv)


def build_launch(case: dict, tensors: dict[str, torch.Tensor]) -> Callable[[], None]:
    """Return the timed launcher: the public entry point for the case's direction."""

    from mu_space.ops import sigmoid_focal_loss

    inputs = tensors["inputs"]
    targets = tensors["targets"]

    def forward() -> torch.Tensor:
        return sigmoid_focal_loss(
            inputs, targets, case["alpha"], case["gamma"], case["reduction"], backend=BACKEND
        )

    if case["direction"] == FORWARD:
        return forward

    def backward() -> None:
        inputs.grad = None
        forward().backward()

    return backward


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Fail before tensors are built when the requested device is absent, so a host that only has one
    # of the two stacks reports that plainly instead of failing inside a device transfer.
    device_module(args.device)
    case = make_named_case(args.case, seed=args.seed)
    dtype = getattr(torch, args.dtype)
    tensors = {
        "inputs": case["inputs"]["inputs"].to(device=args.device, dtype=dtype).requires_grad_(True),
        "targets": case["inputs"]["targets"].to(device=args.device, dtype=dtype),
    }
    measurement = measure(
        build_launch(case, tensors),
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    record = make_record(
        operator=OPERATOR,
        case=args.case,
        dtype=args.dtype,
        scope=SCOPE,
        device=args.device,
        seed=args.seed,
        measurement=measurement,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        derived={
            "direction": case["direction"],
            "shape": list(case["shape"]),
            "alpha": case["alpha"],
            "gamma": case["gamma"],
            "reduction": case["reduction"],
            "backend": BACKEND,
        },
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
