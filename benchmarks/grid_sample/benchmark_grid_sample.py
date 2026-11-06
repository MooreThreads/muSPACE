#!/usr/bin/env python3
"""Benchmark grid_sample on MUSA and CUDA reference devices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch

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
from grid_sample_case import BACKWARD, DEFAULT_CASE, case_ids, make_named_case

OPERATOR = "grid_sample"
SCOPE = "operator"
BACKEND = "musa"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        cases=case_ids(),
        default_case=DEFAULT_CASE,
        dtypes=("float32", "float16", "bfloat16"),
        devices=("musa", "cuda"),
    )
    return parser.parse_args(argv)


def build_launch(case: dict, tensors: dict[str, torch.Tensor]) -> Callable[[], None]:
    from mu_space.ops import grid_sample

    value = tensors["input"]
    grid = tensors["grid"]

    def forward() -> torch.Tensor:
        return grid_sample(
            value,
            grid,
            case["mode"],
            case["padding_mode"],
            case["align_corners"],
            backend="musa" if value.device.type == "musa" else "reference",
        )

    if case["direction"] != BACKWARD:
        return forward

    def backward() -> None:
        value.grad = None
        grid.grad = None
        forward().square().sum().backward()

    return backward


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device_module(args.device)
    case = make_named_case(args.case, seed=args.seed)
    dtype = getattr(torch, args.dtype)
    tensors = {
        name: tensor.to(device=args.device, dtype=dtype).requires_grad_(True)
        for name, tensor in case["inputs"].items()
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
            "input_shape": list(case["input_shape"]),
            "grid_shape": list(case["grid_shape"]),
            "mode": case["mode"],
            "padding_mode": case["padding_mode"],
            "align_corners": case["align_corners"],
            "backend": BACKEND if args.device == "musa" else "reference",
        },
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
