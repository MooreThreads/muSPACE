#!/usr/bin/env python3
"""Benchmark the muSPACE BEVDepth voxel-pooling training operator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from _harness import (
    add_common_args,
    make_record,
    measure,
    print_summary,
    resolve_output,
)
from voxel_pooling_train_case import (
    DEFAULT_CASE,
    case_ids,
    make_named_case,
)

OPERATOR = "voxel_pooling_train"
#: The timed region is the complete training boundary: position-map creation, native
#: forward aggregation and the ``input_features`` backward.
SCOPE = "forward-backward"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cases=case_ids(), default_case=DEFAULT_CASE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = make_named_case(args.case, seed=args.seed)
    dtype = getattr(torch, args.dtype)
    tensors = {
        "geom_xyz": case["inputs"]["geom_xyz"].to(args.device),
        "input_features": case["inputs"]["input_features"].to(device=args.device, dtype=dtype),
        "grad_output": case["inputs"]["grad_output"].to(device=args.device, dtype=dtype),
    }
    voxel_num = case["voxel_num"]

    from mu_space import voxel_pooling_train

    def run() -> None:
        features = tensors["input_features"].detach().requires_grad_()
        output = voxel_pooling_train(tensors["geom_xyz"], features, voxel_num, backend="musa")
        output.backward(tensors["grad_output"])

    measurement = measure(
        run,
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
            "batch": int(tensors["geom_xyz"].shape[0]),
            "points": int(tensors["geom_xyz"].shape[1]),
            "channels": int(tensors["input_features"].shape[2]),
            "voxel_num": [int(value) for value in voxel_num],
        },
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
