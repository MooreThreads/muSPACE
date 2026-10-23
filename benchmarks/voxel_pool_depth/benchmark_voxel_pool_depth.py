#!/usr/bin/env python3
"""Benchmark the muSPACE depth-aware voxel pooling operator."""

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
from voxel_pool_depth_case import (
    DEFAULT_CASE,
    case_ids,
    make_named_case,
)

OPERATOR = "voxel_pool_depth"
#: The timed region is the complete training boundary: Python frustum compaction,
#: the atomic native forward fusion into the zeroed volume and the fused backward
#: pass for both ``img_feats`` and ``img_depth``.
SCOPE = "forward-backward"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        cases=case_ids(),
        default_case=DEFAULT_CASE,
        dtypes=("float32", "float16", "bfloat16"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = make_named_case(args.case, seed=args.seed)
    dtype = getattr(torch, args.dtype)
    tensors = {name: tensor.to(args.device) for name, tensor in case["inputs"].items()}
    for name in ("img_feats", "img_depth", "grad_output"):
        tensors[name] = tensors[name].to(dtype)
    scalars = case["scalars"]

    from mu_space import voxel_pool_depth

    def run() -> None:
        img_feats = tensors["img_feats"].detach().requires_grad_(True)
        img_depth = tensors["img_depth"].detach().requires_grad_(True)
        output = voxel_pool_depth(
            img_feats,
            img_depth,
            tensors["voxel_coords"],
            tensors["mask"],
            scalars["B"],
            scalars["X"],
            scalars["Y"],
            scalars["Z"],
            backend="musa",
        )
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
            "batch": int(tensors["img_feats"].shape[0]),
            "cameras": int(tensors["voxel_coords"].shape[2]),
            "channels": int(tensors["img_feats"].shape[2]),
            "depth_bins": int(tensors["img_depth"].shape[1]),
            "height": int(tensors["img_feats"].shape[3]),
            "width": int(tensors["img_feats"].shape[4]),
            "voxel_num": [int(value) for value in case["voxel_num"]],
            "valid_cells": int(tensors["mask"].sum().item()),
        },
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
