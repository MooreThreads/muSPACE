#!/usr/bin/env python3
"""Benchmark ``points_in_boxes_all`` on MUSA.

``--case`` selects the registered workload; the shape switches override one dimension of it for
local investigation.  The published protocol and the record format live in
``benchmarks/_harness.py`` and ``docs/benchmarking.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Keep the harness and the local case module importable when the runner is executed directly, as
# the operator README documents.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import add_common_args, make_record, measure, print_summary, resolve_output
from points_in_boxes_all_case import (
    DEFAULT_CASE,
    FORMAT,
    case_ids,
    make_named_case,
    output_mask,
)

OPERATOR = "points_in_boxes_all"
SCOPE = "kernel-only"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        cases=case_ids(),
        default_case=DEFAULT_CASE,
        dtypes=("float32",),
    )
    parser.add_argument("--batch", type=int, default=0, help="override the case batch size")
    parser.add_argument("--num-points", type=int, default=0, help="override the case point count")
    parser.add_argument("--num-boxes", type=int, default=0, help="override the case box count")
    return parser.parse_args(argv)


def build_launch(boxes: torch.Tensor, points: torch.Tensor, mask: torch.Tensor):
    """Return the timed launcher: one native kernel call writing into the preallocated mask.

    The mask is allocated by the caller, before the timed region, and nothing is copied back to the
    host while timing.
    """

    from mu_space.runtime.native import native_module_proxy

    native = native_module_proxy()

    def launch() -> None:
        native.points_in_boxes_all_forward(boxes, points, mask)

    return launch


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise SystemExit("MUSA is unavailable")
    fixture = make_named_case(
        args.case,
        seed=args.seed,
        batch=args.batch,
        points=args.num_points,
        boxes=args.num_boxes,
    )
    if fixture["format"] != FORMAT:
        raise ValueError("invalid fixture format")
    boxes = fixture["inputs"]["boxes"].to(args.device)
    points = fixture["inputs"]["points"].to(args.device)
    mask = output_mask(fixture, args.device)
    launch = build_launch(boxes, points, mask)
    measurement = measure(
        launch,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    record = make_record(
        operator=OPERATOR,
        case=str(fixture["case"]),
        dtype=args.dtype,
        scope=SCOPE,
        device=args.device,
        seed=args.seed,
        measurement=measurement,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        derived={
            "batch": int(boxes.size(0)),
            "points": int(points.size(1)),
            "boxes": int(boxes.size(1)),
            "mask_elements": int(mask.numel()),
        },
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
