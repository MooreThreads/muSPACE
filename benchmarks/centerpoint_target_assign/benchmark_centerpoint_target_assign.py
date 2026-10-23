#!/usr/bin/env python3
"""Benchmark ``centerpoint_target_assign`` on MUSA.

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
from centerpoint_target_assign_case import (
    DEFAULT_CASE,
    FORMAT,
    case_ids,
    make_named_case,
)

OPERATOR = "centerpoint_target_assign"
SCOPE = "operator"
REFERENCE_DEVICE = "NVIDIA A100 80GB PCIe"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        cases=case_ids(),
        default_case=DEFAULT_CASE,
        dtypes=("float32",),
    )
    parser.add_argument("--num-objects", type=int, default=0, help="override the object count")
    parser.add_argument("--num-classes", type=int, default=0, help="override the class count")
    parser.add_argument("--max-objs", type=int, default=0, help="override the assignment bound")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the native path against the PyTorch oracle before timing",
    )
    return parser.parse_args(argv)


def build_launch(boxes: torch.Tensor, class_ids: torch.Tensor, parameters: dict):
    """Return the timed callable: the public operator, native backend, one call per sample.

    The operator allocates its heatmap, annotation buffer, index and mask outputs inside the call,
    so the timed region is the public boundary rather than a preallocated kernel launch.
    """

    from mu_space import centerpoint_target_assign

    def launch():
        return centerpoint_target_assign(boxes, class_ids, backend="musa", **parameters)

    return launch


def run_check(launch, boxes: torch.Tensor, class_ids: torch.Tensor, parameters: dict) -> dict:
    """Compare the native outputs against the module's own PyTorch oracle."""

    from mu_space import centerpoint_target_assign_reference

    outputs = launch()
    reference = centerpoint_target_assign_reference(boxes, class_ids, **parameters)
    return {
        "checked": True,
        "oracle": "mu_space.ops.centerpoint_target_assign_reference",
        "heatmap_max_abs_error": float((outputs[0] - reference[0]).abs().max()),
        "anno_box_max_abs_error": float((outputs[1].double() - reference[1].double()).abs().max()),
        "index_mismatches": int((outputs[2] != reference[2]).sum()),
        "mask_mismatches": int((outputs[3] != reference[3]).sum()),
        "assigned_objects": int(reference[3].sum()),
        "positive_heatmap_cells": int((reference[0] > 0).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise SystemExit("MUSA is unavailable")
    fixture = make_named_case(
        args.case,
        seed=args.seed,
        objects=args.num_objects,
        num_classes=args.num_classes,
        max_objs=args.max_objs,
    )
    if fixture["format"] != FORMAT:
        raise ValueError("invalid fixture format")
    boxes = fixture["inputs"]["boxes"].to(args.device)
    class_ids = fixture["inputs"]["class_ids"].to(args.device)
    parameters = dict(fixture["parameters"])
    launch = build_launch(boxes, class_ids, parameters)
    check: dict[str, object] = {"checked": False}
    if args.check:
        check = run_check(launch, boxes, class_ids, parameters)
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
            "objects": int(boxes.size(0)),
            "num_classes": int(parameters["num_classes"]),
            "feature_map_size": list(parameters["feature_map_size"]),
            "out_size_factor": int(parameters["out_size_factor"]),
            "gaussian_overlap": float(parameters["gaussian_overlap"]),
            "min_radius": int(parameters["min_radius"]),
            "voxel_size": list(parameters["voxel_size"]),
            "pc_range": list(parameters["pc_range"]),
            "max_objs": int(parameters["max_objs"]),
            "check": check,
        },
        reference_device=REFERENCE_DEVICE,
    )
    print_summary(record)
    if check.get("checked"):
        print(f"check: {check}")
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
