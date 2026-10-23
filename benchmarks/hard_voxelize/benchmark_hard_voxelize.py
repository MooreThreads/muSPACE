#!/usr/bin/env python3
"""Benchmark the parallel ``hard_voxelize`` path on MUSA.

The published protocol and the record format live in ``benchmarks/_harness.py`` and
``docs/benchmarking.md``.  The timed region is the native launcher of the parallel
(``deterministic=False``) path, called with preallocated outputs and workspaces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Keep the harness and the local case module importable when the runner is executed
# directly, as the operator README documents.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import add_common_args, make_record, measure, print_summary, resolve_output
from hard_voxelize_case import DEFAULT_CASE, DETERMINISTIC, case_ids, make_named_case

OPERATOR = "hard_voxelize"
SCOPE = "kernel-only"
#: The parallel path does not promise an output voxel order; the deterministic path is a
#: different operator boundary and is not published here.
PATH_NAME = "parallel (deterministic=False)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        cases=case_ids(),
        default_case=DEFAULT_CASE,
        dtypes=("float32",),
    )
    return parser.parse_args(argv)


def build_launch(native, case, points, buffers):
    """Return the timed launcher: the native entry point, everything else preallocated."""

    grid = case.grid
    voxel_tensor = points.new_tensor(case.voxel)
    bounds_tensor = points.new_tensor(case.bounds)

    def launch() -> None:
        native.hard_voxelize_forward(
            points,
            voxel_tensor,
            bounds_tensor,
            buffers["voxels"],
            buffers["coors"],
            buffers["counts"],
            buffers["voxel_num"],
            buffers["coord_map"],
            buffers["coord_offsets"],
            buffers["point_keys"],
            case.max_points,
            case.max_voxels,
            grid[0],
            grid[1],
            grid[2],
            DETERMINISTIC,
        )

    return launch


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = make_named_case(args.case, seed=args.seed, device=args.device)
    case = payload["case_parameters"]
    points = payload["points"]
    buffers = payload["buffers"]

    from mu_space.runtime.native import native_module_proxy

    launch = build_launch(native_module_proxy(), case, points, buffers)
    measurement = measure(
        launch,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    # Metadata only, collected after the timed region.
    launch()
    voxels_used = int(buffers["voxel_num"].to("cpu"))

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
            "path": PATH_NAME,
            "num_points": case.num_points,
            "features": case.features,
            "voxel": list(case.voxel),
            "bounds": list(case.bounds),
            "grid": list(case.grid),
            "grid_cells": case.grid_cells,
            "max_points": case.max_points,
            "max_voxels": case.max_voxels,
            "voxels_used": voxels_used,
        },
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
