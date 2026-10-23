#!/usr/bin/env python3
"""Benchmark the FlashBEV and FlashSCA view-transformation operators on MUSA.

``--op`` selects the entry point (`flash_bevpool` or `flash_bevpool_sca`) and ``--case`` selects
the registered workload and direction.  The published protocol and the record format live in
``benchmarks/_harness.py`` and ``docs/benchmarking.md``.

A captured calibration file can be replayed locally with ``--projection-json`` to explain a
number; it is never how a published row is produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import torch

# Allow `python benchmarks/flashbev/benchmark_flashbev.py` without PYTHONPATH manipulation beyond
# the repository root for `mu_space`, while keeping the harness and the local case module
# importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import add_common_args, make_record, measure, print_summary, resolve_output
from flashbev_case import (
    ENTRY_POINTS,
    FLASHBEV,
    FORWARD,
    SCA_HEADS,
    SCA_POINTS,
    case_ids,
    case_parameters,
    forward_case,
    make_named_case,
)

#: A single direction stays inside the operator boundary: one forward call, or one forward call
#: plus the backward pass it triggers.
SCOPE = "operator"
OPERATOR = "flashbev"
DEFAULT_PROJECTION_SOURCE = "synthetic_rig"
REPLAY_PROJECTION_SOURCE = "local_calibration_replay"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cases=case_ids(), default_case=None, dtypes=("float32",))
    parser.add_argument(
        "--op",
        choices=ENTRY_POINTS,
        default=None,
        help="operator entry point; defaults to the entry point the selected case belongs to",
    )
    parser.add_argument(
        "--projection-json",
        type=Path,
        default=None,
        help=(
            "local investigation only: replay a captured [B,N,3,4] calibration file instead of "
            "the synthetic rig; never used for a published row"
        ),
    )
    args = parser.parse_args(argv)
    if args.case is None:
        args.op = args.op or FLASHBEV
        args.case = forward_case(args.op)
    registered = str(case_parameters(args.case)["op"])
    if args.op is not None and args.op != registered:
        raise SystemExit(
            f"--case {args.case} is registered for --op {registered}, not {args.op}; "
            f"registered cases: {', '.join(case_ids())}"
        )
    args.op = registered
    return args


def load_projection(path: Path, batch: int, cameras: int, device: str) -> torch.Tensor:
    """Load a captured calibration file for the local, non-published replay path."""

    projection = torch.tensor(json.loads(path.read_text(encoding="utf-8")), dtype=torch.float32)
    if projection.dim() == 3:
        projection = projection.unsqueeze(0)
    expected = (batch, cameras, 3, 4)
    if tuple(projection.shape) != expected:
        raise ValueError(
            f"calibration file has projection shape {tuple(projection.shape)}, expected {expected}"
        )
    return projection.to(device=device).contiguous()


def build_launch(
    op: str,
    direction: str,
    tensors: dict[str, torch.Tensor],
    grid: tuple[int, int, int],
    roi_range: tuple[float, ...],
) -> Callable[[], None]:
    """Return the timed launcher for one entry point and one direction.

    Both launchers call the public entry point with the arguments the operator's production
    callers use, so the timed region is the operator boundary itself.
    """

    if op == FLASHBEV:
        from mu_space.ops import flash_bevpool

        image_feats = tensors["image_feats"].detach().requires_grad_(True)

        def forward() -> torch.Tensor:
            return flash_bevpool(
                image_feats,
                tensors["projection_matrices"],
                tensors["feature_size"],
                tensors["image_size"],
                grid,
                roi_range=roi_range,
                backend="musa",
            )

        if direction == FORWARD:
            return forward
        grad_output = torch.ones_like(forward())

        def backward() -> None:
            torch.autograd.grad(forward(), image_feats, grad_output, retain_graph=False)

        return backward

    from mu_space.ops import flash_bevpool_sca

    value = tensors["value"].detach().requires_grad_(True)
    sampling_offsets = tensors["sampling_offsets"].detach().requires_grad_(True)
    attention_weights = tensors["attention_weights"].detach().requires_grad_(True)

    def sca_forward() -> torch.Tensor:
        return flash_bevpool_sca(
            value,
            tensors["projection_matrices"],
            tensors["spatial_shapes"],
            tensors["level_start_index"],
            tensors["image_size"],
            sampling_offsets,
            attention_weights,
            grid,
            roi_range=roi_range,
            num_heads=SCA_HEADS,
            num_points=SCA_POINTS,
            backend="musa",
        )

    if direction == FORWARD:
        return sca_forward
    sca_grad_output = torch.ones_like(sca_forward())

    def sca_backward() -> None:
        torch.autograd.grad(
            sca_forward(),
            (value, sampling_offsets, attention_weights),
            sca_grad_output,
            retain_graph=False,
        )

    return sca_backward


def describe(args: argparse.Namespace, case: dict, tensors: dict[str, torch.Tensor]) -> dict:
    """Return the case properties recorded next to the latency."""

    scalars = case["scalars"]
    derived = {
        "entry_point": args.op,
        "direction": case["direction"],
        "grid": list(case["grid"]),
        "roi_range": list(case["roi_range"]),
        "image_size": list(scalars["image_size"]),
        "batch": scalars["batch"],
        "cameras": scalars["cameras"],
        "channels": scalars["channels"],
        "projection_source": (
            REPLAY_PROJECTION_SOURCE if args.projection_json else DEFAULT_PROJECTION_SOURCE
        ),
    }
    if args.op == FLASHBEV:
        derived["image_feats_shape"] = list(tensors["image_feats"].shape)
        derived["feature_size"] = [int(value) for value in tensors["feature_size"][0, 0]]
    else:
        derived["value_shape"] = list(tensors["value"].shape)
        derived["sampling_offsets_shape"] = list(tensors["sampling_offsets"].shape)
        derived["attention_weights_shape"] = list(tensors["attention_weights"].shape)
        derived["levels"] = [list(level) for level in scalars["levels"]]
        derived["heads"] = scalars["heads"]
        derived["points"] = scalars["points"]
    return derived


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = make_named_case(args.case, seed=args.seed)
    tensors = {name: value.to(args.device) for name, value in case["inputs"].items()}
    if args.projection_json is not None:
        scalars = case["scalars"]
        tensors["projection_matrices"] = load_projection(
            args.projection_json, int(scalars["batch"]), int(scalars["cameras"]), args.device
        )
    launch = build_launch(
        args.op, case["direction"], tensors, tuple(case["grid"]), tuple(case["roi_range"])
    )
    measurement = measure(
        launch,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    record = make_record(
        operator=args.op,
        case=args.case,
        dtype=args.dtype,
        scope=SCOPE,
        device=args.device,
        seed=args.seed,
        measurement=measurement,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        derived=describe(args, case, tensors),
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
