#!/usr/bin/env python3
"""Benchmark the muSPACE BEVFusion-compatible BEVPool operator."""

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
from bev_pool_case import (
    DEFAULT_CASE,
    FORMAT,
    case_ids,
    make_named_case,
)

OPERATOR = "bev_pool"
#: A single direction stays inside the operator boundary; the combined mode covers the
#: end-to-end training step and is the published scope.
COMBINED_SCOPE = "forward-backward"
SINGLE_DIRECTION_SCOPE = "operator"
MODES = ("forward", "backward", "forward-backward")

_BYTES_PER_ELEMENT = 4
_BACKWARD_PASSES = 2
_COMBINED_PASSES = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cases=case_ids(), default_case=DEFAULT_CASE, dtypes=("float32",))
    parser.add_argument(
        "--input-file",
        type=Path,
        help="captured artifact for local investigation; not how a published row is built",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=None,
        help="optional point-count override for local investigation",
    )
    parser.add_argument("--mode", choices=MODES, default=COMBINED_SCOPE)
    return parser.parse_args(argv)


def load_artifact(path: Path) -> dict:
    """Load a captured BEVPool artifact for the local replay path."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported BEVPool benchmark artifact: {path}")
    return payload


def scope_for(mode: str) -> str:
    return COMBINED_SCOPE if mode == COMBINED_SCOPE else SINGLE_DIRECTION_SCOPE


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_file is None:
        payload = make_named_case(args.case, seed=args.seed, points=args.points)
    else:
        payload = load_artifact(args.input_file)

    from mu_space import bev_pool

    scalars = payload["scalars"]
    tensors = {name: value.to(args.device) for name, value in payload["inputs"].items()}
    batch, depth, height, width = scalars["B"], scalars["D"], scalars["H"], scalars["W"]

    def run() -> None:
        feats = tensors["feats"].detach().requires_grad_(args.mode != "forward")
        output = bev_pool(
            feats,
            tensors["coords"],
            batch,
            depth,
            height,
            width,
            backend="musa",
        )
        if args.mode != "forward":
            output.backward(tensors["grad_output"])

    measurement = measure(
        run,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    channels = int(tensors["feats"].shape[1])
    traffic_bytes = int(scalars["points"]) * channels * _BYTES_PER_ELEMENT
    if args.mode == COMBINED_SCOPE:
        traffic_bytes *= _COMBINED_PASSES
    elif args.mode == "backward":
        traffic_bytes *= _BACKWARD_PASSES
    record = make_record(
        operator=OPERATOR,
        case=args.case,
        dtype=args.dtype,
        scope=scope_for(args.mode),
        device=args.device,
        seed=args.seed,
        measurement=measurement,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        derived={
            "mode": args.mode,
            "points": int(scalars["points"]),
            "channels": channels,
            "grid": {"B": batch, "D": depth, "H": height, "W": width},
            "effective_gbps": traffic_bytes / measurement.p50_ms / 1e6,
            "artifact_replay": args.input_file is not None,
            "points_override": args.points is not None,
        },
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
