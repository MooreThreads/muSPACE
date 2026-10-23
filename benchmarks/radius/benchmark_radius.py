#!/usr/bin/env python3
"""Benchmark the muSPACE radius neighborhood queries used by the QCNet model recipe.

Three published cases share the deterministic QCNet / Argoverse 2 scene of
:mod:`radius_case` and differ in the entry point and the neighborhood they build:

``av2-enc-pl2a``
    ``mu_space.ops.radius`` for the scene encoder polygon-to-agent query.
``av2-enc-a2a``
    ``mu_space.ops.radius_graph`` for the scene encoder agent-to-agent graph.
``av2-enc-compat``
    ``mu_space.qcnet_compat`` for both queries of one encoder step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from _harness import (
    add_common_args,
    device_module,
    make_record,
    measure,
    print_summary,
    resolve_output,
)
from radius_case import DEFAULT_CASE, case_ids, make_named_case

OPERATOR = "radius"
#: Every case times one operator entry point as a caller issues it: the scene-encoder
#: neighborhood for the two primitives, and the two-query encoder step for the
#: compatibility integration.  The timed region therefore contains the entry point's own
#: fixed-capacity neighbor-buffer allocation and its single device-to-host read-back of
#: the compacted edge count.  The fixture, its transfer to the device and any downstream
#: use of the returned edge index stay outside the timed region.
SCOPE = "operator"
ENTRY_POINTS = ("radius", "radius_graph", "qcnet_compat")
#: The compatibility layer routes MUSA tensors to the native implementation and refuses to
#: fall back; the primitive entry points are measured with the same strict selection.
BACKEND = "musa"
INT64_BYTES = 8
DIRECTION_ORDER = ("pl2a", "a2a")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cases=case_ids(), default_case=DEFAULT_CASE, dtypes=("float32",))
    parser.add_argument(
        "--op",
        choices=ENTRY_POINTS,
        default=None,
        help=(
            "entry point override for local investigation; a published row always uses "
            "the entry point registered for --case"
        ),
    )
    return parser.parse_args(argv)


def primitive_calls(
    tensors: dict[str, torch.Tensor], radius_value: float, max_num_neighbors: int
) -> dict[str, Callable[[], torch.Tensor]]:
    """Return the two native primitive entry-point calls for the shared fixture."""

    from mu_space.ops import radius, radius_graph

    def pl2a() -> torch.Tensor:
        return radius(
            tensors["x"],
            tensors["y"],
            radius_value,
            tensors["batch_x"],
            tensors["batch_y"],
            max_num_neighbors,
            backend=BACKEND,
        )

    def a2a() -> torch.Tensor:
        return radius_graph(
            tensors["x"],
            radius_value,
            tensors["batch_x"],
            loop=False,
            max_num_neighbors=max_num_neighbors,
            backend=BACKEND,
        )

    return {"pl2a": pl2a, "a2a": a2a}


def compat_calls(
    tensors: dict[str, torch.Tensor], radius_value: float, max_num_neighbors: int
) -> dict[str, Callable[[], torch.Tensor]]:
    """Return the two QCNet-compatible calls of one scene-encoder neighborhood step."""

    from mu_space.qcnet_compat import radius as compat_radius
    from mu_space.qcnet_compat import radius_graph as compat_radius_graph

    def pl2a() -> torch.Tensor:
        return compat_radius(
            x=tensors["x"],
            y=tensors["y"],
            r=radius_value,
            batch_x=tensors["batch_x"],
            batch_y=tensors["batch_y"],
            max_num_neighbors=max_num_neighbors,
        )

    def a2a() -> torch.Tensor:
        return compat_radius_graph(
            x=tensors["x"],
            r=radius_value,
            batch=tensors["batch_x"],
            loop=False,
            max_num_neighbors=max_num_neighbors,
        )

    return {"pl2a": pl2a, "a2a": a2a}


def neighbor_capacity(
    tensors: dict[str, torch.Tensor], max_num_neighbors: int, direction: str
) -> int:
    """Return the fixed-capacity int64 neighbor buffer one direction allocates.

    The native path sizes its neighbor buffer as query points times the neighbor cap
    before it compacts the result, so the buffer dwarfs the neighbor count itself.
    ``radius_graph`` additionally uses one slot per query point for the self pair that
    ``loop=False`` drops.
    """

    if direction == "pl2a":
        return int(tensors["y"].shape[0]) * max_num_neighbors
    return int(tensors["x"].shape[0]) * (max_num_neighbors + 1)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    case = make_named_case(args.case, seed=args.seed)
    entry_point = args.op or str(case["entry_point"])
    dtype = getattr(torch, args.dtype)
    scalars = case["scalars"]
    tensors = {
        "x": case["inputs"]["x"].to(device=args.device, dtype=dtype),
        "y": case["inputs"]["y"].to(device=args.device, dtype=dtype),
        "batch_x": case["inputs"]["batch_x"].to(args.device),
        "batch_y": case["inputs"]["batch_y"].to(args.device),
    }
    radius_value = float(scalars["radius"])
    max_num_neighbors = int(scalars["max_num_neighbors"])

    if entry_point == "qcnet_compat":
        calls = compat_calls(tensors, radius_value, max_num_neighbors)
    else:
        calls = primitive_calls(tensors, radius_value, max_num_neighbors)
    directions = {
        "radius": ("pl2a",),
        "radius_graph": ("a2a",),
        "qcnet_compat": DIRECTION_ORDER,
    }[entry_point]

    def launch() -> object:
        edges = [calls[direction]() for direction in directions]
        return edges[0] if len(edges) == 1 else tuple(edges)

    # One extra call outside the timed region measures the neighbor count each direction
    # actually returns, and the fixed-capacity buffer the timed region reserves.
    edges = {direction: int(calls[direction]().shape[1]) for direction in directions}
    buffer_gib = (
        sum(neighbor_capacity(tensors, max_num_neighbors, direction) for direction in directions)
        * INT64_BYTES
        / 2**30
    )

    module = device_module(args.device)
    module.reset_peak_memory_stats()
    measurement = measure(
        launch,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    peak_gib = module.max_memory_allocated() / 2**30

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
            "entry_point": entry_point,
            "backend": BACKEND,
            "scenarios": int(scalars["scenarios"]),
            "historical_steps": int(scalars["historical_steps"]),
            "agents_per_scenario": int(scalars["agents_per_scenario"]),
            "polygons_per_scenario": int(scalars["polygons_per_scenario"]),
            "coordinate_dim": int(scalars["coordinate_dim"]),
            "local_map_extent_m": float(scalars["local_map_extent_m"]),
            "radius_m": radius_value,
            "max_num_neighbors": max_num_neighbors,
            "agent_points": int(tensors["x"].shape[0]),
            "polygon_points": int(tensors["y"].shape[0]),
            "edges": edges,
            "neighbor_buffer_gib": round(buffer_gib, 3),
            "peak_device_gib": round(peak_gib, 3),
        },
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
