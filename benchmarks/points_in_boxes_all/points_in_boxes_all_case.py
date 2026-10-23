"""Deterministic ``points_in_boxes_all`` benchmark cases.

A case id names the workload shape alone, because this directory owns a single entry point:
``b<batch>-n<points>-m<boxes>``.  A case id is stable across releases.  Every published input is
built here from the documented seed; no data set is redistributed.
"""

from __future__ import annotations

import torch

FORMAT = "points_in_boxes_all_v1"
DEFAULT_SEED = 2333

#: Case ids are stable across releases; declaration order is the published order.
CASES: dict[str, dict] = {
    "b1-n10000-m100": {
        "batch": 1,
        "points": 10_000,
        "boxes": 100,
        "description": (
            "one frame of 10 000 points against 100 boxes: a small frame whose latency is "
            "dominated by the fixed per-launch cost and the box transforms"
        ),
    },
    "b1-n100000-m100": {
        "batch": 1,
        "points": 100_000,
        "boxes": 100,
        "description": (
            "one frame of 100 000 points against 100 boxes: the frame size this benchmark has "
            "used as its default workload"
        ),
    },
    "b1-n1000000-m100": {
        "batch": 1,
        "points": 1_000_000,
        "boxes": 100,
        "description": (
            "one frame of 1 000 000 points against 100 boxes: a large frame that emphasizes the "
            "per-point work"
        ),
    },
}
DEFAULT_CASE = "b1-n100000-m100"


def case_ids() -> tuple[str, ...]:
    """Return the canonical case ids in declaration order."""

    return tuple(CASES)


def case_parameters(case_id: str = DEFAULT_CASE) -> dict:
    """Return the registered parameters of ``case_id``."""

    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown points_in_boxes_all case {case_id!r}; expected one of {', '.join(case_ids())}"
        ) from None


def make_named_case(
    case_id: str = DEFAULT_CASE,
    seed: int = DEFAULT_SEED,
    batch: int = 0,
    points: int = 0,
    boxes: int = 0,
) -> dict:
    """Build the fixture registered for ``case_id`` from ``seed``.

    The shape switches override one dimension of the registered workload for local investigation
    and re-derive the case id, so a record always names the workload it measured.  Tensors are
    built on the host: the runner moves them to the measured device without touching dtype or
    layout.
    """

    parameters = case_parameters(case_id)
    shape = (
        int(batch) or parameters["batch"],
        int(points) or parameters["points"],
        int(boxes) or parameters["boxes"],
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    box_tensor = torch.randn((shape[0], shape[2], 7), generator=generator, dtype=torch.float32)
    box_tensor[..., 3:6] = box_tensor[..., 3:6].abs() + 1.0
    point_tensor = torch.randn((shape[0], shape[1], 3), generator=generator, dtype=torch.float32)
    return {
        "format": FORMAT,
        "case": f"b{shape[0]}-n{shape[1]}-m{shape[2]}",
        "seed": seed,
        "batch": shape[0],
        "points": shape[1],
        "boxes": shape[2],
        "description": parameters["description"],
        "inputs": {"boxes": box_tensor, "points": point_tensor},
    }


def make_case(
    seed: int = DEFAULT_SEED, batch: int = 1, points: int = 100_000, boxes: int = 100
) -> dict:
    """Build the canonical fixture; the public helper of this module."""

    return make_named_case(DEFAULT_CASE, seed=seed, batch=batch, points=points, boxes=boxes)


def output_mask(case: dict, device: str) -> torch.Tensor:
    """Preallocate the ``[B, N, M]`` int32 mask the kernel writes into."""

    return torch.empty(
        (int(case["batch"]), int(case["points"]), int(case["boxes"])),
        dtype=torch.int32,
        device=device,
    )
