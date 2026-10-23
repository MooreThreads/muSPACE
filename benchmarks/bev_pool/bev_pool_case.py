"""Deterministic BEVFusion DepthLSSTransform-style BEVPool cases."""

from __future__ import annotations

import torch

FORMAT = "bev_pool_bevfusion_v1"

#: Case ids are stable across releases: an id always describes the same workload.
CASES: dict[str, dict] = {
    "r50-200k": {
        "points": 200_000,
        "channels": 80,
        "grid": (1, 80, 128, 128),
        "description": (
            "ResNet-50 DepthLSSTransform workload: 200000 pooled points, 80 channels, "
            "BEV grid 1 x 80 x 128 x 128"
        ),
    },
}
DEFAULT_CASE = "r50-200k"


def case_ids() -> tuple[str, ...]:
    """Return the canonical case ids in declaration order."""

    return tuple(CASES)


def case_parameters(case_id: str = DEFAULT_CASE) -> dict:
    """Return the registered parameters of ``case_id``."""

    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown BEVPool case {case_id!r}; expected one of {', '.join(case_ids())}"
        ) from None


def make_named_case(
    case_id: str = DEFAULT_CASE, seed: int = 2333, points: int | None = None
) -> dict:
    """Build the fixture registered for ``case_id`` from ``seed``.

    ``points`` optionally overrides the registered point count for local investigation;
    a published row always uses the registered value.
    """

    parameters = case_parameters(case_id)
    batch, depth, height, width = parameters["grid"]
    channels = parameters["channels"]
    total = parameters["points"] if points is None else points
    generator = torch.Generator(device="cpu").manual_seed(seed)
    feats = torch.rand((total, channels), generator=generator)
    coords = torch.stack(
        (
            torch.randint(height, (total,), generator=generator, dtype=torch.int32),
            torch.randint(width, (total,), generator=generator, dtype=torch.int32),
            torch.randint(depth, (total,), generator=generator, dtype=torch.int32),
            torch.zeros(total, dtype=torch.int32),
        ),
        dim=1,
    )
    grad_output = torch.rand((batch, channels, depth, height, width), generator=generator)
    return {
        "format": FORMAT,
        "case": case_id,
        "seed": seed,
        "scalars": {"B": batch, "D": depth, "H": height, "W": width, "points": total},
        "inputs": {"feats": feats, "coords": coords, "grad_output": grad_output},
    }


def make_case(seed: int = 2333, points: int = 200_000) -> dict:
    """Build the canonical BEVPool case; the public helper of this module."""

    return make_named_case(DEFAULT_CASE, seed=seed, points=points)
