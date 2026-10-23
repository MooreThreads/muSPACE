"""Deterministic BEVDepth voxel-pooling training fixture."""

from __future__ import annotations

import torch

FORMAT = "muspace.voxel_pooling_train.v1"

#: Case ids are stable across releases: an id always describes the same workload.
CASES: dict[str, dict] = {
    "bevdepth-60k-mixed-coords": {
        "batch": 2,
        "points": 60_000,
        "channels": 80,
        "voxel_num": (128, 128, 1),
        "description": (
            "BEVDepth LSS training workload: 2 x 60000 points, 80 channels, voxel grid "
            "128 x 128 x 1, with valid, out-of-grid and duplicate voxel coordinates"
        ),
    },
}
DEFAULT_CASE = "bevdepth-60k-mixed-coords"


def case_ids() -> tuple[str, ...]:
    """Return the canonical case ids in declaration order."""

    return tuple(CASES)


def case_parameters(case_id: str = DEFAULT_CASE) -> dict:
    """Return the registered parameters of ``case_id``."""

    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown voxel_pooling_train case {case_id!r}; expected one of {', '.join(case_ids())}"
        ) from None


def make_named_case(case_id: str = DEFAULT_CASE, seed: int = 2333) -> dict:
    """Build the fixture registered for ``case_id`` from ``seed``."""

    parameters = case_parameters(case_id)
    batch, points, channels = parameters["batch"], parameters["points"], parameters["channels"]
    voxel_num = list(parameters["voxel_num"])
    # Coordinates start outside the grid and the z lane spans one step beyond it, so the
    # fixture exercises invalid rejection as well as duplicate accumulation.
    generator = torch.Generator().manual_seed(seed)
    geom = torch.randint(-16, 144, (batch, points, 3), generator=generator, dtype=torch.int32)
    geom[..., 2] = torch.randint(-1, 2, (batch, points), generator=generator, dtype=torch.int32)
    features = torch.rand((batch, points, channels), generator=generator) - 0.5
    grad_output = (
        torch.rand((batch, channels, voxel_num[1], voxel_num[0]), generator=generator) - 0.5
    )
    return {
        "format": FORMAT,
        "case": case_id,
        "seed": seed,
        "inputs": {"geom_xyz": geom, "input_features": features, "grad_output": grad_output},
        "voxel_num": voxel_num,
    }


def make_case(seed: int = 2333) -> dict:
    """Build the canonical voxel-pooling training case; the public helper of this module."""

    return make_named_case(DEFAULT_CASE, seed=seed)
