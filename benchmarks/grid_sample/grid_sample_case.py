"""Deterministic grid_sample benchmark cases."""

from __future__ import annotations

import torch

FORMAT = "grid_sample_v1"
DEFAULT_SEED = 2333
FORWARD = "forward"
BACKWARD = "backward"

WORKLOADS = {
    "bev-2d": {
        "input_shape": (1, 64, 128, 128),
        "grid_shape": (1, 64, 64, 2),
        "description": "2-D image-domain sampling",
    },
    "voxel-3d": {
        "input_shape": (1, 32, 16, 32, 32),
        "grid_shape": (1, 8, 16, 16, 3),
        "description": "3-D voxel-domain sampling",
    },
}

CASES = {
    f"{name}-{direction}": {
        **workload,
        "direction": direction,
    }
    for name, workload in WORKLOADS.items()
    for direction in (FORWARD, BACKWARD)
}
DEFAULT_CASE = "bev-2d-forward"


def case_ids() -> tuple[str, ...]:
    return tuple(CASES)


def case_parameters(case_id: str = DEFAULT_CASE) -> dict:
    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown grid_sample case {case_id!r}; expected one of {', '.join(case_ids())}"
        ) from None


def make_named_case(case_id: str = DEFAULT_CASE, seed: int = DEFAULT_SEED) -> dict:
    parameters = case_parameters(case_id)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_shape = tuple(parameters["input_shape"])
    grid_shape = tuple(parameters["grid_shape"])
    value = torch.randn(*input_shape, generator=generator, dtype=torch.float32)
    grid = torch.rand(*grid_shape, generator=generator, dtype=torch.float32).mul_(2).sub_(1)
    return {
        "format": FORMAT,
        "case": case_id,
        "seed": seed,
        "direction": parameters["direction"],
        "description": parameters["description"],
        "input_shape": input_shape,
        "grid_shape": grid_shape,
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": False,
        "inputs": {"input": value, "grid": grid},
    }


def make_case(seed: int = DEFAULT_SEED) -> dict:
    return make_named_case(DEFAULT_CASE, seed=seed)
