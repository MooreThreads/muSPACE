"""Deterministic temporal_align benchmark cases."""

from __future__ import annotations

import torch

FORMAT = "temporal_align_v1"
DEFAULT_SEED = 2333
FORWARD = "forward"
BACKWARD = "backward"

#: BEVFormer's BEV grid (200x200) with the usual 256-channel feature width.
BEV_H = 200
BEV_W = 200
CHANNELS = 256
#: Corner-based centre of a 200x200 plane; the upstream rotation passes the same value.
CENTER = (float(BEV_W) / 2.0, float(BEV_H) / 2.0)

WORKLOADS = {
    "bev-b1": {"batch": 1, "description": "single-sample temporal alignment"},
    "bev-b4": {"batch": 4, "description": "four-sample temporal alignment"},
    "bev-b8": {"batch": 8, "description": "eight-sample temporal alignment"},
}

CASES = {
    f"{name}-{direction}": {
        **workload,
        "direction": direction,
    }
    for name, workload in WORKLOADS.items()
    for direction in (FORWARD, BACKWARD)
}
DEFAULT_CASE = "bev-b1-forward"


def case_ids() -> tuple[str, ...]:
    return tuple(CASES)


def case_parameters(case_id: str) -> dict:
    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown temporal_align case {case_id!r}; expected one of {', '.join(case_ids())}"
        ) from None


def sample_angles(batch: int) -> torch.Tensor:
    """Per-sample rotations in degrees; never a quarter turn, never zero."""
    return torch.linspace(-12.5, 12.5, batch, dtype=torch.float32)


def make_named_case(case_id: str = DEFAULT_CASE, seed: int = DEFAULT_SEED) -> dict:
    parameters = case_parameters(case_id)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = parameters["batch"]
    bev = torch.randn(batch, CHANNELS, BEV_H, BEV_W, generator=generator, dtype=torch.float32)
    return {
        "format": FORMAT,
        "case": case_id,
        "seed": seed,
        "direction": parameters["direction"],
        "description": parameters["description"],
        "batch": batch,
        "channels": CHANNELS,
        "bev_h": BEV_H,
        "bev_w": BEV_W,
        "bev_shape": (batch, CHANNELS, BEV_H, BEV_W),
        "angle": sample_angles(batch),
        "shift": None,
        "center": CENTER,
        "mode": "bilinear",
        "inputs": {
            "bev": bev,
            "angle": sample_angles(batch),
        },
    }


def make_case(seed: int = DEFAULT_SEED) -> dict:
    return make_named_case(DEFAULT_CASE, seed=seed)
