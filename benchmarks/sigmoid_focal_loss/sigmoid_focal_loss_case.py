"""Deterministic sigmoid focal loss benchmark cases.

The reference implementation of this operator is a plain PyTorch expression, so the same case is
measured on the MUSA device and, later, on the reference device.  Every tensor is built on the
host from the fixed seed and moved to the measured device by the runner.
"""

from __future__ import annotations

import torch

FORMAT = "sigmoid_focal_loss_bevfusion_v1"

DEFAULT_SEED = 2333

#: Contract of the BEVFusion BEVSegmentationHead loss.
ALPHA = 0.25
GAMMA = 2.0
REDUCTION = "mean"

FORWARD = "forward"
BACKWARD = "backward"
DIRECTIONS = (FORWARD, BACKWARD)

#: Workloads.  `bevfusion-seg` is the production segmentation head grid; `small` and `wide` are
#: small-shape sanity rows.  Each workload is published once per direction.
WORKLOADS: dict[str, dict] = {
    "bevfusion-seg": {
        "shape": (4, 3, 200, 200),
        "description": "BEVFusion BEVSegmentationHead workload: batch 4, 3 classes, 200 x 200 grid",
    },
    "small": {
        "shape": (2, 3, 64, 64),
        "description": "64 x 64 grid, batch 2, 3 classes",
    },
    "wide": {
        "shape": (8, 256, 1),
        "description": "256 classes, one spatial position, batch 8",
    },
}

#: Case ids are stable across releases: an id always describes the same workload and direction.
CASES: dict[str, dict] = {
    f"{name}-{direction}": {
        "shape": parameters["shape"],
        "description": parameters["description"],
        "direction": direction,
    }
    for name, parameters in WORKLOADS.items()
    for direction in DIRECTIONS
}
DEFAULT_CASE = "bevfusion-seg-forward"


def case_ids() -> tuple[str, ...]:
    """Return the canonical case ids in declaration order."""

    return tuple(CASES)


def case_parameters(case_id: str = DEFAULT_CASE) -> dict:
    """Return the registered parameters of ``case_id``."""

    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown sigmoid focal loss case {case_id!r}; expected one of {', '.join(case_ids())}"
        ) from None


def make_named_case(case_id: str = DEFAULT_CASE, seed: int = DEFAULT_SEED) -> dict:
    """Build the fixture registered for ``case_id`` from ``seed``.

    Logits are standard normal and targets are binary, both drawn from one deterministic stream in
    this order; the runner later casts both to the measured dtype, so every dtype starts from the
    same sample.
    """

    parameters = case_parameters(case_id)
    shape = tuple(parameters["shape"])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.randn(*shape, generator=generator, dtype=torch.float32)
    targets = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64).to(torch.float32)
    return {
        "format": FORMAT,
        "case": case_id,
        "seed": seed,
        "direction": parameters["direction"],
        "description": parameters["description"],
        "shape": shape,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "reduction": REDUCTION,
        "inputs": {"inputs": inputs, "targets": targets},
    }


def make_case(seed: int = DEFAULT_SEED) -> dict:
    """Build the canonical forward case; the public helper of this module."""

    return make_named_case(DEFAULT_CASE, seed=seed)
