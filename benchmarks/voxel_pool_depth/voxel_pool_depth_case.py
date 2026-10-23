"""Deterministic Qwen-Drive-1.0 depth-aware voxel pooling fixture."""

from __future__ import annotations

import torch

FORMAT = "muspace.voxel_pool_depth.v1"

#: Case ids are stable across releases: an id always describes the same workload.
CASES: dict[str, dict] = {
    "qwen-drive-6cam-frustum": {
        "batch": 1,
        "sweeps": 1,
        "cameras": 6,
        "depth_bins": 32,
        "height": 64,
        "width": 64,
        "channels": 64,
        "voxel_num": (100, 100, 16),
        "mask_ratio": 0.35,
        "description": (
            "Qwen-Drive-1.0 view-transform workload: one batch, one sweep and six cameras "
            "producing a 32 x 64 x 64 frustum per camera, 64 feature channels, on a "
            "100 x 100 x 16 voxel grid; 35 % of the frustum cells project into the volume, "
            "so many cells collide into the same voxel"
        ),
    },
}
DEFAULT_CASE = "qwen-drive-6cam-frustum"


def case_ids() -> tuple[str, ...]:
    """Return the canonical case ids in declaration order."""

    return tuple(CASES)


def case_parameters(case_id: str = DEFAULT_CASE) -> dict:
    """Return the registered parameters of ``case_id``."""

    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown voxel_pool_depth case {case_id!r}; expected one of {', '.join(case_ids())}"
        ) from None


def make_named_case(case_id: str = DEFAULT_CASE, seed: int = 2333) -> dict:
    """Build the fixture registered for ``case_id`` from ``seed``.

    The frustum coordinates are drawn from the voxel grid itself, so the case
    exercises voxel collisions and duplicate accumulation.  ``img_depth`` is a
    normalised distribution over the depth bins and the output gradient is
    centred, matching the view-transform training workload.
    """

    parameters = case_parameters(case_id)
    batch = parameters["batch"]
    sweeps = parameters["sweeps"]
    cameras = parameters["cameras"]
    depth_bins = parameters["depth_bins"]
    height = parameters["height"]
    width = parameters["width"]
    channels = parameters["channels"]
    voxel_x, voxel_y, voxel_z = parameters["voxel_num"]

    generator = torch.Generator().manual_seed(seed)
    cells = (batch, sweeps, cameras, depth_bins, height, width)
    voxel_coords = torch.zeros((*cells, 4), dtype=torch.int32)
    voxel_coords[..., 1] = torch.randint(0, voxel_x, cells, generator=generator)
    voxel_coords[..., 2] = torch.randint(0, voxel_y, cells, generator=generator)
    voxel_coords[..., 3] = torch.randint(0, voxel_z, cells, generator=generator)
    mask = torch.rand(cells, generator=generator) < parameters["mask_ratio"]
    img_feats = torch.randn(batch, sweeps * cameras, channels, height, width, generator=generator)
    depth_logits = torch.randn(
        batch * sweeps * cameras, depth_bins, height, width, generator=generator
    )
    img_depth = torch.softmax(depth_logits, dim=1)
    grad_output = torch.rand(
        batch, cameras, voxel_x, voxel_y, voxel_z, channels, generator=generator
    )
    return {
        "format": FORMAT,
        "case": case_id,
        "seed": seed,
        "scalars": {"B": batch, "X": voxel_x, "Y": voxel_y, "Z": voxel_z},
        "inputs": {
            "img_feats": img_feats,
            "img_depth": img_depth,
            "voxel_coords": voxel_coords,
            "mask": mask,
            "grad_output": grad_output - 0.5,
        },
        "voxel_num": [voxel_x, voxel_y, voxel_z],
    }


def make_case(seed: int = 2333) -> dict:
    """Build the canonical voxel_pool_depth case; the public helper of this module."""

    return make_named_case(DEFAULT_CASE, seed=seed)
