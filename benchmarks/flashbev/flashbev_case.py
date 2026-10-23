"""Deterministic FlashBEV and FlashSCA view-transformation benchmark cases.

Both entry points that `benchmarks/flashbev/` owns share one camera rig, one BEV grid and one
seed, so their published rows describe the same workload geometry.  Every published input is
built here from the seed: the camera rig is an analytic pinhole model rather than a capture, and
no calibration file and no data set is read by the published path.
"""

from __future__ import annotations

import math

import torch

FORMAT = "flashbev_synthetic_v1"

DEFAULT_SEED = 2333

#: Synthetic six-camera pinhole rig: one optical centre 1.6 m above the ego origin, a 60 degree
#: yaw step between cameras, a 5 degree downward pitch, a 630 px focal length and the principal
#: point at the centre of a 448 x 800 image.  The rig is documented in the operator README.
CAMERAS = 6
IMAGE_SIZE = (448, 800)
FOCAL_LENGTH_PX = 630.0
CAMERA_HEIGHT_M = 1.6
CAMERA_PITCH_DEG = 5.0

#: SimpleBEV view-transformation geometry, shared by both entry points.
BATCH = 1
CHANNELS = 128
GRID = (200, 200, 8)
ROI_RANGE = (-50.0, 50.0, -50.0, 50.0, -5.0, 5.0)

#: Image feature map of the FlashBEV path: one strided level of the camera image.
FEATURE_LEVEL = (56, 100)

#: Feature pyramid of the FlashSCA path: three levels of the same camera image.
SCA_LEVELS = ((56, 100), (28, 50), (14, 25))
SCA_HEADS = 8
SCA_POINTS = 2
SCA_OFFSET_SCALE = 0.1

FLASHBEV = "flash_bevpool"
FLASHBEV_SCA = "flash_bevpool_sca"
ENTRY_POINTS = (FLASHBEV, FLASHBEV_SCA)

FORWARD = "forward"
BACKWARD = "backward"
DIRECTIONS = (FORWARD, BACKWARD)

#: Case ids are stable across releases: an id always describes the same workload and direction.
CASES: dict[str, dict] = {
    "simplebev-forward": {
        "op": FLASHBEV,
        "direction": FORWARD,
        "description": (
            "SimpleBEV FlashBEV pooling: one forward flash_bevpool call on six 56 x 100 camera "
            "feature maps over a 200 x 200 x 8 BEV grid"
        ),
    },
    "simplebev-backward": {
        "op": FLASHBEV,
        "direction": BACKWARD,
        "description": (
            "SimpleBEV FlashBEV pooling: one forward flash_bevpool call plus the backward pass "
            "that differentiates image_feats"
        ),
    },
    "simplebev-sca-forward": {
        "op": FLASHBEV_SCA,
        "direction": FORWARD,
        "description": (
            "FlashSCA sampling view transform: one forward flash_bevpool_sca call on a three-level "
            "camera feature pyramid with 8 heads and 2 sample points per level"
        ),
    },
    "simplebev-sca-backward": {
        "op": FLASHBEV_SCA,
        "direction": BACKWARD,
        "description": (
            "FlashSCA sampling view transform: one forward flash_bevpool_sca call plus the backward "
            "pass that differentiates value, sampling_offsets and attention_weights"
        ),
    },
}
DEFAULT_CASE = "simplebev-forward"


def case_ids() -> tuple[str, ...]:
    """Return the canonical case ids in declaration order."""

    return tuple(CASES)


def case_parameters(case_id: str = DEFAULT_CASE) -> dict:
    """Return the registered parameters of ``case_id``."""

    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown FlashBEV case {case_id!r}; expected one of {', '.join(case_ids())}"
        ) from None


def forward_case(op: str = FLASHBEV) -> str:
    """Return the forward case registered for the entry point ``op``."""

    for case_id, parameters in CASES.items():
        if parameters["op"] == op and parameters["direction"] == FORWARD:
            return case_id
    raise KeyError(f"no forward case is registered for {op!r}")


def synthetic_projection(
    batch: int = BATCH,
    cameras: int = CAMERAS,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build the documented synthetic camera rig as ``[batch, cameras, 3, 4]`` matrices.

    The rig is analytic and deterministic, so the geometry of a published row is reproducible
    from the case id alone.  Ego frame: x forward, y left, z up.  Every camera shares the single
    optical centre ``(0, 0, CAMERA_HEIGHT_M)``; camera ``c`` looks along a yaw of
    ``360 / cameras * c`` degrees from the ego x axis, pitched ``CAMERA_PITCH_DEG`` degrees down.
    Intrinsics are ``FOCAL_LENGTH_PX`` in both axes with the principal point at the centre of
    ``IMAGE_SIZE``.  Each returned matrix maps an ego-frame point to ``(u, v, depth)`` in pixels,
    which is the projection convention both entry points consume.
    """

    focal = float(FOCAL_LENGTH_PX)
    intrinsics = torch.tensor(
        [
            [focal, 0.0, IMAGE_SIZE[1] / 2.0],
            [0.0, focal, IMAGE_SIZE[0] / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    optical_centre = torch.tensor([0.0, 0.0, CAMERA_HEIGHT_M], dtype=torch.float64)
    pitch = math.radians(CAMERA_PITCH_DEG)
    body_to_camera = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [-math.sin(pitch), 0.0, -math.cos(pitch)],
            [math.cos(pitch), 0.0, -math.sin(pitch)],
        ],
        dtype=torch.float64,
    )
    projection = torch.empty((batch, cameras, 3, 4), dtype=torch.float64)
    for camera in range(cameras):
        yaw = math.radians(360.0 / cameras * camera)
        ego_to_body = torch.tensor(
            [
                [math.cos(yaw), math.sin(yaw), 0.0],
                [-math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        rotation = body_to_camera @ ego_to_body
        translation = -rotation @ optical_centre
        projection[:, camera] = intrinsics @ torch.cat((rotation, translation[:, None]), dim=1)
    return projection.to(device=device, dtype=dtype)


def _flashbev_inputs(
    generator: torch.Generator, projection: torch.Tensor, image_size: torch.Tensor
) -> dict[str, torch.Tensor]:
    height, width = FEATURE_LEVEL
    return {
        "projection_matrices": projection,
        "image_size": image_size,
        "image_feats": torch.rand((BATCH, CAMERAS, height, width, CHANNELS), generator=generator),
        "feature_size": torch.tensor([[[height, width]] * CAMERAS], dtype=torch.int32),
    }


def _sca_inputs(
    generator: torch.Generator, projection: torch.Tensor, image_size: torch.Tensor
) -> dict[str, torch.Tensor]:
    grid_x, grid_y, grid_z = GRID
    levels = len(SCA_LEVELS)
    level_start_index = []
    stride = 0
    for height, width in SCA_LEVELS:
        level_start_index.append(stride)
        stride += height * width
    return {
        "projection_matrices": projection,
        "image_size": image_size,
        "value": torch.rand((BATCH, CAMERAS, stride, CHANNELS), generator=generator),
        "spatial_shapes": torch.tensor(SCA_LEVELS, dtype=torch.int32),
        "level_start_index": torch.tensor(level_start_index, dtype=torch.int32),
        "sampling_offsets": torch.randn(
            (BATCH, grid_y, grid_x, SCA_HEADS, levels, SCA_POINTS, grid_z, 2),
            generator=generator,
        )
        * SCA_OFFSET_SCALE,
        "attention_weights": torch.rand(
            (BATCH, grid_y, grid_x, SCA_HEADS, levels, SCA_POINTS, grid_z),
            generator=generator,
        ),
    }


def make_named_case(case_id: str = DEFAULT_CASE, seed: int = DEFAULT_SEED) -> dict:
    """Build the fixture registered for ``case_id`` from ``seed``.

    The tensors are built on the host: the runner moves them to the measured device without
    touching dtype or layout.
    """

    parameters = case_parameters(case_id)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = synthetic_projection()
    image_size = torch.tensor(IMAGE_SIZE, dtype=torch.int32)
    if parameters["op"] == FLASHBEV:
        inputs = _flashbev_inputs(generator, projection, image_size)
    else:
        inputs = _sca_inputs(generator, projection, image_size)
    return {
        "format": FORMAT,
        "case": case_id,
        "seed": seed,
        "op": parameters["op"],
        "direction": parameters["direction"],
        "description": parameters["description"],
        "grid": GRID,
        "roi_range": ROI_RANGE,
        "scalars": {
            "batch": BATCH,
            "cameras": CAMERAS,
            "channels": CHANNELS,
            "image_size": list(IMAGE_SIZE),
            "feature_level": list(FEATURE_LEVEL),
            "levels": [list(level) for level in SCA_LEVELS],
            "heads": SCA_HEADS,
            "points": SCA_POINTS,
        },
        "inputs": inputs,
    }


def make_case(seed: int = DEFAULT_SEED) -> dict:
    """Build the canonical forward FlashBEV case; the public helper of this module."""

    return make_named_case(DEFAULT_CASE, seed=seed)
