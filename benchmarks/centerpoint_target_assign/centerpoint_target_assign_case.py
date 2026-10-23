"""Deterministic ``centerpoint_target_assign`` benchmark cases.

A case id names the workload shape: ``c<classes>-n<objects>``.  The feature map, voxel size, point
cloud range, size factor, gaussian overlap and minimum radius are shared by every case and are
recorded in the generated payload.  Every published input is built here from the documented seed;
no data set is redistributed.
"""

from __future__ import annotations

import math

import torch

FORMAT = "centerpoint_target_assign_v1"
DEFAULT_SEED = 2333

#: Shared head configuration: the BEVFusion CenterPoint head grid over a nuScenes point range.
FEATURE_MAP_SIZE = (128, 128)
OUT_SIZE_FACTOR = 8
GAUSSIAN_OVERLAP = 0.1
MIN_RADIUS = 2
VOXEL_SIZE = (0.1, 0.1)
PC_RANGE = (-51.2, -51.2)

#: Case ids are stable across releases; declaration order is the published order.
CASES: dict[str, dict] = {
    "c1-n256": {
        "num_classes": 1,
        "objects": 256,
        "max_objs": 256,
        "description": (
            "one class and 256 objects: a single-class head, where the fixed heatmap and "
            "annotation-buffer cost dominates the per-object inner loop"
        ),
    },
    "c10-n256": {
        "num_classes": 10,
        "objects": 256,
        "max_objs": 256,
        "description": (
            "ten classes and 256 objects: the nuScenes class count at a typical full-frame object "
            "count for this head"
        ),
    },
    "c10-n512": {
        "num_classes": 10,
        "objects": 512,
        "max_objs": 500,
        "description": (
            "ten classes and 512 objects against a 500-object assignment buffer: the inner loop "
            "dominates and the buffer bound is reached"
        ),
    },
}
DEFAULT_CASE = "c10-n256"

#: Objects whose centre repeats an earlier object of the same class, so the heatmap max-update
#: collision path is exercised.
COLLISION_OBJECTS = 16
#: Objects placed beyond the point cloud range, so the out-of-map skip path is exercised.
OUTSIDE_OBJECTS = 4


def case_ids() -> tuple[str, ...]:
    """Return the canonical case ids in declaration order."""

    return tuple(CASES)


def case_parameters(case_id: str = DEFAULT_CASE) -> dict:
    """Return the registered parameters of ``case_id``."""

    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown centerpoint_target_assign case {case_id!r}; "
            f"expected one of {', '.join(case_ids())}"
        ) from None


def make_named_case(
    case_id: str = DEFAULT_CASE,
    seed: int = DEFAULT_SEED,
    objects: int = 0,
    num_classes: int = 0,
    max_objs: int = 0,
) -> dict:
    """Build the fixture registered for ``case_id`` from ``seed``.

    The shape switches override one dimension of the registered workload for local investigation
    and re-derive the case id, so a record always names the workload it measured.  Tensors are
    built on the host: the runner moves them to the measured device without touching dtype or
    layout.
    """

    parameters = case_parameters(case_id)
    count = int(objects) or parameters["objects"]
    classes = int(num_classes) or parameters["num_classes"]
    max_objects = int(max_objs) or parameters["max_objs"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    width, height = FEATURE_MAP_SIZE
    span = width * VOXEL_SIZE[0] * OUT_SIZE_FACTOR
    low = PC_RANGE[0] - 0.25 * span
    boxes = torch.empty((count, 9), dtype=torch.float32)
    boxes[:, 0:2] = (
        torch.rand((count, 2), generator=generator, dtype=torch.float32) * (span + 0.5 * span) + low
    )
    boxes[:, 2] = torch.rand((count,), generator=generator, dtype=torch.float32) * 4.0 - 2.0
    boxes[:, 3:6] = torch.rand((count, 3), generator=generator, dtype=torch.float32) * 3.0 + 1.5
    boxes[:, 6:8] = torch.rand((count, 2), generator=generator, dtype=torch.float32) * 4.0 - 2.0
    boxes[:, 8] = (
        torch.rand((count,), generator=generator, dtype=torch.float32) * 2.0 - 1.0
    ) * math.pi
    class_ids = torch.randint(1, classes + 1, (count,), generator=generator, dtype=torch.int64)
    # Deterministic path coverage: repeat the first objects' centres as same-class collisions, and
    # push a few objects beyond the pooled range so the out-of-map branch runs as well.
    collisions = min(COLLISION_OBJECTS, max(count - COLLISION_OBJECTS, 0))
    if collisions:
        boxes[COLLISION_OBJECTS : COLLISION_OBJECTS + collisions, 0:2] = boxes[:collisions, 0:2]
        class_ids[COLLISION_OBJECTS : COLLISION_OBJECTS + collisions] = class_ids[:collisions]
    outside = min(OUTSIDE_OBJECTS, count)
    if outside:
        boxes[count - outside :, 0] = PC_RANGE[0] - span
        boxes[count - outside :, 1] = PC_RANGE[0] - span
    return {
        "format": FORMAT,
        "case": f"c{classes}-n{count}",
        "seed": seed,
        "num_classes": classes,
        "objects": count,
        "max_objs": max_objects,
        "description": parameters["description"],
        "parameters": {
            "num_classes": classes,
            "out_size_factor": OUT_SIZE_FACTOR,
            "gaussian_overlap": GAUSSIAN_OVERLAP,
            "min_radius": MIN_RADIUS,
            "voxel_size": VOXEL_SIZE,
            "pc_range": PC_RANGE,
            "feature_map_size": FEATURE_MAP_SIZE,
            "max_objs": max_objects,
        },
        "inputs": {"boxes": boxes, "class_ids": class_ids},
    }


def make_case(
    seed: int = DEFAULT_SEED, objects: int = 256, num_classes: int = 10, max_objs: int = 256
) -> dict:
    """Build the canonical fixture; the public helper of this module."""

    return make_named_case(
        DEFAULT_CASE, seed=seed, objects=objects, num_classes=num_classes, max_objs=max_objs
    )
