"""Deterministic ``hard_voxelize`` fixtures for the parallel (``deterministic=False``) path.

A case id is stable across releases: it always names the same point count, feature
width, voxel grid, voxel budget and point distribution, all built from the documented
seed.  The fixtures are synthetic and redistribute no data set.

The parameters mirror two production consumers and one small smoke workload:

* ``pillars``  - PointPillars-style nuScenes voxel configuration at a typical
  full-frame point count;
* ``dense3d``  - a scan-dominated dense 3-D grid that stresses the occupancy scan;
* ``ci-small`` - a small cubic grid dominated by launcher-side cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FORMAT = "hard_voxelize_synthetic_v1"
DEFAULT_SEED = 2333

#: The operator boundary measured here is the parallel path.
DETERMINISTIC = False


@dataclass(frozen=True)
class HardVoxelizeCase:
    """One registered workload."""

    name: str
    num_points: int
    features: int
    voxel: tuple[float, float, float]
    bounds: tuple[float, float, float, float, float, float]
    max_points: int
    max_voxels: int
    description: str

    @property
    def grid(self) -> tuple[int, int, int]:
        """Dense grid the voxel size and range imply."""

        return tuple(
            int(round((self.bounds[axis + 3] - self.bounds[axis]) / self.voxel[axis]))
            for axis in range(3)
        )

    @property
    def grid_cells(self) -> int:
        x, y, z = self.grid
        return x * y * z


#: Case ids are stable across releases; declaration order is the published order.
CASES: dict[str, HardVoxelizeCase] = {
    "pillars": HardVoxelizeCase(
        name="pillars",
        num_points=120_000,
        features=4,
        voxel=(0.16, 0.16, 4.0),
        bounds=(0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
        max_points=32,
        max_voxels=16_000,
        description=(
            "PointPillars nuScenes voxel configuration (0.16 x 0.16 x 4.0 m, range "
            "x 0..69.12, y -39.68..39.68, z -3..1; grid 432 x 496 x 1) at a typical "
            "full-frame point count"
        ),
    ),
    "dense3d": HardVoxelizeCase(
        name="dense3d",
        num_points=500_000,
        features=4,
        voxel=(0.2, 0.2, 0.2),
        bounds=(-51.2, -51.2, -4.0, 51.2, 51.2, 4.0),
        max_points=5,
        max_voxels=60_000,
        description=(
            "Dense 3-D grid (0.2 m cube, range -51.2..51.2 in x and y, z -4..4; "
            "grid 512 x 512 x 40, 10.5M cells) with a scan-dominated point count"
        ),
    ),
    "ci-small": HardVoxelizeCase(
        name="ci-small",
        num_points=30_000,
        features=4,
        voxel=(1.0, 1.0, 1.0),
        bounds=(0.0, 0.0, 0.0, 64.0, 64.0, 64.0),
        max_points=3,
        max_voxels=20_000,
        description=(
            "Small cubic grid (1 m cube, range 0..64 on every axis; grid 64 x 64 x 64) "
            "used as the low-work smoke workload"
        ),
    ),
}
DEFAULT_CASE = "pillars"


def case_ids() -> tuple[str, ...]:
    """Return the canonical case ids in declaration order."""

    return tuple(CASES)


def case_parameters(case_id: str = DEFAULT_CASE) -> HardVoxelizeCase:
    """Return the registered parameters of ``case_id``."""

    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown hard_voxelize case {case_id!r}; expected one of {', '.join(case_ids())}"
        ) from None


def make_points(case: HardVoxelizeCase, seed: int = DEFAULT_SEED, device: str = "cpu"):
    """Build the deterministic point cloud of ``case`` from ``seed``.

    The first three columns are uniform inside the configured range, so every point
    is a candidate voxel member; the remaining feature columns are standard normal.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    low = torch.tensor(case.bounds[:3])
    high = torch.tensor(case.bounds[3:])
    points = (
        torch.rand((case.num_points, 3), generator=generator, dtype=torch.float32) * (high - low)
        + low
    )
    if case.features > 3:
        points = torch.cat(
            [
                points,
                torch.randn(
                    (case.num_points, case.features - 3), generator=generator, dtype=torch.float32
                ),
            ],
            dim=1,
        )
    return points.to(device)


def make_buffers(case: HardVoxelizeCase, points):
    """Allocate the output and workspace tensors the native launcher writes into.

    Everything here is allocated once, before the timed region, exactly as the
    production ``mu_space.ops.hard_voxelize.hard_voxelize`` wrapper allocates it.
    """

    device = points.device
    voxels = torch.zeros(
        (case.max_voxels, case.max_points, case.features), dtype=torch.float32, device=device
    )
    coors = torch.zeros((case.max_voxels, 3), dtype=torch.int32, device=device)
    counts = torch.zeros((case.max_voxels,), dtype=torch.int32, device=device)
    voxel_num = torch.zeros((), dtype=torch.int32, device=device)
    coord_map = torch.empty((case.grid_cells,), dtype=torch.int32, device=device)
    coord_offsets = torch.empty((case.grid_cells,), dtype=torch.int32, device=device)
    point_keys = torch.empty((case.num_points,), dtype=torch.int32, device=device)
    return {
        "voxels": voxels,
        "coors": coors,
        "counts": counts,
        "voxel_num": voxel_num,
        "coord_map": coord_map,
        "coord_offsets": coord_offsets,
        "point_keys": point_keys,
    }


def make_named_case(case_id: str = DEFAULT_CASE, seed: int = DEFAULT_SEED, device: str = "cpu"):
    """Build the fixture registered for ``case_id`` and its preallocated buffers."""

    case = case_parameters(case_id)
    points = make_points(case, seed=seed, device=device)
    return {
        "format": FORMAT,
        "case": case_id,
        "seed": seed,
        "case_parameters": case,
        "points": points,
        "buffers": make_buffers(case, points),
    }
