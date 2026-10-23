"""Deterministic sparse-convolution cases used by the benchmark runner.

A case id is stable across releases: it always names the same sparse shape, active
voxel count, channel pair, kernel, stride and submanifold flag, and every tensor is
built from the documented seed.  The cases are synthetic; they preserve the coordinate
layout and the channel shapes of the model and redistribute no data set.

The rulebook (indice pairs) is generated here, on the host, before the timed region:
the published operator boundary is the convolution itself, not pair construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mu_space.ops.spconv import get_indice_pairs_3d

FORMAT = "spconv_synthetic_v1"
DEFAULT_SEED = 2333


@dataclass(frozen=True)
class SpconvCase:
    """One registered sparse-convolution workload.

    ``spatial_shape`` is ``(x, y, z)``; the generated indices use the spconv
    ``[batch, z, y, x]`` order.
    """

    name: str
    batch_size: int
    spatial_shape: tuple[int, int, int]
    active_per_batch: int
    channels_in: int
    channels_out: int
    kernel_size: tuple[int, int, int]
    stride: tuple[int, int, int]
    padding: tuple[int, int, int]
    subm: bool = False
    dense_active: bool = False
    description: str = ""

    @property
    def active_voxels(self) -> int:
        return self.batch_size * self.active_per_batch

    @property
    def grid_cells(self) -> int:
        x, y, z = self.spatial_shape
        return self.batch_size * x * y * z


#: Case ids are stable across releases; declaration order is the published order.
CASES: dict[str, SpconvCase] = {
    "bevfusion": SpconvCase(
        "bevfusion",
        1,
        (128, 128, 41),
        8192,
        128,
        128,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        description=(
            "BEVFusion LiDAR SparseEncoder shape: 128 x 128 BEV plane with 41 z levels, "
            "8192 of 671088 cells active, 128 -> 128 channels, 3x3x3 submanifold "
            "convolution at stride 1"
        ),
    ),
    "dense-subm-c64": SpconvCase(
        "dense-subm-c64",
        1,
        (8, 8, 8),
        512,
        64,
        64,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        True,
        True,
        description=(
            "Fully occupied 8 x 8 x 8 submanifold cube: all 512 cells active, "
            "64 -> 64 channels, 3x3x3 submanifold convolution at stride 1"
        ),
    ),
}
DEFAULT_CASE = "bevfusion"


def case_ids() -> tuple[str, ...]:
    """Return the canonical case ids in declaration order."""

    return tuple(CASES)


def case_parameters(case_id: str = DEFAULT_CASE) -> SpconvCase:
    """Return the registered parameters of ``case_id``."""

    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f"unknown spconv case {case_id!r}; expected one of {', '.join(case_ids())}"
        ) from None


def make_case(
    name: str = DEFAULT_CASE,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = DEFAULT_SEED,
):
    """Build the case ``name`` from ``seed`` and return it with its device tensors.

    Returns ``(case, tensors, out_shape)`` where ``tensors`` is
    ``(features, filters, outids, indice_pairs, indice_pair_num)``.
    """

    case = case_parameters(name)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = case.batch_size * case.spatial_shape[0] * case.spatial_shape[1] * case.spatial_shape[2]
    active = case.batch_size * case.active_per_batch
    if active > total:
        raise ValueError("active_per_batch exceeds the configured sparse volume")
    linear = (
        torch.arange(total) if case.dense_active else torch.randperm(total, generator=generator)
    )[:active]
    plane = case.spatial_shape[1] * case.spatial_shape[2]
    batch = linear // (case.spatial_shape[0] * plane)
    remainder = linear % (case.spatial_shape[0] * plane)
    z = remainder // plane
    remainder = remainder % plane
    y = remainder // case.spatial_shape[2]
    x = remainder % case.spatial_shape[2]
    indices = torch.stack((batch, z, y, x), dim=1).to(dtype=torch.int32)
    indices = indices[indices[:, 0].argsort(stable=True)]
    features = torch.randn((active, case.channels_in), generator=generator, dtype=dtype)
    filters = torch.randn(
        (
            case.kernel_size[0] * case.kernel_size[1] * case.kernel_size[2],
            case.channels_in,
            case.channels_out,
        ),
        generator=generator,
        dtype=dtype,
    )
    out_shape = tuple(
        (n + 2 * p - k) // s + 1
        for n, p, k, s in zip(case.spatial_shape, case.padding, case.kernel_size, case.stride)
    )
    pairs = get_indice_pairs_3d(
        indices,
        case.batch_size,
        out_shape,
        case.spatial_shape,
        case.kernel_size,
        case.stride,
        case.padding,
        (1, 1, 1),
        (0, 0, 0),
        int(case.subm),
        0,
    )
    return case, tuple(t.to(device=device) for t in (features, filters, *pairs)), out_shape
