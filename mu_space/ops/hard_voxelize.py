"""OpenMMLab-compatible hard and dynamic voxelization."""

from __future__ import annotations

from threading import Lock
from typing import Sequence

import torch
from torch import nn

from ..runtime.kernel_dispatch import BackendName, DispatchReport, explain_dispatch, record_dispatch_report
from ..runtime.native import NativeModuleUnavailable, native_module_proxy

OPERATOR_NAME = "hard_voxelize"
_WORKSPACE_LOCK = Lock()
_COORD_MAP_WORKSPACES: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_PARAM_TENSORS: dict[tuple[str, int, torch.dtype, tuple[float, ...], tuple[float, ...]], tuple[torch.Tensor, torch.Tensor]] = {}


def _coord_map_workspace(points: torch.Tensor, grid_total: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reusable dense-map, scan and point-key workspaces for the native path."""
    index = -1 if points.device.index is None else points.device.index
    key = (points.device.type, index)
    with _WORKSPACE_LOCK:
        workspace = _COORD_MAP_WORKSPACES.get(key)
        required_points = points.shape[0]
        if (workspace is None or workspace[0].numel() < grid_total
                or workspace[2].numel() < required_points):
            workspace = (
                torch.empty((grid_total,), dtype=torch.int32, device=points.device),
                torch.empty((grid_total,), dtype=torch.int32, device=points.device),
                torch.empty((required_points,), dtype=torch.int32, device=points.device),
            )
            _COORD_MAP_WORKSPACES[key] = workspace
    return workspace


def _parameter_tensors(points: torch.Tensor, voxel: tuple[float, ...], bounds: tuple[float, ...]):
    index = -1 if points.device.index is None else points.device.index
    key = (points.device.type, index, points.dtype, voxel, bounds)
    with _WORKSPACE_LOCK:
        tensors = _PARAM_TENSORS.get(key)
        if tensors is None:
            tensors = (points.new_tensor(voxel), points.new_tensor(bounds))
            _PARAM_TENSORS[key] = tensors
    return tensors


def _native():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _record(requested: str, decision, available: bool, fallback: bool = False, reason: str | None = None):
    record_dispatch_report(DispatchReport(OPERATOR_NAME, requested, decision.backend, decision.device,
                                          decision.reason, available, fallback, reason))


def _params(voxel_size: Sequence[float], coors_range: Sequence[float]):
    if len(voxel_size) != 3 or len(coors_range) != 6:
        raise ValueError("voxel_size must contain 3 values and coors_range 6 values")
    voxel = tuple(float(v) for v in voxel_size)
    bounds = tuple(float(v) for v in coors_range)
    if any(v <= 0 for v in voxel) or any(bounds[i + 3] <= bounds[i] for i in range(3)):
        raise ValueError("voxel_size must be positive and coors_range must have increasing bounds")
    return voxel, bounds


def hard_voxelize_reference(points: torch.Tensor, voxel_size, coors_range, max_points: int, max_voxels: int):
    voxel, bounds = _params(voxel_size, coors_range)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N, F] with F >= 3")
    if max_points <= 0 or max_voxels <= 0:
        raise ValueError("max_points and max_voxels must be positive")
    n, features = points.shape
    voxels = points.new_zeros((max_voxels, max_points, features))
    coors = torch.zeros((max_voxels, 3), dtype=torch.int32, device=points.device)
    counts = torch.zeros((max_voxels,), dtype=torch.int32, device=points.device)
    if n == 0:
        return voxels[:0], coors[:0], counts[:0]
    size = points.new_tensor(voxel)
    low = points.new_tensor(bounds[:3])
    high = points.new_tensor(bounds[3:])
    grid = torch.floor((high - low) / size + 0.5).to(torch.int64)
    xyz = torch.floor((points[:, :3] - low) / size).to(torch.int64)
    valid = ((xyz >= 0) & (xyz < grid)).all(dim=1)
    mapping: dict[tuple[int, int, int], int] = {}
    for idx in range(n):
        if not bool(valid[idx]):
            continue
        key = tuple(int(v) for v in xyz[idx])
        voxel_idx = mapping.get(key)
        if voxel_idx is None:
            if len(mapping) >= max_voxels:
                continue
            voxel_idx = len(mapping)
            mapping[key] = voxel_idx
            coors[voxel_idx] = torch.tensor((key[2], key[1], key[0]), dtype=torch.int32,
                                             device=points.device)
        count = int(counts[voxel_idx])
        if count < max_points:
            voxels[voxel_idx, count] = points[idx]
            counts[voxel_idx] += 1
    used = len(mapping)
    return voxels[:used], coors[:used], counts[:used]


def dynamic_voxelize_reference(points: torch.Tensor, voxel_size, coors_range):
    voxel, bounds = _params(voxel_size, coors_range)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N, F] with F >= 3")
    size = points.new_tensor(voxel)
    low, high = points.new_tensor(bounds[:3]), points.new_tensor(bounds[3:])
    grid = torch.floor((high - low) / size + 0.5).to(torch.int64)
    coords = torch.floor((points[:, :3] - low) / size).to(torch.int64)
    valid_x = (coords[:, 0] >= 0) & (coords[:, 0] < grid[0])
    valid_y = valid_x & (coords[:, 1] >= 0) & (coords[:, 1] < grid[1])
    valid_z = valid_y & (coords[:, 2] >= 0) & (coords[:, 2] < grid[2])
    result = torch.zeros((points.shape[0], 3), dtype=torch.int32, device=points.device)
    result[valid_z] = torch.stack(
        (coords[valid_z, 2], coords[valid_z, 1], coords[valid_z, 0]), dim=1
    ).to(torch.int32)
    result[~valid_x, 0] = -1
    result[valid_x & ~valid_y, :2] = -1
    result[valid_y & ~valid_z] = -1
    return result


def hard_voxelize(points: torch.Tensor, voxel_size, coors_range, max_points=35, max_voxels=20000,
                  ndim=3, deterministic=True, *, backend: BackendName = "auto"):
    if ndim != 3:
        raise ValueError("only NDim=3 is supported")
    ext = _native()
    decision = explain_dispatch(points, backend=backend, native_available=ext is not None)
    _record(backend, decision, ext is not None)
    if decision.backend == "error":
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        return hard_voxelize_reference(points, voxel_size, coors_range, max_points, max_voxels)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N, F] with F >= 3")
    if points.shape[0] == 0:
        return hard_voxelize_reference(points, voxel_size, coors_range, max_points, max_voxels)
    voxels = points.new_zeros((max_voxels, max_points, points.shape[1]))
    coors = torch.zeros((max_voxels, 3), dtype=torch.int32, device=points.device)
    counts = torch.zeros((max_voxels,), dtype=torch.int32, device=points.device)
    voxel_num = torch.zeros((), dtype=torch.int32, device=points.device)
    voxel, bounds = _params(voxel_size, coors_range)
    grid = tuple(int(round((bounds[axis + 3] - bounds[axis]) / voxel[axis])) for axis in range(3))
    grid_total = grid[0] * grid[1] * grid[2] if not deterministic else 1
    coord_map, coord_offsets, point_keys = _coord_map_workspace(
        points, grid_total
    )
    voxel_tensor, bounds_tensor = _parameter_tensors(points, voxel, bounds)
    ext.hard_voxelize_forward(points.contiguous(), voxel_tensor,
                              bounds_tensor, voxels, coors, counts, voxel_num,
                              coord_map, coord_offsets, point_keys,
                              int(max_points), int(max_voxels),
                              *grid, bool(deterministic))
    # Compatibility callers require compact tensors. Synchronize once here and
    # use a host index; slicing with a device scalar adds a second device-side
    # indexing path and makes wrapper latency unnecessarily variable.
    compact_num = int(voxel_num.cpu())
    return voxels[:compact_num], coors[:compact_num], counts[:compact_num]


def dynamic_voxelize(points: torch.Tensor, voxel_size, coors_range, ndim=3, *, backend: BackendName = "auto"):
    if ndim != 3:
        raise ValueError("only NDim=3 is supported")
    ext = _native()
    decision = explain_dispatch(points, backend=backend, native_available=ext is not None)
    _record(backend, decision, ext is not None)
    if decision.backend == "error":
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        return dynamic_voxelize_reference(points, voxel_size, coors_range)
    if points.shape[0] == 0:
        return dynamic_voxelize_reference(points, voxel_size, coors_range)
    coors = torch.zeros((points.shape[0], 3), dtype=torch.int32, device=points.device)
    ext.dynamic_voxelize_forward(points.contiguous(), points.new_tensor(voxel_size),
                                 points.new_tensor(coors_range), coors)
    return coors


voxelization = hard_voxelize


class Voxelization(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000, deterministic=True):
        super().__init__()
        self.voxel_size, self.point_cloud_range = voxel_size, point_cloud_range
        self.max_num_points, self.deterministic = max_num_points, deterministic
        self.max_voxels = max_voxels if isinstance(max_voxels, tuple) else (max_voxels, max_voxels)
        pcr = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.grid_size = torch.floor((pcr[3:] - pcr[:3]) / torch.tensor(voxel_size) + 0.5).long()
        self.pcd_shape = [int(self.grid_size[0]), int(self.grid_size[1]), 1]

    def forward(self, points):
        limit = self.max_voxels[0] if self.training else self.max_voxels[1]
        if self.max_num_points == -1 or limit == -1:
            return dynamic_voxelize(points, self.voxel_size, self.point_cloud_range)
        return hard_voxelize(points, self.voxel_size, self.point_cloud_range,
                             self.max_num_points, limit)
