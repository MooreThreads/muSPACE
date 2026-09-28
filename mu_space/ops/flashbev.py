"""FlashBEV and FlashSCA sampling view-transform operators."""

from __future__ import annotations

from typing import Sequence

import torch

from ..runtime.native import NativeModuleUnavailable, native_module_proxy
from ..runtime.kernel_dispatch import (
    BackendName,
    DispatchDecision,
    DispatchReport,
    explain_dispatch,
    record_dispatch_report,
)

OPERATOR_NAME = "flash_bevpool"
SCA_OPERATOR_NAME = "flash_bevpool_sca"


def _native_extension():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _triple(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _grid(values: Sequence[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("grid_size must contain [x, y, z]")
    result = tuple(int(value) for value in values)
    if any(value <= 0 for value in result):
        raise ValueError("grid_size values must be positive")
    return result  # type: ignore[return-value]


def _check_float(tensor: torch.Tensor, name: str) -> None:
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must use float32")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_int(tensor: torch.Tensor, name: str) -> None:
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must use int32")


def _sanitize_value(value: torch.Tensor) -> torch.Tensor:
    """Replace Inf / NaN in *value* with 0.0 so the kernel produces deterministic
    finite results regardless of the GPU's floating-point rounding mode for
    :math:`0 \\times \\infty` and :math:`\\infty - \\infty`.

    The reference path follows IEEE 754 semantics exactly; MUSA hardware may
    produce a different mix of Inf and NaN when the input contains special
    values, so sanitising early avoids non-deterministic Inf-sign mismatches in
    correctness tests (see :file:`tests/ops/test_flashbev.py`).
    """
    if torch.isfinite(value).all():
        return value
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value))


def _metadata(tensor: torch.Tensor, name: str) -> torch.Tensor:
    _check_int(tensor, name)
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _record(
    name: str,
    requested: str,
    decision: DispatchDecision,
    native: bool,
    fallback: bool = False,
    reason: str | None = None,
) -> None:
    record_dispatch_report(
        DispatchReport(
            operator=name,
            requested_backend=requested,
            selected_backend=decision.backend,
            device=decision.device,
            reason=decision.reason,
            native_available=native,
            native_fallback=fallback,
            fallback_reason=reason,
        )
    )


def _project(
    projection: torch.Tensor, x: float, y: float, z: float, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor] | None:
    depth = projection[2, 0] * x + projection[2, 1] * y + projection[2, 2] * z + projection[2, 3]
    if bool(depth.detach().item() <= 0.0):
        return None
    safe = torch.clamp(depth, min=epsilon)
    u = (
        projection[0, 0] * x + projection[0, 1] * y + projection[0, 2] * z + projection[0, 3]
    ) / safe
    v = (
        projection[1, 0] * x + projection[1, 1] * y + projection[1, 2] * z + projection[1, 3]
    ) / safe
    return u, v


def _bilinear(
    feat: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, padding_mode: str = "zero"
) -> torch.Tensor:
    height, width = feat.shape[:2]
    u0 = torch.floor(u).to(torch.long)
    v0 = torch.floor(v).to(torch.long)
    u1, v1 = u0 + 1, v0 + 1
    du, dv = u - u0, v - v0

    def get(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        safe_row = row.clamp(0, height - 1)
        safe_col = col.clamp(0, width - 1)
        result = feat[safe_row, safe_col]
        return result if padding_mode == "border" else result * valid.to(feat.dtype)

    f00, f01 = get(v0, u0), get(v1, u0)
    f10, f11 = get(v0, u1), get(v1, u1)
    return (1 - du) * (1 - dv) * f00 + (1 - du) * dv * f01 + du * (1 - dv) * f10 + du * dv * f11


def flash_bevpool_reference(
    image_feats: torch.Tensor,
    projection_matrices: torch.Tensor,
    feature_size: torch.Tensor,
    image_size: torch.Tensor,
    grid_size: Sequence[int],
    voxel_size: Sequence[float],
    voxel_min: Sequence[float],
    epsilon: float = 1e-5,
) -> torch.Tensor:
    image_feats = _sanitize_value(image_feats)
    gx, gy, gz = _grid(grid_size)
    sx, sy, sz = _triple(voxel_size, "voxel_size")
    xmin, ymin, zmin = _triple(voxel_min, "voxel_min")
    batch, cameras, _, _, channels = image_feats.shape
    image_h, image_w = (int(value) for value in image_size.detach().cpu().tolist())
    values = []
    for b in range(batch):
        for y_index in range(gy):
            for x_index in range(gx):
                x = xmin + (x_index + 0.5) * sx
                y = ymin + (y_index + 0.5) * sy
                for channel in range(channels):
                    result = image_feats.new_zeros(())
                    for z_index in range(gz):
                        z = zmin + (z_index + 0.5) * sz
                        samples = []
                        for camera in range(cameras):
                            projected = _project(
                                projection_matrices[b, camera].view(3, 4), x, y, z, epsilon
                            )
                            if projected is None:
                                continue
                            u, v = projected
                            if bool(
                                (u < 0).detach().item()
                                or (u >= image_w).detach().item()
                                or (v < 0).detach().item()
                                or (v >= image_h).detach().item()
                            ):
                                continue
                            fh, fw = (
                                int(value)
                                for value in feature_size[b, camera].detach().cpu().tolist()
                            )
                            fu, fv = u / image_w * fw, v / image_h * fh
                            if bool(
                                (fu < 0).detach().item()
                                or (fu >= fw).detach().item()
                                or (fv < 0).detach().item()
                                or (fv >= fh).detach().item()
                            ):
                                continue
                            samples.append(
                                _bilinear(image_feats[b, camera], fu, fv, padding_mode="border")[
                                    channel
                                ]
                            )
                        if samples:
                            result = result + torch.stack(samples).sum() / len(samples)
                    values.append(result)
    return torch.stack(values).view(batch, gy, gx, channels)


class _FlashBEVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image_feats, projections, feature_size, image_size, grid, voxel, epsilon):
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        gx, gy, gz = grid
        sx, sy, sz = voxel[:3]
        xmin, ymin, zmin = voxel[3:]
        output = image_feats.new_zeros((image_feats.size(0), gy, gx, image_feats.size(4)))
        ext.flash_bevpool_forward(
            image_feats,
            projections,
            feature_size,
            image_size,
            output,
            gx,
            gy,
            gz,
            sx,
            sy,
            sz,
            xmin,
            ymin,
            zmin,
            epsilon,
        )
        ctx.save_for_backward(
            image_feats,
            projections,
            feature_size,
            image_size,
            image_feats.new_tensor(
                [xmin, xmin + gx * sx, ymin, ymin + gy * sy, zmin, zmin + gz * sz]
            ),
        )
        ctx.grid = grid
        ctx.epsilon = epsilon
        ctx.image_shape = tuple(image_feats.shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        image_feats, projections, feature_size, image_size, roi_range = ctx.saved_tensors
        ext = _native_extension()
        grad_image = torch.zeros_like(image_feats)
        gx, gy, gz = ctx.grid
        ext.flash_bevpool_backward(
            grad_output.contiguous(),
            image_feats,
            projections,
            feature_size,
            image_size,
            roi_range,
            grad_image,
            gx,
            gy,
            gz,
            ctx.epsilon,
        )
        return grad_image, None, None, None, None, None, None


def _resolve_voxels(grid_size, voxel_size, voxel_min, roi_range):
    grid = _grid(grid_size)
    if roi_range is not None:
        if voxel_size is not None or voxel_min is not None:
            raise ValueError("pass either roi_range or voxel_size/voxel_min")
        if len(roi_range) != 6:
            raise ValueError("roi_range must contain [xmin,xmax,ymin,ymax,zmin,zmax]")
        values = (
            [float(item) for item in roi_range.detach().cpu().tolist()]
            if isinstance(roi_range, torch.Tensor)
            else [float(item) for item in roi_range]
        )
        return grid, (
            (values[1] - values[0]) / grid[0],
            (values[3] - values[2]) / grid[1],
            (values[5] - values[4]) / grid[2],
            values[0],
            values[2],
            values[4],
        )
    if voxel_size is None or voxel_min is None:
        raise ValueError("voxel_size and voxel_min are required when roi_range is absent")
    return grid, _triple(voxel_size, "voxel_size") + _triple(voxel_min, "voxel_min")


def flash_bevpool(
    image_feats,
    projection_matrices,
    feature_size,
    image_size,
    grid_size,
    voxel_size=None,
    voxel_min=None,
    *,
    roi_range=None,
    epsilon=1e-5,
    backend: BackendName = "auto",
):
    _check_float(image_feats, "image_feats")
    _check_float(projection_matrices, "projection_matrices")
    feature_size = _metadata(feature_size, "feature_size")
    image_size = _metadata(image_size, "image_size")
    if image_feats.dim() != 5 or projection_matrices.shape != (
        image_feats.size(0),
        image_feats.size(1),
        3,
        4,
    ):
        raise ValueError("image_feats/projection_matrices shapes must be [B,N,H,W,C]/[B,N,3,4]")
    if feature_size.shape != (image_feats.size(0), image_feats.size(1), 2):
        raise ValueError("feature_size must have shape [B,N,2]")
    if image_size.numel() != 2:
        raise ValueError("image_size must have shape [2]")
    if bool((feature_size <= 0).any()) or bool((image_size <= 0).any()):
        raise ValueError("feature_size and image_size values must be positive")
    grid, voxel = _resolve_voxels(grid_size, voxel_size, voxel_min, roi_range)
    native = _native_extension() is not None
    decision = explain_dispatch(image_feats, backend=backend, native_available=native)
    if decision.backend == "error":
        _record(OPERATOR_NAME, backend, decision, native)
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        _record(OPERATOR_NAME, backend, decision, native)
        return flash_bevpool_reference(
            _sanitize_value(image_feats),
            projection_matrices,
            feature_size,
            image_size,
            grid,
            voxel[:3],
            voxel[3:],
            epsilon,
        )
    try:
        result = _FlashBEVFunction.apply(
            _sanitize_value(image_feats).contiguous(),
            projection_matrices.contiguous(),
            feature_size,
            image_size,
            grid,
            voxel,
            float(epsilon),
        )
        _record(OPERATOR_NAME, backend, decision, native)
        return result
    except RuntimeError as exc:
        if backend != "auto":
            raise
        fallback = DispatchDecision(
            "reference", decision.device, "native backend failed; falling back to reference"
        )
        _record(OPERATOR_NAME, backend, fallback, native, True, str(exc))
        return flash_bevpool_reference(
            _sanitize_value(image_feats),
            projection_matrices,
            feature_size,
            image_size,
            grid,
            voxel[:3],
            voxel[3:],
            epsilon,
        )


def flash_bevpool_sca_reference(
    value,
    projection_matrices,
    spatial_shapes,
    level_start_index,
    image_size,
    sampling_offsets,
    attention_weights,
    grid_size,
    voxel_size,
    voxel_min,
    *,
    epsilon=1e-5,
    value_stride_n=None,
):
    value = _sanitize_value(value)
    gx, gy, gz = _grid(grid_size)
    sx, sy, sz = _triple(voxel_size, "voxel_size")
    xmin, ymin, zmin = _triple(voxel_min, "voxel_min")
    batch, cameras, _, channels = value.shape
    heads = sampling_offsets.size(3)
    levels = spatial_shapes.size(0)
    points = sampling_offsets.size(5)
    image_h, image_w = (int(item) for item in image_size.detach().cpu().tolist())
    results = []
    for b in range(batch):
        for y_index in range(gy):
            for x_index in range(gx):
                x = xmin + (x_index + 0.5) * sx
                y = ymin + (y_index + 0.5) * sy
                visible = 0
                for camera in range(cameras):
                    for z_index in range(gz):
                        z = zmin + (z_index + 0.5) * sz
                        projected = _project(
                            projection_matrices[b, camera].view(3, 4), x, y, z, epsilon
                        )
                        if projected is not None and bool(
                            (projected[0] >= 0).detach().item()
                            and (projected[0] < image_w).detach().item()
                            and (projected[1] >= 0).detach().item()
                            and (projected[1] < image_h).detach().item()
                        ):
                            visible += 1
                            break
                inv_count = 1.0 / visible if visible else 0.0
                for channel in range(channels):
                    head = channel // (channels // heads)
                    result = value.new_zeros(())
                    for camera in range(cameras):
                        for z_index in range(gz):
                            z = zmin + (z_index + 0.5) * sz
                            projected = _project(
                                projection_matrices[b, camera].view(3, 4), x, y, z, epsilon
                            )
                            if projected is None or bool(
                                (projected[0] < 0).detach().item()
                                or (projected[0] >= image_w).detach().item()
                                or (projected[1] < 0).detach().item()
                                or (projected[1] >= image_h).detach().item()
                            ):
                                continue
                            u, v = projected
                            for level in range(levels):
                                h, w = (
                                    int(item)
                                    for item in spatial_shapes[level].detach().cpu().tolist()
                                )
                                base_u, base_v = u / image_w * w - 0.5, v / image_h * h - 0.5
                                start = int(level_start_index[level].item())
                                for point in range(points):
                                    off = sampling_offsets[
                                        b, y_index, x_index, head, level, point, z_index
                                    ]
                                    sample = _bilinear(
                                        value[b, camera, start : start + h * w].view(
                                            h, w, channels
                                        ),
                                        base_u + off[0],
                                        base_v + off[1],
                                    )[channel]
                                    weight = attention_weights[
                                        b, y_index, x_index, head, level, point, z_index
                                    ]
                                    result = result + weight * sample
                    results.append(result * inv_count)
    return torch.stack(results).view(batch, gy, gx, channels)


class _FlashSCAModule(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        value,
        projections,
        spatial_shapes,
        level_start,
        image_size,
        offsets,
        weights,
        num_heads,
        num_points,
        grid,
        voxel,
        value_stride_n,
        epsilon,
    ):
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        gx, gy, gz = grid
        sx, sy, sz = voxel[:3]
        xmin, ymin, zmin = voxel[3:]
        output = value.new_zeros((value.size(0), gy, gx, value.size(3)))
        ext.flash_bevpool_sca_forward(
            value,
            projections,
            spatial_shapes,
            level_start,
            image_size,
            offsets,
            weights,
            output,
            num_heads,
            num_points,
            gx,
            gy,
            gz,
            sx,
            sy,
            sz,
            xmin,
            ymin,
            zmin,
            value_stride_n,
            epsilon,
        )
        roi = value.new_tensor([xmin, xmin + gx * sx, ymin, ymin + gy * sy, zmin, zmin + gz * sz])
        ctx.save_for_backward(
            value, projections, spatial_shapes, level_start, image_size, roi, offsets, weights
        )
        ctx.params = (num_heads, num_points, grid, value_stride_n, epsilon)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        value, projections, spatial_shapes, level_start, image_size, roi, offsets, weights = (
            ctx.saved_tensors
        )
        num_heads, num_points, (gx, gy, gz), stride, epsilon = ctx.params
        ext = _native_extension()
        grad_value = torch.zeros_like(value)
        grad_offsets = torch.zeros_like(offsets)
        grad_weights = torch.zeros_like(weights)
        ext.flash_bevpool_sca_backward(
            grad_output.contiguous(),
            value,
            projections,
            spatial_shapes,
            level_start,
            image_size,
            roi,
            offsets,
            weights,
            grad_value,
            grad_offsets,
            grad_weights,
            num_heads,
            num_points,
            gx,
            gy,
            gz,
            stride,
            epsilon,
        )
        return (
            grad_value,
            None,
            None,
            None,
            None,
            grad_offsets,
            grad_weights,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flash_bevpool_sca(
    value,
    projection_matrices,
    spatial_shapes,
    level_start_index,
    image_size,
    sampling_offsets,
    attention_weights,
    grid_size,
    voxel_size=None,
    voxel_min=None,
    *,
    roi_range=None,
    num_heads=None,
    num_points=None,
    value_stride_n=None,
    epsilon=1e-5,
    backend: BackendName = "auto",
):
    for tensor, name in (
        (value, "value"),
        (projection_matrices, "projection_matrices"),
        (sampling_offsets, "sampling_offsets"),
        (attention_weights, "attention_weights"),
    ):
        _check_float(tensor, name)
    for tensor, name in (
        (spatial_shapes, "spatial_shapes"),
        (level_start_index, "level_start_index"),
        (image_size, "image_size"),
    ):
        _check_int(tensor, name)
    if value.dim() != 4 or sampling_offsets.dim() != 8 or attention_weights.dim() != 7:
        raise ValueError("invalid FlashSCA input ranks")
    heads = int(num_heads or sampling_offsets.size(3))
    points = int(num_points or sampling_offsets.size(5))
    if heads <= 0 or points <= 0:
        raise ValueError("num_heads and num_points must be positive")
    if value.size(3) % heads != 0:
        raise ValueError("channels must be divisible by num_heads")
    grid, voxel = _resolve_voxels(grid_size, voxel_size, voxel_min, roi_range)
    gx, gy, gz = grid
    if projection_matrices.shape != (value.size(0), value.size(1), 3, 4):
        raise ValueError("projection_matrices must have shape [B,N,3,4]")
    if spatial_shapes.dim() != 2 or spatial_shapes.size(1) != 2:
        raise ValueError("spatial_shapes must have shape [levels,2]")
    levels = spatial_shapes.size(0)
    if level_start_index.shape != (levels,) or image_size.numel() != 2:
        raise ValueError("level_start_index/image_size shapes must be [levels]/[2]")
    expected_offsets = (value.size(0), gy, gx, heads, levels, points, gz, 2)
    expected_weights = expected_offsets[:-1]
    if tuple(sampling_offsets.shape) != expected_offsets:
        raise ValueError(f"sampling_offsets must have shape {expected_offsets}")
    if tuple(attention_weights.shape) != expected_weights:
        raise ValueError(f"attention_weights must have shape {expected_weights}")
    stride = int(value_stride_n or value.size(2))
    if stride <= 0 or stride > value.size(2):
        raise ValueError("value_stride_n must be in [1, value.size(2)]")
    level_ends = level_start_index + spatial_shapes[:, 0] * spatial_shapes[:, 1]
    if (
        bool((spatial_shapes <= 0).any())
        or bool((level_start_index < 0).any())
        or bool((level_ends > stride).any())
    ):
        raise ValueError("value_stride_n must cover every positive spatial level")
    native = _native_extension() is not None
    decision = explain_dispatch(value, backend=backend, native_available=native)
    if decision.backend == "error":
        _record(SCA_OPERATOR_NAME, backend, decision, native)
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        _record(SCA_OPERATOR_NAME, backend, decision, native)
        return flash_bevpool_sca_reference(
            _sanitize_value(value),
            projection_matrices,
            spatial_shapes,
            level_start_index,
            image_size,
            sampling_offsets,
            attention_weights,
            grid,
            voxel[:3],
            voxel[3:],
            epsilon=epsilon,
            value_stride_n=stride,
        )
    try:
        result = _FlashSCAModule.apply(
            _sanitize_value(value).contiguous(),
            projection_matrices.contiguous(),
            spatial_shapes.contiguous(),
            level_start_index.contiguous(),
            image_size.contiguous(),
            sampling_offsets.contiguous(),
            attention_weights.contiguous(),
            heads,
            points,
            grid,
            voxel,
            stride,
            float(epsilon),
        )
        _record(SCA_OPERATOR_NAME, backend, decision, native)
        return result
    except RuntimeError as exc:
        if backend != "auto":
            raise
        fallback = DispatchDecision(
            "reference", decision.device, "native backend failed; falling back to reference"
        )
        _record(SCA_OPERATOR_NAME, backend, fallback, native, True, str(exc))
        return flash_bevpool_sca_reference(
            _sanitize_value(value),
            projection_matrices,
            spatial_shapes,
            level_start_index,
            image_size,
            sampling_offsets,
            attention_weights,
            grid,
            voxel[:3],
            voxel[3:],
            epsilon=epsilon,
            value_stride_n=stride,
        )
