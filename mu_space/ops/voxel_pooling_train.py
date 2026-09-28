"""BEVDepth training voxel pooling with MUSA and PyTorch backends."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..runtime.kernel_dispatch import (
    BackendName,
    DispatchDecision,
    DispatchReport,
    explain_dispatch as _explain_dispatch,
    record_dispatch_report,
)
from ..runtime.native import NativeModuleUnavailable, native_module_proxy

OPERATOR_NAME = "voxel_pooling_train"


def _native_extension():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _voxel_shape(voxel_num: Sequence[int] | torch.Tensor) -> tuple[int, int, int]:
    if isinstance(voxel_num, torch.Tensor):
        if voxel_num.ndim != 1 or voxel_num.numel() != 3:
            raise ValueError("voxel_num must have shape [3]")
        if voxel_num.dtype not in {torch.int32, torch.int64}:
            raise TypeError("voxel_num must use int32 or int64")
        values = tuple(int(v) for v in voxel_num.detach().cpu().tolist())
    else:
        if len(voxel_num) != 3:
            raise ValueError("voxel_num must contain [X, Y, Z]")
        try:
            values = tuple(int(v) for v in voxel_num)
        except (TypeError, ValueError) as exc:
            raise TypeError("voxel_num must contain integer values") from exc
    if any(v <= 0 for v in values):
        raise ValueError("voxel_num values must be positive")
    return values  # type: ignore[return-value]


def _normalize_inputs(
    geom_xyz: torch.Tensor,
    input_features: torch.Tensor,
    voxel_num: Sequence[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    if geom_xyz.dtype != torch.int32:
        raise TypeError("geom_xyz must use int32")
    if geom_xyz.ndim < 3 or geom_xyz.shape[-1] != 3:
        raise ValueError("geom_xyz must have shape [B, ..., 3]")
    if input_features.dtype not in {torch.float16, torch.float32}:
        raise TypeError("input_features must use float16 or float32")
    if input_features.ndim != geom_xyz.ndim:
        raise ValueError("geom_xyz and input_features must have matching point dimensions")
    if input_features.shape[:-1] != geom_xyz.shape[:-1]:
        raise ValueError("geom_xyz and input_features must have matching point dimensions")
    geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, 3)
    input_features = input_features.reshape(input_features.shape[0], -1, input_features.shape[-1])
    shape = _voxel_shape(voxel_num)
    return geom_xyz.contiguous(), input_features.contiguous(), shape


def voxel_pooling_train_reference(
    geom_xyz: torch.Tensor,
    input_features: torch.Tensor,
    voxel_num: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    geom_xyz, input_features, (x_size, y_size, z_size) = _normalize_inputs(
        geom_xyz, input_features, voxel_num
    )
    batch, points, channels = input_features.shape
    output = input_features.new_zeros((batch, y_size, x_size, channels))
    if points == 0:
        return output.permute(0, 3, 1, 2).contiguous() + input_features.sum() * 0
    valid = (
        (geom_xyz[..., 0] >= 0)
        & (geom_xyz[..., 0] < x_size)
        & (geom_xyz[..., 1] >= 0)
        & (geom_xyz[..., 1] < y_size)
        & (geom_xyz[..., 2] >= 0)
        & (geom_xyz[..., 2] < z_size)
    )
    flat_geom = geom_xyz.reshape(-1, 3)
    flat_features = input_features.reshape(-1, channels)
    flat_batch = torch.arange(batch, device=geom_xyz.device).repeat_interleave(points)
    valid_flat = valid.reshape(-1)
    indices = (
        flat_batch[valid_flat] * y_size + flat_geom[valid_flat, 1].long()
    ) * x_size + flat_geom[valid_flat, 0].long()
    flat_output = output.view(batch * y_size * x_size, channels)
    if indices.numel():
        flat_output.index_add_(0, indices, flat_features[valid_flat])
    # Keep the zero output attached to input_features even when every point is
    # outside the voxel grid, matching BEVDepth's backward contract.
    return output.permute(0, 3, 1, 2).contiguous() + input_features.sum() * 0


class _NativeVoxelPoolingTrainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, geom_xyz, input_features, voxel_shape):
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        x_size, y_size, z_size = voxel_shape
        batch, points, channels = input_features.shape
        output = input_features.new_zeros((batch, y_size, x_size, channels))
        pos_memo = torch.full((batch, points, 3), -1, dtype=torch.int32, device=geom_xyz.device)
        ext.voxel_pooling_train_forward(
            geom_xyz, input_features, output, pos_memo, x_size, y_size, z_size
        )
        ctx.save_for_backward(pos_memo)
        ctx.input_shape = input_features.shape
        return output.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        (pos_memo,) = ctx.saved_tensors
        batch, points, channels = ctx.input_shape
        grad_input = grad_output.new_zeros((batch, points, channels))
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        ext.voxel_pooling_train_backward(
            grad_output.permute(0, 2, 3, 1).contiguous(),
            pos_memo,
            grad_input,
            grad_output.shape[-1],
            grad_output.shape[-2],
        )
        return None, grad_input, None


def _record(
    requested_backend,
    decision,
    native_available,
    native_fallback=False,
    fallback_reason=None,
):
    record_dispatch_report(
        DispatchReport(
            operator=OPERATOR_NAME,
            requested_backend=requested_backend,
            selected_backend=decision.backend,
            device=decision.device,
            reason=decision.reason,
            native_available=native_available,
            native_fallback=native_fallback,
            fallback_reason=fallback_reason,
        )
    )


def voxel_pooling_train(
    geom_xyz: torch.Tensor,
    input_features: torch.Tensor,
    voxel_num: Sequence[int] | torch.Tensor,
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    geom_xyz, input_features, shape = _normalize_inputs(geom_xyz, input_features, voxel_num)
    native_available = _native_extension() is not None
    decision = _explain_dispatch(input_features, backend=backend, native_available=native_available)
    if decision.backend == "error":
        _record(backend, decision, native_available)
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        _record(backend, decision, native_available)
        return voxel_pooling_train_reference(geom_xyz, input_features, shape)
    try:
        output = _NativeVoxelPoolingTrainFunction.apply(geom_xyz, input_features, shape)
        _record(backend, decision, native_available)
        return output
    except RuntimeError as exc:
        if backend != "auto":
            error = DispatchDecision("error", decision.device, str(exc))
            _record(backend, error, native_available)
            raise
        fallback = DispatchDecision(
            "reference", decision.device, "native backend failed; falling back to PyTorch reference"
        )
        _record(backend, fallback, native_available, True, str(exc))
        return voxel_pooling_train_reference(geom_xyz, input_features, shape)


__all__ = ["voxel_pooling_train", "voxel_pooling_train_reference"]
