"""Canonical multi-scale deformable attention op."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..runtime.native import NativeModuleUnavailable, native_module_proxy
from ..runtime.kernel_dispatch import (
    DispatchDecision,
    DispatchReport,
    explain_dispatch as _explain_dispatch,
    record_dispatch_report,
)

OPERATOR_NAME = "multi_scale_deformable_attn"


def _native_extension():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = (
            value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        )
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def multi_scale_deformable_attn_pytorch_compat(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    level_sizes = value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]
    level_start_index = torch.cat([level_sizes.new_zeros(1), level_sizes.cumsum(0)[:-1]])
    return multi_scale_deformable_attn(
        value,
        value_spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
    )


def _to_contiguous(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(t.contiguous() for t in tensors)


def _requires_native_backend(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> bool:
    return any(
        tensor.device.type in {"musa", "privateuseone"} or bool(getattr(tensor, "is_musa", False))
        for tensor in (
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
    )


class _NativeMSDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        im2col_step: int,
    ) -> torch.Tensor:
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        ctx.im2col_step = im2col_step
        ctx.value_dtype = value.dtype
        ctx.sampling_dtype = sampling_locations.dtype
        ctx.attention_dtype = attention_weights.dtype

        value_fp32, sampling_fp32, attn_fp32 = _to_contiguous(
            value.float(), sampling_locations.float(), attention_weights.float()
        )
        output = value_fp32.new_empty(
            (value_fp32.size(0), sampling_fp32.size(1), value_fp32.size(2) * value_fp32.size(3))
        )
        ext.ms_deform_attn_forward(
            value_fp32,
            spatial_shapes.contiguous(),
            level_start_index.contiguous(),
            sampling_fp32,
            attn_fp32,
            output,
            im2col_step,
        )
        ctx.save_for_backward(
            value_fp32,
            spatial_shapes.contiguous(),
            level_start_index.contiguous(),
            sampling_fp32,
            attn_fp32,
        )
        return output.to(dtype=ctx.value_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = (
            ctx.saved_tensors
        )
        grad_value = torch.zeros_like(value)
        grad_sampling = torch.zeros_like(sampling_locations)
        grad_attention = torch.zeros_like(attention_weights)
        ext.ms_deform_attn_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.float().contiguous(),
            grad_value,
            grad_sampling,
            grad_attention,
            ctx.im2col_step,
        )
        return (
            grad_value.to(dtype=ctx.value_dtype),
            None,
            None,
            grad_sampling.to(dtype=ctx.sampling_dtype),
            grad_attention.to(dtype=ctx.attention_dtype),
            None,
        )


def _normalize_inputs(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if spatial_shapes.dtype != torch.int64:
        raise TypeError("spatial_shapes must be int64")
    if level_start_index.dtype != torch.int64:
        raise TypeError("level_start_index must be int64")
    if value.dtype not in {torch.float16, torch.float32}:
        raise TypeError("value must be float16 or float32")
    if sampling_locations.dtype != value.dtype:
        sampling_locations = sampling_locations.to(dtype=value.dtype)
    if attention_weights.dtype != value.dtype:
        attention_weights = attention_weights.to(dtype=value.dtype)
    return _to_contiguous(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights
    )


def multi_scale_deformable_attn(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    *,
    im2col_step: int = 64,
    backend: str = "auto",
) -> torch.Tensor:
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = (
        _normalize_inputs(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
    )
    native_available = _native_extension() is not None
    decision = _explain_dispatch(value, backend=backend, native_available=native_available)
    if decision.backend == "error":
        record_dispatch_report(
            DispatchReport(
                operator=OPERATOR_NAME,
                requested_backend=backend,
                selected_backend="error",
                device=decision.device,
                reason=decision.reason,
                native_available=native_available,
            )
        )
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        record_dispatch_report(
            DispatchReport(
                operator=OPERATOR_NAME,
                requested_backend=backend,
                selected_backend="reference",
                device=decision.device,
                reason=decision.reason,
                native_available=native_available,
            )
        )
        return multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
    if not _requires_native_backend(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights
    ):
        record_dispatch_report(
            DispatchReport(
                operator=OPERATOR_NAME,
                requested_backend=backend,
                selected_backend="reference",
                device=decision.device,
                reason="inputs do not require the native backend",
                native_available=native_available,
            )
        )
        return multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
    if sampling_locations.size(1) == 0:
        output = value.new_zeros((value.size(0), 0, value.size(2) * value.size(3)))
        record_dispatch_report(
            DispatchReport(
                operator=OPERATOR_NAME,
                requested_backend=backend,
                selected_backend="musa",
                device=decision.device,
                reason=decision.reason,
                native_available=native_available,
            )
        )
        return output + value.sum() * 0 + sampling_locations.sum() * 0 + attention_weights.sum() * 0
    try:
        output = _NativeMSDAFunction.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        record_dispatch_report(
            DispatchReport(
                operator=OPERATOR_NAME,
                requested_backend=backend,
                selected_backend="musa",
                device=decision.device,
                reason=decision.reason,
                native_available=native_available,
            )
        )
        return output
    except RuntimeError as exc:
        if backend != "auto":
            record_dispatch_report(
                DispatchReport(
                    operator=OPERATOR_NAME,
                    requested_backend=backend,
                    selected_backend="error",
                    device=decision.device,
                    reason=str(exc),
                    native_available=native_available,
                )
            )
            raise
        record_dispatch_report(
            DispatchReport(
                operator=OPERATOR_NAME,
                requested_backend=backend,
                selected_backend="reference",
                device=decision.device,
                reason="native backend failed; falling back to PyTorch reference",
                native_available=native_available,
                native_fallback=True,
                fallback_reason=str(exc),
            )
        )
        return multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )


def explain_dispatch(
    value: torch.Tensor,
    *,
    backend: str = "auto",
) -> DispatchDecision:
    return _explain_dispatch(
        value, backend=backend, native_available=_native_extension() is not None
    )
