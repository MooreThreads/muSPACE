"""Modulated deformable convolution (DCNv2) on MUSA.

Two native paths back this operator:

``nchw``
    The general path. ``im2col`` produces the deformable column matrix, a muBLAS GEMM
    applies the weights, and the backward pass runs ``col2im`` / ``col2im_coord``. It
    supports arbitrary ``groups``, ``deform_groups``, kernel size, stride and dilation.

``nhwc``
    The offset-conv-fused path used by :class:`ModulatedDeformConv2dPack`. The offset
    convolution runs through muDNN, the deformable gather reads its raw output directly
    (slicing the offsets and applying the mask sigmoid inside the kernel), and the main
    GEMM consumes channel-last columns. It requires ``groups == deform_groups == 1``,
    a 3x3 kernel, stride 1 and dilation 1 with ``channels % 32 == 0``; anything else
    falls back to the ``nchw`` path.

Environment overrides:

``MUSPACE_DCN_FP32_TF32``
    Default ``0``. Runs the float32 GEMMs in strict IEEE float32, matching pure PyTorch
    reference precision. Set to ``1`` for TF32, which matches the reduced-precision
    compute mode muDNN uses for deformable convolution at the cost of numerical accuracy
    on near-zero and large-magnitude values.
``MUSPACE_DCN_RECOMPUTE_COLUMNS``
    Default ``1``. Recomputes the deformable columns in the backward pass of the fused
    path instead of saving them, which is what makes full BEVFormer batches fit.

The forward pass is deterministic; the backward pass is not bit-reproducible, because
``col2im`` accumulates the input, offset and mask gradients with atomics.

The operator is MUSA-only; the differentiable PyTorch model used to validate it lives in
``tests/ops/test_modulated_deform_conv.py``.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Iterator
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.nn.modules.utils import _pair, _single

from ..runtime.native import NativeModuleUnavailable, native_module_proxy

from ..runtime.kernel_dispatch import (
    BackendName,
    DispatchDecision,
    DispatchReport,
    explain_dispatch as _explain_dispatch,
    record_dispatch_report,
)

OPERATOR_NAME = "modulated_deform_conv2d"


def _native_extension():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _output_size(size: int, pad: int, dilation: int, kernel: int, stride: int) -> int:
    return (size + 2 * pad - (dilation * (kernel - 1) + 1)) // stride + 1


class _tf32_matmul:
    """Control TF32 matmul for float32 tensors, restoring the previous flags on exit.

    Strict float32 GEMM has no tensor-core path on MTT S5000 and dominates the DCNv2
    cost, while muDNN runs its own deformable convolution GEMM in a reduced-precision
    mode. When *enabled*, TF32 matches that mode and is worth several-fold on the
    fused path. When *disabled* (default), the operator uses strict IEEE float32 to
    match pure PyTorch reference precision.

    This is a no-op for half and bfloat16 tensors, which already use tensor cores.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        self._use_tf32 = _env_flag("MUSPACE_DCN_FP32_TF32", False) and dtype == torch.float32
        self._saved: list[tuple[object, bool]] = []

    def __enter__(self) -> "_tf32_matmul":
        for backend_name in ("musa", "cuda"):
            backend = getattr(torch.backends, backend_name, None)
            matmul = getattr(backend, "matmul", None) if backend is not None else None
            if matmul is not None and hasattr(matmul, "allow_tf32"):
                self._saved.append((matmul, matmul.allow_tf32))
                matmul.allow_tf32 = self._use_tf32
        return self

    def __exit__(self, *exc: object) -> bool:
        for matmul, previous in self._saved:
            matmul.allow_tf32 = previous
        return False


@contextlib.contextmanager
def _mudnn_tf32() -> Iterator[None]:
    """Scope TF32 for DCNv2 GEMMs across torch_musa 2.7.1 and 2.9.1.

    torch_musa 2.7.1 exposes the switch only through
    ``torch.backends.mudnn.m.flags`` and defaults it to True. Newer versions,
    including torch_musa 2.9.1, expose the regular
    ``torch.backends.musa.matmul`` switch handled by :class:`_tf32_matmul`.
    The flags also control the offset convolution in the fused path, whose
    reference runs under the default flags, so this must scope only the GEMMs,
    never the conv2d calls.
    """
    use_tf32 = _env_flag("MUSPACE_DCN_FP32_TF32", False)
    mudnn = getattr(torch.backends, "mudnn", None)
    flags = getattr(mudnn, "m", None) if mudnn is not None else None
    if flags is None or not hasattr(flags, "flags"):
        yield
        return
    with flags.flags(allow_tf32=use_tf32):
        yield


# ---------------------------------------------------------------------------
# Native NCHW path
# ---------------------------------------------------------------------------


class _NativeModulatedDeformConv2dFunction(Function):
    """im2col + GEMM forward, col2im/col2im_coord + GEMM backward.

    The columns produced by the native kernel are ``[C * kH * kW, B * Ho * Wo]``, i.e.
    channel major, so the weight GEMM is a single ``mm`` for ``groups == 1`` and the
    only transpose in the operator is on the (much smaller) output tensor.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        deform_groups: int,
    ) -> torch.Tensor:
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")

        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        ctx.groups, ctx.deform_groups = groups, deform_groups
        ctx.with_bias = bias is not None
        ctx.save_for_backward(input, offset, mask, weight)

        batch, channels, height, width = input.shape
        channels_out, kernel_h, kernel_w = weight.shape[0], weight.shape[2], weight.shape[3]
        out_per_group = channels_out // groups
        in_per_group = (channels // groups) * kernel_h * kernel_w

        height_out = _output_size(height, padding[0], dilation[0], kernel_h, stride[0])
        width_out = _output_size(width, padding[1], dilation[1], kernel_w, stride[1])
        rows = batch * height_out * width_out

        columns = input.new_empty((channels * kernel_h * kernel_w, rows))
        ext.modulated_deform_conv_im2col(
            input,
            offset,
            mask,
            kernel_h,
            kernel_w,
            padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
            deform_groups,
            columns,
        )

        columns_g = columns.view(groups, in_per_group, rows)
        weight_g = weight.view(groups, out_per_group, in_per_group)
        with _tf32_matmul(columns.dtype), _mudnn_tf32():
            if groups == 1:
                output_flat = torch.mm(weight_g[0], columns_g[0]).unsqueeze(0)
            else:
                output_flat = torch.empty(
                    groups, out_per_group, rows, dtype=columns.dtype, device=columns.device
                )
                for group in range(groups):
                    torch.mm(weight_g[group], columns_g[group], out=output_flat[group])

        output = (
            output_flat.view(channels_out, batch, height_out, width_out)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        input, offset, mask, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation
        groups, deform_groups = ctx.groups, ctx.deform_groups

        grad_output = grad_output.contiguous()
        batch, channels, height, width = input.shape
        channels_out, kernel_h, kernel_w = weight.shape[0], weight.shape[2], weight.shape[3]
        out_per_group = channels_out // groups
        in_per_group = (channels // groups) * kernel_h * kernel_w

        height_out = _output_size(height, padding[0], dilation[0], kernel_h, stride[0])
        width_out = _output_size(width, padding[1], dilation[1], kernel_w, stride[1])
        rows = batch * height_out * width_out

        # grad_output [B, Cout, Ho, Wo] -> [Cout, B * Ho * Wo] mirrors the column
        # layout, so weight^T @ grad_output yields grad_columns directly.
        grad_out_cm = grad_output.permute(1, 0, 2, 3).contiguous().view(channels_out, rows)
        grad_out_g = grad_out_cm.view(groups, out_per_group, rows)
        weight_g = weight.view(groups, out_per_group, in_per_group)

        grad_columns = input.new_empty(groups, in_per_group, rows)
        with _tf32_matmul(input.dtype), _mudnn_tf32():
            for group in range(groups):
                torch.mm(weight_g[group].t(), grad_out_g[group], out=grad_columns[group])
        grad_columns = grad_columns.view(channels * kernel_h * kernel_w, rows)

        geometry = (
            kernel_h,
            kernel_w,
            padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
            deform_groups,
        )
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        ext.modulated_deform_conv_col2im_coord(
            grad_columns, input, offset, mask, *geometry, grad_offset, grad_mask
        )
        grad_input = torch.zeros_like(input)
        ext.modulated_deform_conv_col2im(grad_columns, input, offset, mask, *geometry, grad_input)

        columns = input.new_empty((channels * kernel_h * kernel_w, rows))
        ext.modulated_deform_conv_im2col(
            input,
            offset,
            mask,
            kernel_h,
            kernel_w,
            padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
            deform_groups,
            columns,
        )
        columns_g = columns.view(groups, in_per_group, rows)
        grad_weight = weight.new_empty(groups, out_per_group, in_per_group)
        with _tf32_matmul(input.dtype), _mudnn_tf32():
            for group in range(groups):
                torch.mm(grad_out_g[group], columns_g[group].t(), out=grad_weight[group])
        grad_weight = grad_weight.view_as(weight)

        grad_bias = grad_output.sum(dim=(0, 2, 3)) if ctx.with_bias else None
        return (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Native offset-conv-fused NHWC path
# ---------------------------------------------------------------------------


def _fused_nhwc_supported(
    channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    deform_groups: int,
) -> bool:
    return (
        groups == 1
        and deform_groups == 1
        and kernel_size == (3, 3)
        and stride == (1, 1)
        and dilation == (1, 1)
        and channels % 32 == 0
        and channels <= 1024
    )


class _NativeFusedModulatedDeformConv2dFunction(Function):
    """Offset convolution + deformable gather + GEMM on channel-last tensors.

    ``input`` is ``[B, H, W, C]``; the offset convolution runs through muDNN, its raw
    ``[B, Ho, Wo, 3T]`` output feeds the gather directly (offset slice and mask sigmoid
    happen inside the kernel), and the backward pass gets the pre-sigmoid mask gradient
    straight out of ``col2im``.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        offset_weight: torch.Tensor,
        offset_bias: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
    ) -> torch.Tensor:
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")

        batch, height, width, channels_in = input.shape
        channels_out, _, kernel_h, kernel_w = weight.shape
        taps = kernel_h * kernel_w
        height_out = _output_size(height, padding[0], dilation[0], kernel_h, stride[0])
        width_out = _output_size(width, padding[1], dilation[1], kernel_w, stride[1])

        raw = F.conv2d(
            input.permute(0, 3, 1, 2), offset_weight, offset_bias, stride, padding, dilation
        )
        raw = raw.permute(0, 2, 3, 1).contiguous()  # [B, Ho, Wo, 3T]

        columns = input.new_empty((batch * height_out * width_out, taps * channels_in))
        ext.modulated_deform_conv_im2col_nhwc(
            input,
            raw,
            kernel_h,
            kernel_w,
            padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
            columns,
        )
        weight_flat = (
            weight.permute(2, 3, 1, 0).reshape(taps * channels_in, channels_out).contiguous()
        )
        with _tf32_matmul(input.dtype), _mudnn_tf32():
            output = torch.mm(columns, weight_flat).view(batch, height_out, width_out, channels_out)
        if bias is not None:
            output = output + bias.view(1, 1, 1, -1)

        # The column matrix dominates the memory footprint of a full BEVFormer batch,
        # so by default it is recomputed in the backward pass instead of saved.
        recompute = _env_flag("MUSPACE_DCN_RECOMPUTE_COLUMNS", True)
        ctx.recompute_columns = recompute
        if recompute:
            ctx.save_for_backward(input, raw, offset_weight, weight)
        else:
            ctx.save_for_backward(input, raw, offset_weight, weight, columns)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        ctx.with_bias = bias is not None
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation
        if ctx.recompute_columns:
            input, raw, offset_weight, weight = ctx.saved_tensors
            columns = None
        else:
            input, raw, offset_weight, weight, columns = ctx.saved_tensors

        batch, height, width, channels_in = input.shape
        channels_out, _, kernel_h, kernel_w = weight.shape
        taps = kernel_h * kernel_w
        height_out = _output_size(height, padding[0], dilation[0], kernel_h, stride[0])
        width_out = _output_size(width, padding[1], dilation[1], kernel_w, stride[1])
        rows = batch * height_out * width_out
        depth = taps * channels_in
        dtype = input.dtype

        geometry = (
            kernel_h,
            kernel_w,
            padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
        )
        if columns is None:
            columns = input.new_empty((rows, depth))
            ext.modulated_deform_conv_im2col_nhwc(input, raw, *geometry, columns)

        with _tf32_matmul(dtype):
            grad_out_flat = grad_output.reshape(rows, channels_out).to(dtype)
            weight_flat = weight.permute(2, 3, 1, 0).reshape(depth, channels_out).contiguous()

            with _mudnn_tf32():
                grad_columns = torch.mm(grad_out_flat, weight_flat.t())
                grad_weight = (
                    torch.mm(columns.t().float(), grad_out_flat.float())
                    .view(kernel_h, kernel_w, channels_in, channels_out)
                    .permute(3, 2, 0, 1)
                    .contiguous()
                    .to(dtype)
                )
                grad_bias = grad_out_flat.float().sum(0).to(dtype) if ctx.with_bias else None

            grad_sampled = torch.zeros_like(input, dtype=torch.float32)
            grad_offset = torch.zeros(
                (batch, height_out, width_out, 2 * taps), device=input.device, dtype=torch.float32
            )
            grad_mask = torch.zeros(
                (batch, height_out, width_out, taps), device=input.device, dtype=torch.float32
            )
            ext.modulated_deform_conv_col2im_nhwc(
                grad_columns.contiguous(),
                input,
                raw,
                *geometry,
                grad_sampled,
                grad_offset,
                grad_mask,
            )

            # Offset convolution backward; grad_mask is already the pre-sigmoid gradient.
            grad_raw = torch.cat([grad_offset, grad_mask], dim=-1).permute(0, 3, 1, 2).contiguous()
            input_nchw = input.permute(0, 3, 1, 2).contiguous().float()
            offset_weight_f = offset_weight.float()
            grad_input_offset = conv2d_input(
                input_nchw.shape, offset_weight_f, grad_raw, stride, padding, dilation
            )
            grad_offset_weight = conv2d_weight(
                input_nchw, offset_weight_f.shape, grad_raw, stride, padding, dilation
            )
            grad_offset_bias = grad_raw.sum(dim=(0, 2, 3))
            grad_input_offset = grad_input_offset.permute(0, 2, 3, 1)

        grad_input = (grad_sampled + grad_input_offset).to(dtype)
        return (
            grad_input,
            grad_offset_weight.to(dtype),
            grad_offset_bias.to(dtype),
            grad_weight,
            grad_bias,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Canonical API
# ---------------------------------------------------------------------------


def _record(
    *,
    requested_backend: str,
    decision: DispatchDecision,
    native_available: bool,
) -> None:
    record_dispatch_report(
        DispatchReport(
            operator=OPERATOR_NAME,
            requested_backend=requested_backend,
            selected_backend=decision.backend,
            device=decision.device,
            reason=decision.reason,
            native_available=native_available,
        )
    )


def _require_native(input: torch.Tensor, backend: BackendName) -> tuple[DispatchDecision, bool]:
    """Resolve the backend and fail loudly when the native path cannot run.

    This operator has no PyTorch fallback: the deformable columns are the whole
    operator, and a Python emulation of them would silently replace a production
    kernel with something orders of magnitude slower. Requests that cannot reach
    the MUSA kernels raise instead.
    """
    native_available = _native_extension() is not None
    decision = _explain_dispatch(input, backend=backend, native_available=native_available)
    if decision.backend == "musa":
        return decision, native_available
    reason = (
        decision.reason
        if decision.backend == "error"
        else (
            f"{OPERATOR_NAME} requires MUSA tensors and the muSPACE AOT native module "
            f"({decision.reason}); this operator has no PyTorch reference path"
        )
    )
    failure = DispatchDecision("error", decision.device, reason)
    _record(requested_backend=backend, decision=failure, native_available=native_available)
    raise RuntimeError(reason)


def _check_inputs(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    groups: int,
    deform_groups: int,
) -> None:
    if input.dim() != 4:
        raise ValueError(f"expected a 4D input tensor, got {input.dim()}D")
    if weight.dim() != 4:
        raise ValueError(f"expected a 4D weight tensor, got {weight.dim()}D")
    if groups < 1 or deform_groups < 1:
        raise ValueError("groups and deform_groups must be positive")
    if input.shape[1] % groups or weight.shape[0] % groups:
        raise ValueError("in_channels and out_channels must be divisible by groups")
    if input.shape[1] % deform_groups:
        raise ValueError("in_channels must be divisible by deform_groups")
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    if offset.shape[1] != deform_groups * 2 * kernel_h * kernel_w:
        raise ValueError("offset must have deform_groups * 2 * kH * kW channels")
    if mask.shape[1] != deform_groups * kernel_h * kernel_w:
        raise ValueError("mask must have deform_groups * kH * kW channels")


def modulated_deform_conv2d(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    deform_groups: int = 1,
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    """Modulated deformable convolution with the mmcv DCNv2 signature.

    ``input`` is ``[B, C, H, W]``, ``offset`` is ``[B, dg * 2 * kH * kW, Ho, Wo]``, and
    ``mask`` is ``[B, dg * kH * kW, Ho, Wo]`` (already sigmoid-activated).
    """
    _check_inputs(input, offset, mask, weight, groups, deform_groups)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    input = input.type_as(offset)
    weight = weight.type_as(input)
    mask = mask.type_as(input)

    decision, native_available = _require_native(input, backend)
    output = _NativeModulatedDeformConv2dFunction.apply(
        input.contiguous(),
        offset.contiguous(),
        mask.contiguous(),
        weight.contiguous(),
        bias,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
    )
    _record(requested_backend=backend, decision=decision, native_available=native_available)
    return output


def modulated_deform_conv2d_pack(
    input: torch.Tensor,
    offset_weight: torch.Tensor,
    offset_bias: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 1,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    deform_groups: int = 1,
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    """Packed DCNv2: computes offsets and masks from ``input`` and convolves.

    This is the production entry point used by :class:`ModulatedDeformConv2dPack`. On
    MUSA tensors whose configuration is supported it runs the offset-conv-fused NHWC
    path; otherwise it computes the offset convolution in PyTorch and delegates to
    :func:`modulated_deform_conv2d`.

    ``input`` and the returned tensor are ``[B, C, H, W]``; a ``channels_last`` input
    keeps its memory format through the operator and out the other side.
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    kernel_size = (weight.shape[2], weight.shape[3])

    decision, native_available = _require_native(input, backend)
    fused = _fused_nhwc_supported(
        input.shape[1], kernel_size, stride, dilation, groups, deform_groups
    )
    if not fused:
        raw = F.conv2d(input, offset_weight, offset_bias, stride, padding, dilation)
        # The offset-conv output is laid out as [deform_group][tap][h, w] for the first
        # 2 * dg * kH * kW channels, then [deform_group][tap] of pre-sigmoid mask. That
        # is what both kernels read: the NCHW kernel takes channel `dg*2T + 2t` as the
        # h offset and `+ 1` as the w offset, and the NHWC kernel indexes the raw output
        # the same way. mmcv splits this into three chunks and re-concatenates the first
        # two, which is an identity on these channels; slicing says the same thing without
        # implying the channels are grouped by component.
        offset_channels = 2 * deform_groups * kernel_size[0] * kernel_size[1]
        return modulated_deform_conv2d(
            input,
            raw[:, :offset_channels],
            torch.sigmoid(raw[:, offset_channels:]),
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            deform_groups,
            backend=backend,
        )

    channels_last = input.is_contiguous(memory_format=torch.channels_last)
    input_nhwc = input.permute(0, 2, 3, 1)
    if not channels_last:
        input_nhwc = input_nhwc.contiguous()
    output_nhwc = _NativeFusedModulatedDeformConv2dFunction.apply(
        input_nhwc, offset_weight, offset_bias, weight, bias, stride, padding, dilation
    )
    _record(requested_backend=backend, decision=decision, native_available=native_available)
    output = output_nhwc.permute(0, 3, 1, 2)
    return output if channels_last else output.contiguous()


def explain_modulated_deform_conv2d_dispatch(
    input: torch.Tensor,
    *,
    backend: BackendName = "auto",
) -> DispatchDecision:
    return _explain_dispatch(
        input, backend=backend, native_available=_native_extension() is not None
    )


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class ModulatedDeformConv2d(nn.Module):
    """DCNv2 module taking externally computed offsets and masks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.init_weights()

    def init_weights(self) -> None:
        fan_in = self.in_channels
        for size in self.kernel_size:
            fan_in *= size
        bound = 1.0 / math.sqrt(fan_in)
        self.weight.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )


class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    """DCNv2 module that generates its own offsets and masks.

    Parameter names and layout match mmcv's ``ModulatedDeformConv2dPack``, so mmcv
    checkpoints load unchanged.
    """

    _version = 2

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_weights()

    def init_weights(self) -> None:
        super().init_weights()
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return modulated_deform_conv2d_pack(
            x,
            self.conv_offset.weight,
            self.conv_offset.bias,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            if (
                prefix + "conv_offset.weight" not in state_dict
                and prefix[:-1] + "_offset.weight" in state_dict
            ):
                state_dict[prefix + "conv_offset.weight"] = state_dict.pop(
                    prefix[:-1] + "_offset.weight"
                )
            if (
                prefix + "conv_offset.bias" not in state_dict
                and prefix[:-1] + "_offset.bias" in state_dict
            ):
                state_dict[prefix + "conv_offset.bias"] = state_dict.pop(
                    prefix[:-1] + "_offset.bias"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


__all__ = [
    "ModulatedDeformConv2d",
    "ModulatedDeformConv2dPack",
    "explain_modulated_deform_conv2d_dispatch",
    "modulated_deform_conv2d",
    "modulated_deform_conv2d_pack",
]
