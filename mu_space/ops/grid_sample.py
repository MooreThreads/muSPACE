"""PyTorch-compatible 2-D and 3-D grid sampling."""

from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

from ..runtime.kernel_dispatch import DispatchReport, explain_dispatch, record_dispatch_report
from ..runtime.native import NativeModuleUnavailable, native_module_proxy

OPERATOR_NAME = "grid_sample"
BackendName = Literal["auto", "musa", "reference"]
_MODES = {"bilinear": 0, "nearest": 1}
_PADDING_MODES = {"zeros": 0, "border": 1, "reflection": 2}
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _native():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _validate(input: torch.Tensor, grid: torch.Tensor, mode: str, padding_mode: str):
    if not isinstance(input, torch.Tensor) or not isinstance(grid, torch.Tensor):
        raise TypeError("input and grid must be torch.Tensor objects")
    if input.dim() not in (4, 5):
        raise ValueError("input must be 4-D or 5-D")
    if any(size <= 0 for size in input.shape[2:]):
        raise ValueError("input spatial dimensions must be positive")
    if any(size >= 2**31 - 1 for size in input.shape[2:]):
        raise ValueError("input spatial dimension exceeds indexing range")
    expected_grid_dim = input.dim() - 2
    if grid.dim() != expected_grid_dim + 2:
        raise ValueError(f"grid must be {expected_grid_dim + 2}-D for the input")
    if grid.shape[0] != input.shape[0]:
        raise ValueError("grid batch dimension must match input")
    if grid.shape[-1] != expected_grid_dim:
        raise ValueError(f"grid last dimension must be {expected_grid_dim}")
    if input.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("input must use float32, float16 or bfloat16")
    if grid.dtype != input.dtype:
        raise TypeError("grid dtype must match input")
    if mode not in _MODES:
        raise ValueError("mode must be one of: bilinear, nearest")
    if padding_mode not in _PADDING_MODES:
        raise ValueError("padding_mode must be one of: zeros, border, reflection")
    if input.device != grid.device:
        raise ValueError("input and grid must be on the same device")


def grid_sample_reference(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Run the PyTorch reference implementation after muSPACE validation."""
    _validate(input, grid, mode, padding_mode)
    input_contiguous = input.contiguous()
    grid_contiguous = grid.contiguous()
    # CPU grid_sampler lacks half/bfloat16 kernels on some supported PyTorch
    # stacks. Compute the reference in float32 and preserve the public dtype.
    if input_contiguous.device.type == "cpu" and input_contiguous.dtype != torch.float32:
        return F.grid_sample(
            input_contiguous.float(),
            grid_contiguous.float(),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        ).to(input.dtype)
    return F.grid_sample(
        input_contiguous,
        grid_contiguous,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class _NativeGridSampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, mode, padding_mode, align_corners):
        ext = _native()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        output = torch.empty(
            (input.shape[0], input.shape[1], *grid.shape[1:-1]),
            dtype=input.dtype,
            device=input.device,
        )
        ext.grid_sample_forward(
            input,
            grid,
            output,
            _MODES[mode],
            _PADDING_MODES[padding_mode],
            bool(align_corners),
        )
        ctx.save_for_backward(input, grid)
        ctx.mode = _MODES[mode]
        ctx.padding_mode = _PADDING_MODES[padding_mode]
        ctx.align_corners = bool(align_corners)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input = torch.empty_like(input, dtype=torch.float32)
        grad_grid = torch.empty_like(grid, dtype=torch.float32)
        native_grad_output = grad_output.contiguous()
        # The kernel accumulates gradients in float32 for half storage; the
        # public tensors keep the caller's dtype at the API boundary.
        ext = _native()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        ext.grid_sample_backward(
            input,
            grid,
            native_grad_output,
            grad_input,
            grad_grid,
            ctx.mode,
            ctx.padding_mode,
            ctx.align_corners,
        )
        return grad_input.to(input.dtype), grad_grid.to(grid.dtype), None, None, None


def grid_sample(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    """Sample 2-D or 3-D input at normalized grid coordinates."""
    _validate(input, grid, mode, padding_mode)
    input = input.contiguous()
    grid = grid.contiguous()
    native_available = _native() is not None
    decision = explain_dispatch(input, backend=backend, native_available=native_available)
    if decision.backend == "error":
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME, backend, "error", decision.device, decision.reason, native_available
            )
        )
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME,
                backend,
                "reference",
                decision.device,
                decision.reason,
                native_available,
            )
        )
        return grid_sample_reference(input, grid, mode, padding_mode, align_corners)
    try:
        output = _NativeGridSampleFunction.apply(input, grid, mode, padding_mode, align_corners)
    except RuntimeError as exc:
        if backend != "auto":
            raise
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME,
                backend,
                "reference",
                decision.device,
                "native backend failed; falling back to reference",
                native_available,
                True,
                str(exc),
            )
        )
        return grid_sample_reference(input, grid, mode, padding_mode, align_corners)
    record_dispatch_report(
        DispatchReport(
            OPERATOR_NAME, backend, "musa", decision.device, decision.reason, native_available
        )
    )
    return output


__all__ = ["grid_sample", "grid_sample_reference"]
