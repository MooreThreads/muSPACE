"""BEVFusion-compatible BEV pooling with MUSA and PyTorch backends."""

from __future__ import annotations

import torch

from ..runtime.kernel_dispatch import (
    BackendName,
    DispatchDecision,
    DispatchReport,
    record_dispatch_report,
)
from ..runtime.kernel_dispatch import explain_dispatch as _explain_dispatch
from ..runtime.native import NativeModuleUnavailable, native_module_proxy

OPERATOR_NAME = "bev_pool"


def _native_extension():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _normalize_inputs(
    feats: torch.Tensor, coords: torch.Tensor, batch: int, depth: int, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    if feats.dtype != torch.float32:
        raise TypeError("feats must use float32")
    if feats.ndim != 2:
        raise ValueError("feats must have shape [N, C]")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("coords must have shape [N, 4] in upstream [x, y, z, batch] order")
    if coords.shape[0] != feats.shape[0]:
        raise ValueError("coords and feats must have the same number of points")
    if coords.dtype not in {torch.int32, torch.int64}:
        raise TypeError("coords must use int32 or int64")
    shape = tuple(int(value) for value in (batch, depth, height, width))
    if any(value <= 0 for value in shape):
        raise ValueError("B, D, H, and W must be positive")
    if coords.numel() and (
        (coords[:, 0] < 0).any()
        or (coords[:, 0] >= shape[2]).any()
        or (coords[:, 1] < 0).any()
        or (coords[:, 1] >= shape[3]).any()
        or (coords[:, 2] < 0).any()
        or (coords[:, 2] >= shape[1]).any()
        or (coords[:, 3] < 0).any()
        or (coords[:, 3] >= shape[0]).any()
    ):
        raise ValueError("coords contains an index outside [B, D, H, W]")
    coords = coords.to(dtype=torch.int32).contiguous()
    return feats.contiguous(), coords, shape


def bev_pool_reference(
    feats: torch.Tensor, coords: torch.Tensor, batch: int, depth: int, height: int, width: int
) -> torch.Tensor:
    """Reference implementation of BEVFusion's ``bev_pool`` output layout."""
    feats, coords, (batch, depth, height, width) = _normalize_inputs(
        feats, coords, batch, depth, height, width
    )
    out = feats.new_zeros((batch, depth, height, width, feats.shape[1]))
    if feats.numel():
        out.index_put_(
            (coords[:, 3].long(), coords[:, 2].long(), coords[:, 0].long(), coords[:, 1].long()),
            feats,
            accumulate=True,
        )
        return out.permute(0, 4, 1, 2, 3).contiguous()
    # Keep autograd's empty-input contract: the output is zero but remains connected
    # to ``feats`` so backward returns an empty zero gradient.
    return (out + feats.sum() * 0).permute(0, 4, 1, 2, 3).contiguous()


def _sorted_intervals(coords: torch.Tensor, batch: int, depth: int, width: int):
    ranks = (
        coords[:, 0].to(torch.int64) * (width * depth * batch)
        + coords[:, 1].to(torch.int64) * (depth * batch)
        + coords[:, 2].to(torch.int64) * batch
        + coords[:, 3].to(torch.int64)
    )
    order = ranks.argsort(stable=True)
    sorted_ranks = ranks[order]
    kept = torch.ones(sorted_ranks.shape[0], dtype=torch.bool, device=coords.device)
    kept[1:] = sorted_ranks[1:] != sorted_ranks[:-1]
    starts = torch.where(kept)[0].to(torch.int32).contiguous()
    lengths = torch.empty_like(starts)
    lengths[:-1] = starts[1:] - starts[:-1]
    lengths[-1] = sorted_ranks.numel() - starts[-1]
    return order, starts, lengths


class _NativeBEVPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats: torch.Tensor, coords: torch.Tensor, shape: tuple[int, int, int, int]):
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        batch, depth, _, width = shape
        if feats.shape[0] == 0:
            ctx.save_for_backward(coords, coords, coords, coords)
            return feats.new_zeros((shape[0], feats.shape[1], *shape[1:])) + feats.sum() * 0
        order, starts, lengths = _sorted_intervals(coords, batch, depth, width)
        sorted_feats = feats[order].contiguous()
        sorted_coords = coords[order].contiguous()
        out = feats.new_zeros((*shape, feats.shape[1]))
        ext.bev_pool_forward(sorted_feats, sorted_coords, starts, lengths, out)
        ctx.save_for_backward(order, sorted_coords, starts, lengths)
        return out.permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        order, coords, starts, lengths = ctx.saved_tensors
        if order.numel() == 0:
            return grad_out.new_empty((0, grad_out.shape[1])), None, None
        grad_sorted = grad_out.new_empty((coords.shape[0], grad_out.shape[1]))
        ext.bev_pool_backward(
            grad_out.permute(0, 2, 3, 4, 1).contiguous(), coords, starts, lengths, grad_sorted
        )
        grad_feats = torch.empty_like(grad_sorted)
        grad_feats[order] = grad_sorted
        return grad_feats, None, None


def _record(
    *, requested_backend: str, decision: DispatchDecision, native_available: bool,
    native_fallback: bool = False, fallback_reason: str | None = None,
) -> None:
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


def bev_pool(
    feats: torch.Tensor,
    coords: torch.Tensor,
    B: int,
    D: int,
    H: int,
    W: int,
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    """Pool ``[N, C]`` features into BEVFusion's ``[B, C, D, H, W]`` tensor."""
    feats, coords, shape = _normalize_inputs(feats, coords, B, D, H, W)
    native_available = _native_extension() is not None
    decision = _explain_dispatch(feats, backend=backend, native_available=native_available)
    if decision.backend == "error":
        _record(requested_backend=backend, decision=decision, native_available=native_available)
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        _record(requested_backend=backend, decision=decision, native_available=native_available)
        return bev_pool_reference(feats, coords, *shape)
    try:
        output = _NativeBEVPoolFunction.apply(feats, coords, shape)
        _record(requested_backend=backend, decision=decision, native_available=native_available)
        return output
    except RuntimeError as exc:
        if backend != "auto":
            _record(
                requested_backend=backend,
                decision=DispatchDecision("error", decision.device, str(exc)),
                native_available=native_available,
            )
            raise
        fallback = DispatchDecision(
            "reference", decision.device, "native backend failed; falling back to PyTorch reference"
        )
        _record(
            requested_backend=backend, decision=fallback, native_available=native_available,
            native_fallback=True, fallback_reason=str(exc),
        )
        return bev_pool_reference(feats, coords, *shape)


__all__ = ["bev_pool", "bev_pool_reference"]
