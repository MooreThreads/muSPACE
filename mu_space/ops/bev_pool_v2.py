"""BEVPoolV2 canonical API with MUSA native and PyTorch reference backends."""

from __future__ import annotations

from typing import Sequence

import torch

from ..runtime.native import NativeModuleUnavailable, native_module_proxy
from ..runtime.kernel_dispatch import (
    BackendName,
    DispatchDecision,
    DispatchReport,
    explain_dispatch as _explain_dispatch,
    record_dispatch_report,
)

OPERATOR_NAME = "bev_pool_v2"


def _native_extension():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _shape_tuple(bev_feat_shape: Sequence[int]) -> tuple[int, int, int, int, int]:
    if len(bev_feat_shape) != 5:
        raise ValueError("bev_feat_shape must contain [B, D, H, W, C]")
    return tuple(int(dim) for dim in bev_feat_shape)  # type: ignore[return-value]


def _normalize_inputs(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if depth.dtype != torch.float32:
        raise TypeError("depth must use float32")
    if feat.dtype != torch.float32:
        raise TypeError("feat must use float32")
    for name, tensor in (
        ("ranks_depth", ranks_depth),
        ("ranks_feat", ranks_feat),
        ("ranks_bev", ranks_bev),
        ("interval_starts", interval_starts),
        ("interval_lengths", interval_lengths),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must use int32")
    return tuple(
        tensor.contiguous()
        for tensor in (
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_starts,
            interval_lengths,
        )
    )


def _intervals_from_sorted_ranks(ranks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if ranks.numel() == 0:
        return ranks.new_empty(0), ranks.new_empty(0)
    kept = torch.ones(ranks.shape[0], dtype=torch.bool, device=ranks.device)
    kept[1:] = ranks[1:] != ranks[:-1]
    starts = torch.where(kept)[0].to(dtype=torch.int32)
    lengths = torch.empty_like(starts)
    lengths[:-1] = starts[1:] - starts[:-1]
    lengths[-1] = ranks.shape[0] - starts[-1]
    return starts.contiguous(), lengths.contiguous()


def bev_pool_v2_reference(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    bev_feat_shape: Sequence[int],
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
) -> torch.Tensor:
    del interval_starts, interval_lengths
    b, d, h, w, c = _shape_tuple(bev_feat_shape)
    depth_flat = depth.reshape(-1)
    feat_flat = feat.reshape(-1, c)
    weighted = depth_flat[ranks_depth.long()].unsqueeze(1) * feat_flat[ranks_feat.long()]
    out_flat = feat.new_zeros((b * d * h * w, c))
    out_flat.index_add_(0, ranks_bev.long(), weighted)
    return out_flat.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()


class _NativeBEVPoolV2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        depth: torch.Tensor,
        feat: torch.Tensor,
        ranks_depth: torch.Tensor,
        ranks_feat: torch.Tensor,
        ranks_bev: torch.Tensor,
        bev_feat_shape: tuple[int, int, int, int, int],
        interval_starts: torch.Tensor,
        interval_lengths: torch.Tensor,
    ) -> torch.Tensor:
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        b, d, h, w, c = bev_feat_shape
        out = feat.new_zeros((b, d, h, w, c))
        ext.bev_pool_v2_forward(
            depth, feat, out, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths
        )
        ctx.save_for_backward(depth, feat, ranks_depth, ranks_feat, ranks_bev)
        ctx.bev_feat_shape = bev_feat_shape
        return out.permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        depth, feat, ranks_depth, ranks_feat, ranks_bev = ctx.saved_tensors
        order = ranks_feat.argsort()
        ranks_feat_bp = ranks_feat[order].contiguous()
        ranks_depth_bp = ranks_depth[order].contiguous()
        ranks_bev_bp = ranks_bev[order].contiguous()
        interval_starts_bp, interval_lengths_bp = _intervals_from_sorted_ranks(ranks_feat_bp)
        grad_depth = torch.zeros_like(depth)
        grad_feat = torch.zeros_like(feat)
        grad_out_native = grad_out.permute(0, 2, 3, 4, 1).contiguous()
        ext.bev_pool_v2_backward(
            grad_out_native,
            depth,
            feat,
            ranks_depth_bp,
            ranks_feat_bp,
            ranks_bev_bp,
            interval_starts_bp,
            interval_lengths_bp,
            grad_depth,
            grad_feat,
        )
        return grad_depth, grad_feat, None, None, None, None, None, None


def _record(
    *,
    requested_backend: str,
    decision: DispatchDecision,
    native_available: bool,
    native_fallback: bool = False,
    fallback_reason: str | None = None,
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


def bev_pool_v2(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    bev_feat_shape: Sequence[int],
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    (
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ) = _normalize_inputs(
        depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths
    )
    shape = _shape_tuple(bev_feat_shape)
    native_available = _native_extension() is not None
    decision = _explain_dispatch(feat, backend=backend, native_available=native_available)
    if decision.backend == "error":
        _record(
            requested_backend=backend,
            decision=decision,
            native_available=native_available,
        )
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        _record(
            requested_backend=backend,
            decision=decision,
            native_available=native_available,
        )
        return bev_pool_v2_reference(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            shape,
            interval_starts,
            interval_lengths,
        )
    if ranks_depth.numel() == 0:
        output = feat.new_zeros((shape[0], shape[4], shape[1], shape[2], shape[3]))
        _record(
            requested_backend=backend,
            decision=decision,
            native_available=native_available,
        )
        return output + depth.sum() * 0 + feat.sum() * 0
    try:
        output = _NativeBEVPoolV2Function.apply(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            shape,
            interval_starts,
            interval_lengths,
        )
        _record(
            requested_backend=backend,
            decision=decision,
            native_available=native_available,
        )
        return output
    except RuntimeError as exc:
        if backend != "auto":
            _record(
                requested_backend=backend,
                decision=DispatchDecision("error", decision.device, str(exc)),
                native_available=native_available,
            )
            raise
        fallback_decision = DispatchDecision(
            "reference", decision.device, "native backend failed; falling back to PyTorch reference"
        )
        _record(
            requested_backend=backend,
            decision=fallback_decision,
            native_available=native_available,
            native_fallback=True,
            fallback_reason=str(exc),
        )
        return bev_pool_v2_reference(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            shape,
            interval_starts,
            interval_lengths,
        )


__all__ = ["bev_pool_v2", "bev_pool_v2_reference"]
