"""Depth-aware voxel pooling with MUSA native and PyTorch reference backends.

The canonical API mirrors the Qwen-Drive-1.0 ``voxel_pool_depth`` operator:
image features and per-pixel depth distributions are fused into a per-camera
voxel volume, where every occupied voxel accumulates the depth probability of
the frustum cells that project into it.
"""

from __future__ import annotations

import torch

from ..runtime.kernel_dispatch import (
    BackendName,
    DispatchDecision,
    DispatchReport,
    record_dispatch_report,
)
from ..runtime.kernel_dispatch import (
    explain_dispatch as _explain_dispatch,
)
from ..runtime.native import NativeModuleUnavailable, native_module_proxy

OPERATOR_NAME = "voxel_pool_depth"

SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _native_extension():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _promoted_dtype(img_feats: torch.Tensor, img_depth: torch.Tensor) -> torch.dtype:
    return torch.promote_types(img_feats.dtype, img_depth.dtype)


def _validate_dtypes(img_feats: torch.Tensor, img_depth: torch.Tensor) -> None:
    if img_feats.dtype not in SUPPORTED_DTYPES:
        raise TypeError("img_feats must use float32, float16 or bfloat16")
    if img_depth.dtype not in SUPPORTED_DTYPES:
        raise TypeError("img_depth must use float32, float16 or bfloat16")


def _validate_shapes(
    img_feats: torch.Tensor,
    img_depth: torch.Tensor,
    voxel_coords: torch.Tensor,
    mask: torch.Tensor,
    B: int,
    X: int,
    Y: int,
    Z: int,
) -> tuple[int, int, int, int, int, int]:
    if img_feats.dim() != 5:
        raise ValueError("img_feats must have shape [B, N_sweep * N_cam, C, H, W]")
    if img_depth.dim() != 4:
        raise ValueError("img_depth must have shape [B * N_sweep * N_cam, D, H, W]")
    if voxel_coords.dim() != 7:
        raise ValueError("voxel_coords must have shape [B, N_sweep, N_cam, D, H, W, 4]")
    if tuple(mask.shape) != tuple(voxel_coords.shape[:-1]):
        raise ValueError("mask must match voxel_coords.shape[:-1]")
    if B < 1 or X < 1 or Y < 1 or Z < 1:
        raise ValueError("B, X, Y and Z must be positive")

    _, n_images, feat_channels, H, W = img_feats.shape
    _, N_sweep, N_cam, D, H_coords, W_coords = mask.shape
    if H != H_coords or W != W_coords:
        raise ValueError("img_feats spatial size must match voxel_coords")
    if n_images != N_sweep * N_cam:
        raise ValueError("img_feats must hold N_sweep * N_cam images")
    if img_depth.shape != (B * N_sweep * N_cam, D, H, W):
        raise ValueError("img_depth must have shape [B * N_sweep * N_cam, D, H, W]")
    if feat_channels < 1:
        raise ValueError("img_feats channels must be positive")
    return N_sweep, N_cam, D, H, W, feat_channels


def _compact_frustum(
    voxel_coords: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact the valid frustum cells and keep their linear point indices.

    ``point_indices`` enumerates the ``[batch, sweep, cam, d, h, w]`` frustum
    space; ``coords`` holds the ``(batch, x, y, z)`` voxel coordinate of each
    selected cell.  No rank or interval is built here: the native forward
    accumulates into the zeroed volume with atomics, so the cells never have to
    be grouped by their output voxel.
    """
    flat_mask = mask.reshape(-1)
    point_indices = flat_mask.nonzero().reshape(-1).to(torch.int32)
    coords = voxel_coords.reshape(-1, 4)[flat_mask].to(torch.int32).contiguous()
    return point_indices, coords


def voxel_pool_depth_reference(
    img_feats: torch.Tensor,
    img_depth: torch.Tensor,
    voxel_coords: torch.Tensor,
    mask: torch.Tensor,
    B: int,
    X: int,
    Y: int,
    Z: int,
) -> torch.Tensor:
    """Pure PyTorch depth-aware voxel pooling used for comparison and fallback."""
    _validate_dtypes(img_feats, img_depth)
    N_sweep, N_cam, D, H, W, feat_channels = _validate_shapes(
        img_feats, img_depth, voxel_coords, mask, B, X, Y, Z
    )
    point_indices, coords = _compact_frustum(voxel_coords, mask)
    device = img_feats.device
    out_dtype = _promoted_dtype(img_feats, img_depth)
    out = torch.zeros(B * N_cam * X * Y * Z, feat_channels, dtype=torch.float32, device=device)
    if point_indices.numel() != 0:
        image_index = point_indices // (D * H * W)
        spatial = point_indices % (D * H * W)
        d = spatial // (H * W)
        h = (spatial // W) % H
        w = spatial % W
        feat = img_feats.reshape(B * N_sweep * N_cam, feat_channels, H, W)[image_index, :, h, w]
        depth = img_depth[image_index, d, h, w]
        weighted = depth.to(torch.float32).unsqueeze(1) * feat.to(torch.float32)
        ranks = (
            coords[:, 0].long() * (N_cam * X * Y * Z)
            + (image_index % N_cam).long() * (X * Y * Z)
            + coords[:, 1].long() * (Y * Z)
            + coords[:, 2].long() * Z
            + coords[:, 3].long()
        )
        out.index_add_(0, ranks, weighted)
    else:
        # Keep the empty result connected to the inputs so autograd still reaches
        # them, matching the native path's zero gradients.
        out = out + img_feats.to(torch.float32).sum() * 0 + img_depth.to(torch.float32).sum() * 0
    return out.view(B, N_cam, X, Y, Z, feat_channels).to(out_dtype)


class _NativeVoxelPoolDepthFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        img_feats: torch.Tensor,
        img_depth: torch.Tensor,
        point_indices: torch.Tensor,
        coords: torch.Tensor,
        dims: tuple[int, ...],
    ) -> torch.Tensor:
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        B, N_sweep, N_cam, X, Y, Z, D, H, W = dims
        feat_channels = img_feats.shape[2]
        out = torch.zeros(
            (B, N_cam, X, Y, Z, feat_channels), dtype=torch.float32, device=img_feats.device
        )
        if point_indices.numel() != 0:
            ext.voxel_pool_depth_forward(
                img_feats,
                img_depth,
                out,
                coords,
                point_indices,
                B,
                N_sweep,
                N_cam,
                X,
                Y,
                Z,
                D,
                H,
                W,
            )
        ctx.save_for_backward(img_feats, img_depth, point_indices, coords)
        ctx.dims = dims
        return out.to(_promoted_dtype(img_feats, img_depth))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        img_feats, img_depth, point_indices, coords = ctx.saved_tensors
        B, N_sweep, N_cam, X, Y, Z, D, H, W = ctx.dims
        grad_out_native = grad_out.to(torch.float32).contiguous()
        img_feats_grad = torch.zeros_like(img_feats, dtype=torch.float32)
        img_depth_grad = torch.zeros_like(img_depth, dtype=torch.float32)
        if point_indices.numel() != 0:
            ext.voxel_pool_depth_backward(
                grad_out_native,
                img_feats,
                img_depth,
                coords,
                point_indices,
                img_feats_grad,
                img_depth_grad,
                B,
                N_sweep,
                N_cam,
                X,
                Y,
                Z,
                D,
                H,
                W,
            )
        return (
            img_feats_grad.to(img_feats.dtype),
            img_depth_grad.to(img_depth.dtype),
            None,
            None,
            None,
            None,
        )


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


def voxel_pool_depth(
    img_feats: torch.Tensor,
    img_depth: torch.Tensor,
    voxel_coords: torch.Tensor,
    mask: torch.Tensor,
    B: int,
    X: int,
    Y: int,
    Z: int,
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    """Fuse image features with depth distributions into a voxel volume.

    ``voxel_coords`` has shape ``[B, N_sweep, N_cam, D, H, W, 4]`` holding one
    ``(batch, x, y, z)`` voxel coordinate per frustum cell, and ``mask`` has
    shape ``[B, N_sweep, N_cam, D, H, W]`` selecting the cells that project into
    the volume.  The result has shape ``[B, N_cam, X, Y, Z, C]``.
    """
    for name, tensor in (
        ("img_feats", img_feats),
        ("img_depth", img_depth),
        ("voxel_coords", voxel_coords),
        ("mask", mask),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    _validate_dtypes(img_feats, img_depth)
    N_sweep, N_cam, D, H, W, _ = _validate_shapes(
        img_feats, img_depth, voxel_coords, mask, B, X, Y, Z
    )
    mask = mask.to(torch.bool)
    img_feats = img_feats.contiguous()
    img_depth = img_depth.contiguous()
    voxel_coords = voxel_coords.contiguous()
    mask = mask.contiguous()

    dims = (B, N_sweep, N_cam, X, Y, Z, D, H, W)
    native_available = _native_extension() is not None
    decision = _explain_dispatch(img_feats, backend=backend, native_available=native_available)
    if decision.backend == "error":
        _record(requested_backend=backend, decision=decision, native_available=native_available)
        raise RuntimeError(decision.reason)

    point_indices, coords = _compact_frustum(voxel_coords, mask)

    if decision.backend == "reference":
        _record(requested_backend=backend, decision=decision, native_available=native_available)
        return voxel_pool_depth_reference(img_feats, img_depth, voxel_coords, mask, B, X, Y, Z)

    try:
        output = _NativeVoxelPoolDepthFunction.apply(
            img_feats, img_depth, point_indices, coords, dims
        )
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
        return voxel_pool_depth_reference(img_feats, img_depth, voxel_coords, mask, B, X, Y, Z)


__all__ = ["voxel_pool_depth", "voxel_pool_depth_reference"]
