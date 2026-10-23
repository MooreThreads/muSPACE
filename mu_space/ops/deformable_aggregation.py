"""Deformable aggregation canonical API with MUSA native and PyTorch reference backends.

Adapted from SparseDrive's multi-camera multi-scale deformable feature aggregation. The native
path calls the optimized `*_opt` forward and backward kernels; `docs/provenance.md` records the
lineage of the baseline and optimized pairs.
"""

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

OPERATOR_NAME = "deformable_aggregation"
MAX_GROUPS = 8


def _native_extension():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _normalize_inputs(
    mc_ms_feat: torch.Tensor,
    spatial_shape: torch.Tensor,
    scale_start_index: torch.Tensor,
    sampling_location: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if mc_ms_feat.dtype != torch.float32:
        raise TypeError("mc_ms_feat must use float32")
    if sampling_location.dtype != torch.float32:
        raise TypeError("sampling_location must use float32")
    if weights.dtype != torch.float32:
        raise TypeError("weights must use float32")
    if spatial_shape.dtype != torch.int32:
        raise TypeError("spatial_shape must use int32")
    if scale_start_index.dtype != torch.int32:
        raise TypeError("scale_start_index must use int32")

    if mc_ms_feat.dim() != 3:
        raise ValueError("mc_ms_feat must have shape [B, num_feat, C]")
    if spatial_shape.dim() != 3 or spatial_shape.size(-1) != 2:
        raise ValueError("spatial_shape must have shape [num_cams, num_scale, 2]")
    if scale_start_index.dim() != 2:
        raise ValueError("scale_start_index must have shape [num_cams, num_scale]")
    if sampling_location.dim() != 5 or sampling_location.size(-1) != 2:
        raise ValueError("sampling_location must have shape [B, anchors, pts, cams, 2]")
    if weights.dim() != 6:
        raise ValueError("weights must have shape [B, anchors, pts, cams, scale, groups]")

    num_embeds = mc_ms_feat.size(-1)
    num_groups = weights.size(-1)
    if num_groups <= 0 or num_groups > MAX_GROUPS:
        raise ValueError(f"num_groups must be in [1, {MAX_GROUPS}]")
    if num_embeds % num_groups != 0:
        raise ValueError("num_embeds must be divisible by num_groups")

    return (
        mc_ms_feat.contiguous(),
        spatial_shape.contiguous(),
        scale_start_index.contiguous(),
        sampling_location.contiguous(),
        weights.contiguous(),
    )


def _bilinear_sample(
    feat_map: torch.Tensor,
    h_im: torch.Tensor,
    w_im: torch.Tensor,
) -> torch.Tensor:
    """Sample [H, W, C] feature map at (h_im, w_im); returns [C]."""
    height, width, _channels = feat_map.shape
    h_low = torch.floor(h_im).to(dtype=torch.int64)
    w_low = torch.floor(w_im).to(dtype=torch.int64)
    h_high = h_low + 1
    w_high = w_low + 1
    lh = h_im - h_low.to(dtype=h_im.dtype)
    lw = w_im - w_low.to(dtype=w_im.dtype)
    hh = 1.0 - lh
    hw = 1.0 - lw
    w1, w2, w3, w4 = hh * hw, hh * lw, lh * hw, lh * lw

    value = feat_map.new_zeros(feat_map.shape[-1])
    h_low_i = int(h_low.item())
    w_low_i = int(w_low.item())
    h_high_i = int(h_high.item())
    w_high_i = int(w_high.item())
    if h_low_i >= 0 and w_low_i >= 0:
        value = value + w1 * feat_map[h_low_i, w_low_i]
    if h_low_i >= 0 and w_high_i <= width - 1:
        value = value + w2 * feat_map[h_low_i, w_high_i]
    if h_high_i <= height - 1 and w_low_i >= 0:
        value = value + w3 * feat_map[h_high_i, w_low_i]
    if h_high_i <= height - 1 and w_high_i <= width - 1:
        value = value + w4 * feat_map[h_high_i, w_high_i]
    return value


def deformable_aggregation_reference(
    mc_ms_feat: torch.Tensor,
    spatial_shape: torch.Tensor,
    scale_start_index: torch.Tensor,
    sampling_location: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference matching SparseDrive deformable aggregation semantics."""
    batch_size, _, num_embeds = mc_ms_feat.shape
    num_cams, num_scale, _ = spatial_shape.shape
    num_anchors = sampling_location.size(1)
    num_pts = sampling_location.size(2)
    num_groups = weights.size(-1)
    channels_per_group = num_embeds // num_groups

    output = mc_ms_feat.new_zeros(batch_size, num_anchors, num_embeds)
    spatial_shape_list = spatial_shape.tolist()
    scale_start_list = scale_start_index.tolist()

    for batch_index in range(batch_size):
        for anchor_index in range(num_anchors):
            for pts_index in range(num_pts):
                for cam_index in range(num_cams):
                    loc_w = sampling_location[batch_index, anchor_index, pts_index, cam_index, 0]
                    loc_h = sampling_location[batch_index, anchor_index, pts_index, cam_index, 1]
                    # Match native early-exit on locations outside (0, 1).
                    if not (bool((loc_w > 0) & (loc_w < 1) & (loc_h > 0) & (loc_h < 1))):
                        continue
                    for scale_index in range(num_scale):
                        height = int(spatial_shape_list[cam_index][scale_index][0])
                        width = int(spatial_shape_list[cam_index][scale_index][1])
                        start = int(scale_start_list[cam_index][scale_index])
                        feat_map = mc_ms_feat[batch_index, start : start + height * width].view(
                            height, width, num_embeds
                        )
                        h_im = loc_h * height - 0.5
                        w_im = loc_w * width - 0.5
                        sampled = _bilinear_sample(feat_map, h_im, w_im)
                        group_weights = weights[
                            batch_index, anchor_index, pts_index, cam_index, scale_index
                        ]
                        channel_weights = group_weights.repeat_interleave(channels_per_group)
                        output[batch_index, anchor_index] = (
                            output[batch_index, anchor_index] + sampled * channel_weights
                        )
    return output


class _NativeDeformableAggregationFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mc_ms_feat: torch.Tensor,
        spatial_shape: torch.Tensor,
        scale_start_index: torch.Tensor,
        sampling_location: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        output = mc_ms_feat.new_zeros(
            (mc_ms_feat.size(0), sampling_location.size(1), mc_ms_feat.size(2))
        )
        ext.deformable_aggregation_forward_opt(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            output,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ext = _native_extension()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        ext.deformable_aggregation_backward_opt(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return grad_mc_ms_feat, None, None, grad_sampling_location, grad_weights


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


def deformable_aggregation(
    mc_ms_feat: torch.Tensor,
    spatial_shape: torch.Tensor,
    scale_start_index: torch.Tensor,
    sampling_location: torch.Tensor,
    weights: torch.Tensor,
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    (
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ) = _normalize_inputs(
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )
    native_available = _native_extension() is not None
    decision = _explain_dispatch(mc_ms_feat, backend=backend, native_available=native_available)
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
        return deformable_aggregation_reference(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
    if sampling_location.size(1) == 0:
        output = mc_ms_feat.new_zeros((mc_ms_feat.size(0), 0, mc_ms_feat.size(2)))
        _record(
            requested_backend=backend,
            decision=decision,
            native_available=native_available,
        )
        return output + mc_ms_feat.sum() * 0 + sampling_location.sum() * 0 + weights.sum() * 0
    try:
        output = _NativeDeformableAggregationFunction.apply(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
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
        return deformable_aggregation_reference(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )


def deformable_aggregation_function(
    feature_maps: torch.Tensor,
    spatial_shape: torch.Tensor,
    scale_start_index: torch.Tensor,
    sampling_location: torch.Tensor,
    weights: torch.Tensor,
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    """SparseDrive-compatible alias for :func:`deformable_aggregation`."""
    return deformable_aggregation(
        feature_maps.contiguous().float(),
        spatial_shape.contiguous().to(dtype=torch.int32),
        scale_start_index.contiguous().to(dtype=torch.int32),
        sampling_location.contiguous().float(),
        weights.contiguous().float(),
        backend=backend,
    )


def feature_maps_format(feature_maps, inverse: bool = False):
    """Format multi-camera multi-scale feature maps for deformable aggregation.

    Kept as a SparseDrive-compatible helper used by model glue code.
    """
    if inverse:
        col_feats, spatial_shape, scale_start_index = feature_maps
        num_cams, _num_levels = spatial_shape.shape[:2]

        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]
        split_size = split_size.cpu().numpy().tolist()
        split_size = [[int(element) for element in sub_list] for sub_list in split_size]

        cam_split = [1]
        cam_split_size = [sum(split_size[0])]
        for i in range(num_cams - 1):
            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                cam_split.append(0)
                cam_split_size.append(0)
            cam_split[-1] += 1
            cam_split_size[-1] += sum(split_size[i + 1])
        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]

        spatial_shape_list = spatial_shape.cpu().numpy().tolist()
        spatial_shape_list = [
            [[int(element) for element in element_list] for element_list in sub_list]
            for sub_list in spatial_shape_list
        ]
        mc_ms_feat = []
        shape_index = 0
        for i, feat in enumerate(mc_feat):
            feat = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat):
                feat[j] = f.unflatten(2, spatial_shape_list[shape_index][j])
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
        return mc_ms_feat

    if isinstance(feature_maps[0], (list, tuple)):
        formated = [feature_maps_format(x) for x in feature_maps]
        col_feats = torch.cat([x[0] for x in formated], dim=1)
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        return [col_feats, spatial_shape, scale_start_index]

    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape_rows: list[Sequence[int]] = []
    col_feats = []
    for feat in feature_maps:
        spatial_shape_rows.append(feat.shape[-2:])
        col_feats.append(torch.reshape(feat, (bs, num_cams, feat.shape[2], -1)))

    col_feats_t = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    spatial_shape_t = torch.tensor(
        [spatial_shape_rows] * num_cams,
        dtype=torch.int64,
        device=col_feats_t.device,
    )
    scale_start_index = spatial_shape_t[..., 0] * spatial_shape_t[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [torch.tensor([0], device=scale_start_index.device), scale_start_index[:-1]]
    )
    scale_start_index = scale_start_index.reshape(num_cams, -1)
    return [col_feats_t, spatial_shape_t, scale_start_index]


__all__ = [
    "deformable_aggregation",
    "deformable_aggregation_function",
    "deformable_aggregation_reference",
    "feature_maps_format",
]
