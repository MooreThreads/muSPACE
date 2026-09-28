"""Batch point-in-rotated-3D-box mask for BEVFusion."""

from __future__ import annotations

import math

import torch

from ..runtime.kernel_dispatch import (
    BackendName,
    DispatchDecision,
    DispatchReport,
    explain_dispatch,
    record_dispatch_report,
)
from ..runtime.native import NativeModuleUnavailable, native_module_proxy

OPERATOR_NAME = "points_in_boxes_all"


def _native_extension():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _validate(boxes: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(boxes, torch.Tensor) or not isinstance(points, torch.Tensor):
        raise TypeError("boxes and points must be torch.Tensor")
    if boxes.dtype != torch.float32 or points.dtype != torch.float32:
        raise TypeError("boxes and points must use float32")
    if boxes.ndim != 3 or boxes.shape[-1] != 7:
        raise ValueError("boxes must have shape [B, M, 7]")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, N, 3]")
    if boxes.shape[0] != points.shape[0]:
        raise ValueError("boxes and points must have the same batch size")
    if boxes.device != points.device:
        raise ValueError("boxes and points must be on the same device")
    return boxes.contiguous(), points.contiguous()


def points_in_boxes_all_reference(boxes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    boxes, points = _validate(boxes, points)
    b, m, _ = boxes.shape
    n = points.shape[1]
    if b == 0 or n == 0 or m == 0:
        return torch.zeros((b, n, m), dtype=torch.int32, device=boxes.device)
    # CUDA/OpenMMLab convention: z is the bottom center; x/y are strict.
    center = boxes[:, None, :, :3]
    size = boxes[:, None, :, 3:6]
    angle = boxes[:, None, :, 6]
    delta = points[:, :, None, :] - center
    cz = delta[..., 2] - size[..., 2] * 0.5
    rot_angle = angle + math.pi / 2.0
    cosa, sina = torch.cos(rot_angle), torch.sin(rot_angle)
    local_x = delta[..., 0] * cosa - delta[..., 1] * sina
    local_y = delta[..., 0] * sina + delta[..., 1] * cosa
    inside = (
        (cz.abs() <= size[..., 2] * 0.5)
        & (local_x > -size[..., 1] * 0.5)
        & (local_x < size[..., 1] * 0.5)
        & (local_y > -size[..., 0] * 0.5)
        & (local_y < size[..., 0] * 0.5)
    )
    return inside.to(torch.int32)


def points_in_boxes_all(
    boxes: torch.Tensor, points: torch.Tensor, *, backend: BackendName = "auto"
) -> torch.Tensor:
    boxes, points = _validate(boxes, points)
    # Avoid loading the AOT runtime for an explicitly requested reference path;
    # this also keeps CPU-only validation independent of MUSA loader state.
    native = (
        None
        if backend == "reference" or boxes.device.type not in {"musa", "privateuseone"}
        else _native_extension()
    )
    decision = explain_dispatch(boxes, backend=backend, native_available=native is not None)
    if decision.backend == "error":
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME,
                backend,
                decision.backend,
                decision.device,
                decision.reason,
                native is not None,
            )
        )
        raise RuntimeError(decision.reason)
    if decision.backend == "reference" or boxes.numel() == 0 or points.numel() == 0:
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME,
                backend,
                decision.backend,
                decision.device,
                decision.reason,
                native is not None,
            )
        )
        return points_in_boxes_all_reference(boxes, points)
    output = torch.empty(
        (boxes.shape[0], points.shape[1], boxes.shape[1]), dtype=torch.int32, device=boxes.device
    )
    try:
        native.points_in_boxes_all_forward(boxes, points, output)
    except RuntimeError as exc:
        if backend != "auto":
            record_dispatch_report(
                DispatchReport(OPERATOR_NAME, backend, "error", decision.device, str(exc), True)
            )
            raise
        fallback = DispatchDecision(
            "reference",
            decision.device,
            "native backend failed; falling back to PyTorch reference",
        )
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME,
                backend,
                "reference",
                decision.device,
                fallback.reason,
                True,
                True,
                str(exc),
            )
        )
        return points_in_boxes_all_reference(boxes, points)
    record_dispatch_report(
        DispatchReport(
            OPERATOR_NAME, backend, decision.backend, decision.device, decision.reason, True
        )
    )
    return output


__all__ = ["points_in_boxes_all", "points_in_boxes_all_reference"]
