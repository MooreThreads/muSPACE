"""CenterPoint single-head training target assignment.

This module implements the target-building inner loop from CenterPoint commit
``3cf7d870537e287c99b43b68636ea392a5e6f519``.  It deliberately has no model,
dataset, dataloader, or autograd integration: callers split tasks and batches
before calling this single-sample, single-head operator.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from ..runtime.kernel_dispatch import (
    BackendName,
    DispatchReport,
    explain_dispatch,
    record_dispatch_report,
)
from ..runtime.native import NativeModuleUnavailable, native_module_proxy

OPERATOR_NAME = "centerpoint_target_assign"


def _native():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _parameters(
    voxel_size: Sequence[float], pc_range: Sequence[float], feature_map_size: Sequence[int]
) -> tuple[float, float, float, float, int, int]:
    if len(voxel_size) < 2 or len(pc_range) < 2 or len(feature_map_size) != 2:
        raise ValueError(
            "voxel_size and pc_range need at least 2 values; feature_map_size must be [W, H]"
        )
    vx, vy = float(voxel_size[0]), float(voxel_size[1])
    px, py = float(pc_range[0]), float(pc_range[1])
    width, height = int(feature_map_size[0]), int(feature_map_size[1])
    if vx <= 0 or vy <= 0 or width <= 0 or height <= 0:
        raise ValueError("voxel_size and feature_map_size must be positive")
    return vx, vy, px, py, width, height


def _validate(
    boxes: torch.Tensor, class_ids: torch.Tensor, num_classes: int, max_objs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(boxes, torch.Tensor) or not isinstance(class_ids, torch.Tensor):
        raise TypeError("boxes and class_ids must be torch.Tensor instances")
    if boxes.dtype != torch.float32:
        raise TypeError("boxes must use float32")
    if boxes.ndim != 2 or boxes.shape[1] != 9:
        raise ValueError("boxes must have shape [N, 9] [x, y, z, w, l, h, vx, vy, yaw]")
    if class_ids.dtype != torch.int64:
        raise TypeError("class_ids must use int64 task-local one-based labels")
    if class_ids.ndim != 1 or class_ids.shape[0] != boxes.shape[0]:
        raise ValueError("class_ids must have shape [N] matching boxes")
    if boxes.device != class_ids.device:
        raise ValueError("boxes and class_ids must be on the same device")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if max_objs <= 0:
        raise ValueError("max_objs must be positive")
    return boxes.contiguous(), class_ids.contiguous()


def _check_class_ids(class_ids: torch.Tensor, num_classes: int) -> None:
    if class_ids.numel() and (bool((class_ids < 1).any()) or bool((class_ids > num_classes).any())):
        raise ValueError("class_ids must be task-local one-based values in [1, num_classes]")


def _radius(length: float, width: float, overlap: float) -> int:
    a1, b1 = 1.0, length + width
    c1 = width * length * (1.0 - overlap) / (1.0 + overlap)
    r1 = (b1 + math.sqrt(b1 * b1 - 4.0 * a1 * c1)) / 2.0
    a2, b2 = 4.0, 2.0 * (length + width)
    c2 = (1.0 - overlap) * width * length
    r2 = (b2 + math.sqrt(b2 * b2 - 4.0 * a2 * c2)) / 2.0
    a3, b3 = 4.0 * overlap, -2.0 * overlap * (length + width)
    c3 = (overlap - 1.0) * width * length
    r3 = (b3 + math.sqrt(b3 * b3 - 4.0 * a3 * c3)) / 2.0
    return int(min(r1, r2, r3))


def centerpoint_target_assign_reference(
    boxes: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    num_classes: int,
    out_size_factor: int,
    gaussian_overlap: float,
    min_radius: int,
    voxel_size: Sequence[float],
    pc_range: Sequence[float],
    feature_map_size: Sequence[int],
    max_objs: int = 500,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Literal, non-differentiable PyTorch oracle for the frozen upstream loop."""
    boxes, class_ids = _validate(boxes, class_ids, num_classes, max_objs)
    _check_class_ids(class_ids, num_classes)
    vx, vy, px, py, fmap_w, fmap_h = _parameters(voxel_size, pc_range, feature_map_size)
    if out_size_factor <= 0 or min_radius < 0 or not 0 < gaussian_overlap < 1:
        raise ValueError(
            "out_size_factor must be positive; min_radius non-negative; gaussian_overlap in (0, 1)"
        )
    heatmap = boxes.new_zeros((num_classes, fmap_h, fmap_w))
    anno_box = boxes.new_zeros((max_objs, 10))
    inds = torch.zeros((max_objs,), dtype=torch.int64, device=boxes.device)
    mask = torch.zeros((max_objs,), dtype=torch.uint8, device=boxes.device)
    for index in range(min(boxes.shape[0], max_objs)):
        box = boxes[index]
        scaled_w = float(box[3]) / vx / out_size_factor
        scaled_l = float(box[4]) / vy / out_size_factor
        if scaled_w <= 0 or scaled_l <= 0:
            continue
        radius = max(min_radius, _radius(scaled_l, scaled_w, gaussian_overlap))
        center_x = (float(box[0]) - px) / vx / out_size_factor
        center_y = (float(box[1]) - py) / vy / out_size_factor
        # numpy.astype(np.int32) in the upstream pipeline truncates toward zero.
        int_x, int_y = int(center_x), int(center_y)
        if not (0 <= int_x < fmap_w and 0 <= int_y < fmap_h):
            continue
        left, right = min(int_x, radius), min(fmap_w - int_x, radius + 1)
        top, bottom = min(int_y, radius), min(fmap_h - int_y, radius + 1)
        ys = torch.arange(-top, bottom, device=boxes.device, dtype=torch.float32)
        xs = torch.arange(-left, right, device=boxes.device, dtype=torch.float32)
        gaussian = torch.exp(
            -(ys[:, None].square() + xs[None, :].square())
            / (2.0 * (radius * 2.0 + 1.0) ** 2 / 36.0)
        )
        channel = int(class_ids[index]) - 1
        region = heatmap[channel, int_y - top : int_y + bottom, int_x - left : int_x + right]
        heatmap[channel, int_y - top : int_y + bottom, int_x - left : int_x + right] = (
            torch.maximum(region, gaussian)
        )
        inds[index] = int_y * fmap_w + int_x
        mask[index] = 1
        anno_box[index, 0] = center_x - int_x
        anno_box[index, 1] = center_y - int_y
        anno_box[index, 2] = box[2]
        anno_box[index, 3:6] = torch.log(box[3:6])
        anno_box[index, 6] = torch.sin(box[8])
        anno_box[index, 7] = torch.cos(box[8])
        anno_box[index, 8:10] = box[6:8]
    return heatmap, anno_box, inds, mask


def centerpoint_target_assign(
    boxes: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    num_classes: int,
    out_size_factor: int,
    gaussian_overlap: float,
    min_radius: int,
    voxel_size: Sequence[float],
    pc_range: Sequence[float],
    feature_map_size: Sequence[int],
    max_objs: int = 500,
    backend: BackendName = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build CenterPoint heatmap and regression targets for one sample and one head."""
    boxes, class_ids = _validate(boxes, class_ids, num_classes, max_objs)
    vx, vy, px, py, fmap_w, fmap_h = _parameters(voxel_size, pc_range, feature_map_size)
    if out_size_factor <= 0 or min_radius < 0 or not 0 < gaussian_overlap < 1:
        raise ValueError(
            "out_size_factor must be positive; min_radius non-negative; gaussian_overlap in (0, 1)"
        )
    # Importing TVM-FFI can build its optional PyTorch DLPack bridge.  A CPU
    # reference call never needs that work, so retain the normal dispatch
    # behavior without probing the native module on non-MUSA tensors.
    is_musa = boxes.device.type in {"musa", "privateuseone"} or bool(
        getattr(boxes, "is_musa", False)
    )
    native = _native() if is_musa and backend != "reference" else None
    decision = explain_dispatch(boxes, backend=backend, native_available=native is not None)
    if decision.backend == "error":
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME,
                backend,
                "reference",
                decision.device,
                decision.reason,
                native is not None,
            )
        )
        return centerpoint_target_assign_reference(
            boxes,
            class_ids,
            num_classes=num_classes,
            out_size_factor=out_size_factor,
            gaussian_overlap=gaussian_overlap,
            min_radius=min_radius,
            voxel_size=voxel_size,
            pc_range=pc_range,
            feature_map_size=feature_map_size,
            max_objs=max_objs,
        )
    heatmap = boxes.new_zeros((num_classes, fmap_h, fmap_w))
    anno_box = boxes.new_zeros((max_objs, 10))
    inds = torch.zeros((max_objs,), dtype=torch.int64, device=boxes.device)
    mask = torch.zeros((max_objs,), dtype=torch.uint8, device=boxes.device)
    # TVM-FFI rejects zero-sized tensors as non-contiguous even when PyTorch
    # reports a canonical contiguous layout.  The native kernel has no work
    # to perform for an empty ground-truth set, so preserve its zero-output
    # contract without crossing that incompatible ABI boundary.
    if boxes.shape[0] == 0:
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME,
                backend,
                "musa",
                decision.device,
                "empty boxes fast path",
                True,
            )
        )
        return heatmap, anno_box, inds, mask
    try:
        native.centerpoint_target_assign_forward(
            boxes,
            class_ids,
            heatmap,
            anno_box,
            inds,
            mask,
            int(out_size_factor),
            float(gaussian_overlap),
            int(min_radius),
            float(vx),
            float(vy),
            float(px),
            float(py),
            int(fmap_w),
            int(fmap_h),
            int(max_objs),
        )
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
                True,
                True,
                str(exc),
            )
        )
        return centerpoint_target_assign_reference(
            boxes,
            class_ids,
            num_classes=num_classes,
            out_size_factor=out_size_factor,
            gaussian_overlap=gaussian_overlap,
            min_radius=min_radius,
            voxel_size=voxel_size,
            pc_range=pc_range,
            feature_map_size=feature_map_size,
            max_objs=max_objs,
        )
    record_dispatch_report(
        DispatchReport(OPERATOR_NAME, backend, "musa", decision.device, decision.reason, True)
    )
    return heatmap, anno_box, inds, mask


__all__ = ["centerpoint_target_assign", "centerpoint_target_assign_reference"]
