"""BEVFusion rotated BEV IoU and NMS operators.

The public contract follows BEVFusion's ``mmdet3d.ops.iou3d`` extension:
boxes are ``[x1, y1, x2, y2, ry]`` and all IoU outputs are pairwise matrices.
"""

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

OPERATOR_NAME = "boxes_iou_bev"
NMS_OPERATOR_NAME = "nms_bev"
NMS_NORMAL_OPERATOR_NAME = "nms_bev_normal"


def _native_extension():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _validate_boxes(boxes: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(boxes, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if boxes.dtype != torch.float32:
        raise TypeError(f"{name} must use float32")
    if boxes.dim() != 2 or boxes.size(1) != 5:
        raise ValueError(f"{name} must have shape [N, 5] [x1, y1, x2, y2, ry]")
    return boxes.contiguous()


def _validate_scores(scores: torch.Tensor, count: int) -> torch.Tensor:
    if scores.dtype != torch.float32:
        raise TypeError("scores must use float32")
    if scores.dim() != 1 or scores.size(0) != count:
        raise ValueError("scores must have shape [N] matching boxes")
    return scores.contiguous()


def _box_corners(box: list[float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2, angle = box
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = []
    for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        dx, dy = x - cx, y - cy
        corners.append((dx * cos_a + dy * sin_a + cx, -dx * sin_a + dy * cos_a + cy))
    return corners


def _edge_cross(a, b, point):
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def _line_intersection(start, end, edge_start, edge_end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]
    denominator = dx * ey - dy * ex
    if abs(denominator) <= 1e-12:
        return end
    t = ((edge_start[0] - start[0]) * ey - (edge_start[1] - start[1]) * ex) / denominator
    return (start[0] + t * dx, start[1] + t * dy)


def _clip_polygon(subject, clipper):
    """Independent convex-polygon clipping oracle for the reference path."""
    result = list(subject)
    for edge_index, edge_start in enumerate(clipper):
        if not result:
            return []
        edge_end = clipper[(edge_index + 1) % len(clipper)]
        clipped = []
        previous = result[-1]
        previous_inside = _edge_cross(edge_start, edge_end, previous) >= -1e-10
        for current in result:
            current_inside = _edge_cross(edge_start, edge_end, current) >= -1e-10
            if current_inside != previous_inside:
                clipped.append(_line_intersection(previous, current, edge_start, edge_end))
            if current_inside:
                clipped.append(current)
            previous, previous_inside = current, current_inside
        result = clipped
    return result


def _overlap_area(box_a: list[float], box_b: list[float]) -> float:
    points = _clip_polygon(_box_corners(box_a), _box_corners(box_b))
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            points[i][0] * points[(i + 1) % len(points)][1]
            - points[(i + 1) % len(points)][0] * points[i][1]
            for i in range(len(points))
        )
    ) * 0.5


def boxes_iou_bev_reference(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    boxes_a = _validate_boxes(boxes_a, "boxes_a")
    boxes_b = _validate_boxes(boxes_b, "boxes_b")
    result = boxes_a.new_zeros((boxes_a.size(0), boxes_b.size(0)))
    a_list, b_list = boxes_a.detach().cpu().tolist(), boxes_b.detach().cpu().tolist()
    for i, box_a in enumerate(a_list):
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        for j, box_b in enumerate(b_list):
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            overlap = _overlap_area(box_a, box_b)
            result[i, j] = overlap / max(area_a + area_b - overlap, 1e-8)
    return result


def _nms_reference(boxes, scores, threshold, rotated):
    order = torch.sort(scores, 0, descending=True).indices
    keep = []
    boxes_cpu = boxes.detach().cpu().tolist()
    while order.numel():
        current = int(order[0])
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        current_box = boxes_cpu[current]
        overlaps = []
        for index in remaining.tolist():
            if rotated:
                overlap = _overlap_area(current_box, boxes_cpu[index])
                area_a = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                box = boxes_cpu[index]
                area_b = (box[2] - box[0]) * (box[3] - box[1])
                overlaps.append(overlap / max(area_a + area_b - overlap, 1e-8))
            else:
                a, b = current_box, boxes_cpu[index]
                overlap = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(
                    min(a[3], b[3]) - max(a[1], b[1]), 0.0
                )
                area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
                area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
                overlaps.append(overlap / max(area_a + area_b - overlap, 1e-8))
        order = remaining[torch.tensor(overlaps, device=remaining.device) <= threshold]
    return boxes.new_tensor(keep, dtype=torch.long)


def _record(name, requested, decision, native, fallback=False, fallback_reason=None):
    record_dispatch_report(DispatchReport(name, requested, decision.backend, decision.device,
                                          decision.reason, native, fallback, fallback_reason))


def _dispatch(value, backend, name):
    native = _native_extension() is not None
    decision = explain_dispatch(value, backend=backend, native_available=native)
    if decision.backend == "error":
        _record(name, backend, decision, native)
        raise RuntimeError(decision.reason)
    return native, decision


def boxes_iou_bev(boxes_a, boxes_b, *, backend: BackendName = "auto"):
    boxes_a, boxes_b = _validate_boxes(boxes_a, "boxes_a"), _validate_boxes(boxes_b, "boxes_b")
    if boxes_a.device != boxes_b.device:
        raise ValueError("boxes_a and boxes_b must be on the same device")
    native, decision = _dispatch(boxes_a, backend, OPERATOR_NAME)
    if decision.backend == "reference":
        _record(OPERATOR_NAME, backend, decision, native)
        return boxes_iou_bev_reference(boxes_a, boxes_b)
    if boxes_a.size(0) == 0 or boxes_b.size(0) == 0:
        _record(OPERATOR_NAME, backend, decision, native)
        return boxes_a.new_zeros((boxes_a.size(0), boxes_b.size(0)))
    try:
        output = boxes_a.new_empty((boxes_a.size(0), boxes_b.size(0)))
        _native_extension().boxes_iou_bev_forward(boxes_a, boxes_b, output)
        _record(OPERATOR_NAME, backend, decision, native)
        return output
    except RuntimeError as exc:
        if backend != "auto":
            raise
        fallback = DispatchDecision(
            "reference", decision.device, "native backend failed; falling back"
        )
        _record(OPERATOR_NAME, backend, fallback, native, True, str(exc))
        return boxes_iou_bev_reference(boxes_a, boxes_b)


def _nms(boxes, scores, threshold, *, backend, rotated, pre_maxsize=None, post_max_size=None):
    boxes = _validate_boxes(boxes, "boxes")
    scores = _validate_scores(scores, boxes.size(0))
    if boxes.device != scores.device:
        raise ValueError("boxes and scores must be on the same device")
    name = NMS_OPERATOR_NAME if rotated else NMS_NORMAL_OPERATOR_NAME
    native, decision = _dispatch(boxes, backend, name)
    order = torch.sort(scores, 0, descending=True).indices
    if pre_maxsize is not None:
        if pre_maxsize < 0:
            raise ValueError("pre_maxsize must be non-negative")
        order = order[:pre_maxsize]
    ordered_boxes = boxes.index_select(0, order).contiguous()
    if decision.backend == "reference":
        keep = _nms_reference(ordered_boxes, scores.index_select(0, order), threshold, rotated)
    else:
        if ordered_boxes.size(0) == 0:
            _record(name, backend, decision, native)
            return order
        try:
            keep_buffer = torch.empty(ordered_boxes.size(0), dtype=torch.int64, device=boxes.device)
            mask_buffer = torch.empty(
                (ordered_boxes.size(0), (ordered_boxes.size(0) + 63) // 64),
                dtype=torch.int64,
                device=boxes.device,
            )
            native_nms = (
                _native_extension().nms_bev_forward
                if rotated
                else _native_extension().nms_bev_normal_forward
            )
            count = native_nms(ordered_boxes, float(threshold), mask_buffer, keep_buffer)
            keep = keep_buffer[: int(count)]
        except RuntimeError as exc:
            if backend != "auto":
                raise
            fallback = DispatchDecision(
                "reference", decision.device, "native backend failed; falling back"
            )
            _record(name, backend, fallback, native, True, str(exc))
            keep = _nms_reference(ordered_boxes, scores.index_select(0, order), threshold, rotated)
            keep = order.index_select(0, keep)
            if post_max_size is not None:
                if post_max_size < 0:
                    raise ValueError("post_max_size must be non-negative")
                keep = keep[:post_max_size]
            return keep.contiguous()
    keep = order.index_select(0, keep)
    if post_max_size is not None:
        if post_max_size < 0:
            raise ValueError("post_max_size must be non-negative")
        keep = keep[:post_max_size]
    _record(name, backend, decision, native)
    return keep.contiguous()


def nms_bev(
    boxes,
    scores,
    thresh,
    pre_maxsize=None,
    post_max_size=None,
    *,
    backend: BackendName = "auto",
):
    return _nms(boxes, scores, thresh, backend=backend, rotated=True,
                pre_maxsize=pre_maxsize, post_max_size=post_max_size)


def nms_bev_normal(boxes, scores, thresh, *, backend: BackendName = "auto"):
    return _nms(boxes, scores, thresh, backend=backend, rotated=False)


__all__ = ["boxes_iou_bev", "boxes_iou_bev_reference", "nms_bev", "nms_bev_normal"]
