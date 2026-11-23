"""Reference transcription of the upstream rotated-BEV IoU and NMS kernels.

BEVFusion-style ``iou3d_utils`` wrappers call CUDA-only extension entry points
(``boxes_overlap_bev_gpu``/``boxes_iou_bev_gpu``, ``nms_gpu`` and
``nms_normal_gpu``), so a MUSA machine cannot execute them.  The helpers below
transcribe the upstream algorithm line by line and are used only by the
semantic diff tests, which compare the native MUSA operators against it.

Boxes use the upstream ``[x1, y1, x2, y2, ry]`` layout: ``(x1, y1)`` and
``(x2, y2)`` are the axis-aligned extents of the rectangle before it is rotated
about its center by ``ry``.
"""

from __future__ import annotations

import math

import torch

_EPS = 1e-8
_MARGIN = 1e-5


def _cross(p1, p2, p0):
    """Upstream ``cross(p1, p2, p0)``: signed area of the triangle ``p0p1p2``."""
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])


def _cross_vectors(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _check_rect_cross(p1, p2, q1, q2):
    return (
        min(p1[0], p2[0]) <= max(q1[0], q2[0])
        and min(q1[0], q2[0]) <= max(p1[0], p2[0])
        and min(p1[1], p2[1]) <= max(q1[1], q2[1])
        and min(q1[1], q2[1]) <= max(p1[1], p2[1])
    )


def _check_in_box2d(box, point):
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    angle_cos = math.cos(-box[4])
    angle_sin = math.sin(-box[4])
    rot_x = (point[0] - center_x) * angle_cos + (point[1] - center_y) * angle_sin + center_x
    rot_y = -(point[0] - center_x) * angle_sin + (point[1] - center_y) * angle_cos + center_y
    return (
        rot_x > box[0] - _MARGIN
        and rot_x < box[2] + _MARGIN
        and rot_y > box[1] - _MARGIN
        and rot_y < box[3] + _MARGIN
    )


def _intersection(p1, p0, q1, q0):
    """Upstream ``intersection``: segment ``p0p1`` against segment ``q0q1``."""
    if not _check_rect_cross(p0, p1, q0, q1):
        return None
    s1 = _cross(q0, p1, p0)
    s2 = _cross(p1, q1, p0)
    s3 = _cross(p0, q1, q0)
    s4 = _cross(q1, p1, q0)
    if not (s1 * s2 > 0 and s3 * s4 > 0):
        return None
    s5 = _cross(q1, p1, p0)
    if abs(s5 - s1) > _EPS:
        x = (s5 * q0[0] - s1 * q1[0]) / (s5 - s1)
        y = (s5 * q0[1] - s1 * q1[1]) / (s5 - s1)
        return (x, y)
    a0 = p0[1] - p1[1]
    b0 = p1[0] - p0[0]
    c0 = p0[0] * p1[1] - p1[0] * p0[1]
    a1 = q0[1] - q1[1]
    b1 = q1[0] - q0[0]
    c1 = q0[0] * q1[1] - q1[0] * q0[1]
    determinant = a0 * b1 - a1 * b0
    if determinant == 0:
        return None
    return ((b0 * c1 - b1 * c0) / determinant, (a1 * c0 - a0 * c1) / determinant)


def _rotate_around_center(center, angle_cos, angle_sin, point):
    new_x = (point[0] - center[0]) * angle_cos + (point[1] - center[1]) * angle_sin + center[0]
    new_y = -(point[0] - center[0]) * angle_sin + (point[1] - center[1]) * angle_cos + center[1]
    return (new_x, new_y)


def _box_overlap(box_a, box_b):
    """Upstream ``box_overlap``: intersection area of two rotated rectangles."""
    center_a = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
    center_b = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
    corners_a = [(box_a[0], box_a[1]), (box_a[2], box_a[1]), (box_a[2], box_a[3]), (box_a[0], box_a[3])]
    corners_b = [(box_b[0], box_b[1]), (box_b[2], box_b[1]), (box_b[2], box_b[3]), (box_b[0], box_b[3])]

    cos_a, sin_a = math.cos(box_a[4]), math.sin(box_a[4])
    cos_b, sin_b = math.cos(box_b[4]), math.sin(box_b[4])
    corners_a = [_rotate_around_center(center_a, cos_a, sin_a, corner) for corner in corners_a]
    corners_b = [_rotate_around_center(center_b, cos_b, sin_b, corner) for corner in corners_b]
    corners_a.append(corners_a[0])
    corners_b.append(corners_b[0])

    cross_points = []
    center = (0.0, 0.0)
    for i in range(4):
        for j in range(4):
            point = _intersection(corners_a[i + 1], corners_a[i], corners_b[j + 1], corners_b[j])
            if point is not None:
                cross_points.append(point)
                center = (center[0] + point[0], center[1] + point[1])

    for k in range(4):
        if _check_in_box2d(box_a, corners_b[k]):
            cross_points.append(corners_b[k])
            center = (center[0] + corners_b[k][0], center[1] + corners_b[k][1])
        if _check_in_box2d(box_b, corners_a[k]):
            cross_points.append(corners_a[k])
            center = (center[0] + corners_a[k][0], center[1] + corners_a[k][1])

    count = len(cross_points)
    if count == 0:
        return 0.0
    center = (center[0] / count, center[1] / count)
    cross_points.sort(
        key=lambda point: math.atan2(point[1] - center[1], point[0] - center[0]), reverse=True
    )
    area = 0.0
    for k in range(count - 1):
        area += _cross_vectors(
            _sub(cross_points[k], cross_points[0]), _sub(cross_points[k + 1], cross_points[0])
        )
    return abs(area) / 2.0


def _iou_bev(box_a, box_b):
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    overlap = _box_overlap(box_a, box_b)
    return overlap / max(area_a + area_b - overlap, _EPS)


def _iou_normal(box_a, box_b):
    left = max(box_a[0], box_b[0])
    right = min(box_a[2], box_b[2])
    top = max(box_a[1], box_b[1])
    bottom = min(box_a[3], box_b[3])
    width = max(right - left, 0.0)
    height = max(bottom - top, 0.0)
    intersection = width * height
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / max(area_a + area_b - intersection, _EPS)


def _to_float_rows(boxes: torch.Tensor) -> list[list[float]]:
    return boxes.detach().to(torch.float32).cpu().tolist()


def boxes_overlap_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Upstream ``boxes_overlap_bev_gpu``: ``[A, 5]`` x ``[B, 5]`` -> ``[A, B]`` areas."""
    rows_a = _to_float_rows(boxes_a)
    rows_b = _to_float_rows(boxes_b)
    output = torch.zeros((len(rows_a), len(rows_b)), dtype=torch.float32)
    for row_a, box_a in enumerate(rows_a):
        for row_b, box_b in enumerate(rows_b):
            output[row_a, row_b] = _box_overlap(box_a, box_b)
    return output


def boxes_iou_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Upstream ``boxes_iou_bev_gpu``: ``[A, 5]`` x ``[B, 5]`` -> ``[A, B]`` IoU."""
    rows_a = _to_float_rows(boxes_a)
    rows_b = _to_float_rows(boxes_b)
    output = torch.zeros((len(rows_a), len(rows_b)), dtype=torch.float32)
    for row_a, box_a in enumerate(rows_a):
        for row_b, box_b in enumerate(rows_b):
            output[row_a, row_b] = _iou_bev(box_a, box_b)
    return output


def _greedy_nms(boxes, scores, threshold, rotated):
    order = torch.sort(scores.detach().cpu(), 0, descending=True).indices.tolist()
    rows = _to_float_rows(boxes)
    suppressed = [False] * len(order)
    keep = []
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(i)
        for j in range(i + 1, len(order)):
            if suppressed[j]:
                continue
            overlap = _iou_bev(rows[i], rows[j]) if rotated else _iou_normal(rows[i], rows[j])
            if overlap > threshold:
                suppressed[j] = True
    return order, keep


def nms_gpu(boxes, scores, threshold, pre_maxsize=None, post_max_size=None):
    """Upstream ``nms_gpu``: rotated NMS over ``[N, 5]`` boxes with score-sorted order."""
    order = torch.sort(scores.detach().cpu(), 0, descending=True).indices.tolist()
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    ordered = boxes.detach().to(torch.float32).cpu()[order]
    _, keep = _greedy_nms(ordered, scores.detach().cpu()[order], threshold, rotated=True)
    indices = torch.tensor([order[index] for index in keep], dtype=torch.long)
    if post_max_size is not None:
        indices = indices[:post_max_size]
    return indices


def nms_normal_gpu(boxes, scores, threshold):
    """Upstream ``nms_normal_gpu``: axis-aligned NMS that ignores the yaw column."""
    order = torch.sort(scores.detach().cpu(), 0, descending=True).indices.tolist()
    ordered = boxes.detach().to(torch.float32).cpu()[order]
    _, keep = _greedy_nms(ordered, scores.detach().cpu()[order], threshold, rotated=False)
    return torch.tensor([order[index] for index in keep], dtype=torch.long)
