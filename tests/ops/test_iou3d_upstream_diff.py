"""Semantic diff tests: muSPACE BEV IoU/NMS against the upstream iou3d kernels.

The upstream entry points (``boxes_iou_bev``/``boxes_overlap_bev_gpu``,
``nms_gpu`` and ``nms_normal_gpu``) live in a CUDA-only extension, so the diff
tests compare the muSPACE operators against an independent transcription of the
upstream algorithm in ``upstream_iou3d_reference.py``.  Every native case also
asserts that the MUSA backend was selected, so a silent reference fallback
cannot make the comparison vacuous.
"""

import math

import pytest
import torch

import upstream_iou3d_reference as upstream
from mu_space import (
    boxes_iou_bev,
    clear_dispatch_report,
    get_last_dispatch_report,
    nms_bev,
    nms_bev_normal,
)

MUSA_REQUIRED = not hasattr(torch, "musa") or not torch.musa.is_available()
ATOL = 1e-5
RTOL = 1e-4


def _random_boxes(count, seed, device="cpu"):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.rand(count, 2, generator=generator) * 8.0 - 4.0
    sizes = torch.rand(count, 2, generator=generator) * 2.0 + 0.5
    angles = torch.rand(count, generator=generator) * 2.0 * math.pi - math.pi
    corners = torch.cat(
        (centers - sizes / 2.0, centers + sizes / 2.0, angles.unsqueeze(1)), dim=1
    )
    return corners.to(device)


def _random_scores(count, seed):
    generator = torch.Generator().manual_seed(seed + 977)
    return torch.rand(count, generator=generator)


def test_boxes_iou_bev_reference_matches_upstream_fixed():
    boxes_a = torch.tensor(
        [[0.0, 0.0, 4.0, 4.0, 0.0], [0.0, 0.0, 2.0, 2.0, 0.3], [10.0, 10.0, 12.0, 13.0, -0.7]]
    )
    boxes_b = torch.tensor(
        [[2.0, 2.0, 6.0, 6.0, 0.0], [0.0, 0.0, 2.0, 2.0, 0.3], [0.0, 0.0, 2.0, 2.0, 1.2]]
    )
    actual = boxes_iou_bev(boxes_a, boxes_b, backend="reference")
    expected = upstream.boxes_iou_bev(boxes_a, boxes_b)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def test_boxes_iou_bev_reference_matches_upstream_random():
    boxes = _random_boxes(24, seed=3)
    actual = boxes_iou_bev(boxes[:12], boxes, backend="reference")
    expected = upstream.boxes_iou_bev(boxes[:12], boxes)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(MUSA_REQUIRED, reason="MUSA required")
def test_boxes_iou_bev_musa_matches_upstream():
    boxes = _random_boxes(24, seed=7, device="musa")
    clear_dispatch_report()
    actual = boxes_iou_bev(boxes[:12], boxes, backend="musa").cpu()
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "boxes_iou_bev"
    assert report.selected_backend == "musa"
    expected = upstream.boxes_iou_bev(boxes[:12].cpu(), boxes.cpu())
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(MUSA_REQUIRED, reason="MUSA required")
def test_boxes_iou_bev_musa_identical_rotated_boxes_are_one():
    angles = torch.tensor([0.0, 0.25, 0.5, math.pi / 2, -1.3], device="musa")
    boxes = torch.stack(
        (
            torch.zeros_like(angles),
            torch.zeros_like(angles),
            torch.full_like(angles, 4.0),
            torch.full_like(angles, 2.0),
            angles,
        ),
        dim=1,
    )
    actual = boxes_iou_bev(boxes, boxes, backend="musa").cpu()
    torch.testing.assert_close(actual.diagonal(), torch.ones_like(actual.diagonal()))
    expected = upstream.boxes_iou_bev(boxes.cpu(), boxes.cpu())
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def test_nms_bev_reference_matches_upstream_random():
    boxes = _random_boxes(64, seed=11)
    scores = _random_scores(64, seed=11)
    for threshold in (0.1, 0.3, 0.5, 0.7):
        actual = nms_bev(boxes, scores, threshold, backend="reference")
        expected = upstream.nms_gpu(boxes, scores, threshold)
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(MUSA_REQUIRED, reason="MUSA required")
def test_nms_bev_musa_matches_upstream_random():
    boxes = _random_boxes(96, seed=13, device="musa")
    scores = _random_scores(96, seed=13).to("musa")
    for threshold in (0.2, 0.5, 0.8):
        clear_dispatch_report()
        actual = nms_bev(boxes, scores, threshold, backend="musa").cpu()
        report = get_last_dispatch_report()
        assert report is not None
        assert report.operator == "nms_bev"
        assert report.selected_backend == "musa"
        expected = upstream.nms_gpu(boxes.cpu(), scores.cpu(), threshold)
        torch.testing.assert_close(actual, expected)


def test_nms_bev_pre_and_post_maxsize_match_upstream():
    boxes = _random_boxes(40, seed=17)
    scores = _random_scores(40, seed=17)
    for pre_maxsize, post_max_size in ((10, None), (None, 5), (12, 4)):
        actual = nms_bev(
            boxes,
            scores,
            0.3,
            pre_maxsize=pre_maxsize,
            post_max_size=post_max_size,
            backend="reference",
        )
        expected = upstream.nms_gpu(
            boxes, scores, 0.3, pre_maxsize=pre_maxsize, post_max_size=post_max_size
        )
        torch.testing.assert_close(actual, expected)


def test_nms_bev_threshold_is_strictly_greater():
    # Two unit-height boxes overlapping over half their width: IoU == 1/3 exactly.
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0], [1.0, 0.0, 3.0, 2.0, 0.0]])
    scores = torch.tensor([0.9, 0.8])
    overlap = float(boxes_iou_bev(boxes[:1], boxes[1:], backend="reference")[0, 0])
    actual = nms_bev(boxes, scores, overlap, backend="reference")
    torch.testing.assert_close(actual, upstream.nms_gpu(boxes, scores, overlap))
    assert actual.numel() == 2
    suppressed = nms_bev(boxes, scores, overlap * 0.99, backend="reference")
    torch.testing.assert_close(suppressed, torch.tensor([0]))


def test_nms_bev_normal_ignores_yaw_and_matches_upstream():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 4.0, 4.0, 1.2],
            [0.0, 0.0, 4.0, 4.0, -0.7],
            [6.0, 0.0, 10.0, 4.0, 0.3],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    actual = nms_bev_normal(boxes, scores, 0.5, backend="reference")
    torch.testing.assert_close(actual, upstream.nms_normal_gpu(boxes, scores, 0.5))
    torch.testing.assert_close(actual, torch.tensor([0, 2]))


@pytest.mark.skipif(MUSA_REQUIRED, reason="MUSA required")
def test_nms_bev_normal_musa_matches_upstream():
    boxes = _random_boxes(64, seed=19, device="musa")
    scores = _random_scores(64, seed=19).to("musa")
    for threshold in (0.2, 0.5):
        clear_dispatch_report()
        actual = nms_bev_normal(boxes, scores, threshold, backend="musa").cpu()
        report = get_last_dispatch_report()
        assert report is not None
        assert report.operator == "nms_bev_normal"
        assert report.selected_backend == "musa"
        expected = upstream.nms_normal_gpu(boxes.cpu(), scores.cpu(), threshold)
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(MUSA_REQUIRED, reason="MUSA required")
def test_nms_bev_equal_scores_suppress_identical_boxes():
    # Equal scores leave the tie order to the score sort, so assert the invariant that
    # does not depend on it: exactly one of the identical boxes survives plus the
    # disjoint box.
    boxes = torch.tensor(
        [
            [0.0, 0.0, 4.0, 4.0, 0.0],
            [0.0, 0.0, 4.0, 4.0, 0.0],
            [20.0, 20.0, 24.0, 24.0, 0.0],
            [0.0, 0.0, 4.0, 4.0, 0.0],
        ],
        device="musa",
    )
    scores = torch.full((4,), 0.5, device="musa")
    for backend in ("musa", "reference"):
        kept = set(nms_bev(boxes, scores, 0.5, backend=backend).tolist())
        assert len(kept) == 2
        assert 2 in kept
        assert len(kept & {0, 1, 3}) == 1


@pytest.mark.skipif(MUSA_REQUIRED, reason="MUSA required")
def test_nms_bev_musa_matches_reference_on_large_shape():
    boxes = _random_boxes(512, seed=29, device="musa")
    scores = _random_scores(512, seed=29).to("musa")
    actual = nms_bev(boxes, scores, 0.5, backend="musa")
    expected = nms_bev(boxes, scores, 0.5, backend="reference")
    torch.testing.assert_close(actual, expected)


def test_nms_bev_empty_inputs_match_upstream():
    empty_boxes = torch.empty((0, 5), dtype=torch.float32)
    empty_scores = torch.empty((0,), dtype=torch.float32)
    actual = nms_bev(empty_boxes, empty_scores, 0.5, backend="reference")
    assert actual.numel() == 0
    torch.testing.assert_close(actual, upstream.nms_gpu(empty_boxes, empty_scores, 0.5))
