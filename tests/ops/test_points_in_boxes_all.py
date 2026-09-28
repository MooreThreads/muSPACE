import math
import sys
import types

import pytest
import torch

from mu_space import points_in_boxes_all, points_in_boxes_all_reference
from mu_space.patcher import BEVFusionPatcher


def test_reference_shape_overlap_rotation_and_boundaries():
    boxes = torch.tensor(
        [[[0.0, 0.0, 0.0, 2.0, 4.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0, 4.0, 2.0, 0.0]]]
    )
    points = torch.tensor([[[0.0, 0.0, 0.0], [0.99, 1.99, 2.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]])
    actual = points_in_boxes_all_reference(boxes, points)
    expected = torch.tensor([[[1, 1], [1, 1], [0, 0], [0, 0]]], dtype=torch.int32)
    torch.testing.assert_close(actual, expected)
    rotated = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 4.0, 2.0, math.pi / 2]]])
    torch.testing.assert_close(
        points_in_boxes_all_reference(rotated, torch.tensor([[[1.9, 0.0, 0.0]]])),
        torch.tensor([[[1]]], dtype=torch.int32),
    )


@pytest.mark.parametrize("shape", [(0, 3, 7), (1, 0, 7), (1, 2, 7)])
def test_empty_inputs(shape):
    b, m, _ = shape
    boxes = torch.empty(shape, dtype=torch.float32)
    points = torch.empty((b, 4 if m == 0 else 0, 3), dtype=torch.float32)
    assert points_in_boxes_all(boxes, points, backend="reference").shape == (b, points.shape[1], m)


def test_validation():
    with pytest.raises(ValueError, match=r"\[B, M, 7\]"):
        points_in_boxes_all(torch.zeros(2, 7), torch.zeros(1, 2, 3), backend="reference")
    with pytest.raises(ValueError, match="same batch"):
        points_in_boxes_all(torch.zeros(1, 2, 7), torch.zeros(2, 3, 3), backend="reference")
    with pytest.raises(TypeError, match="float32"):
        points_in_boxes_all(
            torch.zeros(1, 2, 7, dtype=torch.float64),
            torch.zeros(1, 3, 3),
            backend="reference",
        )


def test_bevfusion_patcher_rebinds_batch_points_helper(monkeypatch):
    ops = types.ModuleType("mmdet3d.ops")
    bev_pool = types.ModuleType("mmdet3d.ops.bev_pool.bev_pool")
    structures = types.ModuleType("mmdet3d.core.bbox.structures.depth_box3d")

    def original(points, boxes):
        return "original"

    bev_pool.bev_pool = original
    ops.points_in_boxes_batch = original
    structures.points_in_boxes_batch = original
    monkeypatch.setitem(sys.modules, "mmdet3d.ops", ops)
    monkeypatch.setitem(sys.modules, bev_pool.__name__, bev_pool)
    monkeypatch.setitem(sys.modules, structures.__name__, structures)
    patcher = BEVFusionPatcher()
    report = patcher.apply(strict=False)
    assert report.action == "applied"
    assert ops.points_in_boxes_batch is structures.points_in_boxes_batch
    assert ops.points_in_boxes_batch is not original
    patcher.restore()
    assert ops.points_in_boxes_batch is original


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_musa_matches_reference():
    boxes = torch.tensor(
        [[[0.0, 0.0, 0.0, 2.0, 4.0, 2.0, 0.2], [1.0, 0.0, 0.0, 2.0, 2.0, 3.0, -0.4]]], device="musa"
    )
    points = torch.randn((1, 257, 3), device="musa")
    actual = points_in_boxes_all(boxes, points, backend="musa").cpu()
    expected = points_in_boxes_all_reference(boxes.cpu(), points.cpu())
    torch.testing.assert_close(actual, expected)
