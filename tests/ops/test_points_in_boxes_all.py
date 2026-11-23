import math
import sys
import types

import pytest
import torch

from mu_space import (
    clear_dispatch_report,
    get_last_dispatch_report,
    points_in_boxes_all,
    points_in_boxes_all_reference,
    points_in_boxes_batch,
    points_in_boxes_batch_reference,
    points_in_boxes_cpu,
    points_in_boxes_gpu,
    points_in_boxes_gpu_reference,
)
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


def _upstream_mask(points, boxes):
    """Scalar transcription of the upstream point-in-box kernel: ``[B, N, M]`` mask."""
    batch, npoints, _ = points.shape
    boxes_per_batch = boxes.shape[1]
    output = torch.zeros((batch, npoints, boxes_per_batch), dtype=torch.int32)
    for batch_index in range(batch):
        for point_index in range(npoints):
            px, py, pz = (float(value) for value in points[batch_index, point_index])
            for box_index in range(boxes_per_batch):
                cx, cy, cz, width, length, height, yaw = (
                    float(value) for value in boxes[batch_index, box_index]
                )
                if abs(pz - (cz + height / 2.0)) > height / 2.0:
                    continue
                rotation = yaw + math.pi / 2.0
                shift_x, shift_y = px - cx, py - cy
                local_x = shift_x * math.cos(rotation) - shift_y * math.sin(rotation)
                local_y = shift_x * math.sin(rotation) + shift_y * math.cos(rotation)
                if -length / 2.0 < local_x < length / 2.0 and -width / 2.0 < local_y < width / 2.0:
                    output[batch_index, point_index, box_index] = 1
    return output


def _upstream_gpu_index(points, boxes):
    """Upstream ``points_in_boxes_gpu``: first hit wins, ``-1`` background."""
    mask = _upstream_mask(points, boxes)
    batch, npoints, boxes_per_batch = mask.shape
    output = torch.full((batch, npoints), -1, dtype=torch.int32)
    for batch_index in range(batch):
        for point_index in range(npoints):
            for box_index in range(boxes_per_batch):
                if mask[batch_index, point_index, box_index]:
                    output[batch_index, point_index] = box_index
                    break
    return output


def _upstream_cpu_mask(points, boxes):
    """Upstream ``points_in_boxes_cpu``: unbatched ``[M, npoints]`` mask."""
    return _upstream_mask(points.unsqueeze(0), boxes.unsqueeze(0))[0].transpose(0, 1).contiguous()


def _overlapping_fixture():
    # Boxes are [x, y, z, w, l, h, ry] with z as the bottom center; at ry == 0 the
    # footprint is |x| < w / 2 and |y| < l / 2.
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 4.0, 2.0, 0.0],
                [10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ]
        ]
    )
    points = torch.tensor(
        [
            [
                [0.0, 0.0, 0.5],
                [5.0, 5.0, 0.0],
                [1.5, 0.5, 0.5],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 2.001],
            ]
        ]
    )
    return boxes, points


def test_gpu_adapter_first_hit_and_background():
    boxes, points = _overlapping_fixture()
    actual = points_in_boxes_gpu(points, boxes, backend="reference")
    assert actual.dtype == torch.int32
    assert actual.shape == (1, 6)
    # A point inside boxes 0 and 1 keeps the lowest matching index; points outside
    # every box keep the upstream -1 background; z is inclusive at the half height.
    torch.testing.assert_close(actual, torch.tensor([[0, -1, 1, 0, 0, -1]], dtype=torch.int32))


def test_gpu_adapter_strict_local_bounds():
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 4.0, 2.0, 0.0]]])
    # width 2 => strict |x| < 1, length 4 => strict |y| < 2, height 2 => inclusive |z| <= 1
    points = torch.tensor(
        [[[0.999, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.999, 1.999, 1.0]]]
    )
    torch.testing.assert_close(
        points_in_boxes_gpu(points, boxes, backend="reference"),
        torch.tensor([[0, -1, -1, 0]], dtype=torch.int32),
    )


def test_gpu_adapter_matches_upstream_oracle_on_cpu():
    torch.manual_seed(0)
    boxes = torch.randn(2, 5, 7)
    boxes[..., 3:6] = boxes[..., 3:6].abs() + 0.5
    points = torch.randn(2, 32, 3) * 2.0
    actual = points_in_boxes_gpu(points, boxes, backend="reference")
    expected = _upstream_gpu_index(points, boxes)
    torch.testing.assert_close(actual, expected)


def test_batch_adapter_matches_all_and_upstream_oracle():
    boxes, points = _overlapping_fixture()
    actual = points_in_boxes_batch(points, boxes, backend="reference")
    torch.testing.assert_close(actual, points_in_boxes_all(boxes, points, backend="reference"))
    torch.testing.assert_close(actual, _upstream_mask(points, boxes))
    torch.testing.assert_close(actual, points_in_boxes_batch_reference(points, boxes))


def test_cpu_adapter_unbatched_contract():
    boxes, points = _overlapping_fixture()
    actual = points_in_boxes_cpu(points[0], boxes[0])
    assert actual.dtype == torch.int32
    assert actual.shape == (boxes.shape[1], points.shape[1])
    torch.testing.assert_close(actual, _upstream_cpu_mask(points[0], boxes[0]))


def test_cpu_adapter_reports_reference_dispatch():
    boxes, points = _overlapping_fixture()
    clear_dispatch_report()
    points_in_boxes_cpu(points[0], boxes[0])
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "points_in_boxes_cpu"
    assert report.selected_backend == "reference"


@pytest.mark.parametrize("shape", [(0, 3, 7), (1, 0, 7), (1, 2, 7)])
def test_gpu_adapter_empty_inputs(shape):
    b, m, _ = shape
    boxes = torch.empty(shape, dtype=torch.float32)
    points = torch.empty((b, 4 if m == 0 else 0, 3), dtype=torch.float32)
    for actual in (
        points_in_boxes_gpu(points, boxes, backend="reference"),
        points_in_boxes_gpu_reference(points, boxes),
    ):
        assert actual.shape == (b, points.shape[1])
        if m == 0:
            assert torch.all(actual == -1)


def test_adapter_validation_errors():
    boxes, points = _overlapping_fixture()
    with pytest.raises(ValueError, match=r"\[B, M, 7\]"):
        points_in_boxes_gpu(points, boxes[0], backend="reference")
    with pytest.raises(ValueError, match="same batch"):
        points_in_boxes_gpu(points, boxes.repeat(2, 1, 1), backend="reference")
    with pytest.raises(TypeError, match="float32"):
        points_in_boxes_batch(points.to(torch.float64), boxes, backend="reference")
    with pytest.raises(ValueError, match=r"\[M, 7\]"):
        points_in_boxes_cpu(points[0], boxes)
    with pytest.raises(ValueError, match=r"\[N, 3\]"):
        points_in_boxes_cpu(points, boxes[0])


def test_gpu_adapter_accepts_non_contiguous_inputs():
    boxes, points = _overlapping_fixture()
    strided_boxes = boxes.expand(-1, -1, -1).transpose(1, 2).transpose(1, 2)
    strided_points = points.transpose(1, 2).transpose(1, 2)
    torch.testing.assert_close(
        points_in_boxes_gpu(strided_points, strided_boxes, backend="reference"),
        points_in_boxes_gpu(points, boxes, backend="reference"),
    )


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_gpu_adapter_native_matches_upstream_oracle():
    torch.manual_seed(1234)
    boxes = torch.randn(2, 6, 7, device="musa")
    boxes[..., 3:6] = boxes[..., 3:6].abs() + 0.5
    points = torch.randn(2, 128, 3, device="musa") * 2.0
    clear_dispatch_report()
    actual = points_in_boxes_gpu(points, boxes, backend="musa").cpu()
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "points_in_boxes_gpu"
    assert report.selected_backend == "musa"
    torch.testing.assert_close(actual, _upstream_gpu_index(points.cpu(), boxes.cpu()))


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_batch_adapter_native_matches_all():
    boxes = torch.tensor(
        [[[0.0, 0.0, 0.0, 2.0, 4.0, 2.0, 0.2], [1.0, 0.0, 0.0, 2.0, 2.0, 3.0, -0.4]]],
        device="musa",
    )
    points = torch.randn((1, 257, 3), device="musa")
    actual = points_in_boxes_batch(points, boxes, backend="musa")
    expected = points_in_boxes_all(boxes, points, backend="musa")
    torch.testing.assert_close(actual, expected)
