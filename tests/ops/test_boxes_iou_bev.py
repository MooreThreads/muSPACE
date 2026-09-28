"""Tests for the BEVFusion iou3d BEV IoU/NMS compatibility surface."""

import math
import types
import pytest
import torch

from mu_space import boxes_iou_bev, clear_dispatch_report, get_last_dispatch_report
from mu_space import nms_bev, nms_bev_normal


def test_boxes_iou_bev_axis_aligned():
    boxes_a = torch.tensor([[0.0, 0.0, 4.0, 4.0, 0.0]])
    boxes_b = torch.tensor([[2.0, 2.0, 6.0, 6.0, 0.0]])
    actual = boxes_iou_bev(boxes_a, boxes_b, backend="reference")
    torch.testing.assert_close(actual, torch.tensor([[1.0 / 7.0]]))


@pytest.mark.parametrize("angle", [0.0, 0.25, 0.5, 1.5707963267948966])
def test_boxes_iou_bev_identical_rotated(angle):
    boxes = torch.tensor([[0.0, 0.0, 4.0, 2.0, angle]])
    actual = boxes_iou_bev(boxes, boxes, backend="reference")
    torch.testing.assert_close(actual, torch.ones(1, 1), atol=1e-5, rtol=1e-5)


def test_boxes_iou_bev_disjoint_and_empty():
    boxes_a = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.4]])
    boxes_b = torch.tensor([[20.0, 20.0, 22.0, 22.0, -0.7]])
    torch.testing.assert_close(
        boxes_iou_bev(boxes_a, boxes_b, backend="reference"), torch.zeros(1, 1)
    )
    empty = torch.empty((0, 5), dtype=torch.float32)
    assert boxes_iou_bev(empty, boxes_b, backend="reference").shape == (0, 1)


@pytest.mark.parametrize(
    ("boxes_a", "boxes_b", "expected"),
    [
        ([[0.0, 0.0, 4.0, 4.0, 0.0]], [[1.0, 1.0, 3.0, 3.0, 0.0]], 0.25),
        ([[0.0, 0.0, 2.0, 2.0, 0.0]], [[2.0, 0.0, 4.0, 2.0, 0.0]], 0.0),
        ([[0.0, 0.0, 0.0, 2.0, 0.0]], [[0.0, 0.0, 2.0, 2.0, 0.0]], 0.0),
    ],
)
def test_boxes_iou_bev_containment_tangent_and_degenerate(boxes_a, boxes_b, expected):
    actual = boxes_iou_bev(
        torch.tensor(boxes_a), torch.tensor(boxes_b), backend="reference"
    )
    torch.testing.assert_close(actual, torch.tensor([[expected]]), atol=1e-6, rtol=1e-6)


def test_boxes_iou_bev_validation():
    with pytest.raises(ValueError, match=r"\[N, 5\]"):
        boxes_iou_bev(torch.zeros(2, 4), torch.zeros(2, 5), backend="reference")
    with pytest.raises(TypeError, match="float32"):
        boxes_iou_bev(
            torch.zeros(2, 5, dtype=torch.float64), torch.zeros(2, 5), backend="reference"
        )


def test_rotated_nms_preserves_score_order_and_threshold():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 4.0, 4.0, 0.0],
            [0.1, 0.1, 4.1, 4.1, 0.0],
            [10.0, 10.0, 12.0, 12.0, 0.0],
        ]
    )
    scores = torch.tensor([0.2, 0.9, 0.5])
    actual = nms_bev(boxes, scores, 0.5, backend="reference")
    torch.testing.assert_close(actual, torch.tensor([1, 2]))


def test_normal_nms_uses_axis_aligned_overlap():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 4.0, 4.0, 1.0],
            [1.0, 1.0, 3.0, 3.0, -1.0],
            [10.0, 10.0, 12.0, 12.0, 0.0],
        ]
    )
    scores = torch.tensor([0.8, 0.7, 0.6])
    torch.testing.assert_close(
        nms_bev_normal(boxes, scores, 0.2, backend="reference"), torch.tensor([0, 2])
    )


def test_nms_pre_and_post_max_size():
    boxes = torch.tensor([[float(i), 0.0, float(i + 1), 1.0, 0.0] for i in range(5)])
    scores = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.7])
    actual = nms_bev(boxes, scores, 0.0, 3, 2, backend="reference")
    torch.testing.assert_close(actual, torch.tensor([3, 4]))


def test_nms_equal_scores_and_strict_threshold():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0, 0.0],
            [2.0 / 3.0, 0.0, 8.0 / 3.0, 2.0, 0.0],
            [10.0, 0.0, 12.0, 2.0, 0.0],
        ]
    )
    scores = torch.tensor([0.8, 0.8, 0.1])
    threshold = float(boxes_iou_bev(boxes[:1], boxes[1:2], backend="reference")[0, 0])
    torch.testing.assert_close(
        nms_bev(boxes, scores, threshold, backend="reference"), torch.tensor([0, 1, 2])
    )


def test_auto_cpu_records_reference_dispatch():
    clear_dispatch_report()
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
    boxes_iou_bev(boxes, boxes)
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "boxes_iou_bev"
    assert report.selected_backend == "reference"


def test_bevfusion_patcher_rebinds_iou_and_nms(monkeypatch):
    import sys

    from mu_space.patcher import bevfusion_iou3d_patcher

    iou_module = types.ModuleType("mmdet3d.ops.iou3d.iou3d_utils")
    iou_module.boxes_iou_bev = object()
    iou_module.nms_gpu = object()
    iou_module.nms_normal_gpu = object()
    nms_module = types.ModuleType("mmdet3d.core.post_processing.box3d_nms")
    nms_module.nms_gpu = object()
    nms_module.nms_normal_gpu = object()
    parent_names = ("mmdet3d", "mmdet3d.ops", "mmdet3d.ops.iou3d", "mmdet3d.core",
                    "mmdet3d.core.post_processing")
    modules = {name: types.ModuleType(name) for name in parent_names}
    modules.update({iou_module.__name__: iou_module, nms_module.__name__: nms_module})
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    report = bevfusion_iou3d_patcher.apply(strict=False)
    assert report.action == "applied"
    assert iou_module.boxes_iou_bev.__name__ == "boxes_iou_bev"
    assert iou_module.nms_gpu.__name__ == "nms_bev"
    assert nms_module.nms_normal_gpu.__name__ == "nms_bev_normal"
    restored = bevfusion_iou3d_patcher.restore()
    assert restored.action == "restored"


def test_bevfusion_patcher_supports_reference_backend(monkeypatch):
    import sys

    from mu_space.patcher import bevfusion_iou3d_patcher

    iou_module = types.ModuleType("mmdet3d.ops.iou3d.iou3d_utils")
    nms_module = types.ModuleType("mmdet3d.core.post_processing.box3d_nms")
    for module in (iou_module, nms_module):
        module.nms_gpu = object()
        module.nms_normal_gpu = object()
    iou_module.boxes_iou_bev = object()
    parent_names = ("mmdet3d", "mmdet3d.ops", "mmdet3d.ops.iou3d", "mmdet3d.core",
                    "mmdet3d.core.post_processing")
    modules = {name: types.ModuleType(name) for name in parent_names}
    modules.update({iou_module.__name__: iou_module, nms_module.__name__: nms_module})
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    bevfusion_iou3d_patcher.apply(backend="reference")
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]])
    torch.testing.assert_close(iou_module.boxes_iou_bev(boxes, boxes), torch.ones(1, 1))
    with pytest.raises(ValueError, match="strict=True"):
        bevfusion_iou3d_patcher.apply(strict=True, backend="reference")
    bevfusion_iou3d_patcher.restore()


def test_bevfusion_patcher_rejects_incomplete_install_without_mutation(monkeypatch):
    import sys

    from mu_space.patcher import PatchError, bevfusion_iou3d_patcher

    iou_module = types.ModuleType("mmdet3d.ops.iou3d.iou3d_utils")
    original_iou = object()
    original_nms = object()
    iou_module.boxes_iou_bev = original_iou
    iou_module.nms_gpu = original_nms
    iou_module.nms_normal_gpu = object()
    nms_module = types.ModuleType("mmdet3d.core.post_processing.box3d_nms")
    nms_module.nms_gpu = object()
    # Deliberately omit nms_normal_gpu to emulate an incompatible partial install.
    parent_names = ("mmdet3d", "mmdet3d.ops", "mmdet3d.ops.iou3d", "mmdet3d.core",
                    "mmdet3d.core.post_processing")
    modules = {name: types.ModuleType(name) for name in parent_names}
    modules.update({iou_module.__name__: iou_module, nms_module.__name__: nms_module})
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(PatchError, match="nms_normal_gpu"):
        bevfusion_iou3d_patcher.apply(backend="reference")
    assert iou_module.boxes_iou_bev is original_iou
    assert iou_module.nms_gpu is original_nms
    assert not bevfusion_iou3d_patcher._originals


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
class TestNativeMUSA:
    @staticmethod
    def _random_boxes(seed, count):
        generator = torch.Generator().manual_seed(seed)
        starts = torch.rand((count, 2), generator=generator) * 80.0 - 40.0
        sizes = torch.rand((count, 2), generator=generator) * 5.0 + 0.01
        angles = torch.rand((count, 1), generator=generator) * (2.0 * math.pi) - math.pi
        return torch.cat((starts, starts + sizes, angles), dim=1).float()

    def test_iou_matches_reference(self):
        boxes_a = torch.tensor(
            [[0.0, 0.0, 4.0, 4.0, 0.2], [3.0, 1.0, 5.0, 5.0, -0.4]], device="musa"
        )
        boxes_b = torch.tensor(
            [[1.0, 1.0, 5.0, 4.0, -0.1], [20.0, 20.0, 22.0, 22.0, 0.0]], device="musa"
        )
        expected = boxes_iou_bev(boxes_a.cpu(), boxes_b.cpu(), backend="reference")
        actual = boxes_iou_bev(boxes_a, boxes_b, backend="musa")
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)

    def test_iou_empty_inputs_return_empty_matrix(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]], device="musa")
        empty = torch.empty((0, 5), dtype=torch.float32, device="musa")
        assert empty.is_contiguous()
        assert boxes_iou_bev(empty, boxes, backend="musa").shape == (0, 1)
        assert boxes_iou_bev(boxes, empty, backend="musa").shape == (1, 0)

    def test_nms_matches_reference_and_supports_more_than_one_mask_block(self):
        boxes = torch.tensor(
            [[float(i) * 0.2, 0.0, float(i) * 0.2 + 1.0, 1.0, 0.1] for i in range(70)],
            device="musa",
        )
        scores = torch.arange(70, dtype=torch.float32, device="musa")
        expected = nms_bev(boxes.cpu(), scores.cpu(), 0.1, backend="reference")
        actual = nms_bev(boxes, scores, 0.1, backend="musa")
        torch.testing.assert_close(actual.cpu(), expected)

    def test_normal_nms_matches_reference_for_empty_and_multiple_mask_blocks(self):
        empty_boxes = torch.empty((0, 5), dtype=torch.float32, device="musa")
        empty_scores = torch.empty(0, dtype=torch.float32, device="musa")
        assert nms_bev_normal(empty_boxes, empty_scores, 0.5, backend="musa").numel() == 0
        boxes = torch.tensor(
            [[float(i) * 0.2, 0.0, float(i) * 0.2 + 1.0, 1.0, 0.4] for i in range(70)],
            device="musa",
        )
        scores = torch.arange(70, dtype=torch.float32, device="musa")
        expected = nms_bev_normal(boxes.cpu(), scores.cpu(), 0.1, backend="reference")
        actual = nms_bev_normal(boxes, scores, 0.1, backend="musa")
        torch.testing.assert_close(actual.cpu(), expected)

    @pytest.mark.parametrize("seed", [3, 19, 71])
    def test_random_iou_and_nms_match_reference_across_mask_boundaries(self, seed):
        boxes_a = self._random_boxes(seed, 129)
        boxes_b = self._random_boxes(seed + 1000, 67)
        scores = torch.rand(129, generator=torch.Generator().manual_seed(seed + 2000))
        actual_iou = boxes_iou_bev(boxes_a.musa(), boxes_b.musa(), backend="musa").cpu()
        expected_iou = boxes_iou_bev(boxes_a, boxes_b, backend="reference")
        torch.testing.assert_close(actual_iou, expected_iou, rtol=2e-4, atol=2e-5)
        for threshold in (0.0, 0.1, 0.5, 0.9):
            for nms in (nms_bev, nms_bev_normal):
                expected = nms(boxes_a, scores, threshold, backend="reference")
                actual = nms(boxes_a.musa(), scores.musa(), threshold, backend="musa").cpu()
                torch.testing.assert_close(actual, expected)

    def test_extreme_angles_and_near_tangent_boxes_match_reference(self):
        boxes = torch.tensor(
            [
                [0.0, 0.0, 4.0, 2.0, -math.pi + 1e-6],
                [0.0, 0.0, 4.0, 2.0, math.pi - 1e-6],
                [3.99999, 0.0, 7.99999, 2.0, math.pi / 2.0 - 1e-6],
                [4.0, 0.0, 8.0, 2.0, -math.pi / 2.0 + 1e-6],
                [0.0, 0.0, 1e-4, 2.0, 0.3],
            ],
            dtype=torch.float32,
        )
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5], dtype=torch.float32)
        expected_iou = boxes_iou_bev(boxes, boxes, backend="reference")
        actual_iou = boxes_iou_bev(boxes.musa(), boxes.musa(), backend="musa").cpu()
        # Near-zero positive-area boxes amplify FP32 polygon reconstruction error.
        torch.testing.assert_close(actual_iou, expected_iou, rtol=5e-4, atol=2e-5)
        for nms in (nms_bev, nms_bev_normal):
            expected = nms(boxes, scores, 0.5, backend="reference")
            actual = nms(boxes.musa(), scores.musa(), 0.5, backend="musa").cpu()
            torch.testing.assert_close(actual, expected)
