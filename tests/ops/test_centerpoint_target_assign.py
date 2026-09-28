import math

import pytest
import torch

from mu_space import (
    centerpoint_target_assign,
    centerpoint_target_assign_reference,
    clear_dispatch_report,
    get_last_dispatch_report,
)


_KWARGS = dict(
    num_classes=2,
    out_size_factor=8,
    gaussian_overlap=0.1,
    min_radius=2,
    voxel_size=(0.1, 0.1),
    pc_range=(-51.2, -51.2),
    feature_map_size=(128, 128),
    max_objs=5,
)


def _boxes(device="cpu"):
    # valid, same-class collision, invalid width, just outside, and a second class
    return torch.tensor(
        [
            [-50.0, -50.0, 1.5, 1.8, 4.2, 1.6, 2.0, -1.0, 0.2],
            [-49.9, -50.1, 0.3, 2.0, 3.8, 1.5, 1.0, 0.5, -0.3],
            [-48.0, -48.0, 0.2, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [51.3, 0.0, 0.1, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 1.5, 2.5, 1.2, -2.0, 1.0, math.pi / 2],
        ],
        dtype=torch.float32,
        device=device,
    )


def _classes(device="cpu"):
    return torch.tensor([1, 1, 1, 2, 2], dtype=torch.int64, device=device)


def test_reference_matches_known_centerpoint_encoding():
    heatmap, anno_box, inds, mask = centerpoint_target_assign_reference(
        _boxes(), _classes(), **_KWARGS
    )
    assert heatmap.shape == (2, 128, 128)
    assert anno_box.shape == (5, 10)
    assert inds.dtype == torch.int64 and mask.dtype == torch.uint8
    assert mask.tolist() == [1, 1, 0, 0, 1]
    assert inds[0].item() == 1 * 128 + 1
    assert inds[3].item() == 0
    torch.testing.assert_close(anno_box[0, 3:6], _boxes()[0, 3:6].log())
    torch.testing.assert_close(anno_box[4, 6:8], torch.tensor([1.0, 0.0]), atol=1e-6, rtol=0)
    assert heatmap[0, 1, 1] == 1
    assert heatmap[1].max() == 1


def test_reference_uses_upstream_truncation_and_skips_feature_map_outside():
    boxes = torch.tensor(
        [
            [-51.21, -51.21, 0, 2, 2, 1, 0, 0, 0],  # truncates to (0, 0), like astype(int32)
            [51.2, 0.0, 0, 2, 2, 1, 0, 0, 0],  # exactly beyond W=128, skipped
        ],
        dtype=torch.float32,
    )
    _, _, inds, mask = centerpoint_target_assign_reference(
        boxes, torch.ones(2, dtype=torch.int64), **_KWARGS
    )
    assert mask[:2].tolist() == [1, 0]
    assert inds[0].item() == 0


def test_max_objs_and_empty_input():
    boxes, classes = _boxes(), _classes()
    _, _, _, mask = centerpoint_target_assign_reference(
        boxes, classes, **{**_KWARGS, "max_objs": 2}
    )
    assert mask.tolist() == [1, 1]
    outputs = centerpoint_target_assign_reference(
        torch.empty((0, 9), dtype=torch.float32), torch.empty(0, dtype=torch.int64), **_KWARGS
    )
    assert [tensor.numel() for tensor in outputs] == [2 * 128 * 128, 50, 5, 5]
    assert all(not bool(tensor.any()) for tensor in outputs)


def test_cpu_auto_uses_reference_and_validates_contract():
    clear_dispatch_report()
    actual = centerpoint_target_assign(_boxes(), _classes(), **_KWARGS)
    expected = centerpoint_target_assign_reference(_boxes(), _classes(), **_KWARGS)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
    report = get_last_dispatch_report()
    assert report is not None and report.selected_backend == "reference"
    with pytest.raises(ValueError, match="class_ids"):
        centerpoint_target_assign(_boxes(), torch.zeros(5, dtype=torch.int64), **_KWARGS)
    with pytest.raises(TypeError, match="int64"):
        centerpoint_target_assign(_boxes(), _classes().int(), **_KWARGS)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_musa_matches_reference_for_centerpoint_cases():
    expected = centerpoint_target_assign_reference(_boxes(), _classes(), **_KWARGS)
    actual = centerpoint_target_assign(_boxes("musa"), _classes("musa"), backend="musa", **_KWARGS)
    torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual[2].cpu(), expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3].cpu(), expected[3], rtol=0, atol=0)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_musa_n50_two_class_fixture_matches_reference():
    generator = torch.Generator().manual_seed(2333)
    boxes = torch.empty((50, 9), dtype=torch.float32).uniform_(-45, 45, generator=generator)
    boxes[:, 3:6] = torch.empty((50, 3)).uniform_(0.2, 8.0, generator=generator)
    classes = torch.randint(1, 3, (50,), generator=generator, dtype=torch.int64)
    expected = centerpoint_target_assign_reference(boxes, classes, **_KWARGS)
    actual = centerpoint_target_assign(boxes.musa(), classes.musa(), backend="musa", **_KWARGS)
    torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[2].cpu(), expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3].cpu(), expected[3], rtol=0, atol=0)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_musa_empty_ground_truth_returns_zero_outputs():
    boxes = torch.empty((0, 9), dtype=torch.float32, device="musa")
    classes = torch.empty((0,), dtype=torch.int64, device="musa")
    actual = centerpoint_target_assign(boxes, classes, backend="musa", **_KWARGS)
    assert [tensor.numel() for tensor in actual] == [2 * 128 * 128, 50, 5, 5]
    assert all(not bool(tensor.any().cpu()) for tensor in actual)
