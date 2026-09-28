import pytest
import torch

from mu_space.ops.hard_voxelize import (
    Voxelization,
    dynamic_voxelize,
    hard_voxelize,
)


def _points():
    return torch.tensor(
        [[0.1, 0.1, 0.1, 1.0], [0.2, 0.1, 0.1, 2.0], [1.1, 0.1, 0.1, 3.0],
         [-0.1, 0.1, 0.1, 4.0], [0.1, 1.1, 0.1, 5.0]], dtype=torch.float32
    )


def test_hard_reference_preserves_first_seen_order_and_truncates_points():
    voxels, coors, counts = hard_voxelize(_points(), (1, 1, 1), (0, 0, 0, 2, 2, 2),
                                          max_points=1, max_voxels=2, backend="reference")
    assert coors.tolist() == [[0, 0, 0], [0, 0, 1]]
    assert counts.tolist() == [1, 1]
    assert voxels[:, 0, 3].tolist() == [1.0, 3.0]


def test_dynamic_reference_marks_each_invalid_axis_like_cuda():
    result = dynamic_voxelize(_points(), (1, 1, 1), (0, 0, 0, 2, 2, 2), backend="reference")
    assert result.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 1, 0]]


def test_empty_and_invalid_parameters():
    empty = torch.empty((0, 4), dtype=torch.float32)
    result = hard_voxelize(empty, (1, 1, 1), (0, 0, 0, 2, 2, 2), backend="reference")
    assert all(t.shape[0] == 0 for t in result)
    with pytest.raises(ValueError):
        hard_voxelize(_points(), (0, 1, 1), (0, 0, 0, 2, 2, 2), backend="reference")


def test_max_voxels_and_max_points_are_truncated():
    points = torch.tensor(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]],
        dtype=torch.float32,
    )
    voxels, coors, counts = hard_voxelize(
        points, (1, 1, 1), (0, 0, 0, 2, 2, 2), max_points=1, max_voxels=2, backend="reference"
    )
    assert voxels.shape[0] == coors.shape[0] == counts.shape[0] == 2
    assert counts.tolist() == [1, 1]


def test_voxelization_module_selects_eval_limit():
    module = Voxelization((1, 1, 1), (0, 0, 0, 2, 2, 2), 4, (1, 2))
    module.eval()
    voxels, coors, counts = module(_points())
    assert voxels.shape[0] <= 2 and coors.shape[0] == counts.shape[0]


def test_voxelization_module_selects_training_limit_and_dynamic_mode():
    module = Voxelization((1, 1, 1), (0, 0, 0, 2, 2, 2), 4, (1, 2))
    module.train()
    assert module(_points())[0].shape[0] == 1
    dynamic = Voxelization((1, 1, 1), (0, 0, 0, 2, 2, 2), -1)
    assert dynamic(_points()).shape == (5, 3)


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="requires MUSA")
def test_musa_matches_reference():
    points = _points().to("musa")
    expected = hard_voxelize(_points(), (1, 1, 1), (0, 0, 0, 2, 2, 2), 4, 8, backend="reference")
    actual = hard_voxelize(points, (1, 1, 1), (0, 0, 0, 2, 2, 2), 4, 8, backend="musa")
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="requires MUSA")
def test_musa_dynamic_and_parallel_nondeterministic_semantics():
    points = _points().to("musa")
    expected_dynamic = dynamic_voxelize(_points(), (1, 1, 1), (0, 0, 0, 2, 2, 2), backend="reference")
    actual_dynamic = dynamic_voxelize(points, (1, 1, 1), (0, 0, 0, 2, 2, 2), backend="musa")
    torch.testing.assert_close(actual_dynamic.cpu(), expected_dynamic, rtol=0, atol=0)
    voxels, coors, counts = hard_voxelize(
        points, (1, 1, 1), (0, 0, 0, 2, 2, 2), 4, 8, deterministic=False, backend="musa"
    )
    assert torch.unique(coors, dim=0).shape[0] == coors.shape[0]
    assert int(counts.sum()) == 4
    assert voxels.shape[0] == coors.shape[0] == counts.shape[0]


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="requires MUSA")
def test_musa_hard_voxelize_ignores_nan_points_and_dynamic_empty_is_safe():
    points = torch.tensor([[0.5, 0.5, 0.5], [float("nan"), float("nan"), float("nan")]])
    expected = hard_voxelize(points, (1, 1, 1), (0, 0, 0, 4, 4, 4), 2, 10, backend="reference")
    actual = hard_voxelize(points.musa(), (1, 1, 1), (0, 0, 0, 4, 4, 4), 2, 10, backend="musa")
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0, equal_nan=True)
    empty = dynamic_voxelize(torch.empty((0, 3), device="musa"), (1, 1, 1), (0, 0, 0, 4, 4, 4), backend="musa")
    assert empty.shape == (0, 3)
    assert empty.dtype == torch.int32
    hard_empty = hard_voxelize(
        torch.empty((0, 3), device="musa"), (1, 1, 1), (0, 0, 0, 4, 4, 4), 2, 10, backend="musa"
    )
    assert [tensor.shape for tensor in hard_empty] == [(0, 2, 3), (0, 3), (0,)]


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="requires MUSA")
def test_musa_parallel_respects_voxel_cap_under_concurrent_new_keys():
    points = torch.zeros((4096, 4), dtype=torch.float32)
    points[:, 0] = torch.arange(4096, dtype=torch.float32) + 0.1
    points[:, 1] = 0.1
    points[:, 2] = 0.1
    points = points.to("musa")
    voxels, coors, counts = hard_voxelize(
        points, (1, 1, 1), (0, 0, 0, 4096, 1, 1), 4, 7,
        deterministic=False, backend="musa"
    )
    assert voxels.shape[0] == coors.shape[0] == counts.shape[0] == 7
    assert int(counts.max()) <= 4
    assert int(counts.sum()) <= 28
    assert torch.unique(coors, dim=0).shape[0] == 7


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="requires MUSA")
def test_musa_parallel_clamps_high_collision_counts():
    points = torch.zeros((4096, 4), dtype=torch.float32)
    points[:, 0] = 0.1
    points[:, 1] = 0.1
    points[:, 2] = 0.1
    points[:, 3] = torch.arange(4096, dtype=torch.float32)
    voxels, coors, counts = hard_voxelize(
        points.to("musa"), (1, 1, 1), (0, 0, 0, 2, 2, 2), 4, 8,
        deterministic=False, backend="musa"
    )
    assert voxels.shape[0] == coors.shape[0] == counts.shape[0] == 1
    assert counts.tolist() == [4]
