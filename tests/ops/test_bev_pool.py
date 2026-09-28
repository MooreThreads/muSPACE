import sys
import types

import pytest
import torch

from mu_space import bev_pool, clear_dispatch_report, get_last_dispatch_report
from mu_space.patcher import BEVFusionPatcher


def _fixture(device="cpu"):
    feats = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        device=device,
        requires_grad=True,
    )
    # Upstream BEVFusion order is [x, y, z, batch].
    coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 1], [0, 1, 1, 1]], device=device)
    return feats, coords, 2, 2, 2, 2


def test_reference_forward_backward_and_dispatch():
    clear_dispatch_report()
    feats, coords, *shape = _fixture()
    output = bev_pool(feats, coords, *shape, backend="reference")
    assert output.shape == (2, 2, 2, 2, 2)
    torch.testing.assert_close(output[0, :, 0, 0, 0], torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(output[1, :, 0, 1, 1], torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(output[1, :, 1, 0, 1], torch.tensor([7.0, 8.0]))
    output.sum().backward()
    torch.testing.assert_close(feats.grad, torch.ones_like(feats))
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "bev_pool"
    assert report.selected_backend == "reference"


def test_reference_empty_and_int64_coords():
    feats = torch.empty((0, 3), dtype=torch.float32, requires_grad=True)
    coords = torch.empty((0, 4), dtype=torch.int64)
    output = bev_pool(feats, coords, 1, 2, 3, 4, backend="reference")
    assert output.shape == (1, 3, 2, 3, 4)
    output.sum().backward()
    assert feats.grad.shape == feats.shape


@pytest.mark.parametrize(
    ("coords", "message"),
    [
        (torch.zeros((1, 3), dtype=torch.int32), "coords must have shape"),
        (torch.tensor([[1, 0, 0, 1]], dtype=torch.int32), "coords contains an index"),
    ],
)
def test_invalid_coordinates_fail_explicitly(coords, message):
    with pytest.raises(ValueError, match=message):
        bev_pool(torch.ones((1, 2)), coords, 1, 1, 1, 1, backend="reference")


@pytest.mark.parametrize("value", [2**32, -(2**32), 2**32 + 1])
def test_int64_coordinates_are_validated_before_int32_cast(value):
    coords = torch.tensor([[value, 0, 0, 0]], dtype=torch.int64)
    with pytest.raises(ValueError, match="coords contains an index"):
        bev_pool(torch.ones((1, 2)), coords, 1, 1, 2, 2, backend="reference")


def test_bevfusion_patcher_replaces_helper_and_imported_binding(monkeypatch):
    target_name = "mmdet3d.ops.bev_pool.bev_pool"
    imported_name = "mmdet3d.models.vtransforms.base"
    target = types.ModuleType(target_name)
    imported = types.ModuleType(imported_name)

    def original(*args, **kwargs):
        return "original"

    target.bev_pool = original
    imported.bev_pool = original
    monkeypatch.setitem(sys.modules, target_name, target)
    monkeypatch.setitem(sys.modules, imported_name, imported)
    patcher = BEVFusionPatcher()
    report = patcher.apply()
    assert report.action == "applied"
    assert report.patched_modules == (target_name, imported_name)
    assert target.bev_pool is imported.bev_pool
    assert target.bev_pool is not original
    restored = patcher.restore()
    assert restored.action == "restored"
    assert target.bev_pool is original
    assert imported.bev_pool is original


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_musa_matches_reference_forward_backward():
    native = _fixture(device="musa")
    feats, coords, *shape = native
    ref_feats = feats.detach().cpu().clone().requires_grad_()
    reference = bev_pool(ref_feats, coords.cpu(), *shape, backend="reference")
    actual = bev_pool(feats, coords, *shape, backend="musa")
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-5, atol=1e-5)
    grad = torch.arange(actual.numel(), device="musa", dtype=torch.float32).reshape_as(actual)
    actual.backward(grad)
    reference.backward(grad.cpu())
    torch.testing.assert_close(feats.grad.cpu(), ref_feats.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_musa_strided_feats_match_reference_forward_backward():
    base = torch.tensor([[1.0, -1.0, 2.0, -1.0], [3.0, -1.0, 4.0, -1.0]], device="musa")
    feats = base[:, ::2].detach().requires_grad_()
    assert not feats.is_contiguous()
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32, device="musa")
    expected_feats = feats.detach().cpu().clone().requires_grad_()
    expected = bev_pool(expected_feats, coords.cpu(), 1, 1, 1, 2, backend="reference")
    actual = bev_pool(feats, coords, 1, 1, 1, 2, backend="musa")
    torch.testing.assert_close(actual.cpu(), expected)
    grad = torch.arange(actual.numel(), dtype=torch.float32, device="musa").reshape_as(actual)
    actual.backward(grad)
    expected.backward(grad.cpu())
    torch.testing.assert_close(feats.grad.cpu(), expected_feats.grad)
