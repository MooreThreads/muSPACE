import sys
import types

import pytest
import torch

from mu_space import bev_pool_v2, clear_dispatch_report, get_last_dispatch_report
from mu_space.patcher import FlashOCCPatcher


def _fixture(device="cpu"):
    depth = torch.tensor(
        [[[[[0.2, 0.5], [0.7, 0.9]], [[0.1, 0.3], [0.4, 0.8]]]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    feat = torch.tensor(
        [[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    ranks_depth = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device=device)
    ranks_feat = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], dtype=torch.int32, device=device)
    ranks_bev = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32, device=device)
    interval_starts = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=device)
    interval_lengths = torch.tensor([2, 2, 2, 2], dtype=torch.int32, device=device)
    bev_feat_shape = (1, 1, 2, 2, 2)
    return (
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        bev_feat_shape,
        interval_starts,
        interval_lengths,
    )


def _channel_fixture(channels, device="cpu"):
    batch, cameras, depth_bins, height, width = 1, 1, 2, 3, 4
    depth = (
        torch.linspace(0.1, 2.4, batch * cameras * depth_bins * height * width)
        .view(batch, cameras, depth_bins, height, width)
        .to(device)
        .requires_grad_()
    )
    feat = (
        torch.linspace(
            0.01,
            0.01 * batch * cameras * height * width * channels,
            batch * cameras * height * width * channels,
        )
        .view(batch, cameras, height, width, channels)
        .to(device)
        .requires_grad_()
    )
    ranks_depth = torch.arange(depth.numel(), dtype=torch.int32, device=device)
    ranks_feat = (ranks_depth % (height * width)).to(torch.int32)
    ranks_bev = (ranks_depth * 7 % 16).to(torch.int32)
    order = ranks_bev.argsort()
    ranks_depth = ranks_depth[order].contiguous()
    ranks_feat = ranks_feat[order].contiguous()
    ranks_bev = ranks_bev[order].contiguous()
    kept = torch.ones(ranks_bev.shape[0], dtype=torch.bool, device=device)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].to(torch.int32).contiguous()
    interval_lengths = torch.empty_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.numel() - interval_starts[-1]
    bev_feat_shape = (batch, 1, 4, 4, channels)
    return (
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        bev_feat_shape,
        interval_starts,
        interval_lengths,
    )


def test_reference_forward_backward_on_cpu():
    clear_dispatch_report()
    tensors = _fixture()
    output = bev_pool_v2(*tensors, backend="reference")
    assert output.shape == (1, 2, 1, 2, 2)
    expected = torch.tensor(
        [[[[[1.7, 3.4], [2.6, 7.6]]], [[[2.4, 5.0], [3.0, 8.8]]]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    depth, feat, *_ = tensors
    assert torch.isfinite(depth.grad).all()
    assert torch.isfinite(feat.grad).all()
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "bev_pool_v2"
    assert report.selected_backend == "reference"


def test_auto_cpu_uses_reference():
    clear_dispatch_report()
    output = bev_pool_v2(*_fixture(), backend="auto")
    assert output.shape == (1, 2, 1, 2, 2)
    report = get_last_dispatch_report()
    assert report is not None
    assert report.requested_backend == "auto"
    assert report.selected_backend == "reference"


def test_invalid_dtypes_fail_explicitly():
    tensors = list(_fixture())
    tensors[2] = tensors[2].to(torch.int64)
    with pytest.raises(TypeError, match="ranks_depth must use int32"):
        bev_pool_v2(*tensors, backend="reference")


def test_reference_empty_ranks_returns_zero_and_zero_gradients():
    depth, feat, *_ = _fixture()
    empty = torch.empty(0, dtype=torch.int32)
    output = bev_pool_v2(
        depth,
        feat,
        empty,
        empty,
        empty,
        (1, 1, 2, 2, 2),
        empty,
        empty,
        backend="reference",
    )
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    torch.testing.assert_close(depth.grad, torch.zeros_like(depth.grad))
    torch.testing.assert_close(feat.grad, torch.zeros_like(feat.grad))


def test_bev_feature_shape_requires_all_five_dimensions():
    tensors = _fixture()
    with pytest.raises(ValueError, match="bev_feat_shape must contain"):
        bev_pool_v2(*tensors[:5], (1, 2, 2, 2), *tensors[6:], backend="reference")


def test_flashocc_patcher_replaces_helper_and_imported_binding(monkeypatch):
    target_name = "projects.mmdet3d_plugin.ops.bev_pool_v2.bev_pool"
    imported_name = "projects.mmdet3d_plugin.models.necks.view_transformer"
    target = types.ModuleType(target_name)
    imported = types.ModuleType(imported_name)

    def original(*args, **kwargs):
        return "original"

    target.bev_pool_v2 = original
    imported.bev_pool_v2 = original
    monkeypatch.setitem(sys.modules, target_name, target)
    monkeypatch.setitem(sys.modules, imported_name, imported)
    patcher = FlashOCCPatcher()
    report = patcher.apply()

    assert report.action == "applied"
    assert report.patched_modules == (target_name, imported_name)
    assert target.bev_pool_v2 is imported.bev_pool_v2
    assert target.bev_pool_v2 is not original

    restored = patcher.restore()
    assert restored.action == "restored"
    assert target.bev_pool_v2 is original
    assert imported.bev_pool_v2 is original


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_musa_matches_reference_forward_backward():
    tensors = _fixture(device="musa")
    ref_tensors = tuple(
        tensor.detach().cpu().clone().requires_grad_(tensor.requires_grad)
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
        else tensor.detach().cpu().clone()
        if isinstance(tensor, torch.Tensor)
        else tensor
        for tensor in tensors
    )
    reference = bev_pool_v2(*ref_tensors, backend="reference")
    actual = bev_pool_v2(*tensors, backend="musa")
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)
    actual.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(tensors[0].grad.cpu(), ref_tensors[0].grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tensors[1].grad.cpu(), ref_tensors[1].grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_empty_ranks_returns_zero_and_zero_gradients():
    depth, feat, *_ = _fixture(device="musa")
    empty = torch.empty(0, dtype=torch.int32, device="musa")
    assert empty.is_contiguous()
    output = bev_pool_v2(
        depth, feat, empty, empty, empty, (1, 1, 2, 2, 2), empty, empty, backend="musa"
    )
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    torch.testing.assert_close(depth.grad, torch.zeros_like(depth.grad))
    torch.testing.assert_close(feat.grad, torch.zeros_like(feat.grad))


@pytest.mark.parametrize("channels", [32, 64])
@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_musa_channel_paths_match_reference(channels):
    tensors = _channel_fixture(channels, device="musa")
    ref_tensors = tuple(
        tensor.detach().cpu().clone().requires_grad_(tensor.requires_grad)
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
        else tensor.detach().cpu().clone()
        if isinstance(tensor, torch.Tensor)
        else tensor
        for tensor in tensors
    )

    reference = bev_pool_v2(*ref_tensors, backend="reference")
    actual = bev_pool_v2(*tensors, backend="musa")
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)

    actual.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(tensors[0].grad.cpu(), ref_tensors[0].grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tensors[1].grad.cpu(), ref_tensors[1].grad, rtol=1e-4, atol=1e-4)

    report = get_last_dispatch_report()
    assert report is not None
    assert report.selected_backend == "musa"
