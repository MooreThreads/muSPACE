"""Tests for the Orion flash-attention shim and vendored-mmcv rebinding."""

import sys
import types

import pytest
import torch

from mu_space.patcher.orion_flash_attn import (
    flash_attn_varlen_kvpacked_func,
    install_flash_attn_shim,
)
from mu_space.patcher.orion_patcher import OrionPatcher


def _reference_attention(q, kv):
    """Reference for one packed sequence: plain softmax attention in float64."""
    # q: [T, H, D]; kv: [S, 2, H, D]
    k, v = kv[:, 0], kv[:, 1]
    k = k.transpose(0, 1)  # [H, S, D]
    v = v.transpose(0, 1)  # [H, S, D]
    query = q.transpose(0, 1)  # [H, T, D]
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query.double(), k.double().transpose(1, 2)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v.double()).transpose(0, 1).float()


def test_shim_install_registers_modules():
    assert install_flash_attn_shim(force=True)
    assert "flash_attn.flash_attn_interface" in sys.modules
    assert "flash_attn.bert_padding" in sys.modules


def test_varlen_kvpacked_matches_reference_single_sequence():
    torch.manual_seed(0)
    total_q, total_k, nheads, head_dim = 4, 4, 2, 8
    q = torch.randn(total_q, nheads, head_dim)
    kv = torch.randn(total_k, 2, nheads, head_dim)
    cu = torch.tensor([0, total_q], dtype=torch.int32)
    actual = flash_attn_varlen_kvpacked_func(q, kv, cu, cu, total_q, total_k)
    expected = _reference_attention(q, kv)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_varlen_kvpacked_handles_ragged_batches():
    torch.manual_seed(1)
    lengths_q = [3, 1]
    lengths_k = [3, 5]
    nheads, head_dim = 2, 4
    q = torch.randn(sum(lengths_q), nheads, head_dim)
    kv = torch.randn(sum(lengths_k), 2, nheads, head_dim)
    cu_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths_q), 0)), dtype=torch.int32)
    cu_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths_k), 0)), dtype=torch.int32)
    actual = flash_attn_varlen_kvpacked_func(
        q, kv, cu_q, cu_k, max(lengths_q), max(lengths_k)
    )
    assert actual.shape == q.shape

    offset_q = offset_k = 0
    for length_q, length_k in zip(lengths_q, lengths_k):
        q_slice = q[offset_q : offset_q + length_q]
        kv_slice = kv[offset_k : offset_k + length_k]
        expected = _reference_attention(q_slice, kv_slice)
        actual_slice = actual[offset_q : offset_q + length_q]
        torch.testing.assert_close(actual_slice, expected, atol=1e-5, rtol=1e-5)
        offset_q += length_q
        offset_k += length_k


def test_varlen_kvpacked_ignores_padding_beyond_sequence():
    """A padded tail must not leak into the result of a shorter sequence."""
    torch.manual_seed(2)
    nheads, head_dim = 2, 4
    q = torch.randn(2, nheads, head_dim)
    kv = torch.randn(5, 2, nheads, head_dim)
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 3], dtype=torch.int32)
    actual = flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, 2, 5)
    expected = _reference_attention(q, kv[:3])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _fake_mmcv_modules(monkeypatch):
    """Register every module the patcher touches with sentinel attributes."""
    created = {}
    spec = {
        "mmcv.ops.iou3d_det.iou3d_utils": ("boxes_iou_bev", "nms_gpu", "nms_normal_gpu"),
        "mmcv.ops.iou3d_det": ("boxes_iou_bev", "nms_gpu", "nms_normal_gpu"),
        "mmcv.core.post_processing.box3d_nms": ("nms_gpu", "nms_normal_gpu"),
        "mmcv.core.post_processing.merge_augs": ("nms_gpu", "nms_normal_gpu"),
        "mmcv.ops.roiaware_pool3d.points_in_boxes": (
            "points_in_boxes_gpu",
            "points_in_boxes_batch",
            "points_in_boxes_cpu",
        ),
        "mmcv.ops.roiaware_pool3d": (
            "points_in_boxes_gpu",
            "points_in_boxes_batch",
            "points_in_boxes_cpu",
        ),
        "mmcv.ops.focal_loss": ("sigmoid_focal_loss",),
    }
    for name, attrs in spec.items():
        module = types.ModuleType(name)
        module.__path__ = []
        for attr in attrs:
            setattr(module, attr, lambda *a, _name=f"{name}.{attr}", **k: f"original-{_name}")
        monkeypatch.setitem(sys.modules, name, module)
        created[name] = module
    return created


def test_patcher_rebinds_and_restores(monkeypatch):
    modules = _fake_mmcv_modules(monkeypatch)
    iou_module = modules["mmcv.ops.iou3d_det.iou3d_utils"]
    points_module = modules["mmcv.ops.roiaware_pool3d.points_in_boxes"]
    original_nms = iou_module.nms_gpu

    patcher = OrionPatcher()
    report = patcher.apply(backend="musa")
    assert report.action == "applied"
    assert len(report.patched_modules) == 17
    assert iou_module.nms_gpu is not original_nms
    assert iou_module.nms_gpu.__name__ == "nms_gpu"
    assert points_module.points_in_boxes_gpu.__name__ == "points_in_boxes_gpu"

    patcher.restore()
    assert iou_module.nms_gpu is original_nms


def test_patcher_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        OrionPatcher().apply(backend="cuda")


def test_patcher_reports_missing_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "mmcv.ops.iou3d_det.iou3d_utils", None)
    from mu_space.patcher import PatchError

    with pytest.raises(PatchError, match="unavailable"):
        OrionPatcher().apply(backend="musa")
