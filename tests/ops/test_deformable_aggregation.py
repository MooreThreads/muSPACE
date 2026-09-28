"""Deformable aggregation tests on SparseDrive production shapes.

Shapes (from dump):
  mc_ms_feat         [8, 89760, 256]  float32
  spatial_shape      [6, 4, 2]        int32
  scale_start_index  [6, 4]           int32
  sampling_location  [8, 900, 13, 6, 2] float32
  weights            [8, 900, 13, 6, 4, 8] float32
  output             [8, 900, 256]    float32

Native paths: `*_opt` (production) and `*_base` (baseline comparison).
"""

from __future__ import annotations

import sys
import time
import types

import pytest
import torch

from mu_space.ops.deformable_aggregation import (
    deformable_aggregation,
    deformable_aggregation_reference,
    feature_maps_format,
)
from mu_space.patcher import SparseDrivePatcher

B, NUM_FEAT, C = 8, 89760, 256
NUM_CAMS, NUM_SCALE, NUM_ANCHORS, NUM_PTS, NUM_GROUPS = 6, 4, 900, 13, 8
ACTIVE_FRAC = 0.40  # dump itself is ~18.6%; 40% stresses closer to ref load

SPATIAL_SHAPE = [[[64, 176], [32, 88], [16, 44], [8, 22]]] * NUM_CAMS
SCALE_START_INDEX = [
    [0, 11264, 14080, 14784],
    [14960, 26224, 29040, 29744],
    [29920, 41184, 44000, 44704],
    [44880, 56144, 58960, 59664],
    [59840, 71104, 73920, 74624],
    [74800, 86064, 88880, 89584],
]

FORWARD_OPT_BUDGET_MS = 25.0
BACKWARD_OPT_BUDGET_MS = 90.0

_MUSA = hasattr(torch, "musa") and torch.musa.is_available()
requires_musa = pytest.mark.skipif(not _MUSA, reason="MUSA required")


def _sync() -> None:
    torch.musa.synchronize()


def _inputs(device: str = "musa", seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    spatial_shape = torch.tensor(SPATIAL_SHAPE, dtype=torch.int32, device=device)
    scale_start_index = torch.tensor(SCALE_START_INDEX, dtype=torch.int32, device=device)
    mc_ms_feat = torch.randn(B, NUM_FEAT, C, generator=g, dtype=torch.float32).to(device)

    point_shape = (B, NUM_ANCHORS, NUM_PTS, NUM_CAMS)
    active = torch.rand(*point_shape, generator=g) < ACTIVE_FRAC
    outside = torch.randn(*point_shape, 2, generator=g) * 10.0
    inside = torch.rand(*point_shape, 2, generator=g) * 0.98 + 0.01
    sampling_location = torch.where(active.unsqueeze(-1), inside, outside).to(device)

    weights = (
        torch.rand(B, NUM_ANCHORS, NUM_PTS, NUM_CAMS, NUM_SCALE, NUM_GROUPS, generator=g) * 0.004
    ).to(device=device, dtype=torch.float32)
    grad_output = torch.randn(B, NUM_ANCHORS, C, device=device, dtype=torch.float32)
    return mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, grad_output


def _timed_ms(fn, *, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0 / iters


def _small_inputs(device: str = "cpu"):
    mc_ms_feat = torch.arange(20, dtype=torch.float32, device=device).view(1, 5, 4)
    mc_ms_feat.requires_grad_()
    spatial_shape = torch.tensor([[[2, 2], [1, 1]]], dtype=torch.int32, device=device)
    scale_start_index = torch.tensor([[0, 4]], dtype=torch.int32, device=device)
    sampling_location = torch.tensor(
        [[[[[0.5, 0.5]]], [[[1.0, 0.5]]]]], dtype=torch.float32, device=device
    )
    sampling_location.requires_grad_()
    weights = torch.tensor(
        [[[[[[0.25, 0.5], [0.75, 1.0]]]], [[[[0.5, 0.25], [1.0, 0.75]]]]]],
        dtype=torch.float32,
        device=device,
    )
    weights.requires_grad_()
    return mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights


def test_sparsedrive_patcher(monkeypatch):
    ops = types.ModuleType("projects.mmdet3d_plugin.ops")
    blocks = types.ModuleType("projects.mmdet3d_plugin.models.blocks")

    def original(*args, **kwargs):
        return "original"

    ops.deformable_aggregation_function = original
    blocks.DAF = original
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setitem(sys.modules, blocks.__name__, blocks)

    patcher = SparseDrivePatcher()
    assert patcher.apply().action == "applied"
    assert ops.deformable_aggregation_function is not original
    assert blocks.DAF is ops.deformable_aggregation_function
    assert patcher.restore().action == "restored"


def test_reference_handles_boundary_and_invalid_locations():
    tensors = _small_inputs()
    output = deformable_aggregation(*tensors, backend="reference")
    expected = deformable_aggregation_reference(*tensors)
    torch.testing.assert_close(output, expected)
    # The second anchor is exactly on the native open-interval boundary and must
    # be ignored rather than partially sampled.
    torch.testing.assert_close(output[:, 1], torch.zeros_like(output[:, 1]))
    output.square().sum().backward()
    for tensor in (tensors[0], tensors[3], tensors[4]):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_reference_validates_group_and_metadata_dtypes():
    tensors = list(_small_inputs())
    tensors[-1] = torch.ones((1, 2, 1, 1, 2, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="divisible by num_groups"):
        deformable_aggregation(*tensors, backend="reference")
    tensors = list(_small_inputs())
    tensors[1] = tensors[1].to(torch.int64)
    with pytest.raises(TypeError, match="spatial_shape must use int32"):
        deformable_aggregation(*tensors, backend="reference")


def test_feature_maps_format_round_trip_preserves_values():
    levels = [
        torch.arange(1 * 2 * 4 * 2 * 3, dtype=torch.float32).view(1, 2, 4, 2, 3),
        torch.arange(1 * 2 * 4 * 1 * 2, dtype=torch.float32).view(1, 2, 4, 1, 2),
    ]
    formatted = feature_maps_format(levels)
    restored = feature_maps_format(formatted, inverse=True)
    assert len(restored) == 1  # one homogeneous camera group
    for got, expected in zip(restored[0], levels):
        torch.testing.assert_close(got, expected)


@requires_musa
def test_native_small_case_matches_reference_forward_and_backward():
    native = _small_inputs(device="musa")
    reference = tuple(
        tensor.detach().cpu().clone().requires_grad_(tensor.requires_grad)
        if tensor.dtype == torch.float32
        else tensor.detach().cpu().clone()
        for tensor in native
    )
    expected = deformable_aggregation(*reference, backend="reference")
    actual = deformable_aggregation(*native, backend="musa")
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for actual_tensor, expected_tensor in zip(
        (native[0], native[3], native[4]), (reference[0], reference[3], reference[4])
    ):
        torch.testing.assert_close(
            actual_tensor.grad.cpu(), expected_tensor.grad, rtol=1e-3, atol=1e-3
        )


@requires_musa
def test_native_zero_anchors_returns_empty_output_and_zero_gradients():
    mc, spatial, start, loc, weights = _small_inputs(device="musa")
    loc = loc[:, :0].detach().requires_grad_()
    weights = weights[:, :0].detach().requires_grad_()
    assert loc.is_contiguous() and weights.is_contiguous()
    output = deformable_aggregation(mc, spatial, start, loc, weights, backend="musa")
    assert output.shape == (1, 0, 4)
    output.sum().backward()
    for tensor in (mc, loc, weights):
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor.grad))


@requires_musa
def test_real_shape_accuracy():
    """Compare opt path against baseline on production shapes."""
    from mu_space.runtime.native import native_module_proxy

    native = native_module_proxy()

    mc, spatial, start, loc, weights, grad_out = _inputs()

    out_opt = torch.zeros((B, NUM_ANCHORS, C), device=mc.device, dtype=mc.dtype)
    out_base = torch.zeros_like(out_opt)
    native.deformable_aggregation_forward_opt(mc, spatial, start, loc, weights, out_opt)
    native.deformable_aggregation_forward_base(mc, spatial, start, loc, weights, out_base)
    assert out_opt.shape == (B, NUM_ANCHORS, C)
    torch.testing.assert_close(out_opt, out_base, rtol=1e-4, atol=1e-4)

    g_feat_o, g_loc_o, g_w_o = (
        torch.zeros_like(mc),
        torch.zeros_like(loc),
        torch.zeros_like(weights),
    )
    g_feat_b, g_loc_b, g_w_b = (
        torch.zeros_like(mc),
        torch.zeros_like(loc),
        torch.zeros_like(weights),
    )
    native.deformable_aggregation_backward_opt(
        mc, spatial, start, loc, weights, grad_out, g_feat_o, g_loc_o, g_w_o
    )
    native.deformable_aggregation_backward_base(
        mc, spatial, start, loc, weights, grad_out, g_feat_b, g_loc_b, g_w_b
    )
    torch.testing.assert_close(g_feat_o, g_feat_b, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(g_loc_o, g_loc_b, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(g_w_o, g_w_b, rtol=1e-3, atol=1e-3)


@requires_musa
def test_real_shape_performance():
    """forward_opt / backward_opt vs baseline on production shapes (40% active)."""
    from mu_space.runtime.native import native_module_proxy

    native = native_module_proxy()

    mc, spatial, start, loc, weights, grad_out = _inputs(seed=1)

    def fw_opt():
        output = torch.zeros((B, NUM_ANCHORS, C), device=mc.device, dtype=mc.dtype)
        native.deformable_aggregation_forward_opt(mc, spatial, start, loc, weights, output)
        return output

    def fw_base():
        output = torch.zeros((B, NUM_ANCHORS, C), device=mc.device, dtype=mc.dtype)
        native.deformable_aggregation_forward_base(mc, spatial, start, loc, weights, output)
        return output

    def bw(fn):
        gf, gl, gw = torch.zeros_like(mc), torch.zeros_like(loc), torch.zeros_like(weights)
        fn(mc, spatial, start, loc, weights, grad_out, gf, gl, gw)

    fw_base_ms = _timed_ms(fw_base)
    fw_opt_ms = _timed_ms(fw_opt)
    bw_base_ms = _timed_ms(lambda: bw(native.deformable_aggregation_backward_base))
    bw_opt_ms = _timed_ms(lambda: bw(native.deformable_aggregation_backward_opt))

    print(
        f"\n[real shape, active={ACTIVE_FRAC:.0%}] "
        f"fw_base={fw_base_ms:.2f}ms fw_opt={fw_opt_ms:.2f}ms "
        f"bw_base={bw_base_ms:.2f}ms bw_opt={bw_opt_ms:.2f}ms "
        f"(SparseDrive ref fw 33/14, bw 75/16)"
    )

    assert fw_opt_ms <= FORWARD_OPT_BUDGET_MS
    assert bw_opt_ms <= BACKWARD_OPT_BUDGET_MS
    assert fw_opt_ms < fw_base_ms
    assert bw_opt_ms < bw_base_ms
