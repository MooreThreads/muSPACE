"""Tests for the depth-aware voxel pooling operator."""

from __future__ import annotations

import pytest
import torch

from mu_space import (
    clear_dispatch_report,
    get_last_dispatch_report,
    voxel_pool_depth,
)

_MUSA_REQUIRED = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)


def _hand_case() -> dict:
    """Three frustum cells of one camera collapsing into a single voxel.

    ``img_feats`` is ``[1, 1, 1, 1, 3]``: one feature channel and three image
    columns, so each cell contributes ``img_depth[w] * img_feats[0, 0, 0, 0, w]``.
    """
    return {
        "B": 1,
        "X": 1,
        "Y": 1,
        "Z": 1,
        "img_feats": torch.tensor([[[[[2.0, 3.0, 4.0]]]]], dtype=torch.float32),
        "img_depth": torch.tensor([[[[0.5, 0.25, 0.125]]]], dtype=torch.float32),
        "voxel_coords": torch.zeros((1, 1, 1, 1, 1, 3, 4), dtype=torch.int32),
        "mask": torch.ones((1, 1, 1, 1, 1, 3), dtype=torch.bool),
    }


def _random_case(
    dtype: torch.dtype = torch.float32,
    channels: int = 3,
    mask_ratio: float = 0.6,
    seed: int = 2333,
    dims: tuple[int, ...] = (1, 1, 2, 2, 2, 2, 2, 2, 2),
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    B, N_sweep, N_cam, D, H, W, X, Y, Z = dims
    coords = torch.zeros((B, N_sweep, N_cam, D, H, W, 4), dtype=torch.int32)
    coords[..., 1] = torch.randint(0, X, coords.shape[:-1], generator=generator)
    coords[..., 2] = torch.randint(0, Y, coords.shape[:-1], generator=generator)
    coords[..., 3] = torch.randint(0, Z, coords.shape[:-1], generator=generator)
    mask = torch.rand(coords.shape[:-1], generator=generator) < mask_ratio
    return {
        "B": B,
        "X": X,
        "Y": Y,
        "Z": Z,
        "img_feats": torch.randn(B, N_sweep * N_cam, channels, H, W, generator=generator).to(dtype),
        "img_depth": torch.rand(B * N_sweep * N_cam, D, H, W, generator=generator).to(dtype),
        "voxel_coords": coords,
        "mask": mask,
    }


def _to_device(case: dict, device: str) -> dict:
    moved = dict(case)
    for name in ("img_feats", "img_depth", "voxel_coords", "mask"):
        moved[name] = case[name].to(device)
    return moved


def _apply(case: dict, backend: str, *, clone_inputs: bool = False):
    feats = case["img_feats"]
    depth = case["img_depth"]
    if clone_inputs:
        feats = feats.clone().requires_grad_(True)
        depth = depth.clone().requires_grad_(True)
    output = voxel_pool_depth(
        feats,
        depth,
        case["voxel_coords"],
        case["mask"],
        case["B"],
        case["X"],
        case["Y"],
        case["Z"],
        backend=backend,
    )
    return output, feats, depth


def test_reference_matches_hand_computed_values_and_gradients():
    clear_dispatch_report()
    case = _hand_case()
    output, feats, depth = _apply(case, "reference", clone_inputs=True)
    assert output.shape == (1, 1, 1, 1, 1, 1)
    torch.testing.assert_close(output, torch.tensor([[[[[[2.25]]]]]]))

    output.sum().backward()
    torch.testing.assert_close(feats.grad, torch.tensor([[[[[0.5, 0.25, 0.125]]]]]))
    torch.testing.assert_close(depth.grad, torch.tensor([[[[2.0, 3.0, 4.0]]]]))

    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "voxel_pool_depth"
    assert report.selected_backend == "reference"


def test_reference_random_case_produces_finite_gradients():
    case = _random_case()
    output, feats, depth = _apply(case, "reference", clone_inputs=True)
    assert output.shape == (1, 2, 2, 2, 2, 3)
    output.sum().backward()
    assert torch.isfinite(feats.grad).all()
    assert torch.isfinite(depth.grad).all()


def test_reference_total_mass_matches_depth_weighted_features():
    case = _random_case(mask_ratio=1.0)
    output, _, _ = _apply(case, "reference")
    *_, D, H, W = case["mask"].shape
    C = case["img_feats"].shape[2]
    selected = torch.arange(case["mask"].numel())[case["mask"].reshape(-1)]
    image = selected // (D * H * W)
    spatial = selected % (D * H * W)
    d, h, w = spatial // (H * W), (spatial // W) % H, spatial % W
    feats = case["img_feats"].reshape(-1, C, H, W).to(torch.float32)
    expected = (
        case["img_depth"].to(torch.float32)[image, d, h, w] * feats[image, :, h, w].sum(1)
    ).sum()
    torch.testing.assert_close(output.sum(), expected)


def test_auto_backend_on_cpu_uses_reference():
    clear_dispatch_report()
    case = _random_case()
    output, _, _ = _apply(case, "auto")
    assert output.shape == (1, 2, 2, 2, 2, 3)
    report = get_last_dispatch_report()
    assert report is not None
    assert report.selected_backend == "reference"


def test_unsupported_feature_dtype_is_rejected():
    case = _random_case()
    case["img_feats"] = case["img_feats"].to(torch.int32)
    with pytest.raises(TypeError, match="img_feats must use float32, float16 or bfloat16"):
        _apply(case, "reference")


def test_mask_shape_must_match_voxel_coords():
    case = _random_case()
    case["mask"] = case["mask"][..., :-1]
    with pytest.raises(ValueError, match="mask must match voxel_coords.shape"):
        _apply(case, "reference")


def test_mask_may_be_non_boolean():
    case = _random_case()
    case["mask"] = case["mask"].to(torch.uint8)
    output, _, _ = _apply(case, "reference")
    assert output.shape == (1, 2, 2, 2, 2, 3)


def test_reference_empty_mask_returns_zeros_and_zero_gradients():
    case = _random_case()
    case["mask"] = torch.zeros_like(case["mask"])
    output, feats, depth = _apply(case, "reference", clone_inputs=True)
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    torch.testing.assert_close(feats.grad, torch.zeros_like(feats.grad))
    torch.testing.assert_close(depth.grad, torch.zeros_like(depth.grad))


@_MUSA_REQUIRED
def test_native_matches_reference_forward_and_backward_fp32():
    case = _random_case()
    reference_case = _to_device(case, "cpu")
    native_case = _to_device(case, "musa")
    reference, ref_feats, ref_depth = _apply(reference_case, "reference", clone_inputs=True)
    actual, musa_feats, musa_depth = _apply(native_case, "musa", clone_inputs=True)

    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)
    actual.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(musa_feats.grad.cpu(), ref_feats.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(musa_depth.grad.cpu(), ref_depth.grad, rtol=1e-4, atol=1e-4)

    report = get_last_dispatch_report()
    assert report is not None
    assert report.selected_backend == "musa"


@_MUSA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_half_precision_forward_matches_reference(dtype):
    case = _random_case(dtype=dtype)
    reference, _, _ = _apply(_to_device(case, "cpu"), "reference")
    actual, _, _ = _apply(_to_device(case, "musa"), "musa")
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-2, atol=1e-2)


@_MUSA_REQUIRED
@pytest.mark.parametrize("channels", [3, 32, 64])
def test_native_channel_launch_paths_match_reference(channels):
    case = _random_case(channels=channels)
    reference, _, _ = _apply(_to_device(case, "cpu"), "reference")
    actual, _, _ = _apply(_to_device(case, "musa"), "musa")
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)


@_MUSA_REQUIRED
@pytest.mark.parametrize("channels", [6, 12])
def test_native_non_power_of_two_dims_match_reference(channels):
    """Non-power-of-two divisors exercise the reciprocal multiply-shift path.

    ``channels=6`` takes the linear kernel and ``channels=12`` the packed one;
    the sweep, camera, depth and width divisors are all non-power-of-two.
    """
    dims = (1, 2, 3, 3, 5, 7, 3, 4, 2)
    case = _random_case(channels=channels, mask_ratio=0.8, dims=dims)
    reference, ref_feats, ref_depth = _apply(
        _to_device(case, "cpu"), "reference", clone_inputs=True
    )
    actual, musa_feats, musa_depth = _apply(_to_device(case, "musa"), "musa", clone_inputs=True)
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)
    actual.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(musa_feats.grad.cpu(), ref_feats.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(musa_depth.grad.cpu(), ref_depth.grad, rtol=1e-4, atol=1e-4)


@_MUSA_REQUIRED
@pytest.mark.parametrize("strategy", ["linear", "2d", "packed"])
def test_native_launch_environment_override_matches_reference(monkeypatch, strategy):
    monkeypatch.setenv("MUSPACE_VOXEL_POOL_DEPTH_LAUNCH", strategy)
    case = _random_case(channels=64)
    reference, _, _ = _apply(_to_device(case, "cpu"), "reference")
    actual, _, _ = _apply(_to_device(case, "musa"), "musa")
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)


@_MUSA_REQUIRED
def test_native_empty_mask_returns_zeros_and_zero_gradients():
    case = _random_case()
    case["mask"] = torch.zeros_like(case["mask"])
    native_case = _to_device(case, "musa")
    output, feats, depth = _apply(native_case, "musa", clone_inputs=True)
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    torch.testing.assert_close(feats.grad, torch.zeros_like(feats.grad))
    torch.testing.assert_close(depth.grad, torch.zeros_like(depth.grad))
