"""Tests for the 2-D and 3-D grid_sample operator."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from mu_space import clear_dispatch_report, get_last_dispatch_report, grid_sample


def _case(rank: int, dtype: torch.dtype = torch.float32):
    if rank == 4:
        value = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 7
        grid = torch.tensor(
            [[[[0.0, 0.0], [1.0, -1.0]], [[-1.2, 0.2], [2.0, 2.0]]]], dtype=torch.float32
        )
    else:
        value = torch.arange(48, dtype=torch.float32).reshape(1, 2, 2, 3, 4) / 7
        grid = torch.tensor(
            [
                [
                    [[[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]]],
                    [[[-1.2, 0.2, 0.0], [2.0, 2.0, 2.0]]],
                ]
            ],
            dtype=torch.float32,
        )
    return value.to(dtype).requires_grad_(True), grid.to(dtype).requires_grad_(True)


@pytest.mark.parametrize("rank", [4, 5])
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [False, True])
def test_reference_matches_torch(rank, mode, padding_mode, align_corners):
    value, grid = _case(rank)
    actual = grid_sample(value, grid, mode, padding_mode, align_corners, backend="reference")
    expected = F.grid_sample(value, grid, mode, padding_mode, align_corners)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("rank", [4, 5])
def test_reference_backward_matches_torch(rank):
    value, grid = _case(rank)
    value_ref, grid_ref = (
        value.detach().clone().requires_grad_(),
        grid.detach().clone().requires_grad_(),
    )
    actual = grid_sample(value, grid, backend="reference").square().sum()
    expected = F.grid_sample(value_ref, grid_ref, align_corners=False).square().sum()
    actual.backward()
    expected.backward()
    torch.testing.assert_close(value.grad, value_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grid.grad, grid_ref.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reference_half_dtypes(dtype):
    value, grid = _case(5, dtype)
    actual = grid_sample(value, grid, backend="reference")
    expected = F.grid_sample(value.float(), grid.float(), align_corners=False).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_validation_rejects_bad_shapes_and_modes():
    value, grid = _case(4)
    with pytest.raises(ValueError, match="grid last dimension"):
        grid_sample(value, grid[..., :1], backend="reference")
    with pytest.raises(ValueError, match="mode"):
        grid_sample(value, grid, mode="cubic", backend="reference")
    with pytest.raises(ValueError, match="padding_mode"):
        grid_sample(value, grid, padding_mode="wrap", backend="reference")
    with pytest.raises(ValueError, match="4-D or 5-D"):
        grid_sample(value[0], grid, backend="reference")
    with pytest.raises(TypeError, match="input must use"):
        grid_sample(value.to(torch.int32), grid.to(torch.int32), backend="reference")


def test_cpu_auto_uses_reference_and_forced_musa_fails():
    value, grid = _case(4)
    clear_dispatch_report()
    output = grid_sample(value, grid, backend="auto")
    assert output.shape == (1, 2, 2, 2)
    report = get_last_dispatch_report()
    assert report is not None
    assert report.selected_backend == "reference"
    with pytest.raises(RuntimeError, match="requires a MUSA tensor"):
        grid_sample(value, grid, backend="musa")


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
@pytest.mark.parametrize("rank", [4, 5])
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_native_matches_reference(rank, mode, padding_mode):
    value, grid = _case(rank)
    value = value.musa()
    grid = grid.musa()
    expected = F.grid_sample(value, grid, mode, padding_mode, False)
    actual = grid_sample(value, grid, mode, padding_mode, False, backend="musa")
    if mode == "nearest":
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    report = get_last_dispatch_report()
    assert report is not None
    assert report.selected_backend == "musa"
    assert report.native_fallback is False


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_backward_matches_reference():
    value, grid = _case(4)
    value = value.musa().detach().requires_grad_()
    grid = grid.musa().detach().requires_grad_()
    native = grid_sample(value, grid, "bilinear", "zeros", False, backend="musa")
    native_loss = (native.square() * 0.5).sum()
    native_loss.backward()
    native_value_grad = value.grad.detach().cpu()
    native_grid_grad = grid.grad.detach().cpu()

    ref_value = value.detach().cpu().requires_grad_()
    ref_grid = grid.detach().cpu().requires_grad_()
    reference = F.grid_sample(ref_value, ref_grid, "bilinear", "zeros", False)
    (reference.square() * 0.5).sum().backward()
    torch.testing.assert_close(native_value_grad, ref_value.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(native_grid_grad, ref_grid.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
@pytest.mark.parametrize("rank", [4, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("singleton", [False, True])
def test_native_full_matrix(rank, dtype, mode, padding, align, singleton):
    generator = torch.Generator().manual_seed(2333)
    shape = (2, 3, 1 if singleton else 3, 4) if rank == 4 else (2, 3, 1 if singleton else 2, 3, 4)
    grid_shape = (2, 4, 7, 2) if rank == 4 else (2, 2, 4, 7, 3)
    value = torch.randn(shape, generator=generator).to(dtype)
    grid = (torch.rand(grid_shape, generator=generator) * 12 - 6).to(dtype)
    # Repeated locations, normalized endpoints, and nearest half-integer ties.
    points = grid.reshape(2, -1, rank - 2)
    points[:, :4] = torch.tensor([-1.0, 0.0, 1.0, 3.0], dtype=dtype).view(1, 4, 1)
    points[:, 4:8] = points[:, :4]
    a = value.musa().requires_grad_()
    b = grid.musa().requires_grad_()
    ref_a = value.float().requires_grad_()
    ref_b = grid.float().requires_grad_()
    expected = F.grid_sample(ref_a, ref_b, mode, padding, align)
    actual = grid_sample(a, b, mode, padding, align, backend="musa")
    tolerance = 1e-4 if dtype == torch.float32 else 1e-2
    if mode == "nearest":
        assert torch.equal(actual.cpu(), expected.to(dtype))
    else:
        torch.testing.assert_close(actual.cpu(), expected.to(dtype), rtol=tolerance, atol=tolerance)
    grad = torch.randn(expected.shape, generator=generator).to(dtype)
    actual.backward(grad.musa())
    expected.backward(grad.float())
    torch.testing.assert_close(a.grad.cpu(), ref_a.grad.to(dtype), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(b.grad.cpu(), ref_b.grad.to(dtype), rtol=tolerance, atol=tolerance)
    report = get_last_dispatch_report()
    assert report.selected_backend == "musa"
    assert report.native_fallback is False


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
@pytest.mark.parametrize(
    "shape,grid_shape",
    [
        ((1, 64, 128, 128), (1, 64, 64, 2)),
        ((1, 64, 256, 256), (1, 256, 256, 2)),
        ((2, 64, 512, 512), (2, 512, 512, 2)),
        ((1, 256, 128, 128), (1, 64, 64, 2)),
        ((1, 32, 32, 64, 64), (1, 32, 64, 64, 3)),
    ],
)
def test_native_expanded_random_gradient(shape, grid_shape):
    generator = torch.Generator().manual_seed(2333)
    value = torch.randn(shape, generator=generator).musa().requires_grad_()
    grid = (torch.rand(grid_shape, generator=generator) * 2 - 1).musa().requires_grad_()
    grad = torch.randn((shape[0], shape[1], *grid_shape[1:-1]), generator=generator).musa()
    actual = grid_sample(value, grid, backend="musa")
    # Use the device reference: CPU 2-D backward groups pixel differences instead
    # of accumulating corners, which differs at FP32 cancellation points.
    expected = F.grid_sample(value, grid, align_corners=False)
    actual_grads = torch.autograd.grad(actual, (value, grid), grad)
    expected_grads = torch.autograd.grad(expected, (value, grid), grad)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
@pytest.mark.parametrize("rank", [4, 5])
@pytest.mark.parametrize("empty", ["batch", "channel", "output"])
def test_native_empty_and_noncontiguous(rank, empty):
    shape = [2, 3, 4, 5] if rank == 4 else [2, 3, 2, 4, 5]
    grid_shape = [2, 3, 4, 2] if rank == 4 else [2, 2, 3, 4, 3]
    if empty == "batch":
        shape[0] = grid_shape[0] = 0
    elif empty == "channel":
        shape[1] = 0
    else:
        grid_shape[1] = 0
    value = torch.randn(shape).transpose(-1, -2).musa().detach().requires_grad_()
    grid = torch.randn(grid_shape).transpose(1, 2).musa().detach().requires_grad_()
    out = grid_sample(value, grid, backend="musa")
    out.sum().backward()
    assert out.numel() == 0
    assert torch.count_nonzero(value.grad).item() == 0
    assert torch.count_nonzero(grid.grad).item() == 0


@pytest.mark.parametrize("rank", [4, 5])
def test_validation_rejects_empty_spatial_dimension(rank):
    shape = (1, 2, 0, 4) if rank == 4 else (1, 2, 0, 3, 4)
    _, grid = _case(rank)
    with pytest.raises(ValueError, match="spatial dimensions must be positive"):
        grid_sample(torch.empty(shape), grid, backend="musa")


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
@pytest.mark.parametrize("rank", [4, 5])
def test_native_noncontiguous_forward_backward(rank):
    value, grid = _case(rank)
    value = value.detach().transpose(-1, -2).musa().requires_grad_()
    grid = grid.detach().transpose(1, 2).musa().requires_grad_()
    ref_value = value.detach().cpu().requires_grad_()
    ref_grid = grid.detach().cpu().requires_grad_()
    expected = F.grid_sample(ref_value, ref_grid, padding_mode="reflection", align_corners=True)
    actual = grid_sample(value, grid, padding_mode="reflection", align_corners=True, backend="musa")
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(value.grad.cpu(), ref_value.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grid.grad.cpu(), ref_grid.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_channel_launch_stride():
    value = torch.arange(65536, dtype=torch.float32, device="musa").reshape(1, 65536, 1, 1)
    grid = torch.zeros((1, 1, 1, 2), device="musa")
    actual = grid_sample(value, grid, mode="nearest", backend="musa")
    assert torch.equal(actual, value)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_entry_validation():
    from mu_space.runtime.native import native_module_proxy

    ext = native_module_proxy()
    value, grid = (tensor.detach().musa() for tensor in _case(4))
    output = torch.empty((1, 2, 2, 2), device="musa")
    for mode, padding, message in [(2, 0, "interpolation mode"), (0, 3, "padding mode")]:
        with pytest.raises(RuntimeError, match=message):
            ext.grid_sample_forward(value, grid, output, mode, padding, False)
    with pytest.raises(RuntimeError, match="output spatial shape"):
        ext.grid_sample_forward(value, grid, output[..., :1].contiguous(), 0, 0, False)
    with pytest.raises(RuntimeError, match="grid dtype"):
        ext.grid_sample_forward(value, grid.half(), output, 0, 0, False)
    with pytest.raises(RuntimeError, match="grad_input shape"):
        ext.grid_sample_backward(
            value, grid, output, value[..., :1].contiguous(), torch.empty_like(grid), 0, 0, False
        )
