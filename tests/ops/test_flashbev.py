import pytest
import torch

from mu_space.ops import flash_bevpool, flash_bevpool_sca


def make_uvd(projections, roi, grid, device, epsilon=1e-6):
    gx, gy, gz = grid
    xs = torch.linspace(roi[0], roi[1], gx + 1, device=device)[:-1] + (roi[1] - roi[0]) / gx / 2
    ys = torch.linspace(roi[2], roi[3], gy + 1, device=device)[:-1] + (roi[3] - roi[2]) / gy / 2
    zs = torch.linspace(roi[4], roi[5], gz + 1, device=device)[:-1] + (roi[5] - roi[4]) / gz / 2
    x, y, z = torch.meshgrid(xs, ys, zs, indexing="ij")
    xyz1 = torch.stack((x, y, z, torch.ones_like(x)), dim=-1)
    homogeneous = torch.einsum("bnij,xyzj->bnxyzi", projections, xyz1)
    return torch.cat(
        (homogeneous[..., :2] / (homogeneous[..., 2:3] + epsilon), homogeneous[..., 2:3]), dim=-1
    )


def _basic_fixture(device="cpu", grid_z=2):
    image = torch.linspace(0.1, 3.2, 1 * 2 * 3 * 4 * 2, device=device).view(1, 2, 3, 4, 2)
    image.requires_grad_()
    projections = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 2.0]],
                [[1.0, 0.0, 0.0, 30.0], [0.0, 1.0, 0.0, 30.0], [0.0, 0.0, 1.0, 2.0]],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    feature_size = torch.tensor([[[3, 4], [3, 4]]], dtype=torch.int32, device=device)
    image_size = torch.tensor([3, 4], dtype=torch.int32, device=device)
    return (
        image,
        projections,
        feature_size,
        image_size,
        (2, 2, grid_z),
        (2.0, 2.0, 1.0),
        (0.0, 0.0, 0.0),
    )


def _sca_fixture(device="cpu"):
    value = torch.linspace(0.1, 6.4, 1 * 2 * 16 * 4, device=device).view(1, 2, 16, 4)
    value.requires_grad_()
    projections = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 2.0]],
                [[1.0, 0.0, 0.0, 30.0], [0.0, 1.0, 0.0, 30.0], [0.0, 0.0, 1.0, 2.0]],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    spatial_shapes = torch.tensor([[3, 4], [2, 2]], dtype=torch.int32, device=device)
    level_start = torch.tensor([0, 12], dtype=torch.int32, device=device)
    image_size = torch.tensor([3, 4], dtype=torch.int32, device=device)
    offsets = torch.zeros((1, 2, 2, 2, 2, 2, 2, 2), dtype=torch.float32, device=device)
    offsets[..., 0] = 0.1
    offsets[..., 1] = -0.15
    offsets.requires_grad_()
    weights = torch.rand((1, 2, 2, 2, 2, 2, 2), dtype=torch.float32, device=device)
    weights.requires_grad_()
    return value, projections, spatial_shapes, level_start, image_size, offsets, weights


def test_flash_bevpool_reference_forward_backward():
    args = _basic_fixture()
    output = flash_bevpool(*args, backend="reference")
    assert output.shape == (1, 2, 2, 2)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    assert torch.isfinite(args[0].grad).all()


def test_flash_bevpool_reference_rejects_invalid_inputs():
    args = list(_basic_fixture())
    args[0] = args[0].double()
    with pytest.raises(TypeError, match="image_feats must use float32"):
        flash_bevpool(*args, backend="reference")
    args = list(_basic_fixture())
    args[2] = args[2].to(torch.int64)
    with pytest.raises(TypeError, match="feature_size must use int32"):
        flash_bevpool(*args, backend="reference")
    args = list(_basic_fixture())
    args[0] = args[0].transpose(2, 3)
    with pytest.raises(ValueError, match="image_feats must be contiguous"):
        flash_bevpool(*args, backend="reference")
    args = list(_basic_fixture())
    args[2] = args[2][..., :1].contiguous()
    with pytest.raises(ValueError, match="feature_size must have shape"):
        flash_bevpool(*args, backend="reference")


def test_tensorized_uvd_builder_performs_perspective_division():
    projections = torch.tensor(
        [[[[4.0, 0.0, 0.0, 2.0], [0.0, 6.0, 0.0, 3.0], [0.0, 0.0, 2.0, 1.0]]]],
        dtype=torch.float32,
    )
    # The only voxel centre is (1, 1, 1): homogeneous coordinates are
    # (6, 9, 3), but Tensorized Sampling-VT consumes (2, 3, 3).
    uvd = make_uvd(projections, (0.0, 2.0, 0.0, 2.0, 0.0, 2.0), (1, 1, 1), "cpu")
    torch.testing.assert_close(uvd[0, 0, 0, 0, 0], torch.tensor([2.0, 3.0, 3.0]))


def test_flash_bevpool_reference_handles_boundary_and_zero_valid_cameras():
    args = list(_basic_fixture())
    # Keep camera 0 just inside the right image boundary.  FlashBEV's BEVPool
    # path must clamp its upper-right bilinear corner to the boundary pixel.
    args[1][0, 0, 0, 3] = 8.997
    args[1][0, 0, 1, 3] = 0.5
    boundary = flash_bevpool(*args, backend="reference")
    assert torch.isfinite(boundary).all()
    args[1][..., 0, 3] = 100.0
    zero = flash_bevpool(*args, backend="reference")
    torch.testing.assert_close(zero, torch.zeros_like(zero))


def test_flash_bevpool_reference_rejects_nonpositive_projected_depth():
    args = list(_basic_fixture())
    args[1][..., 2, 3] = -2.0
    output = flash_bevpool(*args, backend="reference")
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_flash_bevpool_sca_rejects_channels_not_divisible_by_heads():
    args = list(_sca_fixture())
    args[0] = args[0][..., :3].contiguous().detach().requires_grad_()
    with pytest.raises(ValueError, match="channels must be divisible by num_heads"):
        flash_bevpool_sca(
            *args,
            (2, 2, 2),
            (2.0, 2.0, 1.0),
            (0.0, 0.0, 0.0),
            num_heads=2,
            backend="reference",
        )


def test_flash_bevpool_sca_reference_forward_backward():
    args = _sca_fixture()
    output = flash_bevpool_sca(
        *args, (2, 2, 2), (2.0, 2.0, 1.0), (0.0, 0.0, 0.0), backend="reference"
    )
    assert output.shape == (1, 2, 2, 4)
    output.sum().backward()
    for tensor in (args[0], args[5], args[6]):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_flash_bevpool_sca_rejects_inconsistent_level_layout():
    args = list(_sca_fixture())
    with pytest.raises(ValueError, match="value_stride_n must cover"):
        flash_bevpool_sca(
            *args,
            (2, 2, 2),
            (2.0, 2.0, 1.0),
            (0.0, 0.0, 0.0),
            value_stride_n=15,
            backend="reference",
        )
    args = list(_sca_fixture())
    with pytest.raises(ValueError, match="sampling_offsets must have shape"):
        flash_bevpool_sca(
            *args[:-2],
            args[-2][..., :1, :].contiguous(),
            args[-1],
            (2, 2, 2),
            (2.0, 2.0, 1.0),
            (0.0, 0.0, 0.0),
            backend="reference",
        )


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_flash_bevpool_native_matches_reference_and_supports_grid_z_65():
    native_args = list(_basic_fixture(device="musa", grid_z=65))
    # Keep all Z samples identical so this test isolates the formerly fixed-size
    # backward/count path from projection interpolation differences.
    native_args[5] = (2.0, 2.0, 0.0)
    native_args = tuple(native_args)
    reference_args = tuple(
        tensor.detach().cpu().clone().requires_grad_(tensor.requires_grad)
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
        else tensor.detach().cpu().clone()
        if isinstance(tensor, torch.Tensor)
        else tensor
        for tensor in native_args
    )
    expected = flash_bevpool(*reference_args, backend="reference")
    actual = flash_bevpool(*native_args, backend="musa")
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        native_args[0].grad.cpu(), reference_args[0].grad, rtol=1e-4, atol=1e-4
    )


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_flash_bevpool_native_matches_reference_at_boundary_and_without_valid_cameras():
    native_args = list(_basic_fixture(device="musa"))
    native_args[1][0, 0, 0, 3] = 8.997
    native_args[1][0, 0, 1, 3] = 0.5
    reference_args = tuple(
        tensor.detach().cpu().clone().requires_grad_(tensor.requires_grad)
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
        else tensor.detach().cpu().clone()
        if isinstance(tensor, torch.Tensor)
        else tensor
        for tensor in native_args
    )
    expected = flash_bevpool(*reference_args, backend="reference")
    actual = flash_bevpool(*native_args, backend="musa")
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)

    native_args[1][..., 0, 3] = 100.0
    no_camera = flash_bevpool(*native_args, backend="musa")
    torch.testing.assert_close(no_camera.cpu(), torch.zeros_like(no_camera.cpu()))
    no_camera.sum().backward()
    torch.testing.assert_close(
        native_args[0].grad.cpu(), torch.zeros_like(native_args[0].grad.cpu())
    )


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_flash_bevpool_native_matches_reference_with_seven_cameras():
    native_args = list(_basic_fixture(device="musa"))
    native_args[0] = (
        native_args[0].detach().repeat(1, 4, 1, 1, 1)[:, :7].contiguous().requires_grad_()
    )
    native_args[1] = native_args[1].repeat(1, 4, 1, 1)[:, :7].contiguous()
    native_args[2] = native_args[2].repeat(1, 4, 1)[:, :7].contiguous()
    reference_args = tuple(
        tensor.detach().cpu().clone().requires_grad_(tensor.requires_grad)
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
        else tensor.detach().cpu().clone()
        if isinstance(tensor, torch.Tensor)
        else tensor
        for tensor in native_args
    )
    expected = flash_bevpool(*reference_args, backend="reference")
    actual = flash_bevpool(*native_args, backend="musa")
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        native_args[0].grad.cpu(), reference_args[0].grad, rtol=1e-4, atol=1e-4
    )


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_flash_bevpool_sca_native_matches_reference_backward():
    native_args = _sca_fixture(device="musa")
    reference_args = tuple(
        tensor.detach().cpu().clone().requires_grad_(tensor.requires_grad)
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
        else tensor.detach().cpu().clone()
        if isinstance(tensor, torch.Tensor)
        else tensor
        for tensor in native_args
    )
    call_args = ((2, 2, 2), (2.0, 2.0, 1.0), (0.0, 0.0, 0.0))
    expected = flash_bevpool_sca(*reference_args, *call_args, backend="reference")
    actual = flash_bevpool_sca(*native_args, *call_args, backend="musa")
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    actual.sum().backward()
    expected.sum().backward()
    for native, reference in (
        (native_args[0], reference_args[0]),
        (native_args[5], reference_args[5]),
        (native_args[6], reference_args[6]),
    ):
        torch.testing.assert_close(native.grad.cpu(), reference.grad, rtol=2e-4, atol=2e-4)
