"""Tests for the BEV temporal align operator."""

from __future__ import annotations

import math

import pytest
import torch

from mu_space import (
    clear_dispatch_report,
    get_last_dispatch_report,
    temporal_align,
    temporal_align_reference,
)

ANGLES = (0.0, 0.1, -0.1, 90.0, -90.0, 180.0, -180.0)
MUSA_REQUIRED = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)


def _bev(batch: int, channels: int, height: int, width: int, dtype=torch.float32, seed=2333):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, channels, height, width, generator=generator).to(dtype)


def _centre(center, height: int, width: int) -> tuple[float, float]:
    if center is None:
        return (width - 1) * 0.5, (height - 1) * 0.5
    return float(center[0]) - 0.5, float(center[1]) - 0.5


def _taps(sample_x: float, sample_y: float, height: int, width: int, mode: str):
    """Yield ``(index_y, index_x, weight)`` taps in source pixel space."""
    if mode == "nearest":
        index_x = int(round(sample_x))
        index_y = int(round(sample_y))
        if 0 <= index_x < width and 0 <= index_y < height:
            yield index_y, index_x, 1.0
        return
    if not (-1.0 < sample_x < width and -1.0 < sample_y < height):
        return
    lower_x = math.floor(sample_x)
    lower_y = math.floor(sample_y)
    frac_x = sample_x - lower_x
    frac_y = sample_y - lower_y
    for step_y in (0, 1):
        for step_x in (0, 1):
            index_x = lower_x + step_x
            index_y = lower_y + step_y
            if not (0 <= index_x < width and 0 <= index_y < height):
                continue
            weight = (frac_x if step_x else 1.0 - frac_x) * (frac_y if step_y else 1.0 - frac_y)
            yield index_y, index_x, weight


def _geometry(angle, shift, sample, center, height, width):
    centre_x, centre_y = _centre(center, height, width)
    radians = math.radians(0.0 if angle is None else float(angle[sample].item()))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    shift_x = 0.0 if shift is None else float(shift[sample, 0].item())
    shift_y = 0.0 if shift is None else float(shift[sample, 1].item())
    return centre_x, centre_y, cosine, sine, shift_x, shift_y


def _naive_forward(bev, angle=None, shift=None, center=None, mode="bilinear"):
    """Independent per-pixel oracle without any grid sampling helper."""
    batch, channels, height, width = bev.shape
    output = torch.zeros_like(bev)
    for sample in range(batch):
        centre_x, centre_y, cosine, sine, shift_x, shift_y = _geometry(
            angle, shift, sample, center, height, width
        )
        for y in range(height):
            for x in range(width):
                delta_x = x - centre_x
                delta_y = y - centre_y
                sample_x = cosine * delta_x - sine * delta_y + centre_x + shift_x
                sample_y = sine * delta_x + cosine * delta_y + centre_y + shift_y
                for index_y, index_x, weight in _taps(sample_x, sample_y, height, width, mode):
                    output[sample, :, y, x] += bev[sample, :, index_y, index_x] * weight
    return output


def _naive_backward(grad_output, angle=None, shift=None, center=None, mode="bilinear"):
    """Independent per-pixel scatter oracle for the backward pass."""
    batch, channels, height, width = grad_output.shape
    grad_bev = torch.zeros_like(grad_output)
    for sample in range(batch):
        centre_x, centre_y, cosine, sine, shift_x, shift_y = _geometry(
            angle, shift, sample, center, height, width
        )
        for y in range(height):
            for x in range(width):
                delta_x = x - centre_x
                delta_y = y - centre_y
                sample_x = cosine * delta_x - sine * delta_y + centre_x + shift_x
                sample_y = sine * delta_x + cosine * delta_y + centre_y + shift_y
                for index_y, index_x, weight in _taps(sample_x, sample_y, height, width, mode):
                    grad_bev[sample, :, index_y, index_x] += grad_output[sample, :, y, x] * weight
    return grad_bev


def _reference_gradients(bev, angle, shift, weight, center=None, mode="bilinear"):
    value = bev.detach().clone().requires_grad_()
    weights = weight.detach().clone()
    output = temporal_align_reference(
        value, angle, shift, bev_h=bev.shape[2], bev_w=bev.shape[3], center=center, mode=mode
    )
    (output * weights).sum().backward()
    return output.detach(), value.grad


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_reference_matches_naive_forward(mode):
    bev = _bev(2, 3, 5, 4)
    angle = torch.tensor([0.0, 37.0], dtype=torch.float32)
    shift = torch.tensor([[0.0, 0.0], [1.5, -2.0]], dtype=torch.float32)
    actual = temporal_align_reference(bev, angle, shift, center=(1.0, 2.0), mode=mode)
    expected = _naive_forward(bev, angle, shift, center=(1.0, 2.0), mode=mode)
    if mode == "nearest":
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_reference_identity_and_default_centre():
    bev = _bev(2, 3, 6, 5)
    zero_angle = torch.zeros(2, dtype=torch.float32)
    identity = temporal_align_reference(bev, zero_angle, bev_h=6, bev_w=5)
    torch.testing.assert_close(identity, bev, rtol=0, atol=1e-6)
    # A corner-based centre of (width / 2, height / 2) is the geometric centre.
    centred = temporal_align_reference(bev, zero_angle, center=(2.5, 3.0), bev_h=6, bev_w=5)
    torch.testing.assert_close(centred, bev, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "offset",
    [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, -3), (5, 0), (-7, 4), (0, 9)],
)
def test_reference_integer_shift_is_a_plane_copy(offset):
    bev = _bev(2, 3, 5, 6)
    shift = torch.tensor([list(offset), [0, 0]], dtype=torch.int32)
    actual = temporal_align_reference(bev, shift=shift, bev_h=5, bev_w=6)
    expected = _naive_forward(bev, shift=shift)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A pure translation is an exact slice copy away from the padded border.
    shift_x, shift_y = offset
    height, width = 5, 6
    expected_plane = torch.zeros_like(bev[0])
    start_y, stop_y = max(0, -shift_y), min(height, height - shift_y)
    start_x, stop_x = max(0, -shift_x), min(width, width - shift_x)
    if stop_y > start_y and stop_x > start_x:
        expected_plane[:, start_y:stop_y, start_x:stop_x] = bev[
            0, :, start_y + shift_y : stop_y + shift_y, start_x + shift_x : stop_x + shift_x
        ]
    torch.testing.assert_close(actual[0], expected_plane, rtol=0, atol=0)


def test_reference_backward_matches_naive_scatter():
    bev = _bev(2, 3, 5, 4)
    angle = torch.tensor([23.0, -37.0], dtype=torch.float32)
    shift = torch.tensor([[0.5, -1.5], [0.0, 0.0]], dtype=torch.float32)
    weight = _bev(2, 3, 5, 4, seed=17)
    output, gradient = _reference_gradients(bev, angle, shift, weight)
    expected_output = _naive_forward(bev, angle, shift)
    torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-6)
    for sample in range(bev.shape[0]):
        expected = _naive_backward(
            weight[sample : sample + 1],
            angle[sample : sample + 1],
            shift[sample : sample + 1],
        )
        torch.testing.assert_close(gradient[sample : sample + 1], expected, rtol=1e-5, atol=1e-5)


def test_reference_integer_shift_backward_round_trip():
    bev = _bev(2, 3, 5, 6, seed=5)
    shift = torch.tensor([[1, -1], [-2, 0]], dtype=torch.int32)
    weight = _bev(2, 3, 5, 6, seed=6)
    _, gradient = _reference_gradients(bev, None, shift, weight)
    for sample in range(bev.shape[0]):
        expected = _naive_backward(
            weight[sample : sample + 1], None, shift[sample : sample + 1].to(torch.float32)
        )
        torch.testing.assert_close(gradient[sample : sample + 1], expected, rtol=1e-6, atol=1e-6)


def test_reference_layouts_agree():
    bev = _bev(2, 3, 5, 4)
    channels_last = bev.permute(0, 2, 3, 1).reshape(2, 20, 3)
    angle = torch.tensor([15.0, -60.0], dtype=torch.float32)
    from_4d = temporal_align_reference(bev, angle, bev_h=5, bev_w=4)
    from_3d = temporal_align_reference(channels_last, angle, bev_h=5, bev_w=4)
    torch.testing.assert_close(from_3d, from_4d.permute(0, 2, 3, 1).reshape(2, 20, 3))


def test_reference_matches_torchvision_rotate():
    pytest.importorskip("torchvision")
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import rotate

    bev = _bev(2, 3, 7, 5)
    for center in (None, (1.0, 2.0)):
        for angle in ANGLES:
            for mode in ("nearest", "bilinear"):
                actual = temporal_align_reference(
                    bev,
                    torch.full((2,), angle, dtype=torch.float32),
                    center=center,
                    mode=mode,
                )
                interpolation = (
                    InterpolationMode.BILINEAR if mode == "bilinear" else InterpolationMode.NEAREST
                )
                expected = torch.stack(
                    [
                        rotate(bev[sample], angle, interpolation=interpolation, center=center)
                        for sample in range(bev.shape[0])
                    ]
                )
                if mode == "nearest":
                    assert torch.equal(actual, expected), (center, angle, mode)
                else:
                    torch.testing.assert_close(
                        actual, expected, rtol=1e-4, atol=1e-5, msg=f"{center} {angle}"
                    )


def test_validation_rejects_bad_geometry():
    bev = _bev(2, 3, 5, 4)
    with pytest.raises(ValueError, match="3-D .* or 4-D"):
        temporal_align(bev[0, 0], bev_h=5, bev_w=4)
    with pytest.raises(ValueError, match="bev_h and bev_w are required"):
        temporal_align(bev.permute(0, 2, 3, 1).reshape(2, 20, 3))
    with pytest.raises(ValueError, match="must equal the bev plane size"):
        temporal_align(bev.permute(0, 2, 3, 1).reshape(2, 20, 3), bev_h=4, bev_w=6)
    with pytest.raises(ValueError, match="must match the bev height"):
        temporal_align(bev, bev_h=6)
    with pytest.raises(ValueError, match="must match the bev width"):
        temporal_align(bev, bev_w=7)
    with pytest.raises(ValueError, match="mode must be one of"):
        temporal_align(bev, mode="bicubic")
    with pytest.raises(ValueError, match="center must be a pair"):
        temporal_align(bev, center=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="center must be a pair"):
        temporal_align(bev, center=(float("nan"), 2.0))
    with pytest.raises(TypeError, match="bev must use float32"):
        temporal_align(bev.to(torch.int32))
    with pytest.raises(ValueError, match="bev channels and spatial"):
        temporal_align(torch.zeros(1, 0, 5, 4))


def test_validation_rejects_bad_angle_and_shift():
    bev = _bev(2, 3, 5, 4)
    with pytest.raises(TypeError, match="angle must use float32"):
        temporal_align(bev, torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="angle must be a 1-D tensor"):
        temporal_align(bev, torch.zeros(3, dtype=torch.float32))
    with pytest.raises(ValueError, match="angle must be a 1-D tensor"):
        temporal_align(bev, torch.zeros(2, 1, dtype=torch.float32))
    with pytest.raises(TypeError, match="shift must use int32 or float32"):
        temporal_align(bev, shift=torch.zeros(2, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="shift must be a \\(batch, 2\\) tensor"):
        temporal_align(bev, shift=torch.zeros(2, 3, dtype=torch.float32))
    with pytest.raises(ValueError, match="shift must be a \\(batch, 2\\) tensor"):
        temporal_align(bev, shift=torch.zeros(3, 2, dtype=torch.float32))
    with pytest.raises(ValueError, match="integer shift cannot be combined"):
        temporal_align(
            bev,
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, 2, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="non-trainable"):
        temporal_align(bev, torch.zeros(2, dtype=torch.float32).requires_grad_())
    with pytest.raises(ValueError, match="non-trainable"):
        temporal_align(bev, shift=torch.zeros(2, 2, requires_grad=True))


def test_cpu_auto_uses_reference_and_forced_musa_fails():
    bev = _bev(2, 3, 5, 4)
    clear_dispatch_report()
    output = temporal_align(bev, torch.zeros(2, dtype=torch.float32))
    assert output.shape == bev.shape
    report = get_last_dispatch_report()
    assert report is not None
    assert report.selected_backend == "reference"
    with pytest.raises(RuntimeError, match="requires a MUSA tensor"):
        temporal_align(bev, backend="musa")


@MUSA_REQUIRED
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("angle", ANGLES)
def test_native_matches_reference(mode, angle):
    bev = _bev(2, 3, 5, 4)
    value = torch.full((2,), angle, dtype=torch.float32)
    expected = temporal_align_reference(bev, value, bev_h=5, bev_w=4, mode=mode)
    clear_dispatch_report()
    actual = temporal_align(bev.musa(), value.musa(), bev_h=5, bev_w=4, mode=mode, backend="musa")
    if mode == "nearest":
        assert torch.equal(actual.cpu(), expected)
    else:
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    report = get_last_dispatch_report()
    assert report.selected_backend == "musa"
    assert report.native_fallback is False


@MUSA_REQUIRED
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["channels_last", "channels_first"])
@pytest.mark.parametrize("geometry", ["rotation", "rotation_shift", "integer_shift"])
def test_native_full_matrix(mode, dtype, layout, geometry):
    bev = _bev(2, 5, 6, 9, dtype)
    angle = torch.tensor([0.0, -37.0], dtype=torch.float32)
    shift = torch.tensor([[0.0, 0.0], [1.25, -2.5]], dtype=torch.float32)
    if geometry == "integer_shift":
        if mode != "bilinear":
            pytest.skip("the integer path has a single interpolation semantics")
        angle = None
        shift = torch.tensor([[0, 0], [2, -3]], dtype=torch.int32)
    elif geometry == "rotation_shift":
        angle = torch.tensor([45.0, -10.0], dtype=torch.float32)
    else:
        shift = None
    expected = temporal_align_reference(bev.float(), angle, shift, bev_h=6, bev_w=9, mode=mode).to(
        dtype
    )
    source = bev.permute(0, 2, 3, 1).reshape(2, 54, 5) if layout == "channels_last" else bev
    if layout == "channels_last":
        expected = expected.permute(0, 2, 3, 1).reshape(2, 54, 5)
    device_source = source.musa().requires_grad_()
    actual = temporal_align(
        device_source,
        None if angle is None else angle.musa(),
        None if shift is None else shift.musa(),
        bev_h=6,
        bev_w=9,
        mode=mode,
        backend="musa",
    )
    tolerance = 1e-4 if dtype == torch.float32 else 2e-2
    if mode == "nearest":
        assert torch.equal(actual.cpu(), expected)
    else:
        torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
    weight = _bev(2, 5, 6, 9, dtype, seed=11) * 0.1
    expected_gradient = _naive_backward(weight.float(), angle, shift, mode=mode).to(dtype)
    device_weight = (
        weight.permute(0, 2, 3, 1).reshape(2, 54, 5) if layout == "channels_last" else weight
    )
    gradient = torch.autograd.grad(actual, device_source, device_weight.musa())[0]
    if layout == "channels_last":
        gradient = gradient.reshape(2, 6, 9, 5).permute(0, 3, 1, 2)
    torch.testing.assert_close(gradient.cpu(), expected_gradient, rtol=tolerance, atol=tolerance)
    assert get_last_dispatch_report().selected_backend == "musa"


@MUSA_REQUIRED
def test_native_acceptance_angles_200x200():
    """The JIRA acceptance angles, on a BEV-sized plane.

    The kernel samples directly in pixel space and reproduces the input exactly
    at zero degrees; the CPU reference round-trips through normalised grid
    coordinates and carries about 1e-4 of absolute error at this plane size, so
    the bilinear comparison allows for that reference error.
    """
    generator = torch.Generator().manual_seed(2333)
    channels, side = 8, 200
    bev = torch.randn(2, channels, side, side, generator=generator)
    source = bev.permute(0, 2, 3, 1).reshape(2, side * side, channels)
    for angle in ANGLES:
        angle_tensor = torch.full((2,), angle, dtype=torch.float32)
        for mode in ("bilinear", "nearest"):
            expected = temporal_align_reference(
                bev, angle_tensor, bev_h=side, bev_w=side, mode=mode
            )
            actual = temporal_align(
                source.musa(),
                angle_tensor.musa(),
                bev_h=side,
                bev_w=side,
                mode=mode,
                backend="musa",
            )
            restored = actual.cpu().reshape(2, side, side, channels).permute(0, 3, 1, 2)
            if mode == "nearest":
                assert torch.equal(restored, expected), angle
            else:
                torch.testing.assert_close(restored, expected, rtol=1e-4, atol=3e-4, msg=str(angle))
            if angle == 0.0:
                assert torch.equal(restored, bev), mode


@MUSA_REQUIRED
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_native_identity_is_exact(mode):
    bev = _bev(2, 3, 200, 200)
    source = bev.permute(0, 2, 3, 1).reshape(2, 200 * 200, 3)
    actual = temporal_align(
        source.musa(),
        torch.zeros(2, dtype=torch.float32, device="musa"),
        bev_h=200,
        bev_w=200,
        mode=mode,
        backend="musa",
    )
    restored = actual.cpu().reshape(2, 200, 200, 3).permute(0, 3, 1, 2)
    assert torch.equal(restored, bev)


@MUSA_REQUIRED
def test_native_acceptance_non_square_and_out_of_bounds():
    bev = _bev(2, 4, 200, 100)
    source = bev.permute(0, 2, 3, 1).reshape(2, 200 * 100, 4)
    cases = [
        {"angle": torch.tensor([0.0, 0.0]), "shift": None},
        {"angle": None, "shift": torch.tensor([[0, 0], [40, -70]], dtype=torch.int32)},
        {"angle": torch.tensor([180.0, -180.0]), "shift": None},
        {"angle": None, "shift": torch.tensor([[9_999, -9_999], [0, 0]], dtype=torch.int32)},
    ]
    for case in cases:
        expected = temporal_align_reference(bev, case["angle"], case["shift"], bev_h=200, bev_w=100)
        actual = temporal_align(
            source.musa(),
            None if case["angle"] is None else case["angle"].musa(),
            None if case["shift"] is None else case["shift"].musa(),
            bev_h=200,
            bev_w=100,
            backend="musa",
        )
        restored = actual.cpu().reshape(2, 200, 100, 4).permute(0, 3, 1, 2)
        torch.testing.assert_close(restored, expected, rtol=1e-4, atol=2e-4)
        if case["shift"] is not None and int(case["shift"][0, 0].abs()) > 1_000:
            # The first sample is shifted entirely off the plane.
            assert torch.count_nonzero(restored[0]) == 0


@MUSA_REQUIRED
def test_native_handles_empty_batch_and_single_pixel():
    empty = torch.zeros(0, 3, 5, 4, dtype=torch.float32, device="musa")
    assert temporal_align(empty, bev_h=5, bev_w=4, backend="musa").shape == (0, 3, 5, 4)
    single = torch.ones(1, 2, 1, 1, dtype=torch.float32, device="musa")
    angle = torch.zeros(1, dtype=torch.float32, device="musa")
    actual = temporal_align(single, angle, mode="nearest", backend="musa")
    torch.testing.assert_close(actual.cpu(), single.cpu())


@MUSA_REQUIRED
def test_native_non_contiguous_input():
    bev = _bev(2, 3, 5, 4).transpose(2, 3)
    expected = temporal_align_reference(bev, bev_h=4, bev_w=5)
    actual = temporal_align(bev.musa(), bev_h=4, bev_w=5, backend="musa")
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)


@MUSA_REQUIRED
def test_planar_layout_keeps_the_input_layout():
    """A float32 channels-first call runs the planar kernels, not a conversion."""
    bev = _bev(2, 8, 200, 200)
    clear_dispatch_report()
    actual = temporal_align(
        bev.musa(), torch.full((2,), 17.0, dtype=torch.float32, device="musa"), backend="musa"
    )
    assert "planar" in get_last_dispatch_report().reason
    expected = temporal_align_reference(
        bev, torch.full((2,), 17.0, dtype=torch.float32), bev_h=200, bev_w=200
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=3e-4)


@MUSA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("geometry", ["rotation", "integer_shift"])
def test_planar_selection_by_dtype_and_shift(dtype, geometry):
    """Only a float32 rotation keeps the layout; other cases take the packed path."""
    bev = _bev(2, 5, 6, 9, dtype)
    if geometry == "integer_shift":
        angle = None
        shift = torch.tensor([[0, 0], [2, -3]], dtype=torch.int32)
    else:
        angle = torch.tensor([45.0, -10.0], dtype=torch.float32)
        shift = None
    clear_dispatch_report()
    temporal_align(
        bev.musa(),
        None if angle is None else angle.musa(),
        None if shift is None else shift.musa(),
        bev_h=6,
        bev_w=9,
        backend="musa",
    )
    expected_kernel = "planar" if dtype == torch.float32 and geometry == "rotation" else "packed"
    assert expected_kernel in get_last_dispatch_report().reason


@MUSA_REQUIRED
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_planar_matches_packed_layout(mode):
    """The two layouts of one plane sample identically."""
    bev = _bev(2, 5, 37, 6)
    angle = torch.tensor([33.0, -70.0], dtype=torch.float32)
    shift = torch.tensor([[1.25, -2.5], [0.0, 0.5]], dtype=torch.float32)
    packed_source = bev.permute(0, 2, 3, 1).reshape(2, 37 * 6, 5)
    planar = temporal_align(
        bev.musa(),
        angle.musa(),
        shift.musa(),
        bev_h=37,
        bev_w=6,
        mode=mode,
        backend="musa",
    )
    packed = temporal_align(
        packed_source.musa(),
        angle.musa(),
        shift.musa(),
        bev_h=37,
        bev_w=6,
        mode=mode,
        backend="musa",
    )
    restored = packed.cpu().reshape(2, 37, 6, 5).permute(0, 3, 1, 2)
    torch.testing.assert_close(planar.cpu(), restored, rtol=1e-6, atol=1e-6)


@MUSA_REQUIRED
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("width", [37, 16, 1])
def test_planar_unaligned_width(mode, width):
    """Widths that do not fill a vector or a whole x tile still hit the reference."""
    bev = _bev(2, 7, 5, width)
    angle = torch.tensor([37.0, -120.0], dtype=torch.float32)
    expected = temporal_align_reference(bev, angle, bev_h=5, bev_w=width, mode=mode)
    actual = temporal_align(
        bev.musa(), angle.musa(), bev_h=5, bev_w=width, mode=mode, backend="musa"
    )
    if mode == "nearest":
        assert torch.equal(actual.cpu(), expected)
    else:
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)


@MUSA_REQUIRED
def test_planar_gradient_matches_packed_layout():
    """The gradient of a planar call matches the gradient of the packed layout."""
    bev = _bev(2, 5, 37, 6)
    angle = torch.tensor([33.0, -70.0], dtype=torch.float32)
    packed_source = bev.permute(0, 2, 3, 1).reshape(2, 37 * 6, 5)
    weight = _bev(2, 5, 37, 6, seed=11) * 0.1
    planar_source = bev.musa().requires_grad_()
    packed_device = packed_source.musa().requires_grad_()
    planar_out = temporal_align(planar_source, angle.musa(), bev_h=37, bev_w=6, backend="musa")
    packed_out = temporal_align(packed_device, angle.musa(), bev_h=37, bev_w=6, backend="musa")
    packed_weight = weight.permute(0, 2, 3, 1).reshape(2, 37 * 6, 5).musa()
    planar_grad = torch.autograd.grad(planar_out, planar_source, weight.musa())[0]
    packed_grad = torch.autograd.grad(packed_out, packed_device, packed_weight)[0]
    restored = packed_grad.cpu().reshape(2, 37, 6, 5).permute(0, 3, 1, 2)
    torch.testing.assert_close(planar_grad.cpu(), restored, rtol=1e-5, atol=1e-5)
