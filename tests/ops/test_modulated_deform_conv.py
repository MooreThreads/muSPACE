"""Correctness tests for the modulated deformable convolution operator.

The differentiable PyTorch model in this file is the golden reference. It mirrors the
kernel semantics exactly -- a sampling position outside ``(-1, H) x (-1, W)`` contributes
nothing, and every bilinear corner outside ``[0, H-1] x [0, W-1]`` is dropped from the
interpolation -- and every gradient is derived from it by autograd, so the native
backward pass is checked against the derivative of the reference forward pass rather
than against a second hand-written implementation.
"""

import sys
import types
from typing import Sequence

import pytest
import torch
from torch.nn.modules.utils import _pair

from mu_space import clear_dispatch_report, get_last_dispatch_report
from mu_space.ops.modulated_deform_conv import (
    ModulatedDeformConv2d,
    ModulatedDeformConv2dPack,
    _mudnn_tf32,
    modulated_deform_conv2d,
    modulated_deform_conv2d_pack,
)
from mu_space.patcher import ModulatedDeformConv2dPatcher, PatchError

musa_only = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------


def _reference_columns(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    deform_groups: int,
) -> torch.Tensor:
    """Deformable im2col: ``[B, C * kH * kW, Ho * Wo]``, channel major, tap minor."""
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    batch, channels, height, width = input.shape
    height_col, width_col = offset.shape[2], offset.shape[3]
    taps = kernel_h * kernel_w
    positions = height_col * width_col
    channels_per_group = channels // deform_groups

    device = input.device
    rows = torch.arange(height_col, device=device, dtype=input.dtype).view(height_col, 1)
    cols = torch.arange(width_col, device=device, dtype=input.dtype).view(1, width_col)
    h_in = (rows * stride_h - pad_h).expand(height_col, width_col).reshape(positions)
    w_in = (cols * stride_w - pad_w).expand(height_col, width_col).reshape(positions)

    offset_view = offset.view(batch, deform_groups, taps, 2, positions)
    mask_view = mask.view(batch, deform_groups, taps, positions)
    flat_input = input.view(batch, deform_groups, channels_per_group, height * width)

    tap_columns = []
    for tap in range(taps):
        ki, kj = tap // kernel_w, tap % kernel_w
        h_im = h_in.view(1, 1, positions) + ki * dilation_h + offset_view[:, :, tap, 0]
        w_im = w_in.view(1, 1, positions) + kj * dilation_w + offset_view[:, :, tap, 1]
        inside = (h_im > -1) & (w_im > -1) & (h_im < height) & (w_im < width)

        h_low, w_low = torch.floor(h_im), torch.floor(w_im)
        lh, lw = h_im - h_low, w_im - w_low
        hh, hw = 1.0 - lh, 1.0 - lw
        h_low_i, w_low_i = h_low.long(), w_low.long()

        value = None
        for dy, dx, corner_weight in (
            (0, 0, hh * hw),
            (0, 1, hh * lw),
            (1, 0, lh * hw),
            (1, 1, lh * lw),
        ):
            ys, xs = h_low_i + dy, w_low_i + dx
            keep = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width) & inside
            index = ys.clamp(0, height - 1) * width + xs.clamp(0, width - 1)
            gathered = flat_input.gather(
                3, index.unsqueeze(2).expand(-1, -1, channels_per_group, -1)
            )
            weighted = gathered * (corner_weight * keep).unsqueeze(2)
            value = weighted if value is None else value + weighted

        tap_columns.append(value * mask_view[:, :, tap].unsqueeze(2))

    columns = torch.stack(tap_columns, dim=3)  # [B, dg, cpg, T, P]
    return columns.reshape(batch, channels * taps, positions)


def _reference_dcn(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    deform_groups: int = 1,
) -> torch.Tensor:
    stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    batch, channels_out = input.shape[0], weight.shape[0]
    height_col, width_col = offset.shape[2], offset.shape[3]

    columns = _reference_columns(
        input, offset, mask, (kernel_h, kernel_w), stride, padding, dilation, deform_groups
    )
    in_per_group = weight.shape[1] * kernel_h * kernel_w
    out_per_group = channels_out // groups
    columns = columns.view(batch, groups, in_per_group, height_col * width_col)
    weight_g = weight.reshape(groups, out_per_group, in_per_group)
    output = torch.matmul(weight_g.unsqueeze(0), columns)
    output = output.reshape(batch, channels_out, height_col, width_col)
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output


def _reference_pack(
    input: torch.Tensor,
    offset_weight: torch.Tensor,
    offset_bias: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=1,
    dilation=1,
    groups: int = 1,
    deform_groups: int = 1,
) -> torch.Tensor:
    stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)
    raw = torch.nn.functional.conv2d(input, offset_weight, offset_bias, stride, padding, dilation)
    # The first 2 * dg * kH * kW channels are the offsets, laid out per deform group as
    # (h, w) interleaved per tap; the rest is the pre-sigmoid mask. See
    # test_offset_channel_layout_contract, which pins this down without going through
    # either this model or the kernel's own indexing.
    offset_channels = 2 * deform_groups * weight.shape[2] * weight.shape[3]
    return _reference_dcn(
        input,
        raw[:, :offset_channels],
        torch.sigmoid(raw[:, offset_channels:]),
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
    )


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _out_size(size, pad, dilation, kernel, stride):
    return (size + 2 * pad - (dilation * (kernel - 1) + 1)) // stride + 1


def _dcn_fixture(
    device="cpu",
    dtype=torch.float32,
    *,
    batch=2,
    channels=32,
    channels_out=16,
    height=7,
    width=9,
    kernel=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    deform_groups=1,
):
    generator = torch.Generator(device="cpu").manual_seed(20260729)
    taps = kernel * kernel
    height_col = _out_size(height, padding, dilation, kernel, stride)
    width_col = _out_size(width, padding, dilation, kernel, stride)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=generator) * scale).to(device=device, dtype=dtype)

    return (
        rand(batch, channels, height, width),
        # Offsets of this magnitude put a healthy share of samples outside the image,
        # which is where the boundary handling of the kernel actually gets exercised.
        rand(batch, deform_groups * 2 * taps, height_col, width_col, scale=1.5),
        torch.sigmoid(rand(batch, deform_groups * taps, height_col, width_col)),
        rand(channels_out, channels // groups, kernel, kernel, scale=0.2),
        rand(channels_out, scale=0.1),
    )


def _pack_fixture(device="musa", dtype=torch.float32, *, batch=2, channels=32, channels_out=32):
    generator = torch.Generator(device="cpu").manual_seed(20260730)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=generator) * scale).to(device=device, dtype=dtype)

    return (
        rand(batch, channels, 7, 9),
        rand(27, channels, 3, 3, scale=0.2),
        rand(27, scale=0.5),
        rand(channels_out, channels, 3, 3, scale=0.1),
        rand(channels_out, scale=0.1),
    )


def _grads(fn, tensors, *, seed=7):
    """Run ``fn`` on clones of ``tensors`` and return (output, gradients).

    The whole computation, including the lazy autograd backward, runs under
    ``_mudnn_tf32`` so the reference GEMMs use the same strict-fp32 semantics as
    the native kernels with torch_musa 2.7.1, where the batched matmuls of the
    backward would otherwise execute in TF32 by default. With torch_musa 2.9.1,
    the regular ``torch.backends.musa.matmul`` switch is used instead.
    """
    leaves = [t.clone().detach().requires_grad_(True) for t in tensors]
    with _mudnn_tf32():
        output = fn(*leaves)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weights = torch.randn(*output.shape, generator=generator).to(
            device=output.device, dtype=output.dtype
        )
        (output * weights).sum().backward()
    return output.detach(), [leaf.grad for leaf in leaves]


def _assert_close(actual, expected, *, rtol, atol, name):
    """Compare with an absolute tolerance that also tracks the magnitude of the tensor.

    Gradients here are reductions over every sampling position, and the kernel sums
    them in a different order than the reference does, so an element that lands near
    zero through cancellation can differ well beyond its own magnitude while the
    tensor as a whole is accurate. The extra slack is a fixed fraction of the largest
    value in the tensor. ``rtol == atol == 0`` asks for bit-exactness and is left alone.
    """
    actual = actual.detach().float()
    expected = expected.detach().float()
    if rtol or atol:
        atol = atol + 1e-5 * expected.abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=lambda m: f"{name}: {m}")


# ---------------------------------------------------------------------------
# Host-side behaviour
# ---------------------------------------------------------------------------


def test_cpu_tensors_are_rejected_with_a_dispatch_report():
    input, offset, mask, weight, bias = _dcn_fixture()
    clear_dispatch_report()
    with pytest.raises(RuntimeError, match="requires MUSA tensors"):
        modulated_deform_conv2d(input, offset, mask, weight, bias, 1, 1, 1)
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "modulated_deform_conv2d"
    assert report.selected_backend == "error"


def test_unknown_backend_is_rejected():
    input, offset, mask, weight, bias = _dcn_fixture()
    with pytest.raises(RuntimeError, match="unknown backend"):
        modulated_deform_conv2d(input, offset, mask, weight, bias, 1, 1, 1, backend="cudnn")


def test_packed_cpu_tensors_are_rejected_with_a_dispatch_report():
    input, offset_weight, offset_bias, weight, bias = _pack_fixture(device="cpu")
    clear_dispatch_report()
    with pytest.raises(RuntimeError, match="requires MUSA tensors"):
        modulated_deform_conv2d_pack(input, offset_weight, offset_bias, weight, bias)
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "modulated_deform_conv2d"
    assert report.selected_backend == "error"


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"input_3d": True}, "expected a 4D input tensor"),
        ({"weight_3d": True}, "expected a 4D weight tensor"),
        ({"offset_channels": 4}, "offset must have"),
        ({"mask_channels": 4}, "mask must have"),
        ({"groups": 5}, "divisible by groups"),
        ({"deform_groups": 5}, "divisible by deform_groups"),
    ],
)
def test_input_validation(mutation, message):
    input, offset, mask, weight, bias = _dcn_fixture()
    groups = mutation.get("groups", 1)
    deform_groups = mutation.get("deform_groups", 1)
    if mutation.get("input_3d"):
        input = input[0]
    if mutation.get("weight_3d"):
        weight = weight[0]
    if "offset_channels" in mutation:
        offset = offset[:, : mutation["offset_channels"]]
    if "mask_channels" in mutation:
        mask = mask[:, : mutation["mask_channels"]]
    with pytest.raises(ValueError, match=message):
        modulated_deform_conv2d(
            input,
            offset,
            mask,
            weight,
            bias,
            1,
            1,
            1,
            groups=groups,
            deform_groups=deform_groups,
        )


class _FakeRegistry:
    """Minimal stand-in for mmcv's ``CONV_LAYERS`` registry."""

    def __init__(self, entries):
        self._module_dict = dict(entries)

    def get(self, key):
        return self._module_dict.get(key)

    def register_module(self, name=None, force=False, module=None):
        if name in self._module_dict and not force:
            raise KeyError(f"{name} is already registered")
        self._module_dict[name] = module


def _install_fake_mmcv(monkeypatch):
    """Install the mmcv modules the DCNv2 patcher reaches into."""

    class MMCVModulatedDeformConv2dPack:
        pass

    registry_module = types.ModuleType("mmcv.cnn.bricks.registry")
    registry_module.CONV_LAYERS = _FakeRegistry({"DCNv2": MMCVModulatedDeformConv2dPack})
    target = types.ModuleType("mmcv.ops.modulated_deform_conv")
    target.ModulatedDeformConv2dPack = MMCVModulatedDeformConv2dPack
    ops = types.ModuleType("mmcv.ops")
    ops.ModulatedDeformConv2dPack = MMCVModulatedDeformConv2dPack

    for name, module in (
        ("mmcv.cnn.bricks.registry", registry_module),
        ("mmcv.ops.modulated_deform_conv", target),
        ("mmcv.ops", ops),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return registry_module.CONV_LAYERS, target, ops, MMCVModulatedDeformConv2dPack


def test_patcher_replaces_the_mmcv_dcnv2_layer_and_restores(monkeypatch):
    registry, target, ops, original = _install_fake_mmcv(monkeypatch)
    patcher = ModulatedDeformConv2dPatcher()

    report = patcher.apply()

    assert report.action == "applied"
    assert report.patched_modules == (
        "mmcv.cnn.bricks.registry",
        "mmcv.ops.modulated_deform_conv",
        "mmcv.ops",
    )
    # build_conv_layer(dict(type='DCNv2', ...)) resolves through the registry, and a
    # direct import resolves through the module attributes; both must move.
    assert registry.get("DCNv2") is ModulatedDeformConv2dPack
    assert target.ModulatedDeformConv2dPack is ModulatedDeformConv2dPack
    assert ops.ModulatedDeformConv2dPack is ModulatedDeformConv2dPack

    restored = patcher.restore()
    assert restored.action == "restored"
    assert registry.get("DCNv2") is original
    assert target.ModulatedDeformConv2dPack is original
    assert ops.ModulatedDeformConv2dPack is original


def test_patcher_is_idempotent(monkeypatch):
    registry, _, _, original = _install_fake_mmcv(monkeypatch)
    patcher = ModulatedDeformConv2dPatcher()

    patcher.apply()
    second = patcher.apply()

    assert second.action == "already-applied"
    assert registry.get("DCNv2") is ModulatedDeformConv2dPack
    patcher.restore()
    assert registry.get("DCNv2") is original


def test_patcher_reports_a_missing_openmmlab_stack(monkeypatch):
    monkeypatch.setitem(sys.modules, "mmcv.ops.modulated_deform_conv", None)
    with pytest.raises(PatchError, match="OpenMMLab stack"):
        ModulatedDeformConv2dPatcher().apply()


def test_pack_module_loads_legacy_mmcv_checkpoints():
    module = ModulatedDeformConv2dPack(16, 16, 3, padding=1)
    state = {
        "dcn_offset.weight": torch.zeros_like(module.conv_offset.weight),
        "dcn_offset.bias": torch.zeros_like(module.conv_offset.bias),
        "dcn.weight": torch.zeros_like(module.weight),
        "dcn.bias": torch.zeros_like(module.bias),
    }
    missing, unexpected = [], []
    module._load_from_state_dict(state, "dcn.", {}, True, missing, unexpected, [])
    assert not missing and not unexpected


# ---------------------------------------------------------------------------
# Native accuracy (MUSA)
# ---------------------------------------------------------------------------


@musa_only
@pytest.mark.parametrize(
    "config",
    [
        pytest.param({}, id="base"),
        pytest.param({"groups": 2}, id="groups2"),
        pytest.param({"deform_groups": 2}, id="deform_groups2"),
        pytest.param({"stride": 2}, id="stride2"),
        pytest.param({"kernel": 1, "padding": 0}, id="kernel1"),
        pytest.param({"dilation": 2, "padding": 2}, id="dilation2"),
        pytest.param({"channels": 24, "channels_out": 24}, id="channels24"),
    ],
)
def test_native_nchw_matches_reference(config):
    """Forward and every gradient of the general path against the reference."""
    tensors = _dcn_fixture(device="musa", **config)
    kwargs = dict(
        stride=config.get("stride", 1),
        padding=config.get("padding", 1),
        dilation=config.get("dilation", 1),
        groups=config.get("groups", 1),
        deform_groups=config.get("deform_groups", 1),
    )

    expected, expected_grads = _grads(lambda *t: _reference_dcn(*t, **kwargs), tensors)
    actual, actual_grads = _grads(
        lambda *t: modulated_deform_conv2d(*t, backend="musa", **kwargs), tensors
    )

    _assert_close(actual, expected, rtol=2e-5, atol=2e-5, name="output")
    for name, got, want in zip(
        ("grad_input", "grad_offset", "grad_mask", "grad_weight", "grad_bias"),
        actual_grads,
        expected_grads,
    ):
        _assert_close(got, want, rtol=2e-5, atol=3e-5, name=name)


@musa_only
def test_native_nchw_reports_the_musa_backend():
    tensors = _dcn_fixture(device="musa")
    clear_dispatch_report()
    modulated_deform_conv2d(*tensors, stride=1, padding=1, dilation=1)
    report = get_last_dispatch_report()
    assert report is not None
    assert report.selected_backend == "musa"
    assert report.native_available


@musa_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_native_nchw_dtypes(dtype):
    tensors = _dcn_fixture(device="musa", dtype=dtype)
    reference_inputs = [t.float() for t in tensors]
    expected = _reference_dcn(*reference_inputs, stride=1, padding=1, dilation=1)
    actual = modulated_deform_conv2d(*tensors, stride=1, padding=1, dilation=1, backend="musa")
    assert actual.dtype == dtype
    tolerance = dict(rtol=2e-5, atol=2e-5) if dtype == torch.float32 else dict(rtol=4e-3, atol=4e-3)
    _assert_close(actual, expected, name="output", **tolerance)


@musa_only
@pytest.mark.parametrize("channels_last", [False, True])
def test_fused_pack_matches_reference(monkeypatch, channels_last):
    """The offset-conv-fused NHWC path, forward and backward, in strict float32."""
    monkeypatch.setenv("MUSPACE_DCN_FP32_TF32", "0")
    tensors = _pack_fixture()
    if channels_last:
        tensors = (tensors[0].to(memory_format=torch.channels_last),) + tensors[1:]

    expected, expected_grads = _grads(lambda *t: _reference_pack(*t, padding=1), tensors)
    actual, actual_grads = _grads(lambda *t: modulated_deform_conv2d_pack(*t, padding=1), tensors)

    assert actual.is_contiguous(memory_format=torch.channels_last) == channels_last
    _assert_close(actual, expected, rtol=2e-5, atol=2e-5, name="output")
    for name, got, want in zip(
        ("grad_input", "grad_offset_weight", "grad_offset_bias", "grad_weight", "grad_bias"),
        actual_grads,
        expected_grads,
    ):
        _assert_close(got, want, rtol=1e-4, atol=1e-4, name=name)


@musa_only
def test_fused_pack_matches_reference_with_tf32(monkeypatch):
    """With TF32 explicitly enabled, check the looser tolerance holds."""
    monkeypatch.setenv("MUSPACE_DCN_FP32_TF32", "1")
    tensors = _pack_fixture()
    expected = _reference_pack(*tensors, padding=1)
    actual = modulated_deform_conv2d_pack(*tensors, padding=1)
    _assert_close(actual, expected, rtol=5e-3, atol=5e-3, name="output")


@musa_only
def test_fused_pack_half_precision(monkeypatch):
    monkeypatch.setenv("MUSPACE_DCN_FP32_TF32", "0")
    tensors = _pack_fixture(dtype=torch.float16)
    expected = _reference_pack(*[t.float() for t in tensors], padding=1)
    actual = modulated_deform_conv2d_pack(*tensors, padding=1)
    assert actual.dtype == torch.float16
    _assert_close(actual, expected, rtol=6e-3, atol=6e-3, name="output")


@musa_only
def test_fused_pack_agrees_with_the_general_path(monkeypatch):
    """Both native paths must produce the same operator, not just the same shape."""
    monkeypatch.setenv("MUSPACE_DCN_FP32_TF32", "0")
    input, offset_weight, offset_bias, weight, bias = _pack_fixture()

    def general(x, w_off, b_off, w, b):
        raw = torch.nn.functional.conv2d(x, w_off, b_off, 1, 1, 1)
        offset_y, offset_x, raw_mask = torch.chunk(raw, 3, dim=1)
        return modulated_deform_conv2d(
            x,
            torch.cat((offset_y, offset_x), dim=1),
            torch.sigmoid(raw_mask),
            w,
            b,
            stride=1,
            padding=1,
            dilation=1,
            backend="musa",
        )

    tensors = (input, offset_weight, offset_bias, weight, bias)
    expected, expected_grads = _grads(general, tensors)
    actual, actual_grads = _grads(lambda *t: modulated_deform_conv2d_pack(*t, padding=1), tensors)

    _assert_close(actual, expected, rtol=2e-5, atol=2e-5, name="output")
    for name, got, want in zip(
        ("grad_input", "grad_offset_weight", "grad_offset_bias", "grad_weight", "grad_bias"),
        actual_grads,
        expected_grads,
    ):
        _assert_close(got, want, rtol=1e-4, atol=1e-4, name=name)


@musa_only
def test_pack_falls_back_to_the_general_path(monkeypatch):
    """Configurations outside the fused domain still run, through the NCHW kernels."""
    monkeypatch.setenv("MUSPACE_DCN_FP32_TF32", "0")
    generator = torch.Generator(device="cpu").manual_seed(11)
    input = torch.randn(2, 32, 7, 9, generator=generator).musa()
    offset_weight = (torch.randn(54, 32, 3, 3, generator=generator) * 0.2).musa()
    offset_bias = (torch.randn(54, generator=generator) * 0.5).musa()
    weight = (torch.randn(32, 32, 3, 3, generator=generator) * 0.1).musa()
    bias = (torch.randn(32, generator=generator) * 0.1).musa()
    tensors = (input, offset_weight, offset_bias, weight, bias)
    kwargs = dict(padding=1, deform_groups=2)

    expected, expected_grads = _grads(lambda *t: _reference_pack(*t, **kwargs), tensors)
    actual, actual_grads = _grads(lambda *t: modulated_deform_conv2d_pack(*t, **kwargs), tensors)

    _assert_close(actual, expected, rtol=2e-5, atol=2e-5, name="output")
    for name, got, want in zip(
        ("grad_input", "grad_offset_weight", "grad_offset_bias", "grad_weight", "grad_bias"),
        actual_grads,
        expected_grads,
    ):
        _assert_close(got, want, rtol=1e-4, atol=1e-4, name=name)


@musa_only
def test_saved_and_recomputed_columns_agree(monkeypatch):
    """Recomputing the columns in backward must not change the operator.

    The forward pass is deterministic and must match bit for bit. The gradients only
    have to match closely: col2im accumulates into grad_input, grad_offset and
    grad_mask with atomics, so the summation order -- and therefore the last bits --
    varies between runs no matter how the columns were obtained.
    """
    tensors = _pack_fixture()
    monkeypatch.setenv("MUSPACE_DCN_RECOMPUTE_COLUMNS", "1")
    recomputed, recomputed_grads = _grads(
        lambda *t: modulated_deform_conv2d_pack(*t, padding=1), tensors
    )
    monkeypatch.setenv("MUSPACE_DCN_RECOMPUTE_COLUMNS", "0")
    saved, saved_grads = _grads(lambda *t: modulated_deform_conv2d_pack(*t, padding=1), tensors)

    _assert_close(recomputed, saved, rtol=0, atol=0, name="output")
    for name, got, want in zip(
        ("grad_input", "grad_offset_weight", "grad_offset_bias", "grad_weight", "grad_bias"),
        recomputed_grads,
        saved_grads,
    ):
        _assert_close(got, want, rtol=1e-5, atol=1e-6, name=name)


@musa_only
@pytest.mark.parametrize("deform_groups", [1, 2])
def test_offset_channel_layout_contract(deform_groups):
    """Pin which offset channel moves a tap in which direction.

    The offset tensor holds, per deform group, the (h, w) pair of each tap interleaved:
    channel ``g * 2T + 2t`` is the h offset of tap ``t`` in group ``g`` and ``+ 1`` is its
    w offset. mmcv's kernels use the same layout, which is what makes its checkpoints
    load unchanged, and both muSPACE kernels index the offset-conv output this way.

    This measures the layout physically instead of asserting it against the reference
    model: the input is constant along a row and increases down the column, so an h
    offset changes the sampled value and a w offset does not. It therefore fails if the
    kernel, the reference, or the packed wrapper ever disagree about the layout.
    """
    batch, channels, height, width = 1, 32, 10, 10
    kernel = 3
    taps = kernel * kernel
    centre_tap = taps // 2
    shift = 0.5

    # value == row index, so only a vertical shift changes what is sampled
    x = (
        torch.arange(height, dtype=torch.float32)
        .view(1, 1, height, 1)
        .expand(batch, channels, height, width)
        .contiguous()
        .musa()
    )
    weight = torch.zeros(1, channels, kernel, kernel, device="musa")
    weight[0, :, 1, 1] = 1.0 / channels  # keep the centre tap only
    mask = torch.ones(batch, deform_groups * taps, height, width, device="musa")

    def sample(channel):
        offset = torch.zeros(batch, deform_groups * 2 * taps, height, width, device="musa")
        if channel is not None:
            offset[:, channel] = shift
        return modulated_deform_conv2d(
            x, offset, mask, weight, None, 1, 1, 1, deform_groups=deform_groups, backend="musa"
        )

    # Interior only: a shifted sample near the border leaves the image and is zeroed,
    # which would move the output for either component.
    base = sample(None)

    def moved(channel):
        return (sample(channel) - base)[..., 1 : height - 1, 1 : width - 1].abs().max().item()

    # Each deform group drives channels / deform_groups input channels, so a single
    # group's shift moves the averaged output by that fraction.
    expected = shift * (1.0 / deform_groups)
    for group in range(deform_groups):
        h_channel = group * 2 * taps + 2 * centre_tap
        assert moved(h_channel) == pytest.approx(expected, abs=1e-4), (
            f"channel {h_channel} should be the h offset of tap {centre_tap} in group {group}"
        )
        assert moved(h_channel + 1) == pytest.approx(0.0, abs=1e-6), (
            f"channel {h_channel + 1} should be the w offset, which cannot move this input"
        )
        # A block layout would put the h offset of the centre tap here instead.
        assert moved(group * 2 * taps + centre_tap) == pytest.approx(0.0, abs=1e-6)


@musa_only
def test_layer_built_the_way_mmdet_builds_it(monkeypatch):
    """The whole integration chain, from the model config entry to a MUSA gradient.

    mmdet's ResNet bottleneck calls ``build_conv_layer(dict(type='DCNv2', deform_groups=1),
    planes, planes, kernel_size=3, stride=..., padding=dilation, dilation=dilation,
    bias=False)``, which is a registry lookup followed by a call with those arguments.
    Afterwards ``ResNet.init_weights`` zero-initialises ``conv_offset`` if the layer has
    one. This resolves the layer through the patched registry, exactly as mmdet would.
    """
    registry, _, _, _ = _install_fake_mmcv(monkeypatch)
    patcher = ModulatedDeformConv2dPatcher()
    patcher.apply()

    dcn_cfg = dict(type="DCNv2", deform_groups=1)
    layer_cfg = {key: value for key, value in dcn_cfg.items() if key != "type"}
    layer_cls = registry.get(dcn_cfg["type"])
    assert layer_cls is ModulatedDeformConv2dPack

    layer = layer_cls(
        64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, **layer_cfg
    ).musa()
    assert layer.bias is None
    assert hasattr(layer, "conv_offset")

    generator = torch.Generator(device="cpu").manual_seed(5)
    x = torch.randn(2, 64, 7, 9, generator=generator).musa().requires_grad_(True)
    output = layer(x)
    output.sum().backward()

    assert output.shape == (2, 64, 7, 9)
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.conv_offset.weight.grad is not None
    assert layer.conv_offset.bias.grad is not None


@musa_only
def test_modules_run_on_musa():
    """With zero offsets the modules degenerate to a plain convolution.

    The comparison runs against muDNN's ``conv2d``, which computes in a reduced
    precision mode, so the tolerance here is muDNN's and not the operator's; the
    tight accuracy checks are the reference comparisons above.
    """
    convolution_tolerance = dict(rtol=5e-3, atol=5e-3)
    generator = torch.Generator(device="cpu").manual_seed(3)
    x = torch.randn(2, 32, 7, 9, generator=generator).musa()

    packed = ModulatedDeformConv2dPack(32, 32, 3, padding=1).musa()
    output = packed(x)
    assert output.shape == (2, 32, 7, 9)
    # conv_offset is zero-initialised, so the packed module starts as a plain
    # convolution with a constant 0.5 modulation (the bias is added after it).
    reference = torch.nn.functional.conv2d(
        x, packed.weight, None, 1, 1, 1
    ) * 0.5 + packed.bias.view(1, -1, 1, 1)
    _assert_close(output, reference, name="packed output", **convolution_tolerance)

    plain = ModulatedDeformConv2d(32, 16, 3, padding=1).musa()
    offset = torch.zeros(2, 18, 7, 9, device="musa")
    mask = torch.ones(2, 9, 7, 9, device="musa")
    plain_output = plain(x, offset, mask)
    plain_reference = torch.nn.functional.conv2d(x, plain.weight, plain.bias, 1, 1, 1)
    _assert_close(plain_output, plain_reference, name="plain output", **convolution_tolerance)
