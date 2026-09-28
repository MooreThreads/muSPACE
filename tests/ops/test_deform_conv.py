import pytest
import torch

from mu_space.ops import DeformConv2d, DeformConv2dPack, modulated_deform_conv2d


def test_standard_deform_conv_pack_matches_mmcv_parameter_surface():
    layer = DeformConv2dPack(
        8,
        16,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        groups=4,
        deform_groups=2,
    )
    assert layer.weight.shape == (16, 2, 3, 3)
    assert layer.conv_offset.weight.shape == (36, 8, 3, 3)
    assert layer.conv_offset.bias.shape == (36,)
    assert torch.count_nonzero(layer.conv_offset.weight) == 0
    assert torch.count_nonzero(layer.conv_offset.bias) == 0


def test_standard_deform_conv_rejects_bias():
    with pytest.raises(ValueError, match="does not support bias"):
        DeformConv2dPack(4, 4, 3, bias=True)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_standard_deform_conv_mask_one_matches_modulated_reference():
    torch.manual_seed(2333)
    layer = DeformConv2d(4, 4, 3, padding=1).to("musa")
    x = torch.randn(1, 4, 8, 8, device="musa", requires_grad=True)
    offset = torch.randn(1, 18, 8, 8, device="musa", requires_grad=True)
    actual = layer(x, offset)
    expected = modulated_deform_conv2d(
        x,
        offset,
        torch.ones(1, 9, 8, 8, device="musa"),
        layer.weight,
        None,
        (1, 1),
        (1, 1),
        (1, 1),
        1,
        1,
        backend="musa",
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    actual.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
