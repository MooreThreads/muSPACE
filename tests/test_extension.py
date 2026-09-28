import pytest
import torch


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_add_one_extension_on_musa():
    from mu_space.runtime.native import native_function

    value = torch.arange(4, dtype=torch.float32, device="musa")
    actual = torch.empty_like(value)
    native_function("add_one")(value, actual)
    torch.testing.assert_close(actual.cpu(), torch.arange(1, 5, dtype=torch.float32))


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_aot_module_exports_core_operator_entrypoints():
    from mu_space.runtime.native import native_function

    for name in (
        "bev_pool_forward",
        "bev_pool_backward",
        "bev_pool_v2_forward",
        "flash_bevpool_forward",
        "flash_bevpool_sca_forward",
        "deformable_aggregation_forward_base",
        "ms_deform_attn_forward",
        "modulated_deform_conv_im2col",
        "modulated_deform_conv_col2im",
        "modulated_deform_conv_col2im_coord",
        "modulated_deform_conv_im2col_nhwc",
        "modulated_deform_conv_col2im_nhwc",
        "radius_forward",
        "radius_batch_x_forward",
        "radius_compact_forward",
    ):
        assert native_function(name) is not None
