"""Import and MUSA MSDA-reference smoke test for the fixed OpenMMLab stack."""

import importlib

import torch
import torch_musa  # noqa: F401


def main() -> None:
    modules = [
        "mmcv",
        "mmcv.runner",
        "mmcv.cnn",
        "mmcv.ops",
        "mmdet",
        "mmdet.models",
        "mmseg",
        "mmseg.models",
        "mmdet3d",
        "mmdet3d.core.bbox",
    ]
    for name in modules:
        importlib.import_module(name)

    from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch

    value = torch.randn(1, 4, 2, 4, device="musa", requires_grad=True)
    spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long)
    sampling = torch.rand(1, 3, 2, 1, 2, 2, device="musa", requires_grad=True)
    weights_raw = torch.randn(1, 3, 2, 1, 2, device="musa", requires_grad=True)
    output = multi_scale_deformable_attn_pytorch(
        value, spatial_shapes, sampling, weights_raw.softmax(-1)
    )
    output.sum().backward()

    assert output.device.type == "musa"
    for tensor in (value, sampling, weights_raw):
        assert tensor.grad is not None
        assert tensor.grad.device.type == "musa"
        assert torch.isfinite(tensor.grad).all()

    from mmcv.ops import nms
    from mmcv.utils.ext_loader import check_ops_exist

    assert not check_ops_exist()
    try:
        nms(torch.zeros((1, 4)), torch.zeros(1), 0.5)
    except RuntimeError as error:
        assert "unavailable" in str(error)
    else:
        raise AssertionError("disabled mmcv native op unexpectedly succeeded")

    print(f"OpenMMLab imports and MUSA MSDA reference passed: output={tuple(output.shape)}")


if __name__ == "__main__":
    main()
