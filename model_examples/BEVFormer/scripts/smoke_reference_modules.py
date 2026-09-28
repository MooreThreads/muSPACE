"""Small MUSA-only forward/backward smoke for the three BEVFormer attentions."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch_musa  # noqa: F401  # registers the MUSA device with torch


def _assert_musa_finite(output: torch.Tensor, *inputs: torch.Tensor) -> None:
    assert output.device.type == "musa", output.device
    assert torch.isfinite(output).all().item()
    for tensor in inputs:
        assert tensor.grad is not None and tensor.grad.device.type == "musa"
        assert torch.isfinite(tensor.grad).all().item()


def _load_attention_modules(checkout: Path):
    """Load only the attention package, avoiding optional DD3D dependencies."""
    root = checkout.resolve()
    packages = {
        "projects": root / "projects",
        "projects.mmdet3d_plugin": root / "projects/mmdet3d_plugin",
        "projects.mmdet3d_plugin.models": root / "projects/mmdet3d_plugin/models",
        "projects.mmdet3d_plugin.models.utils": root / "projects/mmdet3d_plugin/models/utils",
        "projects.mmdet3d_plugin.bevformer": root / "projects/mmdet3d_plugin/bevformer",
        "projects.mmdet3d_plugin.bevformer.modules": root
        / "projects/mmdet3d_plugin/bevformer/modules",
    }
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load(
        "projects.mmdet3d_plugin.models.utils.bricks",
        root / "projects/mmdet3d_plugin/models/utils/bricks.py",
    )
    image_ops = load(
        "projects.mmdet3d_plugin.models.utils.image_ops",
        root / "projects/mmdet3d_plugin/models/utils/image_ops.py",
    )
    spatial = load(
        "projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention",
        root / "projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py",
    )
    temporal = load(
        "projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention",
        root / "projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py",
    )
    return spatial, temporal, image_ops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, default=Path("BEVFormer"))
    args = parser.parse_args()
    spatial, temporal, image_ops = _load_attention_modules(args.checkout)
    MSDeformableAttention3D = spatial.MSDeformableAttention3D
    SpatialCrossAttention = spatial.SpatialCrossAttention
    TemporalSelfAttention = temporal.TemporalSelfAttention

    rotation_input = (
        torch.arange(9, dtype=torch.float32, device="musa").reshape(1, 3, 3).requires_grad_()
    )
    rotated = image_ops.rotate(rotation_input, 90, center=(1, 1))
    rotation_expected = torch.tensor(
        [[[2.0, 5.0, 8.0], [1.0, 4.0, 7.0], [0.0, 3.0, 6.0]]], device="musa"
    )
    assert torch.equal(rotated, rotation_expected)
    rotated.sum().backward()
    assert rotation_input.grad is not None and torch.isfinite(rotation_input.grad).all().item()

    device = torch.device("musa")
    shapes = torch.tensor([[1, 1], [1, 2]], device=device, dtype=torch.long)
    starts = torch.tensor([0, 1], device=device, dtype=torch.long)

    msda = MSDeformableAttention3D(
        embed_dims=32, num_heads=4, num_levels=2, num_points=2, batch_first=True
    ).to(device)
    query = torch.randn(1, 3, 32, device=device, requires_grad=True)
    value = torch.randn(1, 3, 32, device=device, requires_grad=True)
    reference = torch.rand(1, 3, 1, 2, device=device, requires_grad=True)
    output = msda(
        query=query,
        value=value,
        reference_points=reference,
        spatial_shapes=shapes,
        level_start_index=starts,
    )
    output.square().mean().backward()
    _assert_musa_finite(output, query, value, reference)

    tsa = TemporalSelfAttention(
        embed_dims=32, num_heads=4, num_levels=2, num_points=2, batch_first=True
    ).to(device)
    query = torch.randn(1, 3, 32, device=device, requires_grad=True)
    reference = torch.rand(1, 3, 1, 2, device=device, requires_grad=True)
    output = tsa(
        query=query,
        reference_points=reference,
        spatial_shapes=shapes,
        level_start_index=starts,
    )
    output.square().mean().backward()
    _assert_musa_finite(output, query, reference)

    sca = SpatialCrossAttention(
        embed_dims=32,
        num_cams=2,
        deformable_attention=dict(
            type="MSDeformableAttention3D",
            embed_dims=32,
            num_heads=4,
            num_levels=2,
            num_points=2,
            batch_first=True,
        ),
    ).to(device)
    query = torch.randn(1, 3, 32, device=device, requires_grad=True)
    key = torch.randn(2, 3, 1, 32, device=device, requires_grad=True)
    value = torch.randn(2, 3, 1, 32, device=device, requires_grad=True)
    reference_cam = torch.rand(2, 1, 3, 1, 2, device=device)
    bev_mask = torch.ones(2, 1, 3, 1, dtype=torch.bool, device=device)
    output = sca(
        query=query,
        key=key,
        value=value,
        reference_points_cam=reference_cam,
        bev_mask=bev_mask,
        spatial_shapes=shapes,
        level_start_index=starts,
    )
    output.square().mean().backward()
    # SCA intentionally uses value as the image feature; key is API-compatible
    # metadata and is not part of this module's computation graph.
    _assert_musa_finite(output, query, value)
    print("BEVFormer rotate/MSDA/SCA/TSA MUSA reference forward/backward passed")


if __name__ == "__main__":
    main()
