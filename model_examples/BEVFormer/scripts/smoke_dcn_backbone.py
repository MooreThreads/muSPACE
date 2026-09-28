"""Smoke the BEVFormer ResNet-DCN backbone against the muSPACE DCNv2 operator.

BEVFormer's base and small configurations build their image backbone with
``dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False)`` and
``stage_with_dcn=(False, False, True, True)``. mmdet resolves that ``type`` through
mmcv's ``CONV_LAYERS`` registry, so the muSPACE runtime patcher is what puts the MUSA
operator in the backbone; no model source is edited.

This script applies the patcher, builds the backbone, checks that the deformable layers
really are muSPACE modules, and runs one forward and backward pass on MUSA.

Requires the OpenMMLab compatibility stack from
``scripts/build_openmmlab_compat_wheels.sh``. The BEVFormer-tiny fixture uses a plain
ResNet-50 without DCN, so this is the check that covers the base and small recipes.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch_musa  # noqa: F401  # registers the MUSA device with torch

from mu_space import (
    clear_dispatch_report,
    enable_eval_batch_norm_reference,
    get_last_dispatch_report,
)
from mu_space.ops import ModulatedDeformConv2dPack
from mu_space.patcher import modulated_deform_conv_patcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, choices=(50, 101), default=50)
    parser.add_argument("--input-size", type=int, nargs=2, default=(128, 352))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--strict", action="store_true", help="request the strict patch mode")
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def _build_backbone(depth: int):
    from mmdet.models import ResNet

    # The img_backbone of projects/configs/bevformer/bevformer_base.py, with the
    # depth left configurable so the smoke can run on ResNet-50.
    backbone = ResNet(
        depth=depth,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    )
    backbone.init_weights()
    return backbone


def main() -> None:
    if not torch.musa.is_available():
        raise SystemExit("this smoke requires MUSA")

    args = parse_args()
    torch.manual_seed(args.seed)

    patch_report = modulated_deform_conv_patcher.apply(strict=args.strict or None)
    backbone = _build_backbone(args.depth)

    deform_layers = [
        name
        for name, module in backbone.named_modules()
        if isinstance(module, ModulatedDeformConv2dPack)
    ]
    if not deform_layers:
        raise SystemExit(
            "the backbone has no muSPACE deformable layers; the runtime patch did not take"
        )

    backbone = backbone.musa().train()
    eval_batch_norm_reference = enable_eval_batch_norm_reference(backbone)

    height, width = args.input_size
    images = torch.randn(args.batch, 3, height, width, device="musa", requires_grad=True)

    clear_dispatch_report()
    outputs = backbone(images)
    loss = sum(output.float().sum() for output in outputs)
    loss.backward()
    torch.musa.synchronize()

    dispatch_report = get_last_dispatch_report()
    trained = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    payload = {
        "patch": patch_report.as_dict(),
        "backbone": f"ResNet-{args.depth}",
        "deform_layers": len(deform_layers),
        "first_deform_layer": deform_layers[0],
        "eval_batch_norm_reference": eval_batch_norm_reference,
        "output_shapes": [list(output.shape) for output in outputs],
        "loss": float(loss.detach()),
        "finite_loss": bool(torch.isfinite(loss.detach()).item()),
        "finite_input_gradient": bool(torch.isfinite(images.grad).all().item()),
        "trained_parameters_with_gradients": sum(
            1 for parameter in trained if parameter.grad is not None
        ),
        "trained_parameters": len(trained),
        "dispatch": dispatch_report.as_dict() if dispatch_report else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if payload["dispatch"] is not None and payload["dispatch"]["selected_backend"] != "musa":
        raise SystemExit("the deformable convolution did not run on the MUSA backend")
    if not payload["finite_loss"] or not payload["finite_input_gradient"]:
        raise SystemExit("the backbone produced non-finite values")


if __name__ == "__main__":
    main()
