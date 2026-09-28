#!/usr/bin/env python3
"""Run one camera-only BEVFusion segmentation optimizer step."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

UPSTREAM_REVISION = "326653dc06e0938edf1aae7d01efcd158ba83de5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--device", choices=("musa", "cuda"), default="musa")
    parser.add_argument("--backend", choices=("auto", "musa", "reference"), default="musa")
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def add_paths(checkout: Path) -> None:
    root = Path(__file__).resolve()
    for parent in root.parents:
        if (parent / "mu_space").is_dir():
            sys.path.insert(0, str(parent))
            break
    sys.path.insert(0, str(checkout))


def ensure_patch(checkout: Path) -> None:
    script = Path(__file__).with_name("apply_source_patch.py")
    subprocess.run(
        [sys.executable, str(script), "--checkout", str(checkout), "--action", "apply"],
        check=True,
    )


def install_camera_only_import_surface(checkout: Path) -> None:
    """Import only the official camera-only modules without optional CUDA extensions."""
    import importlib

    def package(name: str, path: Path) -> None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    import mmdet3d  # noqa: F401

    models = checkout / "mmdet3d" / "models"
    package("mmdet3d.models", models)
    for name in ("backbones", "necks", "vtransforms", "heads", "fusion_models"):
        package(f"mmdet3d.models.{name}", models / name)
    package("mmdet3d.models.heads.segm", models / "heads" / "segm")

    ops = types.ModuleType("mmdet3d.ops")

    def bev_pool(feats, coords, batch_size, depth, height, width):
        ranks = (
            coords[:, 0].long() * (width * depth * batch_size)
            + coords[:, 1].long() * (depth * batch_size)
            + coords[:, 2].long() * batch_size
            + coords[:, 3].long()
        )
        output = feats.new_zeros((batch_size * depth * height * width, feats.shape[-1]))
        output.index_add_(0, ranks, feats)
        return output.view(height, width, depth, batch_size, -1).permute(3, 4, 2, 0, 1).contiguous()

    class UnusedCameraOnlyOperator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("this camera-only fixture does not use the requested 3D operator")

    ops.bev_pool = bev_pool
    ops.DynamicScatter = UnusedCameraOnlyOperator
    ops.Voxelization = UnusedCameraOnlyOperator
    sys.modules["mmdet3d.ops"] = ops
    builder = importlib.import_module("mmdet3d.models.builder")
    models_module = sys.modules["mmdet3d.models"]
    for name in ("BACKBONES", "NECKS", "VTRANSFORMS", "FUSIONMODELS", "HEADS"):
        if hasattr(builder, name):
            setattr(models_module, name, getattr(builder, name))
    for name in (
        "mmdet3d.models.backbones.resnet",
        "mmdet3d.models.necks.generalized_lss",
        "mmdet3d.models.necks.lss",
        "mmdet3d.models.vtransforms.base",
        "mmdet3d.models.heads.segm.vanilla",
        "mmdet3d.models.fusion_models.base",
        "mmdet3d.models.fusion_models.bevfusion",
    ):
        importlib.import_module(name)
    base = importlib.import_module("mmdet3d.models.vtransforms.base")
    original_forward = base.BaseTransform.forward

    def camera_only_forward(
        self,
        image,
        points,
        radar,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        image_aug_matrix,
        lidar_aug_matrix,
        metas=None,
        **kwargs,
    ):
        del metas
        return original_forward(
            self,
            image,
            points,
            radar,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            image_aug_matrix,
            lidar_aug_matrix,
            **kwargs,
        )

    base.BaseTransform.forward = camera_only_forward
    lss = importlib.import_module("mmdet3d.models.vtransforms.lss")
    original_get_cam_feats = lss.LSSTransform.get_cam_feats

    def camera_only_get_cam_feats(self, image, mats_dict):
        del mats_dict
        return original_get_cam_feats(self, image)

    lss.LSSTransform.get_cam_feats = camera_only_get_cam_feats


def configure_fixture(config) -> None:
    """Keep the official camera-only graph while making one-step validation compact."""
    model = config.model
    model.pretrained = None
    camera = model.encoders.camera
    camera.backbone.embed_dims = 32
    camera.backbone.depths = [1, 1, 1, 1]
    camera.backbone.num_heads = [1, 2, 4, 8]
    camera.backbone.drop_path_rate = 0.0
    camera.backbone.init_cfg = None
    camera.neck.in_channels = [64, 128, 256]
    camera.neck.out_channels = 32
    camera.vtransform.in_channels = 32
    camera.vtransform.out_channels = 16
    camera.vtransform.image_size = [64, 64]
    camera.vtransform.feature_size = [8, 8]
    camera.vtransform.xbound = [-4.0, 4.0, 1.0]
    camera.vtransform.ybound = [-4.0, 4.0, 1.0]
    camera.vtransform.zbound = [-2.0, 2.0, 4.0]
    camera.vtransform.dbound = [1.0, 9.0, 1.0]

    decoder = model.decoder
    decoder.backbone.in_channels = 16
    decoder.backbone.blocks = [[1, 32, 1], [1, 64, 1], [1, 128, 1]]
    decoder.neck.in_channels = [128, 32]
    decoder.neck.out_channels = 32

    head = model.heads.map
    head.in_channels = 32
    head.grid_transform.input_scope = [[-4.0, 4.0, 1.0], [-4.0, 4.0, 1.0]]
    head.grid_transform.output_scope = [[-4.0, 4.0, 1.0], [-4.0, 4.0, 1.0]]


def recursive_eval(value, globals_=None):
    if globals_ is None:
        import copy

        globals_ = copy.deepcopy(value)
    if isinstance(value, dict):
        for key in value:
            value[key] = recursive_eval(value[key], globals_)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = recursive_eval(item, globals_)
    elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        value = recursive_eval(eval(value[2:-1], globals_), globals_)
    return value


def matrix_batch(batch: int, cameras: int, device: torch.device) -> torch.Tensor:
    return torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch, cameras, 1, 1)


def fixture(device: torch.device, classes: int) -> dict[str, object]:
    batch, cameras, height, width = 1, 6, 64, 64
    matrices = matrix_batch(batch, cameras, device)
    return {
        "img": torch.randn(batch, cameras, 3, height, width, device=device),
        "points": [torch.empty(0, 5, device=device)],
        "camera2ego": matrices.clone(),
        "lidar2ego": torch.eye(4, device=device).view(1, 4, 4).repeat(batch, 1, 1),
        "lidar2camera": matrices.clone(),
        "lidar2image": matrices.clone(),
        "camera_intrinsics": matrices.clone(),
        "camera2lidar": matrices.clone(),
        "img_aug_matrix": matrices.clone(),
        "lidar_aug_matrix": torch.eye(4, device=device).view(1, 4, 4).repeat(batch, 1, 1),
        "metas": [{} for _ in range(batch)],
        "depths": None,
        "radar": None,
        "gt_masks_bev": torch.randint(0, 2, (batch, classes, 8, 8), device=device).float(),
    }


def main() -> None:
    args = parse_args()
    checkout = args.checkout.resolve()
    actual_revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={checkout}", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_revision != UPSTREAM_REVISION:
        raise SystemExit(f"expected {UPSTREAM_REVISION}, got {actual_revision}")
    if args.device == "musa" and not (hasattr(torch, "musa") and torch.musa.is_available()):
        raise SystemExit("--device musa requires an available MUSA device")

    add_paths(checkout)
    ensure_patch(checkout)
    os.environ["MUSPACE_SIGMOID_FOCAL_LOSS_BACKEND"] = args.backend
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if "bool" not in np.__dict__:
        np.bool = np.bool_  # type: ignore[attr-defined]

    from mmcv import Config
    from torchpack.utils.config import configs
    install_camera_only_import_surface(checkout)
    from mmdet3d.models.builder import build_fusion_model
    from mu_space import clear_dispatch_report, get_last_dispatch_report

    config_path = checkout / "configs/nuscenes/seg/camera-bev256d2.yaml"
    configs.load(str(config_path), recursive=True)
    config = Config(recursive_eval(configs), filename=str(config_path))
    configure_fixture(config)
    model = build_fusion_model(config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg"))
    device = torch.device(f"{args.device}:0")
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = fixture(device, len(config.model.heads.map.classes))
    first_parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    before = first_parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    clear_dispatch_report()
    losses = model(**batch)
    loss_values = {name: value for name, value in losses.items() if value.requires_grad}
    loss = sum(loss_values.values())
    if not torch.isfinite(loss).item():
        raise RuntimeError("BEVFusion fixture loss is non-finite")
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise RuntimeError("BEVFusion fixture has non-finite gradients")
    optimizer.step()
    report = get_last_dispatch_report()
    payload = {
        "device": str(device),
        "focal_loss_dispatch": None if report is None else report.as_dict(),
        "finite_gradients": finite_gradients,
        "loss": float(loss.detach()),
        "losses": {name: float(value.detach()) for name, value in loss_values.items()},
        "optimizer_step": not torch.equal(before, first_parameter.detach()),
        "seed": args.seed,
        "upstream_revision": actual_revision,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
