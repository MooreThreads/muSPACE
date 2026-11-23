#!/usr/bin/env python3
"""Run a short VAD training invocation with the muSPACE MSDA patch."""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import pickle
import random
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VAD_MODULES = (
    "projects.mmdet3d_plugin.VAD.modules.multi_scale_deformable_attn_function",
    "projects.mmdet3d_plugin.VAD.modules.spatial_cross_attention",
    "projects.mmdet3d_plugin.VAD.modules.temporal_self_attention",
    "projects.mmdet3d_plugin.VAD.modules.decoder",
)


def _scratch_dir(parent: Path, prefix: str) -> Path:
    """Create a writable scratch directory below ``parent``."""
    try:
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent)))
    except OSError as exc:
        raise RuntimeError(f"work directory is not writable: {parent}") from exc


def _scene_token(scene_root: Path, scene_name: str | None, infos: list[dict]) -> tuple[str, str]:
    """Resolve a scene name to its token, defaulting to the first scene in the infos."""
    scenes = json.loads((scene_root / "scene.json").read_text())
    names = {item["name"]: item["token"] for item in scenes}
    if scene_name is not None:
        try:
            return names[scene_name], scene_name
        except KeyError as exc:
            raise FileNotFoundError(f"scene {scene_name!r} not found in {scene_root}") from exc
    tokens = {item["token"]: item["name"] for item in scenes}
    for info in infos:
        if info.get("scene_token") in tokens:
            return info["scene_token"], tokens[info["scene_token"]]
    raise ValueError(f"no metadata scene matches {scene_root / 'scene.json'}")


def _scene_subset(ann_file: Path, data_root: Path, scene_name: str | None, work_dir: Path) -> Path:
    """Write a one-scene copy of the temporal metadata."""
    with ann_file.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "infos" not in payload:
        raise ValueError(f"unsupported VAD metadata format: {ann_file}")
    version = payload.get("metadata", {}).get("version", "v1.0-trainval")
    token, name = _scene_token(data_root / version, scene_name, payload["infos"])
    infos = [info for info in payload["infos"] if info.get("scene_token") == token]
    if not infos:
        raise ValueError(f"metadata {ann_file} contains no samples from {name}")
    output = _scratch_dir(work_dir, "vad-") / ann_file.name
    with output.open("wb") as stream:
        pickle.dump({**payload, "infos": infos}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"vad_scene_selected scene={name} samples={len(infos)}")
    return output


def _apply_patch() -> None:
    """Rebind VAD's MSDA call sites and fail if a module kept the legacy entry."""
    from mu_space.patcher import default_patcher

    report = default_patcher.apply(strict=True)
    missing = [name for name in VAD_MODULES if name not in report.patched_modules]
    if missing:
        raise RuntimeError(f"MSDA patch did not rebind: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--scene-name", default=None, help="scene name; defaults to the first scene in the infos"
    )
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument(
        "--infos",
        type=Path,
        help="use this temporal infos file directly instead of a one-scene subset",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=0,
        help="draw a new sample every N iterations; 0 reuses the first usable sample",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="initialise the ResNet backbone from ImageNet weights instead of training from scratch",
    )
    args = parser.parse_args()
    if args.max_iters < 1:
        parser.error("--max-iters must be >= 1")
    if not args.device.startswith("musa"):
        parser.error("--device must be a MUSA device, such as musa:0")

    checkout = args.checkout.resolve()
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(checkout))
    os.environ.setdefault("MUSPACE_STRICT_PATCH", "1")

    import torch

    import torch_musa  # noqa: F401
    from mmcv import Config
    from mmcv.parallel import DataContainer, collate
    from mmcv.runner import build_optimizer
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset

    import projects.mmdet3d_plugin  # noqa: F401

    torch.manual_seed(0)
    np.random.seed(0)
    # VAD binds the legacy MSDA entries in its own modules, so import the plugin
    # first and let the patcher rewrite the loaded globals.
    _apply_patch()

    config = (
        str((checkout / args.config).resolve()) if not os.path.isabs(args.config) else args.config
    )
    if args.infos is not None:
        ann_file = args.infos.resolve()
    else:
        ann_file = _scene_subset(
            (args.data_root / "vad_nuscenes_infos_temporal_train.pkl").resolve(),
            args.data_root.resolve(),
            args.scene_name,
            args.work_dir,
        )
    cfg = Config.fromfile(config)
    cfg.data.train.data_root = f"{args.data_root.resolve()}/"
    cfg.data.train.ann_file = str(ann_file)
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    # The launch check runs offline, so skip the torchvision backbone download.
    cfg.model.pretrained = None
    if cfg.model.get("img_backbone", {}).get("pretrained"):
        cfg.model.img_backbone.pretrained = None
    if args.pretrained_backbone:
        cfg.model.img_backbone.init_cfg = dict(
            type="Pretrained", checkpoint="torchvision://resnet50"
        )

    dataset = custom_build_dataset(cfg.data.train)
    device = torch.device(args.device)

    def unwrap(value):
        if isinstance(value, DataContainer):
            value = value.data[0]
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: unwrap(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(unwrap(item) for item in value)
        return value

    def take_sample(index: int):
        for offset in range(len(dataset)):
            prepared = dataset.prepare_train_data((index + offset) % len(dataset))
            if prepared is not None:
                return collate([prepared], samples_per_gpu=1)
        raise RuntimeError("no usable training sample found")

    batch = unwrap(take_sample(0))
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    # Upstream initialises the transformer with xavier_uniform; skipping it leaves
    # PyTorch defaults that make the BEV encoder activations overflow.
    model.init_weights()
    model.to(device).train()
    optimizer = build_optimizer(model, cfg.optimizer)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    loss_value = None
    log_vars = {}
    for step in range(args.max_iters):
        if step and args.sample_stride:
            batch = unwrap(take_sample(step * args.sample_stride))
        optimizer.zero_grad(set_to_none=True)
        losses = model(return_loss=True, **batch)
        loss, log_vars = model._parse_losses(losses)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
    grad_norm = torch.sqrt(
        sum(
            (parameter.grad.detach() ** 2).sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    param_delta = max(
        (
            float((parameter.detach() - before[name]).abs().max())
            for name, parameter in model.named_parameters()
            if name in before and before[name].shape == parameter.shape
        ),
        default=0.0,
    )
    if not torch.isfinite(grad_norm) or not torch.isfinite(torch.tensor(loss_value)):
        raise RuntimeError(f"non-finite training result: loss={loss_value} grad_norm={grad_norm}")
    if param_delta == 0.0:
        raise RuntimeError("optimizer step did not update any parameter")
    print(
        f"vad_train_step_ok optimizer={type(optimizer).__name__} "
        f"param_groups={len(optimizer.param_groups)} loss={loss_value:.6f} "
        f"grad_norm={float(grad_norm.cpu()):.6f} param_delta={param_delta:.3e} logs={log_vars}"
    )


if __name__ == "__main__":
    main()
