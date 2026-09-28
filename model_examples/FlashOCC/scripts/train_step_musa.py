#!/usr/bin/env python3
"""Run a short FlashOcc training invocation with the muSPACE BEVPool patch."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _scratch_dir(parent: Path, prefix: str) -> Path:
    """Allocate a scratch directory below an explicit writable work directory."""
    try:
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent)))
    except OSError as exc:
        raise RuntimeError(f"work directory is not writable: {parent}") from exc


def _build_scene_metadata(data_root: Path, scene_name: str, work_dir: Path) -> Path:
    """Build FlashOcc metadata from one local nuScenes scene only."""
    import numpy as np

    from nuscenes import NuScenes
    from tools.create_data_bevdet import get_gt
    from tools.data_converter import nuscenes_converter

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(data_root), verbose=False)
    scene = next((item for item in nusc.scene if item["name"] == scene_name), None)
    if scene is None:
        raise FileNotFoundError(f"scene {scene_name!r} is absent from {data_root}")
    scene_token = scene["token"]
    original_samples = nusc.sample
    nusc.sample = [item for item in original_samples if item["scene_token"] == scene_token]
    try:
        infos, _ = nuscenes_converter._fill_trainval_infos(
            nusc, [scene_token], [], test=False, max_sweeps=0
        )
    finally:
        nusc.sample = original_samples
    if not infos:
        raise ValueError(f"no metadata was generated for {scene_name}")
    for info in infos:
        sample = nusc.get("sample", info["token"])
        annotations = []
        for token in sample["anns"]:
            annotation = nusc.get("sample_annotation", token)
            velocity = nusc.box_velocity(token)
            annotation["velocity"] = (
                np.zeros(3, dtype=np.float32) if np.any(np.isnan(velocity)) else velocity
            )
            annotations.append(annotation)
        info["ann_infos"] = annotations
        info["ann_infos"] = get_gt(info)
        info["scene_token"] = scene_token
        info["scene_name"] = scene_name
        info["occ_path"] = str(data_root / "gts" / scene_name / info["token"])
    output = _scratch_dir(work_dir, "flashocc-") / "infos_train.pkl"
    with output.open("wb") as stream:
        pickle.dump(
            {"infos": infos, "metadata": {"version": "v1.0-trainval"}},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-name", default="scene-0001")
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    if args.max_iters < 1:
        parser.error("--max-iters must be >= 1")
    if not args.device.startswith("musa"):
        parser.error("--device must be a MUSA device, such as musa:0")

    sys.path.insert(0, str(REPO_ROOT))
    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout))
    os.environ.setdefault("MUSPACE_STRICT_PATCH", "1")
    from mu_space.patcher import flashocc_patcher

    flashocc_patcher.apply(strict=True)
    config = (
        str((checkout / args.config).resolve()) if not os.path.isabs(args.config) else args.config
    )
    ann_file = _build_scene_metadata(args.data_root.resolve(), args.scene_name, args.work_dir)
    import torch
    import torch_musa  # noqa: F401
    from mmcv import Config
    from mmcv.parallel import DataContainer, collate
    from mmcv.runner import build_optimizer
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    import projects.mmdet3d_plugin  # noqa: F401

    cfg = Config.fromfile(config)
    cfg.data.train.data_root = f"{args.data_root.resolve()}/"
    cfg.data.train.ann_file = str(ann_file)
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    cfg.model.img_backbone.pretrained = None
    cfg.model.img_backbone.init_cfg = None
    dataset = build_dataset(cfg.data.train)
    sample = next(
        (
            prepared
            for index in range(len(dataset))
            if (prepared := dataset.prepare_train_data(index)) is not None
        ),
        None,
    )
    if sample is None:
        raise RuntimeError(f"no usable training sample found in {args.scene_name}")
    packed = collate([sample], samples_per_gpu=1)
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

    batch = unwrap(packed)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.to(device).train()
    optimizer = build_optimizer(model, cfg.optimizer)
    for _ in range(args.max_iters):
        optimizer.zero_grad(set_to_none=True)
        losses = model(return_loss=True, **batch)
        loss, log_vars = model._parse_losses(losses)
        loss.backward()
        optimizer.step()
    print(
        f"flashocc_train_step_ok optimizer={type(optimizer).__name__} "
        f"param_groups={len(optimizer.param_groups)} loss={float(loss.detach().cpu()):.6f} "
        f"logs={log_vars}"
    )


if __name__ == "__main__":
    main()
