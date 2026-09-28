#!/usr/bin/env python3
"""Run a short UniAD training invocation with the muSPACE MSDA patch."""

from __future__ import annotations

import argparse
import os
import json
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


def _scene_subset(ann_file: Path, data_root: Path, scene_name: str, work_dir: Path) -> Path:
    """Create a deterministic one-scene temporal metadata file."""
    version_root = data_root / "v1.0-trainval"
    scenes = json.loads((version_root / "scene.json").read_text())
    scene_tokens = {item["name"]: item["token"] for item in scenes}
    try:
        scene_token = scene_tokens[scene_name]
    except KeyError as exc:
        raise FileNotFoundError(
            f"scene {scene_name!r} not found in {version_root / 'scene.json'}"
        ) from exc
    with ann_file.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "infos" not in payload:
        raise ValueError(f"unsupported UniAD metadata format: {ann_file}")
    infos = [
        info
        for info in payload["infos"]
        if info.get("scene_token") == scene_token or info.get("scene_name") == scene_name
    ]
    if not infos:
        raise ValueError(f"metadata {ann_file} contains no samples from {scene_name}")

    source_root = "data/nuscenes/"

    def relocate(value):
        if isinstance(value, str) and value.startswith(source_root):
            return str(data_root / value[len(source_root) :])
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, tuple):
            return tuple(relocate(item) for item in value)
        return value

    infos = [relocate(info) for info in infos]
    payload["infos"] = infos
    output = _scratch_dir(work_dir, "uniad-") / ann_file.name
    with output.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
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
    parser.add_argument("--dump-batch", type=Path)
    parser.add_argument("--batch-file", type=Path)
    parser.add_argument("--dump-only", action="store_true")
    args = parser.parse_args()
    if args.max_iters < 1:
        parser.error("--max-iters must be >= 1")
    if not args.device.startswith("musa"):
        parser.error("--device must be a MUSA device, such as musa:0")

    checkout = args.checkout.resolve()
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(checkout))
    if args.device.startswith("musa") or os.getenv("UNIAD_LEGACY_IMPORT_GUARD"):
        os.environ.setdefault("MUSPACE_STRICT_PATCH", "1")
        from mu_space.patcher import default_patcher

        default_patcher.apply(strict=True)
    config = (
        str((checkout / args.config).resolve()) if not os.path.isabs(args.config) else args.config
    )
    ann_file = None
    if args.batch_file is None:
        ann_file = _scene_subset(
            (args.data_root / "nuscenes_infos_temporal_train.pkl").resolve(),
            args.data_root.resolve(),
            args.scene_name,
            args.work_dir,
        )
    import torch

    if args.device.startswith("musa"):
        import torch_musa  # noqa: F401
    from mmcv import Config
    from mmcv.parallel import DataContainer, collate
    from mmcv.runner import build_optimizer
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset

    import projects.mmdet3d_plugin  # noqa: F401

    cfg = Config.fromfile(config)
    cfg.data.train.data_root = f"{args.data_root.resolve()}/"
    if ann_file is not None:
        cfg.data.train.ann_file = str(ann_file)
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    # MMCV's legacy DCNv2 extension has no MUSA replacement in this smoke path.
    # Use the same ResNet stages with regular convolutions for launch validation.
    cfg.model.img_backbone.dcn = None
    cfg.model.img_backbone.stage_with_dcn = (False, False, False, False)

    def set_img_root(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "img_root":
                    value[key] = str(args.data_root.resolve())
                else:
                    set_img_root(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                set_img_root(item)

    set_img_root(cfg.data.train.pipeline)
    if args.batch_file:
        packed = torch.load(args.batch_file, map_location="cpu", weights_only=False)
    else:
        dataset = custom_build_dataset(cfg.data.train)
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
    if args.dump_batch:
        args.dump_batch.parent.mkdir(parents=True, exist_ok=True)
        torch.save(packed, args.dump_batch)
    if args.dump_only:
        return
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
        f"uniad_train_step_ok optimizer={type(optimizer).__name__} "
        f"param_groups={len(optimizer.param_groups)} loss={float(loss.detach().cpu()):.6f} "
        f"logs={log_vars}"
    )


if __name__ == "__main__":
    main()
