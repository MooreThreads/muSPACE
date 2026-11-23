#!/usr/bin/env python3
"""Run a single-frame VAD inference with the muSPACE MSDA patch."""

from __future__ import annotations

import argparse
import json
import os
import pickle
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
PREDICTION_GROUPS = (
    ("det", ("all_cls_scores", "all_bbox_preds")),
    ("map", ("map_all_cls_scores", "map_all_pts_preds")),
    ("traj", ("ego_fut_preds", "all_traj_preds")),
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
    output = _scratch_dir(work_dir, "vad-infer-") / ann_file.name
    with output.open("wb") as stream:
        pickle.dump({**payload, "infos": infos}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"vad_infer_scene scene={name} samples={len(infos)}")
    return output


def _apply_patch() -> None:
    """Rebind VAD's MSDA call sites and fail if a module kept the legacy entry."""
    from mu_space.patcher import default_patcher

    report = default_patcher.apply(strict=True)
    missing = [name for name in VAD_MODULES if name not in report.patched_modules]
    if missing:
        raise RuntimeError(f"MSDA patch did not rebind: {', '.join(missing)}")


def _summary(outs: dict) -> dict:
    """Validate the prediction tensors and summarize their shapes."""
    import torch

    report = {}
    for group, keys in PREDICTION_GROUPS:
        tensors = [outs[key] for key in keys if torch.is_tensor(outs.get(key))]
        if not tensors:
            continue
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise RuntimeError(f"non-finite {group} predictions")
        report[group] = {
            "keys": list(keys),
            "shapes": [tuple(tensor.shape) for tensor in tensors],
            "max_abs": max(float(tensor.abs().max()) for tensor in tensors),
        }
    if not report:
        raise RuntimeError(f"no prediction tensors in head output: {sorted(outs)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--config", default="projects/configs/VAD/VAD_base_stage_1.py")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--scene-name", default=None, help="scene name; defaults to the first scene in the infos"
    )
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    if not args.device.startswith("musa"):
        parser.error("--device must be a MUSA device, such as musa:0")

    checkout = args.checkout.resolve()
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(checkout))
    os.environ.setdefault("MUSPACE_STRICT_PATCH", "1")

    import torch

    import torch_musa  # noqa: F401
    from mmcv import Config
    from mmcv.parallel import DataContainer
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset

    import projects.mmdet3d_plugin  # noqa: F401

    torch.manual_seed(0)
    # VAD binds the legacy MSDA entries in its own modules, so import the plugin
    # first and let the patcher rewrite the loaded globals.
    _apply_patch()

    cfg = Config.fromfile(str((checkout / args.config).resolve()))
    ann_file = _scene_subset(
        (args.data_root / "vad_nuscenes_infos_temporal_val.pkl").resolve(),
        args.data_root.resolve(),
        args.scene_name,
        args.work_dir,
    )
    cfg.data.test.data_root = f"{args.data_root.resolve()}/"
    cfg.data.test.ann_file = str(ann_file)
    # Evaluation-only artefacts are not part of this fixture.
    cfg.data.test.map_ann_file = None
    cfg.data.test.pop("samples_per_gpu", None)
    cfg.data.test.pop("workers_per_gpu", None)
    cfg.model.pretrained = None
    if cfg.model.get("img_backbone", {}).get("pretrained"):
        cfg.model.img_backbone.pretrained = None

    dataset = custom_build_dataset(cfg.data.test)
    sample = None
    for index in range(len(dataset)):
        try:
            sample = dataset.prepare_test_data(index)
            break
        except (IndexError, KeyError, TypeError):
            continue
    if sample is None:
        raise RuntimeError("no usable test sample found")
    device = torch.device(args.device)

    def take(value):
        """Unwrap the per-sample ``[DataContainer]`` entries of the pipeline."""
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, DataContainer):
            value = value.data
        if torch.is_tensor(value):
            value = value.to(device)
        return value

    img = take(sample["img"])
    if isinstance(img, (list, tuple)) and img and torch.is_tensor(img[0]):
        img = torch.stack(list(img))
    while img.dim() < 5:
        img = img.unsqueeze(0)
    img_metas = [take(sample["img_metas"])]

    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    # Match the upstream initialisation so the BEV encoder stays finite.
    model.init_weights()
    model.to(device).eval()
    with torch.no_grad():
        img_feats = model.extract_feat(img=img, img_metas=img_metas)
        # The ego-conditioned planner needs the stage-2/e2e inputs; this fixture
        # covers the perception and agent-trajectory heads.
        outs = model.pts_bbox_head(img_feats, img_metas, None)
    print(f"vad_infer_ok device={args.device} report={_summary(outs)}")


if __name__ == "__main__":
    main()
