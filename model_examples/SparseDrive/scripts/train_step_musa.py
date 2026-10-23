#!/usr/bin/env python3
"""Run a short SparseDrive training invocation with the muSPACE deformable aggregation patch.

The recipe ships no checkout patches, so the run only rebinds the upstream
``deformable_aggregation_function`` entry point before the model package is imported.
The command is a launch and dispatch check on a single scene, not a benchmark.

SparseDrive's info files carry far more per-frame preprocessing than the other recipes
(map vectors, motion targets, planning targets, ego status), so this script consumes the
info file produced by the recipe dataset scripts instead of rebuilding it in process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_REVISION = "fbadea693cbef3f3daea7705e521c0dd3321605c"
OPERATOR_NAME = "deformable_aggregation"
DEFAULT_CONFIG = "projects/configs/sparsedrive_small_stage1.py"
DEFAULT_INFOS = "data/infos/nuscenes_infos_train.pkl"


def _scratch_dir(parent: Path, prefix: str) -> Path:
    """Allocate a scratch directory below an explicit writable work directory."""
    try:
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent)))
    except OSError as exc:
        raise RuntimeError(f"work directory is not writable: {parent}") from exc


def _verify_revision(checkout: Path) -> str:
    """Confirm the checkout holds the revision this recipe pins."""
    try:
        revision = subprocess.check_output(
            ["git", "-c", f"safe.directory={checkout}", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{checkout} is not a readable git checkout") from exc
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"expected upstream revision {UPSTREAM_REVISION}, found {revision}")
    return revision


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_token(data_root: Path, scene_name: str) -> str:
    """Resolve a scene name to its token using the local nuScenes tree."""
    from nuscenes import NuScenes

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(data_root), verbose=False)
    scene = next((item for item in nusc.scene if item["name"] == scene_name), None)
    if scene is None:
        raise FileNotFoundError(f"scene {scene_name!r} is absent from {data_root}")
    return scene["token"]


def _subset_info_file(infos_path: Path, scene_token: str, work_dir: Path) -> Path:
    """Write a training info file that holds a single scene."""
    with infos_path.open("rb") as stream:
        payload = pickle.load(stream)
    infos = payload.get("infos", []) if isinstance(payload, Mapping) else payload
    if not infos:
        raise ValueError(f"{infos_path} contains no scene infos")
    selected = [info for info in infos if info.get("scene_token") == scene_token]
    if not selected:
        raise FileNotFoundError(
            f"{infos_path} has no frames for the requested scene; regenerate the info "
            "files with the recipe dataset scripts"
        )
    metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
    output = _scratch_dir(work_dir, "sparsedrive-") / "infos_train.pkl"
    with output.open("wb") as stream:
        pickle.dump(
            {"infos": selected, "metadata": metadata},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return output


def _move_to(value: Any, device: Any) -> Any:
    """Move a collated batch onto the target device, unwrapping DataContainer."""
    import torch
    from mmcv.parallel import DataContainer

    if isinstance(value, DataContainer):
        return _move_to(value.data[0], device)
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to(item, device) for item in value)
    return value


def _unique_reports(reports: list) -> list[dict]:
    """Collapse per-call dispatch reports into the distinct decisions they record."""
    unique: list[dict] = []
    seen: set[tuple] = set()
    for report in reports:
        entry = report.as_dict()
        key = tuple(sorted((name, str(value)) for name, value in entry.items()))
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--infos", default=DEFAULT_INFOS)
    parser.add_argument("--scene-name", default="scene-0001")
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument(
        "--result-artifact",
        type=Path,
        default=None,
        help="optional path that receives the loss and gradients of the run",
    )
    args = parser.parse_args()

    if args.max_iters < 1:
        parser.error("--max-iters must be >= 1")
    if not args.device.startswith("musa"):
        parser.error("--device must be a MUSA device, such as musa:0")
    checkout = args.checkout.resolve()
    if not (checkout / "projects" / "mmdet3d_plugin").is_dir():
        parser.error(f"not a SparseDrive checkout: {checkout}")
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        parser.error(f"data root does not exist: {data_root}")
    infos_path = Path(args.infos)
    if not infos_path.is_absolute():
        infos_path = checkout / infos_path
    if not infos_path.is_file():
        parser.error(f"training info file does not exist: {infos_path}")
    config = Path(args.config)
    if not config.is_absolute():
        config = checkout / config
    if not config.is_file():
        parser.error(f"config does not exist: {config}")

    revision = _verify_revision(checkout)

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(checkout))
    os.environ.setdefault("MUSPACE_STRICT_PATCH", "1")

    from mu_space.patcher import sparsedrive_patcher

    # The rebind has to happen before the model package is imported: the upstream
    # blocks module captures the ops binding while it is being imported, and the
    # patcher updates both the ops package attribute and that captured binding.
    patcher = sparsedrive_patcher
    patcher.apply(strict=True)

    import torch
    import torch_musa  # noqa: F401
    from mmcv import Config
    from mmcv.parallel import collate
    from mmcv.runner import build_optimizer
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    from mu_space import clear_dispatch_report, get_dispatch_reports

    import projects.mmdet3d_plugin  # noqa: F401

    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        ann_file = _subset_info_file(
            infos_path, _scene_token(data_root, args.scene_name), args.work_dir
        )
        cfg = Config.fromfile(str(config))
        cfg.data.train.data_root = f"{data_root}/"
        cfg.data.train.ann_file = str(ann_file)
        cfg.data.samples_per_gpu = 1
        cfg.data.workers_per_gpu = 0
        cfg.model.img_backbone.pretrained = None
        cfg.model.img_backbone.init_cfg = None
        dataset = build_dataset(cfg.data.train)

        sample = None
        skipped: Exception | None = None
        for index in range(len(dataset)):
            try:
                candidate = dataset[index]
            except Exception as exc:  # noqa: BLE001 - one unusable frame must not end the check
                skipped = skipped or exc
                continue
            if candidate is not None:
                sample = candidate
                break
        if sample is None:
            raise RuntimeError(f"no usable training sample in {args.scene_name}: {skipped}")

        device = torch.device(args.device)
        batch = _move_to(collate([sample], samples_per_gpu=1), device)
        images = batch["img"]
        metadata = {key: item for key, item in batch.items() if key != "img"}
        model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
        model.to(device).train()
        optimizer = build_optimizer(model, cfg.optimizer)

        selected_name, selected = next(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        before = selected.detach().float().cpu().clone()
        clear_dispatch_report()

        loss = None
        log_vars: dict[str, Any] = {}
        for _ in range(args.max_iters):
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, **metadata)
            loss, log_vars = model._parse_losses(losses)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("SparseDrive loss is not finite")
            loss.backward()
            optimizer.step()

        after = selected.detach().float().cpu().clone()
        if torch.equal(before, after):
            raise RuntimeError("the optimizer did not update any parameter")
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not finite_gradients:
            raise RuntimeError("SparseDrive has non-finite gradients")

        reports = [item for item in get_dispatch_reports() if item.operator == OPERATOR_NAME]
        if not reports:
            raise RuntimeError("SparseDrive never dispatched the deformable aggregation operator")
        if args.device.startswith("musa") and any(
            item.selected_backend != "musa" or item.native_fallback or not item.native_available
            for item in reports
        ):
            raise RuntimeError("SparseDrive did not use the strict muSPACE native kernel")

        result = {
            "device": str(device),
            "scene": args.scene_name,
            "seed": args.seed,
            "config": str(config),
            "upstream_revision": revision,
            "infos_sha256": _sha256_file(ann_file),
            "torch_version": torch.__version__,
            "max_iters": args.max_iters,
            "loss": float(loss.detach().float().cpu()),
            "log_vars": {key: float(value) for key, value in log_vars.items()},
            "selected_parameter": selected_name,
            "selected_parameter_before_sha256": hashlib.sha256(
                before.numpy().tobytes()
            ).hexdigest(),
            "selected_parameter_after_sha256": hashlib.sha256(after.numpy().tobytes()).hexdigest(),
            "finite_gradients": bool(finite_gradients),
            "optimizer_step": True,
            "dispatch": _unique_reports(reports),
            "dispatch_calls": len(reports),
        }
        if args.result_artifact is not None:
            args.result_artifact.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "result": result,
                    "gradients": {
                        name: parameter.grad.detach().float().cpu()
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None
                    },
                },
                args.result_artifact,
            )
        print(json.dumps(result, sort_keys=True))
    finally:
        patcher.restore()


if __name__ == "__main__":
    main()
