#!/usr/bin/env python3
"""Run one Sparse4D training step on a single nuScenes scene with muSPACE.

The recipe ships no checkout patches.  The run only rebinds the upstream
``deformable_aggregation_function`` binding, registers a placeholder for the
unbuilt ``deformable_aggregation_ext`` extension, and routes the checkout's
hardcoded ``.cuda()`` calls to MUSA.  The command is a launch and dispatch check,
not a benchmark or an accuracy run.
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
UPSTREAM_REVISION = "4ec96edf0e7d9e80fbd75ef22e1d51a937d3a750"
OPERATOR_NAME = "deformable_aggregation"
DEFAULT_CONFIG = "projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py"
DEFAULT_INFOS = "data/infos/nuscenes_infos_train.pkl"
DEFAULT_ANCHORS = "data/anchors/nuscenes_kmeans900.npy"


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
            [
                "git",
                "-c",
                f"safe.directory={checkout}",
                "-C",
                str(checkout),
                "rev-parse",
                "HEAD",
            ],
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


def _scene_sample_tokens(data_root: Path, scene_name: str, version: str) -> tuple[str, set[str]]:
    """Resolve a scene name to its token and the set of sample tokens in it."""
    from nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    scene = next((item for item in nusc.scene if item["name"] == scene_name), None)
    if scene is None:
        raise FileNotFoundError(f"scene {scene_name!r} is absent from {data_root}")
    tokens: set[str] = set()
    token = scene["first_sample_token"]
    while token:
        tokens.add(token)
        token = nusc.get("sample", token)["next"]
    return scene["token"], tokens


def _absolutize_paths(infos: list[dict[str, Any]], data_root: Path) -> None:
    """Resolve relative image and lidar paths against the nuScenes root."""

    def _resolve(value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        path = Path(value)
        return value if path.is_absolute() else str(data_root / path)

    for info in infos:
        info["lidar_path"] = _resolve(info.get("lidar_path"))
        for cam in info.get("cams", {}).values():
            if isinstance(cam, Mapping):
                cam["data_path"] = _resolve(cam.get("data_path"))
        for sweep in info.get("sweeps", []):
            if isinstance(sweep, Mapping):
                sweep["data_path"] = _resolve(sweep.get("data_path"))


def _subset_info_file(
    infos_path: Path, sample_tokens: set[str], data_root: Path, work_dir: Path
) -> Path:
    """Write a training info file that holds a single scene."""
    with infos_path.open("rb") as stream:
        payload = pickle.load(stream)
    infos = payload.get("infos", []) if isinstance(payload, Mapping) else payload
    if not infos:
        raise ValueError(f"{infos_path} contains no scene infos")
    selected = [info for info in infos if info.get("token") in sample_tokens]
    if not selected:
        raise FileNotFoundError(
            f"{infos_path} has no frames for the requested scene; regenerate the "
            "info files with model_examples/Sparse4D/scripts/prepare_dataset.sh"
        )
    _absolutize_paths(selected, data_root)
    result = dict(payload) if isinstance(payload, Mapping) else {"infos": selected}
    result["infos"] = selected
    output = _scratch_dir(work_dir, "sparse4d-") / "infos_train.pkl"
    with output.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
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


def _configure_fixture(cfg: Any, data_root: Path, ann_file: Path, anchors: Path) -> None:
    """Apply the single-scene, single-card overrides to the released config."""
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    train = cfg.data.train
    train.data_root = f"{data_root}/"
    train.ann_file = str(ann_file)
    train.test_mode = False
    if "samples_per_gpu" in train:
        train.samples_per_gpu = 1
    if "workers_per_gpu" in train:
        train.workers_per_gpu = 0
    cfg.total_batch_size = 1
    cfg.num_gpus = 1
    cfg.batch_size = 1
    cfg.model.img_backbone.pretrained = None
    cfg.model.img_backbone.init_cfg = None
    if not bool(cfg.model.use_deformable_func):
        raise RuntimeError("the pinned config must enable use_deformable_func")
    # The released training config enables denoising.  The single-step fixture
    # disables it so the check stays focused on the deformable aggregation path and
    # does not depend on the negative-DN focal-loss convention.
    cfg.model.head.sampler.num_dn_groups = 0
    cfg.model.head.sampler.num_temp_dn_groups = 0
    cfg.model.head.instance_bank.anchor = str(anchors)

    # A reduced anchor file (for example a mini-split sanity set) cannot hold more
    # temporal instances than it has anchors.
    import numpy as np

    anchor_count = int(np.load(anchors, mmap_mode="r").shape[0])
    temporal_instances = int(cfg.model.head.instance_bank.get("num_temp_instances", 0))
    if temporal_instances > 0:
        cfg.model.head.instance_bank.num_temp_instances = min(temporal_instances, anchor_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixture",), default="fixture")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--infos", default=DEFAULT_INFOS)
    parser.add_argument("--anchors", default=DEFAULT_ANCHORS)
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--scene-name", default="scene-0001")
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--backend", choices=("strict", "auto"), default="strict")
    parser.add_argument("--no-depth-branch", action="store_true")
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
        parser.error(f"not a Sparse4D checkout: {checkout}")
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        parser.error(f"data root does not exist: {data_root}")
    infos_path = Path(args.infos)
    if not infos_path.is_absolute():
        infos_path = checkout / infos_path
    if not infos_path.is_file():
        parser.error(f"training info file does not exist: {infos_path}")
    anchors_path = Path(args.anchors)
    if not anchors_path.is_absolute():
        anchors_path = checkout / anchors_path
    if not anchors_path.is_file():
        parser.error(f"anchor file does not exist: {anchors_path}")
    config = Path(args.config)
    if not config.is_absolute():
        config = checkout / config
    if not config.is_file():
        parser.error(f"config does not exist: {config}")

    revision = _verify_revision(checkout)

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(checkout))
    os.environ.setdefault("MUSPACE_STRICT_PATCH", "1")

    try:
        import torch
        import torch_musa  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Sparse4D fixture requires torch and torch_musa; install the platform "
            "pair listed in model_examples/Sparse4D/README.md"
        ) from exc

    from mu_space.patcher import sparse4d_patcher

    # The rebind must happen before the model package is imported: blocks.py captures
    # the ops binding while it imports, and Sparse4D's package __init__ imports mmdet,
    # which needs the patcher's lite-mmcv extension guard.
    patcher = sparse4d_patcher
    patcher.apply(strict=args.backend == "strict")

    try:
        from mmcv import Config
        from mmcv.parallel import collate
        from mmcv.runner import build_optimizer
        from mmdet.datasets import build_dataset
        from mmdet.models import build_detector

        from mu_space import clear_dispatch_report, get_dispatch_reports

        import projects.mmdet3d_plugin  # noqa: F401

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if hasattr(torch, "musa") and hasattr(torch.musa, "manual_seed_all"):
            torch.musa.manual_seed_all(args.seed)

        with infos_path.open("rb") as stream:
            info_payload = pickle.load(stream)
        metadata = info_payload.get("metadata", {}) if isinstance(info_payload, Mapping) else {}
        nusc_version = metadata.get("version") or args.nuscenes_version
        _, scene_samples = _scene_sample_tokens(data_root, args.scene_name, nusc_version)
        ann_file = _subset_info_file(
            infos_path,
            scene_samples,
            data_root,
            args.work_dir,
        )

        cfg = Config.fromfile(str(config))
        _configure_fixture(cfg, data_root, ann_file, anchors_path)
        if args.no_depth_branch:
            cfg.model.depth_branch = None

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
        if device.type == "musa" and hasattr(torch, "musa") and hasattr(torch.musa, "set_device"):
            torch.musa.set_device(device)
        batch = _move_to(collate([sample], samples_per_gpu=1), device)
        images = batch["img"]
        model_metadata = {key: item for key, item in batch.items() if key != "img"}
        model = build_detector(
            cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
        )
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
            losses = model(images, **model_metadata)
            loss, log_vars = model._parse_losses(losses)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Sparse4D loss is not finite")
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
            raise RuntimeError("Sparse4D has non-finite gradients")

        reports = [item for item in get_dispatch_reports() if item.operator == OPERATOR_NAME]
        if not reports:
            raise RuntimeError("Sparse4D never dispatched the deformable aggregation operator")
        if args.backend == "strict" and any(
            item.selected_backend != "musa" or item.native_fallback or not item.native_available
            for item in reports
        ):
            raise RuntimeError("Sparse4D did not use the strict muSPACE native kernel")

        result = {
            "device": str(device),
            "mode": args.mode,
            "scene": args.scene_name,
            "seed": args.seed,
            "config": str(config),
            "upstream_revision": revision,
            "infos_sha256": _sha256_file(ann_file),
            "anchors": str(anchors_path),
            "anchors_sha256": _sha256_file(anchors_path),
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
            "backend": args.backend,
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
