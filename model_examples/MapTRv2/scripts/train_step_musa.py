#!/usr/bin/env python3
"""Run one deterministic MapTRv2 optimizer step on a real nuScenes mini scene."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_REVISION = "e03f097abef19e1ba3fed5f471a8d80fbfa0a064"
CONFIG = "projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--can-bus-root", type=Path, required=True)
    parser.add_argument("--scene-name", default="scene-0061")
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--device", choices=("musa:0",), default="musa:0")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--input-artifact", type=Path)
    parser.add_argument("--result-artifact", type=Path)
    return parser.parse_args()


def _scratch_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="maptrv2-", dir=parent))


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.musa.manual_seed_all(seed)


def _prepare_infos(args: argparse.Namespace, scratch: Path) -> Path:
    output = args.work_dir / "infos"
    source = output / "nuscenes_map_infos_temporal_train.pkl"
    if not source.is_file():
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(args.checkout / "tools/maptrv2/custom_nusc_map_converter.py"),
            "--root-path", str(args.data_root), "--out-dir", str(output),
            "--extra-tag", "nuscenes", "--version", "v1.0-mini",
            "--canbus", str(args.can_bus_root), "--max-sweeps", "0",
        ]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(args.checkout))))
        subprocess.run(command, cwd=args.checkout, env=env, check=True)
    if not source.is_file():
        raise RuntimeError(f"MapTRv2 mini converter did not create {source.name}")
    with source.open("rb") as stream:
        payload = pickle.load(stream)
    infos = payload["infos"]
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version="v1.0-mini", dataroot=str(args.data_root), verbose=False)
    scene_token = next((scene["token"] for scene in nusc.scene if scene["name"] == args.scene_name), None)
    selected = [item for item in infos if item.get("scene_token") == scene_token]
    if not selected:
        raise RuntimeError(f"no training samples for {args.scene_name}")
    payload = dict(payload, infos=selected)
    subset = scratch / "scene_train.pkl"
    with subset.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return subset


def _cpu(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _to_device(value: Any, device) -> Any:
    import torch
    from mmcv.parallel import DataContainer

    if isinstance(value, DataContainer):
        return _to_device(value.data[0], device)
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _pick_tensors(value: Any, prefix: str = "") -> dict[str, Any]:
    import torch

    if torch.is_tensor(value):
        return {prefix: _cpu(value)}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(_pick_tensors(item, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(_pick_tensors(item, f"{prefix}.{index}"))
        return result
    return {}


def _gradient_summary(model) -> dict[str, dict[str, float]]:
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad = parameter.grad.detach().float()
            result[name] = {"max_abs": float(grad.abs().max()), "mean_abs": float(grad.abs().mean())}
    return result


def main() -> None:
    args = _parse_args()
    if args.max_iters < 1:
        raise SystemExit("--max-iters must be >= 1")
    args.checkout, args.data_root, args.can_bus_root = (
        path.resolve() for path in (args.checkout, args.data_root, args.can_bus_root))
    args.work_dir = args.work_dir.resolve()
    checkout, data_root, can_bus_root = args.checkout, args.data_root, args.can_bus_root
    if not (checkout / ".git").is_dir():
        raise SystemExit(f"not a MapTR checkout: {checkout}")
    if args.input_artifact is None:
        if not (data_root / "v1.0-mini").is_dir():
            raise SystemExit(f"not a nuScenes mini root: {data_root}")
        if not (can_bus_root / "can_bus" / f"{args.scene_name}_pose.json").is_file():
            raise SystemExit(f"CAN-bus expansion missing below: {can_bus_root}")
    sys.path.insert(0, str(checkout))
    sys.path.insert(0, str(ROOT))
    import torch
    import torch_musa  # noqa: F401
    from mu_space.patcher import default_patcher

    default_patcher.apply(strict=True)
    from mmcv import Config
    from mmcv.runner import build_optimizer
    from mmdet3d.models import build_model
    if args.input_artifact is None:
        from mmcv.parallel import collate
        from mmdet3d.datasets import build_dataset
    _seed(args.seed)
    import projects.mmdet3d_plugin  # noqa: F401

    scratch = _scratch_dir(args.work_dir)
    cfg = Config.fromfile(str(checkout / CONFIG))
    cfg.data.train.data_root = f"{data_root}/"
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    cfg.data.train.ann_file = str(_prepare_infos(args, scratch)) if args.input_artifact is None else ""
    cfg.model.pretrained = None
    cfg.model.img_backbone.pretrained = None
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    device = torch.device(args.device)
    model.to(device).train()
    if args.input_artifact is None:
        dataset = build_dataset(cfg.data.train)
        sample = None
        for index in range(len(dataset)):
            prepared = dataset.prepare_train_data(index)
            if prepared is not None:
                sample = prepared
                break
        if sample is None:
            raise RuntimeError(f"no usable sample for {args.scene_name}")
        batch_cpu = _cpu(collate([sample], samples_per_gpu=1))
        input_artifact = scratch / "input.pt"
        torch.save({"batch": batch_cpu, "initial_state": _cpu(model.state_dict())}, input_artifact)
    else:
        input_artifact = args.input_artifact.resolve()
        shared = torch.load(input_artifact, map_location="cpu", weights_only=False)
        batch_cpu = shared["batch"]
        model.load_state_dict(shared["initial_state"], strict=True)
    batch = _to_device(batch_cpu, device)
    optimizer = build_optimizer(model, cfg.optimizer)
    for _ in range(args.max_iters):
        optimizer.zero_grad(set_to_none=True)
        losses = model(return_loss=True, **batch)
        loss, log_vars = model._parse_losses(losses)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss}")
        loss.backward()
        optimizer.step()
    result = {
        "payload": {"scene": args.scene_name, "seed": args.seed, "upstream_revision": UPSTREAM_REVISION,
                    "device": args.device, "loss": float(loss.detach().cpu()), "input_artifact": str(input_artifact)},
        "losses": _pick_tensors(losses), "gradients": _gradient_summary(model),
        "state_after": _cpu(model.state_dict()), "log_vars": {key: float(value) for key, value in log_vars.items()},
    }
    destination = args.result_artifact or scratch / f"result-{args.device.split(':')[0]}.pt"
    torch.save(result, destination)
    print(json.dumps({"result_artifact": str(destination), **result["payload"], "losses": result["log_vars"]}, sort_keys=True))


if __name__ == "__main__":
    main()
