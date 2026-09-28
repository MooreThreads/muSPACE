#!/usr/bin/env python3
"""Run deterministic real-data BEVDepth training on MUSA or upstream CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch


UPSTREAM_REVISION = "d78c7b58b10b9ada940462ba83ab24d99cae5833"
EXPERIMENT_MODULE = "bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float16"), required=True)
    parser.add_argument("--device", choices=("musa", "cuda"), default="musa")
    parser.add_argument("--scene-name", default="scene-0001")
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON artifact path")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def verify_revision(checkout: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={checkout}", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"expected {UPSTREAM_REVISION}, got {revision}")
    return revision


def move_to(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=False)
    if isinstance(value, Mapping):
        return type(value)((key, move_to(item, device)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(move_to(item, device) for item in value)
    if isinstance(value, list):
        return [move_to(item, device) for item in value]
    return value


def find_optimizer(value: Any) -> torch.optim.Optimizer:
    if isinstance(value, torch.optim.Optimizer):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return find_optimizer(item)
            except LookupError:
                pass
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return find_optimizer(item)
            except LookupError:
                pass
    raise LookupError("BEVDepth configure_optimizers did not return an optimizer")


def loss_value(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping) and torch.is_tensor(value.get("loss")):
        return value["loss"]
    raise TypeError("BEVDepth training_step did not return a loss tensor")


def configure_device(kind: str) -> torch.device:
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable")
        return torch.device("cuda:0")
    import torch_musa  # noqa: F401

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA device is unavailable")
    if hasattr(torch.backends, "musa"):
        torch.backends.musa.matmul.allow_tf32 = False
    if hasattr(torch.backends, "mudnn"):
        torch.backends.mudnn.allow_tf32 = False
    return torch.device("musa:0")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_experiment(data_root: Path, num_workers: int):
    module = importlib.import_module(EXPERIMENT_MODULE)
    model_class = getattr(module, "BEVDepthLightningModel")
    experiment = model_class(gpus=1, data_root=str(data_root), batch_size_per_device=1)
    # The pinned experiment exposes this mutable loader setting.  Keep loading
    # deterministic and avoid worker-local random augmentation in the smoke run.
    experiment.num_workers = num_workers
    return experiment


def deterministic_train_loader(experiment, num_workers: int):
    """Rebuild the upstream loader with one ordered sample and no worker RNG by default."""

    source = experiment.train_dataloader()
    return torch.utils.data.DataLoader(
        source.dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=source.collate_fn,
        pin_memory=False,
    )


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.max_hours <= 0 or args.max_steps < 0:
        raise ValueError("epochs must be >=1, max-hours >0 and max-steps >=0")
    checkout, data_root = args.checkout.resolve(), args.data_root.resolve()
    revision = verify_revision(checkout)
    if not (data_root / "nuscenes_infos_train.pkl").is_file():
        raise FileNotFoundError("prepare_scene_data.py must create nuscenes_infos_train.pkl first")
    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root), str(checkout)]
    os.environ.setdefault("MUSPACE_STRICT_PATCH", "1")
    device = configure_device(args.device)

    from mu_space import clear_dispatch_report, get_dispatch_reports
    from mu_space.patcher import BEVDepthPatcher, deform_conv_patcher

    patcher = None
    dcn_patcher = None
    if args.device == "musa":
        patcher = BEVDepthPatcher()
        patcher.apply(strict=True)
        # Bind mmcv's standard DCN after mmcv is importable and before model construction.
        import mmcv  # noqa: F401

        dcn_patcher = deform_conv_patcher
        dcn_patcher.apply(strict=True)
    else:
        # CUDA runs the pinned upstream extension without loading the muSPACE patchers.
        import mmcv  # noqa: F401

    clear_dispatch_report()

    try:
        seed_everything(args.seed)
        experiment = build_experiment(data_root, args.num_workers)
        experiment = experiment.to(device).train()
        optimizer = find_optimizer(experiment.configure_optimizers())
        scaler = None
        if args.dtype == "float16":
            scaler = (
                torch.musa.amp.GradScaler()
                if args.device == "musa"
                else torch.cuda.amp.GradScaler()
            )
        selected_name, selected = next(
            (name, parameter)
            for name, parameter in experiment.named_parameters()
            if parameter.requires_grad
        )
        before = selected.detach().float().cpu().clone()

        def autocast_context():
            # The upstream revision disables ``torch.cuda.amp.autocast`` around
            # BCE depth loss.  On MUSA the patcher maps that legacy context to
            # the active MUSA autocast implementation; CUDA keeps the original.
            return (
                torch.autocast(device_type=args.device, dtype=torch.float16)
                if args.dtype == "float16"
                else nullcontext()
            )

        deadline = time.monotonic() + args.max_hours * 3600
        epoch_losses: list[float] = []
        step_losses: list[float] = []
        steps = 0
        for epoch in range(args.epochs):
            total = 0.0
            count = 0
            for batch in deterministic_train_loader(experiment, args.num_workers):
                if time.monotonic() >= deadline or (args.max_steps and steps >= args.max_steps):
                    break
                optimizer.zero_grad(set_to_none=True)
                batch = move_to(batch, device)
                with autocast_context():
                    loss = loss_value(experiment.training_step(batch))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"BEVDepth loss is non-finite at step {steps}")
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                total += float(loss.detach().float().cpu())
                step_losses.append(float(loss.detach().float().cpu()))
                count += 1
                steps += 1
            if count == 0:
                raise RuntimeError("BEVDepth produced no training batches before the time limit")
            epoch_losses.append(total / count)
            if time.monotonic() >= deadline or (args.max_steps and steps >= args.max_steps):
                break
        after = selected.detach().float().cpu().clone()
        if torch.equal(before, after):
            raise RuntimeError("BEVDepth optimizer did not update the selected parameter")
        reports = get_dispatch_reports()
        native_reports = [
            report
            for report in reports
            if report.operator in {"voxel_pooling_train", "deform_conv2d"}
        ]
        if args.device == "musa":
            required = {"voxel_pooling_train", "deform_conv2d"}
            observed = {report.operator for report in native_reports}
            if observed != required or any(
                report.selected_backend != "musa"
                or report.native_fallback
                or not report.native_available
                for report in native_reports
            ):
                raise RuntimeError(
                    "BEVDepth training did not use strict muSPACE native voxel pooling and DCN"
                )
        finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in experiment.parameters()
        )
        if not finite:
            raise RuntimeError("BEVDepth contains non-finite gradients")
        payload = {
            "device": str(device),
            "dtype": args.dtype,
            "scene": args.scene_name,
            "seed": args.seed,
            "upstream_revision": revision,
            "data_sha256": sha256_file(data_root / "nuscenes_infos_train.pkl"),
            "torch_version": torch.__version__,
            "loss": epoch_losses[-1],
            "epoch_losses": epoch_losses,
            "step_losses": step_losses,
            "epochs_completed": len(epoch_losses),
            "steps": steps,
            "selected_parameter": selected_name,
            "selected_parameter_before_sha256": hashlib.sha256(
                before.numpy().tobytes()
            ).hexdigest(),
            "selected_parameter_after_sha256": hashlib.sha256(after.numpy().tobytes()).hexdigest(),
            "finite_gradients": bool(finite),
            "optimizer_step": True,
            "dispatch": [report.as_dict() for report in native_reports],
        }
        encoded = json.dumps(payload, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n")
        print(encoded)
    finally:
        if dcn_patcher is not None:
            dcn_patcher.restore()
        if patcher is not None:
            patcher.restore()


if __name__ == "__main__":
    main()
