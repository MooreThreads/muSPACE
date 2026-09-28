#!/usr/bin/env python3
"""Compare the patched BEVFusion focal-loss call with the CUDA expression."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def reference(inputs: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - pt) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss


def metrics(actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> dict[str, object]:
    delta = (actual.float() - expected.float()).abs()
    relative = delta / expected.float().abs().clamp_min(1e-12)
    return {
        "max_absolute_error": float(delta.max()) if delta.numel() else 0.0,
        "max_relative_error": float(relative.max()) if relative.numel() else 0.0,
        "mean_absolute_error": float(delta.mean()) if delta.numel() else 0.0,
        "mismatch_count": int((delta > tolerance).sum()),
        "tolerance": tolerance,
        "pass": bool(torch.all(delta <= tolerance)),
    }


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    shape = (1, args.classes, args.height, args.width)
    cpu_inputs = torch.randn(shape, dtype=args.dtype)
    cpu_targets = torch.randint(0, 2, shape, dtype=torch.int64).to(args.dtype)
    expected = reference(cpu_inputs, cpu_targets, args.alpha, args.gamma)

    if args.backend == "musa":
        if args.checkout is None:
            raise SystemExit("--checkout is required for --backend musa")
        root = Path(__file__).resolve()
        for parent in root.parents:
            if (parent / "mu_space").is_dir():
                sys.path.insert(0, str(parent))
                break
        sys.path.insert(0, str(args.checkout.resolve()))
        os.environ["MUSPACE_SIGMOID_FOCAL_LOSS_BACKEND"] = "musa"
        import subprocess

        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("apply_source_patch.py")),
             "--checkout", str(args.checkout), "--action", "apply"], check=True
        )
        from train_step_musa import install_camera_only_import_surface

        install_camera_only_import_surface(args.checkout.resolve())
        from mmdet3d.models.heads.segm.vanilla import sigmoid_focal_loss
        from mu_space import get_last_dispatch_report
        device = torch.device("musa:0")
        inputs = cpu_inputs.to(device).requires_grad_()
        targets = cpu_targets.to(device)
        actual = sigmoid_focal_loss(inputs, targets, args.alpha, args.gamma, "none")
        actual.sum().backward()
        grad = inputs.grad.detach().cpu()
        dispatch = get_last_dispatch_report()
        selected_backend = None if dispatch is None else dispatch.selected_backend
    else:
        device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable")
        inputs = cpu_inputs.to(device).requires_grad_()
        targets = cpu_targets.to(device)
        actual = reference(inputs, targets, args.alpha, args.gamma)
        actual.sum().backward()
        grad = inputs.grad.detach().cpu()
        selected_backend = "cuda-reference"

    ref_inputs = cpu_inputs.detach().float().requires_grad_()
    reference(ref_inputs, cpu_targets, args.alpha, args.gamma).sum().backward()
    payload = {
        "backend": args.backend,
        "dtype": str(args.dtype).replace("torch.", ""),
        "shape": list(shape),
        "seed": args.seed,
        "selected_backend": selected_backend,
        "forward": metrics(actual.detach().cpu(), expected, args.tolerance),
        "gradient": metrics(grad, ref_inputs.grad, args.tolerance),
    }
    if args.artifact:
        torch.save({"inputs": cpu_inputs, "targets": cpu_targets,
                    "output": actual.detach().cpu(), "gradient": grad}, args.artifact)
    print(json.dumps(payload, sort_keys=True))


def compare(args: argparse.Namespace) -> None:
    left = torch.load(args.left, map_location="cpu", weights_only=True)
    right = torch.load(args.right, map_location="cpu", weights_only=True)
    payload = {
        "forward": metrics(left["output"], right["output"], args.tolerance),
        "gradient": metrics(left["gradient"], right["gradient"], args.tolerance),
        "shape": list(left["output"].shape),
        "tolerance": args.tolerance,
    }
    payload["pass"] = payload["forward"]["pass"] and payload["gradient"]["pass"]
    print(json.dumps(payload, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("musa", "cuda"))
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--tolerance", type=float, default=3e-3)
    args = parser.parse_args()
    args.dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.left and args.right:
        compare(args)
    elif args.backend:
        run(args)
    else:
        parser.error("provide --backend or both --left and --right")


if __name__ == "__main__":
    main()
