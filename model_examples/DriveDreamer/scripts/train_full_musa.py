#!/usr/bin/env python3
"""Launch the official DriveDreamer image trainer on a real converted dataset."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

UPSTREAM_REVISION = "da1ca92f831bc23d91b59ad418eb47b41cbb1fa9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config(checkout: Path) -> dict:
    path = checkout / "dreamer-train/projects/DriveDreamer/configs/drivedreamer-img_sd15_corners_hdmap_res448.py"
    spec = importlib.util.spec_from_file_location("drivedreamer_upstream_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load upstream config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def main() -> None:
    args = parse_args()
    checkout = args.checkout.resolve()
    data_root = args.data_root.resolve()
    models_dir = args.models_dir.resolve()
    output_dir = args.output_dir.resolve()
    actual = subprocess.check_output(
        ["git", "-c", f"safe.directory={checkout}", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != UPSTREAM_REVISION:
        raise RuntimeError(f"unexpected DriveDreamer revision: {actual}; expected {UPSTREAM_REVISION}")
    if not data_root.is_dir():
        raise RuntimeError(f"converted training data does not exist: {data_root}")
    sd15 = models_dir / "runwayml/stable-diffusion-v1-5"
    missing = [name for name in ("vae", "unet", "scheduler") if not (sd15 / name).is_dir()]
    if missing:
        raise RuntimeError("missing SD 1.5 model subfolders: " + ", ".join(missing))
    if not (models_dir / "openai/clip-vit-large-patch14").is_dir():
        raise RuntimeError("missing local openai/clip-vit-large-patch14 model")
    if args.epochs <= 0 or args.gpu_id < 0:
        raise ValueError("--epochs must be positive and --gpu-id must be non-negative")

    sys.path.insert(0, str(checkout))
    import ENV

    ENV.init_paths(base_dir=str(checkout), project_name="DriveDreamer")
    os.environ["DREAMER_MODELS_DIR"] = str(models_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch_musa  # noqa: F401
    except ModuleNotFoundError as error:
        raise RuntimeError("torch_musa is required for the full MUSA launch") from error
    import torch

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA device is unavailable")
    torch.manual_seed(args.seed)
    for backend_name in ("cuda", "cudnn", "musa", "mudnn"):
        backend = getattr(torch.backends, backend_name, None)
        if backend is not None and hasattr(backend, "allow_tf32"):
            backend.allow_tf32 = False
        matmul = getattr(backend, "matmul", None)
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = False

    from dreamer_train import Config
    from dreamer_train.distributed.run_task import run_tasks

    config = Config(load_config(checkout))
    config.project_dir = str(output_dir)
    config.launch.gpu_ids = [args.gpu_id]
    config.launch.num_machines = 1
    config.launch.distributed_type = None
    config.launch.deepspeed_config = None
    config.dataloaders.train.data_or_config = str(data_root)
    # The official mini config uses a constant default prompt. Cache its CLIP
    # embedding per worker to avoid reloading the text encoder for every sample.
    config.dataloaders.train.transform.with_cache = True
    config.train.max_epochs = args.epochs
    # The upstream trainer interprets interval=1 as every step when max_epochs=1;
    # retain epoch checkpoints for multi-epoch runs and save only at the end of
    # a one-epoch launch validation.
    config.train.checkpoint_interval = 10**9 if args.epochs == 1 else 1
    config.train.resume = args.resume
    config.train.mixed_precision = None
    config.train.allow_tf32 = False
    config.train.seed = args.seed
    config.models.pretrained = "runwayml/stable-diffusion-v1-5"
    print(f"DriveDreamer full image launch: revision={actual} data={data_root} output={output_dir}")
    run_tasks(config, "drivedreamer.DriveDreamerTrainer")


if __name__ == "__main__":
    main()
