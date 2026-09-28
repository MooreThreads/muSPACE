#!/usr/bin/env python3
"""Run and record one real DriveDreamer image-stage FP32 optimizer step.

This intentionally mirrors ``DriveDreamerTrainer.forward_step`` instead of using a
synthetic UNet substitute.  The portable input artifact fixes the real collated batch,
model initialization, VAE sampling noise, diffusion noise, and timesteps for MUSA/A100
comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

UPSTREAM_REVISION = "da1ca92f831bc23d91b59ad418eb47b41cbb1fa9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="converted cam_all_train directory produced by prepare_assets.sh")
    parser.add_argument("--models-dir", type=Path, required=True,
                        help="model root containing runwayml/stable-diffusion-v1-5 and openai/clip-vit-large-patch14")
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--device", choices=("musa", "cuda"), default="musa")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    return parser.parse_args()


def add_upstream_paths(checkout: Path) -> None:
    for path in (
        checkout / "dreamer-datasets",
        checkout / "dreamer-models",
        checkout / "dreamer-train",
        checkout / "dreamer-train" / "projects" / "DriveDreamer",
    ):
        sys.path.insert(0, str(path))


def cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def device_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {key: device_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [device_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(device_tree(item, device) for item in value)
    return value


def float_tree(value: Any) -> Any:
    """Match dreamer_train.utils.to_dtype without importing distributed helpers."""
    if torch.is_tensor(value):
        return value.to(dtype=torch.float32) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: float_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [float_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(float_tree(item) for item in value)
    return value


def set_reproducibility(seed: int) -> dict[str, bool]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    configured: dict[str, bool] = {}
    for backend_name in ("cuda", "cudnn", "musa", "mudnn"):
        backend = getattr(torch.backends, backend_name, None)
        if backend is not None and hasattr(backend, "allow_tf32"):
            backend.allow_tf32 = False
            configured[f"torch.backends.{backend_name}.allow_tf32"] = bool(backend.allow_tf32)
        matmul = getattr(backend, "matmul", None)
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = False
            configured[f"torch.backends.{backend_name}.matmul.allow_tf32"] = bool(matmul.allow_tf32)
    return configured


def revision(checkout: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={checkout.resolve()}", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot determine checkout revision: {error}") from error


def validate_assets(args: argparse.Namespace) -> None:
    if not args.checkout.is_dir():
        raise RuntimeError(f"DriveDreamer checkout does not exist: {args.checkout}")
    if revision(args.checkout) != UPSTREAM_REVISION:
        raise RuntimeError("unexpected DriveDreamer revision; run prepare_checkout.sh first")
    if not args.data_root.is_dir() and not args.input_artifact.exists():
        raise RuntimeError(f"converted training data does not exist: {args.data_root}")
    sd15 = args.models_dir / "runwayml" / "stable-diffusion-v1-5"
    missing = [name for name in ("vae", "unet", "scheduler") if not (sd15 / name).is_dir()]
    if missing:
        raise RuntimeError("invalid local Stable Diffusion 1.5 directory; missing " + ", ".join(missing))
    if not args.input_artifact.exists() and not (args.models_dir / "openai" / "clip-vit-large-patch14").is_dir():
        raise RuntimeError("missing local openai/clip-vit-large-patch14 required by the official transform")


def build_batch(data_root: Path, sample_index: int) -> dict[str, Any]:
    from dreamer_datasets import DefaultCollator, load_dataset
    from drivedreamer.transforms import DriveDreamerTransform

    dataset = load_dataset(str(data_root))
    if not 0 <= sample_index < len(dataset):
        raise RuntimeError(f"sample index {sample_index} is outside converted dataset length {len(dataset)}")
    dataset.set_transform(DriveDreamerTransform(
        dst_size=448, mode="long", pos_name="corner", max_objs=100,
        random_choice=False, default_prompt="a realistic driving scene.",
        prompt_name="sd", dd_name="image_hdmap", is_train=True,
    ))
    return DefaultCollator()([dataset[sample_index]])


def build_models(pretrained_dir: Path, *, replay_initial_state: bool):
    from diffusers import AutoencoderKL, DDPMScheduler
    from dreamer_models import DriveDreamerModel, UNet2DConditionModel

    vae = AutoencoderKL.from_pretrained(str(pretrained_dir), subfolder="vae", local_files_only=True)
    scheduler = DDPMScheduler.from_pretrained(str(pretrained_dir), subfolder="scheduler", local_files_only=True)
    if replay_initial_state:
        # The portable artifact contains the exact FP32 UNet state.  Loading
        # the architecture config only avoids copying a duplicate SD UNet to
        # the A100 reference host while retaining the real execution graph.
        unet = UNet2DConditionModel.from_config(str(pretrained_dir / "unet"))
    else:
        unet = UNet2DConditionModel.from_pretrained(
            str(pretrained_dir), subfolder="unet", local_files_only=True
        )
    unet.requires_grad_(False)
    drivedreamer = DriveDreamerModel.from_unet(
        unet,
        position_net_cfg={"type": "PositionNet", "in_dim": 768, "mid_dim": 512, "box_dim": 16,
                          "feature_type": "text_only"},
        grounding_downsampler_cfg={"type": "GroundingDownSampler", "in_dim": 3, "mid_dim": 4, "out_dim": 8},
        add_in_channels=8,
    )
    drivedreamer.to_unet(unet)
    vae.requires_grad_(False).eval()
    unet.train()
    drivedreamer.train()
    return vae, scheduler, unet, drivedreamer


def make_input_artifact(args: argparse.Namespace, unet, drivedreamer) -> dict[str, Any]:
    batch = cpu_tree(build_batch(args.data_root, args.sample_index))
    image = batch.get("image")
    if not torch.is_tensor(image) or image.ndim != 4:
        raise RuntimeError("official DriveDreamer transform did not produce a BCHW image tensor")
    # Stable Diffusion v1 VAE downsamples by eight and has four latent channels.
    latent_shape = (image.shape[0], 4, image.shape[-2] // 8, image.shape[-1] // 8)
    return {
        "schema_version": 1,
        "upstream_revision": UPSTREAM_REVISION,
        "seed": args.seed,
        "sample_index": args.sample_index,
        "batch": batch,
        "vae_epsilon": torch.randn(latent_shape, dtype=torch.float32),
        "diffusion_noise": torch.randn(latent_shape, dtype=torch.float32),
        "timesteps": torch.randint(0, 1000, (image.shape[0],), dtype=torch.long),
        "initial_state": {"unet": cpu_tree(unet.state_dict()), "drivedreamer": cpu_tree(drivedreamer.state_dict())},
    }


def tracked_parameter(model: torch.nn.Module) -> tuple[str, torch.nn.Parameter]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            return name, parameter
    raise RuntimeError("DriveDreamer has no trainable parameter")


def run_step(payload: dict[str, Any], vae, scheduler, unet, drivedreamer, device: torch.device, lr: float):
    batch = device_tree(payload["batch"], device)
    vae = vae.to(device=device, dtype=torch.float32)
    unet = unet.to(device=device, dtype=torch.float32)
    drivedreamer = drivedreamer.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW((p for p in drivedreamer.parameters() if p.requires_grad), lr=lr, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)

    image = batch["image"].to(dtype=torch.float32)
    with torch.no_grad():
        posterior = vae.encode(image).latent_dist
        vae_epsilon = payload["vae_epsilon"].to(device=device, dtype=posterior.mean.dtype)
        if posterior.mean.shape != vae_epsilon.shape:
            raise RuntimeError(f"saved VAE noise shape {tuple(vae_epsilon.shape)} differs from posterior {tuple(posterior.mean.shape)}")
        latents = (posterior.mean + posterior.std * vae_epsilon) * vae.config.scaling_factor
    noise = payload["diffusion_noise"].to(device=device, dtype=torch.float32)
    timesteps = payload["timesteps"].to(device=device)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    grounding_input = batch["grounding_downsampler_input"].to(dtype=torch.float32)
    grounding = drivedreamer("grounding_downsampler", grounding_input).to(dtype=torch.float32)
    noisy_latents = torch.cat((noisy_latents, grounding), dim=1)
    position_input = float_tree(device_tree(batch["position_net_input"], device))
    positions = drivedreamer("position_net", **position_input).to(dtype=torch.float32)
    prediction = unet(
        noisy_latents, timesteps, encoder_hidden_states=batch["prompt_embeds"].to(dtype=torch.float32),
        cross_attention_kwargs={"gligen": {"objs": positions}},
    ).sample
    prediction = prediction.reshape(noise.shape)
    loss = F.mse_loss(prediction.float(), noise.float(), reduction="mean")
    if loss.device.type != device.type or not torch.isfinite(loss):
        raise RuntimeError(f"invalid loss on {loss.device}: {loss}")
    name, parameter = tracked_parameter(drivedreamer)
    before = parameter.detach().clone()
    loss.backward()
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise RuntimeError(f"invalid gradient for tracked parameter {name}")
    gradient = parameter.grad.detach().clone()
    optimizer.step()
    after = parameter.detach().clone()
    if torch.equal(before, after):
        raise RuntimeError(f"optimizer did not update tracked parameter {name}")
    stages = {
        "vae_latents": cpu_tree(latents), "grounding_latents": cpu_tree(grounding),
        "position_features": cpu_tree(positions), "model_prediction": cpu_tree(prediction),
        "tracked_gradient": cpu_tree(gradient), "tracked_parameter_before": cpu_tree(before),
        "tracked_parameter_after": cpu_tree(after), "tracked_parameter_delta": cpu_tree(after - before),
    }
    return float(loss.detach().cpu()), name, stages


def main() -> None:
    args = parse_args()
    validate_assets(args)
    add_upstream_paths(args.checkout.resolve())
    if args.device == "musa":
        try:
            import torch_musa  # noqa: F401
        except ModuleNotFoundError as error:
            raise RuntimeError("torch_musa is required for --device musa") from error
        if not hasattr(torch, "musa") or not torch.musa.is_available():
            raise RuntimeError("MUSA device is unavailable")
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable")
    tf32 = set_reproducibility(args.seed)
    device = torch.device(f"{args.device}:0")
    os.environ["DREAMER_MODELS_DIR"] = str(args.models_dir.resolve())
    sd15_dir = args.models_dir / "runwayml" / "stable-diffusion-v1-5"
    if args.input_artifact.exists():
        payload = torch.load(args.input_artifact, map_location="cpu", weights_only=False)
        if payload.get("upstream_revision") != UPSTREAM_REVISION:
            raise RuntimeError("input artifact was created for a different DriveDreamer revision")
        vae, scheduler, unet, drivedreamer = build_models(sd15_dir, replay_initial_state=True)
        unet.load_state_dict(payload["initial_state"]["unet"], strict=True)
        drivedreamer.load_state_dict(payload["initial_state"]["drivedreamer"], strict=True)
    else:
        vae, scheduler, unet, drivedreamer = build_models(sd15_dir, replay_initial_state=False)
        payload = make_input_artifact(args, unet, drivedreamer)
        args.input_artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.input_artifact)
    loss, parameter_name, stages = run_step(payload, vae, scheduler, unet, drivedreamer, device, args.learning_rate)
    result = {
        "schema_version": 1,
        "payload": {"upstream_revision": UPSTREAM_REVISION, "seed": payload["seed"],
                    "sample_index": payload["sample_index"], "loss": loss, "tracked_parameter": parameter_name},
        "environment": {"device": str(device), "torch": torch.__version__,
                        "torch_musa": getattr(sys.modules.get("torch_musa"), "__version__", None),
                        "tf32": tf32, "muspace_patchers": []},
        "stages": stages,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.artifact)
    print(json.dumps({"artifact": str(args.artifact), **result["payload"], "device": str(device)}, sort_keys=True))


if __name__ == "__main__":
    main()
