"""Run one unoptimized BEVFormer-tiny optimizer step on MUSA."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import torch_musa  # noqa: F401
except ModuleNotFoundError:
    torch_musa = None  # type: ignore[assignment]


def _add_repo_paths(checkout: Path) -> None:
    script = Path(__file__).resolve()
    sys.path.insert(0, str(script.parent))
    for parent in script.parents:
        if (parent / "mu_space").is_dir():
            sys.path.insert(0, str(parent))
            break
    sys.path.insert(0, str(checkout))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixture", "full-random", "real"), default="fixture")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--classification-only", action="store_true")
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--device", choices=("musa", "cuda"), default="musa")
    return parser.parse_args()


def _configure_model(config, mode: str, train_backbone: bool) -> None:
    model = config.model
    model.pretrained = None
    model.img_backbone.pretrained = None
    if not train_backbone:
        # The unoptimized reference MSDA input-gradient path produces non-finite
        # image-backbone gradients on MUSA for both synthetic and real inputs.
        # The fixture therefore trains the detection stack by default; the
        # explicit diagnostic flag retains the official frozen_stages=1 policy.
        model.img_backbone.frozen_stages = 4
    if mode == "fixture":
        model.use_grid_mask = False
        head = model.pts_bbox_head
        head.bev_h = 8
        head.bev_w = 8
        head.num_query = 16
        head.transformer.encoder.num_layers = 1
        head.transformer.decoder.num_layers = 1
        head.positional_encoding.row_num_embed = 8
        head.positional_encoding.col_num_embed = 8


def _metadata(queue: int, cameras: int, height: int, width: int) -> list[list[dict]]:
    depth = 20.0
    projection = np.array(
        [
            [1.0, 0.0, 0.0, 0.5 * width * depth],
            [0.0, 1.0, 0.0, 0.5 * height * depth],
            [0.0, 0.0, 0.0, depth],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return [
        [
            {
                "can_bus": np.zeros(18, dtype=np.float32),
                "lidar2img": [projection.copy() for _ in range(cameras)],
                "img_shape": [(height, width, 3)] * cameras,
                "pad_shape": [(height, width, 3)] * cameras,
                "prev_bev_exists": frame > 0,
                "scene_token": "mu-space-fixed-fixture",
            }
            for frame in range(queue)
        ]
    ]


def _random_batch(mode: str, device: torch.device):
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    if mode == "fixture":
        queue, cameras, height, width = 2, 6, 64, 64
    else:
        queue, cameras, height, width = 3, 6, 450, 800
    image = torch.randn(1, queue, cameras, 3, height, width, device=device)
    boxes = [
        LiDARInstance3DBoxes(
            torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]), box_dim=9
        )
    ]
    labels = [torch.tensor([0], dtype=torch.long, device=device)]
    return {
        "img": image,
        "img_metas": _metadata(queue, cameras, height, width),
        "gt_bboxes_3d": boxes,
        "gt_labels_3d": labels,
    }


def _require_nuscenes_root(path: Path | None) -> Path:
    raw = path or (Path(os.environ["NUSCENES_ROOT"]) if "NUSCENES_ROOT" in os.environ else None)
    if raw is None:
        raise RuntimeError(
            "real mode requires --data-root or NUSCENES_ROOT; data is never downloaded"
        )
    root = raw.resolve()
    required = (root / "v1.0-mini", root / "samples", root / "nuscenes_infos_temporal_train.pkl")
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise RuntimeError("incomplete nuScenes mini mount; missing: " + ", ".join(missing))
    return root


def _real_batch(config, root: Path, device: torch.device):
    from mmcv.parallel import DataContainer, collate
    from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset

    train = config.data.train
    train.data_root = str(root) + "/"
    train.ann_file = str(root / "nuscenes_infos_temporal_train.pkl")
    dataset = custom_build_dataset(train)
    candidates = [
        index
        for index, info in enumerate(dataset.data_infos)
        if info["frame_idx"] >= dataset.queue_length
    ]
    sample = next(
        (
            prepared
            for index in candidates
            if (prepared := dataset.prepare_train_data(index)) is not None
        ),
        None,
    )
    if sample is None:
        raise RuntimeError("nuScenes mini has no valid full temporal training queue")
    packed = collate([sample], samples_per_gpu=1)

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

    return unwrap(packed)


def main() -> None:
    args = _parse_args()
    data_root = _require_nuscenes_root(args.data_root) if args.mode == "real" else None
    checkout = args.checkout.resolve()
    _add_repo_paths(checkout)
    from mu_space.patcher import default_patcher, modulated_deform_conv_patcher

    default_patcher.apply()
    # A no-op for the tiny fixture config, whose backbone is a plain ResNet-50. Configs
    # that declare dcn=dict(type='DCNv2', ...) resolve the layer through mmcv's registry
    # at build_model time, and mmcv's own DCNv2 raises on its disabled native extension.
    modulated_deform_conv_patcher.apply()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # mmdet3d 0.17.1 still uses the aliases removed by NumPy 1.24.
    if "bool" not in np.__dict__:
        np.bool = np.bool_  # type: ignore[attr-defined]

    from mmcv import Config
    from mmdet3d.models import build_model

    import projects.mmdet3d_plugin  # noqa: F401

    config = Config.fromfile(str(checkout / "projects/configs/bevformer/bevformer_tiny.py"))
    _configure_model(config, args.mode, args.train_backbone)
    logging.disable(logging.CRITICAL)
    model = build_model(
        config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg")
    )
    model.init_weights()
    logging.disable(logging.NOTSET)

    if args.device == "musa" and torch_musa is None:
        raise RuntimeError("torch_musa is required for --device musa")
    device = torch.device(f"{args.device}:0")
    model = model.to(device).train()
    eval_batch_norm_reference_count = 0
    if args.train_backbone and args.device == "musa":
        from mu_space import enable_eval_batch_norm_reference

        eval_batch_norm_reference_count = enable_eval_batch_norm_reference(model.img_backbone)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = (
        _real_batch(config, data_root, device)
        if args.mode == "real"
        else _random_batch(args.mode, device)
    )

    backend = getattr(torch, args.device)
    backend.reset_peak_memory_stats(device)
    backend.synchronize(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    losses = model(return_loss=True, **batch)
    tensor_losses = {key: value for key, value in losses.items() if torch.is_tensor(value)}
    selected_losses = (
        {key: value for key, value in tensor_losses.items() if key.endswith("loss_cls")}
        if args.classification_only
        else tensor_losses
    )
    loss = sum(selected_losses.values())
    if loss.device.type != args.device or not torch.isfinite(loss).item():
        raise RuntimeError(f"invalid loss on {loss.device}: {loss}")
    (loss * args.loss_scale).backward()
    nonfinite_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
    ]
    finite_gradients = not nonfinite_gradients
    if not finite_gradients:
        raise RuntimeError(
            f"{len(nonfinite_gradients)} non-finite model gradients; first: "
            + ", ".join(nonfinite_gradients[:10])
            + "; last: "
            + ", ".join(nonfinite_gradients[-10:])
        )
    optimizer.step()
    backend.synchronize(device)

    result = {
        "mode": args.mode,
        "backbone_frozen": config.model.img_backbone.frozen_stages == 4,
        "eval_batch_norm_reference_count": eval_batch_norm_reference_count,
        "seed": args.seed,
        "device": str(loss.device),
        "loss": float(loss.detach()),
        "loss_scale": args.loss_scale,
        "losses": {key: float(value.detach()) for key, value in tensor_losses.items()},
        "selected_losses": sorted(selected_losses),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_memory_bytes": backend.max_memory_allocated(device),
        "optimizer_step": True,
        "finite_gradients": finite_gradients,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
