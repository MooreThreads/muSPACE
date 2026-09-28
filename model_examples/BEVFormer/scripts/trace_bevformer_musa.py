"""Trace one BEVFormer forward pass with compact tensor summaries."""

from __future__ import annotations

import argparse
import json
import logging
import random
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

try:
    import torch_musa  # noqa: F401
except ModuleNotFoundError:
    torch_musa = None  # type: ignore[assignment]

from train_step_musa import _configure_model, _random_batch, _real_batch, _require_nuscenes_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixture", "full-random", "real"), default="real")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--sample-elements", type=int, default=4096)
    parser.add_argument("--trace-grid-sample", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("musa", "cuda"), default="musa")
    return parser.parse_args()


def _tensor_summary(tensor: torch.Tensor, sample_elements: int) -> dict:
    with torch.no_grad():
        flat = tensor.detach().reshape(-1)
        sample = flat[:sample_elements].float().cpu()
        finite = torch.isfinite(sample)
        finite_sample = sample[finite]
        summary = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": int(tensor.numel()),
            "sample_numel": int(sample.numel()),
            "sample_finite": int(finite.sum().item()),
            "sample_sum": float(sample.sum().item()) if sample.numel() else 0.0,
            "sample_l1": float(sample.abs().sum().item()) if sample.numel() else 0.0,
            "sample_values": sample.tolist(),
        }
        if finite_sample.numel():
            summary.update(
                {
                    "sample_mean": float(finite_sample.mean().item()),
                    "sample_std": float(finite_sample.std(unbiased=False).item()),
                    "sample_min": float(finite_sample.min().item()),
                    "sample_max": float(finite_sample.max().item()),
                }
            )
        return summary


def _summarize(value, sample_elements: int, prefix: str = "out", limit: int = 8) -> list[dict]:
    if torch.is_tensor(value):
        item = _tensor_summary(value, sample_elements)
        item["path"] = prefix
        return [item]
    if isinstance(value, Mapping):
        records = []
        for key in sorted(value):
            records.extend(_summarize(value[key], sample_elements, f"{prefix}.{key}", limit))
            if len(records) >= limit:
                return records[:limit]
        return records
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        records = []
        for index, item in enumerate(value):
            records.extend(_summarize(item, sample_elements, f"{prefix}.{index}", limit))
            if len(records) >= limit:
                return records[:limit]
        return records
    return []


def _should_trace(name: str, module: torch.nn.Module) -> bool:
    if not name:
        return False
    key_names = {
        "img_backbone",
        "img_backbone.conv1",
        "img_backbone.maxpool",
        "img_backbone.layer1",
        "img_backbone.layer2",
        "img_backbone.layer3",
        "img_backbone.layer4",
        "img_neck",
        "pts_bbox_head",
        "pts_bbox_head.transformer",
        "pts_bbox_head.transformer.encoder",
        "pts_bbox_head.transformer.decoder",
    }
    if name in key_names:
        return True
    if name.startswith("pts_bbox_head.transformer.encoder.layers."):
        return True
    if name.startswith("pts_bbox_head.transformer.decoder.layers."):
        return True
    return not any(module.children())


def _state_summaries(model: torch.nn.Module, sample_elements: int) -> dict[str, dict]:
    interesting = (
        "img_backbone.conv1.weight",
        "img_backbone.layer1.0.conv1.weight",
        "img_backbone.layer2.0.conv1.weight",
        "img_neck.lateral_convs.0.conv.weight",
        "pts_bbox_head.cls_branches.0.0.weight",
        "pts_bbox_head.reg_branches.0.0.weight",
    )
    params = dict(model.named_parameters())
    return {
        name: _tensor_summary(params[name], sample_elements)
        for name in interesting
        if name in params
    }


@contextmanager
def _capture_grid_sample(sample_elements: int):
    import mmcv.ops.multi_scale_deform_attn as msda

    records = []
    original = msda.F.grid_sample

    def wrapped(input, grid, *args, **kwargs):
        call = {
            "input": _tensor_summary(input, sample_elements),
            "grid": _tensor_summary(grid, sample_elements),
        }
        output = original(input, grid, *args, **kwargs)
        call["output"] = _tensor_summary(output, sample_elements)
        call["index"] = len(records)
        records.append(call)
        return output

    msda.F.grid_sample = wrapped
    try:
        yield records
    finally:
        msda.F.grid_sample = original


@contextmanager
def _capture_point_sampling(model: torch.nn.Module, sample_elements: int):
    encoder = model.pts_bbox_head.transformer.encoder
    original = encoder.point_sampling
    records = []

    def wrapped(reference_points, pc_range, img_metas):
        record = {
            "reference_points": _tensor_summary(reference_points, sample_elements),
            "num_img_metas": len(img_metas),
        }
        output = original(reference_points, pc_range, img_metas)
        record["reference_points_cam"] = _tensor_summary(output[0], sample_elements)
        record["bev_mask"] = _tensor_summary(output[1], sample_elements)
        records.append(record)
        return output

    encoder.point_sampling = wrapped
    try:
        yield records
    finally:
        encoder.point_sampling = original


def main() -> None:
    args = _parse_args()
    data_root = _require_nuscenes_root(args.data_root) if args.mode == "real" else None
    checkout = args.checkout.resolve()
    from train_step_musa import _add_repo_paths

    _add_repo_paths(checkout)
    from mu_space.patcher import default_patcher, modulated_deform_conv_patcher

    default_patcher.apply()
    # A no-op for the tiny fixture config; see train_step_musa.py.
    modulated_deform_conv_patcher.apply()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
    state = _state_summaries(model, args.sample_elements)
    logging.disable(logging.NOTSET)

    if args.device == "musa" and torch_musa is None:
        raise RuntimeError("torch_musa is required for --device musa")
    device = torch.device(f"{args.device}:0")
    model = model.to(device).train()
    eval_batch_norm_reference_count = 0
    if args.train_backbone and args.device == "musa":
        from mu_space import enable_eval_batch_norm_reference

        eval_batch_norm_reference_count = enable_eval_batch_norm_reference(model.img_backbone)

    records = []
    handles = []
    order = {"value": 0}

    def hook(name: str):
        def collect(_module, _inputs, output):
            order["value"] += 1
            records.append(
                {
                    "order": order["value"],
                    "name": name,
                    "inputs": _summarize(_inputs, args.sample_elements, prefix="in"),
                    "outputs": _summarize(output, args.sample_elements),
                }
            )

        return collect

    for name, module in model.named_modules():
        if _should_trace(name, module):
            handles.append(module.register_forward_hook(hook(name)))

    batch = (
        _real_batch(config, data_root, device)
        if args.mode == "real"
        else _random_batch(args.mode, device)
    )
    if args.trace_grid_sample:
        with _capture_grid_sample(args.sample_elements) as grid_records:
            with _capture_point_sampling(model, args.sample_elements) as point_records:
                with torch.enable_grad():
                    losses = model(return_loss=True, **batch)
    else:
        grid_records = []
        with _capture_point_sampling(model, args.sample_elements) as point_records:
            with torch.enable_grad():
                losses = model(return_loss=True, **batch)
    for handle in handles:
        handle.remove()

    tensor_losses = {key: value for key, value in losses.items() if torch.is_tensor(value)}
    loss = sum(tensor_losses.values())
    backend = getattr(torch, args.device)
    backend.synchronize(device)
    result = {
        "torch": {
            "version": torch.__version__,
            "git_version": torch.version.git_version,
        },
        "torch_musa": getattr(torch_musa, "__version__", None),
        "mode": args.mode,
        "seed": args.seed,
        "train_backbone": args.train_backbone,
        "eval_batch_norm_reference_count": eval_batch_norm_reference_count,
        "state": state,
        "loss": float(loss.detach()),
        "losses": {key: float(value.detach()) for key, value in tensor_losses.items()},
        "records": records,
        "grid_sample": grid_records,
        "point_sampling": point_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
