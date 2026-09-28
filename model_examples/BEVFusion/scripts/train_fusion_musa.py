#!/usr/bin/env python3
"""Run one pinned BEVFusion camera+LiDAR training step and postprocess pass.

The script deliberately uses the upstream config and dataset sample.  It only filters
the prepared training infos to one scene, so it is a launch/regression check rather
than a synthetic model substitute.
"""

from __future__ import annotations
from contextlib import nullcontext

import argparse
import copy
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# BEVFusion's pinned preprocessing still references the NumPy-1.23 alias.
if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]
if "long" not in np.__dict__:
    np.long = np.int64  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_REVISION = "326653dc06e0938edf1aae7d01efcd158ba83de5"


def _scratch_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="bevfusion-", dir=parent))


def _subset_infos(data_root: Path, scene_name: str, work_dir: Path) -> Path:
    candidates = sorted(data_root.glob("*infos_train*.pkl"))
    if candidates:
        source = candidates[0]
        with source.open("rb") as stream:
            payload = pickle.load(stream)
        infos = payload["infos"] if isinstance(payload, dict) else payload
        filtered = [
            info for info in infos
            if info.get("scene_name") == scene_name or info.get("scene_token") == scene_name
        ]
        if not filtered:
            raise ValueError(f"prepared infos {source} contain no samples from {scene_name}")
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["infos"] = filtered
        else:
            payload = filtered
        output_name = source.name
    else:
        from nuscenes import NuScenes
        from tools.data_converter.nuscenes_converter import _fill_trainval_infos

        nusc = NuScenes(version="v1.0-trainval", dataroot=str(data_root), verbose=False)
        scene = next((item for item in nusc.scene if item["name"] == scene_name), None)
        if scene is None:
            raise FileNotFoundError(f"scene {scene_name!r} is absent from {data_root}")
        original_samples = nusc.sample
        nusc.sample = [item for item in original_samples if item["scene_token"] == scene["token"]]
        try:
            filtered, _ = _fill_trainval_infos(
                nusc, {scene["token"]}, set(), test=False, max_sweeps=0, max_radar_sweeps=0
            )
        finally:
            nusc.sample = original_samples
        if not filtered:
            raise ValueError(f"no metadata was generated for {scene_name}")
        payload = {"infos": filtered, "metadata": {"version": "v1.0-trainval"}}
        output_name = "nuscenes_infos_train.pkl"
    for info in filtered:
        info.pop("radar", None)
        info.pop("radars", None)
    if any("location" not in info for info in filtered):
        from nuscenes import NuScenes

        nusc = NuScenes(version="v1.0-trainval", dataroot=str(data_root), verbose=False)
        locations = {
            scene["token"]: nusc.get("log", scene["log_token"])["location"]
            for scene in nusc.scene
        }
        for info in filtered:
            sample = nusc.get("sample", info["token"])
            info["location"] = locations[sample["scene_token"]]
    output = _scratch_dir(work_dir) / output_name
    with output.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return output


def _remove_unused_radar_pipeline(pipeline):
    """The selected BEVFusion fixture has camera/LiDAR encoders only."""
    if isinstance(pipeline, list):
        pipeline[:] = [
            item for item in pipeline
            if not (
                isinstance(item, dict)
                and item.get("type") in {"LoadRadarPointsMultiSweeps", "ObjectPaste"}
            )
        ]
        for item in pipeline:
            _remove_unused_radar_pipeline(item)
    elif isinstance(pipeline, dict):
        if pipeline.get("type") == "Collect3D" and "keys" in pipeline:
            pipeline["keys"] = [key for key in pipeline["keys"] if key != "radar"]
        for value in pipeline.values():
            _remove_unused_radar_pipeline(value)


def _set_dataset_paths(config, data_root: Path, ann_file: Path) -> None:
    """Set every nested NuScenes dataset leaf to the one-scene fixture."""
    if isinstance(config, dict):
        if "dataset_root" in config:
            config["dataset_root"] = f"{data_root}/"
        if config.get("type") == "NuScenesDataset":
            if "data_root" in config:
                config["data_root"] = f"{data_root}/"
            config["ann_file"] = str(ann_file)
        for value in config.values():
            _set_dataset_paths(value, data_root, ann_file)
    elif isinstance(config, list):
        for value in config:
            _set_dataset_paths(value, data_root, ann_file)


def _unwrap_scene_only_wrappers(config) -> None:
    """CBGS requires every class and is undefined for a one-scene subset."""
    if isinstance(config, dict):
        if config.get("type") == "CBGSDataset":
            dataset = config["dataset"]
            config.clear()
            config.update(dataset)
        for value in config.values():
            _unwrap_scene_only_wrappers(value)
    elif isinstance(config, list):
        for value in config:
            _unwrap_scene_only_wrappers(value)


def _normalize_voxel_limits(config) -> None:
    """Upstream stores train/test max_voxels as a pair; native op takes one int."""
    if isinstance(config, dict):
        if "max_voxels" in config and isinstance(config["max_voxels"], (list, tuple)):
            config["max_voxels"] = config["max_voxels"][0]
        for value in config.values():
            _normalize_voxel_limits(value)
    elif isinstance(config, list):
        for value in config:
            _normalize_voxel_limits(value)


def _unwrap(value, device):
    import torch
    from mmcv.parallel import DataContainer

    if isinstance(value, DataContainer):
        value = value.data[0]
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _unwrap(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap(item, device) for item in value)
    return value


def _tensor_summary(value):
    import torch

    if isinstance(value, dict):
        return {key: _tensor_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_summary(item) for item in value]
    if not torch.is_tensor(value):
        return (
            value
            if value is None or isinstance(value, (str, int, float, bool))
            else type(value).__name__
        )
    result = value.detach().float().cpu()
    return {
        "shape": list(result.shape),
        "mean": float(result.mean()) if result.numel() else 0.0,
        "max_abs": float(result.abs().max()) if result.numel() else 0.0,
    }


def _trace_tensor(value, *, limit=None):
    """Preserve a CPU sample for cross-device stage attribution."""
    import torch

    result = value.detach().float().cpu().contiguous()
    flat = result.flatten()
    return {
        "shape": list(result.shape),
        "mean": float(flat.mean()) if flat.numel() else 0.0,
        "max_abs": float(flat.abs().max()) if flat.numel() else 0.0,
        # Materialize values rather than retaining a tensor view.  Some device
        # bridges preserve the original activation storage through ``clone``;
        # that makes a small attribution sample serialize as gigabytes.
        "sample": torch.tensor(
            (flat if limit is None else flat[:limit]).tolist(), dtype=torch.float32
        ),
    }


def _resolve_expressions(value, globals_=None):
    """Resolve the pinned BEVFusion YAML interpolation syntax."""
    if globals_ is None:
        globals_ = copy.deepcopy(value)
    if isinstance(value, dict):
        for key in value:
            value[key] = _resolve_expressions(value[key], globals_)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _resolve_expressions(item, globals_)
    elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        value = _resolve_expressions(eval(value[2:-1], globals_), globals_)
    return value


def _model_forward(model, batch, device, fp16, return_loss, rescale):
    import torch
    context = torch.autocast(device_type=device.type, dtype=torch.float16) if fp16 else nullcontext()
    with context:
        return model(return_loss=return_loss, rescale=rescale, **batch)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-name", default="scene-0001")
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--batch-file", type=Path)
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument(
        "--trace-init-only",
        action="store_true",
        help="save shared-input and initial-parameter attribution data without executing a model step",
    )
    parser.add_argument(
        "--trace-train-forward-only",
        action="store_true",
        help="capture the seeded training forward before backward or an optimizer update",
    )
    args = parser.parse_args()
    if args.max_iters < 1:
        parser.error("--max-iters must be >= 1")
    if not args.device.startswith(("musa", "cuda")):
        parser.error("--device must be a MUSA or CUDA device, such as musa:0 or cuda:0")

    checkout, data_root = args.checkout.resolve(), args.data_root.resolve()
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={checkout}", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"expected {UPSTREAM_REVISION}, got {revision}")
    sys.path[:0] = [str(ROOT), str(checkout)]
    os.environ.setdefault("MUSPACE_STRICT_PATCH", "1")
    os.environ.setdefault("MUSPACE_SPCONV_FP32_TF32", "0")

    import torch
    from mmcv import Config
    from mmcv.parallel import collate
    from mmcv.runner import build_optimizer
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from torchpack.utils.config import configs


    is_musa = args.device.startswith("musa")
    if is_musa:
        import torch_musa  # noqa: F401

        from mu_space import clear_dispatch_report, get_last_dispatch_report
        from mu_space.patcher import bevfusion_iou3d_patcher, bevfusion_patcher

        if not torch.musa.is_available():
            raise RuntimeError("MUSA device is unavailable")
        # The model's first divergent FP32 operation is a cuDNN/muDNN-style
        # convolution, not a matmul.  Disable both MUSA TF32 paths so the
        # regression runs do not silently use reduced-precision convolution.
        torch.backends.musa.matmul.allow_tf32 = False
        torch.backends.mudnn.allow_tf32 = False
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable")
    else:
        # Keep the CUDA reference on the matching non-TF32 convolution path.
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = Path(args.config)
    config = config if config.is_absolute() else checkout / config
    if not config.is_file():
        raise FileNotFoundError(f"BEVFusion config does not exist: {config}")
    configs.load(str(config), recursive=True)
    cfg = Config(_resolve_expressions(configs), filename=str(config))
    fixture_infos = _subset_infos(data_root, args.scene_name, args.work_dir)
    _set_dataset_paths(cfg.data.train, data_root, fixture_infos)
    _unwrap_scene_only_wrappers(cfg.data.train)
    _remove_unused_radar_pipeline(cfg.data.train)
    _normalize_voxel_limits(cfg.model)
    for key in ("pretrained", "init_cfg"):
        if hasattr(cfg.model, "img_backbone"):
            setattr(cfg.model.img_backbone, key, None)
    camera = cfg.model.get("encoders", {}).get("camera", {})
    if "backbone" in camera:
        camera["backbone"].pop("init_cfg", None)
        camera["backbone"].pop("pretrained", None)

    # The CUDA reference uses the unmodified upstream checkout. Importing the
    # MUSA checkout creates the target modules, then strict replacements are
    # installed before model construction so imported bindings are rebound.
    dispatch = None
    if is_musa:
        bevfusion_patcher.apply(strict=True)
        bevfusion_iou3d_patcher.apply(strict=True, backend="musa")
    if args.batch_file and args.batch_file.exists():
        packed = torch.load(args.batch_file, map_location="cpu", weights_only=False)
    else:
        dataset = build_dataset(cfg.data.train)
        sample = next((dataset.prepare_train_data(i) for i in range(len(dataset))), None)
        if sample is None:
            raise RuntimeError(f"no usable training sample found in {args.scene_name}")
        packed = collate([sample], samples_per_gpu=1)
        if args.batch_file:
            torch.save(packed, args.batch_file)
    device = torch.device(args.device)
    batch = _unwrap(packed, device)
    # Dataset preprocessing may consume RNG; initialize model from the shared seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = build_model(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    ).to(device)
    trace = {}
    hooks = []
    if args.trace_file:
        image = batch.get("img")
        if isinstance(image, torch.Tensor):
            trace["__input_img__"] = _trace_tensor(image, limit=4096)
        for name, parameter in model.named_parameters():
            if name in {
                "encoders.camera.backbone.patch_embed.projection.weight",
                "encoders.camera.backbone.patch_embed.projection.bias",
                "encoders.camera.backbone.patch_embed.norm.weight",
                "encoders.camera.backbone.patch_embed.norm.bias",
            }:
                # These tensors are small enough to preserve in full.  This
                # distinguishes initialization/config drift from a backend
                # numerical difference at the first image operator.
                trace[f"__parameter__.{name}"] = _trace_tensor(parameter)
            elif name.startswith("encoders.lidar.backbone.conv_input.0."):
                # The first SparseEncoder SubMConv is the first large residual
                # in a training-forward trace; retain its small filter/BN
                # tensors to separate rulebook arithmetic from initialization.
                trace[f"__parameter__.{name}"] = _trace_tensor(parameter)
        # Keep the saved trace intentionally narrow.  Registering every Swin
        # submodule materializes hundreds of large device-to-CPU activations
        # and obscures the first divergent operation under memory pressure.
        targets = {
            "encoders.camera.backbone.patch_embed.adap_padding",
            "encoders.camera.backbone.patch_embed.projection",
            "encoders.camera.backbone.patch_embed.norm",
            "encoders.camera.backbone.stages.0.blocks.0.norm1",
            "encoders.camera.backbone.stages.0.blocks.0.attn.w_msa.qkv",
            "encoders.camera.backbone.stages.0.blocks.0.attn.w_msa.softmax",
            "encoders.camera.backbone.stages.0.blocks.0.attn",
            "encoders.camera.backbone.stages.0.blocks.0.ffn.layers.0.0",
            "encoders.camera.backbone.stages.0.blocks.0",
            "encoders.camera.neck.lateral_convs.0.conv",
            "encoders.camera.neck.lateral_convs.0.bn",
            "encoders.camera.neck.lateral_convs.0",
            "encoders.camera.vtransform.depthnet",
            "encoders.camera.vtransform",
            "encoders.lidar.backbone",
            "fuser",
            "heads.map",
        }
        for name, module in model.named_modules():
            if name in targets or name.startswith("encoders.lidar.backbone."):
                def capture(_module, _inputs, output, *, name=name):
                    tensor = output.features if hasattr(output, "features") else output
                    if isinstance(tensor, torch.Tensor):
                        trace[name] = _trace_tensor(tensor, limit=4096)
                        if hasattr(output, "indices") and isinstance(output.indices, torch.Tensor):
                            trace[name]["indices"] = _trace_tensor(output.indices, limit=None)
                        if name == "encoders.lidar.backbone.conv_input.0":
                            for key, data in getattr(output, "indice_dict", {}).items():
                                if not isinstance(data, tuple) or len(data) < 4:
                                    continue
                                outids, inids, pairs, pair_num = data[:4]
                                if all(isinstance(item, torch.Tensor) for item in (outids, inids, pairs, pair_num)):
                                    prefix = f"__rulebook__.{name}.{key}"
                                    trace[prefix] = {
                                        "outids": _trace_tensor(outids, limit=None),
                                        "inids": _trace_tensor(inids, limit=None),
                                        "pairs": _trace_tensor(pairs, limit=None),
                                        "pair_num": _trace_tensor(pair_num, limit=None),
                                    }
                hooks.append(module.register_forward_hook(capture))
            if name == "encoders.lidar.backbone.conv_input.0":
                def capture_sparse_input(_module, inputs, *, name=name):
                    if not inputs:
                        return
                    tensor = inputs[0].features if hasattr(inputs[0], "features") else inputs[0]
                    if isinstance(tensor, torch.Tensor):
                        trace[f"__input__.{name}"] = _trace_tensor(tensor, limit=4096)
                        if hasattr(inputs[0], "indices") and isinstance(inputs[0].indices, torch.Tensor):
                            trace[f"__input__.{name}"]["indices"] = _trace_tensor(
                                inputs[0].indices, limit=None
                            )
                hooks.append(module.register_forward_pre_hook(capture_sparse_input))
    if args.trace_init_only:
        if not args.trace_file:
            parser.error("--trace-init-only requires --trace-file")
        args.trace_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trace, args.trace_file)
        print(json.dumps({"device": str(device), "trace_init_only": True, "upstream_revision": revision}))
        return
    model.train()
    if args.trace_train_forward_only:
        if not args.trace_file:
            parser.error("--trace-train-forward-only requires --trace-file")
        # Separate random training-path effects (DropPath/Dropout) from
        # initialization and optimizer-update effects.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        # BEVFusion's map head emits a Python zero for absent loss branches
        # under ``no_grad``.  Preserve the upstream training-forward autograd
        # semantics here, while intentionally stopping before backward/step.
        losses = _model_forward(model, batch, device, args.dtype == "float16", True, False)
        loss, _ = model._parse_losses(losses)
        for hook in hooks:
            hook.remove()
        args.trace_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trace, args.trace_file)
        print(json.dumps({"device": str(device), "loss": float(loss.cpu()),
                          "trace_train_forward_only": True, "upstream_revision": revision}))
        return
    optimizer = build_optimizer(model, cfg.optimizer)
    first = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    before = first.detach().clone()
    if is_musa:
        clear_dispatch_report()
    for _ in range(args.max_iters):
        optimizer.zero_grad(set_to_none=True)
        losses = _model_forward(model, batch, device, args.dtype == "float16", True, False)
        loss, log_vars = model._parse_losses(losses)
        if not torch.isfinite(loss):
            raise RuntimeError("BEVFusion loss is non-finite")
        loss.backward()
        optimizer.step()
    finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if not finite or torch.equal(before, first.detach()):
        raise RuntimeError("BEVFusion optimizer/gradient regression")
    model.eval()
    with torch.no_grad():
        predictions = _model_forward(model, batch, device, args.dtype == "float16", False, True)
    for hook in hooks:
        hook.remove()
    if args.trace_file:
        args.trace_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trace, args.trace_file)
    if is_musa:
        dispatch = get_last_dispatch_report()
    if is_musa and (dispatch is None or dispatch.selected_backend != "musa"):
        raise RuntimeError("strict native BEVFusion run did not record a MUSA dispatch")
    payload = {
        "device": str(device),
        "seed": args.seed,
        "scene": args.scene_name,
        "upstream_revision": revision,
        "loss": float(loss.detach().cpu()),
        "log_vars": {key: float(value) for key, value in log_vars.items()},
        "finite_gradients": bool(finite),
        "optimizer_step": not torch.equal(before, first.detach()),
        "dispatch": dispatch.as_dict() if dispatch is not None else {"selected_backend": "cuda-reference"},
        "prediction": _tensor_summary(predictions),
    }
    if args.artifact:
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "payload": payload,
                "initial_parameter": before.cpu(),
                "parameter": first.detach().cpu(),
            },
            args.artifact,
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
