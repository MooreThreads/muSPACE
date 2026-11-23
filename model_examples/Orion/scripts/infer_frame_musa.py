#!/usr/bin/env python3
"""Run a single-frame Orion forward pass on MUSA with the muSPACE operator patch.

Input modes:

``--input synthetic`` (default)
    One synthetic frame shaped like Orion's Bench2Drive test pipeline output.

``--input bench2drive --clip-dir DIR --frame-id N``
    One real frame read from ``DIR/camera/rgb_*``; the temporal history is zero-filled.

The fixture succeeds when every prediction is finite and at least one replaced
operator reported the MUSA backend. ``--benchmark N`` additionally reports the
steady-state latency of N warm repetitions.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PAST_FRAMES = 2
NUM_CAMERAS = 6
IMAGE_SIZE = 640
SOURCE_HEIGHT, SOURCE_WIDTH = 900, 1600
# Orion gates the class-agnostic NMS pass of get_motion_bboxes behind this variable.
# Without it the decoded boxes skip NMS and no replaced operator runs.
NMS_DEBUG_ENV = "DEBUG_SHOW_DIR_PRED"
CAMERA_FOLDERS = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back",
    "rgb_back_left",
    "rgb_back_right",
)
IMAGE_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
IMAGE_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


def _prepare_import_path(checkout: Path) -> None:
    for entry in (REPO_ROOT, checkout):
        resolved = str(entry)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _apply_patcher(backend: str):
    from mu_space.patcher.orion_patcher import OrionPatcher

    return OrionPatcher().apply(backend=backend)


def _stage3_transform(llm_path: Path):
    """The upstream transform that tokenizes an Orion generation prompt."""
    from mmcv.datasets.pipelines.transforms_3d import LoadAnnoatationCriticalVQATest

    return LoadAnnoatationCriticalVQATest(
        tokenizer=str(llm_path),
        max_length=2048,
        load_type=["critical_qa"],
        use_gen_token=True,
    )


def _conversation_ids(transform, device: torch.device) -> torch.Tensor:
    """Tokenize the prompt. The stage-3 conversation ends in ``<waypoint_ego>``, which
    is what marks the sample as a final generation round."""
    pieces = transform({"use_gen_token": True})["input_ids"]
    ids = torch.cat([p if torch.is_tensor(p) else torch.tensor(p) for p in pieces])
    return ids.reshape(1, -1).to(device)


def _assemble_frame(img: torch.Tensor, device: torch.device, transform):
    from mmcv.core.bbox.structures.utils import get_box_type

    box_type_3d, _ = get_box_type("LiDAR")
    conversation = _conversation_ids(transform, device)
    intrinsic = torch.eye(3, dtype=torch.float32)
    intrinsic[0, 0] = intrinsic[1, 1] = 400.0
    intrinsic[0, 2], intrinsic[1, 2] = IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0
    eye = torch.eye(4, dtype=torch.float32)

    def to_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=device, dtype=torch.float32)
        return value

    img_metas = {
        "scene_token": "orion-fixture-scene",
        "frame_idx": 0,
        "box_type_3d": box_type_3d,
        "vlm_labels": torch.full((1, 32), -100, dtype=torch.long),
        "lidar2img": eye.unsqueeze(0).repeat(NUM_CAMERAS, 1, 1).numpy(),
        "cam_intrinsic": intrinsic.unsqueeze(0).repeat(NUM_CAMERAS, 1, 1).numpy(),
        "lidar2cam": eye.unsqueeze(0).repeat(NUM_CAMERAS, 1, 1).numpy(),
        "img_shape": [(IMAGE_SIZE, IMAGE_SIZE, 3)] * NUM_CAMERAS,
        "ori_shape": [(SOURCE_HEIGHT, SOURCE_WIDTH, 3)] * NUM_CAMERAS,
        "pad_shape": [(IMAGE_SIZE, IMAGE_SIZE, 3)] * NUM_CAMERAS,
        "scale_factor": 1.0,
    }
    return {
        # 6-D ``[B, T, N, C, H, W]``; a 5-D stack collapses B to the camera count.
        "img": [to_device(img[-1].unsqueeze(0).unsqueeze(0))],
        # ``forward_test`` strips one level, the generation step strips another.
        "input_ids": [[conversation]],
        "vlm_labels": [[torch.full((1, 32), -100, dtype=torch.long, device=device)]],
        # Every remaining key is reduced with ``value[0][0].unsqueeze(0)``, so it has to
        # arrive nested two deep to come out as a ``[1, ...]`` batch.
        "ego_his_trajs": [[to_device(torch.zeros(PAST_FRAMES + 1, 3))]],
        "ego_lcf_feat": [[to_device(torch.zeros(9))]],
        "can_bus": [[to_device(torch.zeros(18))]],
        "timestamp": [to_device(torch.zeros(1))],
        "command": [to_device(torch.zeros(1))],
        "cam_intrinsic": [[to_device(intrinsic.unsqueeze(0).repeat(NUM_CAMERAS, 1, 1))]],
        "lidar2img": [[to_device(eye.unsqueeze(0).repeat(NUM_CAMERAS, 1, 1))]],
        "ego_pose": [[to_device(eye.unsqueeze(0))]],
        "ego_pose_inv": [[to_device(eye.unsqueeze(0))]],
        # A fixture has no ground truth, so the planning targets are zero-filled.
        "ego_fut_cmd": [[to_device(torch.zeros(1, 1, 6))]],
        "ego_fut_trajs": [[to_device(torch.zeros(1, 1, 6, 2))]],
        "ego_fut_masks": [[to_device(torch.zeros(1, 1, 6))]],
        "fut_valid_flag": [[to_device(torch.zeros(1, dtype=torch.bool))]],
        "img_metas": [[img_metas]],
    }


def _synthetic_frame(device: torch.device, transform):
    generator = torch.Generator(device="cpu").manual_seed(0)
    img = torch.randn(
        PAST_FRAMES + 1, NUM_CAMERAS, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator
    )
    return _assemble_frame((img - 0.5) * 2.0, device, transform)


def _load_rgb(path: Path) -> torch.Tensor:
    """Read one camera image and apply the pipeline's crop, resize and normalize."""
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(path)
    # Bench2Drive frames carry a hood band at the bottom; the pipeline crops it.
    image = raw.astype(np.float32)[: SOURCE_HEIGHT - 320]
    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1)
    return (tensor.flip(0) - IMAGE_MEAN) / IMAGE_STD


def _read_clip_frame(clip_dir: Path, frame_id: str, device: torch.device, transform):
    """Assemble one frame from ``<clip>/camera/rgb_*/<frame_id>.*``."""
    camera_root = clip_dir / "camera"
    camera_images = []
    missing = []
    for folder in CAMERA_FOLDERS:
        candidates = sorted((camera_root / folder).glob(f"{frame_id}.*"))
        if not candidates:
            missing.append(folder)
            continue
        camera_images.append(_load_rgb(candidates[0]))
    if missing:
        raise FileNotFoundError(f"{camera_root} is missing {frame_id} in: {missing}")

    current = torch.stack(camera_images)  # [N, C, H, W]
    history = torch.zeros(
        PAST_FRAMES, NUM_CAMERAS, 3, current.shape[-2], current.shape[-1]
    )
    frame = _assemble_frame(
        torch.cat((history, current.unsqueeze(0)), dim=0), device, transform
    )

    can_bus = torch.zeros(18)
    anno = clip_dir / "anno" / f"{frame_id}.json.gz"
    if anno.is_file():
        meta = json.loads(gzip.decompress(anno.read_bytes()))
        values = [float(meta.get(key, 0.0)) for key in ("x", "y", "theta", "speed", "steer")]
        can_bus[: len(values)] = torch.tensor(values)
    frame["can_bus"] = [[can_bus.to(device)]]
    return frame


def _build_model(
    config_path: Path,
    checkpoint: Path | None,
    llm_path: Path,
    device,
    lm_fp16: bool = False,
):
    from mmcv.models.builder import build_detector
    from mmcv.utils import Config

    config = Config.fromfile(str(config_path))
    model_config = config.model.copy()
    model_config["tokenizer"] = str(llm_path)
    model_config["lm_head"] = str(llm_path)
    # Upstream inference defaults. ``use_gen_token`` adds ``<waypoint_ego>`` and grows
    # the vocabulary to 32001 as the published checkpoint stores it; turning it off
    # leaves the LM head one row short and the load fails on a size mismatch.
    model_config["use_lora"] = True
    model_config["use_gen_token"] = True
    # A fixture has no ``gt_attr_labels``, so skip the metric block that reads them.
    model_config["fp32_infer"] = True

    model = build_detector(model_config, train_cfg=None, test_cfg=None)
    if checkpoint is not None:
        state = torch.load(str(checkpoint), map_location="cpu")
        state = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"orion_checkpoint missing={len(missing)} unexpected={len(unexpected)}")
    if lm_fp16:
        # Cast after the load, otherwise the float32 weights overwrite the cast.
        model.lm_head = model.lm_head.half()
        model.lm_head.config.torch_dtype = torch.float16
    return model.to(device).eval()


def _collect_tensors(value, output):
    """Collect every tensor reachable from the prediction structure."""
    if isinstance(value, torch.Tensor):
        output.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_tensors(item, output)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_tensors(item, output)
    return output


def _run_benchmark(model, frame, args) -> int:
    """Time repeated forwards, resetting the detector's memory state each time."""
    accelerator = getattr(torch, args.device, None)

    def one_forward():
        model.test_flag = False
        model.pts_bbox_head.reset_memory()
        model.map_head.reset_memory()
        with torch.no_grad():
            output = model(frame, return_loss=False)
        accelerator.synchronize()
        return output

    start = time.perf_counter()
    one_forward()
    warmup = time.perf_counter() - start

    samples, peak = [], 0.0
    for _ in range(args.benchmark):
        accelerator.reset_peak_memory_stats()
        start = time.perf_counter()
        one_forward()
        samples.append(time.perf_counter() - start)
        peak = max(peak, accelerator.max_memory_allocated() / 1024**3)

    summary = {
        "input": args.input,
        "benchmark": True,
        "warmup_s": round(warmup, 3),
        "runs_s": [round(sample, 3) for sample in samples],
        "steady_mean_s": round(statistics.mean(samples), 3),
        "steady_median_s": round(statistics.median(samples), 3),
        "steady_min_s": round(min(samples), 3),
        "peak_gb": round(peak, 2),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", default="Orion", help="Orion source checkout")
    parser.add_argument("--checkpoint", default=None, help="Orion.pth path")
    parser.add_argument("--llm-path", required=True, help="pretrain_qformer directory")
    parser.add_argument("--input", default="synthetic", choices=("synthetic", "bench2drive"))
    parser.add_argument("--clip-dir", default=None, help="one Bench2Drive clip directory")
    parser.add_argument("--frame-id", default="00000", help="frame index inside the clip")
    parser.add_argument("--backend", default="musa", choices=("auto", "musa", "reference"))
    parser.add_argument("--device", default="musa")
    parser.add_argument("--output", default=None, help="write the summary JSON here")
    parser.add_argument(
        "--lm-fp16",
        action="store_true",
        help="run only the LM head in float16; the detector graph stays float32",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        metavar="N",
        help="time N warm repetitions after one warmup and report steady-state latency",
    )
    args = parser.parse_args()

    checkout = Path(args.checkout).resolve()
    if not (checkout / "mmcv").is_dir():
        raise SystemExit(f"Orion checkout not found at {checkout}")
    if args.input == "bench2drive" and not args.clip_dir:
        raise SystemExit("--input bench2drive requires --clip-dir")

    os.environ.setdefault(NMS_DEBUG_ENV, "1")
    # The patcher imports the vendored mmcv modules, so the checkout has to be
    # importable before it runs.
    _prepare_import_path(checkout)
    report = _apply_patcher(args.backend)
    print(
        f"orion_patcher action={report.action} backend={report.backend} "
        f"patched={len(report.patched_modules)} flash_attn_shim={report.flash_attn_shim}"
    )

    from mu_space import clear_dispatch_report, get_dispatch_reports

    torch.manual_seed(0)
    device = torch.device(args.device)
    llm_path = Path(args.llm_path)
    model = _build_model(
        checkout / "adzoo" / "orion" / "configs" / "orion_stage3_infer.py",
        Path(args.checkpoint) if args.checkpoint else None,
        llm_path,
        device,
        args.lm_fp16,
    )

    transform = _stage3_transform(llm_path)
    if args.input == "bench2drive":
        frame = _read_clip_frame(Path(args.clip_dir), args.frame_id, device, transform)
    else:
        frame = _synthetic_frame(device, transform)

    if args.benchmark:
        return _run_benchmark(model, frame, args)

    clear_dispatch_report()
    with torch.no_grad():
        result = model(frame, return_loss=False)

    tensors = _collect_tensors(result, [])
    if not tensors:
        raise RuntimeError("orion fixture produced no tensor outputs")

    finite = all(bool(torch.isfinite(t.to(torch.float32)).all().item()) for t in tensors)
    reports = get_dispatch_reports()
    musa = [r for r in reports if r.selected_backend == "musa"]
    failures = [r for r in reports if r.native_fallback or r.selected_backend == "error"]
    summary = {
        "input": args.input,
        "outputs": len(tensors),
        "finite": finite,
        "dispatch_total": len(reports),
        "dispatch_musa": len(musa),
        "dispatch_operators": sorted({r.operator for r in musa}),
        "dispatch_failures": len(failures),
        "patched_modules": len(report.patched_modules),
        "flash_attn_shim": report.flash_attn_shim,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["finite"] and summary["dispatch_musa"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
