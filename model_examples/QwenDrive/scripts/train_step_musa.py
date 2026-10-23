#!/usr/bin/env python3
"""Run one frozen-backbone Qwen-Drive perception-head step on MUSA.

``--mode fixture`` builds the published perception head from its config and takes
one AdamW step on synthetic features at the released 896x512 geometry. It does
not load a VLM.

``--mode real`` runs the frozen VLM once on a packed demo frame, detaches the
two feature taps the published head consumes, and takes the same step. The VLM
stays in ``eval`` with ``requires_grad_(False)``. ``--device-count`` repeats
that head step on consecutive MUSA devices in a Python loop. The loop is not
data parallel and is not a throughput measurement.

The loss is a stand-in over the tensors the published head returns. It is not
the upstream detection-matching objective. ``QwenDrivePerception.infer`` is not
used: that entry point is wrapped in ``torch.no_grad`` and decodes predictions
instead of returning trainable logits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Published nuScenes camera count. The fixture only needs a fixed N; real mode
# takes the count from the packed frame.
NUSCENES_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
# The released processor merges 2x2 patches, so the language-model tap is half
# the pre-merge grid. The frustum step equals that patch size.
MERGE_SIZE = 2
FEATURE_SCALE = 0.02


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixture", "real"), default="fixture")
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--device-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--vlm", type=Path, default=None)
    parser.add_argument("--perception", type=Path, default=None)
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject bad arguments before importing torch, the patcher, or the model."""

    if args.device_count < 1:
        parser.error("--device-count must be at least 1")
    if args.device.split(":", 1)[0] != "musa":
        parser.error("--device must be a MUSA device")
    checkout = Path(args.checkout)
    package = checkout / "src" / "qwen_drive_perception"
    if not package.is_dir():
        parser.error(f"not a Qwen-Drive checkout: {checkout}")
    if args.mode != "real":
        return
    if args.vlm is None or not Path(args.vlm).is_dir():
        parser.error("--vlm must be a directory for --mode real")
    if args.perception is None or not Path(args.perception).is_dir():
        parser.error("--perception must be a directory for --mode real")
    frames = Path(args.frames) if args.frames is not None else None
    if frames is None or not frames.is_dir():
        parser.error("--frames must be a directory for --mode real")
    # The published loader opens calib.npz and gt.npz for every packed frame.
    # Reject that gap here so the process exits before the model is imported.
    if not _packed_frames(frames):
        parser.error("--frames has no packed demo frame (missing frame.json)")
    if _first_complete_frame(frames) is None:
        parser.error("--frames has no packed demo frame with calib.npz and gt.npz")


def _device_list(first: str, count: int) -> list[str]:
    kind, sep, index = first.partition(":")
    start = int(index) if sep else 0
    return [f"{kind}:{start + offset}" for offset in range(count)]


def _select_device(device) -> None:
    """Select ``device`` before moving tensors onto it.

    The multi-card loop moves one head from card to card. The current device
    stays on the previous card unless it is selected explicitly, and a launch
    from that context makes the next card's step non-finite.
    """
    import torch

    torch.musa.set_device(device)


def _add_repo_paths(checkout: Path) -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "mu_space").is_dir():
            sys.path.insert(0, str(parent))
            break
    sys.path.insert(0, str(checkout / "src"))


def _pinhole(width: int, height: int, focal: float = 500.0):
    """Image-plane projection that leaves some frustum cells inside the BEV grid."""

    import numpy as np

    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0
    return matrix


def _synthetic_inputs(device, dtype, seed: int):
    """Features and calibration at the published frustum, not a shrunk stand-in."""

    import numpy as np
    import torch

    from qwen_drive_perception.configuration_perception import FRUSTUM_RANGE, FRUSTUM_SIZE

    grid_w = int(round((FRUSTUM_RANGE[3] - FRUSTUM_RANGE[0]) / FRUSTUM_SIZE[0]))
    grid_h = int(round((FRUSTUM_RANGE[4] - FRUSTUM_RANGE[1]) / FRUSTUM_SIZE[1]))
    cameras = len(NUSCENES_CAMERAS)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vit = torch.randn(cameras, grid_h, grid_w, 1024, generator=generator)
    llm = torch.randn(
        cameras,
        grid_h // MERGE_SIZE,
        grid_w // MERGE_SIZE,
        2560,
        generator=generator,
    )
    vit = vit.mul(FEATURE_SCALE).to(device=device, dtype=dtype)
    llm = llm.mul(FEATURE_SCALE).to(device=device, dtype=dtype)

    width = int(FRUSTUM_RANGE[3] - FRUSTUM_RANGE[0])
    height = int(FRUSTUM_RANGE[4] - FRUSTUM_RANGE[1])
    projection = _pinhole(width, height)
    identity = np.eye(4, dtype=np.float32)
    metas = [
        {
            "dataset_type": "nuscenes",
            "cam_order": list(NUSCENES_CAMERAS),
            "lidar2img": np.stack([projection for _ in range(cameras)]),
            "lidar2ego": np.stack([identity for _ in range(cameras)]),
            "img_shape": [(height, width)] * cameras,
        }
    ]
    return vit, llm, metas, None


def _load_gt(frame_dir: Path):
    path = frame_dir / "gt.npz"
    if not path.is_file():
        return None
    import numpy as np

    loaded = np.load(path)
    try:
        return {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()


def _packed_frames(frames: Path) -> list[Path]:
    return sorted(
        path for path in frames.iterdir() if path.is_dir() and (path / "frame.json").is_file()
    )


def _frame_is_complete(path: Path) -> bool:
    return (path / "calib.npz").is_file() and (path / "gt.npz").is_file()


def _first_complete_frame(frames: Path) -> Path | None:
    found = [path for path in _packed_frames(frames) if _frame_is_complete(path)]
    return found[0] if found else None


def _first_frame(frames: Path) -> Path:
    found = _first_complete_frame(frames)
    if found is None:
        raise RuntimeError(f"no packed demo frame with calib.npz and gt.npz under {frames}")
    return found


def _tap_vlm(vlm, inputs: dict, img_metas: dict, perception_cls):
    """Replay the published taps without entering ``infer``."""

    import torch

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, args, _output=None):
        captured["patches"] = args[0]

    visual = vlm.model.visual
    handle = visual.merger.register_forward_hook(_hook)
    try:
        outputs = vlm(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            mm_token_type_ids=(inputs["input_ids"] == vlm.config.image_token_id).long(),
            use_cache=False,
            output_hidden_states=True,
        )
    finally:
        handle.remove()
    if "patches" not in captured:
        raise RuntimeError("VLM forward did not reach the vision merger tap")

    hidden = vlm.model.language_model.norm(outputs.hidden_states[-1])
    with torch.no_grad():
        patches = visual.merger.norm(captured["patches"])
    vit_feats = perception_cls._premerge_grids(patches, inputs["image_grid_thw"])

    num_cams = len(img_metas["cam_order"])
    grid_h = int(inputs["image_grid_thw"][-1, 1])
    grid_w = int(inputs["image_grid_thw"][-1, 2])
    tokens_per_image = (grid_h // MERGE_SIZE) * (grid_w // MERGE_SIZE)
    needed = num_cams * tokens_per_image
    image_mask = inputs["input_ids"][0] == vlm.config.image_token_id
    llm_tokens = hidden[0][image_mask]
    if llm_tokens.shape[0] < needed:
        raise RuntimeError(
            f"VLM image tokens ({llm_tokens.shape[0]}) are fewer than "
            f"{num_cams} cameras x {tokens_per_image} tokens"
        )
    llm_tokens = llm_tokens[-needed:]
    img_llm = llm_tokens.view(num_cams, grid_h // MERGE_SIZE, grid_w // MERGE_SIZE, -1)
    img_vit = torch.stack(vit_feats[-num_cams:], dim=0)
    return img_vit.detach(), img_llm.detach()


def _real_inputs(args, device):
    import torch
    from transformers import AutoTokenizer

    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor

    frame_dir = _first_frame(Path(args.frames))
    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    vlm = holder.vlm
    if hasattr(holder, "planning_expert"):
        del holder.planning_expert
    del holder
    vlm.requires_grad_(False)
    vlm.eval()
    vlm.to(device)

    processor = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    frame = PerceptionFrame(frame_dir)
    inputs, img_metas = processor(frame, device=str(device))
    with torch.no_grad():
        vit, llm = _tap_vlm(vlm, inputs, img_metas, QwenDrivePerception)

    perception = QwenDrivePerception.from_pretrained(args.perception, dtype=torch.bfloat16)
    return vit, llm, [img_metas], _load_gt(frame_dir), perception.bev_modeling, vlm, frame_dir.name


def _label_tensor(logits, label, class_dim: int):
    import torch

    labels = torch.as_tensor(label, device=logits.device)
    if labels.dtype.is_floating_point:
        return None
    spatial = [size for axis, size in enumerate(logits.shape) if axis != class_dim]
    if list(labels.shape) == spatial[1:]:
        labels = labels.unsqueeze(0)
    if list(labels.shape) != spatial:
        return None
    return labels.to(dtype=torch.long)


def _cross_entropy(logits, label, class_dim: int):
    """Class cross-entropy when the label grid matches, otherwise no term."""

    import torch.nn.functional as F

    labels = _label_tensor(logits, label, class_dim)
    if labels is None:
        return None
    classes = logits.shape[class_dim]
    flat_logits = logits.float().movedim(class_dim, -1).reshape(-1, classes)
    flat_labels = labels.reshape(-1)
    valid = (flat_labels >= 0) & (flat_labels < classes)
    if int(valid.sum()) == 0:
        return None
    flat_labels = flat_labels.masked_fill(~valid, -100)
    return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)


def _stand_in_loss(outs: dict, gt: dict | None):
    """Logit penalty, replaced by cross-entropy only for a matching label grid."""

    terms = {
        "cls": outs["all_cls_scores"].float().square().mean(),
        "bbox": outs["all_bbox_preds"].float().square().mean(),
        "occ": outs["occ_pred"].float().square().mean(),
        "map": outs["seg_preds"].float().square().mean(),
    }
    kinds = {name: "logit_penalty" for name in terms}
    if not gt:
        return sum(terms.values()), terms, kinds

    occ_label = next(
        (gt[name] for name in ("occupancy", "occ", "voxel_semantics", "semantics") if name in gt),
        None,
    )
    map_label = next(
        (gt[name] for name in ("map", "maps", "semantic_map", "gt_map", "map_seg") if name in gt),
        None,
    )
    if occ_label is not None:
        supervised = _cross_entropy(outs["occ_pred"], occ_label, class_dim=-1)
        if supervised is not None:
            terms["occ"] = supervised
            kinds["occ"] = "cross_entropy"
    if map_label is not None:
        supervised = _cross_entropy(outs["seg_preds"], map_label, class_dim=1)
        if supervised is not None:
            terms["map"] = supervised
            kinds["map"] = "cross_entropy"
    return sum(terms.values()), terms, kinds


def _grad_report(module):
    """L2 norm of produced head gradients. Unused parameters may have no gradient."""

    import torch

    total = torch.zeros((), dtype=torch.float32)
    produced = 0
    nonfinite = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        produced += 1
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            nonfinite.append(name)
            continue
        total = total + gradient.float().pow(2).sum().cpu()
    if nonfinite:
        detail = ", ".join(nonfinite[:10])
        raise RuntimeError(f"head gradients are non-finite: {len(nonfinite)}; first: {detail}")
    if produced == 0:
        raise RuntimeError("head produced no gradients")
    norm = total.sqrt()
    if not torch.isfinite(norm) or float(norm) == 0.0:
        raise RuntimeError(f"head gradient norm is {float(norm)}")
    return float(norm)


def _backbone_is_clean(vlm) -> bool:
    if vlm is None:
        return True
    leaked = [name for name, parameter in vlm.named_parameters() if parameter.grad is not None]
    if leaked:
        raise RuntimeError("frozen backbone received gradients: " + ", ".join(leaked[:10]))
    return True


def _step_on_device(head, vit, llm, metas, gt, device, lr: float) -> dict:
    import torch

    _select_device(device)
    head.to(device=device, dtype=torch.bfloat16)
    head.train()
    dtype = next(head.parameters()).dtype
    vit = vit.to(device=device, dtype=dtype)
    llm = llm.to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    torch.musa.reset_peak_memory_stats(device)
    torch.musa.synchronize(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    outputs = head(vit, llm, metas)
    loss, terms, kinds = _stand_in_loss(outputs, gt)
    if loss.device.type != device.type or not torch.isfinite(loss).item():
        raise RuntimeError(f"invalid loss on {loss.device}: {loss}")
    loss.backward()
    grad_norm = _grad_report(head)
    optimizer.step()
    torch.musa.synchronize(device)
    return {
        "device": str(device),
        "loss": float(loss.detach().float().cpu()),
        "loss_terms": {name: float(value.detach().float().cpu()) for name, value in terms.items()},
        "loss_kinds": kinds,
        "head_grad_norm": grad_norm,
        "head_finite_gradients": True,
        "optimizer_step": True,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_memory_bytes": int(torch.musa.max_memory_allocated(device)),
    }


def _run(args: argparse.Namespace) -> dict:
    import torch

    try:
        import torch_musa  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch_musa is required for --device musa") from exc
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")

    devices = _device_list(args.device, args.device_count)
    available = torch.musa.device_count()
    last_index = int(devices[-1].split(":", 1)[1])
    if available <= last_index:
        raise RuntimeError(
            f"need MUSA device index {last_index} for --device-count {args.device_count}, "
            f"found {available}"
        )

    _add_repo_paths(Path(args.checkout).resolve())
    from mu_space.patcher import qwendrive_patcher

    torch.manual_seed(args.seed)
    torch.musa.manual_seed_all(args.seed)
    report = qwendrive_patcher.apply(strict=args.strict)
    try:
        first = torch.device(devices[0])
        _select_device(first)
        frame_name = None
        vlm = None
        if args.mode == "fixture":
            from qwen_drive_perception.configuration_perception import QwenDrivePerceptionConfig
            from qwen_drive_perception.modeling_perception import BEVFormerModelV2

            head = BEVFormerModelV2(QwenDrivePerceptionConfig())
            vit, llm, metas, gt = _synthetic_inputs(first, torch.bfloat16, args.seed)
        else:
            vit, llm, metas, gt, head, vlm, frame_name = _real_inputs(args, first)

        steps = [
            _step_on_device(head, vit, llm, metas, gt, torch.device(name), args.lr)
            for name in devices
        ]
        backbone_clean = _backbone_is_clean(vlm)
    finally:
        qwendrive_patcher.restore()

    return {
        "status": "qwendrive_perception_train_step_ok",
        "mode": args.mode,
        "strict": bool(args.strict),
        "seed": args.seed,
        "lr": args.lr,
        "patcher_action": report.action,
        "frame": frame_name,
        "backbone_present": vlm is not None,
        "backbone_grad_is_none": backbone_clean,
        "devices": devices,
        "steps": steps,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate(parser, args)
    print(json.dumps(_run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
