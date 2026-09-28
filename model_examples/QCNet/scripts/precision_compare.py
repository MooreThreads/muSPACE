"""Run or compare a shared real-batch QCNet precision artifact."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--dtype", choices=("float32", "float16"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--atol", type=float)
    p.add_argument("--rtol", type=float)
    return p.parse_args()


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _run(args) -> dict[str, object]:
    checkout = args.checkout.resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "mu_space").is_dir():
            sys.path.insert(0, str(parent))
            break
    sys.path.insert(0, str(checkout))
    from predictors import QCNet

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "musa" and (not hasattr(torch, "musa") or not torch.musa.is_available()):
        raise RuntimeError("MUSA is unavailable")
    saved = _load(args.state)
    model = QCNet(**saved["model_kwargs"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model = model.to(device).eval()
    batch = _load(args.batch).to(device)
    context = (
        torch.autocast(device_type=device.type, dtype=torch.float16)
        if args.dtype == "float16"
        else nullcontext()
    )
    with torch.no_grad(), context:
        output = model(batch)
        loss = model.training_step(copy.deepcopy(batch), 0)
    tensors = {
        name: value.detach().float().cpu()
        for name, value in output.items()
        if torch.is_tensor(value)
    }
    tensors["loss"] = loss.detach().float().cpu().reshape(1)
    return {"device": str(device), "dtype": args.dtype, "tensors": tensors}


def _compare(actual: dict, reference: dict, atol: float, rtol: float) -> dict[str, object]:
    if set(actual["tensors"]) != set(reference["tensors"]):
        raise RuntimeError("actual and reference tensor keys differ")
    metrics, passed = {}, True
    for name in sorted(actual["tensors"]):
        value = actual["tensors"][name]
        expected = reference["tensors"][name]
        if value.shape != expected.shape:
            raise RuntimeError(f"shape mismatch for {name}: {value.shape} != {expected.shape}")
        diff = (value - expected).abs()
        rel = diff / expected.abs().clamp_min(1e-12)
        close = torch.isclose(value, expected, atol=atol, rtol=rtol)
        mismatch = int((~close).sum().item())
        flat_index = int(diff.argmax().item()) if diff.numel() else 0
        item = {
            "shape": list(value.shape),
            "max_abs_error": float(diff.max()) if diff.numel() else 0.0,
            "mean_abs_error": float(diff.mean()) if diff.numel() else 0.0,
            "max_rel_error": float(rel.max()) if rel.numel() else 0.0,
            "mismatch_count": mismatch,
            "worst_flat_index": flat_index,
        }
        metrics[name] = item
        passed = passed and mismatch == 0 and torch.isfinite(value).all().item()
    return {"atol": atol, "rtol": rtol, "passed": passed, "metrics": metrics}


def main() -> None:
    args = _args()
    artifact = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    atol = args.atol if args.atol is not None else (3e-4 if args.dtype == "float32" else 5e-3)
    rtol = args.rtol if args.rtol is not None else (1e-3 if args.dtype == "float32" else 5e-2)
    report = {
        "device": artifact["device"],
        "dtype": args.dtype,
        "artifact": str(args.output.resolve()),
    }
    if args.reference:
        report["comparison"] = _compare(artifact, _load(args.reference), atol, rtol)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.reference and not report["comparison"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
