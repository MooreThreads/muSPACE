#!/usr/bin/env python3
"""Compare two SparseDrive step artifacts produced by train_step_musa.py.

Run the launch check twice with ``--result-artifact`` and pass the two files here to
check that the loss, the logged losses, and every parameter gradient agree within a
tolerance. No fixture is bundled: both artifacts come from the run being compared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load(path: Path) -> dict:
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or "result" not in artifact or "gradients" not in artifact:
        raise SystemExit(f"{path} is not a train_step_musa.py result artifact")
    return artifact


def _compare_scalar(name: str, value: float, expected: float, atol: float, rtol: float) -> dict:
    difference = abs(value - expected)
    tolerance = atol + rtol * abs(expected)
    return {
        "reference": expected,
        "candidate": value,
        "abs_error": difference,
        "within_tolerance": difference <= tolerance,
    }


def _compare_tensor(name: str, value, expected, atol: float, rtol: float) -> dict:
    import torch

    if value.shape != expected.shape:
        raise SystemExit(
            f"shape mismatch for {name}: {tuple(value.shape)} != {tuple(expected.shape)}"
        )
    difference = (value - expected).abs()
    relative = difference / expected.abs().clamp_min(1e-12)
    close = torch.isclose(value, expected, atol=atol, rtol=rtol)
    return {
        "shape": list(value.shape),
        "max_abs_error": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs_error": float(difference.mean()) if difference.numel() else 0.0,
        "max_rel_error": float(relative.max()) if relative.numel() else 0.0,
        "mismatch_count": int((~close).sum().item()),
        "finite": bool(torch.isfinite(value).all()),
    }


def main() -> None:
    args = _args()
    # Exit 1 means the two artifacts disagree, so a bad invocation must not use it.
    for label, path in (("reference", args.reference), ("candidate", args.candidate)):
        if not path.is_file():
            print(f"{label} artifact does not exist: {path}", file=sys.stderr)
            raise SystemExit(2)

    reference = _load(args.reference)
    candidate = _load(args.candidate)

    reference_result = reference["result"]
    candidate_result = candidate["result"]
    reference_gradients = reference["gradients"]
    candidate_gradients = candidate["gradients"]
    if set(reference_gradients) != set(candidate_gradients):
        missing = sorted(set(reference_gradients) - set(candidate_gradients))
        extra = sorted(set(candidate_gradients) - set(reference_gradients))
        raise SystemExit(f"gradient key mismatch; missing={missing[:5]} extra={extra[:5]}")

    scalars = {
        "loss": _compare_scalar(
            "loss", candidate_result["loss"], reference_result["loss"], args.atol, args.rtol
        )
    }
    reference_logs = reference_result.get("log_vars", {})
    candidate_logs = candidate_result.get("log_vars", {})
    if set(reference_logs) != set(candidate_logs):
        raise SystemExit("log_vars key mismatch")
    for name in sorted(reference_logs):
        scalars[name] = _compare_scalar(
            name, candidate_logs[name], reference_logs[name], args.atol, args.rtol
        )

    gradients = {
        name: _compare_tensor(
            name, candidate_gradients[name], reference_gradients[name], args.atol, args.rtol
        )
        for name in sorted(reference_gradients)
    }
    passed = all(item["within_tolerance"] for item in scalars.values()) and all(
        item["mismatch_count"] == 0 and item["finite"] for item in gradients.values()
    )

    report = {
        "reference": {
            "path": str(args.reference.resolve()),
            "device": reference_result.get("device"),
            "scene": reference_result.get("scene"),
            "upstream_revision": reference_result.get("upstream_revision"),
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "device": candidate_result.get("device"),
            "scene": candidate_result.get("scene"),
            "upstream_revision": candidate_result.get("upstream_revision"),
        },
        "atol": args.atol,
        "rtol": args.rtol,
        "passed": passed,
        "scalars": scalars,
        "gradients": gradients,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
