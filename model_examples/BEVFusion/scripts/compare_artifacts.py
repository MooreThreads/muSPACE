#!/usr/bin/env python3
"""Compare MUSA and A100 BEVFusion regression artifacts without rounding metrics."""

from __future__ import annotations

import argparse
import json

import torch


def metric(left: torch.Tensor, right: torch.Tensor, tolerance: float) -> dict[str, object]:
    delta = (left.float() - right.float()).abs()
    relative = delta / right.float().abs().clamp_min(1e-12)
    return {"max_absolute_error": float(delta.max()), "max_relative_error": float(relative.max()),
            "mean_absolute_error": float(delta.mean()), "mismatch_count": int((delta > tolerance).sum()),
            "tolerance": tolerance, "pass": bool(torch.all(delta <= tolerance))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--musa", required=True)
    parser.add_argument("--a100", required=True)
    parser.add_argument("--dtype", choices=("float32", "float16"), required=True)
    args = parser.parse_args()
    tolerance = 1e-3 if args.dtype == "float32" else 2e-2
    musa, a100 = torch.load(args.musa, map_location="cpu", weights_only=False), torch.load(args.a100, map_location="cpu", weights_only=False)
    for key in ("scene", "seed", "upstream_revision"):
        if musa["payload"].get(key) != a100["payload"].get(key):
            raise SystemExit(f"artifact mismatch for {key}")
    result = {"dtype": args.dtype, "parameter": metric(musa["parameter"], a100["parameter"], tolerance),
              "loss_absolute_error": abs(musa["payload"]["loss"] - a100["payload"]["loss"]),
              "tolerance": tolerance}
    result["pass"] = result["parameter"]["pass"] and result["loss_absolute_error"] <= tolerance
    print(json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise SystemExit("BEVFusion MUSA/A100 comparison failed")


if __name__ == "__main__":
    main()
