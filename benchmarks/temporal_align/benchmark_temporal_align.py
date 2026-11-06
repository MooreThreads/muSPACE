#!/usr/bin/env python3
"""Benchmark temporal_align on MUSA and the CUDA reference device."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import (
    add_common_args,
    device_module,
    make_record,
    measure,
    print_summary,
    resolve_output,
)
from temporal_align_case import BACKWARD, DEFAULT_CASE, case_ids, make_named_case

OPERATOR = "temporal_align"
SCOPE = "operator"
#: The CUDA row reproduces the production path this operator replaces: one
#: ``torchvision.transforms.functional.rotate`` call per sample. The MUSA row
#: forces the native backend.
UPSTREAM_HINT = "torchvision rotate loop"
NATIVE_HINT = "muSPACE native"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        cases=case_ids(),
        default_case=DEFAULT_CASE,
        dtypes=("float32",),
        devices=("musa", "cuda"),
    )
    return parser.parse_args(argv)


def _upstream_forward(case: dict, tensors: dict[str, torch.Tensor]) -> Callable[[], torch.Tensor]:
    """Reproduce the upstream per-sample rotation loop on the reference device."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import rotate

    bev = tensors["bev"]
    # Upstream keeps the rotation angle as a host scalar per sample, so the
    # device-to-host transfer belongs outside the timed region.
    angles = [float(value) for value in case["angle"].tolist()]
    center = list(case["center"])
    interpolation = (
        InterpolationMode.BILINEAR if case["mode"] == "bilinear" else InterpolationMode.NEAREST
    )

    def forward() -> torch.Tensor:
        return torch.stack(
            [
                rotate(bev[sample], angles[sample], interpolation=interpolation, center=center)
                for sample in range(bev.shape[0])
            ]
        )

    return forward


def build_launch(case: dict, tensors: dict[str, torch.Tensor]) -> Callable[[], None]:
    bev = tensors["bev"]
    angle = tensors["angle"]
    if bev.device.type == "musa":
        from mu_space.ops import temporal_align

        def forward() -> torch.Tensor:
            return temporal_align(
                bev,
                angle,
                None,
                bev_h=case["bev_h"],
                bev_w=case["bev_w"],
                center=case["center"],
                mode=case["mode"],
                backend="musa",
            )

    else:
        forward = _upstream_forward(case, tensors)

    if case["direction"] != BACKWARD:
        return forward

    def backward() -> None:
        bev.grad = None
        forward().square().sum().backward()

    return backward


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device_module(args.device)
    case = make_named_case(args.case, seed=args.seed)
    dtype = getattr(torch, args.dtype)
    tensors = {
        name: tensor.to(device=args.device, dtype=dtype) for name, tensor in case["inputs"].items()
    }
    # Only the BEV feature map is differentiable; angle and shift are inputs.
    tensors["bev"].requires_grad_(True)
    measurement = measure(
        build_launch(case, tensors),
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    native = args.device == "musa"
    record = make_record(
        operator=OPERATOR,
        case=args.case,
        dtype=args.dtype,
        scope=SCOPE,
        device=args.device,
        seed=args.seed,
        measurement=measurement,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        derived={
            "direction": case["direction"],
            "bev_shape": list(case["bev_shape"]),
            "bev_h": case["bev_h"],
            "bev_w": case["bev_w"],
            "channels": case["channels"],
            "batch": case["batch"],
            "mode": case["mode"],
            "center": list(case["center"]),
            "shift": "none",
            "implementation": NATIVE_HINT if native else UPSTREAM_HINT,
        },
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
