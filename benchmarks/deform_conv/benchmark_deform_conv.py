#!/usr/bin/env python3
"""Benchmark standard deformable convolution (DCN) on MUSA.

Two entry points of the same operator family are measured on the BEVDepth image-backbone
shapes: the general ``mu_space.ops.DeformConv2d``, which takes the sampling offsets as an
input, and the packed ``mu_space.ops.DeformConv2dPack``, whose offset convolution runs
inside the measured boundary.  The published protocol and the record format live in
``benchmarks/_harness.py`` and ``docs/benchmarking.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

# Allow `python benchmarks/deform_conv/benchmark_*.py` without PYTHONPATH manipulation
# beyond the repo root for `mu_space`, while keeping the harness and the local case
# module importable.
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
from deform_conv_case import CASES, DEFAULT_CASE, DEFAULT_OP, OPS, case_id, make_case

OPERATOR = "deform_conv"
SCOPE = "operator"
IMPLEMENTATION = "mu_space_native"
REFERENCE_DEVICE = "NVIDIA A100 80GB PCIe"
REFERENCE_IMPLEMENTATION = "mmcv-compatible CUDA DCN reference"
CHECK_REFERENCE = "modulated_deform_conv2d with an all-ones mask"
TOLERANCE = {"float32": 1e-5, "float16": 1e-3}


def build_layer(payload: dict, op: str, dtype: torch.dtype, device: str):
    """Build the measured layer and load the case's weights into it.

    ``DeformConv2dPack`` zeroes its offset convolution, which is the mmcv initialisation
    but not what a trained backbone holds, so the case's seeded weights are copied over
    both convolutions before anything is timed.
    """
    from mu_space.ops import DeformConv2d, DeformConv2dPack

    scalars = payload["scalars"]
    factory = DeformConv2dPack if op == "pack" else DeformConv2d
    layer = factory(
        scalars["channels"],
        scalars["channels_out"],
        scalars["kernel_size"],
        stride=scalars["stride"],
        padding=scalars["padding"],
        dilation=scalars["dilation"],
        groups=scalars["groups"],
        deform_groups=scalars["deform_groups"],
    ).to(device=device, dtype=dtype)
    inputs = payload["inputs"]
    with torch.no_grad():
        layer.weight.copy_(inputs["weight"])
        if op == "pack":
            layer.conv_offset.weight.copy_(inputs["offset_weight"])
            layer.conv_offset.bias.copy_(inputs["offset_bias"])
    return layer


def to_device(payload: dict, op: str, dtype: torch.dtype, device: str) -> tuple:
    """Return the case's feature map and, for the general entry point, its offsets."""
    inputs = payload["inputs"]
    feature = inputs["input"].to(device=device, dtype=dtype)
    offset = inputs["offset"].to(device=device, dtype=dtype) if op == "general" else None
    return feature, offset


def make_runner(layer, op: str, feature: torch.Tensor, offset) -> Callable[[], torch.Tensor]:
    """Return the exactly-one-operator-call launch the harness times."""
    if op == "general":
        return lambda: layer(feature, offset)
    return lambda: layer(feature)


def check_layer(layer, op: str, payload: dict, feature: torch.Tensor, offset, dtype: str) -> dict:
    """Compare the measured entry point with the modulated kernel at mask = 1.

    Standard DCN is the modulated operator with an all-ones mask, which is the reference
    ``tests/ops/test_deform_conv.py`` validates against, so both entry points have to
    reproduce it from the same inputs.
    """
    from mu_space.ops import modulated_deform_conv2d

    scalars = payload["scalars"]
    taps = scalars["deform_groups"] * scalars["kernel_size"] ** 2
    mask = torch.ones(
        feature.shape[0],
        taps,
        feature.shape[2],
        feature.shape[3],
        dtype=feature.dtype,
        device=feature.device,
    )
    if op == "general":
        measured_offset = offset
    else:
        raw = F.conv2d(
            feature,
            layer.conv_offset.weight,
            layer.conv_offset.bias,
            scalars["stride"],
            scalars["padding"],
            scalars["dilation"],
        )
        measured_offset = raw[:, : scalars["offset_channels"]]
    expected = modulated_deform_conv2d(
        feature,
        measured_offset.contiguous(),
        mask,
        layer.weight,
        None,
        (scalars["stride"], scalars["stride"]),
        (scalars["padding"], scalars["padding"]),
        (scalars["dilation"], scalars["dilation"]),
        scalars["groups"],
        scalars["deform_groups"],
        backend="musa",
    )
    actual = layer(feature, offset) if op == "general" else layer(feature)
    tolerance = TOLERANCE[dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    difference = (actual.detach().float() - expected.detach().float()).abs()
    return {
        "checked": True,
        "reference": CHECK_REFERENCE,
        "tolerance": tolerance,
        "max_abs_error": float(difference.max()),
    }


def describe(payload: dict, op: str, layer, check: dict) -> dict:
    """Return the case properties recorded next to the latency."""
    scalars = payload["scalars"]
    return {
        "implementation": IMPLEMENTATION,
        "entry_point": op,
        "input_shape": list(payload["inputs"]["input"].shape),
        "weight_shape": list(layer.weight.shape),
        "kernel_size": scalars["kernel_size"],
        "stride": scalars["stride"],
        "padding": scalars["padding"],
        "dilation": scalars["dilation"],
        "groups": scalars["groups"],
        "deform_groups": scalars["deform_groups"],
        "check": check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cases=CASES, default_case=DEFAULT_CASE)
    parser.add_argument(
        "--op",
        choices=OPS,
        default=DEFAULT_OP,
        help=(
            "entry point: 'pack' measures mu_space.ops.DeformConv2dPack, with the offset "
            "convolution inside the timed region; 'general' measures "
            "mu_space.ops.DeformConv2d, which takes the case's offsets as an input"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the measured entry point with the modulated kernel at mask = 1 first",
    )
    args = parser.parse_args()

    device_module(args.device)
    dtype = getattr(torch, args.dtype)
    payload = make_case(args.case, args.seed, dtype)
    feature, offset = to_device(payload, args.op, dtype, args.device)
    layer = build_layer(payload, args.op, dtype, args.device)
    check = (
        check_layer(layer, args.op, payload, feature, offset, args.dtype)
        if args.check
        else {"checked": False}
    )
    measurement = measure(
        make_runner(layer, args.op, feature, offset),
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    record = make_record(
        operator=OPERATOR,
        case=case_id(args.op, args.case),
        dtype=args.dtype,
        scope=SCOPE,
        device=args.device,
        seed=args.seed,
        measurement=measurement,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        derived=describe(payload, args.op, layer, check),
        reference_device=REFERENCE_DEVICE,
        reference_implementation=REFERENCE_IMPLEMENTATION,
    )
    record.save(resolve_output(args, OPERATOR))
    print_summary(record)


if __name__ == "__main__":
    main()
