#!/usr/bin/env python3
"""Benchmark the BEVFormer single-level MSDA case on the measured device.

The measured work is the pair of native kernels that
``mu_space.ops.multi_scale_deformable_attn`` dispatches to on MUSA tensors:
``ms_deform_attn_forward`` and ``ms_deform_attn_backward``.  The timed region covers one
forward launch on the case's forward inputs followed by one backward launch on its backward
inputs -- the same entry points, the same arguments and the same contiguity handling the
shipped autograd function uses -- so the published row describes the training path rather
than an inference-only call.

Measurement protocol, record format and table rules live in ``docs/benchmarking.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch

from benchmarks._harness import (
    add_common_args,
    device_module,
    make_record,
    measure,
    print_summary,
    resolve_output,
)
from msda_single_level_case import (
    CASES,
    DEFAULT_CASE,
    DTYPES,
    FORMAT as SYNTHETIC_FORMAT,
    make_case,
)
from mu_space.runtime.native import native_module_proxy

SCOPE = "forward-backward"
ARTIFACT_FORMATS = {"ms_deform_attn_single_level_v1", SYNTHETIC_FORMAT}
INPUT_NAMES = (
    "value",
    "value_spatial_shapes",
    "value_level_start_index",
    "sampling_locations",
    "attention_weights",
)


def load_case(path: Path) -> dict:
    """Load a captured input artifact.

    Captured artifacts are a local investigation path only: they are never committed and
    they never produce a published row, which always comes from the fixed-seed case.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") not in ARTIFACT_FORMATS:
        raise ValueError(f"unsupported MSDA benchmark artifact: {path.name}")
    for group, extra in (("forward_inputs", set()), ("backward_inputs", {"grad_output"})):
        missing = (set(INPUT_NAMES) | extra).difference(payload.get(group, {}))
        if missing:
            raise ValueError(f"{path.name} is missing {group}: {sorted(missing)}")
    return payload


def move_tensors(payload: dict, device: str) -> dict[str, dict[str, torch.Tensor]]:
    """Copy the case to the device without changing layout or dtype."""
    return {
        group: {name: tensor.to(device) for name, tensor in payload[group].items()}
        for group in ("forward_inputs", "backward_inputs")
    }


def accumulator_dtype(storage_dtype: torch.dtype) -> torch.dtype:
    """Return the gradient accumulator dtype the shipped backward uses.

    ``mu_space.ops.multi_scale_deformable_attn`` accumulates in fp32 for a bfloat16 storage
    dtype and in the storage dtype otherwise, so the timed region must allocate the same
    accumulator buffers.
    """
    return torch.float32 if storage_dtype == torch.bfloat16 else storage_dtype


def make_launcher(
    tensors: dict[str, dict[str, torch.Tensor]], im2col_step: int
) -> Callable[[], None]:
    """Return the timed closure: one forward launch, then one backward launch."""
    native = native_module_proxy()
    forward = tuple(tensors["forward_inputs"][name] for name in INPUT_NAMES)
    backward = tuple(tensors["backward_inputs"][name] for name in INPUT_NAMES)
    grad_output = tensors["backward_inputs"]["grad_output"]
    accumulators = accumulator_dtype(backward[0].dtype)

    def new_accumulator(tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(tensor.shape, dtype=accumulators, device=tensor.device)

    def launch() -> None:
        output = forward[0].new_empty(
            (
                forward[0].size(0),
                forward[3].size(1),
                forward[0].size(2) * forward[0].size(3),
            )
        )
        native.ms_deform_attn_forward(*forward, output, im2col_step)
        grad_value = new_accumulator(backward[0])
        grad_sampling = new_accumulator(backward[3])
        grad_weights = new_accumulator(backward[4])
        native.ms_deform_attn_backward(
            *backward,
            grad_output.contiguous(),
            grad_value,
            grad_sampling,
            grad_weights,
            im2col_step,
        )

    return launch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cases=tuple(CASES), default_case=DEFAULT_CASE, dtypes=tuple(DTYPES))
    parser.add_argument(
        "--input-file",
        type=Path,
        help="captured input artifact to replay locally; published rows use the fixed-seed case",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device_module(args.device)
    if args.input_file is None:
        payload = make_case(args.seed, args.case, DTYPES[args.dtype])
        case = args.case
    else:
        payload = load_case(args.input_file)
        case = str(payload.get("case") or args.input_file.stem)
    tensors = move_tensors(payload, args.device)
    im2col_step = int(payload["scalars"]["im2col_step"])
    measurement = measure(
        make_launcher(tensors, im2col_step),
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    record = make_record(
        operator="multi_scale_deformable_attn",
        case=case,
        dtype=args.dtype,
        scope=SCOPE,
        device=args.device,
        seed=int(payload.get("seed", args.seed)),
        measurement=measurement,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        derived={
            "entry_point": "ms_deform_attn_forward + ms_deform_attn_backward",
            "storage_dtype": str(tensors["forward_inputs"]["value"].dtype).split(".")[-1],
            "accumulator_dtype": str(
                accumulator_dtype(tensors["forward_inputs"]["value"].dtype)
            ).split(".")[-1],
            "im2col_step": im2col_step,
            "forward_value_shape": list(tensors["forward_inputs"]["value"].shape),
            "forward_sampling_shape": list(tensors["forward_inputs"]["sampling_locations"].shape),
            "backward_grad_output_stride": list(tensors["backward_inputs"]["grad_output"].stride()),
            "input_file": args.input_file.name if args.input_file else None,
        },
    )
    record.save(resolve_output(args, "msda"))
    print_summary(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
