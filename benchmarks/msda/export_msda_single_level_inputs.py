#!/usr/bin/env python3
"""Create the compact, fixed MSDA benchmark input artifact from raw dumps."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


OPERATOR_INPUTS = (
    "value",
    "value_spatial_shapes",
    "value_level_start_index",
    "sampling_locations",
    "attention_weights",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "dumped_data"
        / "ms_deform_attn_single_level.pt",
    )
    args = parser.parse_args()
    forward = torch.load(
        args.dump_dir / "ms_deform_attn_forward0.pt", map_location="cpu", weights_only=False
    )
    backward = torch.load(
        args.dump_dir / "ms_deform_attn_backward7.pt", map_location="cpu", weights_only=False
    )
    if forward["scalars"] != backward["scalars"]:
        raise ValueError("Forward and backward im2col settings differ")
    artifact = {
        "format": "ms_deform_attn_single_level_v1",
        "description": "Tensors from real single-level MSDA forward/backward dumps",
        "scalars": forward["scalars"],
        "forward_inputs": {name: forward["inputs"][name]["tensor"] for name in OPERATOR_INPUTS},
        "backward_inputs": {
            **{name: backward["inputs"][name]["tensor"] for name in OPERATOR_INPUTS},
            "grad_output": backward["inputs"]["grad_output"]["tensor"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
