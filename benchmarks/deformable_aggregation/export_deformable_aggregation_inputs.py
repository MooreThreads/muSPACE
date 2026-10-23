#!/usr/bin/env python3
"""Create a compact deformable aggregation benchmark artifact from SparseDrive dumps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deformable_aggregation_case import DUMP_FORMAT, TENSOR_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("data") / "deformable_aggregation.pt",
    )
    args = parser.parse_args()
    forward = torch.load(
        args.dump_dir / "deformable_aggregation_forward.pt",
        map_location="cpu",
        weights_only=False,
    )
    backward = torch.load(
        args.dump_dir / "deformable_aggregation_backward.pt",
        map_location="cpu",
        weights_only=False,
    )
    for name in TENSOR_NAMES:
        if name not in forward or name not in backward:
            raise ValueError(f"Dump is missing tensor {name}")
    if "grad_output" not in backward:
        raise ValueError("Backward dump is missing grad_output")

    artifact = {
        "format": DUMP_FORMAT,
        "description": "Tensors from SparseDrive deformable aggregation forward/backward dumps",
        "scalars": {},
        "forward_inputs": {name: forward[name].contiguous() for name in TENSOR_NAMES},
        "backward_inputs": {
            **{name: backward[name].contiguous() for name in TENSOR_NAMES},
            "grad_output": backward["grad_output"].contiguous(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
