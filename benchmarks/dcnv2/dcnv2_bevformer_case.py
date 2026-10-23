"""Deterministic, shape-faithful DCNv2 benchmark inputs.

The cases mirror the deformable convolutions of the BEVFormer image backbone: the two
feature resolutions the ResNet-DCN stages run at, for the single-frame and multi-frame
batch sizes. Weight scales follow the fan-in initialisation the model uses, which keeps
the generated sampling offsets in the range a trained model produces -- that range is
what determines how irregular the gather's memory access becomes.
"""

from __future__ import annotations

import torch

FORMAT = "modulated_deform_conv_bevformer_synthetic_v1"

# name -> (batch, in_channels, out_channels, height, width)
CASES = {
    "high-b6": (6, 256, 256, 58, 100),
    "high-b18": (18, 256, 256, 58, 100),
    "low-b6": (6, 512, 512, 29, 50),
    "low-b18": (18, 512, 512, 29, 50),
    "tiny": (2, 32, 32, 7, 9),
}

#: Stable case ids published for the BEVFormer image backbone.  ``tiny`` stays available
#: for smoke runs but is not a published row.
BEVFORMER_CASES = ("high-b6", "high-b18", "low-b6", "low-b18")

# 3x3, stride 1, padding 1, dilation 1, groups 1, deform_groups 1 -- the configuration
# BEVFormer's `dcn=dict(type='DCNv2', deform_groups=1)` produces in the backbone.
KERNEL_SIZE = 3
OFFSET_CHANNELS = 3 * KERNEL_SIZE * KERNEL_SIZE


def make_case(name: str, seed: int, dtype: torch.dtype = torch.float32) -> dict:
    """Create a deterministic case with the workload's tensor layouts."""
    if name not in CASES:
        raise ValueError(f"unknown case {name!r}; choose from {sorted(CASES)}")
    batch, channels, channels_out, height, width = CASES[name]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fan_in = channels * KERNEL_SIZE * KERNEL_SIZE

    def rand(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * scale).to(dtype)

    return {
        "format": FORMAT,
        "description": "Deterministic synthetic BEVFormer ResNet-DCN benchmark inputs",
        "case": name,
        "seed": seed,
        "scalars": {
            "batch": batch,
            "channels": channels,
            "channels_out": channels_out,
            "height": height,
            "width": width,
            "kernel_size": KERNEL_SIZE,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "deform_groups": 1,
        },
        "inputs": {
            "input": rand(batch, channels, height, width),
            "offset_weight": rand(
                OFFSET_CHANNELS, channels, KERNEL_SIZE, KERNEL_SIZE, scale=fan_in**-0.5
            ),
            "offset_bias": rand(OFFSET_CHANNELS, scale=0.1),
            "weight": rand(channels_out, channels, KERNEL_SIZE, KERNEL_SIZE, scale=fan_in**-0.5),
            "bias": rand(channels_out, scale=0.1),
        },
    }
