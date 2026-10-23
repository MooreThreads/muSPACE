"""Deterministic, fixed-seed cases for the standard deformable convolution benchmark.

The cases mirror the 3x3 deformable convolutions of the BEVDepth image backbone: a ResNet-50
whose image stages convolve with a standard ``DCN`` layer instead of a plain 3x3, fed by the
256x704 six-camera image batch the training recipe gives it.  At that input the backbone's
layer3 features are 16x44 with 512 channels and its layer4 features 8x22 with 1024 channels,
so a stride-1 layer of either stage deforms exactly the shapes published here, at the channel
counts the layer convolves over.

Both entry points of the operator family are built from the same tensors: ``general``
consumes the offsets the offset convolution produced, ``pack`` recomputes them inside the
timed region.  Weights use the fan-in scale a trained backbone produces, which is what puts
the sampling offsets in the range that decides how irregular the gather's access is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

FORMAT = "deform_conv_bevdepth_r50_v1"

DEFAULT_SEED = 2333

KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
DILATION = 1
GROUPS = 1
DEFORM_GROUPS = 1
OFFSET_CHANNELS = 2 * DEFORM_GROUPS * KERNEL_SIZE * KERNEL_SIZE

#: Entry points of ``mu_space.ops``: ``general`` takes the offsets as an input, ``pack``
#: computes them with the layer's own offset convolution.
OPS = ("pack", "general")
DEFAULT_OP = "pack"

#: Case name -> (batch, in_channels, out_channels, height, width).  ``b6`` is the
#: six-camera image batch the training recipe feeds the backbone, ``b12`` its twofold
#: accumulation of the same batch.
LAYOUTS = {
    "stage3-b6": (6, 512, 512, 16, 44),
    "stage4-b6": (6, 1024, 1024, 8, 22),
    "stage3-b12": (12, 512, 512, 16, 44),
    "stage4-b12": (12, 1024, 1024, 8, 22),
}
CASES = tuple(LAYOUTS)
DEFAULT_CASE = "stage3-b6"


def case_id(op: str, case: str) -> str:
    """Return the published case id: the entry point and the layout it is measured on."""
    if op not in OPS:
        raise ValueError(f"unknown entry point {op!r}; choose from {sorted(OPS)}")
    if case not in LAYOUTS:
        raise ValueError(f"unknown case {case!r}; choose from {sorted(LAYOUTS)}")
    return f"{op}-{case}"


def make_case(
    case: str = DEFAULT_CASE, seed: int = DEFAULT_SEED, dtype: torch.dtype = torch.float32
) -> dict:
    """Build one deterministic case; every tensor follows from ``case`` and ``seed`` alone.

    The tensors are drawn in float32 and cast once at the end, so a case measured in another
    dtype holds the same values and can be replayed on another platform unchanged.
    """
    if case not in LAYOUTS:
        raise ValueError(f"unknown case {case!r}; choose from {sorted(LAYOUTS)}")
    batch, channels, channels_out, height, width = LAYOUTS[case]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fan_in = channels * KERNEL_SIZE * KERNEL_SIZE

    def rand(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * scale

    input_fp32 = rand(batch, channels, height, width)
    weight_fp32 = rand(
        channels_out, channels // GROUPS, KERNEL_SIZE, KERNEL_SIZE, scale=fan_in**-0.5
    )
    offset_weight_fp32 = rand(
        OFFSET_CHANNELS, channels, KERNEL_SIZE, KERNEL_SIZE, scale=fan_in**-0.5
    )
    offset_bias_fp32 = rand(OFFSET_CHANNELS, scale=0.1)
    # The general entry point takes the offsets as an input, so they are materialised here
    # from the same offset convolution the packed entry point runs internally.
    offset_fp32 = F.conv2d(
        input_fp32, offset_weight_fp32, offset_bias_fp32, STRIDE, PADDING, DILATION
    )
    inputs = {
        "input": input_fp32,
        "weight": weight_fp32,
        "offset_weight": offset_weight_fp32,
        "offset_bias": offset_bias_fp32,
        "offset": offset_fp32,
    }
    return {
        "format": FORMAT,
        "case": case,
        "seed": seed,
        "scalars": {
            "batch": batch,
            "channels": channels,
            "channels_out": channels_out,
            "height": height,
            "width": width,
            "kernel_size": KERNEL_SIZE,
            "stride": STRIDE,
            "padding": PADDING,
            "dilation": DILATION,
            "groups": GROUPS,
            "deform_groups": DEFORM_GROUPS,
            "offset_channels": OFFSET_CHANNELS,
        },
        "inputs": {name: tensor.to(dtype).contiguous() for name, tensor in inputs.items()},
    }
