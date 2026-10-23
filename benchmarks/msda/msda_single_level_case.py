"""Deterministic, shape-faithful MSDA single-level benchmark inputs.

The published case mirrors the single-level multi-scale deformable attention of a BEVFormer
encoder: 40,000 queries, 8 heads, 32 embed dims, one 200x200 feature level and four sampling
points.  Every tensor is drawn from a fixed seed, so a published row is reproducible from
committed code alone and no captured artifact has to be redistributed.

The case is drawn in float32 and optionally narrowed to a storage dtype.  bfloat16 changes
storage only: the kernel keeps the sampling geometry, the attention weights and every
accumulator in fp32, so a bfloat16 row measures the storage dtype, not reduced-precision math.
"""

from __future__ import annotations

import torch


FORMAT = "ms_deform_attn_single_level_synthetic_v1"
TENSOR_NAMES = (
    "value",
    "value_spatial_shapes",
    "value_level_start_index",
    "sampling_locations",
    "attention_weights",
)

#: Stable case ids.  Each id names one workload and never changes once published.
CASES = {
    "bevformer-1l": (
        "BEVFormer single-level MSDA: batch 2, 40000 queries, 8 heads, 32 embed dims, "
        "one 200x200 level, 4 sampling points"
    ),
}
DEFAULT_CASE = "bevformer-1l"

#: Storage dtypes a published row may use.  float32 is the reference dtype; bfloat16 is the
#: training dtype of the perception heads this operator serves.
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def _inputs(
    generator: torch.Generator, value_scale: float, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    batch, queries, heads, channels, points = 2, 40000, 8, 32, 4
    return {
        "value": (
            torch.randn((batch, 40000, heads, channels), generator=generator) * value_scale
        ).to(dtype),
        "value_spatial_shapes": torch.tensor([[200, 200]], dtype=torch.int64),
        "value_level_start_index": torch.tensor([0], dtype=torch.int64),
        # The workload has a near-uniform coordinate distribution with a small
        # boundary margin. Keep that distribution without storing dumps.
        "sampling_locations": torch.rand((batch, queries, heads, 1, points, 2), generator=generator)
        .mul_(1.035)
        .add_(-0.0175)
        .to(dtype),
        "attention_weights": torch.full((batch, queries, heads, 1, points), 0.25, dtype=dtype),
    }


def make_case(seed: int, case: str = DEFAULT_CASE, dtype: torch.dtype = torch.float32) -> dict:
    """Create a deterministic case with the workload's tensor layouts."""
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; choose from {sorted(CASES)}")
    forward_generator = torch.Generator(device="cpu").manual_seed(seed)
    backward_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    forward_inputs = _inputs(forward_generator, value_scale=1.38, dtype=dtype)
    backward_inputs = _inputs(backward_generator, value_scale=1.25, dtype=dtype)
    # Preserve the workload's backward layout: stride (1, 512, 2), so the
    # benchmark keeps grad_output.contiguous() in the timed path.
    grad_output = torch.randn((40000, 256, 2), generator=backward_generator).permute(2, 0, 1)
    grad_output.mul_(6.3e-6)
    grad_output = grad_output.to(dtype)
    return {
        "format": FORMAT,
        "case": case,
        "description": CASES[case],
        "seed": seed,
        "scalars": {"im2col_step": 64},
        "forward_inputs": forward_inputs,
        "backward_inputs": {**backward_inputs, "grad_output": grad_output},
    }
