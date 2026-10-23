"""Deterministic SparseDrive-shaped deformable aggregation benchmark inputs.

Shapes match production dumps:
  mc_ms_feat         [8, 89760, 256]  float32
  spatial_shape      [6, 4, 2]        int32
  scale_start_index  [6, 4]           int32
  sampling_location  [8, 900, 13, 6, 2] float32
  weights            [8, 900, 13, 6, 4, 8] float32
  output / grad_out  [8, 900, 256]    float32

Sampling locations use the dump active-point fraction (~18.6%) so synthetic
latency stays close to real-dump profiling without shipping large .pt files.

Forward and backward read separate tensors built from the same seed, so each pass is
published under its own stable case id.
"""

from __future__ import annotations

import torch

FORMAT = "deformable_aggregation_synthetic_v1"
DUMP_FORMAT = "deformable_aggregation_dump_v1"

DEFAULT_SEED = 2333

FORWARD_CASE = "sparsedrive-forward"
BACKWARD_CASE = "sparsedrive-backward"
CASES = (FORWARD_CASE, BACKWARD_CASE)
DEFAULT_CASE = FORWARD_CASE

BATCH = 8
NUM_FEAT = 89760
CHANNELS = 256
NUM_CAMS = 6
NUM_SCALE = 4
NUM_ANCHORS = 900
NUM_PTS = 13
NUM_GROUPS = 8
# Fraction of (anchor, pts, cam) points with both u,v in (0, 1) on the dump.
ACTIVE_FRAC = 0.18635328114032745

SPATIAL_SHAPE = [[[64, 176], [32, 88], [16, 44], [8, 22]]] * NUM_CAMS
SCALE_START_INDEX = [
    [0, 11264, 14080, 14784],
    [14960, 26224, 29040, 29744],
    [29920, 41184, 44000, 44704],
    [44880, 56144, 58960, 59664],
    [59840, 71104, 73920, 74624],
    [74800, 86064, 88880, 89584],
]

TENSOR_NAMES = (
    "mc_ms_feat",
    "spatial_shape",
    "scale_start_index",
    "sampling_location",
    "weights",
)


def _sampling_location(generator: torch.Generator) -> torch.Tensor:
    point_shape = (BATCH, NUM_ANCHORS, NUM_PTS, NUM_CAMS)
    active = torch.rand(*point_shape, generator=generator) < ACTIVE_FRAC
    outside = torch.randn(*point_shape, 2, generator=generator) * 10.0
    inside = torch.rand(*point_shape, 2, generator=generator).mul_(0.98).add_(0.01)
    return torch.where(active.unsqueeze(-1), inside, outside)


def _inputs(
    generator: torch.Generator, *, feat_scale: float, weight_scale: float
) -> dict[str, torch.Tensor]:
    return {
        "mc_ms_feat": torch.randn((BATCH, NUM_FEAT, CHANNELS), generator=generator) * feat_scale,
        "spatial_shape": torch.tensor(SPATIAL_SHAPE, dtype=torch.int32),
        "scale_start_index": torch.tensor(SCALE_START_INDEX, dtype=torch.int32),
        "sampling_location": _sampling_location(generator),
        "weights": torch.rand(
            (BATCH, NUM_ANCHORS, NUM_PTS, NUM_CAMS, NUM_SCALE, NUM_GROUPS),
            generator=generator,
        )
        * weight_scale,
    }


def make_case(seed: int = DEFAULT_SEED) -> dict:
    """Create a deterministic case with dump-faithful shapes and active density."""
    forward_generator = torch.Generator(device="cpu").manual_seed(seed)
    backward_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    forward_inputs = _inputs(forward_generator, feat_scale=1.0, weight_scale=0.004)
    backward_inputs = _inputs(backward_generator, feat_scale=1.0, weight_scale=0.004)
    grad_output = torch.randn((BATCH, NUM_ANCHORS, CHANNELS), generator=backward_generator)
    return {
        "format": FORMAT,
        "description": (
            "Deterministic synthetic deformable aggregation inputs with SparseDrive "
            "production shapes and dump-matched active sampling fraction"
        ),
        "seed": seed,
        "scalars": {"active_frac": ACTIVE_FRAC},
        "forward_inputs": forward_inputs,
        "backward_inputs": {**backward_inputs, "grad_output": grad_output},
    }
