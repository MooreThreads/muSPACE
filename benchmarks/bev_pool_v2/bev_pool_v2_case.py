"""Deterministic r50-ish BEVPoolV2 benchmark case."""

from __future__ import annotations

import torch


FORMAT = "bev_pool_v2_r50ish_v1"

DEFAULT_SEED = 2333

CASE_ID = "r50-200k"
CASES = (CASE_ID,)
DEFAULT_CASE = CASE_ID

_DEPTH_SHAPE = (1, 6, 88, 16, 44)
_FEAT_SHAPE = (1, 6, 16, 44, 64)
_BEV_FEAT_SHAPE = (1, 1, 128, 128, 64)
_POINTS = 200_000


def _intervals(sorted_ranks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kept = torch.ones(sorted_ranks.numel(), dtype=torch.bool)
    kept[1:] = sorted_ranks[1:] != sorted_ranks[:-1]
    starts = torch.where(kept)[0].to(torch.int32)
    lengths = torch.empty_like(starts)
    lengths[:-1] = starts[1:] - starts[:-1]
    lengths[-1] = sorted_ranks.numel() - starts[-1]
    return starts.contiguous(), lengths.contiguous()


def make_case(seed: int = DEFAULT_SEED) -> dict:
    """Build a deterministic case matching the published r50-ish comparison."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    depth = torch.rand(_DEPTH_SHAPE, generator=generator)
    feat = torch.rand(_FEAT_SHAPE, generator=generator)
    # A depth logit is sampled at most once by the projection. Keeping this
    # invariant avoids a write race in grad_depth on every native backend.
    ranks_depth = torch.randperm(depth.numel(), generator=generator, dtype=torch.int32)[:_POINTS]
    ranks_feat = torch.randint(
        _FEAT_SHAPE[0] * _FEAT_SHAPE[1] * _FEAT_SHAPE[2] * _FEAT_SHAPE[3],
        (_POINTS,),
        generator=generator,
        dtype=torch.int32,
    )
    ranks_bev = torch.randint(
        _BEV_FEAT_SHAPE[0] * _BEV_FEAT_SHAPE[1] * _BEV_FEAT_SHAPE[2] * _BEV_FEAT_SHAPE[3],
        (_POINTS,),
        generator=generator,
        dtype=torch.int32,
    )
    order = ranks_bev.argsort(stable=True)
    ranks_depth, ranks_feat, ranks_bev = (
        tensor[order].contiguous() for tensor in (ranks_depth, ranks_feat, ranks_bev)
    )
    interval_starts, interval_lengths = _intervals(ranks_bev)
    grad_out = torch.rand(
        (_BEV_FEAT_SHAPE[0], _BEV_FEAT_SHAPE[4], *_BEV_FEAT_SHAPE[1:4]), generator=generator
    )
    inputs = {
        "depth": depth,
        "feat": feat,
        "ranks_depth": ranks_depth,
        "ranks_feat": ranks_feat,
        "ranks_bev": ranks_bev,
        "interval_starts": interval_starts,
        "interval_lengths": interval_lengths,
    }
    return {
        "format": FORMAT,
        "seed": seed,
        "scalars": {"bev_feat_shape": _BEV_FEAT_SHAPE, "points": _POINTS},
        "forward_inputs": inputs,
        "backward_inputs": {**inputs, "grad_output": grad_out},
    }
