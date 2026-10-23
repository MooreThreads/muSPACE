"""Check a deformable aggregation launch whose flat index space exceeds 32 bits.

The forward kernel tiles batch * anchors * points * cameras * scales * channels, so about a
hundred sampling points already exceeds 2**32 threads. This script runs that launch once as
a single call and once as the sum over point chunks whose flat indices stay below 2**32. A
32-bit launch index wraps past that boundary and attributes threads to the wrong sample, so
the two disagree.

The boundary is 2**32 rather than 2**31 because the kernel advances the flat index in
unsigned arithmetic: it stays exact while the count fits in 32 bits, and the launcher's
32-bit kernel argument only reintroduces the value modulo 2**32.

Usage:
    python benchmarks/deformable_aggregation/large_batch_check.py [--points 100] [--chunk 10]
"""

from __future__ import annotations

import argparse
import sys

import torch

from mu_space.ops.deformable_aggregation import deformable_aggregation

BATCH, NUM_FEAT, NUM_EMBEDS = 8, 89760, 256
NUM_CAMS, NUM_SCALE, NUM_ANCHORS, NUM_GROUPS = 6, 4, 900, 8
ACTIVE_FRAC = 0.4

SPATIAL_SHAPE = [[[64, 176], [32, 88], [16, 44], [8, 22]]] * NUM_CAMS
SCALE_START_INDEX = [
    [0, 11264, 14080, 14784],
    [14960, 26224, 29040, 29744],
    [29920, 41184, 44000, 44704],
    [44880, 56144, 58960, 59664],
    [59840, 71104, 73920, 74624],
    [74800, 86064, 88880, 89584],
]

THREADS_PER_POINT = BATCH * NUM_EMBEDS * NUM_ANCHORS * NUM_CAMS * NUM_SCALE
UINT32_LIMIT = 2**32


def build_inputs(num_pts: int, device: str, seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    spatial_shape = torch.tensor(SPATIAL_SHAPE, dtype=torch.int32, device=device)
    scale_start_index = torch.tensor(SCALE_START_INDEX, dtype=torch.int32, device=device)
    mc_ms_feat = torch.randn(BATCH, NUM_FEAT, NUM_EMBEDS, generator=generator, dtype=torch.float32)
    mc_ms_feat = mc_ms_feat.to(device)

    point_shape = (BATCH, NUM_ANCHORS, num_pts, NUM_CAMS)
    active = torch.rand(*point_shape, generator=generator) < ACTIVE_FRAC
    outside = torch.randn(*point_shape, 2, generator=generator) * 10.0
    inside = torch.rand(*point_shape, 2, generator=generator) * 0.98 + 0.01
    sampling_location = torch.where(active.unsqueeze(-1), inside, outside).to(device)

    weights = torch.rand(
        BATCH, NUM_ANCHORS, num_pts, NUM_CAMS, NUM_SCALE, NUM_GROUPS, generator=generator
    )
    return (
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        (weights * 0.004).to(device=device, dtype=torch.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=100, help="points in the large launch")
    parser.add_argument("--chunk", type=int, default=10, help="points per chunked call")
    parser.add_argument("--device", default="musa")
    args = parser.parse_args()

    threads = THREADS_PER_POINT * args.points
    chunk_threads = THREADS_PER_POINT * args.chunk
    if threads <= UINT32_LIMIT:
        parser.error(f"--points {args.points} gives {threads} threads, not above 2**32")
    if chunk_threads > UINT32_LIMIT:
        parser.error(f"--chunk {args.chunk} gives {chunk_threads} threads, which wraps 32 bits")

    inputs = build_inputs(args.points, args.device)
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights = inputs

    full = deformable_aggregation(*inputs)
    torch.musa.synchronize()

    chunked = torch.zeros_like(full)
    for begin in range(0, args.points, args.chunk):
        end = min(begin + args.chunk, args.points)
        chunked += deformable_aggregation(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location[:, :, begin:end],
            weights[:, :, begin:end],
        )
    torch.musa.synchronize()

    scale = chunked.abs().max().item()
    difference = (full - chunked).abs().max().item()
    print(f"threads: {threads:,} in one launch, {chunk_threads:,} per chunk")
    print(f"max |chunked| = {scale:.6g}, max |full - chunked| = {difference:.6g}")
    if scale == 0.0:
        print("chunked result is all zeros; no active samples", file=sys.stderr)
        return 2
    if difference > 1e-4 * scale:
        print("mismatch: the large launch disagrees with the chunked computation", file=sys.stderr)
        return 1
    print("match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
