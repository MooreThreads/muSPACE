#!/usr/bin/env python3
"""Benchmark the BEVFusion SparseEncoder sparse-convolution primitive on MUSA.

The published protocol and the record format live in ``benchmarks/_harness.py`` and
``docs/benchmarking.md``.  The timed region is one forwarded ``indice_conv`` call on a
rulebook that was built before the timed region.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Keep the harness and the local case module importable when the runner is executed
# directly, as the operator README documents.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import add_common_args, make_record, measure, print_summary, resolve_output
from spconv_case import DEFAULT_CASE, case_ids, make_case

from mu_space.ops.spconv import indice_conv

OPERATOR = "spconv"
#: The published entry point is the operator call itself: it owns its output allocation
#: and its host-side dispatch, so the row is not a kernel-only measurement.
SCOPE = "operator"
ENTRY_POINT = "mu_space.ops.spconv.indice_conv"
PUBLISHED_DTYPES = ("float32", "float16")
_DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        cases=case_ids(),
        default_case=DEFAULT_CASE,
        dtypes=PUBLISHED_DTYPES,
    )
    parser.add_argument(
        "--peak-bandwidth-gbps",
        type=float,
        default=None,
        help="nominal device bandwidth used for the derived utilization figure only",
    )
    return parser.parse_args(argv)


def describe(case, features, pairs, pair_num, extra: dict) -> dict:
    """Return the case properties and the derived bandwidth models."""

    derived = {
        "entry_point": ENTRY_POINT,
        "active_features": int(features.shape[0]),
        "active_pairs": int(pair_num.sum().item()),
        "spatial_shape": list(case.spatial_shape),
        "active_voxels": case.active_voxels,
        "grid_cells": case.grid_cells,
        "channels_in": case.channels_in,
        "channels_out": case.channels_out,
        "kernel_size": list(case.kernel_size),
        "stride": list(case.stride),
        "subm": bool(case.subm),
        "rulebook_shape": list(pairs.shape),
    }
    derived.update(extra)
    return derived


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dtype = _DTYPE_MAP[args.dtype]
    case, tensors, _ = make_case(args.case, device=args.device, dtype=dtype, seed=args.seed)
    features, filters, outids, pairs, pair_num = tensors
    # Mirror the BEVFusion SparseConv / SubMConv dispatch: submanifold convolution
    # enables the centre-offset direct-GEMM shortcut used by the CUDA reference path.
    subm = bool(case.subm)
    num_activate_out = int(outids.shape[0])

    def run() -> None:
        indice_conv(features, filters, pairs, pair_num, num_activate_out, inverse=False, subm=subm)

    # One untimed call provides the output shape used by the traffic model below.
    output = indice_conv(
        features, filters, pairs, pair_num, num_activate_out, inverse=False, subm=subm
    )
    measurement = measure(
        run,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    active_pairs = int(pair_num.sum().item())
    bytes_per_value = 2 if dtype == torch.float16 else 4
    traffic = (
        (features.numel() + filters.numel() + output.numel() + output.numel()) * bytes_per_value
        + pairs.numel() * 4
        + pair_num.numel() * 4
    )
    # Per-pair working set: what one pair touches if nothing is reused.  It counts a full
    # filter read per pair, which the batched GEMM does not do, so it models the
    # algorithm's working set and is not a bandwidth figure.  Do not divide it by a device
    # peak; use profiler DRAM counters for measured throughput.
    pair_working_set = (
        active_pairs
        * (case.channels_in + case.channels_in * case.channels_out + 2 * case.channels_out)
        * bytes_per_value
    )
    effective_gbps = traffic / (measurement.p50_ms * 1e6)
    derived = describe(
        case,
        features,
        pairs,
        pair_num,
        {
            "traffic_bytes_lower_bound": traffic,
            "effective_gbps": effective_gbps,
            "pair_working_set_bytes_model": pair_working_set,
            "pair_working_set_model_gbps": pair_working_set / (measurement.p50_ms * 1e6),
            "peak_bandwidth_gbps": args.peak_bandwidth_gbps,
            "bandwidth_utilization": (
                effective_gbps / args.peak_bandwidth_gbps if args.peak_bandwidth_gbps else None
            ),
        },
    )
    record = make_record(
        operator=OPERATOR,
        case=args.case,
        dtype=args.dtype,
        scope=SCOPE,
        device=args.device,
        seed=args.seed,
        measurement=measurement,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        derived=derived,
    )
    print_summary(record)
    target = record.save(resolve_output(args, OPERATOR))
    print(f"record: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
