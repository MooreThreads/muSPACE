"""Export deterministic sparse-convolution inputs for MUSA/A100 runs."""

from __future__ import annotations

import argparse
import subprocess

import torch

from spconv_case import CASES, DEFAULT_SEED, make_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), default="bevfusion")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    case, tensors, out_shape = make_case(args.case, seed=args.seed)
    features, filters, outids, pairs, pair_num = tensors
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    torch.save(
        {
            "features": features,
            "filters": filters,
            "outids": outids,
            "indice_pairs": pairs,
            "indice_pair_num": pair_num,
            "metadata": {
                "case": case.name,
                "seed": args.seed,
                "spatial_shape": case.spatial_shape,
                "out_shape": out_shape,
                "dtype": str(features.dtype),
                "commit": commit,
            },
        },
        args.output,
    )


if __name__ == "__main__":
    main()
