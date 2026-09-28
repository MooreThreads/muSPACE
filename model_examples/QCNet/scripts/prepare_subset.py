"""Create a small read-only AV2 raw subset and let QCNet preprocess it."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    source = (args.source_root / args.split / "raw").resolve()
    if not source.is_dir():
        raise SystemExit(f"missing AV2 raw directory: {source}")
    scenarios = sorted(path for path in source.iterdir() if path.is_dir())[: args.count]
    if len(scenarios) < args.count:
        raise SystemExit(f"requested {args.count} scenarios but found {len(scenarios)}")
    destination = args.output_root / args.split / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        link = destination / scenario.name
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == scenario:
                continue
            raise SystemExit(f"refusing to replace existing path: {link}")
        link.symlink_to(scenario, target_is_directory=True)
    print(f"linked {len(scenarios)} {args.split} scenarios under {destination}")


if __name__ == "__main__":
    main()
