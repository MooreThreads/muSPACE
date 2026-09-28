#!/usr/bin/env python3
"""Apply, inspect, or restore the pinned BEVFusion source patch safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REFERENCE_COMMIT = "326653dc06e0938edf1aae7d01efcd158ba83de5"
MARKER = ".mu_space_bevfusion_sigmoid_focal_loss.json"


def run(checkout: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={checkout}", *args],
        cwd=checkout,
        check=check,
        text=True,
        capture_output=True,
    )


def patch_path() -> Path:
    return Path(__file__).resolve().parents[1] / "patches" / "use-muspace-sigmoid-focal-loss.patch"


def revision(checkout: Path) -> str:
    return run(checkout, "rev-parse", "HEAD").stdout.strip()


def can_apply(checkout: Path, *, reverse: bool = False) -> bool:
    args = ["apply", "--check", "--unidiff-zero", "--ignore-space-change"]
    if reverse:
        args.append("--reverse")
    args.append(str(patch_path()))
    return run(checkout, *args, check=False).returncode == 0


def state(checkout: Path) -> str:
    if can_apply(checkout):
        return "not-applied"
    if can_apply(checkout, reverse=True):
        return "applied"
    return "conflict"


def digest() -> str:
    return hashlib.sha256(patch_path().read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--action", choices=("inspect", "apply", "restore"), default="apply")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    if not (checkout / ".git").exists():
        raise SystemExit(f"not a git checkout: {checkout}")
    actual_revision = revision(checkout)
    if actual_revision != REFERENCE_COMMIT:
        raise SystemExit(f"expected {REFERENCE_COMMIT}, got {actual_revision}")
    before = state(checkout)
    if args.action == "apply":
        if before == "not-applied":
            run(checkout, "apply", "--unidiff-zero", "--ignore-space-change", str(patch_path()))
            (checkout / MARKER).write_text(
                json.dumps({"patch_sha256": digest(), "reference_commit": actual_revision}, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        elif before != "applied":
            raise SystemExit(f"cannot apply patch from state: {before}")
    elif args.action == "restore":
        if before == "applied":
            run(checkout, "apply", "--reverse", "--unidiff-zero", "--ignore-space-change", str(patch_path()))
        elif before != "not-applied":
            raise SystemExit(f"cannot restore patch from state: {before}")
        (checkout / MARKER).unlink(missing_ok=True)
    after = state(checkout)
    print(json.dumps({"action": args.action, "before": before, "after": after,
                      "reference_commit": actual_revision}, sort_keys=True))


if __name__ == "__main__":
    main()
