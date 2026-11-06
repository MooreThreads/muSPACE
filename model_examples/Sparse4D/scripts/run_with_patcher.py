#!/usr/bin/env python3
"""Run a Sparse4D checkout script with the muSPACE runtime patcher installed.

The upstream data preparation tools are plain scripts.  Some of them import
``projects.mmdet3d_plugin`` while the checkout's ``deformable_aggregation_ext``
extension is not built, so they need the same runtime patch as a training run.
This helper keeps the patch scoped to the child script and restores it afterwards.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: run_with_patcher.py CHECKOUT SCRIPT [ARGS...]")

    checkout = Path(sys.argv[1]).resolve()
    script = Path(sys.argv[2])
    if not script.is_absolute():
        script = checkout / script
    if not script.is_file():
        raise SystemExit(f"script does not exist: {script}")

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(checkout))

    from mu_space.patcher import sparse4d_patcher

    sparse4d_patcher.apply(strict=False)
    try:
        sys.argv = [str(script), *sys.argv[3:]]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sparse4d_patcher.restore()


if __name__ == "__main__":
    main()
