"""Small, dependency-free helpers used by the PEP 517 backend."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

# Native code is a TVM FFI/DLPack module.  Do not add PyTorch extension
# sources here: keeping the source list independent of torch headers is the
# ABI boundary that lets one wheel work with several PyTorch releases.
EXTENSION_SOURCES = (
    "csrc/ffi/muspace_ops.mu",
    "csrc/pooling/bev_pool_musa.mu",
    "csrc/pooling/bev_pool_v2_musa.mu",
    "csrc/pooling/voxel_pool_depth_musa.mu",
    "csrc/flashbev/flash_bevpool_sca_musa.mu",
    "csrc/flashbev/flash_bevpool_musa.mu",
    "csrc/deformable_aggregation/deformable_aggregation_musa.mu",
    "csrc/ms_deformable_attn/ms_deform_attn_musa.mu",
    "csrc/deformable_conv/modulated_deform_conv_musa.mu",
    "csrc/sigmoid_focal_loss/sigmoid_focal_loss_musa.mu",
    "csrc/spconv/spconv_musa.mu",
    "csrc/geometry/boxes_iou_bev_musa.mu",
    "csrc/centerpoint/centerpoint_target_assign_musa.mu",
    "csrc/voxelize/hard_voxelize_musa.mu",
    "csrc/pooling/voxel_pooling_train_musa.mu",
    "csrc/geometry/points_in_boxes_all_musa.mu",
    "csrc/geometry/radius_musa.mu",
)


def get_git_version(cwd: Optional[Path] = None) -> str:
    """Return the full git commit hash, or ``unknown`` outside a git checkout."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def build_version_string(
    base_version: str,
    cwd: Optional[Path] = None,
    dev_suffix: str = "",
    local_version: Optional[str] = None,
) -> tuple[str, str]:
    """Build the package version and return it with the full git revision."""
    version = base_version
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"
    if local_version:
        version = f"{version}+{local_version}"
    return version, get_git_version(cwd=cwd)


def get_extension_sources(root: Path) -> list[str]:
    del root
    return list(EXTENSION_SOURCES)
