#!/usr/bin/env python3
"""Prepare a real, scene-filtered nuScenes subset for pinned BEVDepth.

The input metadata and sensor directories are never modified.  The generated data
root contains only links plus scene-filtered train/validation info files.  BEVDepth
creates the camera depth supervision from the linked LIDAR_TOP sample at data-loader
time, matching the unmodified upstream training path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path


UPSTREAM_REVISION = "d78c7b58b10b9ada940462ba83ab24d99cae5833"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-name", default=None, help="legacy alias for one train scene")
    parser.add_argument("--train-scenes", default="scene-0001:scene-0010")
    parser.add_argument("--val-scenes", default="scene-0011:scene-0012")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _link(destination: Path, source: Path, overwrite: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise FileExistsError(f"refusing to replace {destination}; pass --overwrite")
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.symlink_to(source, target_is_directory=True)


def _revision(checkout: Path) -> None:
    actual = subprocess.check_output(
        ["git", "-c", f"safe.directory={checkout}", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != UPSTREAM_REVISION:
        raise RuntimeError(f"expected {UPSTREAM_REVISION}, got {actual}")


def _scene_names(spec: str) -> list[str]:
    names = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            names.append(item)
            continue
        first, last = item.split(":", 1)
        prefix_a, number_a = first.rsplit("-", 1)
        prefix_b, number_b = last.rsplit("-", 1)
        if prefix_a != prefix_b:
            raise ValueError(f"scene range must share a prefix: {item}")
        names.extend(f"{prefix_a}-{index:04d}" for index in range(int(number_a), int(number_b) + 1))
    if not names:
        raise ValueError("at least one scene must be selected")
    return names


def _filter_infos(data_root: Path, scene_names: list[str], filename: str) -> int:
    from nuscenes import NuScenes

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(data_root), verbose=False)
    scene_map = {item["name"]: item for item in nusc.scene}
    missing = [name for name in scene_names if name not in scene_map]
    if missing:
        raise ValueError(f"scenes are absent from supplied metadata: {missing}")
    sample_tokens: set[str] = set()
    scene_tokens = {scene_map[name]["token"] for name in scene_names}
    for name in scene_names:
        token = scene_map[name]["first_sample_token"]
        while token:
            sample_tokens.add(token)
            token = nusc.get("sample", token)["next"]

    info_path = data_root / filename
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    with info_path.open("rb") as stream:
        payload = pickle.load(stream)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    filtered = [
        info
        for info in infos
        if info.get("sample_token") in sample_tokens or info.get("scene_token") in scene_tokens
    ]
    if not filtered:
        raise RuntimeError(f"generated infos contain no samples for {scene_names}")
    missing_paths = []
    for info in filtered:
        records = list(info.get("cam_infos", {}).values()) + [info.get("lidar_infos") or {}]
        for record in records:
            path = record.get("data_path") or record.get("filename")
            if path and not (data_root / path).is_file():
                missing_paths.append(path)
    if missing_paths:
        sample = missing_paths[:3]
        raise FileNotFoundError(
            f"{len(missing_paths)} sensor files are missing for {scene_names}; examples: {sample}"
        )
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["infos"] = filtered
    else:
        payload = filtered
    with info_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(info_path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {"file": filename, "scenes": scene_names, "samples": len(filtered), "sha256": digest}
        )
    )
    return len(filtered)


def _generate_infos(checkout: Path, data_root: Path, scene_names: list[str], filename: str) -> None:
    """Generate infos directly for requested names, independent of split lists."""

    sys.path.insert(0, str(checkout))
    from nuscenes.nuscenes import NuScenes
    from scripts.gen_info import generate_info

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(data_root), verbose=False)
    infos = generate_info(nusc, scene_names)
    with (data_root / filename).open("wb") as stream:
        pickle.dump(infos, stream, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    checkout = args.checkout.resolve()
    meta_root, scene_root, output_root = (
        args.meta_root.resolve(),
        args.scene_root.resolve(),
        args.output_root.resolve(),
    )
    _revision(checkout)
    output_root.mkdir(parents=True, exist_ok=True)
    _link(output_root / "maps", meta_root / "maps", args.overwrite)
    _link(output_root / "samples", scene_root / "samples", args.overwrite)
    _link(output_root / "v1.0-trainval", meta_root / "v1.0-trainval", args.overwrite)

    legacy_root = checkout / "data" / "nuScenes"
    legacy_root.parent.mkdir(parents=True, exist_ok=True)
    created_legacy_link = False
    if legacy_root.exists() or legacy_root.is_symlink():
        if legacy_root.resolve() != output_root:
            raise RuntimeError(
                f"BEVDepth checkout already has a different data root: {legacy_root}"
            )
    else:
        legacy_root.symlink_to(output_root, target_is_directory=True)
        created_legacy_link = True
    train_names = (
        _scene_names(args.scene_name) if args.scene_name else _scene_names(args.train_scenes)
    )
    val_names = _scene_names(args.val_scenes)
    try:
        _generate_infos(checkout, output_root, train_names, "nuscenes_infos_train.pkl")
        _generate_infos(checkout, output_root, val_names, "nuscenes_infos_val.pkl")
    finally:
        if created_legacy_link:
            legacy_root.unlink()
    train_count = _filter_infos(output_root, train_names, "nuscenes_infos_train.pkl")
    val_count = _filter_infos(output_root, val_names, "nuscenes_infos_val.pkl")
    print(f"prepared train={train_count} infos and val={val_count} infos")


if __name__ == "__main__":
    main()
