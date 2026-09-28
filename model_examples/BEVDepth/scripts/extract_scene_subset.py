#!/usr/bin/env python3
"""Extract only selected nuScenes scenes from the official tar archives."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def scene_names(spec: str) -> list[str]:
    names = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            names.append(part)
            continue
        first, last = part.split(":", 1)
        prefix_a, number_a = first.rsplit("-", 1)
        prefix_b, number_b = last.rsplit("-", 1)
        if prefix_a != prefix_b:
            raise ValueError(f"scene range must share a prefix: {part}")
        names.extend(f"{prefix_a}-{i:04d}" for i in range(int(number_a), int(number_b) + 1))
    if not names:
        raise ValueError("at least one scene is required")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenes", default="scene-0001:scene-0012")
    args = parser.parse_args()
    selected = set(scene_names(args.scenes))
    scenes = json.loads((args.metadata_root / "scene.json").read_text())
    samples = {
        item["token"]: item for item in json.loads((args.metadata_root / "sample.json").read_text())
    }
    sample_data = json.loads((args.metadata_root / "sample_data.json").read_text())
    wanted: set[str] = set()
    matched = []
    for scene in scenes:
        if scene["name"] not in selected:
            continue
        matched.append(scene["name"])
    selected_sample_tokens = set()
    for scene in scenes:
        if scene["name"] not in selected:
            continue
        token = scene["first_sample_token"]
        while token:
            selected_sample_tokens.add(token)
            token = samples[token]["next"]
    wanted.update(
        item["filename"].lstrip("./")
        for item in sample_data
        if item["sample_token"] in selected_sample_tokens
    )
    missing_scenes = selected - set(matched)
    if missing_scenes:
        raise ValueError(f"scenes absent from metadata: {sorted(missing_scenes)}")
    destination = args.output_root / "samples"
    destination.mkdir(parents=True, exist_ok=True)
    # prepare_scene_data.py expects a self-contained scene root. Metadata and
    # map files remain read-only symlinks into the supplied metadata checkout.
    for name, source in (
        ("maps", args.metadata_root.parent / "maps"),
        ("v1.0-trainval", args.metadata_root),
    ):
        link = args.output_root / name
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == source.resolve():
                continue
            raise FileExistsError(link)
        link.symlink_to(source, target_is_directory=True)
    archives = sorted(args.download_root.glob("v1.0-trainval*_blobs.tgz"))
    if not archives:
        raise FileNotFoundError(f"no trainval blob archives under {args.download_root}")
    extracted: set[str] = set()
    for archive in archives:
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                name = member.name.lstrip("./")
                if name not in wanted:
                    continue
                target = args.output_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as data, target.open("wb") as output:
                    output.write(data.read())
                extracted.add(name)
    missing_files = sorted(wanted - extracted)
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} selected sensor files were not found; examples: {missing_files[:3]}"
        )
    print(
        json.dumps(
            {
                "scenes": sorted(selected),
                "sensor_files": len(extracted),
                "output_root": str(args.output_root),
            }
        )
    )


if __name__ == "__main__":
    main()
