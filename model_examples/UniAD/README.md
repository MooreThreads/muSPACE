# UniAD for muSPACE

## Contents

- [Overview](#overview)
  - [Model description](#model-description)
  - [Code implementation](#code-implementation)
- [Preparing the environment](#preparing-the-environment)
  - [Install dependencies](#install-dependencies)
  - [Install the MUSA stack](#install-the-musa-stack)
  - [Prepare the checkout](#prepare-the-checkout)
  - [Prepare the dataset](#prepare-the-dataset)
  - [Prepare pretrained weights](#prepare-pretrained-weights)
- [Quick start](#quick-start)
  - [Training](#training)
  - [Results](#results)
- [Changelog](#changelog)
- [FAQ](#faq)

# Overview

## Model description

UniAD is a planning-oriented end-to-end autonomous-driving model. It chains perception,
tracking, mapping, motion forecasting, and occupancy prediction into one network so that
every upstream task is queried by the planner rather than optimized in isolation. Its
BEV encoder and motion head are built from multi-scale deformable attention, which is
the operator muSPACE replaces on MUSA.

## Code implementation

- Reference implementation:

  ```shell
  url=https://github.com/OpenDriveLab/UniAD
  commit_id=7b5bf15e0e49522b6553ddc48e67833e8f5f0f52
  ```

- muSPACE recipe:

  ```shell
  code_path=model_examples/UniAD
  ```

muSPACE maintains no model fork. The recipe pins the upstream commit, applies the
compatibility patches under `patches/`, and routes multi-scale deformable attention to
`mu_space.ops.multi_scale_deformable_attn` at runtime.

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Upstream repository, pinned commit, patch marker, and patch digests |
| `requirements.lock` | Python 3.10 compatibility overrides for the legacy stack |
| `patches/` | Compatibility patches applied to the official checkout |
| `scripts/prepare_checkout.sh` | Fetches and verifies the pinned upstream commit |
| `scripts/train_1p_stage1_smoke.sh` | Single-card stage-1 training launch check |
| `scripts/train_step_musa.py` | Runs the training iteration behind the launch check |

# Preparing the environment

## Install dependencies

**Table 1** Third-party version support

| Package | Supported version |
| --- | --- |
| Python | `3.10` |
| mmcv-full | `1.5.3` |
| mmdet | `2.25.1` |
| mmseg | `0.25.0` |
| mmdet3d | `1.0.0rc4` |
| numpy | `1.26.4` |
| pytorch-lightning | `1.9.5` |
| torchmetrics | `0.11.4` |
| shapely | `2.0.x` |

The upstream requirements pin NumPy 1.20 and other Python 3.7-era releases that are not
installable on this stack. Install the overrides recorded in `requirements.lock` instead
of restoring the legacy pins, and use `numba`/`networkx` releases that match the
installed NumPy.

Upstream `docs/INSTALL.md` builds against mmdet3d `v0.17.1`. That release does not build
on Python 3.10, so this recipe targets `1.0.0rc4` instead — the same version FlashOcc
uses, which lets one environment serve both recipes. The compatibility patches under
`patches/` close the API gaps that the version bump opens.

UniAD also expects a sibling mmdetection3d checkout:

```bash
git clone --branch v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git UniAD/mmdetection3d
pip install -e UniAD/mmdetection3d --no-deps
pip install -r model_examples/UniAD/requirements.lock
```

## Install the MUSA stack

**Table 2** MUSA software support

| Component | Supported version |
| --- | --- |
| PyTorch | `2.9.1` |
| torch_musa | `2.9.1` |
| MUSA Toolkit | `5.1.0` |
| Device | MTT S5000, `mp_31` |

Install muSPACE with its native extension, then run `mu-space env` from the repository
root to confirm the active environment is reported as supported.

## Prepare the checkout

From the muSPACE repository root:

```bash
model_examples/UniAD/scripts/prepare_checkout.sh ./UniAD
mu-space migrate uniad --checkout ./UniAD
```

`prepare_checkout.sh` only fetches and verifies the pinned commit; it exits non-zero if
the origin or the resulting `HEAD` does not match. `mu-space migrate` applies the patch
recipe and writes `.mu_space_uniad.json` in the checkout. Re-running is safe: an already
migrated checkout reports `already-applied`, a partially patched one reports `upgraded`,
and a patch that no longer applies fails loudly instead of being skipped.

The eleven patches cover the Python 3.10 and device-compatibility gaps in this path,
including the `DiceCost` and match-cost registry collisions with newer mmdet releases,
NumPy boolean aliases, Shapely 2 geometry access, the occupancy-metrics backend swap
from the removed Lightning metrics API to torchmetrics, GridMask device placement, and
the panoptic matcher's CPU/MUSA index placement. Upstream implementations are preserved.

`mu-space restore uniad --checkout ./UniAD` reverses the recipe.

## Prepare the dataset

Obtain nuScenes from [nuscenes.org](https://www.nuscenes.org/nuscenes), including the
CAN bus expansion and the map expansion, then link the extracted tree into the checkout:

```bash
mkdir -p UniAD/data
ln -s /path/to/nuscenes UniAD/data/nuscenes
```

```
UniAD/data/nuscenes
├── can_bus
├── maps
│   ├── basemap
│   ├── expansion
│   └── prediction
├── samples
├── sweeps
├── v1.0-trainval
├── nuscenes_infos_temporal_train.pkl
└── nuscenes_infos_temporal_val.pkl
```

The temporal `.pkl` infos are generated by UniAD's own data-preparation tooling and are
not produced by this recipe. The launch check filters the supplied temporal train infos
down to one scene at runtime, so the full `v1.0-trainval/scene.json` metadata is
required while the sensor files only need to cover that scene (normally nuScenes
`85 scenes part 1`). CAN bus and map-expansion files are required in either case.

## Prepare pretrained weights

The launch check disables checkpoint loading so it can run offline. Full training uses
the upstream BEVFormer R101-DCN backbone weights and, for stage 2, the stage-1
track-and-map checkpoint; place both under `UniAD/ckpts/`.

# Quick start

## Training

Run one stage-1 (track and map) training iteration on a single card:

```bash
bash model_examples/UniAD/scripts/train_1p_stage1_smoke.sh --checkout ./UniAD
```

The launcher defaults to `projects/configs/stage1_track_map/base_track_map.py`, scene
`scene-0001`, one iteration, device `musa:0`, and `<checkout>/data/nuscenes` as the data
root. Each default is overridable by flag (`--config`, `--scene-name`, `--max-iters`,
`--device`, `--data-root`, `--work-dir`) or by the matching environment variable. The
`--device` value must be a MUSA device. Scratch metadata uses the system temporary
location by default so a read-only checkout remains runnable; pass `--work-dir` explicitly
when a controlled location is required (for repository-local validation, use
`<repository>/.tmp`). It applies the muSPACE deformable-attention patch in strict mode before
any upstream module is imported, so the run exercises the native MUSA kernel rather than a
fallback. The optimizer is built from the config through OpenMMLab's `build_optimizer`,
including its configured parameter rules.

A successful run prints one line:

```
uniad_train_step_ok optimizer=AdamW param_groups=<N> loss=<finite> logs={...}
```

The launch check replaces the image backbone's legacy MMCV DCNv2 blocks with regular
convolutions, because that optional extension has no MUSA replacement in the
compatibility environment. The rest of the stage-1 data, temporal, map, forward, loss,
backward, and optimizer path is unchanged. DCNv2-enabled training is outside this check.

The stage-1 config matches with `HungarianAssigner`, `HungarianAssigner_multi_info`, and
`HungarianAssigner3D`, so the launch check does not execute `HungarianAssigner_filter`.
The device-placement fix in `uniad-10-panseg-assigner-device.patch` is applied in the
checkout but is not covered by this run.

## Results

No muSPACE accuracy or throughput numbers are published for UniAD. The script above is a
launch check that proves the stage-1 path completes on MUSA with a finite loss; it is not
a benchmark and must not be quoted as performance data. It is also not a substitute for
full training, which needs all trainval temporal infos, tracking and map annotations, and
the stage-specific checkpoints.

# Changelog

2026-08-13: First release of the UniAD recipe — pinned checkout, eleven compatibility
patches, deformable-attention runtime integration, and the single-card stage-1 launch
check.

# FAQ

1. **`prepare_checkout.sh` exits with `unexpected origin`.** The destination directory is
   an unrelated git repository. Remove it or pass a fresh path.
2. **`mu-space migrate` reports `conflict`.** The checkout is not at the pinned commit or
   has local edits that overlap the recipe. Restore it to a clean pinned state and retry.
3. **`does not match its manifest digest`.** A patch file was modified after the manifest
   was written. Restore the patch from git rather than updating the digest.
4. **`ModuleNotFoundError` for `mmdet3d`.** The sibling `v1.0.0rc4` checkout is missing, or
   the environment still has upstream's `v0.17.1`, which does not build on Python 3.10.
5. **The launch check fails to find the scene.** The data root must contain the full
   `v1.0-trainval/scene.json` metadata, and the temporal train infos must include samples
   from the requested scene.
6. **A DCNv2 extension error appears.** Only the full training path builds DCNv2; the
   launch check disables it on purpose.
