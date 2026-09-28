# FlashOcc for muSPACE

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

FlashOcc is a camera-only 3D occupancy prediction model. It replaces the voxel-level
3D convolutions used by earlier occupancy networks with a bird's-eye-view feature map
and a channel-to-height transform, which keeps the occupancy head cheap enough for
deployment. The BEV features are produced by the BEVPoolV2 view transformer, which is
the operator muSPACE replaces on MUSA.

## Code implementation

- Reference implementation:

  ```shell
  url=https://github.com/Yzichen/FlashOCC
  commit_id=449dc7947d6549ccb72a99d6f6d8c02f166ec0dd
  ```

- muSPACE recipe:

  ```shell
  code_path=model_examples/FlashOCC
  ```

muSPACE maintains no model fork. The recipe pins the upstream commit, applies the
compatibility patches under `patches/`, and routes BEVPoolV2 to
`mu_space.ops.bev_pool_v2` at runtime.

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Upstream repository, pinned commit, patch marker, and patch digests |
| `requirements.lock` | Python 3.10 compatibility overrides for the legacy stack |
| `patches/` | Compatibility patches applied to the official checkout |
| `scripts/prepare_checkout.sh` | Fetches and verifies the pinned upstream commit |
| `scripts/train_1p_occ_smoke.sh` | Single-card occupancy training launch check |
| `scripts/train_step_musa.py` | Runs the training iteration behind the launch check |

# Preparing the environment

## Install dependencies

**Table 1** Third-party version support

| Package | Supported version |
| --- | --- |
| Python | `3.10` |
| mmcv-full | `1.5.3` |
| mmdet | `2.25.1` |
| mmdet3d | `1.0.0rc4` |
| numpy | `1.26.4` |

The upstream requirements pin Python 3.7-era releases that are not installable on this
stack. Install the overrides recorded in `requirements.lock` instead of restoring the
legacy pins.

```bash
pip install -e FlashOCC/projects --no-deps
pip install -r model_examples/FlashOCC/requirements.lock
```

The FlashOcc configs inherit from mmdetection3d by relative path
(`_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py', ...]`), so the checkout
needs a sibling `mmdetection3d/configs/` tree. Either clone the matching upstream tag:

```bash
git clone --branch v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git FlashOCC/mmdetection3d
pip install -e FlashOCC/mmdetection3d --no-deps
```

or, when the mmdet3d wheel is already installed, point at the configs it ships:

```bash
python - <<'PY'
import mmdet3d, os
print(os.path.join(os.path.dirname(mmdet3d.__file__), ".mim"))
PY
ln -s <printed path> FlashOCC/mmdetection3d
```

Both give the config loader the same `_base_` files. The launch check below was validated with
the second form.

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
model_examples/FlashOCC/scripts/prepare_checkout.sh ./FlashOCC
mu-space migrate flashocc --checkout ./FlashOCC
```

`prepare_checkout.sh` only fetches and verifies the pinned commit; it exits non-zero if
the origin or the resulting `HEAD` does not match. `mu-space migrate` applies the patch
recipe and writes `.mu_space_flashocc.json` in the checkout. Re-running is safe: an
already migrated checkout reports `already-applied`, a partially patched one reports
`upgraded`, and a patch that no longer applies fails loudly instead of being skipped.

The patches guard the CUDA-only BEVPool, nearest-assign, and ray-metrics extensions so
they do not need to be built when the muSPACE replacement is used, keep the pose
transforms in `float32`, and defer ray-metrics compilation until that optional
evaluation metric is called.

`mu-space restore flashocc --checkout ./FlashOCC` reverses the recipe.

## Prepare the dataset

Obtain nuScenes from [nuscenes.org](https://www.nuscenes.org/nuscenes) together with the
Occ3D-nuScenes occupancy labels, then link the extracted tree into the checkout:

```bash
mkdir -p FlashOCC/data
ln -s /path/to/nuscenes FlashOCC/data/nuscenes
```

```
FlashOCC/data/nuscenes
├── can_bus
├── gts
│   └── scene-0001
├── maps
├── samples
├── sweeps
└── v1.0-trainval
```

The launch check builds its metadata at runtime from a single scene, so the full
`v1.0-trainval/` metadata is required while the sensor files only need to cover that
scene (normally nuScenes `85 scenes part 1`). The matching occupancy labels under
`gts/<scene>/` are also required. Full training still requires all trainval sensor data
and occupancy annotations; nuScenes-mini alone is not sufficient.

## Prepare pretrained weights

The launch check initializes the image backbone randomly and needs no checkpoint. Full
training uses the upstream ResNet-50 BEVDet backbone weights; place them under
`FlashOCC/ckpts/` and keep the config's `pretrained` entry pointing at that path.

# Quick start

## Training

Run one occupancy training iteration on a single card:

```bash
bash model_examples/FlashOCC/scripts/train_1p_occ_smoke.sh --checkout ./FlashOCC
```

The launcher defaults to `projects/configs/flashocc/flashocc-r50.py`, scene
`scene-0001`, one iteration, device `musa:0`, and `<checkout>/data/nuscenes` as the data
root. Each default is overridable by flag (`--config`, `--scene-name`, `--max-iters`,
`--device`, `--data-root`, `--work-dir`) or by the matching environment variable. The
`--device` value must be a MUSA device. Scratch metadata uses the system temporary
location by default so a read-only checkout remains runnable; pass `--work-dir` explicitly
when a controlled location is required (for repository-local validation, use
`<repository>/.tmp`). It applies the muSPACE BEVPoolV2 patch in strict mode before any
upstream module is imported, so the run exercises the native MUSA kernel rather than a
fallback. The optimizer is built from the config through OpenMMLab's `build_optimizer`,
including its configured parameter rules.

A successful run prints one line:

```
flashocc_train_step_ok optimizer=AdamW param_groups=<N> loss=<finite> logs={...}
```

## Results

No muSPACE accuracy or throughput numbers are published for FlashOcc. The script above
is a launch check that proves the data, forward, loss, backward, and optimizer path
completes on MUSA with a finite loss; it is not a benchmark and must not be quoted as
performance data.

# Changelog

2026-08-13: First release of the FlashOcc recipe — pinned checkout, five compatibility
patches, BEVPoolV2 runtime integration, and the single-card occupancy launch check.

# FAQ

1. **`prepare_checkout.sh` exits with `unexpected origin`.** The destination directory is
   an unrelated git repository. Remove it or pass a fresh path.
2. **`mu-space migrate` reports `conflict`.** The checkout is not at the pinned commit or
   has local edits that overlap the recipe. Restore it to a clean pinned state and retry.
3. **`does not match its manifest digest`.** A patch file was modified after the manifest
   was written. Restore the patch from git rather than updating the digest.
4. **The build asks for the CUDA BEVPool extension.** Run `mu-space migrate flashocc`
   first; the import guards are part of the recipe, not of the upstream checkout.
5. **The launch check fails to find the scene.** The data root must contain the full
   `v1.0-trainval/` metadata and the `gts/<scene>/` occupancy labels for the requested
   scene.
