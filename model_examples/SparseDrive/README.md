# SparseDrive for muSPACE

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
  - [Connect the muSPACE operator](#connect-the-muspace-operator)
  - [Training](#training)
  - [Results](#results)
- [Changelog](#changelog)
- [FAQ](#faq)

# Overview

## Model description

SparseDrive is a sparse end-to-end autonomous-driving model. Following the Sparse4D
design, it performs detection and online mapping on sparse instance representations
instead of dense BEV grids, and runs motion prediction and planning in parallel so the
ego state is part of scene understanding. Its per-instance feature gathering is done by a
multi-camera multi-scale deformable aggregation operator, which is the operator muSPACE
replaces on MUSA.

## Code implementation

- Reference implementation:

  ```shell
  url=https://github.com/swc-17/SparseDrive
  commit_id=52c4c05b6d446b710c8a12eb9fb19d698b33cb2b
  ```

- muSPACE recipe:

  ```shell
  code_path=model_examples/SparseDrive
  ```

**The pinned commit is an unverified candidate.** It was taken from another integration
of the same upstream repository, and the operator sources at that commit could not be
compared against the lineage recorded in `docs/provenance.md` because the upstream
repository was unreachable from the validation environment. Confirm that
`projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu` at this commit matches
that lineage before treating the pin as authoritative.

muSPACE maintains no model fork. Unlike the other recipes, this one ships **no checkout
patches**: the integration is purely a runtime rebinding, so `manifest.yaml` declares the
checkout contract with an empty `patches:` list.

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Upstream repository, candidate pinned commit, and patch marker |
| `scripts/prepare_checkout.sh` | Fetches and verifies the pinned upstream commit |

# Preparing the environment

## Install dependencies

**Table 1** Third-party version support

| Package | Supported version |
| --- | --- |
| Python | `3.10` |
| mmcv | `1.x` |
| mmdet | `2.28.2` |

Install the remaining dependencies from the upstream checkout's own requirements. This
recipe ships no `requirements.lock` because no muSPACE-validated dependency set has been
established for SparseDrive yet.

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
model_examples/SparseDrive/scripts/prepare_checkout.sh ./SparseDrive
mu-space migrate sparsedrive --checkout ./SparseDrive
```

`prepare_checkout.sh` only fetches and verifies the pinned commit; it exits non-zero if
the origin or the resulting `HEAD` does not match. `mu-space migrate` verifies the commit
and writes `.mu_space_sparsedrive.json`; because the recipe has no patches, it leaves the
checkout contents untouched. Once the pin is confirmed, the checkout is only needed for
the model code itself — the operator is rebound in the running process.

## Prepare the dataset

Obtain nuScenes from [nuscenes.org](https://www.nuscenes.org/nuscenes) and link the
extracted tree into the checkout:

```bash
mkdir -p SparseDrive/data
ln -s /path/to/nuscenes SparseDrive/data/nuscenes
```

```
SparseDrive/data/nuscenes
├── can_bus
├── maps
├── samples
├── sweeps
└── v1.0-trainval
```

Generate the info files and initial anchors with SparseDrive's own preprocessing tooling;
this recipe does not produce them.

## Prepare pretrained weights

Full training uses an ImageNet ResNet-50 backbone, and stage 2 uses the stage-1
checkpoint. Both come from upstream; place them under `SparseDrive/ckpt/`.

# Quick start

## Connect the muSPACE operator

Either call muSPACE directly:

```python
from mu_space.ops import deformable_aggregation_function

out = deformable_aggregation_function(
    feature_maps, spatial_shape, scale_start_index, sampling_location, weights
)
```

Or rebind the upstream import at runtime, before the model modules are used:

```python
from mu_space.patcher import sparsedrive_patcher

sparsedrive_patcher.apply()               # backend=auto
# sparsedrive_patcher.apply(strict=True)  # require the native MUSA kernel
```

The patcher replaces `projects.mmdet3d_plugin.ops.deformable_aggregation_function` and
the already-imported `DAF` binding in `projects.mmdet3d_plugin.models.blocks`.
`sparsedrive_patcher.restore()` reverses it.

The native path requires MUSA tensors: `float32` features, sampling locations, and
weights; `int32` shapes and indices. `num_groups` must be in `[1, 8]` and must divide
`num_embeds`. The helper `feature_maps_format` is exported from `mu_space.ops` for model
glue code.

## Training

This recipe has no MUSA training launch check. Unlike FlashOcc and UniAD, SparseDrive is
integrated at the operator level only; no single-card training script has been validated,
and none is shipped here rather than shipping one that has not been run.

## Results

No muSPACE accuracy or throughput numbers are published for SparseDrive.

# Changelog

2026-08-13: Recipe restructured to the shared layout — added the checkout manifest and
`prepare_checkout.sh` with a candidate pin, and recorded the unverified-pin gap.

# FAQ

1. **Is the pinned commit trustworthy?** Not yet. See
   [Code implementation](#code-implementation); the operator lineage check is still open.
2. **Why are there no patches?** The integration is a runtime rebinding, so nothing in the
   checkout needs to change. `mu-space migrate sparsedrive` therefore only verifies the
   commit and writes the marker.
3. **`strict=True` raises instead of falling back.** That is intended: strict mode is how
   you assert the native MUSA kernel is in use. Drop `strict` to allow the reference path.
4. **`num_groups` errors.** The upstream constraint is carried over — `num_groups` must be
   in `[1, 8]` and divide `num_embeds`.
