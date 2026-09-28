# BEVDepth for muSPACE

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
  - [Runtime patching](#runtime-patching)
  - [Validation](#validation)
  - [Results](#results)
- [Changelog](#changelog)
- [FAQ](#faq)

## Overview

### Model description

| Component | Requirement |
| --- | --- |
| Python | `3.10` |
| PyTorch | MUSA build matching the installed `torch_musa` |
| torch_musa | Matching the installed PyTorch build |
| MUSA device | MTT S5000 |
| BEVDepth | `d78c7b58b10b9ada940462ba83ab24d99cae5833` |
| Dataset fixture | nuScenes trainval, scenes `scene-0001`–`scene-0012` |

BEVDepth is a multi-camera 3D detection model. Its training pipeline projects image
features into BEV voxels with `voxel_pooling_train`; muSPACE provides the MUSA native
implementation for this training operator.

The recipe covers deterministic single-card training (up to five epochs, batch size 1)
with real nuScenes data. It is a launch and semantic-regression check, not a replacement
for full nuScenes multi-card convergence training.

### Code implementation

- Reference implementation:

  ```shell
  url=https://github.com/Megvii-BaseDetection/BEVDepth
  commit_id=d78c7b58b10b9ada940462ba83ab24d99cae5833
  ```

- muSPACE recipe:

  ```shell
  code_path=model_examples/BEVDepth
  ```

muSPACE maintains no BEVDepth fork and does not edit the upstream checkout. The runtime
patcher redirects the training `voxel_pooling_train` import and mmcv's standard `DCN`
registry to canonical muSPACE APIs. Standard DCN uses the existing native modulated
kernel with a fixed all-ones mask. In strict mode, reference fallback is rejected.

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Upstream repository, pinned commit, and training fixture |
| `requirements.lock` | Shared Python dependencies for the upstream training stack |
| `requirements-runtime.lock` | Explicit legacy runtime closure without changing platform PyTorch |
| `scripts/prepare_checkout.sh` | Fetches and verifies the pinned upstream commit |
| `scripts/install_dependencies.sh` | Installs the shared dependency lock |
| `scripts/prepare_scene_data.py` | Links read-only nuScenes data and filters train/validation scenes |
| `scripts/extract_scene_subset.py` | Extracts selected scenes from official nuScenes blob archives |
| `scripts/train_step_musa.py` | Convenience entry point for MUSA training |
| `scripts/train_step.py` | Shared MUSA/native-CUDA training implementation |
| `scripts/validate_fixture.sh` | Runs and validates the real-data single-step check |
| `fixture_validation.md` | Validation boundary, acceptance criteria, and troubleshooting |

## Preparing the environment

### Install dependencies

Install the matching MUSA PyTorch build first. Then install the shared OpenMMLab and
data-loader dependencies:

```bash
model_examples/BEVDepth/scripts/install_dependencies.sh
```

The lock file keeps the legacy BEVDepth/OpenMMLab versions together. The MUSA/CUDA
PyTorch package is intentionally installed separately because it is platform-specific.

### Install the MUSA stack

Install muSPACE with its native extension, then verify that PyTorch and `torch_musa` are
matching builds and that an MTT S5000 device is visible. The training entry point fails
if MUSA is unavailable or if strict native dispatch cannot be established.

### Prepare the checkout

From the muSPACE repository root:

```bash
model_examples/BEVDepth/scripts/prepare_checkout.sh ./BEVDepth
```

The script only fetches and verifies the pinned upstream commit. It does not apply a
source patch or maintain a fork. Re-running it is safe when the checkout has the expected
origin and revision.

### Prepare the dataset

Obtain the nuScenes trainval metadata and sensor files from [nuscenes.org](https://www.nuscenes.org/nuscenes).
Keep those inputs read-only and pass their two roots to the preparation script:

```bash
model_examples/BEVDepth/scripts/extract_scene_subset.py \
  --download-root <NUSCENES_DOWNLOAD_ROOT> \
  --metadata-root <NUSCENES_METADATA_ROOT>/v1.0-trainval \
  --output-root .tmp/nuscenes-scene-subset \
  --scenes scene-0001:scene-0012

model_examples/BEVDepth/scripts/prepare_scene_data.py \
  --checkout ./BEVDepth \
  --meta-root <NUSCENES_METADATA_ROOT> \
  --scene-root .tmp/nuscenes-scene-subset \
  --output-root <PREPARED_NUSCENES_ROOT> \
  --train-scenes scene-0001:scene-0010 \
  --val-scenes scene-0011:scene-0012
```

The script links `maps`, `samples`, and `v1.0-trainval`, runs BEVDepth's own info
generator, and keeps only the selected scene in `nuscenes_infos_train.pkl`. It does not
copy or mutate the supplied dataset.

### Prepare pretrained weights

The single-step recipe initializes the upstream model weights from its default model
construction path and does not require a checkpoint. For a longer training run, follow
the upstream BEVDepth instructions for obtaining and loading pretrained weights.

## Quick start

### Runtime patching

For a custom BEVDepth entry point, apply the patcher before importing or constructing the
upstream training model:

```python
from mu_space.patcher import BEVDepthPatcher

patcher = BEVDepthPatcher()
patcher.apply(strict=True)
```

The supplied `train_step_musa.py` entry point applies and restores this patcher for you.
When the upstream CUDA extension is unavailable on MUSA, the patcher supplies only the
known training-module import surface and binds it to muSPACE. `strict=True` requires
native MUSA dispatch and never silently selects the reference implementation.

### Validation

After preparing the checkout and scene data, run the real-data training recipe. The
default is five epochs and an eight-hour wall-clock limit; `--max-steps` is useful for
a quick smoke test:

```bash
model_examples/BEVDepth/scripts/validate_fixture.sh \
  --checkout ./BEVDepth \
  --data-root <PREPARED_NUSCENES_ROOT> \
  --dtype float32 --epochs 5 --max-hours 8

model_examples/BEVDepth/scripts/validate_fixture.sh \
  --checkout ./BEVDepth \
  --data-root <PREPARED_NUSCENES_ROOT> \
  --dtype float16 --epochs 5 --max-hours 8

# On an A100, omit the muSPACE patcher and use the pinned upstream CUDA extension:
DEVICE=cuda model_examples/BEVDepth/scripts/validate_fixture.sh \
  --checkout ./BEVDepth --data-root <PREPARED_NUSCENES_ROOT> \
  --dtype float32 --device cuda --max-steps 1
```

The command checks that loss and gradients are finite, an optimizer update occurs, and
the MUSA run reports native voxel pooling and standard DCN dispatch. CUDA mode leaves
the upstream extension untouched. Add `--output .tmp/results/musa-fp32.json` to persist
the structured result (loss trend, commit/data hash, parameter update hash and dispatch).
See
[`fixture_validation.md`](fixture_validation.md) for the exact acceptance boundary.

### Results

The command prints a compact JSON result for the step. The recipe does not publish
accuracy or throughput numbers.

## Changelog

2026-08-31: Restructured the BEVDepth recipe to follow the shared model-example workflow
used by BEVFormer, including environment preparation, dataset preparation, quick start,
and fixture validation documentation.

## FAQ

### Does this modify BEVDepth source code?

No. `prepare_checkout.sh` pins a clean upstream checkout, and the runtime patcher changes
bindings only for the duration of the process.

### Why is the fixture limited to twelve scenes and five epochs?

It provides a reproducible real-data launch check while keeping the validation boundary
small. It does not claim full-dataset training convergence.

### Why does validation fail when the native module is unavailable?

The recipe intentionally uses strict dispatch. Install a muSPACE wheel containing the
MUSA `voxel_pooling_train` native symbols and verify the active MUSA device before retrying.
