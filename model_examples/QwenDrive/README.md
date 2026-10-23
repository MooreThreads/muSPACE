# Qwen-Drive for muSPACE

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
  - [Training step](#training-step)
  - [Results](#results)
- [Changelog](#changelog)
- [FAQ](#faq)

## Overview

### Model description

Qwen-Drive-1.0 is a vision-language driving model. This recipe covers only its perception head:
a BEV encoder and decoder, fed by two frozen Qwen3.5-4B feature taps, predicting 3D boxes,
occupancy, and a BEV map. The planning head is not part of the recipe.

The published perception package is an inference implementation. It has no loss, dataloader, or
optimizer, and `QwenDrivePerception.infer` is wrapped in `torch.no_grad`. muSPACE does not add
those to the checkout. The training entry point calls the head (`bev_modeling`) on the same taps
`infer` would have built, with the backbone left frozen.

Two custom operators are on that path. Depth-aware voxel pooling lifts ViT features into the BEV
volume. Multi-scale deformable attention samples the language-model feature pyramid. The published
attention call is a CUDA bfloat16 forward without a backward, and the view-transform frustum is
built on CUDA or CPU. Both are rebound at runtime onto the muSPACE autograd operators.

| Component | Requirement |
| --- | --- |
| Python | `3.10` |
| PyTorch | MUSA build matching the installed `torch_musa` |
| torch_musa | Matching the installed PyTorch build |
| MUSA device | MTT S5000 |
| Qwen-Drive | `28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0` |
| Head dtype | `bfloat16` |

The check is one optimizer step. It does not claim convergence, paper-training parity, or a
multi-card throughput.

### Code implementation

- Reference implementation:

  ```shell
  url=https://github.com/QwenLM/Qwen-Drive-1.0
  commit_id=28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0
  ```

- muSPACE recipe:

  ```shell
  code_path=model_examples/QwenDrive
  ```

muSPACE maintains no fork and ships no checkout patches. `qwendrive_patcher` rebinds three
published entry points in the running process:

- `qwen_drive_perception.attention.multi_scale_deformable_attn_cuda`
- `qwen_drive_perception.ops.ms_deform_attn_bf16_forward` and `voxel_pool_depth`
- the `frustum` property on `Uni3DVoxelPoolDepth`, so the grid is built on the MUSA device and
  stays fp32 across `model.to(bfloat16)`

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Upstream repository, pinned commit, and the runtime patcher |
| `requirements.lock` | Perception-side packages that must not replace the MUSA stack |
| `scripts/prepare_checkout.sh` | Fetches and verifies the pinned upstream commit |
| `scripts/train_step_musa.py` | One frozen-backbone perception-head step |
| `fixture_validation.md` | What the step proves, and what it does not |

## Preparing the environment

### Install dependencies

Install the matching MUSA PyTorch build first. Then install the perception-side packages named in
`requirements.lock` (`transformers`, `accelerate`, `safetensors`, `numpy`, `pillow`). The lock
intentionally does not pin them and does not list torch, torch_musa, flash-attn,
flash-linear-attention, or causal-conv1d.

The pinned upstream requirements were verified by the publisher on CUDA 12.8 with torch 2.8.0.
Installing that file into a MUSA environment replaces the supported stack. The frozen VLM forward
can need extra packages from the checkout; add those only after checking they do not downgrade
torch or torch_musa.

### Install the MUSA stack

Install muSPACE with its native extension, then run `mu-space env` and confirm the active
environment is a supported MUSA stack with a visible device. The training entry point rejects a
non-MUSA `--device`.

### Prepare the checkout

From the muSPACE repository root:

```bash
model_examples/QwenDrive/scripts/prepare_checkout.sh ./QwenDrive
mu-space migrate qwendrive --checkout ./QwenDrive
```

`prepare_checkout.sh` only fetches and verifies the pinned commit. It exits non-zero when the
origin or the resulting `HEAD` does not match. `mu-space migrate qwendrive` checks that commit and
writes `.mu_space_qwendrive.json`. The patch list is empty, so a second migrate reports
`already-applied` and still does not edit the checkout. `mu-space restore qwendrive` removes the
marker.

### Prepare the dataset

Fixture mode does not read a dataset. Real mode reads one packed demo frame:

```text
<frame-token>/
├── frame.json
├── images/
├── calib.npz
├── gt.npz          # opened by the published loader; used when a label grid matches
└── lidar.npy       # not read by the training step
```

Point `--frames` at the directory that contains those frame directories. The script does not
download frames. A nuScenes or nuPlan root in the upstream training layout is not consumed
directly; pack a frame the way the published `PerceptionFrame` loader expects, or use a released
demo frame.

### Prepare pretrained weights

Fixture mode constructs `BEVFormerModelV2` from `QwenDrivePerceptionConfig` and does not load a
checkpoint. Real mode expects a Qwen-Drive-1.0-4B directory and the released `perception/`
weights inside it. The script does not download either.

## Quick start

### Runtime patching

Apply the patcher after the checkout's `src` directory is on `PYTHONPATH`, and before the head is
constructed. `train_step_musa.py` does this itself.

```python
import sys

sys.path.insert(0, "QwenDrive/src")

from mu_space.patcher import qwendrive_patcher

qwendrive_patcher.apply()               # backend=auto
# qwendrive_patcher.apply(strict=True)  # require the native MUSA kernels
```

`qwendrive_patcher.restore()` puts the published callables and the frustum property back. On a
MUSA tensor that is not bfloat16, the rebound attention raises; the published kernel has the same
dtype constraint. Non-MUSA tensors keep the published PyTorch fallback when `--strict` is not set.
Strict mode passes `backend="musa"` and fails if the native kernel cannot run.
`MUSPACE_STRICT_PATCH=1` selects strict mode when `apply()` is called without an argument.

### Training step

```bash
python model_examples/QwenDrive/scripts/train_step_musa.py \
  --checkout ./QwenDrive --mode fixture --device musa:0 --strict
```

`--device-count 4` repeats the head step on `musa:0` through `musa:3`, one card after another, on a
single head moved between devices. The script selects each card as the current MUSA device before
the move. In real mode the frozen VLM runs once on the first device. The count is a placement check,
not a throughput.

The head is the only module in the optimizer. The step prints one JSON object whose `status` is
`qwendrive_perception_train_step_ok`. See [`fixture_validation.md`](fixture_validation.md) for the
fields that decide a pass.

### Results

No accuracy or throughput number is published for this recipe. A passing step shows a finite
stand-in loss, a positive finite head-gradient norm, an optimizer update, and no backbone
gradient. It does not show that training converges or that the stand-in matches the upstream
objective.

## Changelog

2026-10-16: Added the runtime-only recipe, `prepare_checkout.sh`, and a single-step MUSA training
entry point for the frozen-backbone perception head.

## FAQ

### Does this modify the Qwen-Drive checkout?

No. The checkout stays at the pinned commit. The patcher changes bindings for the process that
applies it, and `restore()` reverses them.

### Why is the loss not the training loss from the paper?

The published perception code does not include a supervised detection loss. Matching predictions to
ground-truth boxes would be a new training objective, not a check that the released head can take
a gradient step. The stand-in only requires the returned logits to be finite and connected to the
head parameters. Cross-entropy is used for occupancy or map only when a packed `gt.npz` already
contains a label grid of the right shape.

### Why does fixture mode not load the 4B backbone?

The backbone is frozen either way, and its published forward is not part of the head's backward.
Fixture mode is the check that still runs when the VLM cannot be loaded or its forward is not
available. Real mode is the check that the frozen taps and the head step agree on one frame.
Neither mode unfreezes the backbone, and neither trains the planning head.

### Why does `--device-count` not use DDP or FSDP?

The acceptance bar is one finite step on more than one device, not a scaling run. A Python loop
keeps that obvious. Do not report its wall time as throughput.

### `strict=True` raises instead of falling back.

That is intended. Drop `--strict` to allow the reference path. A reference step does not show that
the native MUSA kernels ran.
