# OpenVLA

## Contents

- [Overview](#overview)
- [Code implementation](#code-implementation)
- [Preparing the environment](#preparing-the-environment)
- [Quick start](#quick-start)
- [Validation](#validation)
- [Known limitations](#known-limitations)

## Overview

This model example integrates the public `openvla/openvla` repository at the pinned revision
`c8f03f48af692657d3060c19588038c7220e9af9`. muSPACE does not maintain a model fork. The recipe
applies one explicit patch to the official fine-tuning entry point for a single MUSA card.

The current scope is a launchable MUSA training smoke with deterministic synthetic image, language
token, and 7-DoF action-token inputs. It checks the training loop and device path; it is not a
quality claim for the `openvla-7b` checkpoint.

## Code implementation

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Pinned upstream revision, patch metadata, and recipe entrypoints |
| `patches/` | MUSA device/autocast and single-card training patch |
| `scripts/prepare_checkout.sh` | Fetch and verify the pinned official checkout |
| `scripts/validate_fixture.sh` | Apply the patch and run the fixture contract checks |
| `scripts/train_step_musa.py` | Single-card MUSA training entrypoint |
| `fixture_validation.md` | Reproducible fixture command and acceptance contract |
| `requirements.lock` | Scope note for the supported MUSA smoke dependencies |

## Preparing the environment

Build and install the current muSPACE wheel before running the fixture so
`mu_space/native/libmuspace_ops.so` matches the checkout:

```bash
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir .tmp/wheelhouse
python -m pip install --force-reinstall --no-deps .tmp/wheelhouse/mu_space-*.whl
```

Prepare the upstream checkout in disposable storage:

```bash
model_examples/OpenVLA/scripts/prepare_checkout.sh .tmp/OpenVLA
```

The script verifies `origin`, the pinned commit, and `vla-scripts/finetune.py`.

## Quick start

Apply the recipe patch and launch the MUSA fixture:

```bash
python -m mu_space migrate openvla --checkout .tmp/OpenVLA
bash model_examples/OpenVLA/scripts/validate_fixture.sh \
  --checkout .tmp/OpenVLA \
  --output-dir .tmp/openvla_fixture
```

The command writes the JSON result and checkpoint below `.tmp/openvla_fixture`. The default run
uses three FP32 optimizer steps on `musa:0` with a fixed seed.

## Validation

The validation entrypoint requires `mode=fixture`, `device=musa:0`, finite loss and gradients,
a successful optimizer update, and a checkpoint. It keeps generated artifacts under `.tmp/`.

On a clean checkout, the upstream patch can be checked independently:

```bash
git -C .tmp/OpenVLA apply --check \
  "$PWD/model_examples/OpenVLA/patches/openvla-01-musa-single-card-finetune.patch"
```

## Known limitations

- Only one MUSA card is supported by the patch; multi-card DDP is rejected explicitly.
- bitsandbytes quantization, FlashAttention, TensorFlow/RLDS installation, and the full upstream
  dependency stack are outside this recipe.
- Real `openvla-7b` LoRA training and real dataset preparation are follow-up work.
