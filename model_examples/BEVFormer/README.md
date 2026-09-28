# BEVFormer for muSPACE

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

# Overview

## Model description

| Component | Requirement |
| --- | --- |
| Python | `3.10` |
| PyTorch | `>=2.7` (recommended: `2.9.1`) |
| torch_musa | `>=2.7` (recommended: `2.9.1`, matching PyTorch) |
| MUSA Toolkit | `5.2.0` |
| Device | MTT S5000, `mp_31` |
| BEVFormer | `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` |

Run `mu-space env` from the muSPACE repository root to inspect the active environment.
PyTorch and torch_musa must use matching versions. Version 2.7 is the minimum supported version;
2.9.1 is recommended for both.

BEVFormer is a camera-only 3D perception model. It builds bird's-eye-view features with
spatial cross-attention over multi-camera images and temporal self-attention over the
previous BEV state, so the same BEV representation serves detection and segmentation.
Both attention modules are multi-scale deformable attention, which is the operator
muSPACE replaces on MUSA. BEVFormer is muSPACE's first supported model target.

## Code implementation

- Reference implementation:

  ```shell
  url=https://github.com/fundamentalvision/BEVFormer
  commit_id=66b65f3a1f58caf0507cb2a971b9c0e7f842376c
  ```

- muSPACE recipe:

  ```shell
  code_path=model_examples/BEVFormer
  ```

muSPACE maintains no model fork. The recipe pins the upstream commit, applies the
compatibility patches under `patches/`, and routes multi-scale deformable attention to
`mu_space.ops.multi_scale_deformable_attn` at runtime.

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Upstream repository, pinned commit, patch marker, and patch digests |
| `requirements.lock` | Python dependencies for the legacy BEVFormer/OpenMMLab stack |
| `patches/` | Compatibility patches applied to the official checkout |
| `scripts/prepare_checkout.sh` | Fetches and verifies the pinned upstream commit |
| `scripts/build_openmmlab_compat_wheels.sh` | Builds the legacy OpenMMLab compatibility wheels |
| `scripts/validate_fixture.sh` | Runs the one-command fixture validation |
| `scripts/profile_fixture.sh` | Runs fixture profiling and optional forward tracing |
| `scripts/train_step_musa.py` | Runs one BEVFormer-tiny optimizer step |
| `scripts/trace_bevformer_musa.py` | Captures compact forward trace diagnostics |
| `scripts/diff_bevformer_trace.py` | Compares two trace JSON files |
| `scripts/smoke_openmmlab.py` | Checks the legacy OpenMMLab imports and disabled native ops |
| `scripts/audit_cuda_usage.py` | Reports CUDA-specific source usage in a checkout |
| `scripts/smoke_reference_modules.py` | Runs small attention-module MUSA smoke tests |
| `scripts/smoke_dcn_backbone.py` | Runs the ResNet-DCN backbone through the muSPACE DCNv2 patcher |
| `fixture_validation.md` | Acceptance criteria and troubleshooting for fixture validation |
| `fixture_profiling.md` | Profiling outputs, trace mode, and result interpretation |

# Preparing the environment

## Install dependencies

**Table 1** Third-party version support

| Package | Supported version |
| --- | --- |
| Python | `3.10` |
| mmcv-full | `1.4.0` |
| mmdet | `2.14.0` |
| mmsegmentation | `0.14.1` |
| mmdet3d | `0.17.1` |
| numpy | `1.24.0` |

`mmcv-full` 1.4.0 and `mmdet3d` 0.17.1 predate Python 3.10 and do not build as
published. Build the compatibility wheels first, then install the pinned stack:

```bash
model_examples/BEVFormer/scripts/build_openmmlab_compat_wheels.sh
pip install --no-deps wheelhouse/mmcv_full-1.4.0-*.whl
pip install --no-deps wheelhouse/mmdet3d-0.17.1-*.whl
pip install --no-deps mmdet==2.14.0 mmsegmentation==0.14.1
pip install -r model_examples/BEVFormer/requirements.lock --no-deps
```

That script applies the `deps-*.patch` files under `patches/` to the extracted sdists.
Those patches target the dependency sources rather than the model checkout, so they are
tracked separately in `manifest.yaml` and `mu-space migrate` never applies them.

If optional mmdet3d evaluation imports are missing:

```bash
pip install lyft_dataset_sdk nuscenes-devkit plyfile pyquaternion shapely
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
model_examples/BEVFormer/scripts/prepare_checkout.sh ./BEVFormer
mu-space migrate bevformer --checkout ./BEVFormer
```

`prepare_checkout.sh` only fetches and verifies the pinned commit; it exits non-zero if
the origin or the resulting `HEAD` does not match. `mu-space migrate` applies the patch
recipe and writes `.mu_space_bevformer.json` in the checkout. Re-running is safe: an
already migrated checkout reports `already-applied`, a partially patched one reports
`upgraded`, and a patch that no longer applies fails loudly instead of being skipped.

`mu-space restore bevformer --checkout ./BEVFormer` reverses the recipe.

## Prepare the dataset

The fixture validation below is fully synthetic and needs no dataset. For full training,
obtain nuScenes from [nuscenes.org](https://www.nuscenes.org/nuscenes), including the CAN
bus expansion, and link the extracted tree into the checkout:

```bash
mkdir -p BEVFormer/data
ln -s /path/to/nuscenes BEVFormer/data/nuscenes
```

```
BEVFormer/data/nuscenes
├── can_bus
├── maps
├── samples
├── sweeps
└── v1.0-trainval
```

Generate the info files with BEVFormer's own data-preparation tooling; this recipe does
not produce them.

## Prepare pretrained weights

The fixture validation initializes weights randomly and needs no checkpoint. Full
training uses the upstream R101-DCN backbone weights; place them under `BEVFormer/ckpts/`.

# Quick start

## Runtime patching

Call the muSPACE runtime patcher before importing BEVFormer modules that bind mmcv's
deformable-attention helper:

```python
from mu_space.patcher import default_patcher

default_patcher.apply()
```

The default mode is compatibility-first. Set `MUSPACE_STRICT_PATCH=1` to require the
native MUSA path during validation.

The base and small configurations build their image backbone with
`dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False)` and
`stage_with_dcn=(False, False, True, True)`. mmdet resolves that `type` through mmcv's
`CONV_LAYERS` registry, so a second patcher installs the muSPACE DCNv2 layer:

```python
from mu_space.patcher import modulated_deform_conv_patcher

modulated_deform_conv_patcher.apply()
```

Apply it before `build_model`. Parameter names and the state dict layout match mmcv's module, so
BEVFormer checkpoints load unchanged. The tiny configuration used by the fixture is a plain
ResNet-50 with no DCN, so this patch only affects the base and small recipes.

`scripts/train_step_musa.py` and `scripts/trace_bevformer_musa.py` already apply both patchers;
the snippets above are for your own entry points. Without the DCNv2 patch a base or small
configuration builds mmcv's own module, which raises on its disabled native extension at the
first forward pass.

## Validation

Run the fixture correctness check:

```bash
model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer
model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer --strict
```

The check runs one small BEVFormer-tiny optimizer step on synthetic inputs and accepts
the result only when the JSON reports `mode: "fixture"`, `finite_gradients: true`,
`optimizer_step: true`, a finite `loss`, and a non-empty `losses` object.

Run diagnostic profiling:

```bash
model_examples/BEVFormer/scripts/profile_fixture.sh --checkout ./BEVFormer
model_examples/BEVFormer/scripts/profile_fixture.sh --checkout ./BEVFormer --trace
```

Check that the DCN backbone actually runs on the muSPACE operator (needs the OpenMMLab stack):

```bash
python model_examples/BEVFormer/scripts/smoke_dcn_backbone.py
python model_examples/BEVFormer/scripts/smoke_dcn_backbone.py --depth 101
```

It applies the DCNv2 patcher, builds the base-config image backbone, asserts the deformable
layers are muSPACE modules, and runs one forward and backward pass on MUSA.

Fixture validation checks a small synthetic BEVFormer-tiny optimizer step. It requires
`finite_gradients: true`, `optimizer_step: true`, a finite `loss`, and a non-empty `losses` object.
Profiling output is for triage and comparison only; it is not a release performance benchmark.

## Results

No muSPACE accuracy or throughput numbers are published for BEVFormer. The fixture
validation is a correctness check, and the profiling output is for triage and comparison
only; neither is a release benchmark and neither may be quoted as performance data.

# Changelog

2026-08-13: Recipe restructured to the shared layout — ordered patch names, per-patch
digests in the manifest, and dependency patches tracked separately from checkout patches.

# FAQ

1. **`prepare_checkout.sh` exits with `unexpected origin`.** The destination directory is
   an unrelated git repository. Remove it or pass a fresh path.
2. **`mu-space migrate` reports `conflict`.** The checkout is not at the pinned commit or
   has local edits that overlap the recipe. Restore it to a clean pinned state and retry.
3. **`does not match its manifest digest`.** A patch file was modified after the manifest
   was written. Restore the patch from git rather than updating the digest.
4. **Imports fail after installation.** Rebuild the OpenMMLab compatibility wheels and
   check `mu-space env`.
5. **Default validation passes but `--strict` fails.** The fallback path works but the
   native path is unavailable or failing. Inspect `MUSPACE_DISPATCH_LOG=1` output.
6. **CUDA-specific code remains after migration.** Run
   `python model_examples/BEVFormer/scripts/audit_cuda_usage.py ./BEVFormer`.

See `docs/install.md`, `fixture_validation.md`, and `fixture_profiling.md` for the full
workflow.
