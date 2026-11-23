# VAD for muSPACE

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
  - [Inference](#inference)
- [Results](#results)
- [Changelog](#changelog)
- [FAQ](#faq)

# Overview

## Model description

VAD (Vectorized Autonomous Driving) is a planning-oriented end-to-end autonomous-driving
model built on a vectorized scene representation. It encodes the multi-camera surround view
into a bird's-eye-view feature map, decodes vectorized map elements and surrounding agents,
and predicts its own future trajectory, so the planning query conditions every upstream
representation. Its spatial cross attention and temporal self attention are built from
multi-scale deformable attention, which is the operator muSPACE replaces on MUSA.

## Code implementation

- Reference implementation:

  ```shell
  url=https://github.com/hustvl/VAD
  commit_id=1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8
  ```

- muSPACE recipe:

  ```shell
  code_path=model_examples/VAD
  ```

muSPACE maintains no model fork. The recipe pins the upstream commit, applies the
compatibility patches under `patches/`, and routes multi-scale deformable attention to
`mu_space.ops.multi_scale_deformable_attn` at runtime.

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Upstream repository, pinned commit, patch marker, and patch digests |
| `requirements.lock` | Python 3.10 compatibility overrides for the legacy stack |
| `patches/` | Compatibility patches, including the pure-Python mmcv extension patch |
| `scripts/prepare_checkout.sh` | Fetches and verifies the pinned upstream commit |
| `scripts/build_openmmlab_compat_wheels.sh` | Builds the pure-Python mmcv wheel |
| `scripts/train_1p_stage1_smoke.sh` | Single-card stage-1 training launch check |
| `scripts/train_step_musa.py` | Runs the training iteration behind the launch check |
| `scripts/infer_frame_musa.py` | Single-frame inference fixture |
| `fixture_validation.md` | Acceptance contract for both fixtures |

# Preparing the environment

## Install dependencies

**Table 1** Third-party version support

| Package | Supported version |
| --- | --- |
| Python | `3.10` |
| mmcv | `1.7.0` (Python-only build) |
| mmdet | `2.28.0` |
| mmseg | `0.30.0` |
| mmdet3d | `1.0.0rc4` |
| numpy | `1.26.4` |
| shapely | `2.0.x` |

Upstream documents Python 3.8 with torch 1.9.1+cu111, mmcv-full 1.4.0, mmdet 2.14.0 and
mmdetection3d from master. Those pins do not install on this stack, so the recipe targets the
same OpenMMLab compatibility set as the UniAD recipe: mmcv 1.7.0 built without compiled ops,
mmdet 2.28.0, mmseg 0.30.0 and the mmdet3d 1.0.0rc4 checkout. The compatibility patches close
the API gaps that the version bump opens.

The Python-only mmcv distribution ships no `mmcv/_ext` module, and importing `mmcv.ops`
aborts on the first `ext_loader.load_ext` call. Build the patched wheel, which replaces the
loader with a placeholder that raises only when an operator is actually called:

```bash
model_examples/VAD/scripts/build_openmmlab_compat_wheels.sh
pip install --no-deps wheelhouse/mmcv-1.7.0-*.whl
```

Then install mmdetection3d from its pinned source checkout and the Python 3.10 overrides:

```bash
pip install -e mmdetection3d --no-deps
pip install -r model_examples/VAD/requirements.lock
```

`scikit-image` is imported lazily by the planning metrics (see
`patches/vad-01-lazy-metrics-import.patch`), so it is only required for the planner
evaluation, not for the launch checks.

## Install the MUSA stack

**Table 2** MUSA software support

| Component | Supported version |
| --- | --- |
| PyTorch | `2.9.1` / `2.11.0` |
| torch_musa | matching PyTorch version |
| MUSA Toolkit | `5.2.0` |
| Device | MTT S5000, `mp_31` |

Install muSPACE with its native extension, then run `mu-space env --strict` from the
repository root to confirm the active environment is reported as supported.

## Prepare the checkout

From the muSPACE repository root:

```bash
model_examples/VAD/scripts/prepare_checkout.sh ./VAD
mu-space migrate vad --checkout ./VAD
```

`prepare_checkout.sh` only fetches and verifies the pinned commit; it exits non-zero if the
origin or the resulting `HEAD` does not match. `mu-space migrate` applies the patch recipe and
writes `.mu_space_vad.json` in the checkout. Re-running is safe: an already migrated checkout
reports `already-applied`, a partially patched one reports `upgraded`, and a patch that no
longer applies fails loudly instead of being skipped. `mu-space restore vad --checkout ./VAD`
reverses the recipe.

The patches cover the Python 3.10 gaps in this path: the lazy `scikit-image` import in the
planning metrics, a `points_in_boxes_gpu` fallback backed by
`mu_space.ops.points_in_boxes_all` for the compatibility mmdet3d build, and device placement in
GridMask. Upstream implementations are preserved.

## Prepare the dataset

Obtain nuScenes from [nuscenes.org](https://www.nuscenes.org/nuscenes), including the CAN bus
expansion and the map expansion, then point the data root at it. VAD uses the temporal metadata
that its own converter produces:

```bash
cd VAD
python tools/data_converter/vad_nuscenes_converter.py nuscenes \
    --root-path /path/to/nuscenes \
    --canbus /path/to/nuscenes \
    --version v1.0 \
    --out-dir /path/to/nuscenes \
    --extra-tag vad_nuscenes
```

The converter writes `vad_nuscenes_infos_temporal_train.pkl` and
`vad_nuscenes_infos_temporal_val.pkl` into the output directory. A data root that satisfies the
configs contains the nuScenes tree plus those two files:

```
nuscenes
├── can_bus
├── maps
├── samples
├── sweeps
├── v1.0-trainval
├── vad_nuscenes_infos_temporal_train.pkl
└── vad_nuscenes_infos_temporal_val.pkl
```

The launch checks select one scene from the infos at runtime, so the sensor files only need to
cover that scene while the full version metadata (`<version>/scene.json`) and the CAN bus files
are required. Passing `--version v1.0-mini` produces a small metadata set for a quick check.

## Prepare pretrained weights

The launch checks build the model without loading a checkpoint so they can run offline. Full
training uses the upstream ResNet-50 backbone initialisation and, for the stage-2 configs, the
stage-1 checkpoint.

# Quick start

## Training

Run one stage-1 (detection and map) training iteration on a single card:

```bash
bash model_examples/VAD/scripts/train_1p_stage1_smoke.sh --checkout ./VAD --data-root /path/to/nuscenes
```

The launcher defaults to `projects/configs/VAD/VAD_base_stage_1.py`, the first scene present in
the infos, one iteration, device `musa:0`, and `<checkout>/data/nuscenes` as the data root. Each
default is overridable by flag (`--config`, `--scene-name`, `--max-iters`, `--device`,
`--data-root`, `--work-dir`) or by the matching environment variable. Scratch metadata is
written below the system temporary location by default; pass `--work-dir` explicitly when a
controlled location is required (for repository-local validation, use `<repository>/.tmp`).

A successful run prints one line:

```
vad_train_step_ok optimizer=AdamW param_groups=<N> loss=<finite> grad_norm=<finite> param_delta=<positive> logs={...}
```

The launcher applies the muSPACE patch in strict mode after importing the plugin modules, so
every MSDA call site — the helper function and VAD's own `MultiScaleDeformableAttnFunction_fp32`
/ `_fp16` entries — is rebound to `mu_space.ops.multi_scale_deformable_attn` before the model is
built. Strict mode makes a fallback to the PyTorch reference path fail instead of passing
silently. The optimizer is built from the config through OpenMMLab's `build_optimizer`.

The launcher calls `model.init_weights()` after building the model, as the upstream training
entry point does. Upstream initialises the transformer with `xavier_uniform`; skipping that call
leaves PyTorch's default initialisation on the attention projections, and the BEV encoder
activations then grow by roughly an order of magnitude per layer until they overflow float32.

Pass `--pretrained-backbone` to initialise the ResNet backbone from the ImageNet weights that
`torchvision://resnet50` resolves to. The flag is off by default so the launch check can run
without network access; it downloads the checkpoint on first use.

## Inference

Run one single-frame inference forward:

```bash
python model_examples/VAD/scripts/infer_frame_musa.py --checkout ./VAD --data-root /path/to/nuscenes \
    --device musa:0
```

The fixture builds the test dataset, takes one prepared frame, runs the image backbone and the
VAD head under `torch.no_grad()`, and checks that the prediction tensors are finite with the
expected shapes. A successful run prints:

```
vad_infer_ok device=musa:0 scene=<name> report={...}
```

`report` covers detection (`all_cls_scores`, `all_bbox_preds`), map (`map_all_cls_scores`,
`map_all_pts_preds`) and agent trajectory (`all_traj_preds`) tensors. The ego-conditioned
planning branch (`ego_fut_preds`) is fed by the stage-2/e2e planner inputs and is outside this
fixture's scope.

# Results

No muSPACE accuracy or throughput numbers are published for VAD. The scripts above are launch
and correctness checks that prove the training and inference paths complete on MUSA with finite
tensors and the native deformable-attention operator; they are not benchmarks and must not be
quoted as performance data. They are also not a substitute for full training, which needs all
trainval temporal infos, the stage-specific checkpoints, and the upstream training schedule.

Alignment with the published VAD metrics is not claimed. Out of scope for this recipe: strict
reproduction of the paper's numbers, multi-card scaling, the CARLA closed-loop evaluation,
`aidisdk` integration, VADv2, and full training convergence.

# Changelog

2026-11-09: First release of the VAD recipe — pinned checkout, three compatibility patches, the
pure-Python mmcv extension patch, deformable-attention runtime integration, and the single-card
training and inference fixtures.

# FAQ

1. **`prepare_checkout.sh` exits with `unexpected origin`.** The destination directory is an
   unrelated git repository. Remove it or pass a fresh path.
2. **`mu-space migrate` reports `conflict`.** The checkout is not at the pinned commit or has
   local edits that overlap the recipe. Restore it to a clean pinned state and retry.
3. **`does not match its manifest digest`.** A patch file was modified after the manifest was
   written. Restore the patch from git rather than updating the digest.
4. **`ModuleNotFoundError` for `mmdet3d.ops.roiaware_pool3d`.** The compatibility mmdet3d build
   ships without that extension; `patches/vad-02-points-in-boxes-fallback.patch` routes the call
   through `mu_space.ops.points_in_boxes_all` instead.
5. **`mmcv._ext... is unavailable`.** The environment still has the compiled mmcv distribution or
   an unpatched Python-only build. Install the wheel produced by
   `scripts/build_openmmlab_compat_wheels.sh`.
6. **The fixtures report non-finite detection tensors.** The model was built without
   `model.init_weights()`. Upstream relies on that call to apply `xavier_uniform` to the
   transformer; without it the BEV encoder activations overflow float32 within one forward pass.
7. **`--pretrained-backbone` fails to download.** The flag needs access to the torchvision
   checkpoint host. Omit it to train from scratch.
