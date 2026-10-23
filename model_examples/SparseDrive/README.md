# SparseDrive for muSPACE

## Contents

- [Overview](#overview)
  - [Model description](#model-description)
  - [Code implementation](#code-implementation)
  - [Operator provenance](#operator-provenance)
- [Preparing the environment](#preparing-the-environment)
  - [Install dependencies](#install-dependencies)
  - [Install the MUSA stack](#install-the-musa-stack)
  - [Prepare the checkout](#prepare-the-checkout)
  - [Prepare the dataset](#prepare-the-dataset)
  - [Prepare pretrained weights](#prepare-pretrained-weights)
- [Quick start](#quick-start)
  - [Connect the muSPACE operator](#connect-the-muspace-operator)
  - [Training](#training)
  - [Validation](#validation)
  - [Results](#results)
- [Known limitations](#known-limitations)
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
  commit_id=fbadea693cbef3f3daea7705e521c0dd3321605c
  ```

- muSPACE recipe:

  ```shell
  code_path=model_examples/SparseDrive
  ```

muSPACE maintains no model fork. Unlike the other recipes, this one ships **no checkout
patches**: the integration is purely a runtime rebinding, so `manifest.yaml` declares the
checkout contract with an empty `patches:` list.

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Upstream repository, pinned commit, and patch marker |
| `requirements.lock` | Dependency stack for the pinned OpenMMLab 1.0.0rc4 line |
| `scripts/prepare_checkout.sh` | Fetches and verifies the pinned upstream commit |
| `scripts/prepare_dataset.sh` | Generates the training info files with the checkout's converter |
| `scripts/prepare_anchors.sh` | Generates the detection, map, motion, and planning anchors |
| `scripts/train_step_musa.py` | Single-scene training step with strict muSPACE dispatch |
| `scripts/train_1p_sparsedrive_smoke.sh` | Single-card launch check around `train_step_musa.py` |
| `scripts/validate_fixture.sh` | Validates the launch-check result and native dispatch |
| `scripts/precision_compare.py` | Compares two step artifacts key by key |
| `fixture_validation.md` | Fixture prerequisites, acceptance criteria, and troubleshooting |

## Operator provenance

The pinned commit is authoritative for the operator lineage. Its
`projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu` baseline device functions
— `bilinear_sampling`, `bilinear_sampling_grad`, `deformable_aggregation_kernel`, and
`deformable_aggregation_grad_kernel` — are token-identical to the corresponding kernels in
`csrc/deformable_aggregation/deformable_aggregation_musa.mu`, so the file mapping recorded
in [docs/provenance.md](../../docs/provenance.md) is confirmed at this revision.

The optimized `*_opt` kernels in the same muSPACE source are not part of the public
upstream repository: a walk of its complete history finds no `forward_v1`, `backward_v3`,
or FastDivmod variant of this operator. They come from an internal MUSA port of the same
operator, and
[Deformable Aggregation (SparseDrive)](../../docs/provenance.md#deformable-aggregation-sparsedrive)
records the split lineage.

# Preparing the environment

## Install dependencies

**Table 1** Third-party version support

| Package | Supported version |
| --- | --- |
| Python | `3.10` |
| mmcv | `1.7.0` |
| mmdet | `2.28.0` |
| mmdet3d | `1.0.0rc4` |

Install the platform-specific PyTorch build first, then the stack in `requirements.lock`:

```bash
python -m pip install -r model_examples/SparseDrive/requirements.lock
```

`requirements.lock` follows the same OpenMMLab 1.0.0rc4 line as the BEVDepth and FlashOCC
recipes, with the upstream pins moved to releases that still publish Python 3.10 wheels.
It deliberately omits `flash-attn`; see [Known limitations](#known-limitations).

## Install the MUSA stack

**Table 2** MUSA software support

| Platform | MUSA SDK / Toolkit | PyTorch | torch_musa | Architecture |
| --- | --- | --- | --- | --- |
| Linux x86_64 / MTT S5000 | `4.3.8` | `2.7.1` | `2.7.1` | `MUSA_ARCH=31` |
| Linux x86_64 / MTT S5000 | `5.2.0` | `2.9.1` | `2.9.1` | `MUSA_ARCH=31` |
| Linux x86_64 / MTT S5000 | `5.2.0` | `2.11.0` | `2.11.0` | `MUSA_ARCH=31` |

Python `3.10` is required for every row. Install muSPACE with its native extension, then
run `mu-space env` from the repository root to confirm the active environment is reported
as supported.

## Prepare the checkout

From the muSPACE repository root:

```bash
model_examples/SparseDrive/scripts/prepare_checkout.sh ./SparseDrive
mu-space migrate sparsedrive --checkout ./SparseDrive
```

`prepare_checkout.sh` only fetches and verifies the pinned commit; it exits non-zero if
the origin or the resulting `HEAD` does not match. `mu-space migrate` verifies the commit
and writes `.mu_space_sparsedrive.json`; because the recipe has no patches, it leaves the
checkout contents untouched. The checkout is only needed for the model code itself — the
operator is rebound in the running process.

Do not follow the upstream instruction to run `setup.py` under
`projects/mmdet3d_plugin/ops`. That build produces the CUDA extension
`deformable_aggregation_ext`, which has no MUSA build and is never called under this
recipe, because `sparsedrive_patcher` replaces the Python binding that would reach it. The
patcher also registers a placeholder for the missing extension so the ops package imports
without it, and raises a descriptive error if anything still calls into the extension
directly.

## Prepare the dataset

Obtain nuScenes from [nuscenes.org](https://www.nuscenes.org/nuscenes), including the
trainval metadata, `samples`, `sweeps`, `maps`, and `can_bus`, and link the extracted tree
into the checkout:

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

Generate the training info files with the checkout's own converter:

```bash
model_examples/SparseDrive/scripts/prepare_dataset.sh \
  --checkout ./SparseDrive \
  --data-root ./SparseDrive/data/nuscenes
```

Then generate the anchors the model configuration loads:

```bash
model_examples/SparseDrive/scripts/prepare_anchors.sh --checkout ./SparseDrive
```

`prepare_dataset.sh` wraps `tools/data_converter/nuscenes_converter.py`, writes
`data/infos/nuscenes_infos_train.pkl` and `data/infos/nuscenes_infos_val.pkl`, and checks
that both files exist; use `--out-dir` to write them elsewhere and `--version v1.0-mini`
for a quick sanity pass below `data/infos/mini`. `prepare_anchors.sh` runs the checkout's
four K-means tools from the checkout root — they hardcode the relative `data/infos` and
`data/kmeans` paths — and checks that `kmeans_det_900.npy`, `kmeans_map_100.npy`,
`kmeans_motion_6.npy`, and `kmeans_plan_6.npy` were written. Both outputs are required
before a training run.

The converter runs the checkout's own `tools/data_converter/nuscenes_converter.py`, and that
module imports `projects.mmdet3d_plugin`, whose package `__init__` pulls in the model
package: converting data therefore needs `flash_attn` importable as well, and the operator
module it binds expects the `deformable_aggregation_ext` extension to exist, so that
extension has to be built or stubbed in whatever environment runs the converter.

A mini split exercises the converter but not the whole anchor set: `kmeans_plan.py` clusters
ego trajectories per navigation command and needs frames for all three commands, and nuScenes
mini contains only two of them, so `kmeans_plan_6.npy` requires the trainval info file.
`prepare_anchors.sh` reports that reason when the clustering run fails instead of stopping at
the clustering traceback.

With `v1.0-trainval` the split is complete: the converter writes 28130 train and 6019 val
info records, the navigation-command buckets are `{0: 1940, 1: 1527, 2: 24663}` and
`{0: 410, 1: 303, 2: 5306}`, and all four anchor files are written (clustering takes about
two and a half minutes on 28130 records). Note that `--version v1.0-mini` writes below
`<out-dir>/mini`, so pass the same `--out-dir` for both versions; and keep
`data/infos/nuscenes_infos_train.pkl` a regular file, because the converter opens the target
path for writing and would follow a symlink left there by a mini run.

## Prepare pretrained weights

Full training uses an ImageNet ResNet-50 backbone, and stage 2 uses the stage-1
checkpoint. Both come from upstream; place them under `SparseDrive/ckpt/`. The launch
check below does not need them: it clears the backbone initialization so the model can be
built without a downloaded checkpoint.

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
the already-imported `DAF` binding in `projects.mmdet3d_plugin.models.blocks`. It also
reinstate the import probes the checkout caches at module scope (`DAF_VALID` and the
`feature_maps_format` name), and, on a MUSA-only process, route the hardcoded
`Tensor.cuda()` calls in the grid mask and the detection head to the current MUSA device.
`sparsedrive_patcher.restore()` reverses all of it.

The native path requires MUSA tensors: `float32` features, sampling locations, and
weights; `int32` shapes and indices. `num_groups` must be in `[1, 8]` and must divide
`num_embeds`. The helper `feature_maps_format` is exported from `mu_space.ops` for model
glue code.

## Training

Run one training iteration on a single scene as a launch check:

```bash
bash model_examples/SparseDrive/scripts/train_1p_sparsedrive_smoke.sh \
  --checkout ./SparseDrive \
  --data-root ./SparseDrive/data/nuscenes \
  --infos ./SparseDrive/data/infos/nuscenes_infos_train.pkl
```

The wrapper forwards to `train_step_musa.py`, which applies `sparsedrive_patcher` in
strict mode before the model package is imported, selects one scene from the prepared info
file, and runs the forward, loss, backward, and optimizer step. It prints a single JSON
line with this shape:

```text
{"device": "<musa device>", "scene": "<scene>", "seed": <int>, "config": "<path>",
 "upstream_revision": "<commit>", "infos_sha256": "<digest>", "torch_version": "<version>",
 "max_iters": <int>, "loss": <float>, "log_vars": {<loss name>: <float>, ...},
 "selected_parameter": "<name>", "selected_parameter_before_sha256": "<digest>",
 "selected_parameter_after_sha256": "<digest>", "finite_gradients": true,
 "optimizer_step": true, "dispatch": [<one entry per distinct backend decision>],
 "dispatch_calls": <int>}
```

Each `dispatch` entry reports `operator`, `requested_backend`, `selected_backend`,
`native_available`, and `native_fallback`, so the line shows whether the run really used
the native MUSA kernel. Pass `--result-artifact PATH` to also save the loss and every
parameter gradient for `precision_compare.py`.

## Validation

`validate_fixture.sh` runs the same step and checks the result JSON; see
[fixture_validation.md](fixture_validation.md) for the prerequisites, the acceptance
criteria, and troubleshooting.

## Results

One training iteration on the full `v1.0-trainval` split completes on MUSA: 12
deformable-aggregation dispatches on the MUSA backend, attention on MATE, finite gradients,
and a real optimizer step on `scene-0001`. It is a launch check, not a benchmark — the
per-iteration time must not be quoted as performance data, and a single step is not a
convergence result.

The deformable aggregation operator is aligned against the checkout's own CUDA extension on
an NVIDIA A100 80GB. Both sides consumed bit-identical inputs (one numpy seed, and the
sha256 digests of all four input tensors agree) at the production feature and point layout
with batch 2, and the outputs are compared elementwise:

| tensor | elements | max abs | RMAE | outside 1e-4 + 1e-4 x abs(ref) |
| --- | ---: | ---: | ---: | ---: |
| forward output | 460800 | 2.61e-08 | 1.86e-07 | 0 |
| grad wrt feature maps | 45957120 | 1.12e-08 | 9.16e-08 | 0 |
| grad wrt sampling locations | 280800 | 6.87e-05 | 7.58e-07 | 0 |
| grad wrt weights | 4492800 | 3.81e-06 | 8.88e-08 | 0 |

Every tensor matches to float32 rounding. The largest relative deviations (0.23 on
`grad_weights`) sit on elements whose reference value is close to zero, which is why RMAE
and the absolute figures carry the verdict rather than a max-relative column. The A100 side
is the pinned checkout's `deformable_aggregation_ext`, built with nvcc 12.6 for the
container's torch; the MUSA side is `deformable_aggregation_function(..., backend="musa")`.
Throughput is published separately in the
[deformable aggregation benchmark](../../benchmarks/deformable_aggregation/README.md).

# Known limitations

- **`flash_attn` is a compute dependency, not only an import.** The pinned
  `projects/mmdet3d_plugin/models/attention.py` imports `flash_attn` at module scope, and
  `FlashAttention.forward` calls `flash_attn_unpadded_kvpacked_func` — the CUDA
  variable-length kernel — after asserting `q.is_cuda`; `FlashMHA` is built with
  `dtype=torch.float16`. An importable build is therefore necessary but not sufficient: the
  kernels have to run on the target device. With MATE installed the patcher bridges this
  block onto `mate.flash_attn_varlen_func`: it registers a `flash_attn` package backed by
  MATE before the checkout imports it, and rebinds `FlashAttention.forward`, which drops the
  `is_cuda` assertion and keeps the upstream signature, return shape, and padded-key path.
  Without MATE nothing is registered, upstream `flash-attn` stays CUDA-only, and the launch
  check reports the missing dependency instead of producing model numbers. MATE is an
  optional runtime dependency: the operator replacement works either way.
- **MATE's attention backward needs `tilelang-musa`.** MATE implements the variable-length
  attention forward natively but compiles its backward with TileLang, so `loss.backward()`
  imports `tilelang` and, for MATE 0.2.7, requires `tilelang-musa==0.1.12` —
  `pip install "mate[all]"` pulls that pin in. That build links
  `libz3.so.4.15` from `z3-solver==4.15.0.0`, which has to be reachable through
  `LD_LIBRARY_PATH` (the wheel does not set an rpath), and the first backward JIT-compiles
  the kernels, which takes about half a minute. A forward-only run does not need any of it.
- **The launch check needs the fully prepared dataset.** `train_step_musa.py` consumes the
  info file produced by `prepare_dataset.sh` and the anchors produced by
  `prepare_anchors.sh`, and the model configuration loads the anchors by relative path.
  The script's argument contract and dispatch assertions are covered by the test suite; no
  full-run log is published with the recipe.
- **No model-level performance numbers.** Only the operator-level benchmark linked above
  is published.

# Changelog

2026-10-19: Exercised the data preparation scripts against a real nuScenes mini split:
`prepare_dataset.sh` produced the train/val info files from 404 samples, and
`prepare_anchors.sh` produced the detection, map, and motion anchors. `--infos` now resolves
relative paths against the checkout like `train_step_musa.py`, and a failing plan clustering
run is reported with its command-bucket requirement instead of the raw traceback. The
`flash_attn` limitation now states that the attention blocks call the variable-length CUDA
kernel and assert a CUDA device, so an importable build alone does not make the model runnable
on MUSA.

2026-10-19: Moved the pin to `fbadea6`. Its only operator change since `52c4c05b` is upstream's
large-batch indexing fix (`4958e1e`), and the MUSA kernels now carry the same 64-bit launch
indexing, so the baseline lineage statement holds against `52c4c05b`.

2026-10-19: Ran the recipe end to end on the full `v1.0-trainval` split. Conversion produced
28130 train and 6019 val info records with all three navigation-command buckets populated, the
four anchor files were clustered, and one training iteration on `scene-0001` went through the
whole path: 12 deformable-aggregation dispatches on the MUSA backend, attention on MATE with
the TileLang-compiled backward, finite gradients, and a real optimizer step
(`img_backbone.conv1.weight` changes from `31b6ae9b` to `a109fb6f`). Loss was 2050.66 with
`det_loss_cls_5 = 355.24` and `map_loss_cls_2 = 8.05`; the run takes 1m43s including model
construction.

2026-10-19: Bridged the attention blocks onto MATE. When MATE is importable the patcher
registers a `flash_attn` package backed by `mate.flash_attn_varlen_func` before the checkout
imports it, and rebinds `FlashAttention.forward`, which otherwise asserts `q.is_cuda`. Verified
on S5000 at SparseDrive's attention geometry (1x900x8x32, fp16): 2.0e-4 against a float64
reference, with `FlashMHA`, `MultiheadFlashAttention`, and the padded-key path running end to
end. MATE stays optional; without it the previous limitation stands.

2026-10-12: Completed the support workflow — verified the pinned commit against the
operator lineage, added `requirements.lock`, the dataset preparation scripts, the
single-card training launch check with fixture validation, and the artifact comparison
script, and recorded the `*_opt` kernel attribution and the `flash_attn` prerequisite.

2026-08-13: Recipe restructured to the shared layout — added the checkout manifest and
`prepare_checkout.sh` with a candidate pin, and recorded the unverified-pin gap.

# FAQ

1. **Is the pinned commit trustworthy?** Yes. The baseline kernels are token-identical to the
   muSPACE implementation at the revision they were carried over from, and the pinned commit
   only widens the launch indexing since then; see
   [Operator provenance](#operator-provenance).
2. **Why are there no patches?** The integration is a runtime rebinding, so nothing in the
   checkout needs to change. `mu-space migrate sparsedrive` therefore only verifies the
   commit and writes the marker.
3. **Do I have to build `projects/mmdet3d_plugin/ops`?** No. That build produces a CUDA
   extension the recipe never calls; the patcher replaces the Python binding and tolerates
   the missing extension.
4. **`strict=True` raises instead of falling back.** That is intended: strict mode is how
   you assert the native MUSA kernel is in use. Drop `strict` to allow the reference path.
5. **`num_groups` errors.** The upstream constraint is carried over — `num_groups` must be
   in `[1, 8]` and divide `num_embeds`.
6. **`ModuleNotFoundError: flash_attn`.** The pinned upstream imports `flash_attn`
   unconditionally. Install MATE to let the patcher supply a MUSA-backed `flash_attn`, or
   provide an importable build for the target platform; see
   [Known limitations](#known-limitations).
7. **Why are no accuracy or throughput numbers published?** No validated end-to-end
   measurements exist for this recipe yet, and the launch check is not a benchmark.
