# Sparse4D for muSPACE

## Contents

- [Overview](#overview)
- [Preparing the environment](#preparing-the-environment)
- [Quick start](#quick-start)
- [Known limitations](#known-limitations)
- [Changelog](#changelog)
- [FAQ](#faq)

## Overview

### Model description

Sparse4D v3 is a sparse, temporal multi-view 3D detection and tracking model. Its
perception stack builds sparse instance features, projects 3D key points into the
multi-view feature maps, and fuses the sampled features with
`deformable_aggregation`.

### Code implementation

| Item | Value |
| --- | --- |
| Upstream repository | `https://github.com/HorizonRobotics/Sparse4D.git` |
| Pinned revision | `4ec96edf0e7d9e80fbd75ef22e1d51a937d3a750` (`v3.0`) |
| Recipe path | `model_examples/Sparse4D` |
| Integration | runtime patcher, no checkout patches, no model fork |

The recipe keeps the official checkout unmodified. `mu-space migrate sparse4d`
only validates the pinned revision and writes a marker file; the operator binding is
replaced at runtime.

### Operator provenance

Sparse4D v3.0's `projects/mmdet3d_plugin/ops` exposes the same
`deformable_aggregation_function` and `feature_maps_format` bindings as SparseDrive.
The four upstream device functions are the same lineage recorded in
[`docs/provenance.md`](../../docs/provenance.md). muSPACE reuses
`mu_space.ops.deformable_aggregation`; this recipe adds no operator.

Sparse4D's plugin package imports `mmcv` and `mmdet` but not `mmdet3d`. The
mmdet3d 1.0.0rc4 / 1.0.0rc6 compatibility question therefore does not apply to
this recipe: the fixture builds the model and dataset through `mmdet`, as the
upstream tools do.

### Recipe files

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Pinned repository, revision, patcher, marker |
| `requirements.lock` | OpenMMLab-side dependency pins |
| `scripts/prepare_checkout.sh` | Fetch and verify the pinned revision |
| `scripts/prepare_dataset.sh` | Generate nuScenes info files |
| `scripts/prepare_anchors.sh` | Generate the K-means anchor file |
| `scripts/run_with_patcher.py` | Run an upstream script with the patcher installed |
| `scripts/train_step_musa.py` | Single-card single-step fixture |
| `scripts/validate_fixture.sh` | Strict fixture and dispatch validation |
| `fixture_validation.md` | Fixture acceptance criteria and troubleshooting |

## Preparing the environment

### Install dependencies

Install the platform PyTorch and torch_musa build separately, then install the
OpenMMLab-side pins:

```bash
python -m pip install -r model_examples/Sparse4D/requirements.lock
```

| Component | Version |
| --- | --- |
| Python | 3.10 |
| MUSA SDK | 5.2.0 |
| torch / torch_musa | 2.9.1 / 2.9.1 |
| MUSA architecture | `mp_31` |
| mmcv | 1.7.0 (pure Python) |
| mmdet | 2.28.2 |
| nuscenes-devkit | 1.1.10 |

`requirements.lock` deliberately replaces upstream `mmcv_full` 1.7.1 with the
pure-Python `mmcv` 1.7.0. `sparse4d_patcher` installs the `mmcv._ext` guard that
`mmdet` needs during import and releases it on `restore()`. The lock also upgrades
`motmetrics` to a release that imports on Python 3.10 and keeps `ipython`
importable because an upstream sampler imports it.

### Install the MUSA stack

Build and install the muSPACE wheel for the target architecture as described in
[`docs/install.md`](../../docs/install.md). The fixture requires the native
`deformable_aggregation` exports.

### Prepare the checkout

```bash
model_examples/Sparse4D/scripts/prepare_checkout.sh ./Sparse4D
```

The script initializes the checkout, checks the origin, fetches the pinned commit
with `--depth 1`, detaches at that commit, and verifies both the commit and the
required upstream files. It never patches the checkout.

`mu-space migrate sparse4d --checkout ./Sparse4D` writes
`.mu_space_sparse4d.json`. `mu-space restore sparse4d --checkout ./Sparse4D`
removes it and leaves the tracked tree unchanged.

### Prepare the dataset

Point the converter at a nuScenes root that contains the metadata, samples,
sweeps, and maps trees:

```bash
model_examples/Sparse4D/scripts/prepare_dataset.sh \
  --checkout ./Sparse4D \
  --data-root /path/to/nuscenes
```

The default output is `<checkout>/data/infos/nuscenes_infos_train.pkl` plus the
validation info file. Pass `--version v1.0-mini` for a small sanity split.

### Prepare the anchors

```bash
model_examples/Sparse4D/scripts/prepare_anchors.sh \
  --checkout ./Sparse4D \
  --infos ./Sparse4D/data/infos/nuscenes_infos_train.pkl
```

The default output is `<checkout>/data/anchors/nuscenes_kmeans900.npy`. The
released configs load `nuscenes_kmeans900.npy` by name; the script passes the
output name explicitly because the upstream tool's default has a leading underscore.
`tools/anchor_generator.py` imports the plugin package, so the script runs it
through `run_with_patcher.py` by default.

With `v1.0-mini`, K-means needs at least `--num-anchor` ground-truth boxes. Reduce
`--num-anchor` for a mini sanity run, or generate the anchors from the trainval
info file. The fixture lowers `num_temp_instances` to the anchor count when the
anchor file is smaller than the released 900.

### Prepare pretrained weights

The fixture does not load pretrained weights. For a longer training run, place the
upstream ResNet-50 checkpoint at the path referenced by the config and remove the
fixture's `init_cfg` override in a copy of the script.

## Quick start

### Connect the muSPACE operator

Call the patcher before importing the model package:

```python
from mu_space.patcher import sparse4d_patcher

sparse4d_patcher.apply(strict=True)
try:
    import projects.mmdet3d_plugin  # noqa: F401
    # Build the model and run training here.
finally:
    sparse4d_patcher.restore()
```

The patcher rebinds `ops.deformable_aggregation_function`, the cached
`models.blocks.DAF` binding, the `DAF_VALID` probe and `feature_maps_format`. It
registers a placeholder for the unbuilt `deformable_aggregation_ext`, routes
hardcoded `.cuda()` calls to MUSA when CUDA is unavailable, and installs the
`mmcv._ext` guard needed by `mmdet`. Sparse4D has no attention module, so the
MATE/flash-attn bridge used by the SparseDrive recipe is skipped.

### Training

```bash
python model_examples/Sparse4D/scripts/train_step_musa.py \
  --mode fixture \
  --checkout ./Sparse4D \
  --data-root /path/to/nuscenes \
  --infos ./Sparse4D/data/infos/nuscenes_infos_train.pkl \
  --anchors ./Sparse4D/data/anchors/nuscenes_kmeans900.npy \
  --device musa:0 \
  --max-iters 1
```

The script subsets the info file to one scene, builds the released
`sparse4dv3_temporal_r50_1x8_bs6_256x704` config with batch size 1, clears the
backbone pretrained initializer, points the instance bank at the generated anchors,
and runs forward, loss, backward, and one optimizer step. It prints one JSON line.

### Validation

```bash
model_examples/Sparse4D/scripts/validate_fixture.sh --strict \
  --checkout ./Sparse4D \
  --data-root /path/to/nuscenes \
  --infos ./Sparse4D/data/infos/nuscenes_infos_train.pkl \
  --anchors ./Sparse4D/data/anchors/nuscenes_kmeans900.npy
```

See [`fixture_validation.md`](fixture_validation.md) for the accepted JSON fields.

### Results

This recipe publishes no accuracy or performance number. The fixture is a launch
and dispatch check, not a benchmark.

The fixture was exercised on the pinned checkout with the nuScenes mini split
(`scene-0061`): one step completed with a finite loss, finite gradients, an
optimizer step, and six native `deformable_aggregation` dispatches on a single
MUSA device.

## Known limitations

- The fixture is a single-card, single-step smoke test. It does not validate
  convergence, learning-rate schedules, or multi-card behavior.
- The fixture covers one nuScenes scene. It is not full-dataset training.
- No pretrained weights are loaded, so loss values are not accuracy results.
- The temporal instance cache is not exercised beyond the first step.
- Denoising training is disabled in the fixture; the released training config keeps
  it enabled.
- No model-level performance or memory number is published.
- Only the MUSA 5.2.0 + torch/torch_musa 2.11.0 stack is validated. Other
  supported MUSA stacks may work but are untested.
- `mmdet3d` is not installed by this recipe. The upstream plugin does not import
  it; downstream tooling that requires `mmdet3d` must provide its own compatible
  stack.

## Changelog

- Initial Sparse4D v3.0 runtime-only recipe with a real nuScenes single-scene
  fixture and strict dispatch validation.

## FAQ

### Does this modify the Sparse4D checkout?

No. `prepare_checkout.sh` only fetches and verifies. `mu-space migrate` writes a
marker file because the recipe has no patches. `mu-space restore` removes it.

### Why is there no `mmdet3d` pin?

Sparse4D v3.0's `projects/mmdet3d_plugin` imports `mmcv` and `mmdet` only. The
dataset and detector builders come from `mmdet`, as in the upstream tools. The
repository's existing `mmdet3d` recipes are independent of this one.

### Why does the recipe use `mmcv` instead of `mmcv_full`?

The only compiled mmcv operator this model would use is replaced by muSPACE.
`mmdet` still imports `mmcv.ops`, so `sparse4d_patcher` installs the legacy
extension guard for the duration of the patched run.

### Why does `prepare_anchors.sh` use the runtime patcher?

`tools/anchor_generator.py` imports `projects.mmdet3d_plugin.core.box3d`, which
initializes the plugin package. The patcher lets that import succeed without the
unbuilt `deformable_aggregation_ext`.

### Why is `ipython` in the lock file?

An upstream sampler imports `IPython.embed` for debugging. The fixture never calls
it, but the import must succeed.

### Why does the fixture use a single scene?

The goal is to prove that the patched operator runs through the model graph and
that the optimizer step is finite. A single scene keeps the check reproducible and
short; it is not an accuracy or convergence test.
