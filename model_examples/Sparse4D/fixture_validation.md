# Sparse4D Fixture Validation

The recipe runs a deterministic single-card Sparse4D training step on one nuScenes
scene. It validates model launch and native dispatch; it is not full-dataset
training, an accuracy benchmark, or a performance benchmark.

## What the check uses

- The pinned Sparse4D v3.0 checkout (`4ec96edf0e7d9e80fbd75ef22e1d51a937d3a750`).
- The released `sparse4dv3_temporal_r50_1x8_bs6_256x704` config with batch size 1,
  no backbone pretrained initializer, and the generated K-means anchors.
- A single-scene subset of the nuScenes training info file produced by
  `prepare_dataset.sh`, with relative image and lidar paths resolved against the
  data root.
- `sparse4d_patcher.apply(strict=True)` before the model package is imported.

## Prerequisites

Prepare the pinned checkout, dependencies, info files, and anchors as described in
[README.md](README.md). Install muSPACE with the native `deformable_aggregation`
symbols and run inside an environment with an available MUSA device.

The checkout must contain `projects/mmdet3d_plugin`, `tools/nuscenes_converter.py`,
and `tools/anchor_generator.py`. The info file must contain the requested scene and
the anchor file must be non-empty.

## One-command check

```bash
model_examples/Sparse4D/scripts/validate_fixture.sh --strict \
  --checkout ./Sparse4D \
  --data-root <PREPARED_NUSCENES_ROOT> \
  --infos ./Sparse4D/data/infos/nuscenes_infos_train.pkl \
  --anchors ./Sparse4D/data/anchors/nuscenes_kmeans900.npy
```

`--scene-name`, `--seed`, and `--max-iters` change the fixture. `--device` must be
a MUSA device because the check asserts native MUSA dispatch. Mini splits use
different scene names; pass one explicitly, such as `--scene-name scene-0061`.
`--no-strict` is a debugging aid and is not accepted as release evidence.

## Acceptance criteria

Validation passes only when the result JSON reports:

- a finite `loss`;
- `finite_gradients: true`;
- `optimizer_step: true`;
- different `selected_parameter_before_sha256` and
  `selected_parameter_after_sha256`;
- `dispatch_calls >= 1`;
- a dispatch operator set equal to `{"deformable_aggregation"}`;
- with `--strict`, `upstream_revision` equal to the pinned commit and every
  dispatch entry with `selected_backend: "musa"`, `native_available: true`, and no
  fallback.

The pinned upstream revision is checked before model construction. A mismatched
checkout, a missing info or anchor file, an unavailable MUSA device, a non-finite
update, or a non-native dispatch fails the command.

## What the check does not prove

- Convergence or learning-rate schedule behavior.
- Full-dataset or multi-card training.
- Accuracy against the paper or an NVIDIA reference.
- Model-level throughput or memory.
- Temporal cache behavior beyond the first step.
- Denoising training, which the fixture disables.

## Troubleshooting

If the info or anchor file is missing, run `prepare_dataset.sh` and
`prepare_anchors.sh` first. If the revision check fails, re-run
`prepare_checkout.sh`.

If dependency imports fail, install the versions in `requirements.lock` alongside
the matching platform PyTorch build. If `mmcv._ext` is reported missing, confirm
that the run goes through `sparse4d_patcher.apply()` before importing `mmdet`; the
patcher installs the guard.

If anchor generation fails on the plugin import, run it through
`prepare_anchors.sh` without `--no-patch`; that routes the tool through
`run_with_patcher.py`.

If strict dispatch fails, verify that the installed muSPACE wheel exports the
native `deformable_aggregation` entry points and that the run was started with
`--backend strict` (the validation script's default).
