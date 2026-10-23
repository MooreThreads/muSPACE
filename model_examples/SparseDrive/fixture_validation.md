# SparseDrive Fixture Validation

The recipe runs a deterministic single-card SparseDrive training step on one nuScenes
scene. It validates model launch and native dispatch; it is not full-dataset training, an
accuracy benchmark, or a performance benchmark.

## Prerequisites

Prepare the pinned checkout, the dependencies, and the info files and anchors as described
in [README.md](README.md). Install muSPACE with the native `deformable_aggregation`
symbols, and run inside an environment with an available MUSA device.

The pinned upstream imports `flash_attn` while the model package is imported, and its
attention blocks call flash-attn's variable-length CUDA kernel on a device that reports
itself as CUDA, so the launch check needs a `flash_attn` that can actually run there; see
[Known limitations](README.md#known-limitations).

## One-command validation

```bash
model_examples/SparseDrive/scripts/validate_fixture.sh \
  --checkout ./SparseDrive \
  --data-root <PREPARED_NUSCENES_ROOT> \
  --infos ./SparseDrive/data/infos/nuscenes_infos_train.pkl
```

`--scene-name`, `--seed`, and `--max-iters` change the fixture. `--device` must be a MUSA
device, because the check asserts native MUSA dispatch and has no reference path to
select.

## Acceptance criteria

Validation passes only when the result JSON reports:

- a finite loss;
- `finite_gradients: true`;
- `optimizer_step: true`;
- a dispatch report whose only operator is `deformable_aggregation`, with
  `selected_backend: "musa"`, `native_available: true`, and no fallback.

The pinned upstream revision is checked before model construction. A mismatched checkout,
a missing info file, an unavailable MUSA device, a non-finite update, or a non-native
dispatch fails the command.

## Troubleshooting

If the info or anchor files are missing, run `prepare_dataset.sh` and
`prepare_anchors.sh` first. If the revision check fails, re-run `prepare_checkout.sh`.

If dependency imports fail, install the versions in `requirements.lock` alongside the
matching platform PyTorch build, and make `flash_attn` importable. If strict dispatch
fails, verify that the installed muSPACE wheel exports the native `deformable_aggregation`
entry points; the recipe does not silently fall back to the PyTorch reference path.
