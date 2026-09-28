# BEVDepth Fixture Validation

The recipe runs deterministic single-card BEVDepth training (up to five epochs) using
real nuScenes scenes `scene-0001`–`scene-0010` for train and `scene-0011`–`scene-0012`
for validation. It validates model launch and native dispatch; it is not full-dataset
training, an accuracy benchmark, or a performance benchmark.

## Prerequisites

Prepare the pinned checkout and the 12-scene subset as described in [README.md](README.md), install
muSPACE with the native `voxel_pooling_train` symbols, and run inside an environment with
an available MUSA device.

## One-command validation

```bash
model_examples/BEVDepth/scripts/validate_fixture.sh \
  --checkout ./BEVDepth \
  --data-root <PREPARED_NUSCENES_ROOT> \
  --dtype float32
```

Use `--dtype float16` to exercise the official mixed-precision path. The command accepts
`--epochs`, `--max-hours`, and `--max-steps`; defaults are five epochs, eight hours, and
unlimited steps. `--device cuda` runs the pinned upstream CUDA extension without applying
the muSPACE patcher.

## Acceptance criteria

Validation passes only when the result JSON reports:

- a finite loss;
- `finite_gradients: true`;
- `optimizer_step: true`;
- a dispatch report with `selected_backend: "musa"`, native availability, and no reference fallback.

The selected upstream revision is checked before model construction. A mismatched checkout,
missing info file, unavailable MUSA device, non-finite update, or non-native dispatch fails
the command.

## Troubleshooting

If checkout preparation fails, confirm that the destination has the official BEVDepth origin
and that the pinned revision can be fetched. If scene preparation fails, verify that metadata
contains `v1.0-trainval` and that the sensor root contains `samples`.

If dependency imports fail, reinstall the versions in `requirements.lock` alongside the
matching platform-specific PyTorch build. If strict dispatch fails, verify the installed
muSPACE wheel exports the native `voxel_pooling_train` entry points; the recipe does not
silently fall back to the PyTorch reference path.
