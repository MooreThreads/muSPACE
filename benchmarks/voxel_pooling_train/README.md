# BEVDepth voxel_pooling_train benchmark

## Scope

The measured entry point is `mu_space.ops.voxel_pooling_train` with `backend="musa"`. The timed
region is the complete training operator boundary: position-map (`pos_memo`) creation, native
forward aggregation of `input_features [B, N, C]` into the `[B, C, Y, X]` voxel grid, and the
backward pass that scatters the output gradient back to `input_features`. The voxel output and
the position map are allocated inside the operator call, as they are in training; only the
fixture tensors are moved to the device before timing. The surrounding model work — image
backbone, depth distribution, and the detection loss — is outside the timed region.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `bevdepth-60k-mixed-coords` | `geom_xyz [2, 60000, 3]` (int32) and `input_features [2, 60000, 80]` on a `128 x 128 x 1` voxel grid; the fixture mixes valid coordinates, coordinates outside the grid, and duplicates that accumulate into one voxel | `float32` | `2333` |

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

The published row declares the `forward-backward` scope because the operator always carries its
training backward. The timer is device events on the measured device: one sample is 100 timed
launches divided by 100, five independent samples are collected, and the median of the samples is one
execution. Run-to-run spread on this launcher is larger than the ±5 % this operator accepts, so the
published value is the median of **five executions**, each a separate process: 0.2059, 0.2274,
0.2055, 0.2058 and 0.2048 ms. The protocol is `--seed 2333 --warmup 20 --iterations 100
--repeats 5` with one event timer per device; the same execution count is used on the reference
device.

Position-map creation, forward aggregation, and the `input_features` backward all run inside
the timed region; the fixture tensors are moved to the device before it, and the output gradient
is a contiguous `float32` tensor. The operator also accepts `float16` features, and
`--dtype float16` measures that path for local investigation, but only the `float32` row is
published.

## Run

```bash
PYTHONPATH=. python benchmarks/voxel_pooling_train/benchmark_voxel_pooling_train.py \
  --case bevdepth-60k-mixed-coords --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/voxel_pooling_train-bevdepth-60k-mixed-coords.json
```

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `bevdepth-60k-mixed-coords` | `float32` | `forward-backward` | 0.2058 | 0.1702 |  |  | reference unavailable |

## Acceptance

The row is accepted while
`PYTHONPATH=. python -m pytest tests/ops/test_voxel_pooling_train.py -q` passes on the measured
build and re-running the documented command of the case reproduces the published S5000 p50 within
5 %, compared against the five-execution median. No reference row is published, so the reference
and speedup cells stay empty and the row says `reference unavailable` in `Note`; `## Notes` names
the reference implementation that would be used.

## Notes

- The intended reference is the original BEVDepth CUDA extension on the NVIDIA A100 80GB PCIe.
  No reference latency is published yet, so the reference and speedup cells stay empty and the
  missing reference is outstanding work; the normalized column is still filled from the nominal
  bandwidth ratio.
- The fixture deliberately contains out-of-grid coordinates and duplicates, so the row covers
  rejection of invalid points and accumulation into a single voxel, not only the collision-free
  path.
- The numbers describe the training operator boundary; they are not a model-level latency.
