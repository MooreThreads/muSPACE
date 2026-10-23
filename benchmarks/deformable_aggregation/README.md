# Deformable aggregation benchmark

## Scope

Measured: the production native entry points `deformable_aggregation_forward_opt` and
`deformable_aggregation_backward_opt` of `mu_space.ops.deformable_aggregation`, called directly with
the canonical argument order. Forward and backward are distinct entry points, so each is published as
its own row.

Outside the timed region: case construction, host-to-device transfer, the forward output tensor, the
backward gradient tensors, and the autograd function wrapper. Multi-camera feature formatting
(`feature_maps_format`), the PyTorch reference backend, and any model-level pipeline are not
measured.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `sparsedrive-forward` | `mc_ms_feat [8, 89760, 256]`, `spatial_shape [6, 4, 2]`, `scale_start_index [6, 4]`, `sampling_location [8, 900, 13, 6, 2]`, `weights [8, 900, 13, 6, 4, 8]`; 6 cameras x 4 scales, 900 anchors, 13 sample points, 8 groups, active sampling fraction 0.1864 | `float32` | `2333` |
| `sparsedrive-backward` | the shapes of `sparsedrive-forward` plus `grad_output [8, 900, 256]`, driven through the backward entry point | `float32` | `2333` |

Both cases derive their tensors from the same seed: the forward tensors come from one deterministic
stream of it and the backward tensors, including `grad_output`, from a second one. The active
sampling fraction is the share of `(anchor, sample point, camera)` triples whose normalized
coordinates fall strictly inside `(0, 1)`; triples outside that interval leave the aggregation loop
early, so the fraction is part of the workload definition.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

Both rows use `kernel-only` scope. The timed region contains one native launcher call per launch: the
output tensor of the forward pass and the three gradient tensors of the backward pass are allocated
once, before the timed region, and reused by every launch. The native kernels accumulate into those
buffers with atomics, so a launch adds to what earlier launches left behind; that changes the stored
values but not the work a launch performs or its latency.

Timer: a MUSA event pair recorded around the timed launches, with the warmup untimed. The published
number is the median of the samples. The default protocol is used unchanged: `--warmup 20
--iterations 100 --repeats 5`, seed `2333`.

## Run

```bash
PYTHONPATH=. python benchmarks/deformable_aggregation/benchmark_deformable_aggregation.py \
  --case sparsedrive-forward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/deformable_aggregation-forward.json
PYTHONPATH=. python benchmarks/deformable_aggregation/benchmark_deformable_aggregation.py \
  --case sparsedrive-backward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/deformable_aggregation-backward.json
```

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `sparsedrive-forward` | `float32` | `kernel-only` | 7.7067 | 6.3725 |  |  | reference unavailable |
| `sparsedrive-backward` | `float32` | `kernel-only` | 25.1164 | 20.7681 |  |  | reference unavailable |

## Acceptance

`pytest tests/ops/test_deformable_aggregation.py -q` passes on the measured build, and re-running the
documented commands reproduces each published S5000 p50 within 5% on the measured device. A
re-measurement outside that band re-opens the row.

## Notes

- No reference measurement is published for either row: the SparseDrive CUDA deformable aggregation
  extension has not been measured on the reference device under this case and protocol. The rows
  therefore keep an empty reference cell, and the missing reference is outstanding work.
- The published path builds its inputs from the fixed seed. `--input-file` replays a captured
  forward/backward artifact for local investigation; captured artifacts are never committed and are
  not a published input.
- The case is synthetic but follows the production shapes and active sampling density. The rows
  describe the operator boundary only and are not model-level latencies.
- `large_batch_check.py` launches the forward kernel with more threads than a 32-bit flat index
  can address (4.42e9 threads at 100 sampling points) and compares that single launch against the
  sum over ten-point chunks; the two must agree, while 32-bit launch indexing dropped the whole
  result. It is a correctness check, not a published row.
