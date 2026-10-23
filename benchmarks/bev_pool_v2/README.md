# BEVPoolV2 benchmark

## Scope

Measured: `mu_space.ops.bev_pool_v2` (re-exported as `mu_space.bev_pool_v2`) with `backend="musa"`,
through its public autograd boundary. The timed region covers the whole `forward-backward` path: the
native forward launcher, and the backward pass that `output.backward(grad_output)` triggers,
including the backward launcher's rank sort, its interval construction, and the operator's own output
and gradient allocation.

Outside the timed region: case construction, host-to-device transfer of the case tensors, and the
dispatch report. The PyTorch reference backend and the view-transformer wrapper that calls this
operator are not measured.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `r50-200k` | `depth [1, 6, 88, 16, 44]`, `feat [1, 6, 16, 44, 64]`, `bev_feat_shape [1, 1, 128, 128, 64]`, 200,000 projected points, every `ranks_depth` value unique, `ranks_bev` sorted with its matching intervals | `float32` | `2333` |

The case is one workload: the depth logits, the image features and the projection index tensors are
drawn from the fixed seed, and the backward pass reuses the same tensors with a fixed `grad_output`.
`ranks_depth` is drawn as a permutation, so every value stays unique; that is what the native
backward's projection-input contract assumes.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

Scope is `forward-backward`. The timed region contains the public operator call and the backward pass
it triggers, including the output and gradient tensors the operator allocates for itself, because
they belong to the entry point being measured. The leaf `depth` and `feat` tensors are re-derived per
launch with `detach().requires_grad_(True)`, so gradients never accumulate across launches.

Timer: a MUSA event pair recorded around the timed launches, with the warmup untimed. The published
number is the median of the samples. The default protocol is used unchanged: `--warmup 20
--iterations 100 --repeats 5`, seed `2333`.

## Run

```bash
PYTHONPATH=. python benchmarks/bev_pool_v2/benchmark_bev_pool_v2.py --case r50-200k \
  --warmup 20 --iterations 100 --repeats 5 --output .tmp/benchmarks/bev_pool_v2-r50-200k.json
```

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `r50-200k` | `float32` | `forward-backward` | 1.0404 | 0.8603 |  |  | reference unavailable |

## Acceptance

`pytest tests/ops/test_bev_pool_v2.py -q` passes on the measured build, and re-running the documented
command reproduces the published S5000 p50 within 5% on the measured device. A re-measurement outside
that band re-opens the row.

## Notes

- No reference measurement is published: the FlashOCC CUDA BEVPoolV2 extension has not been measured
  on the reference device under this case and protocol. The row keeps an empty reference cell, and
  the missing reference is outstanding work.
- The published path builds its inputs from the fixed seed. `--input-file` replays a captured case
  artifact for local investigation; captured artifacts are never committed and are not a published
  input.
- The case is a synthetic r50-ish projection with a fixed point count and fixed index distributions.
  The row describes the operator boundary only and is not end-to-end model latency.
