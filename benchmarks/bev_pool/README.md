# BEVPool benchmark

## Scope

The measured entry point is `mu_space.ops.bev_pool` with `backend="musa"`, called on
`feats [N, C]` and `coords [N, 4]` in upstream `(x, y, z, batch)` order. The published row
covers the combined forward and backward pass over a `1 x 80 x 128 x 128` BEV grid:
coordinate rank construction, the stable rank sort, interval metadata construction, the native
pooling kernel, and the gradient with respect to `feats`. Model work around the operator —
image backbone, depth distribution, feature lifting, and the detection loss — is outside the
timed region.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `r50-200k` | `feats [200000, 80]` and `coords [200000, 4]` pooled into a `1 x 80 x 128 x 128` grid: the DepthLSSTransform pooling workload of a ResNet-50 BEVFusion configuration | `float32` | `2333` |

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

Scope is `forward-backward` for the published row, which times the forward pass and the
backward pass together; a single direction (`--mode forward` or `--mode backward`) declares the
`operator` scope and stays a local investigation path. The timer is MUSA device events: one
sample is 100 timed launches divided by 100, five independent samples are collected, and the
median is published. The protocol is `--seed 2333 --warmup 20 --iterations 100 --repeats 5`.

The fixture and `grad_output` are moved to the device before the timed region; rank
construction, the stable sort, interval metadata construction, the pooling call, and the
backward pass all run inside it, with no cached geometry between launches. `--mode backward`
times the forward pass that builds the autograd graph together with the backward pass, so that
mode is declared `operator` as well and is not published. `--points` and `--input-file` only
override or replay the fixture for local investigation.

## Run

```bash
PYTHONPATH=. python benchmarks/bev_pool/benchmark_bev_pool.py --case r50-200k \
  --warmup 20 --iterations 100 --repeats 5 --output .tmp/benchmarks/bev_pool-r50-200k.json
```

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `r50-200k` | `float32` | `forward-backward` | 4.9574 | 4.0991 | 8.5476 | 2.09x | reference includes a 419 MB layout conversion |

## Acceptance

The row is accepted while `PYTHONPATH=. python -m pytest tests/ops/test_bev_pool.py -q` passes
on the measured build and the S5000 p50 for `r50-200k` at `float32` shows no regression against
the previously published value. A reference row for the same case, dtype, scope and protocol is
published as well, and the comparison against it is reported in the table.

## Notes

- The reference is the original BEVFusion CUDA extension reached through
  `mmdet3d.ops.bev_pool.bev_pool` on the NVIDIA A100 80GB PCIe, called with the same fixture, the
  same seed and the same protocol, and measured over the same combined forward and backward pass.
  Both directions were verified against an independent PyTorch accumulation and gather reference
  with a maximum absolute error of `0.0`.
- The reference entry point does not write the published layout directly: it materialises an
  intermediate `[B, H, W, C, D]` tensor and converts it with `permute(...).contiguous()`. On this
  case that strided copy moves 419 MB and accounts for about 5.1 ms of the 8.55 ms reference
  latency, while the measured operator writes `[B, C, D, H, W]` from its kernel and pays no such
  copy. The row therefore compares the two public entry points end to end, not the two pooling
  kernels, and the gap must not be read as a device-level bandwidth comparison of the same kernel.
- `bev_pool` accepts `float32` features only, so the case has a single dtype row.
- The numbers describe the operator boundary on the BEV grid; they are not a model-level
  latency.
