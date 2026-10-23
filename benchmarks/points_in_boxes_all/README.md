# points_in_boxes_all benchmark

## Scope

The measured entry point is the native launcher `points_in_boxes_all_forward` that
`mu_space.ops.points_in_boxes_all` calls on the MUSA path: framed boxes `[B, M, 7]` (centre xyz,
size dxdydz, yaw) and points `[B, N, 3]`, with the `[B, N, M]` int32 mask written directly into a
preallocated output.

Scope is `kernel-only`: the timed region is one launcher call. The launcher-side work it excludes
is the public wrapper's validation, its dispatch report and its output allocation, the
host-to-device transfer of the fixture, and the device-to-host copy or reduction a caller performs
on the mask.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `b1-n10000-m100` | `boxes [1, 100, 7]`, `points [1, 10000, 3]`, mask `[1, 10000, 100]` int32: a small frame whose latency is dominated by the fixed per-launch cost | `float32` | `2333` |
| `b1-n100000-m100` | `boxes [1, 100, 7]`, `points [1, 100000, 3]`, mask `[1, 100000, 100]` int32: the default workload of this benchmark | `float32` | `2333` |
| `b1-n1000000-m100` | `boxes [1, 100, 7]`, `points [1, 1000000, 3]`, mask `[1, 1000000, 100]` int32: a large frame that emphasizes the per-point work | `float32` | `2333` |

Every fixture is built from the seed alone: box centres and sizes are standard normal with the
size made strictly positive by `abs(dxdydz) + 1`, points are standard normal, and all draws come
from one host generator, so the fixture does not depend on the measured device. The shape switches
`--batch`, `--num-points` and `--num-boxes` override one dimension of a case for local
investigation and re-derive the case id.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

Scope is `kernel-only`; the launcher-side work the timed region excludes is enumerated in
`## Scope`. The fixture and the `[B, N, M]` int32 output are allocated before the timed region and
reused by every launch, so no allocation happens inside it and nothing is copied back to the host.

The timer is device events (`torch.musa.Event(enable_timing=True)`): one sample records an event
pair around `--iterations` launches, synchronises, and divides the elapsed time by `--iterations`;
warmup launches are untimed and the published number is the median of `--repeats` independent
samples. The local record names this timer `device-event`. The default protocol is used unchanged:
`--warmup 20 --iterations 100 --repeats 5`, seed `2333`.

No reference measurement is published, so the reference and speedup cells are empty. The
`S5000 normalized (ms)` cell applies the registered nominal bandwidth ratio `1600 / 1935`. That
adjustment is indicative only for these rows: the kernel writes each mask element once and the
small case is dominated by fixed per-launch cost, so the ratio of nominal device bandwidths does
not describe it, and a `kernel-only` row is never comparable with a reference `operator` row. The
normalized cell is a scaling of the measured latency, not a prediction of a measurement on the
reference device.

## Run

```bash
PYTHONPATH=. python benchmarks/points_in_boxes_all/benchmark_points_in_boxes_all.py \
  --case b1-n10000-m100 --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/points_in_boxes_all-b1-n10000-m100.json
PYTHONPATH=. python benchmarks/points_in_boxes_all/benchmark_points_in_boxes_all.py \
  --case b1-n100000-m100 --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/points_in_boxes_all-b1-n100000-m100.json
PYTHONPATH=. python benchmarks/points_in_boxes_all/benchmark_points_in_boxes_all.py \
  --case b1-n1000000-m100 --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/points_in_boxes_all-b1-n1000000-m100.json
```

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `b1-n10000-m100` | `float32` | `kernel-only` | 0.0180 | 0.0149 |  |  | reference unavailable |
| `b1-n100000-m100` | `float32` | `kernel-only` | 0.1365 | 0.1129 |  |  | reference unavailable |
| `b1-n1000000-m100` | `float32` | `kernel-only` | 1.2935 | 1.0695 |  |  | reference unavailable |

## Acceptance

Correctness is the precondition and is established by the operator test:
`PYTHONPATH=. python -m pytest tests/ops/test_points_in_boxes_all.py -q` passes on the measured
build, and the native MUSA mask equals the independent PyTorch reference element for element,
including the rotated and the boundary cases.

The pass criterion for a published row is reproducibility: re-running the documented command with
the default protocol reproduces the published S5000 p50 within 5% on the measured device. No
reference row is published, so the reference and speedup cells stay empty and every row says
`reference unavailable` in `Note`; `## Notes` names the reference implementation that would be
used.

## Notes

- No reference measurement is published for any row: the original BEVFusion CUDA
  `points_in_boxes_batch` kernel has not been measured on the reference device under these cases
  and this protocol. The reference cell stays empty, the rows carry `reference unavailable`, and
  the missing reference is outstanding work.
- The `S5000 normalized (ms)` cell is an indicative bandwidth scaling, not a roofline result and
  not a prediction of a measurement on the reference device. Rows with different scopes are never
  merged through it.
- The fixture is synthetic and the repository redistributes no data set. The rows describe the
  kernel boundary for one frame shape each, not model-level latency, and they say nothing about
  the cost of the wrapper or of the reduction that consumes the mask.
- Earlier measurements of this operator were produced under a different protocol and a different
  record format. They are not republished here, no regression claim is made against them, and a
  baseline of muSPACE itself is not a reference platform.
