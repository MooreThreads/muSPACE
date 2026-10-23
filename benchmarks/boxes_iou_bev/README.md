# boxes_iou_bev benchmark

## Scope

This directory publishes the three kernels of the BEVFusion iou3d surface that
`mu_space.ops.boxes_iou_bev` owns: `boxes_iou_bev` (pairwise rotated BEV IoU), `nms_bev` (rotated
NMS) and `nms_bev_normal` (axis-aligned NMS). `--op` selects the entry point and `--case` names
the workload; the case id carries the entry point, so `n256-iou`, `n256-nms` and `n256-nms-normal`
are three published cases over the same boxes.

Every row is `kernel-only`. The timed region is one native launcher call and nothing else: no
buffer allocation, no host synchronisation and no device-to-host copy. The launcher-side work it
excludes is:

- `boxes_iou_bev`: the public wrapper's validation, its `[N, N]` float32 output allocation, the
  dispatch report and the reference fallback;
- `nms_bev` and `nms_bev_normal`: everything except the bitmask kernel, that is the public
  wrapper's score sort and box gather, the `[N, ceil(N/64)]` int64 mask allocation, the host-side
  mask reduction that turns the bitmask into keep indices, the keep-index compaction, and the
  synchronisation the public function performs to return a variable-length index tensor.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `n256-iou` | 256 boxes `[x1, y1, x2, y2, ry]` and the `256 x 256` pairwise rotated IoU matrix: a dense single-frame box count | `float32` | `2333` |
| `n256-nms` | the same 256 boxes and the `256 x 4` rotated NMS bitmask at `thresh 0.5` | `float32` | `2333` |
| `n256-nms-normal` | the same 256 boxes and the `256 x 4` axis-aligned NMS bitmask at `thresh 0.5` | `float32` | `2333` |

Every fixture is built from the seed alone: box starts uniform in `[-40, 40)`, sizes uniform in
`[0.5, 3.5)`, yaw uniform in `[-pi/2, pi/2)` and scores uniform in `[0, 1)`. The registered fixture
sweep is `n32`, `n64`, `n128`, `n256`, `n512`, `n1024` and `n2048` boxes, and each fixture can be
measured on each entry point by qualifying its id (`--case n512-nms`); only the `n256` workloads
are published here. `--num-boxes` overrides the box count of a case for local investigation and
re-derives the case id.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

Scope is `kernel-only`; the launcher-side work the timed region excludes is enumerated in
`## Scope`. The fixture is moved to the measured device before the timed region, and the buffers
the kernel writes into — the `[N, N]` float32 IoU matrix and the `[N, ceil(N/64)]` int64 NMS
bitmask — are allocated once, before the timed region, and reused by every launch.

The timer is device events (`torch.musa.Event(enable_timing=True)`): one sample records an event
pair around `--iterations` launches, synchronises, and divides the elapsed time by `--iterations`;
warmup launches are untimed and the published number is the median of `--repeats` independent
samples. The local record names this timer `device-event`. The default protocol is used unchanged:
`--warmup 20 --iterations 100 --repeats 5`, seed `2333`. `--check` is not part of a published run;
it compares the complete public API with the independent CPU reference before timing.

Each row pairs with the reference kernel for the same case. The
`S5000 normalized (ms)` cell applies the registered nominal bandwidth ratio `1600 / 1935`. That
adjustment is indicative only for these rows: all three kernels are latency- and occupancy-bound
rather than bandwidth-bound, the box count is small enough that launch cost is a visible share of
the measured time, and a `kernel-only` row is never comparable with a reference `operator` row.
The normalized cell is a scaling of the measured latency, not a prediction of a measurement on the
reference device.

## Run

```bash
PYTHONPATH=. python benchmarks/boxes_iou_bev/benchmark_boxes_iou_bev.py --case n256-iou \
  --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/boxes_iou_bev-n256-iou.json
PYTHONPATH=. python benchmarks/boxes_iou_bev/benchmark_boxes_iou_bev.py --case n256-nms \
  --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/boxes_iou_bev-n256-nms.json
PYTHONPATH=. python benchmarks/boxes_iou_bev/benchmark_boxes_iou_bev.py --case n256-nms-normal \
  --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/boxes_iou_bev-n256-nms-normal.json
```

`--op` is optional and must agree with the case when it is given: `--case n256 --op boxes_iou_bev`
is the same workload as `--case n256-iou`, and the record always carries the qualified case id. Add
`--check` to validate the public API against the CPU reference before timing.

## Results

The three rows are the three published cases; the case id identifies both the fixture and the
measured entry point.

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `n256-iou` | `float32` | `kernel-only` | 0.0370 | 0.0306 | 0.0242 | 0.79x |  |
| `n256-nms` | `float32` | `kernel-only` | 0.4039 | 0.3339 | 0.3868 | 1.16x |  |
| `n256-nms-normal` | `float32` | `kernel-only` | 0.0459 | 0.0380 | 0.0185 | 0.49x |  |

## Acceptance

Correctness is the precondition and is established by the operator tests:
`PYTHONPATH=. python -m pytest tests/ops/test_boxes_iou_bev.py -q` passes on the measured build.
For `boxes_iou_bev` the native IoU matrix stays within `rtol 1e-4` and `atol 1e-5` of the
independent CPU polygon-clipping reference; both NMS entry points return exactly the reference keep
indices, including across mask block boundaries.

The pass criterion for a published row is reproducibility: re-running the documented command with
the default protocol reproduces the published S5000 p50 within 5% on the measured device. Each row
also has a reference row measured on the reference device with the same case, dtype, scope and
protocol, and the comparison against it is reported in the table.

Two of the three rows land below `1.00x` (`n256-iou` and `n256-nms-normal`). Each compares one
kernel launch against the reference kernel for the same 256 boxes, so a per-launch cost difference of
a few microseconds moves the ratio by tens of percent. The rows are published with their numbers
rather than omitted, and the gap is outstanding optimization work.

## Notes

- The reference rows are the original BEVFusion CUDA iou3d kernels measured on the reference
  device with the same case, dtype, scope and protocol: `boxes_iou_bev_gpu` for the IoU row and the
  exported NMS mask launch entry points behind `nms_gpu` and `nms_normal_gpu` for the two NMS rows.
  The `kernel-only` boundary is identical on both sides: one kernel launch into buffers that are
  allocated before the timed region.
- The public NMS functions return a variable-length index tensor and synchronise to read its
  length, so their end-to-end latency is not comparable with these kernel-only rows. The score
  sort and the box gather the public wrapper performs before the kernel are outside the timed
  region as well: a row describes the bitmask kernel, not the cost of a complete NMS call.
- The `S5000 normalized (ms)` cell is an indicative bandwidth scaling, not a roofline result and
  not a prediction of a measurement on the reference device. Rows with different scopes are never
  merged through it.
- The fixtures are synthetic and the repository redistributes no data set. The rows describe the
  kernel boundary at one box count, not model-level latency, and they do not cover the rest of the
  registered fixture sweep.
