# `centerpoint_target_assign` benchmark

## Scope

The measured entry point is `mu_space.ops.centerpoint_target_assign` with `backend="musa"`, called
on `boxes [N, 9]` in `[x, y, z, w, l, h, vx, vy, yaw]` order and one-based `class_ids [N]`. A
published row covers the whole public callable: argument validation, the dispatch decision, the
allocation of the heatmap, annotation buffer, index and mask outputs, and the single native
launcher call that fills them. The operator is not differentiable — there is no backward pass to
time — so every row uses `operator` scope.

The model work around the operator is outside the timed region, as is the host-to-device transfer
of the boxes: in a detector this operator runs between the dense head's convolution and the loss,
and neither of those is part of this benchmark.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `c1-n256` | `boxes [256, 9]`, one class, feature map `128 x 128`, `out_size_factor 8`, `gaussian_overlap 0.1`, `min_radius 2`, voxel size `(0.1, 0.1)` m, point range starting at `-51.2` m, `max_objs 256`: a single-class head, where the fixed grid pass dominates the per-object loop | `float32` | `2333` |
| `c10-n256` | the same head with ten classes at 256 objects: the nuScenes class count at a typical full-frame object count for this head | `float32` | `2333` |
| `c10-n512` | the same head with ten classes at 512 objects against a 500-object assignment buffer: the per-object loop dominates and the buffer bound is reached | `float32` | `2333` |

Every fixture is built from the seed alone: centres, sizes, velocities and yaws are drawn from the
documented head grid, sixteen objects repeat an earlier same-class centre so the heatmap
max-update collision path runs, and four objects sit beyond the pooled range so the out-of-map
skip path runs as well. No data set is redistributed.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

Scope is `operator`; the work the timed region includes is enumerated in `## Scope`. The fixture is
built on the host before timing and moved to the measured device once, so no host transfer happens
inside the timed region. The operator allocates its own outputs, so the allocation is inside the
timed region and is part of what the rows measure.

The timer is device events (`torch.musa.Event(enable_timing=True)`): one sample records an event
pair around `--iterations` calls, synchronises, and divides the elapsed time by `--iterations`;
warmup calls are untimed and the published number is the median of `--repeats` independent samples.
The local record names this timer `device-event`. The default protocol is used unchanged:
`--warmup 20 --iterations 100 --repeats 5`, seed `2333`.

This operator is bound by a fixed per-call cost rather than by its workload shape, so single
executions of it are not stable enough to publish. Every published row is therefore the median over
five full executions, as `docs/benchmarking.md` requires when the run-to-run spread exceeds the
acceptance band. The per-execution medians were `c1-n256` 0.0749, 0.0752, 0.0749, 0.0768, 0.0749;
`c10-n256` 0.0765, 0.0751, 0.0882, 0.0745, 0.0945; `c10-n512` 0.0752, 0.0765, 0.0768, 0.0762,
0.0765 ms. The `c10-n256` set contains two executions 18–24% above the rest — the case was measured
while the device was shared — and the median absorbs them: the published 0.0765 ms is the same
number the two clean cases report. The records keep every per-execution value and the execution
count in `protocol` and `derived`.

No reference measurement is published, so the reference and speedup cells are empty. The
`S5000 normalized (ms)` cell applies the registered nominal bandwidth ratio `1600 / 1935`. That
adjustment is indicative only for these rows: the operator moves a few hundred kilobytes of
heatmap and annotation data, so the ratio of nominal device bandwidths does not describe it. The
normalized cell is a scaling of the measured latency, not a prediction of a measurement on the
reference device.

## Run

```bash
PYTHONPATH=. python benchmarks/centerpoint_target_assign/benchmark_centerpoint_target_assign.py \
  --case c1-n256 --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/executions/centerpoint_target_assign-c1-n256-e1.json
PYTHONPATH=. python benchmarks/centerpoint_target_assign/benchmark_centerpoint_target_assign.py \
  --case c10-n256 --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/executions/centerpoint_target_assign-c10-n256-e1.json
PYTHONPATH=. python benchmarks/centerpoint_target_assign/benchmark_centerpoint_target_assign.py \
  --case c10-n512 --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/executions/centerpoint_target_assign-c10-n512-e1.json
```

Repeat each command for `-e2` to `-e5`, then merge the five executions of a case into the published
record:

```bash
PYTHONPATH=. python benchmarks/_harness.py aggregate \
  .tmp/benchmarks/executions/centerpoint_target_assign-c10-n256-e*.json \
  --output .tmp/benchmarks/centerpoint_target_assign-c10-n256.json
```

Add `--check` to compare the native outputs against the module's own PyTorch oracle before timing.

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `c1-n256` | `float32` | `operator` | 0.0749 | 0.0619 |  |  | reference unavailable |
| `c10-n256` | `float32` | `operator` | 0.0765 | 0.0632 |  |  | reference unavailable |
| `c10-n512` | `float32` | `operator` | 0.0765 | 0.0633 |  |  | reference unavailable |

## Acceptance

Correctness is the precondition and is established by the operator test:
`PYTHONPATH=. python -m pytest tests/ops/test_centerpoint_target_assign.py -q` passes on the
measured build. It is confirmed on the published cases by the module's own PyTorch oracle, which
the native path reproduces with a heatmap maximum absolute error of `7.5e-08`, an annotation
maximum absolute error of `9.5e-06`, zero index mismatches and zero mask mismatches; the oracle
assigns 115, 115 and 245 objects in the three cases, and the records keep those numbers.

The pass criterion for a published row is reproducibility: re-running the documented commands with
the default protocol on a quiet device reproduces the published S5000 p50 within 5%. Per-execution
medians inside one measurement set may spread further than that when the device is shared — the
spread is recorded in every record and in `## Measurement` — so a row is compared through its
five-execution median, never through a single run. No reference row is published, so the reference
and speedup cells stay empty and every row says `reference unavailable` in `Note`; `## Notes` names
the reference implementation that would be used.

## Notes

- No reference measurement is published for any row. The upstream CenterPoint target builder this
  operator ports is a host-side NumPy loop — `draw_umich_gaussian` plus a Python loop over objects,
  at the upstream revision the operator documents — and not a device implementation. Timing it
  would compare a host loop against a device kernel rather than two implementations of the same
  boundary, which is the comparison `docs/benchmarking.md` rejects; the same reason kept a
  host-expression reference out of the `sigmoid_focal_loss` rows. The reference cell stays empty,
  the rows carry `reference unavailable`, and a device-side upstream implementation would have to
  appear before a speedup could be published here.
- The three cases sit within about 2% of each other. On this build the operator is bound by its
  fixed per-call cost — validation, the dispatch decision, two output allocations and the launcher
  — rather than by the object count or the class count. The shapes are published anyway so that a
  future change to that fixed cost becomes visible instead of hiding inside a single row.
- The rows describe the operator boundary only. They are not model-level or pipeline latencies and
  they say nothing about the cost of the dense-head convolution or of the loss that surround this
  operator in a detector.
- The `S5000 normalized (ms)` cell is an indicative bandwidth scaling, not a roofline result and
  not a prediction of a measurement on the reference device.
- The fixture is synthetic and the repository redistributes no data set. Its collision and
  out-of-map objects are deliberate: they exercise the max-update and skip branches, and they are
  the same objects the PyTorch oracle is compared against.
