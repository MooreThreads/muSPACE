# FlashBEV pooling benchmark

## Scope

Measured: the two view-transformation entry points that share this directory,
`mu_space.ops.flash_bevpool` (FlashBEV) and `mu_space.ops.flash_bevpool_sca` (FlashSCA). Each is
called with `backend="musa"` through its public autograd boundary and with the argument order its
production callers use. `--op` selects the entry point, `--case` selects the workload and the
direction.

- `flash_bevpool` samples six strided camera feature maps `[1, 6, 56, 100, 128]` onto a
  `200 x 200 x 8` BEV grid spanning `[-50, 50] m` in x, `[-50, 50] m` in y and `[-5, 5] m` in z.
- `flash_bevpool_sca` samples a three-level camera feature pyramid `[1, 6, 7350, 128]` with learned
  sampling offsets and attention weights onto the same grid.

Outside the timed region: case construction, the synthetic camera rig, host-to-device transfer of the
case tensors, and the dispatch report. A forward row times one entry-point call. A backward row times
one entry-point call plus the backward pass it triggers, which is how the operator's autograd
boundary exposes the backward direction.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `simplebev-forward` | `image_feats [1, 6, 56, 100, 128]`, `feature_size [[56, 100]] x 6`, `image_size [448, 800]`, `projection_matrices [1, 6, 3, 4]` from the synthetic rig; BEV grid `200 x 200 x 8`; forward call only | `float32` | `2333` |
| `simplebev-backward` | the `simplebev-forward` inputs, with `image_feats` as the differentiated leaf and an all-ones output gradient; times the forward call plus the backward pass it triggers | `float32` | `2333` |
| `simplebev-sca-forward` | `value [1, 6, 7350, 128]` over levels `56 x 100`, `28 x 50`, `14 x 25` with `level_start_index [0, 5600, 7000]`, `sampling_offsets [1, 200, 200, 8, 3, 2, 8, 2]`, `attention_weights [1, 200, 200, 8, 3, 2, 8]`, `image_size [448, 800]`, `num_heads 8`, `num_points 2`; same BEV grid; forward call only | `float32` | `2333` |
| `simplebev-sca-backward` | the `simplebev-sca-forward` inputs, with `value`, `sampling_offsets` and `attention_weights` as differentiated leaves and an all-ones output gradient; times the forward call plus the backward pass it triggers | `float32` | `2333` |

Every row builds its inputs from the seed alone. The camera rig is synthetic but analytic and
deterministic, so the workload geometry is reproducible from the case id: one optical centre `1.6 m`
above the ego origin, six cameras `60` degrees apart in yaw and pitched `5` degrees down, focal
length `630 px` in both axes, and the principal point at the centre of the `448 x 800` image. Ego
frame is x forward, y left, z up; each camera matrix maps an ego-frame point to `(u, v, depth)` in
pixels, which is the projection convention both entry points consume. With this rig, `97.8%` of the
voxel centres at a given z are visible to at least one camera and the average centre is visible to
`1.05` cameras, so the workload is neither fully visible nor fully culled.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

All four rows use `operator` scope: the timed region covers one direction of the operator boundary
and nothing else. Each sample records one MUSA event pair around `--iterations` launches and divides
the elapsed time by `--iterations`; warmup launches are untimed; the published number is the median
of `--repeats` samples. Wall-clock timing is not used.

Preallocated before the timed region: the case tensors on the measured device, the synthetic
projection matrices, and the all-ones output gradient of the backward rows. The image-feature leaf
that the backward rows differentiate is derived once with `detach().requires_grad_(True)`, so
gradients never accumulate across launches; each launch builds and frees its own autograd graph
inside the timed region.

The default protocol is used unchanged: `--warmup 20 --iterations 100 --repeats 5`, seed `2333`.

## Run

```bash
PYTHONPATH=. python benchmarks/flashbev/benchmark_flashbev.py --op flash_bevpool \
  --case simplebev-forward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/flashbev-simplebev-forward.json
PYTHONPATH=. python benchmarks/flashbev/benchmark_flashbev.py --op flash_bevpool \
  --case simplebev-backward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/flashbev-simplebev-backward.json
PYTHONPATH=. python benchmarks/flashbev/benchmark_flashbev.py --op flash_bevpool_sca \
  --case simplebev-sca-forward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/flashbev-simplebev-sca-forward.json
PYTHONPATH=. python benchmarks/flashbev/benchmark_flashbev.py --op flash_bevpool_sca \
  --case simplebev-sca-backward --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/flashbev-simplebev-sca-backward.json
```

`--op` may be omitted; it is then taken from the selected case.

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `simplebev-forward` | `float32` | `operator` | 3.1167 | 2.5771 |  |  | reference unavailable |
| `simplebev-backward` | `float32` | `operator` | 5.5587 | 4.5964 |  |  | reference unavailable |
| `simplebev-sca-forward` | `float32` | `operator` | 8.5633 | 7.0808 |  |  | reference unavailable |
| `simplebev-sca-backward` | `float32` | `operator` | 37.0339 | 30.6223 |  |  | reference unavailable |

## Acceptance

`pytest tests/ops/test_flashbev.py -q` passes on the measured build, and re-running the documented
command of a case reproduces its published S5000 p50 within 5% on the measured device. A
re-measurement outside that band re-opens the row.

## Notes

- No reference measurement is published for any row: neither the original BEVFusion CUDA FlashBEV
  extension nor a CUDA FlashSCA implementation has been measured on the reference device under these
  cases and this protocol. The rows therefore keep an empty reference cell, and the missing reference
  is outstanding work.
- The `S5000 normalized (ms)` cell adjusts the measured latency by nominal device bandwidth: the
  S5000 p50 multiplied by `1600 / 1935`, the registered bandwidths of the measured and the reference
  device. `Speedup` stays empty because it needs a published reference measurement; rendering a row
  with `python benchmarks/_harness.py row --s5000 <record> --reference <record>` fills it once a
  matching reference record exists.
- A bandwidth-normalised comparison against a CUDA implementation is only indicative for this
  operator. FlashBEV and FlashSCA are memory-bound scatter and gather-pool kernels: their traffic is
  dominated by irregular per-camera bilinear gathers, by the atomic accumulation of the backward
  pass, and by the camera-visibility culling that decides how much of that traffic happens at all.
  None of those are described by the ratio of nominal device bandwidths, so the normalised column can
  rank two devices but cannot predict the measured ratio on a different access pattern or a different
  camera rig. A published reference row still needs the same case, the same protocol and the same
  calibration-free input construction on the reference device.
- The cases are synthetic: they use the published SimpleBEV geometry and a documented camera rig
  rather than a captured calibration file, and the repository redistributes no data set. The rows
  describe the operator boundary, not end-to-end model latency.
- A captured calibration file can be replayed with `--projection-json` for local investigation, for
  example to explain a visibility difference against a captured rig. That path is local by
  construction: it is never used for a published row and no captured file is committed.
