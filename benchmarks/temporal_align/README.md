# temporal_align gather/scatter benchmark

## Scope

`mu_space.ops.temporal_align` replaces the per-sample `torchvision.transforms.functional.rotate`
loop that BEV temporal alignment runs in BEVFormer-style pipelines. One BEV plane per sample is
rotated about a per-sample pixel centre, optionally translated, and resampled with bilinear or
nearest filtering; gradients are scattered back onto the source plane.

The measured region is the operator boundary, not the kernel: one call with a channels-first BEV
tensor, autograd enabled and wrapper validation included. The native kernels read and write that
channels-first layout directly, so the boundary carries no layout conversion.
The backward rows measure the full training step a model performs — zero the gradients, call the
operator, reduce the output to a scalar, run backward — so they contain the forward pass and the
atomic gradient scatter.

The reference rows reproduce what this operator replaces: one `torchvision` rotation per sample,
in a Python loop over the batch, on the reference device. That loop is dispatch-bound at small
batches, so part of a reference row is per-sample host work rather than sampler work. The
comparison is published anyway because it is the call the pipeline makes today.

## Cases

Every case uses seed `2333`, `float32` channels-first BEV input of shape `(batch, 256, 200, 200)`,
`mode="bilinear"`, no `shift`, and `center=(100.0, 100.0)` — the operator default for a 200x200
plane, expressed as a torchvision corner-based centre so both sides use the general affine path.

| Case | batch | BEV plane | channels | dtype | Mode | Center | Shift | Angle (deg) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bev-b1-forward` | 1 | 200 x 200 | 256 | `float32` | `bilinear` | 100, 100 | none | linspace(-12.5, 12.5, 1) |
| `bev-b1-backward` | 1 | 200 x 200 | 256 | `float32` | `bilinear` | 100, 100 | none | linspace(-12.5, 12.5, 1) |
| `bev-b4-forward` | 4 | 200 x 200 | 256 | `float32` | `bilinear` | 100, 100 | none | linspace(-12.5, 12.5, 4) |
| `bev-b4-backward` | 4 | 200 x 200 | 256 | `float32` | `bilinear` | 100, 100 | none | linspace(-12.5, 12.5, 4) |
| `bev-b8-forward` | 8 | 200 x 200 | 256 | `float32` | `bilinear` | 100, 100 | none | linspace(-12.5, 12.5, 8) |
| `bev-b8-backward` | 8 | 200 x 200 | 256 | `float32` | `bilinear` | 100, 100 | none | linspace(-12.5, 12.5, 8) |

The angles span the per-sample temporal offset range without landing on a multiple of 90 degrees,
where both paths would collapse to an exact plane rotation.

## Test environment

Measured device: `MTT S5000`, one device of the node, verified idle before the run (no other
process holding memory and no utilisation reported). The four GPUs used by earlier work were left
untouched; the whole measurement ran on a single device.

Reference device: `NVIDIA A100 80GB PCIe`, registered in `benchmarks/_harness.py` with its nominal
bandwidth. The reference rows run the same case inputs, the same warmup, iteration and repeat
counts, and the same event-timer protocol inside a task container owned by the benchmarking user;
the container was created for this run because the pre-existing containers on that host could not
initialise their device stack, and no output from a different host is mixed in.

Clock behaviour is not pinned on either side. Each record keeps the per-execution medians and the
minimum/maximum spread, so a run-to-run drift above the spread is visible rather than averaged
away.

## Measurement

Both sides use `benchmarks/_harness.py`, which times one launch region with device events:

- 20 warmup launches, then 100 timed launches per repeat, 5 repeats per execution, 5 executions
  per row, in separate processes.
- A row reports the median of the five execution medians; the spread across executions is printed
  next to it and is below one percent on every published row.
- Forward rows time a single operator call. Backward rows time the zero-grad, forward, reduce and
  backward sequence described in [Scope](#scope).
- The event timer brackets only the launches; the Python-level argument handling that the harness
  performs outside the timed region is not part of a row.

The published latency is compared after normalising for the nominal bandwidth difference between
the two devices, so the ratio reflects work rather than memory size:

- `S5000 normalized (ms) = S5000 p50 (ms) * 1600 / 1935`
- `Speedup = Reference p50 (ms) / S5000 normalized (ms)`

## Run

Measure one case on the measured device, from the repository root:

```bash
PYTHONPATH=. python benchmarks/temporal_align/benchmark_temporal_align.py \
  --device musa --case bev-b8-forward --dtype float32 \
  --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/temporal-align-bev-b8-forward.json
```

Measure the same case on the reference device by swapping `--device musa` for `--device cuda`.
Repeat the command five times with distinct `--output` paths, then fold the executions into one
record and render the table row:

```bash
python benchmarks/_harness.py aggregate .tmp/benchmarks/temporal-align-bev-b8-forward-*.json \
  --output .tmp/benchmarks/temporal-align-bev-b8-forward.json
python benchmarks/_harness.py row \
  --s5000 .tmp/benchmarks/temporal-align-bev-b8-forward.json \
  --reference .tmp/benchmarks/temporal-align-bev-b8-forward-reference.json \
  --note forward
```

Every record and row is written under `.tmp/`, which is git-ignored.

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bev-b1-forward` | `float32` | `operator` | 0.1795 | 0.1485 | 0.3783 | 2.55x | forward |
| `bev-b1-backward` | `float32` | `operator` | 0.7260 | 0.6004 | 1.2087 | 2.01x | with backward |
| `bev-b4-forward` | `float32` | `operator` | 0.6762 | 0.5592 | 1.5110 | 2.70x | forward |
| `bev-b4-backward` | `float32` | `operator` | 2.6617 | 2.2009 | 5.4389 | 2.47x | with backward |
| `bev-b8-forward` | `float32` | `operator` | 1.2979 | 1.0732 | 3.0191 | 2.81x | forward |
| `bev-b8-backward` | `float32` | `operator` | 5.0798 | 4.2004 | 13.7524 | 3.27x | with backward |

The normalized column applies `p50 * 1600 / 1935`; the speedup column is the reference p50 divided
by that normalized value, so a value above one means the native path does less work per unit of
nominal bandwidth than the loop it replaces.

### Kernel revision

The native builds below differ only in the kernel. Each pair was timed alternately within one
session, three rounds, alternating the recorded build before each round, on the same device and
with the same protocol as the published rows, so a device-side drift shows up as a disagreement
between the repeated columns rather than as a revision effect.

| Revision | BEV b8 forward (ms) | BEV b8 backward (ms) | Change |
| --- | --- | --- | --- |
| first native kernel | 14.4004 | 30.9652 | one thread per output location, channel-strided loop, per-thread plane arithmetic |
| tiled kernel | 1.9953 | 9.7461 | plane tiled over x, row and sample; 128-bit channel accesses; geometry planned once per pixel |
| planar tiled kernel | 1.2978 | 5.4211 | channels-first plane, x as the fast axis: one thread per pixel, a warp per 32 consecutive pixels of one channel, no layout conversion |
| planar gather kernel | 1.2979 | 5.0798 | adjoint by inverted rotation: every source pixel writes its own gradient once, coalesced, with no atomics and no clearing pass |

The first kernel is 7.22 times slower in the forward pass and 3.18 times slower in the backward
pass than the tiled kernel. Two effects dominate: the first kernel gave every thread one output
element and let each thread decode its plane coordinates with 64-bit division, and its channel
loop walked a channel-strided address pattern, so a warp touched a kilobyte-wide span instead of
consecutive addresses. The tiled kernel takes its indices from the grid, keeps the four-tap
sampler unrolled, and moves four channels per thread through 128-bit accesses. The backward pass
gains less because it is limited by atomic accumulation on the source plane rather than by
instruction issue.

The planar kernel is a further 1.54 times faster in the forward pass and 1.80 times faster in the
backward pass than the tiled kernel, and it is what the published rows now measure. Two changes
carry it: the four-dimensional input and output stay in their own layout, which removes the two
full-plane conversions that used to cost 1.12 ms of the 2.00 ms batch-eight forward row, and the
kernel vectors along x instead of along channels. The second change matters for the training step
too: x is the contiguous axis, so a warp owns 32 consecutive pixels of one channel and the
gradient scatter of a warp lands inside a few cache lines instead of spreading four channels
across four planes.

Vectorising along x is only faster if the x axis is also the block's fast axis. A first attempt
gave each thread four consecutive pixels and each block four pixel groups across 32 channels; it
measured 4.2 to 4.7 times slower than the tiled kernel on all four cases, because a warp then
touched eight channels of 64 bytes each per tap instead of one contiguous 512-byte span. The
published kernel keeps one pixel per thread and the whole x range inside one warp.

The gather kernel replaces the backward pass and is 6.3 percent faster on the batch-eight training
step (5.4211 to 5.0798 ms), with the backward kernel itself falling from 1.921 to 1.566 ms. The
scatter could not be made faster by writing fewer atomics: replacing the atomic with a plain store
made the step 32 percent slower, and merging the sixteen contributions of a two-by-two output
block into nine atomics was 15 percent slower still, because the scatter is limited by the sector
traffic of a scattered read-modify-write rather than by the atomic operation. Inverting the affine
map instead makes every source pixel the owner of its own gradient: it tests the sixteen output
pixels around the inverted position, keeps the corners that land on it, and writes once with a
coalesced store, so the gradient buffer no longer needs clearing either.

One implementation detail dominates that kernel. The first gather kept a compacted list of
`(index, weight)` hits and walked it per channel; the run-time index forced the list into local
memory and the kernel measured 5.6 ms. Indexing a weight grid by compile-time slots and predicating
the loads on a non-zero weight brought the same arithmetic to 1.79 ms, and folding two channels
into one iteration, so the candidate test is paid once for both, took it to 1.57 ms.

## Acceptance

- Every published row reports a speedup of at least 1.00x.
- `python -m pytest tests/ops/test_temporal_align.py` passes, including the acceptance angles, the
  out-of-bounds and non-square planes, and the illegal-input cases.
- `python benchmarks/check_benchmarks.py --check` reports no findings for this directory.
- Re-running the commands in [Run](#run) on an idle device reproduces each median within five
  percent; a larger deviation means the device was shared or throttled, not that the row moved.

## Notes

- The operator accepts both layouts. A `float32` channels-first input runs the planar kernels and
  keeps its own layout, which is the published case. A caller that holds the channels-last plane,
  or that passes a low-precision plane, runs the packed kernels and pays the two conversions of the
  four-dimensional boundary; both paths sample identically.
- FP16 and BF16 storage, nearest filtering and the integer-shift fast path are supported and
  covered by the operator tests, but they are not published: they have no upstream counterpart to
  compare against, and a speedup against a hand-written expression would measure the expression.
- The integer-shift path is an exact plane copy and cannot be compared with a rotation loop. It is
  documented in the operator tests instead of being given a row here.
- The planar backward accumulates in a fixed order, so repeated runs of the published case give the
  same gradient bits. The packed path still scatters with atomics, so a channels-last or
  low-precision caller can differ in the last bits.
- The reference rows are dispatch-bound at small batches, where the per-sample loop spends a visible
  share of its time outside the sampler. The forward speedup therefore grows with the batch as the
  sampler work dominates the reference loop: 2.55x, 2.70x and 2.81x at batch one, four and eight.
- The batch-eight training row breaks down as 1.30 ms of forward pass, 1.57 ms of gradient gather
  and about 2.2 ms of reduction and autograd outside the operator. The gather reads up to five
  neighbouring output gradients per source pixel, and that traffic is the whole cost: the forward
  pass moves five words per pixel-channel (four taps and one store) in 1.2933 ms and the backward
  moves six (five taps and one store) in 1.5662 ms, which is 1.267 and 1.255 TB/s of effective
  traffic, or 79 and 78 percent of the nominal 1600 GB/s. Both kernels sit on the same bandwidth
  ceiling, and the backward costs 1.21 times the forward against an access ratio of 6:5. Neither a
  shared-memory tile over the source plane nor a wider per-thread channel block can lower that
  traffic — the wider block was built and measured 35 percent slower — so the only remaining lever
  would be fewer taps, which a bilinear rotation does not offer.
- The batch-one backward row is 1.5 percent slower than the previous revision (0.7260 against
  0.7150 ms): the gather pays a fixed sixteen-candidate test per source pixel and at batch one the
  atomics it removes do not pay it back yet. The forward row at the same batch is unchanged, and
  from batch four up the gather is faster.
