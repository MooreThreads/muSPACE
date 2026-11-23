# `grid_sample` benchmark

## Scope

This benchmark measures the `grid_sample` operator boundary, including Python dispatch and
allocation overhead. The forward cases time the forward call with autograd enabled; the backward
cases clear existing gradients, run the forward call, form `output.square().sum()`, and execute its
autograd backward pass. Tensor construction and device transfer are outside the timed region.

No case feeds NaN or infinite grid coordinates. The `reference` backend calls PyTorch's CPU
`grid_sample`, whose backward can crash the process (SIGSEGV, not a catchable exception) for
`padding_mode="reflection"` with `align_corners=True` and a non-finite grid; the MUSA kernels do
not. Accuracy runs that compare against the reference backend must validate the grid first.

## Cases

| Case | Input | Grid | dtype | Mode | Padding | `align_corners` | Seed |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `bev-2d-forward` | `[1, 64, 128, 128]` | `[1, 64, 64, 2]` | `float32` | bilinear | zeros | false | 2333 |
| `bev-2d-backward` | `[1, 64, 128, 128]` | `[1, 64, 64, 2]` | `float32` | bilinear | zeros | false | 2333 |
| `voxel-3d-forward` | `[1, 32, 16, 32, 32]` | `[1, 8, 16, 16, 3]` | `float32` | bilinear | zeros | false | 2333 |
| `voxel-3d-backward` | `[1, 32, 16, 32, 32]` | `[1, 8, 16, 16, 3]` | `float32` | bilinear | zeros | false | 2333 |

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

The shared benchmark harness uses device-event timing with 20 warmup iterations, 100 timed
iterations, and 5 repeats, repeated over five full executions on each device. The reported value
is the median of the per-execution medians in milliseconds. Only FP32 rows are published.
Some operator-boundary timings vary by more than 5% across executions; the five-execution
aggregation is required and does not establish a stable single-run latency.

## Run

```bash
PYTHONPATH=. python benchmarks/grid_sample/benchmark_grid_sample.py \
  --device musa --case bev-2d-forward --dtype float32 --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/grid_sample-bev-2d-forward-musa.json
```

Use `--device cuda` for the A100 reference run and substitute any case from the table above.
The MUSA runner forces the native backend; the CUDA runner uses the PyTorch reference backend.

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `bev-2d-forward` | float32 | operator | 0.0311 | 0.0257 | 0.1905 | 7.41x | Forward, autograd enabled; wrapper-inclusive |
| `bev-2d-backward` | float32 | operator | 0.3065 | 0.2535 | 0.6245 | 2.46x | Forward + loss + backward |
| `voxel-3d-forward` | float32 | operator | 0.0172 | 0.0142 | 0.1897 | 13.34x | Forward, autograd enabled; wrapper-inclusive |
| `voxel-3d-backward` | float32 | operator | 0.1942 | 0.1606 | 0.6141 | 3.82x | Forward + loss + backward |

The normalized latency convention is `S5000 latency * 1600 / 1935`; speedup is
`A100 latency / normalized S5000 latency`. These are operator-boundary measurements, not
isolated kernel speedups. In particular, the reference rows include the muSPACE wrapper's
validation, native-availability probing and reporting, not just a direct PyTorch call. A separate
2-D forward check measured about 0.1641 ms through that wrapper versus 0.0342 ms through direct
PyTorch. The operator-table ratios must not be interpreted as a native-kernel or hardware advantage.
Per-execution median ranges (ms) are S5000/A100: 2-D forward 0.0311–0.0315 /
0.1898–0.1914; 2-D backward 0.3063–0.3067 / 0.3913–0.6490; 3-D forward
0.0172–0.0374 / 0.1896–0.1907; 3-D backward 0.1941–0.2469 / 0.4358–0.6328.
The wrapper-inclusive ratios above are not native-kernel speedups.

### Forward-kernel revision

The forward sampling kernel now hoists the per-location corner geometry out of the
channel loop and issues every bilinear corner load from one straight-line expression.
The previous per-corner bounds branches serialized those loads and left the memory
pipeline under-filled. The backward kernel keeps one thread per output location so
the grid-gradient accumulation order stays identical.

A paired same-device A/B on an idle MTT S5000, alternating the previous and current
native builds within one session and using operator-boundary event timing (20 warmup
iterations, 15 timed calls, median of 3 repeats), measured 0.403 -> 0.249 ms for the
2-D spatial forward, 0.110 -> 0.072 ms for the 2-D channel forward, and 3.40 -> 2.11 ms
for the batch-2 512x512 forward. Backward times are unchanged within run-to-run
variance. The small published forward cases are dominated by wrapper dispatch and do
not change at the operator boundary. In the profiler-based table below the forward
sampling kernel drops from 399.2 to 245.7 µs (2-D spatial), 3391.7 to 2102.4 µs
(batch-2 512²), 109.8 to 69.6 µs (2-D channel) and 23.9 to 20.2 µs (3-D); backward
stays within run-to-run variance.
The sampling-kernel table below was re-measured with the same protocol on the current
build; the operator-boundary rows above are wrapper-dominated and were left as
originally measured.

### Sampling kernel durations

A separate GPU timeline measurement excludes host submission gaps and auxiliary GPU work.
Both devices use FP32, bilinear interpolation, zeros padding, `align_corners=False`, seed 2333,
and the shapes listed below. After 20 warmup calls, the profiler records 10 calls and reports the
median sampling-kernel duration. An unprofiled event check uses 50 calls and 5 repeats.
The diagnostic table reports the median of five full executions on each device, not a release event-timing row.
Backward uses a saved forward result and a preallocated all-ones output gradient with
`torch.autograd.grad(..., retain_graph=True)`; it excludes forward, loss and gradient clearing.
Input-gradient zeroing is measured separately, not included in the sampling-kernel column.

| Case | Input / grid | S5000 sampling kernel (µs) | A100 sampling kernel (µs) | S5000 / A100 raw time |
| --- | --- | ---: | ---: | ---: |
| 2-D forward | `[1,64,128,128]` / `[1,64,64,2]` | 29.941 | 32.496 | 0.92 |
| 2-D backward | same | 128.124 | 85.822 | 1.49 |
| 2-D spatial forward | `[1,64,256,256]` / `[1,256,256,2]` | 245.688 | 124.177 | 1.98 |
| 2-D spatial backward | same | 419.253 | 399.524 | 1.05 |
| 2-D large forward | `[2,64,512,512]` / `[2,512,512,2]` | 2102.385 | 1434.236 | 1.47 |
| 2-D large backward | same | 4543.421 | 6224.812 | 0.73 |
| 2-D channel forward | `[1,256,128,128]` / `[1,64,64,2]` | 69.622 | 137.744 | 0.51 |
| 2-D channel backward | same | 509.396 | 370.562 | 1.37 |
| 3-D forward | `[1,32,16,32,32]` / `[1,8,16,16,3]` | 20.200 | 43.151 | 0.47 |
| 3-D backward | same | 123.244 | 105.600 | 1.17 |

Ratios above 1 mean the S5000 kernel takes longer. No bandwidth normalization is applied to
these ratios. Forward distributes channels over blocks; backward reuses channel-invariant corner
geometry while retaining an ordered per-location grid-gradient reduction. These are diagnostic
measurements: 3-D forward S5000 execution medians range from 19.921 to 20.361 µs, A100
large-forward medians range from 1296.058 to 1552.686 µs, and S5000 channel-forward medians
range from 69.402 to 70.302 µs. Do not interpret those rows as stable speedup guarantees.
Input-gradient zeroing is excluded: small-shape S5000 memset medians are approximately
2.7 µs (2-D) and 2.1 µs (3-D); A100 fill medians are approximately 3.5 µs (2-D) and 2.7 µs (3-D).


To reproduce the timeline measurement, prepare the case tensors before profiling, warm up and
synchronize the device, then use `torch.profiler.profile` with CPU and the device activity
(`ProfilerActivity.PrivateUse1` for MUSA, `ProfilerActivity.CUDA` for CUDA). Capture 10 calls,
synchronize before leaving the profiling context, and inspect the exported Chrome trace.
On MUSA, call `mu_space.grid_sample(..., backend="musa")`; on CUDA, call
`torch.nn.functional.grid_sample` directly. Select only `sample_forward_kernel` and
`sample_backward_kernel` on MUSA and the corresponding `grid_sampler_*_kernel` on CUDA;
report memset/fill events separately.
The optimized 2-D small forward sampling kernel is about 30 µs; expanded workloads remain
hundreds of microseconds to milliseconds. Event-call times can include host submission gaps,
especially for the shorter 3-D forward kernel, so they must not replace timeline kernel durations.

The general-purpose runner automates event timing and sampling-kernel extraction:

```bash
PYTHONPATH=. python benchmarks/grid_sample/benchmark_kernels.py \
  --device musa --input-shape 1 64 128 128 --grid-shape 1 64 64 2 \
  --direction forward --warmup 20 --iterations 50 --repeats 5 --profile-calls 10 \
  --output grid-sample-kernels.json
```

Use `--device cuda` for direct PyTorch on the reference device, and `--direction backward`
for the retained-graph measurement. The runner fails if device sampling events are unavailable.


The repaired implementation passes FP32 random-output-gradient checks against hash-identical
A100 inputs at the small, 256², batch-2 512² and channel-256 2-D shapes, and the expanded 3-D
shape. Grid gradients are bit-identical in those five checks; forward and input gradients pass
`rtol=atol=1e-4`. The repair preserves corner/channel order and explicitly fuses the final
multiply-add. A separate 144-case rank/dtype/mode/padding/alignment/singleton matrix also passes
against CUDA computation in FP32, with FP16/BF16 results cast back to the storage dtype.

CPU 2-D backward instead groups adjacent pixel differences before channel accumulation. Its
FP32 grid gradients can differ beyond `1e-4` at cancellation points, including between CPU and
CUDA themselves. The expanded regression uses the framework device reference; this is not a
claim of strict CPU agreement for arbitrary shapes. Input-gradient atomics remain nondeterministic.


## Acceptance

The native implementation must pass `tests/ops/test_grid_sample.py` before performance results are
added. The target is to reproduce its S5000 median within 5%; when execution medians exceed
that spread, use the documented five-execution aggregation and disclose the range rather than
claim single-run stability. The A100 row must use the same execution count, case, seed, dtype,
mode, padding, `align_corners`, and timing protocol.

## Notes

- FP16 and BF16 are available for local investigation; only FP32 is a published comparison dtype.
- The benchmark measures the operator, not a model-level view-transform latency.
- Scratch outputs and comparison artifacts are not part of the repository results.
