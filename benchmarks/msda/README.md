# MSDA benchmark

## Scope

The measured entry point is the pair of native kernels that
`mu_space.ops.multi_scale_deformable_attn` dispatches to on MUSA tensors:
`ms_deform_attn_forward` and `ms_deform_attn_backward`, launched through the muSPACE AOT module
with the same arguments, the same dtypes and the same contiguity handling the shipped autograd
function uses.

The timed region covers one forward launch on the case's forward inputs followed by one backward
launch on its backward inputs: the pair of kernels a BEVFormer training step runs. It contains the
output allocation of the forward pass, the three zeroed gradient accumulators of the backward pass
and the `contiguous()` copy of the non-contiguous `grad_output`, because the shipped backward
function performs exactly those steps on every call. Excluded from the timed region: dispatch and
dispatch-report recording, the autograd wrapper, float32 dtype normalisation, the device copy of
the case, and every model-level computation around the operator (image backbone, BEV encoder
attention, detection head).

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `bevformer-1l` | `value [2, 40000, 8, 32]` over one 200x200 level, `sampling_locations [2, 40000, 8, 1, 4, 2]` near-uniform with a small boundary margin, `attention_weights [2, 40000, 8, 1, 4]` uniform at 0.25, `im2col_step 64`; the single-level MSDA workload of a BEVFormer encoder layer, forward and backward measured together | `float32` | `2333` |
| `bevformer-1l` | the same workload, the same seed and the same distributions with bfloat16 storage; the kernel keeps the sampling geometry, the bilinear weights and every accumulator in fp32 | `bfloat16` | `2333` |

Both directions are built from the seed alone: the forward pass draws `value` with scale 1.38, and
the backward pass draws its own tensors with scale 1.25 and a `grad_output [2, 40000, 256]` that
keeps the non-contiguous stride `(1, 512, 2)`. No captured artifact, dump or data set is
redistributed. Both rows are drawn in float32 from that seed and narrowed to the row's
storage dtype, so they differ in storage only.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

Scope `forward-backward`: the forward and the backward kernel are timed together in one region, and
the published value is their sum. The timer is a MUSA event pair recorded around the launches; one
sample is 100 timed launches divided by 100, five independent samples are collected, and the median
is published with minimum, mean, maximum and standard deviation in the local record. The default
protocol is used unchanged — `--warmup 20 --iterations 100 --repeats 5`, seed `2333` — so nothing
deviates from it.

Both dtypes are published, in one row each: the native kernels take float32 or bfloat16 tensors
directly. A bfloat16 row measures storage, not reduced-precision math, because the kernel
evaluates the sampling geometry, the bilinear weights and every accumulator in fp32 and narrows
only the forward result and the returned gradients back to bfloat16. Its backward accumulates in
fp32 gradient buffers, and the timed region allocates exactly those buffers.

`--input-file` replays a captured artifact through the same code path for local investigation. It
never produces a published row, because a row replayed from a captured artifact is not reproducible
from committed code; captured artifacts are never committed.

## Run

```bash
PYTHONPATH=. python benchmarks/msda/benchmark_msda.py --case bevformer-1l --dtype float32 \
  --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/msda-bevformer-1l.json
PYTHONPATH=. python benchmarks/msda/benchmark_msda.py --case bevformer-1l --dtype bfloat16 \
  --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/msda-bevformer-1l-bfloat16.json
```

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `bevformer-1l` | `float32` | `forward-backward` | 4.2387 | 3.5049 | 4.0895 | 1.17x |  |
| `bevformer-1l` | `bfloat16` | `forward-backward` | 3.9933 | 3.3020 |  |  | reference unavailable: no upstream bfloat16 implementation to time |

## Acceptance

Correctness is the precondition: `tests/ops/test_multi_scale_deformable_attn.py` passes on the
measured build. The pass criterion for the published row is reproducibility and no regression:
re-running the documented command reproduces the published S5000 p50 within 1% on `bevformer-1l`,
`float32`, `forward-backward`, and the value does not regress against the previously accepted S5000
number for the same case, dtype and scope. The reference measurement for the same case, dtype,
scope and protocol is published, and the comparison against it is reported in the table. The
`bfloat16` row is accepted on reproducibility plus the accuracy alignment recorded in `## Notes`,
because no reference implementation exists to time.

## Notes

- The reference is the original multi-scale deformable attention CUDA implementation of the mmcv
  stack on the NVIDIA A100 80GB PCIe: the raw forward and backward extension entry points that the
  mmcv autograd function calls, timed at the same two-launch boundary, with the same case and the
  same protocol. It reproduces the measured forward output exactly and the backward gradients to
  atomic-ordering rounding.
- The normalized column applies the registered nominal bandwidth ratio 1600 / 1935 to the S5000
  p50. It needs the reference device model only, not a reference measurement; it is an indicative
  scaling, not a roofline result; the comparison with the reference row is exactly this scaling.
- The row is the fused forward and backward pass, not an inference-only latency. It must not be
  compared with a forward-only measurement of another implementation, and it is not a model-level
  or pipeline latency.
- The backward pass accumulates its gradients with atomics, so this row is not bit-reproducible
  across runs the way the forward pass is; only latency is published here.
- The `bfloat16` row has no reference latency: the upstream mmcv multi-scale deformable attention
  extension has no bfloat16 path, so there is nothing to time on the reference device. Its accuracy
  is aligned against the PyTorch reference evaluated in fp32 on the reference device instead: the
  native forward and backward stay within 3.1e-3 of the peak magnitude at this shape, the float32
  control stays within 5e-7, and evaluating that reference itself in bfloat16 drifts by up to 1.7.
- The `bfloat16` row was measured on a host where the same float32 case reproduces at
  4.3620 ms rather than the published 4.2387 ms, so comparing the two rows carries that
  host difference; on that host bfloat16 runs 8.5% faster than float32 on this case.
