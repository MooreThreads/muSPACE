# Sparse convolution benchmark

## Scope

Measured: `mu_space.ops.spconv.indice_conv`, the BEVFusion sparse-convolution primitive, called
once per launch with `inverse=False` and the submanifold flag of the case. The legacy
`sparse_conv_ext` shim exposes the same function as `indice_conv_fp32` and `indice_conv_half`.

Excluded from the timed region: indice-pair (rulebook) construction, the host-to-device transfer
of the features, filters and rulebook, the untimed call that yields the output shape used by the
traffic model, convolution backward, `SparseConvTensor` orchestration, the dense conversion and
the multi-layer `SparseEncoder` pipeline. The timed region does contain the operator's own output
allocation and its host-side dispatch, which the published entry point owns and which cannot be
hoisted without changing that entry point; the rows are therefore `operator` scope and not
`kernel-only`.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `bevfusion` | BEVFusion LiDAR SparseEncoder shape: `features [8192, 128]`, filters `[27, 128, 128]`, spatial shape `128 x 128 x 41` (671744 cells, 8192 active), kernel `3 x 3 x 3`, stride `1`, padding `1`, sparse convolution; rulebook `[27, 2, 8192]` with 215010 live pairs | `float32` | `2333` |
| `bevfusion` | the same tensors as the `float32` row, with `float16` features and filters through the same entry point | `float16` | `2333` |
| `dense-subm-c64` | fully occupied submanifold cube: `features [512, 64]`, filters `[27, 64, 64]`, spatial shape `8 x 8 x 8` (all 512 cells active), kernel `3 x 3 x 3`, stride `1`, padding `1`, submanifold convolution; rulebook `[27, 2, 512]` with 10648 live pairs | `float32` | `2333` |

Each case builds its indices, features and filters from the seed alone: indices are drawn from a
permutation of the sparse volume (`bevfusion`) or from the fully dense coordinate set
(`dense-subm-c64`), features and filters are standard normal. The cases are synthetic and
redistribute no data set; they preserve the coordinate layout and the channel shapes of the model
but are not a trained nuScenes dump.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

Every row uses `operator` scope: one `indice_conv` call per launch, with the features, filters and
rulebook built and moved to the device before the timed region. The operator allocates its output
inside the call, so that allocation is inside the timed region; no host synchronisation or
device-to-host copy happens there.

Timer: a MUSA event pair recorded around the timed launches, with the warmup untimed. The
published number is the median of the samples, reported with minimum, mean, maximum and standard
deviation in the local record. The default protocol is used unchanged: `--warmup 20
--iterations 100 --repeats 5`, seed `2333`. No deviation from the default protocol is
documented or required.

dtype: the `float16` row enters the same `indice_conv` entry point with `float16` features and
filters; the operator selects its half-precision kernels from the feature dtype, and no separate
runner or entry point is involved.

Executions: the `bevfusion` rows repeated within 0.4% (`float32`) and 0.3% (`float16`) over eleven
executions of the documented command each, so both are published from a single execution
(`executions: 1`). `dense-subm-c64` is launcher-bound and is published as the median over eleven
executions, so its record states `executions: 11` and lists every per-execution median: those span
0.0950-0.1149 ms, i.e. -5.9%/+13.8% around the published 0.1010 ms, while each execution's own
samples agreed to about 1%.

## Run

```bash
PYTHONPATH=. python benchmarks/spconv/benchmark_spconv.py --case bevfusion --dtype float32 \
  --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/spconv-bevfusion-fp32.json
PYTHONPATH=. python benchmarks/spconv/benchmark_spconv.py --case bevfusion --dtype float16 \
  --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/spconv-bevfusion-fp16.json
PYTHONPATH=. python benchmarks/spconv/benchmark_spconv.py --case dense-subm-c64 --dtype float32 \
  --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/spconv-dense-subm-c64-fp32.json
```

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `bevfusion` | `float32` | `operator` | 0.6260 | 0.5176 | 1.6737 | 3.23x |  |
| `bevfusion` | `float16` | `operator` | 0.3594 | 0.2972 | 1.4424 | 4.85x |  |
| `dense-subm-c64` | `float32` | `operator` | 0.1010 | 0.0835 | 0.7754 | 9.28x |  |

## Acceptance

Correctness is the precondition: `pytest tests/ops/test_spconv.py -q` passes on the measured
build. The pass criterion for a row is reproducibility on re-measurement of the documented command:
within 1% for the `bevfusion` rows, whose observed per-execution spread is +/-0.4%, and within 15%
for `dense-subm-c64`, whose observed spread is -5.9%/+13.8%. Each band comes from the observed
spread of the recorded executions rather than from a guess; a re-measurement outside its band
re-opens that row. Every case also has a published reference row, and the comparison against it is
reported in the table.

## Notes

- The normalized column is filled per the format contract: it applies the registered nominal
  bandwidth ratio `1600 / 1935` to the S5000 p50, so it needs the reference device model only and
  not a reference measurement. For the `float16` row it is indicative only, because a half-precision
  measurement is not comparable with an FP32 reference row; the column is nevertheless filled so
  that every row keeps the canonical shape.
- The reference is the original BEVFusion CUDA `sparse_conv_ext.indice_conv` on the NVIDIA A100
  80GB PCIe, timed with the same rulebook, indice-pair layout and boundary as the measured
  operator. The two implementations agree bit-for-bit where both run on the same device, so a row
  compares the same convolution surface. It is the kernel BEVFusion itself runs, not a tuned
  third-party sparse-convolution library, so a speedup here describes the model's own stack rather
  than the fastest CUDA implementation of the same operation.
- Derived traffic figures, not table entries. `effective_gbps` is the lower bound on traffic
  (`traffic_bytes_lower_bound`, which charges the features, the filters, the output read and
  written, and the rulebook once) divided by the published p50: 313.86 GB/s for `bevfusion`
  float32, 275.75 GB/s for `bevfusion` float16 and 9.37 GB/s for `dense-subm-c64` float32.
  `pair_working_set_model_gbps` is 23037.75 GB/s, 20060.29 GB/s and 1808.14 GB/s respectively; it
  charges a full filter read per live indice pair (215010 for `bevfusion`, 10648 for
  `dense-subm-c64`), which the batched GEMM does not do. It models the algorithm's per-pair
  working set and is not a bandwidth figure: it exceeds the 1600 GB/s nominal bandwidth of the
  measured device by construction and must never be compared with a device peak. Both figures
  live in the `derived` field of the local record; use profiler DRAM counters for measured
  throughput. `--peak-bandwidth-gbps` optionally adds a nominal utilization ratio to the record;
  it does not change the timed region or the published latency.
- The `float32` rows run the operator's default float32 sparse GEMM precision, which uses the
  reduced-precision tensor-core path on the measured device; `MUSPACE_SPCONV_FP32_TF32=0` forces
  strict IEEE float32 and changes the latency. Prefer the `float32` rows when claiming numerical
  alignment, and read the `float16` row as latency only.
- The scatter-add stage accumulates contributions of several kernel offsets into one output with
  atomics, so neither the forward nor the backward pass of this operator is bit-reproducible
  across runs. A re-measurement therefore reproduces within a band, not exactly.
- `dense-subm-c64` is launcher-bound: the eleven executions behind its published row span
  0.0950-0.1149 ms although each execution's own samples agreed to about 1%. Its published value is
  the median over those executions and should be read as a band. The `bevfusion` rows are not
  affected: eleven executions each repeated within 0.4%.
- Baseline, not a reference platform: earlier muSPACE revisions of this primitive were measured
  under a different seed and protocol. Their numbers are not republished here, no regression
  claim is made against them, and they must not be normalized as a reference row.
- `export_spconv_inputs.py` writes a case to an artifact and `bevfusion_spconv.patch` shims the
  legacy sparse-convolution import in an untracked upstream checkout. Both support cross-device
  replay for local investigation; neither is part of the published path, and captured artifacts
  are never committed.
