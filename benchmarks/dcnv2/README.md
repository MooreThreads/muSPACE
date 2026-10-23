# DCNv2 benchmark

## Scope

The measured entry point is `mu_space.ops.modulated_deform_conv2d_pack`, the `pack` path of the
operator: the offset convolution through muDNN, the fused NHWC deformable gather, and the main GEMM
on channel-last columns.

The timed region is the whole operator call. It therefore contains the offset convolution, the
deformable column matrix, the GEMM, the operator output, and the workspace allocations the operator
performs internally on every call. Excluded from the timed region: the host-to-device copy of the
case, the dtype cast, the channel-last conversion, `requires_grad_(True)`, the dispatch report, and
every model-level computation around the operator in the BEVFormer image backbone.

`--path general` measures the general NCHW path instead — offset convolution and sigmoid in
PyTorch, then `mu_space.ops.modulated_deform_conv2d`. It is kept as an investigation path and no row
of it is published here.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `high-b6` | `input [6, 256, 58, 100]`, `offset_weight [27, 256, 3, 3]`, `weight [256, 256, 3, 3]`, 3x3 kernel, stride 1, padding 1, dilation 1, groups 1; the 58x100 ResNet-DCN stage of the BEVFormer image backbone at the single-frame batch size | `float32` | `2333` |
| `high-b18` | `input [18, 256, 58, 100]`, `offset_weight [27, 256, 3, 3]`, `weight [256, 256, 3, 3]`, 3x3 kernel, stride 1, padding 1, dilation 1, groups 1; the same stage at the multi-frame batch size | `float32` | `2333` |
| `low-b6` | `input [6, 512, 29, 50]`, `offset_weight [27, 512, 3, 3]`, `weight [512, 512, 3, 3]`, 3x3 kernel, stride 1, padding 1, dilation 1, groups 1; the 29x50 DCN stage of the backbone at the single-frame batch size | `float32` | `2333` |
| `low-b18` | `input [18, 512, 29, 50]`, `offset_weight [27, 512, 3, 3]`, `weight [512, 512, 3, 3]`, 3x3 kernel, stride 1, padding 1, dilation 1, groups 1; the same stage at the multi-frame batch size | `float32` | `2333` |

Each case builds its own tensors from the seed: inputs and biases are standard normal scaled by
0.1, and both weight tensors use the fan-in initialisation the model uses, which keeps the sampling
offsets in the range a trained model produces. No captured artifact and no data set is
redistributed.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

Scope `operator` for the four published rows: the timed region is the `pack` call itself, including
the allocations it makes internally. The timer is a MUSA event pair recorded around the launches;
one sample is 100 timed launches divided by 100, five independent samples are collected, and the
median is published with minimum, mean, maximum and standard deviation in the local record. The
default protocol is used unchanged — `--warmup 20 --iterations 100 --repeats 5`, seed `2333` — so
nothing deviates from it.

The published rows are the operator's default float32 precision mode, in which the GEMMs run in
strict IEEE float32 (`MUSPACE_DCN_FP32_TF32` unset). The operator also supports
`MUSPACE_DCN_FP32_TF32=1`, which runs the same GEMMs in TF32 and is documented as matching the
reduced-precision compute mode muDNN uses for deformable convolution; that is a different and
faster configuration with reduced numerical accuracy, and its rows are not published here. Every
record carries the mode it was measured in.

`--include-backward` times the backward pass with the forward pass and declares the
`forward-backward` scope; no such row is published. `--channels-last`, `--dtype float16` and
`--check` (which compares the two native paths before timing) are investigation switches; every
published row is `pack`, `float32`, forward only, with the default input layout. `--input-file`
replays a captured artifact for local investigation and never produces a published row.

## Run

```bash
PYTHONPATH=. python benchmarks/dcnv2/benchmark_dcnv2.py --case high-b6 --path pack \
  --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/dcnv2-high-b6.json
```

Repeat with `--case high-b18`, `--case low-b6` and `--case low-b18`, changing the record name
accordingly. `--case` and `--path` are repeatable; a run that selects more than one case or path
writes one record per row beside `--output`.

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `high-b6` | `float32` | `operator` | 2.9456 | 2.4357 | 3.5750 | 1.47x |  |
| `high-b18` | `float32` | `operator` | 8.3928 | 6.9398 | 10.6259 | 1.53x |  |
| `low-b6` | `float32` | `operator` | 2.4848 | 2.0546 | 3.1459 | 1.53x |  |
| `low-b18` | `float32` | `operator` | 6.7906 | 5.6149 | 9.4036 | 1.67x |  |

## Acceptance

Correctness is the precondition: `tests/ops/test_modulated_deform_conv.py` passes on the measured
build. The pass criterion for each published row is reproducibility and no regression: re-running
the documented command reproduces the published S5000 p50 within 1% for the case, dtype and scope
of that row, and the value does not regress against the previously accepted S5000 number for the
same case, dtype and scope. Reference measurements for the same case, dtype, scope and protocol
are published, and the comparison against them is reported in the table.

## Notes

- The reference is the original mmcv DCNv2 CUDA implementation on the NVIDIA A100 80GB PCIe,
  measured with TF32 disabled so that the reference runs the strict float32 path. On the measured
  side the deformable-convolution GEMMs are strict IEEE float32 while the offset convolution runs
  at the device library's default reduced precision, so a row compares the same work at the same
  precision class rather than bit-identical arithmetic.
- The normalized column applies the registered nominal bandwidth ratio 1600 / 1935 to the S5000
  p50. It needs the reference device model only, not a reference measurement; it is an indicative
  scaling, not a roofline result; the comparison with the reference row is exactly this scaling.
- The rows describe the operator boundary, not the backbone. They are not model-level or pipeline
  latencies, and they say nothing about the rest of the ResNet-DCN stage.
- A record also carries `effective_gbps`, computed from a traffic model of the implemented
  dataflow that charges each tensor once per pass touching it and ignores cache reuse in the
  four-corner gather. It is a lower bound on real DRAM traffic, not a measured bandwidth; use
  hardware counters when a measured value is required.
- The case module also builds a `tiny` smoke case `[2, 32, 7, 9]`. It exists for quick local runs
  and has no published row, because its latency is dominated by fixed launch cost and does not
  represent a backbone workload.
- The backward pass accumulates its input, offset and mask gradients with atomics, so backward rows
  are not bit-reproducible across runs. Only forward latency is published here.
