# Deformable convolution (DCN) benchmark

## Scope

Measured: the two entry points of the standard deformable convolution layer of `mu_space.ops`,
on MUSA tensors, on the shapes of the BEVDepth image backbone.

- `mu_space.ops.DeformConv2dPack` (`--op pack`), the path the backbone uses through the `DCN`
  registry patcher: the layer's own offset convolution and the deformable convolution it feeds.
- `mu_space.ops.DeformConv2d` (`--op general`): the same deformable convolution with the sampling
  offsets supplied by the caller, which leaves the offset convolution outside the operator.

Both are measured through the public module call, so the timed region is the whole `forward`: the
offset convolution of the packed entry point, the native forward launcher, the all-ones mask the
standard layer builds for itself, the output it allocates, and the gradient bookkeeping a training
forward pays for. No backward pass is timed.

Outside the timed region: case construction, host-to-device transfer of the case tensors, weight
loading, the `--check` comparison, and the offset convolution of the `general` entry point.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `pack-stage3-b6` | `--op pack --case stage3-b6`; input `[6, 512, 16, 44]`, weight `[512, 512, 3, 3]`, 3x3, stride 1, padding 1, dilation 1, `groups=1`, `deform_groups=1`; layer3 deformable convolution of the ResNet-50 backbone at the 256x704 six-camera image batch, with the offset convolution inside the timed region | `float32` | `2333` |
| `pack-stage4-b6` | `--op pack --case stage4-b6`; input `[6, 1024, 8, 22]`, weight `[1024, 1024, 3, 3]`, same layer configuration; layer4 deformable convolution at the same batch | `float32` | `2333` |
| `pack-stage3-b12` | `--op pack --case stage3-b12`; input `[12, 512, 16, 44]`, otherwise `pack-stage3-b6` | `float32` | `2333` |
| `pack-stage4-b12` | `--op pack --case stage4-b12`; input `[12, 1024, 8, 22]`, otherwise `pack-stage4-b6` | `float32` | `2333` |
| `general-stage3-b6` | `--op general --case stage3-b6`; the tensors of `pack-stage3-b6`, with the offsets precomputed outside the timed region | `float32` | `2333` |
| `general-stage4-b6` | `--op general --case stage4-b6`; the tensors of `pack-stage4-b6`, with the offsets precomputed outside the timed region | `float32` | `2333` |
| `general-stage3-b12` | `--op general --case stage3-b12`; the tensors of `pack-stage3-b12`, with the offsets precomputed outside the timed region | `float32` | `2333` |
| `general-stage4-b12` | `--op general --case stage4-b12`; the tensors of `pack-stage4-b12`, with the offsets precomputed outside the timed region | `float32` | `2333` |

A published case id is `<entry point>-<case>`; `--case` selects the layout and `--op` the entry
point. `stage3` is the 16x44, 512-channel deformable convolution of the backbone's layer3 and
`stage4` the 8x22, 1024-channel one of its layer4, at the resolutions the 256x704 image input
produces. `b6` is the six-camera image batch the training recipe feeds the backbone and `b12` its
twofold accumulation of that batch.

Every tensor of a case follows from the seed alone: the feature map, both convolutions' weights and
the offset convolution's bias are drawn in float32 and cast once. The offsets a trained backbone
produces are restored over the layer's own initialisation, so the sampling positions are the
irregular ones the operator exists for. The single-precision latency of a case is published;
`--dtype float16` is available and is not part of the published table.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe (1935 GB/s nominal, as registered in
  [benchmarks/README.md](../README.md))

## Measurement

Scope is `operator`: one call of the selected entry point on the case's tensors. The timer is a pair
of MUSA events recorded around `--iterations` timed launches, with `--warmup` untimed launches
before them, and the published number is the median of `--repeats` samples. The default protocol is
used unchanged: `--warmup 20 --iterations 100 --repeats 5`, seed `2333`.

The case tensors, the layer and its weights are built and moved to the device before the timed
region. The operator's own mask and output allocations stay inside it, because they belong to the
entry point being measured. Latency is therefore the forward pass of the layer as a training step
calls it, with the operator's default strict float32 compute mode and no TF32 substitution.

`--check` compares the measured entry point with the operator's modulated kernel at an all-ones
mask, the reference used by `tests/ops/test_deform_conv.py`, on `stage3-b6` in the published dtype
before anything is timed; both entry points reproduce it and the comparison is outside the timed
region.

## Run

```bash
PYTHONPATH=. python benchmarks/deform_conv/benchmark_deform_conv.py --case stage3-b6 --op pack \
  --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/deform_conv-pack-stage3-b6.json
```

The other cases are the same command with `--case` and `--op` set to the case id's two parts, for
example `--case stage4-b12 --op general --output .tmp/benchmarks/deform_conv-general-stage4-b12.json`.

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `pack-stage3-b6` | `float32` | `operator` | 1.4954 | 1.2365 | 1.3927 | 1.13x |  |
| `pack-stage4-b6` | `float32` | `operator` | 1.9038 | 1.5742 | 1.6371 | 1.04x |  |
| `pack-stage3-b12` | `float32` | `operator` | 2.5959 | 2.1465 | 2.7887 | 1.30x |  |
| `pack-stage4-b12` | `float32` | `operator` | 2.7193 | 2.2485 | 2.8262 | 1.26x |  |
| `general-stage3-b6` | `float32` | `operator` | 1.3452 | 1.1123 | 1.2859 | 1.16x |  |
| `general-stage4-b6` | `float32` | `operator` | 1.6470 | 1.3618 | 1.4214 | 1.04x |  |
| `general-stage3-b12` | `float32` | `operator` | 2.4414 | 2.0187 | 2.6876 | 1.33x |  |
| `general-stage4-b12` | `float32` | `operator` | 2.4641 | 2.0375 | 2.6242 | 1.29x |  |

## Acceptance

`pytest tests/ops/test_deform_conv.py -q` passes on the measured build, and re-running the
documented command reproduces the published S5000 p50 within 5% on the measured device. A
re-measurement outside that band re-opens the row. Every row also has a published reference row,
and the comparison against it is reported in the table. A row stays valid only while the operator
test passes.

## Notes

- The reference is the upstream mask-less deformable convolution on the NVIDIA A100 80GB PCIe:
  mmcv's `DeformConv2dPack` for the packed entry point and `DeformConv2d` for the general one, run
  with TF32 disabled so both sides of a comparison are strict `float32`. The measured operator
  reaches the same result through its modulated kernel with a self-built all-ones mask; that is an
  implementation detail of the measured side, not part of the reference boundary.
- The two entry points are different workloads, not two implementations of one: the packed row
  contains the offset convolution, the general one does not. Their latencies are not compared with
  each other, and the difference between them is the offset convolution's cost.
- The cases are single-operator workloads on the backbone's real shapes. The rows describe the
  operator boundary only and are not end-to-end model latency.
