# Qwen-Drive-1.0 voxel_pool_depth benchmark

## Scope

The measured entry point is `mu_space.ops.voxel_pool_depth` with `backend="musa"`, the depth-aware
voxel pooling of the Qwen-Drive-1.0 view transform. The timed region is the complete training
operator boundary:

- Python frustum compaction: selecting the masked cells of
  `voxel_coords [B, N_sweep, N_cam, D, H, W, 4]` and keeping their linear point indices;
- the native forward fusion of `img_feats [B, N_sweep * N_cam, C, H, W]` and
  `img_depth [B * N_sweep * N_cam, D, H, W]` into `[B, N_cam, X, Y, Z, C]`, where every cell
  accumulates into its output voxel with `atomicAdd`;
- the backward pass for `img_feats` and for `img_depth`, fused into one warp-per-cell kernel.

The surrounding view transform is outside the timed region. The fixture tensors are moved to the
device before timing; the output volume and the gradient buffers are allocated inside the operator
call, exactly as they are in training.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `qwen-drive-6cam-frustum` | `B=1`, `N_sweep=1`, `N_cam=6`, `D=32`, `H=W=64`, `C=64`, voxel grid `100 x 100 x 16`, mask ratio `0.35`; 275,908 valid frustum cells collide into a subset of the 160,000 voxels per camera, so the case exercises duplicate accumulation | `float32` | `2333` |

The image plane is reduced relative to the Qwen-Drive training resolution so the case fits a single
benchmark process, while the voxel grid and the six-camera sweep keep the operator's memory-bound
character. The Python compaction is part of the measured operator, exactly as in the upstream
implementation.

## Test environment

- Measured device: MTT S5000
- Reference device: NVIDIA A100 80GB PCIe

## Measurement

The published row declares the `forward-backward` scope because the operator always carries its
training backward. The timer is a device event with `enable_timing=True`: one sample is 100 timed
launches divided by 100, and five independent samples are collected. The protocol is
`--seed 2333 --warmup 20 --iterations 100 --repeats 5` on both devices.

Run-to-run spread on the measured device is larger than the ±5 % this operator accepts, so the
published value is the median of **five executions**, each a separate process of the documented
command. The per-execution medians are 1.2698, 1.2422, 1.2160, 1.2223 and 1.2257 ms. The record
states the execution count, and the reference row uses the same execution count; its per-execution
medians are 1.3537, 1.3492, 1.3507, 1.3495 and 1.3463 ms.

Only the `float32` row is published. The operator also accepts `float16` and `bfloat16` features and
depth, and `--dtype float16|bfloat16` measures those paths for local investigation, but no published
comparison uses them.

Because the Python compaction is a non-trivial share of the row, a `kernel-only` measurement of the
native kernels is a different scope; it is never merged with this row.

## Run

```bash
PYTHONPATH=. python benchmarks/voxel_pool_depth/benchmark_voxel_pool_depth.py \
  --case qwen-drive-6cam-frustum --warmup 20 --iterations 100 --repeats 5 \
  --output .tmp/benchmarks/voxel_pool_depth-qwen-drive-6cam-frustum.json
```

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `qwen-drive-6cam-frustum` | `float32` | `forward-backward` | 1.2257 | 1.0135 | 1.3495 | 1.33x | |

## Acceptance

The row is accepted while
`PYTHONPATH=. python -m pytest tests/ops/test_voxel_pool_depth.py -q` passes on the measured build
and re-running the documented command of the case reproduces the published S5000 p50 within 5 % of
the five-execution median. The forward-and-backward path must not regress against the previous
muSPACE implementation of this operator on the same case.

## Notes

- The reference implementation is the original Qwen-Drive-1.0 CUDA extension, built with fast math
  on the reference device. Its autograd wrapper defines forward only, so the reference
  `forward-backward` row calls the extension's backward entry point with the compacted coordinate,
  point-index and interval arrays that the upstream forward derives.
- The S5000 and reference rows both include their own Python compaction, so the comparison covers the
  same operator boundary. On both devices this host-side step is a non-trivial share of the row,
  which is why the row is not a pure kernel comparison.
- The measured device is shared during these runs. The row is therefore the median over five
  executions of the documented command, as `## Measurement` records, so the published latency is a
  central estimate of a noisy device rather than an exclusive-access latency.
- Regression statement for the muSPACE implementation only, not a reference row: the backward pass
  fuses the `img_feats` and `img_depth` gradients into one warp-per-cell kernel by default, and the
  depth gradient reduces over the channel dimension with the warp shuffle tree. Forcing the two
  separate passes with `MUSPACE_VOXEL_POOL_DEPTH_FUSED_BACKWARD=0` adds about 0.19 ms to the row on
  this case, and `MUSPACE_VOXEL_POOL_DEPTH_DEPTH_WARP=0` then selects the earlier serial per-cell
  depth reduction. The shuffle reduction fixes the summation order, so the fused kernel stays
  deterministic and differs from the serial order only by float32 rounding, inside the tolerance of
  the operator test. The forward kernel is selected by `MUSPACE_VOXEL_POOL_DEPTH_LAUNCH` (`auto`,
  `linear`, `2d`, `packed`).
- The operator supports `float16` and `bfloat16` forward and backward; only the `float32` row is
  published.
- The row is the operator boundary, not a model-level view-transform latency.