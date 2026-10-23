# Operator matrix

| Operator | Canonical API | MUSA native | Reference | Model integration | Benchmark |
|---|---:|---:|---:|---:|---|
| multi-scale deformable attention | `mu_space.ops.multi_scale_deformable_attn` | available for MUSA FP32/BF16/FP16 tensors | PyTorch reference fallback | BEVFormer and UniAD through `default_patcher`; Qwen-Drive through `qwendrive_patcher` | [msda](../benchmarks/msda/README.md) |
| BEVPool | `mu_space.ops.bev_pool` | available for MUSA tensors | PyTorch reference fallback | BEVFusion runtime patcher | [bev_pool](../benchmarks/bev_pool/README.md) |
| BEVDepth voxel pooling (train) | `mu_space.ops.voxel_pooling_train` | available for MUSA tensors | PyTorch reference fallback | BEVDepth runtime patcher | [voxel_pooling_train](../benchmarks/voxel_pooling_train/README.md) |
| Depth-aware voxel pooling | `mu_space.ops.voxel_pool_depth` | FP32/FP16/BF16 forward and backward for MUSA tensors | PyTorch reference fallback | Qwen-Drive through `qwendrive_patcher` | [voxel_pool_depth](../benchmarks/voxel_pool_depth/README.md) |
| BEVPoolV2 | `mu_space.ops.bev_pool_v2` | available for MUSA tensors | PyTorch reference fallback | FlashOcc through `flashocc_patcher` | [bev_pool_v2](../benchmarks/bev_pool_v2/README.md) |
| deformable aggregation | `mu_space.ops.deformable_aggregation` | available for MUSA tensors | PyTorch reference fallback | SparseDrive runtime patcher | [deformable_aggregation](../benchmarks/deformable_aggregation/README.md) |
| modulated deformable conv (DCNv2) | `mu_space.ops.modulated_deform_conv2d` | required, MUSA only | test-only golden model | BEVFormer ResNet-DCN runtime patcher | [dcnv2](../benchmarks/dcnv2/README.md) |
| standard deformable conv (DCN) | `mu_space.ops.DeformConv2d` / `DeformConv2dPack` | required, MUSA native via mask=1 | mmcv-compatible reference surface | BEVDepth `DCN` registry patcher | [deform_conv](../benchmarks/deform_conv/README.md) |
| FlashBEV pooling | `mu_space.ops.flash_bevpool` | available for MUSA tensors | PyTorch reference fallback | standalone operator API | [flashbev](../benchmarks/flashbev/README.md) |
| FlashSCA pooling | `mu_space.ops.flash_bevpool_sca` | available for MUSA tensors | PyTorch reference fallback | standalone operator API | [flashbev](../benchmarks/flashbev/README.md) |
| sigmoid focal loss | `mu_space.ops.sigmoid_focal_loss` | available for MUSA tensors | PyTorch BCE-with-logits reference fallback | BEVFusion source patch recipe | [sigmoid_focal_loss](../benchmarks/sigmoid_focal_loss/README.md) |
| BEVFusion SparseEncoder spconv | `mu_space.ops.spconv` / legacy `sparse_conv_ext` shim | 3-D SubM/SparseConv FP32/FP16 forward and backward | PyTorch reference implementation | BEVFusion LiDAR SparseEncoder | [spconv](../benchmarks/spconv/README.md) |
| BEVFusion BEV IoU/NMS | `mu_space.ops.boxes_iou_bev`, `nms_bev`, `nms_bev_normal` | available for MUSA tensors | independent polygon/NMS reference | source patch or BEVFusion iou3d patcher | [boxes_iou_bev](../benchmarks/boxes_iou_bev/README.md) |
| Hard voxelize | `mu_space.ops.hard_voxelize` / `Voxelization` | available for MUSA tensors | PyTorch reference fallback | mmdet3d/BEVFusion compatibility patch | [hard_voxelize](../benchmarks/hard_voxelize/README.md) |
| points in rotated 3-D boxes | `mu_space.ops.points_in_boxes_all` | available for MUSA tensors | PyTorch reference fallback | BEVFusion Depth batch point-in-box path | [points_in_boxes_all](../benchmarks/points_in_boxes_all/README.md) |
| radius neighborhood query | `mu_space.ops.radius`, `radius_graph` | FP32, D < 9, sorted non-negative batches, int32 indexing range | upstream torch-cluster, with `radius_reference`, `radius_graph_reference` as the local fallback | standalone operator API | [radius](../benchmarks/radius/README.md) |
| QCNet radius integration | `mu_space.qcnet_compat` | native on MUSA tensors | upstream torch-cluster on CUDA/CPU | QCNet recipe | [radius](../benchmarks/radius/README.md) |
| CenterPoint target assignment | `mu_space.ops.centerpoint_target_assign` | available for MUSA tensors | host-side NumPy loop upstream, no device reference | BEVFusion CenterPoint head training path | [centerpoint_target_assign](../benchmarks/centerpoint_target_assign/README.md) |

The `Benchmark` column links to the published benchmark record for each operator. Measured rows
follow the [operator benchmark standard](benchmarking.md).

Hard voxelize accepts FP32 points `[N, F]` and emits voxel coordinates in OpenMMLab
`[z, y, x]` order. The default deterministic path preserves first-seen order; the
parallel `deterministic=False` path does not guarantee voxel order. On rare FP32
voxel-boundary inputs, MUSA division rounding can place a point in a neighboring bin
compared with CUDA.

## Multi-scale deformable attention

`multi_scale_deformable_attn` accepts FP32, BF16 and FP16 MUSA tensors; `sampling_locations`
and `attention_weights` follow the `value` dtype. BF16 is a native storage dtype: the kernel
evaluates the sampling geometry, the bilinear weights and every accumulator in FP32, and only
narrows the forward result and the returned gradients back to BF16. Gradients always accumulate
in FP32, so a native BF16 backward allocates FP32 gradient buffers and casts them to the input
dtypes before returning. FP16 has no native kernel and is computed through the FP32 path.

Sampling geometry is never evaluated in the storage dtype, so BF16 keeps the FP32 sub-pixel
resolution of `sampling_locations`: `tests/ops/test_multi_scale_deformable_attn.py` pins this
with a query placed exactly between two value pixels, where a BF16 geometry would sample the
neighbouring pixel and double the result.

Measured BF16 alignment: against the PyTorch reference evaluated in FP32, the native forward and
backward stay within 3.1e-3 of the peak magnitude at the published MSDA shape (2,40000,8,32),
and the FP32 control stays within 5e-7. Evaluating that reference in BF16 instead drifts by up
to 1.7 there, because it builds its sampling grid in the storage dtype.

## Depth-aware voxel pooling

`voxel_pool_depth` follows the Qwen-Drive-1.0 view-transform contract:
`img_feats [B, N_sweep * N_cam, C, H, W]`, `img_depth [B * N_sweep * N_cam, D, H, W]`,
`voxel_coords [B, N_sweep, N_cam, D, H, W, 4]`, `mask` over the frustum cells, and output
`[B, N_cam, X, Y, Z, C]`. The Python layer compacts the masked cells, ranks them by output voxel
and derives the sorted intervals; the native kernels unpack the linear frustum index, weighted-sum
the depth probability against the image features, and back-propagate to both inputs.

Environment override:

| Variable | Default | Effect |
| --- | --- | --- |
| `MUSPACE_VOXEL_POOL_DEPTH_LAUNCH` | `auto` | Launch variant. `auto` uses the packed kernel when `C % 4 == 0` and the output is 16-byte aligned, otherwise the linear kernel. `linear`, `2d` and `packed` force one variant. The `2d` variant is an experimental mapping that is slower on every measured shape. |
| `MUSPACE_VOXEL_POOL_DEPTH_DEPTH_WARP` | `1` | Backward depth-gradient reduction. `1` spreads the channels over a warp and reduces with a shuffle tree when `C >= 32`; `0` forces the serial per-cell kernel. |

The backward pass accumulates the feature gradient with atomics, so the feature gradient is not
bit-reproducible across runs. The depth gradient is a per-cell reduction; the warp reduction fixes
the summation order, so it is deterministic, but it differs from the serial per-cell order by float32
rounding.

## BEVFusion SparseEncoder spconv

The current spconv compatibility scope is limited to the 3-D operators exercised by
BEVFusion SparseEncoder.  2-D/4-D, transpose/inverse convolution, fused-BN and sparse
maxpool entrypoints are intentionally not exposed by the muSPACE shim.

Environment overrides:

| Variable | Default | Effect |
| --- | --- | --- |
| `MUSPACE_SPCONV_FP32_TF32` | `1` | Runs the float32 GEMMs of the sparse convolution in TF32. `0` forces strict IEEE float32. |

The scatter-add stage accumulates contributions from several kernel offsets into the same
output voxel with atomics, so neither the forward nor the backward pass is bit-reproducible
across runs.

## Modulated Deformable Convolution

Two native paths back the operator; both are the production kernels and neither has a
PyTorch fallback.

| Path | Entry point | Kernels | Domain |
| --- | --- | --- | --- |
| NHWC, offset-conv fused | `mu_space.ops.modulated_deform_conv2d_pack`, `ModulatedDeformConv2dPack` | `modulated_deform_conv_im2col_nhwc`, `modulated_deform_conv_col2im_nhwc` | `groups == deform_groups == 1`, 3x3, stride 1, dilation 1, `channels % 32 == 0`, `channels <= 1024` |
| NCHW, general | `mu_space.ops.modulated_deform_conv2d`, `ModulatedDeformConv2d` | `modulated_deform_conv_im2col`, `modulated_deform_conv_col2im`, `modulated_deform_conv_col2im_coord` | any groups, deform_groups, kernel size, stride, dilation |

The packed entry point picks the fused path when the configuration allows it and routes
everything else to the general path, so `ModulatedDeformConv2dPack` is a drop-in replacement
for mmcv's module regardless of configuration.

### Model integration

mmdet-based backbones do not import the layer class; they declare it in the model config
(`dcn=dict(type='DCNv2', ...)`) and `build_conv_layer` resolves that string through mmcv's
`CONV_LAYERS` registry. `mu_space.patcher.modulated_deform_conv_patcher` therefore replaces the
registry entry as well as the `ModulatedDeformConv2dPack` attribute of `mmcv.ops` and
`mmcv.ops.modulated_deform_conv`, and `restore()` puts mmcv's own class back:

```python
from mu_space.patcher import modulated_deform_conv_patcher

modulated_deform_conv_patcher.apply()   # before the model is built
```

muSPACE's module keeps mmcv's constructor signature, parameter names, and state dict layout,
including the legacy `*_offset` to `conv_offset` key rename, so mmcv checkpoints load unchanged.
Unlike MSDA, DCNv2 has no compatibility fallback, so `strict` is recorded in the patch report but
does not change execution.

Environment overrides:

| Variable | Default | Effect |
| --- | --- | --- |
| `MUSPACE_DCN_FP32_TF32` | `1` | Runs the float32 GEMMs of the fused path in TF32, matching the reduced-precision compute mode muDNN uses for deformable convolution. `0` forces strict IEEE float32. |
| `MUSPACE_DCN_RECOMPUTE_COLUMNS` | `1` | Recomputes the deformable columns in the fused backward pass instead of saving them. The column matrix dominates the operator's memory footprint; recomputing is what makes a full BEVFormer batch fit. |

The forward pass is deterministic. The backward pass is not bit-reproducible: neighbouring
sampling positions scatter into the same input pixel, so `col2im` accumulates `grad_input`,
`grad_offset` and `grad_mask` with atomics and the summation order varies between runs.

Disabled legacy `mmcv` and `mmdet3d` native extensions are not muSPACE operators and
must fail explicitly if called.

Validated stack status is reported by `mu-space env`, which uses
`mu_space.compatibility.compatibility_report(profile="core")` as the source of truth for the
SDK-bound core matrix: SDK 4.3.8 with torch/torch_musa 2.7.1, or SDK 5.2.0 with
torch/torch_musa 2.9.1 or 2.11.0. The BEVFormer recipe uses the separate `profile="bevformer"` pin set.

## Dispatch Reporting

MSDA records the most recent backend decision so users can confirm whether a call used the native
MUSA path or the PyTorch reference fallback:

```python
from mu_space import get_last_dispatch_report, multi_scale_deformable_attn

output = multi_scale_deformable_attn(...)
print(get_last_dispatch_report().as_dict())
```

Set `MUSPACE_DISPATCH_LOG=1` to print each report to stderr. A native fallback is visible as
`selected_backend: "reference"` with `native_fallback: true`. Strict validation should call the op
with `backend="musa"` or run the BEVFormer fixture with `--strict`.