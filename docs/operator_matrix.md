# Operator matrix

| Operator | Canonical API | MUSA native | Reference | Model integration |
|---|---:|---:|---:|---|
| multi-scale deformable attention | `mu_space.ops.multi_scale_deformable_attn` | available for MUSA tensors | PyTorch reference fallback | BEVFormer and UniAD through `default_patcher` |
| BEVPool | `mu_space.ops.bev_pool` | available for MUSA tensors | PyTorch reference fallback | BEVFusion runtime patcher |
| BEVDepth voxel pooling (train) | `mu_space.ops.voxel_pooling_train` | available for MUSA tensors | PyTorch reference fallback | BEVDepth runtime patcher |
| BEVPoolV2 | `mu_space.ops.bev_pool_v2` | available for MUSA tensors | PyTorch reference fallback | FlashOcc through `flashocc_patcher` |
| deformable aggregation | `mu_space.ops.deformable_aggregation` | available for MUSA tensors | PyTorch reference fallback | SparseDrive runtime patcher |
| modulated deformable conv (DCNv2) | `mu_space.ops.modulated_deform_conv2d` | required, MUSA only | test-only golden model | BEVFormer ResNet-DCN runtime patcher |
| standard deformable conv (DCN) | `mu_space.ops.DeformConv2d` / `DeformConv2dPack` | required, MUSA native via mask=1 | mmcv-compatible reference surface | BEVDepth `DCN` registry patcher |
| FlashBEV pooling | `mu_space.ops.flash_bevpool` | available for MUSA tensors | PyTorch reference fallback | standalone operator API |
| FlashSCA pooling | `mu_space.ops.flash_bevpool_sca` | available for MUSA tensors | PyTorch reference fallback | standalone operator API |
| sigmoid focal loss | `mu_space.ops.sigmoid_focal_loss` | available for MUSA tensors | PyTorch BCE-with-logits reference fallback | BEVFusion source patch recipe |
| BEVFusion SparseEncoder spconv | `mu_space.ops.spconv` / legacy `sparse_conv_ext` shim | 3-D SubM/SparseConv FP32/FP16 forward and backward | PyTorch reference implementation | BEVFusion LiDAR SparseEncoder |
| BEVFusion BEV IoU/NMS | `mu_space.ops.boxes_iou_bev`, `nms_bev`, `nms_bev_normal` | available for MUSA tensors | independent polygon/NMS reference | source patch or BEVFusion iou3d patcher |
| Hard voxelize | `mu_space.ops.hard_voxelize` / `Voxelization` | available for MUSA tensors | PyTorch reference fallback | mmdet3d/BEVFusion compatibility patch |
| points in rotated 3-D boxes | `mu_space.ops.points_in_boxes_all` | available for MUSA tensors | PyTorch reference fallback | BEVFusion Depth batch point-in-box path |
| radius neighborhood query | `mu_space.ops.radius`, `radius_graph` | FP32, D < 9, sorted non-negative batches, int32 indexing range | `radius_reference`, `radius_graph_reference` | standalone operator API |
| QCNet radius integration | `mu_space.qcnet_compat` | native on MUSA tensors | upstream torch-cluster on CUDA/CPU | QCNet recipe |

Hard voxelize accepts FP32 points `[N, F]` and emits voxel coordinates in OpenMMLab
`[z, y, x]` order. The default deterministic path preserves first-seen order; the
parallel `deterministic=False` path does not guarantee voxel order. On rare FP32
voxel-boundary inputs, MUSA division rounding can place a point in a neighboring bin
compared with CUDA.

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
