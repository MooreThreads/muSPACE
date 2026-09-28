# Source Provenance

## Multi-Scale Deformable Attention

muSPACE provides a canonical MUSA implementation of multi-scale deformable attention (MSDA). The
implementation is derived from OpenMMLab MMCV's MSDA operator and adapted into the package-local
TVM FFI AOT library with an explicit Python dispatch layer.

File mapping:

| Source lineage | muSPACE file | Notes |
| --- | --- | --- |
| MMCV MSDA MUSA kernel helper code | `csrc/ms_deformable_attn/musa_kernel_helper.muh` | Reduced to the helper pieces needed by the local MSDA kernel. |
| MMCV MSDA MUSA kernel implementation | `csrc/ms_deformable_attn/ms_deform_attn_musa_kernel.muh` | Vendored and adapted for the muSPACE extension. |
| MSDA FFI launcher | `csrc/ms_deformable_attn/ms_deform_attn_musa.mu` | Reduced to canonical MSDA forward/backward launchers. |
| MMCV Python reference path | `mu_space/ops/multi_scale_deformable_attn.py` | Rewritten as a mmcv-free canonical API with explicit backend dispatch and reporting. |

Copyright and license:

- OpenMMLab MMCV is distributed under the Apache License 2.0.
- The original Deformable DETR lineage carries SenseTime copyright and Apache License 2.0 notices.
- muSPACE exports stable TVM FFI function names from `libmuspace_ops.so`; it does not reuse `mmcv._ext`.

Building muSPACE does not require a sibling MMCV checkout.

## Modulated Deformable Convolution

muSPACE provides a canonical MUSA implementation of modulated deformable convolution (DCNv2).
The kernels come from the MUSA-tuned DCNv2 operator used in production for the BEVFormer
ResNet-DCN backbone, which itself descends from OpenMMLab MMCV's `modulated_deform_conv`.
Only the kernels on the production path were carried over; the experimental variants that
accompanied them (fully fused offset convolution, channel-vectorised and atomic-free col2im,
shared-memory geometry, TileLang gather) were left behind, and version and variant suffixes
were dropped from the names that remain.

File mapping:

| Source lineage | muSPACE file | Notes |
| --- | --- | --- |
| MMCV DCNv2 CUDA kernels, MUSA-tuned | `csrc/deformable_conv/modulated_deform_conv_musa_kernel.muh` | NCHW im2col / col2im / col2im_coord plus the NHWC gather and per-element col2im. |
| Native FFI launchers | `csrc/deformable_conv/modulated_deform_conv_musa.mu` | Rewritten around tensor-derived geometry and explicit DLPack/FFI validation; launch policy is local to this file. |
| Python operator wrapper | `mu_space/ops/modulated_deform_conv.py` | Rewritten as an mmcv-free canonical API with autograd wrappers, dispatch reporting, and mmcv-compatible modules. |

Boundary semantics follow MMCV: a sampling position outside `(-1, H) x (-1, W)` contributes
nothing, and bilinear corners outside `[0, H-1] x [0, W-1]` are dropped. The NHWC kernels rely
on per-corner zeroing alone, which is equivalent except exactly on the `-1` boundary.

Model integration goes through `mu_space.patcher.modulated_deform_conv_patcher`, which replaces
mmcv's `CONV_LAYERS['DCNv2']` entry and the `ModulatedDeformConv2dPack` class attribute at
runtime. No model source is edited, and `restore()` reinstates mmcv's own class.

Copyright and license:

- The DCNv2 lineage carries a Microsoft copyright and MIT license notice.
- OpenMMLab MMCV is distributed under the Apache License 2.0.

## Runtime Model Integration

Runtime MSDA integration is provided by `mu_space.patcher.default_patcher.apply()`, which serves both
BEVFormer and UniAD. It replaces mmcv's Python MSDA helper at runtime, including already imported
model module globals when present. Default mode is compatibility-first and may fall back to the
PyTorch reference path; `MUSPACE_STRICT_PATCH=1` forces the native MUSA path for validation.

The other two operators follow the same pattern with their own patchers:
`mu_space.patcher.flashocc_patcher` rebinds FlashOcc's BEVPoolV2 binding, and
`mu_space.patcher.sparsedrive_patcher` rebinds SparseDrive's deformable aggregation binding.

## Sigmoid Focal Loss (BEVFusion)

muSPACE's elementwise binary sigmoid focal loss follows mit-han-lab/BEVFusion's
`mmdet3d/models/heads/segm/vanilla.py` contract: matching logits and binary targets,
`alpha`, `gamma`, and `none`/`mean`/`sum` reduction. The PyTorch reference and the MUSA
kernel use the same numerically stable BCE-with-logits focal-loss formulation. The optional
BEVFusion recipe applies a reversible source patch only to route that upstream function to
`mu_space.ops.sigmoid_focal_loss`; the upstream float32 conversions remain in place.

## BEVFusion iou3d BEV IoU and NMS

The BEVFusion `main` revision `326653dc06e0938edf1aae7d01efcd158ba83de5` uses the
`mmdet3d/ops/iou3d` extension for rotated BEV IoU and post-processing NMS. Its public
Python entry points are `boxes_iou_bev`, `nms_gpu`, and `nms_normal_gpu`; the boxes passed
to these functions have shape `(N, 5)` and format `[x1, y1, x2, y2, ry]`. The extension's
3D-volume IoU calculator is a separate box-structure path and is not replaced here.

muSPACE ports these entry points to TVM FFI/MUSA in `csrc/geometry/boxes_iou_bev_musa.mu`
and exposes them as `boxes_iou_bev`, `nms_bev`, and `nms_bev_normal`. The
`bevfusion_iou3d_patcher` replaces both the iou3d utility functions and the names imported
by `mmdet3d.core.post_processing.box3d_nms`, while preserving the original score ordering,
threshold, and pre/post limit semantics. The native `boxes_overlap_bev_gpu` helper is not exposed
as a public muSPACE API because the fixed BEVFusion Python path does not call it. For full NMS
calls, the Python wrapper supplies a MUSA mask workspace so native code can use the device
caching allocator; the variable-length result still requires synchronization and host mask
reduction to preserve the upstream return contract. The standalone benchmark measures only the
mask kernel and does not include that wrapper path.

| Pinned upstream source | muSPACE destination | Role |
| --- | --- | --- |
| `mmdet3d/ops/iou3d/iou3d_utils.py` | `mu_space/ops/boxes_iou_bev.py` | Python API and dispatch compatibility |
| `mmdet3d/ops/iou3d/src/iou3d.cpp` | `csrc/geometry/boxes_iou_bev_musa.mu` | FFI entry-point contract |
| `mmdet3d/ops/iou3d/src/iou3d_kernel.cu` | `csrc/geometry/boxes_iou_bev_musa.mu` | BEV overlap and NMS kernel lineage |

Source URLs are pinned to
[`326653dc`](https://github.com/mit-han-lab/bevfusion/tree/326653dc06e0938edf1aae7d01efcd158ba83de5/mmdet3d/ops/iou3d).
The upstream implementation is derived from OpenPCDet's iou3d NMS kernels. The BEVFusion
repository and its inherited source notices are the provenance baseline; muSPACE remains
Apache-2.0 and retains only the MUSA port and compatibility layer.

## BEVPoolV2

muSPACE's BEVPoolV2 API follows the FlashOCC/BEVDet-style interface:

```python
bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
            bev_feat_shape, interval_starts, interval_lengths)
```

It returns BEV features in `(B, C, Dz, Dy, Dx)` format and provides a PyTorch reference path for
correctness comparison and fallback. The native MUSA path is compiled into `libmuspace_ops.so` from
`csrc/pooling/bev_pool_v2_musa.mu`.

## BEVPool

muSPACE also implements mit-han-lab/BEVFusion's original coordinate-based API:

```python
bev_pool(feats, coords, B, D, H, W)
```

`feats` has shape `(N, C)` and `coords` has shape `(N, 4)` in `(x, y, z, batch)`
order. The result preserves the upstream Python API layout `(B, C, D, H, W)`. The MUSA
implementation is independent of BEVPoolV2 because BEVFusion does not provide its rank/interval
inputs. `mu_space.patcher.bevfusion_patcher` installs it into a loaded BEVFusion runtime.

## Deformable Aggregation (SparseDrive)

muSPACE provides a canonical MUSA implementation of SparseDrive's multi-camera multi-scale
deformable feature aggregation. The native path uses SparseDrive's production kernels
(FastDivmod forward + optimized backward, exposed as `*_opt`) and is compiled into `libmuspace_ops.so`.

File mapping:

| Source lineage | muSPACE file | Notes |
| --- | --- | --- |
| SparseDrive `deformable_aggregation_cuda.mu` | `csrc/deformable_aggregation/deformable_aggregation_musa.mu` | Baseline + optimized (`*_opt`, from former forward_v1 / backward_v3) kernels. |
| SparseDrive `deformable_aggregation.cpp` | merged into the `.mu` above | Tensor API adapted to muSPACE conventions (`PrivateUse1`, contiguous checks). |
| SparseDrive Python autograd wrapper | `mu_space/ops/deformable_aggregation.py` | Canonical API with `backend=auto|musa|reference`, plus SparseDrive aliases. |

Public API:

```python
from mu_space.ops import deformable_aggregation, deformable_aggregation_function

output = deformable_aggregation(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights
)
```

`deformable_aggregation_function` matches SparseDrive's call signature and casts inputs to
`float32` / `int32`. Runtime replacement is available through `SparseDrivePatcher`.

Constraint carried from upstream: `num_groups <= 8` and `num_embeds % num_groups == 0`.

`model_examples/SparseDrive/manifest.yaml` pins
`52c4c05b6d446b710c8a12eb9fb19d698b33cb2b`. That commit is a candidate pin: it has not been
verified against the kernel lineage recorded above, so the recipe ships no checkout patches and
the integration stays runtime-only until the comparison is done.

## Radius neighborhood query

The radius API follows the public torch-cluster contract: FP32 point coordinates,
sorted batch segments, strict squared-distance comparison, and index-order truncation. The
MUSA implementation is a clean-room TVM FFI kernel in
`csrc/geometry/radius_musa.mu`; the Python reference and `radius_graph` adapter
are implemented in `mu_space/ops/radius.py`.

The native path accepts FP32 coordinates with fewer than nine dimensions,
sorted non-negative batch labels, and tensors whose point and output indexing
fits the internal 32-bit launch dimensions. The implementation scans each
segment in deterministic x-index order and does not impose a per-segment
point-count limit.

# BEVDepth voxel pooling

`voxel_pooling_train` follows the public training operator contract and reference semantics from
Megvii-BaseDetection/BEVDepth at `d78c7b58b10b9ada940462ba83ab24d99cae5833`. The MUSA source is
a clean-room TVM FFI/DLPack implementation and does not include PyTorch extension source from upstream.

BEVDepth's standard mmcv `DCN` is exposed through `mu_space.ops.DeformConv2dPack` and
patched only at runtime. It reuses the native modulated deformable-convolution kernel with
an all-ones mask, preserving the standard DCN equation and mmcv parameter/state-dict surface.

## FlashBEV and FlashSCA pooling

`mu_space.ops.flash_bevpool` and `mu_space.ops.flash_bevpool_sca` are MUSA sampling view-transform
operators backed by `csrc/flashbev/flash_bevpool_musa.mu` and
`csrc/flashbev/flash_bevpool_sca_musa.mu`. Both provide a PyTorch reference path for correctness
comparison and fallback.

Provenance boundary: no third-party source lineage is recorded in these files, and no model recipe
pins an upstream project for them. The workload is a SimpleBEV-style case and consumes an official
nuScenes projection-matrix JSON, which the repository does not redistribute. The kernels are treated
as muSPACE-local implementations pending a provenance review that confirms whether any upstream
project contributed to them.

## SparseEncoder sparse convolution

`mu_space.ops.spconv` exposes the 3-D SubM/SparseConv entry points exercised by BEVFusion's
SparseEncoder, together with the legacy `sparse_conv_ext` shim. Scope is deliberately limited to
that contract; 2-D/4-D, transpose/inverse convolution, fused-BN and sparse maxpool entrypoints are
not exposed.

The operator contract follows BEVFusion's `spconv` bindings (Apache-2.0), which build on
`traveller59/spconv` (Apache-2.0) and MMCV's `sparse_conv_ext`. The MUSA implementation in
`csrc/spconv/spconv_musa.mu` is a local gather/GEMM/scatter-add kernel and does not include
PyTorch extension source from upstream. FP32 GEMMs run in TF32 by default; set
`MUSPACE_SPCONV_FP32_TF32=0` to force strict IEEE float32.

## Hard and dynamic voxelization

`mu_space.ops.hard_voxelize` implements the OpenMMLab mmdetection3d `Voxelization` contract
(Apache-2.0): FP32 points `[N, F]` mapped to `[z, y, x]` voxel coordinates, with a deterministic
first-seen-order path and a parallel path that does not guarantee voxel order. The MUSA kernels in
`csrc/voxelize/hard_voxelize_musa.mu` are a local implementation; no upstream CUDA source is
vendored. On rare FP32 voxel-boundary inputs, MUSA division rounding can place a point in a
neighbouring bin compared with CUDA.

## Points in rotated 3-D boxes

`mu_space.ops.points_in_boxes_all` provides the batch rotated-3D point-in-box mask used by the
BEVFusion Depth path, replacing `points_in_boxes_batch` from BEVFusion/mmdetection3d
(Apache-2.0). The MUSA kernel in `csrc/geometry/points_in_boxes_all_musa.mu` is a local
implementation with its own PyTorch reference path.

## CenterPoint target assignment

`mu_space.ops.centerpoint_target_assign` implements the single-sample, single-head target-building
inner loop from CenterPoint commit `3cf7d870537e287c99b43b68636ea392a5e6f519`
(https://github.com/tianweiy/CenterPoint, MIT). It deliberately has no model, dataset, dataloader
or autograd integration: callers split tasks and batches before calling it.
