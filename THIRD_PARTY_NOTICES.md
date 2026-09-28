# Third-party notices

muSPACE is distributed under the Apache License 2.0. Parts of this repository are derived from,
or adapt the interface of, the third-party projects listed below. The complete license texts —
including the MIT and BSD 3-Clause texts that must accompany source and binary distribution — are
reproduced in [`NOTICE`](NOTICE). Per-file copyright headers are retained in the kernel sources.

## Code derived into this repository

| Project | Source | License | muSPACE location |
| --- | --- | --- | --- |
| OpenMMLab MMCV | https://github.com/open-mmlab/mmcv | Apache-2.0 | `csrc/ms_deformable_attn/*`, `csrc/deformable_conv/*`, `mu_space/ops/multi_scale_deformable_attn.py`, `mu_space/ops/modulated_deform_conv.py` |
| Deformable DETR (SenseTime) | https://github.com/fundamentalvision/Deformable-DETR | Apache-2.0 | MSDA kernel lineage and headers |
| Deformable-Convolution-V2-PyTorch (DCNv2) | https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch | MIT | `csrc/deformable_conv/modulated_deform_conv_musa_kernel.muh` |
| MMDetection3D | https://github.com/open-mmlab/mmdetection3d | Apache-2.0 | `mu_space/ops/hard_voxelize.py` |
| BEVFusion | https://github.com/mit-han-lab/bevfusion | Apache-2.0 | `csrc/geometry/boxes_iou_bev_musa.mu`, `csrc/pooling/bev_pool_musa.mu`, `csrc/spconv/spconv_musa.mu`, `csrc/geometry/points_in_boxes_all_musa.mu`, `mu_space/ops/sigmoid_focal_loss.py` |
| OpenPCDet | https://github.com/open-mmlab/OpenPCDet | Apache-2.0 | rotated BEV IoU / NMS lineage, inherited through BEVFusion |
| SparseDrive | https://github.com/swc-17/SparseDrive | MIT | `csrc/deformable_aggregation/deformable_aggregation_musa.mu` |
| BEVDepth | https://github.com/Megvii-BaseDetection/BEVDepth | MIT | `csrc/pooling/voxel_pooling_train_musa.mu` |
| CenterPoint | https://github.com/tianweiy/CenterPoint | MIT | `mu_space/ops/centerpoint_target_assign.py` |
| torch-cluster (PyTorch Geometric) | https://github.com/pyg-team/pytorch_geometric | MIT | `csrc/geometry/radius_musa.mu`, `mu_space/ops/radius.py`, `mu_space/qcnet_compat.py` |
| spconv | https://github.com/traveller59/spconv | Apache-2.0 | `mu_space/ops/spconv.py` |
| FlashInfer | https://github.com/flashinfer-ai/flashinfer | Apache-2.0 | `format.sh` |
| tilelang | https://github.com/tile-ai/tilelang | MIT (with a dated collaboration-terms note upstream) | `format.sh` |
| NVIDIA CUTLASS | https://github.com/NVIDIA/cutlass | BSD-3-Clause | fast division / modulo arithmetic in `csrc/ffi/fast_divmod.hpp` |
| PyTorch / torch_musa | https://github.com/pytorch/pytorch, https://github.com/MooreThreads/torch_musa | BSD-3-Clause | FastDivmod lineage in `csrc/ffi/fast_divmod.hpp` |

`MATE` (Moore Threads, Apache-2.0) is the other origin of `format.sh`; it is a Moore Threads
project rather than a third-party dependency.

## Upstream model recipe targets

`model_examples/` ships patch recipes against official upstream checkouts pinned to a fixed
commit. muSPACE maintains no forks; the patches contain upstream context lines and retain the
upstream attribution. See [`docs/provenance.md`](docs/provenance.md) for the pinned commits.

| Project | Source | License |
| --- | --- | --- |
| BEVFormer | https://github.com/fundamentalvision/BEVFormer | Apache-2.0 |
| BEVFusion | https://github.com/mit-han-lab/bevfusion | Apache-2.0 |
| BEVDepth | https://github.com/Megvii-BaseDetection/BEVDepth | MIT |
| DriveDreamer | https://github.com/JeffWang987/DriveDreamer | Apache-2.0 |
| FlashOCC | https://github.com/Yzichen/FlashOCC | Apache-2.0 |
| lingbot-va | https://github.com/Robbyant/lingbot-va | Apache-2.0 |
| MapTR | https://github.com/hustvl/MapTR | MIT |
| OpenVLA | https://github.com/openvla/openvla | MIT |
| QCNet | https://github.com/ZikangZhou/QCNet | Apache-2.0 |
| SparseDrive | https://github.com/swc-17/SparseDrive | MIT |
| UniAD | https://github.com/OpenDriveLab/UniAD | Apache-2.0 |

The OpenMMLab compatibility patches target MMCV 1.4.0 and MMDetection3D 0.17.1, both distributed
under Apache License 2.0. Patch files contain only the local compatibility delta.

## Dependencies not redistributed

PyTorch (BSD-3-Clause), apache-tvm-ffi (Apache-2.0), click (BSD-3-Clause), PyYAML (MIT),
packaging (Apache-2.0/BSD), pytest (MIT), ruff (MIT), pre-commit (MIT) and
setuptools/wheel/ninja (MIT/Apache-2.0) are runtime or build dependencies. They are not
redistributed in this repository. torch_musa and the MUSA SDK are Moore Threads products
governed by their own terms.

## Open provenance items

The following operator families are muSPACE-local implementations with no third-party source
lineage recorded in the source files or pinned by a recipe. They are listed here so the boundary
stays explicit; see [`docs/provenance.md`](docs/provenance.md) for details.

- FlashBEV and FlashSCA pooling (`csrc/flashbev/*`, `mu_space/ops/flashbev.py`)
- BEV IoU / NMS and spconv MUSA kernels are ports of the BEVFusion contract rather than copies of
  its CUDA sources.
