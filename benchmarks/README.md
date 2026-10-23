# muSPACE operator benchmarks

Each operator owns its benchmark case, runner, reference-platform evidence, and published
measurements. The rules that make those records comparable live in
[docs/benchmarking.md](../docs/benchmarking.md) and are normative for this directory: measurement
protocol, case construction, README structure, result table, reference-platform statement, acceptance
criterion, and the privacy rules for published files.

`benchmarks/check_benchmarks.py` enforces the mechanical part of those rules:

```bash
python benchmarks/check_benchmarks.py --check
python benchmarks/check_benchmarks.py --check --records .tmp/benchmarks
```

## Coverage

Every operator in [docs/operator_matrix.md](../docs/operator_matrix.md) has exactly one benchmark
directory. The mapping is:

| Operator | Benchmark | Entry point |
| --- | --- | --- |
| Multi-scale deformable attention | [msda/](msda/README.md) | `mu_space.ops.multi_scale_deformable_attn` |
| BEVPool | [bev_pool/](bev_pool/README.md) | `mu_space.ops.bev_pool` |
| BEVDepth voxel pooling (train) | [voxel_pooling_train/](voxel_pooling_train/README.md) | `mu_space.ops.voxel_pooling_train` |
| Depth-aware voxel pooling | [voxel_pool_depth/](voxel_pool_depth/README.md) | `mu_space.ops.voxel_pool_depth` |
| BEVPoolV2 | [bev_pool_v2/](bev_pool_v2/README.md) | `mu_space.ops.bev_pool_v2` |
| Deformable aggregation | [deformable_aggregation/](deformable_aggregation/README.md) | `mu_space.ops.deformable_aggregation` |
| Modulated deformable convolution (DCNv2) | [dcnv2/](dcnv2/README.md) | `mu_space.ops.modulated_deform_conv2d` |
| Standard deformable convolution (DCN) | [deform_conv/](deform_conv/README.md) | `mu_space.ops.DeformConv2d` |
| FlashBEV pooling | [flashbev/](flashbev/README.md) | `mu_space.ops.flash_bevpool` |
| FlashSCA pooling | [flashbev/](flashbev/README.md) | `mu_space.ops.flash_bevpool_sca` |
| Sigmoid focal loss | [sigmoid_focal_loss/](sigmoid_focal_loss/README.md) | `mu_space.ops.sigmoid_focal_loss` |
| BEVFusion SparseEncoder spconv | [spconv/](spconv/README.md) | `mu_space.ops.spconv`, legacy `sparse_conv_ext` shim |
| BEVFusion BEV IoU / NMS | [boxes_iou_bev/](boxes_iou_bev/README.md) | `mu_space.ops.boxes_iou_bev`, `nms_bev`, `nms_bev_normal` |
| Hard voxelize | [hard_voxelize/](hard_voxelize/README.md) | `mu_space.ops.hard_voxelize` |
| Points in rotated 3-D boxes | [points_in_boxes_all/](points_in_boxes_all/README.md) | `mu_space.ops.points_in_boxes_all` |
| Radius neighborhood query | [radius/](radius/README.md) | `mu_space.ops.radius`, `radius_graph` |
| QCNet radius integration | [radius/](radius/README.md) | `mu_space.qcnet_compat` |
| CenterPoint target assignment | [centerpoint_target_assign/](centerpoint_target_assign/README.md) | `mu_space.ops.centerpoint_target_assign` |

## Comparison rules

All published comparisons use FP32 latency. S5000 has 1600 GB/s nominal memory bandwidth; each
operator README states its reference device and applies the same indicative normalization:

```text
S5000 normalized latency = S5000 measured latency x 1600 / reference bandwidth
Speedup                  = reference latency / S5000 normalized latency
```

Values above `1.00x` favor S5000. This is a nominal-bandwidth adjustment, not a roofline model, and it
never makes two rows with different scopes comparable.

Registered devices:

| Device | Role | Nominal memory bandwidth |
| --- | --- | ---: |
| MTT S5000 | measured device | 1600 GB/s |
| NVIDIA A100 80GB PCIe | reference device | 1935 GB/s |

A new reference device is registered here, with its nominal bandwidth, before any row measured on it
is published.

## Reproducing benchmark inputs

Published measurements are not redistributed with this repository. Every case builds its own
deterministic inputs from a fixed seed, documented in the operator README together with the exact
command. Captured artifacts, calibration files, and data sets that the repository does not
redistribute are not part of the published path; where an operator can also replay a captured
artifact through `--input-file` for local investigation, the operator README says so.

Run any benchmark from the repository root, for example:

```bash
PYTHONPATH=. python benchmarks/msda/benchmark_msda.py --case bevformer-1l --warmup 20 \
  --iterations 100 --repeats 5 --output .tmp/benchmarks/msda-bevformer-1l.json
```

Scratch output belongs in a git-ignored location such as `.tmp/`, never in the tracked tree.
