# BEVFusion for muSPACE

This recipe runs the official `mit-han-lab/BEVFusion` revision
`326653dc06e0938edf1aae7d01efcd158ba83de5`; muSPACE does not maintain a model fork.
It provides a deterministic one-scene camera-and-LiDAR training launch check, an inference
pass, and an artifact format for MUSA/A100 numerical comparison. It is not full nuScenes
training and does not publish throughput or task accuracy.

## Integration

| Path | Purpose |
| --- | --- |
| `manifest.yaml` | Pinned checkout contract and source-patch digest |
| `patches/` | Reversible sigmoid focal-loss source routing |
| `scripts/prepare_checkout.sh` | Fetches and verifies the official revision |
| `scripts/train_1p_fusion_smoke.sh` | One-card fusion training and inference launcher |
| `scripts/train_fusion_musa.py` | Strict native MUSA regression implementation |
| `scripts/compare_artifacts.py` | MUSA/A100 artifact comparison gate |

Run checkout migration before importing the model:

```bash
model_examples/BEVFusion/scripts/prepare_checkout.sh ./BEVFusion
mu-space migrate bevfusion --checkout ./BEVFusion
```

The migration only rewrites upstream `sigmoid_focal_loss`. The runtime launcher installs the
BEV pool, batch `points_in_boxes_all`, and iou3d/NMS patchers before model construction. The
points-in-box replacement covers `DepthInstance3DBoxes.points_in_boxes()` and preserves the
`[B, N, M]` mask contract. The required 3-D SparseEncoder and
voxelization compatibility paths remain strict native requirements: a missing replacement or a
reference fallback is a failure, not a compatibility success.

## Data and launch check

Mount the same complete nuScenes trainval metadata and the camera, LiDAR, and annotation files
into both comparison environments. The default scene is `scene-0001`; choose another scene only
explicitly. Prepared `*infos_train*.pkl` metadata must be present at the data root.

```bash
bash model_examples/BEVFusion/scripts/train_1p_fusion_smoke.sh \
  --checkout ./BEVFusion --data-root ./BEVFusion/data/nuscenes \
  --scene-name scene-0001 --artifact .tmp/bevfusion-musa-fp32.pt
```

The launcher defaults to `configs/nuscenes/seg/fusion-bev256d2-lss.yaml`, one iteration,
`musa:0`, `MUSPACE_STRICT_PATCH=1`, and strict FP32 sparse-convolution math. It writes filtered
metadata only below `--work-dir`; no dataset or checkpoint is modified. A successful JSON result
requires finite loss and gradients, a changed parameter, and a native MUSA dispatch report.

The camera-map fixture remains available through `validate_fixture.sh`; it specifically covers
the patched sigmoid focal loss. The fusion launcher covers the actual one-scene detection graph
and its postprocessing path.

## A100 comparison

Run the same pinned revision, seed, batch serialization policy, and scene in the verified A100
container, but keep that checkout unmodified by `mu-space migrate`. Disable CUDA TF32 there and
run `train_fusion_musa.py --device cuda:0 --artifact <cuda-artifact>`; it emits the identical
artifact schema, then compare:

```bash
python model_examples/BEVFusion/scripts/compare_artifacts.py \
  --musa .tmp/bevfusion-musa-fp32.pt --a100 .tmp/bevfusion-a100-fp32.pt --dtype float32
```

FP32 uses `rtol=atol=1e-3`; FP16 uses `2e-2`. The compared artifact checks the pinned revision,
scene, seed, post-step parameter, and loss. A failed comparison is a regression failure. Keep
all raw artifacts outside tracked source.
