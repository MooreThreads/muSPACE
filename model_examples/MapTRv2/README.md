# MapTRv2

This recipe runs a deterministic one-step MapTRv2 training smoke test on a
real nuScenes mini scene on a MUSA device. It applies muSPACE's MUSA
replacements for BEVPoolV2 and multi-scale deformable attention before
importing the MapTRv2 project.

Prepare the upstream checkout, install the locked Python dependencies in the
muSPACE development container, then run:

```bash
DATA_ROOT=/path/to/nuscenes \
CAN_BUS_ROOT=/path/to/datasets \
./model_examples/MapTRv2/scripts/train_1p_maptrv2_smoke.sh
```

`DATA_ROOT` must contain `v1.0-mini`; `CAN_BUS_ROOT` must contain the
expanded `can_bus/` directory. The default scene is `scene-0061`, batch size
is one, and the script writes training artifacts below `.tmp/maptrv2`.

The locked `numba` and `llvmlite` versions are required for the NumPy 1.26
runtime used by the compatible mmdetection3d release.  Install
`requirements.lock` after the OpenMMLab packages so its LYFT evaluation import
dependency is available during dataset registration.
