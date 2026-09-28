# muSPACE

muSPACE (**MUSA Spatial and Physical AI Compute Engine**) is a MUSA compatibility and
optimized-operator layer for spatial and physical AI and autonomous-driving workloads. It provides
canonical MUSA operator APIs, thin framework compatibility adapters, and reproducible model
migration recipes.

The repository provides launch recipes for official BEVFormer, BEVFusion, FlashOcc, OpenVLA, UniAD, QCNet, and
SparseDrive
checkouts. muSPACE does not maintain model forks; it prepares official upstream checkouts with
explicit compatibility patches and runtime replacement hooks.

## Highlights

- Canonical `mu_space.ops.multi_scale_deformable_attn` API with MUSA native and PyTorch reference paths
- Canonical `mu_space.ops.bev_pool_v2` API for BEV-style feature pooling
- Canonical `mu_space.ops.deformable_aggregation` API for SparseDrive multi-camera feature aggregation
- Canonical `mu_space.ops.modulated_deform_conv2d` API (DCNv2) with an offset-conv-fused NHWC path
- `mu-space migrate` and `mu-space restore` for every pinned upstream checkout
- Runtime patcher that replaces BEVFormer's and UniAD's MSDA helper through `default_patcher.apply()`
- Runtime patcher that installs the muSPACE DCNv2 layer in mmcv's `CONV_LAYERS` registry
- FlashOcc BEVPoolV2 launch recipe through `flashocc_patcher.apply()`
- SparseDrive deformable aggregation through `sparsedrive_patcher.apply()`
- Dispatch reports for confirming native, reference, and fallback execution
- Fixture validation and profiling scripts for repeatable correctness and triage runs
- Batch rotated-3D point-in-box masks for BEVFusion Depth box processing

## Requirements

| Component | Requirement |
| --- | --- |
| Python | `>=3.10,<4` (validated: `3.10`) |
| MUSA SDK / PyTorch / torch_musa | `4.3.8` + `2.7.1` + `2.7.1`, or `5.2.0` + `2.9.1` + `2.9.1`, or `5.2.0` + `2.11.0` + `2.11.0` |
| Architecture | MTT S5000, `mp_31` |
| Model recipes | BEVFormer, BEVFusion, FlashOcc, OpenVLA, UniAD, QCNet, SparseDrive |

The AOT native library uses TVM FFI and DLPack rather than the PyTorch C++ ABI, but the supported
MUSA SDK and PyTorch/torch_musa combinations remain version-bound. Use
`mu-space env --profile core` for the core check and `mu-space env --profile bevformer` for the
pinned model recipe. Add `--strict` to fail when the SDK, versions, or MUSA device are not
validated. Building or loading the AOT library requires the MUSA-enabled
`apache-tvm-ffi>=0.1.9.post3` distribution in the active Python environment.
PyTorch and torch_musa must use the exact matching versions from one supported SDK row; other
combinations are not declared supported by v0.0.3.

## Quick Start

Build and install muSPACE in a configured MUSA development environment:

```bash
cd muSPACE
export MUSA_ARCH=31

# Install the MUSA-enabled TVM FFI runtime (if not already present). This
# documented package index can be used from any environment that has access
# to the index; it is not bundled into the muSPACE wheel.
pip install "apache-tvm-ffi>=0.1.9.post3" --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

# Verify the installation.
python - <<'PY'
import importlib.metadata
import tvm_ffi
from packaging.version import Version

version = importlib.metadata.version("apache-tvm-ffi")
assert Version(version) >= Version("0.1.9.post3"), version
print("apache-tvm-ffi:", version)
print("tvm_ffi:", tvm_ffi.__file__)
PY

python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
python -m pip install --force-reinstall --no-deps dist/mu_space-*.whl
mu-space env --profile core --strict
python -m pytest -q
```

The package metadata requires Python `>=3.10,<4`; v0.1.0 release validation
uses Python `3.10`. The platform wheel uses a `py3-none-linux_x86_64` tag
because its native library is not linked to the CPython ABI. Install the exact
`apache-tvm-ffi>=0.1.9.post3` distribution from the documented package index
before installing muSPACE with `--no-deps`, so pip does not replace the
matched PyTorch/torch_musa stack. If the index is unreachable, resolve that
environment access issue before installing the wheel.

Prepare an official checkout and apply its compatibility recipe. Every recipe follows the same
two steps — `prepare_checkout.sh` fetches and verifies the pinned commit, `mu-space migrate`
applies the patches and writes the migration marker:

```bash
model_examples/BEVFormer/scripts/prepare_checkout.sh ./BEVFormer
mu-space migrate bevformer --checkout ./BEVFormer
```

The same pair works for `flashocc`, `openvla`, `uniad`, and `sparsedrive` with their own recipe directory
and checkout path. `mu-space restore <model> --checkout <path>` reverses the recipe.

Build and install the legacy OpenMMLab compatibility stack:

```bash
model_examples/BEVFormer/scripts/build_openmmlab_compat_wheels.sh
pip install --no-deps wheelhouse/mmcv_full-1.4.0-*.whl
pip install --no-deps wheelhouse/mmdet3d-0.17.1-*.whl
pip install --no-deps mmdet==2.14.0 mmsegmentation==0.14.1
pip install -r model_examples/BEVFormer/requirements.lock --no-deps
```

Do not let pip downgrade NumPy, Numba, NetworkX, or Trimesh to legacy mmdet3d pins. If optional
mmdet3d evaluation imports are missing, install `lyft_dataset_sdk`, `nuscenes-devkit`, `plyfile`,
`pyquaternion`, and `shapely`.

Validate the fixture path:

```bash
model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer
model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer --strict
```

The default validation is compatibility-first and may use the PyTorch reference fallback. The
`--strict` run requires the native MUSA MSDA path.

## Architecture

muSPACE keeps model-specific code thin and places reusable acceleration logic in the package:

```text
canonical operator API
  -> OpenMMLab-compatible wrapper
    -> opt-in patch/dispatch replacement
```

`mu_space` owns MUSA kernels, TVM FFI/DLPack bindings, autograd integration, validation, fallback policy,
and dispatch reporting. Model recipes adapt official upstream repositories to those APIs without
maintaining forks.

## Operator Support

| Operator | Canonical API | MUSA native | Reference | Model integration |
| --- | --- | --- | --- | --- |
| multi-scale deformable attention | `mu_space.ops.multi_scale_deformable_attn` | available for MUSA tensors | PyTorch reference fallback | BEVFormer and UniAD runtime patcher |
| BEVPoolV2 | `mu_space.ops.bev_pool_v2` | available for MUSA tensors | PyTorch reference fallback | FlashOcc runtime patcher |
| BEVDepth voxel pooling (train) | `mu_space.ops.voxel_pooling_train` | available for MUSA tensors | PyTorch reference fallback | BEVDepth runtime patcher |
| BEVFusion SparseEncoder spconv | `mu_space.ops.spconv` / legacy `sparse_conv_ext` shim | 3-D SubM/SparseConv FP32/FP16 forward and backward | PyTorch reference implementation | BEVFusion LiDAR SparseEncoder |
| deformable aggregation | `mu_space.ops.deformable_aggregation` | available for MUSA tensors | PyTorch reference fallback | SparseDrive runtime patcher |
| modulated deformable conv (DCNv2) | `mu_space.ops.modulated_deform_conv2d` | required, MUSA only | test-only golden model | BEVFormer ResNet-DCN runtime patcher |
| FlashBEV pooling | `mu_space.ops.flash_bevpool` | available for MUSA tensors | PyTorch reference fallback | standalone operator API |
| FlashSCA pooling | `mu_space.ops.flash_bevpool_sca` | available for MUSA tensors | PyTorch reference fallback | standalone operator API |
| BEVFusion BEV IoU/NMS | `mu_space.ops.boxes_iou_bev`, `nms_bev`, `nms_bev_normal` | available for MUSA tensors | PyTorch reference fallback | BEVFusion iou3d patcher |
| Hard voxelize | `mu_space.ops.hard_voxelize` | available for MUSA tensors | PyTorch reference fallback | mmdet3d voxelization compatibility |
| points in rotated 3-D boxes | `mu_space.ops.points_in_boxes_all` | available for MUSA tensors | PyTorch reference fallback | BEVFusion Depth batch point-in-box path |
| radius neighborhood query | `mu_space.ops.radius`, `radius_graph` | FP32, D < 9, sorted non-negative batches, int32 indexing range | `radius_reference`, `radius_graph_reference` | standalone operator API |

DCNv2 is native-only: the deformable columns are the whole operator, so a PyTorch emulation
would silently replace a production kernel with something orders of magnitude slower. Calls on
non-MUSA tensors raise with a dispatch report instead of falling back. The differentiable golden
model used to validate it lives in `tests/ops/test_modulated_deform_conv.py`.

Legacy `mmcv` and `mmdet3d` native extensions are intentionally disabled in the compatibility
wheels. They are not muSPACE operators and should fail explicitly if called.

## Runtime Patcher

Call the runtime patcher before importing BEVFormer modules that bind mmcv's MSDA helper:

```python
from mu_space.patcher import default_patcher

default_patcher.apply()
```

Default mode is compatibility-first. MUSA tensors use the native muSPACE MSDA operator when
available and fall back to the PyTorch reference path otherwise. Set `MUSPACE_STRICT_PATCH=1` for
native-path validation.

For SparseDrive deformable aggregation:

```python
from mu_space.patcher import sparsedrive_patcher

sparsedrive_patcher.apply()
```

Backbones that declare `dcn=dict(type='DCNv2', ...)` — BEVFormer base and small — resolve that
config entry through mmcv's `CONV_LAYERS` registry, so a second patcher installs the muSPACE
DCNv2 layer:

```python
from mu_space.patcher import modulated_deform_conv_patcher

modulated_deform_conv_patcher.apply()
```

Apply it before the model is built. muSPACE's module keeps mmcv's constructor signature and
state dict layout, so checkpoints load unchanged. DCNv2 has no compatibility fallback, so this
patcher's strict flag is recorded but does not change execution.

## Validation

Model-specific validation entrypoints live with each model recipe.

| Model | Primary validation | Details |
| --- | --- | --- |
| BEVFormer | `model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer` | [BEVFormer guide](model_examples/BEVFormer/README.md), [fixture validation](model_examples/BEVFormer/fixture_validation.md), [fixture profiling](model_examples/BEVFormer/fixture_profiling.md) |
| BEVFusion | `bash model_examples/BEVFusion/scripts/train_1p_fusion_smoke.sh --checkout ./BEVFusion` | [BEVFusion guide](model_examples/BEVFusion/README.md) |

| FlashOcc | `bash model_examples/FlashOCC/scripts/train_1p_occ_smoke.sh --checkout ./FlashOCC` | [FlashOcc guide](model_examples/FlashOCC/README.md) |
| UniAD | `bash model_examples/UniAD/scripts/train_1p_stage1_smoke.sh --checkout ./UniAD` | [UniAD guide](model_examples/UniAD/README.md) |
| OpenVLA | `bash model_examples/OpenVLA/scripts/validate_fixture.sh --checkout ./OpenVLA` | [OpenVLA guide](model_examples/OpenVLA/README.md) |
| LingBotVA | `bash model_examples/LingBotVA/scripts/validate_fixture.sh --checkout ./LingBotVA` | [LingBotVA guide](model_examples/LingBotVA/README.md), [fixture validation](model_examples/LingBotVA/fixture_validation.md) |
| QCNet | `bash model_examples/QCNet/scripts/validate_fixture.sh --checkout ./QCNet --data-root <AV2_SUBSET_ROOT>` | [QCNet guide](model_examples/QCNet/README.md), [validation contract](model_examples/QCNet/fixture_validation.md) |
| SparseDrive | operator-level integration only; no training launch check | [SparseDrive guide](model_examples/SparseDrive/README.md) |

Validation scripts are correctness and integration launch checks, not benchmarks. The FlashOcc and
UniAD launchers run one training iteration on one scene and print a single `*_train_step_ok` line
with a finite loss; their timings must not be quoted as performance data.

## Dispatch Diagnostics

Inspect the most recent MSDA backend decision from Python:

```python
from mu_space import get_last_dispatch_report

print(get_last_dispatch_report().as_dict())
```

Set `MUSPACE_DISPATCH_LOG=1` to print dispatch reports to stderr. Use
`selected_backend: "musa"` to confirm native execution. Auto fallback appears as
`selected_backend: "reference"` with `native_fallback: true`.

## Source Provenance

muSPACE's MSDA implementation is derived from OpenMMLab MMCV's MSDA operator and adapted into the
package-local TVM FFI AOT library:

| Source lineage | muSPACE file |
| --- | --- |
| MMCV MSDA MUSA kernel helper code | `csrc/ms_deformable_attn/musa_kernel_helper.muh` |
| MMCV MSDA MUSA kernel implementation | `csrc/ms_deformable_attn/ms_deform_attn_musa_kernel.muh` |
| MSDA FFI launcher | `csrc/ms_deformable_attn/ms_deform_attn_musa.mu` |
| MMCV Python reference path | `mu_space/ops/multi_scale_deformable_attn.py` |

OpenMMLab MMCV and the original Deformable DETR lineage use Apache License 2.0. muSPACE exports
stable function names from `libmuspace_ops.so` and does not reuse `mmcv._ext`.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `mu_space/` | Public Python package, CLI, compatibility checks, patchers, and operator APIs |
| `csrc/` | MUSA/TVM FFI AOT sources |
| `model_examples/BEVFormer/` | Pinned BEVFormer manifest, patch recipe, validation scripts, and model guide |
| `model_examples/FlashOCC/` | Pinned FlashOcc manifest, patch recipe, training launch check, and model guide |
| `model_examples/UniAD/` | Pinned UniAD manifest, patch recipe, training launch check, and model guide |
| `model_examples/OpenVLA/` | Pinned OpenVLA manifest, synthetic training smoke, and model guide |
| `model_examples/LingBotVA/` | Pinned LingBot-VA manifest, MUSA patches, smoke and post-training entrypoints, and model guide |
| `model_examples/QCNet/` | Pinned QCNet manifest, AV2 training smoke, radius integration, and model guide |
| `model_examples/SparseDrive/` | Pinned SparseDrive manifest, checkout script, and model guide |
| `docs/` | Cross-model installation, architecture, operator support, provenance, and contributor docs |
| `tests/` | Unit and integration tests |

Cross-model guidance belongs in `docs/`. Model-specific manifests, patches, scripts, validation
notes, and profiling notes belong under `model_examples/<ModelName>/` so additional model support
can be added without expanding the top-level documentation surface.

## Documentation

The README contains the normal installation, architecture, validation, profiling, and provenance
workflow. The docs directory keeps focused references for maintainers and advanced troubleshooting:

- [Installation](docs/install.md)
- [Architecture](docs/architecture.md)
- [Operator support and dispatch reporting](docs/operator_matrix.md)
- [Operator development guide](docs/operator_development.md)
- [BEVFormer model example](model_examples/BEVFormer/README.md)
- [FlashOcc model example](model_examples/FlashOCC/README.md)
- [UniAD model example](model_examples/UniAD/README.md)
- [OpenVLA model example](model_examples/OpenVLA/README.md)
- [LingBotVA model example](model_examples/LingBotVA/README.md)
- [QCNet model example](model_examples/QCNet/README.md)
- [SparseDrive model example](model_examples/SparseDrive/README.md)
- [Source provenance](docs/provenance.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow, testing expectations, and pull
request guidelines.

## License

muSPACE is distributed under the Apache License 2.0. See [LICENSE](LICENSE) for the license
text, [NOTICE](NOTICE) for the copyright and license declaration (including the full texts of the
third-party licenses), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the third-party
inventory.
