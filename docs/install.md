# Installation

## Supported Stack

The supported stack is:

| Platform | MUSA SDK / Toolkit | PyTorch | torch_musa | Python | Architecture |
| --- | --- | --- | --- | --- | --- |
| Linux x86_64 / MTT S5000 | 4.3.8 | 2.7.1 | 2.7.1 | 3.10 | `MUSA_ARCH=31` |
| Linux x86_64 / MTT S5000 | 5.2.0 | 2.9.1 | 2.9.1 | 3.10 | `MUSA_ARCH=31` |
| Linux x86_64 / MTT S5000 | 5.2.0 | 2.11.0 | 2.11.0 | 3.10 | `MUSA_ARCH=31` |

Model recipes pin one upstream commit each:

| Recipe slug | Upstream commit |
| --- | --- |
| `bevfusion` | `326653dc06e0938edf1aae7d01efcd158ba83de5` |
| `bevformer` | `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` |
| `flashocc` | `449dc7947d6549ccb72a99d6f6d8c02f166ec0dd` |
| `uniad` | `7b5bf15e0e49522b6553ddc48e67833e8f5f0f52` |
| `sparsedrive` | `fbadea693cbef3f3daea7705e521c0dd3321605c` |
| `qwendrive` | `28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0` |

PyTorch and torch_musa must use matching versions from the same SDK row. The wheel tag does not
encode the MUSA device, SDK, Toolkit, or torch_musa version; use the matrix and
`mu-space env --strict` to validate the runtime stack.

The package metadata accepts Python `>=3.10,<4`, while the v0.1.0 release matrix above is
validated with Python 3.10. The platform wheel uses `py3-none-linux_x86_64`: the bundled native
library is platform-specific but does not link against the CPython ABI.

## Install muSPACE

Build and install the wheel inside a MUSA development image. Both the build environment and the
runtime environment require the MUSA-enabled `apache-tvm-ffi>=0.1.9.post3` distribution. Install
it from the documented package index (this works from environments with access to that index):

```bash
pip install "apache-tvm-ffi>=0.1.9.post3" --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple
```

Verify that it is available before using the AOT wheel:

```bash
python - <<'PY'
import importlib.metadata
import tvm_ffi
from packaging.version import Version

version = importlib.metadata.version("apache-tvm-ffi")
assert Version(version) >= Version("0.1.9.post3"), version
print("apache-tvm-ffi:", version)
print("tvm_ffi:", tvm_ffi.__file__)
PY
```

Then build and install:

```bash
cd muSPACE
export MUSA_ARCH=31
rm -rf build dist
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
pip install --no-deps --force-reinstall dist/mu_space-*.whl
```

The wheel contains an AOT TVM FFI/DLPack native library; runtime JIT compilation is intentionally
not supported. For local development, rebuild and reinstall the wheel after native changes:

```bash
export MUSA_ARCH=31
python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
pip install --no-deps --force-reinstall dist/mu_space-*.whl
python -m mu_space
mu-space env --profile core --strict
```

## Prepare a Model Checkout

muSPACE maintains no model forks. Every recipe uses the same two steps: `prepare_checkout.sh`
fetches and verifies the pinned commit, and `mu-space migrate` applies the patch recipe and writes
the migration marker.

```bash
model_examples/BEVFormer/scripts/prepare_checkout.sh ./BEVFormer
mu-space migrate bevformer --checkout ./BEVFormer
```

The same pair works for `flashocc`, `uniad`, `sparsedrive`, and `qwendrive` with their own recipe
directory and checkout path. `prepare_checkout.sh` never patches: it exits non-zero when the origin
or the resulting `HEAD` does not match the manifest, so a stale or wrong checkout fails before
migration.
`mu-space restore <slug> --checkout <path>` reverses the recipe.

BEVFormer additionally needs the legacy OpenMMLab compatibility wheels. Build them, then install
them from the local wheelhouse:

```bash
model_examples/BEVFormer/scripts/build_openmmlab_compat_wheels.sh
pip install --no-deps wheelhouse/mmcv_full-1.4.0-*.whl
pip install --no-deps wheelhouse/mmdet3d-0.17.1-*.whl
pip install --no-deps mmdet==2.14.0 mmsegmentation==0.14.1
pip install -r model_examples/BEVFormer/requirements.lock --no-deps
```

Some optional mmdet3d evaluation imports are still pulled in by the legacy package. If your base
image does not already provide them, install:

```bash
pip install lyft_dataset_sdk nuscenes-devkit plyfile pyquaternion shapely
```

Do not allow pip to downgrade NumPy, Numba, NetworkX, or Trimesh to mmdet3d's Python-3.6-era pins.
The lock file intentionally keeps Python-3.10-compatible versions.

## Runtime Patching

Call the runtime patcher before importing model modules that bind the legacy operator helper.
`default_patcher` replaces mmcv's MSDA helper and serves both BEVFormer and UniAD:

```python
from mu_space.patcher import default_patcher

default_patcher.apply()
```

FlashOcc, SparseDrive, and Qwen-Drive use their own patchers for their own operators. Qwen-Drive
also needs the checkout `src` directory on `PYTHONPATH` before `apply()`:

```python
from mu_space.patcher import flashocc_patcher, qwendrive_patcher, sparsedrive_patcher

flashocc_patcher.apply()
sparsedrive_patcher.apply()
qwendrive_patcher.apply()
```

Default mode is compatibility-first: MUSA tensors use the native muSPACE operator when available and
fall back to the PyTorch reference path otherwise. A configuration whose backbone declares
`dcn=dict(type='DCNv2', ...)` also needs `modulated_deform_conv_patcher.apply()`; see the
[BEVFormer guide](../model_examples/BEVFormer/README.md). Use strict mode for native-path validation:

```bash
MUSPACE_STRICT_PATCH=1 python your_model_entry.py
```

## Quick Validation

BEVFormer runs a one-command fixture validation:

```bash
model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer
```

Expected result: JSON output with `finite_gradients: true` and `optimizer_step: true`.
Strict native-path validation is available with:

```bash
model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer --strict
```

See `model_examples/BEVFormer/fixture_validation.md` for acceptance criteria and troubleshooting.

QCNet uses the same migration workflow and native MUSA `radius`/`radius_graph` dispatch:

```bash
model_examples/QCNet/scripts/prepare_checkout.sh ./QCNet
mu-space migrate qcnet --checkout ./QCNet
```

Run the bounded real-AV2 training check with the QCNet model guide.

FlashOcc and UniAD run one training iteration on one scene:

```bash
bash model_examples/FlashOCC/scripts/train_1p_occ_smoke.sh --checkout ./FlashOCC
bash model_examples/UniAD/scripts/train_1p_stage1_smoke.sh --checkout ./UniAD
```

Each prints a single `*_train_step_ok` line with a finite loss. These are launch checks: they prove
the path runs end to end and measure nothing. See each model README for dataset layout and the
limitations of the check.

Qwen-Drive runs one frozen-backbone perception-head step. Fixture mode needs no VLM weights:

```bash
python model_examples/QwenDrive/scripts/train_step_musa.py \
  --checkout ./QwenDrive --mode fixture --device musa:0
```

A passing run prints JSON whose `status` is `qwendrive_perception_train_step_ok`. It is one step,
not a convergence or throughput result. See `model_examples/QwenDrive/fixture_validation.md`.
SparseDrive needs its info files and anchors first, then runs the same kind of single-scene launch
check and additionally asserts strict native dispatch:

```bash
model_examples/SparseDrive/scripts/prepare_dataset.sh --checkout ./SparseDrive --data-root ./SparseDrive/data/nuscenes
model_examples/SparseDrive/scripts/prepare_anchors.sh --checkout ./SparseDrive
model_examples/SparseDrive/scripts/validate_fixture.sh --checkout ./SparseDrive --data-root ./SparseDrive/data/nuscenes
```

See `model_examples/SparseDrive/fixture_validation.md` for the acceptance criteria and the
prerequisites, including the `flash_attn` import the pinned model package requires.
