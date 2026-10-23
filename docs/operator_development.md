# Operator Development Guide

This guide describes the expected structure for adding a new muSPACE operator. Keep the canonical
operator reusable and place model-specific adaptation in `model_examples/` or `mu_space/patcher/`.

## Source Layout

Use one directory per operator family under `csrc/`:

| Path | Purpose |
| --- | --- |
| `csrc/<family>/<operator>_musa.mu` | MUSA/TVM FFI launcher and kernels |
| `csrc/<family>/*.muh` | Shared device helpers or kernel headers |
| `mu_space/ops/<operator>.py` | Public Python API, autograd wrapper, reference path, dispatch |
| `tests/ops/test_<operator>.py` | Unit and MUSA comparison tests |
| `model_examples/<ModelName>/` | Optional model recipe, patches, scripts, and README |

Add new `.mu` files to `EXTENSION_SOURCES` in `build_utils.py`. Shared FFI helpers belong under
`csrc/ffi/`.

`csrc/ffi/muspace_ops.mu` contains the `add_one` build smoke entrypoint used to verify FFI
compilation and DLPack binding. It is not a public operator and should not be listed in the
operator support matrix.

## Native Extension Requirements

Native entrypoints should validate device, dtype, contiguity, shape, and index tensor types before
launching kernels. Prefer explicit `TVM_FFI_ICHECK` messages that name the offending argument. Keep
launch policy local to the operator source and make profiling-only overrides opt-in through clearly
named environment variables.

Use float/index conventions that match the model interface. For example, BEV-style operators
usually require `float32` tensors and `int32` ranks or interval arrays.

## Python API and Dispatch

Expose the stable API from `mu_space/ops/<operator>.py` and re-export it from `mu_space/ops/__init__.py`
and `mu_space/__init__.py` when it is user-facing. Provide:

- a PyTorch reference implementation for correctness comparison and fallback;
- a native autograd wrapper when backward is implemented natively;
- `backend="auto" | "musa" | "reference"` behavior when dispatch is relevant;
- dispatch reports through `mu_space.runtime.kernel_dispatch`.

Default mode should be compatibility-first. Strict validation should request `backend="musa"` or use
a script flag that fails when native execution is unavailable.

## Tests

Add CPU/reference tests for shape validation, dtype errors, and deterministic outputs. Add MUSA tests
with `pytest.mark.skipif` guards so the suite still passes without MUSA hardware. Native tests should
compare against the reference path with tolerances documented in the test.

Run at least:

```bash
python -m py_compile mu_space/ops/<operator>.py
python -m pytest tests/ops/test_<operator>.py -q
python -m pytest -q
```

When the native AOT library changes, rebuild before hardware validation:

```bash
MUSA_ARCH=31 python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist
python -m pip install --no-deps --force-reinstall dist/mu_space-*.whl
```

## Model Integration

Keep model patches thin. A model recipe should call the canonical operator API rather than importing
extension internals. Add a model README with the supported upstream commit, required patches,
validation commands, and known limitations. Do not commit external checkouts, generated logs,
profiling outputs, or temporary benchmark data.

Model-specific scripts must live under `model_examples/<ModelName>/scripts/` from the start. Use the
top-level `scripts/` directory only for cross-project diagnostics or maintenance helpers.

## Documentation Checklist

Before submitting an operator, update:

- `docs/operator_matrix.md` with native/reference/model support;
- `docs/provenance.md` when code is adapted from upstream projects;
- `README.md` only when the operator is part of the public support surface;
- model example docs when a model workflow depends on the operator;
- the operator benchmark under `benchmarks/` when performance is published, following the
  [benchmark standard](benchmarking.md).
