# Architecture

muSPACE is organized as a MUSA compatibility and optimized-operator layer for spatial and physical AI
workloads. It does not maintain model forks. Instead, it keeps reusable acceleration code in
`mu_space` and applies explicit, auditable recipes to upstream model checkouts.

## Design Goals

- Keep canonical operator APIs independent from OpenMMLab internals.
- Preserve upstream model repository structure and make local changes reversible.
- Prefer compatibility-first execution for users, with strict native validation available when
  diagnosing release readiness.
- Make backend decisions observable through dispatch reports and logs.
- Keep generated artifacts, external model checkouts, and diagnostic outputs outside public source
  commits.

## Layered Integration Model

The integration stack is intentionally cumulative:

```text
MUSA runtime and torch_musa
  -> canonical muSPACE operator API
    -> OpenMMLab-compatible wrapper
      -> opt-in runtime patch/dispatch replacement
        -> upstream model checkout
```

`torch_musa` registers the MUSA device with PyTorch and enables ordinary tensor and module execution
on MUSA. muSPACE then handles model-specific gaps and optimized kernels that are not covered by the
base PyTorch stack.

The canonical operator layer is the stable boundary. For MSDA, that boundary is
`mu_space.ops.multi_scale_deformable_attn`. Wrappers and patchers may adapt legacy function names or
module globals, but they should not duplicate kernel logic.

## Repository Boundaries

| Path | Responsibility |
| --- | --- |
| `mu_space/ops/` | Public canonical operator APIs and autograd wrappers |
| `mu_space/runtime/` | Backend selection, dispatch explanations, and report state |
| `mu_space/patcher/` | Model and framework compatibility patchers |
| `mu_space/compatibility.py` | Supported stack inspection for `mu-space env` |
| `csrc/ffi/` | MUSA/C++ TVM FFI sources compiled into package-local `libmuspace_ops.so` |
| `model_examples/` | Upstream model recipes: manifests, patches, launch scripts, and model guides |
| `tests/` | Unit and integration coverage for package behavior |

Generated build trees, wheel outputs, validation output, profiling output, and downloaded model
checkouts should remain ignored by git.

## Model Integration Flow

Every model is integrated as an official upstream checkout pinned to one commit. The recipe for
each model is described entirely by `model_examples/<ModelName>/manifest.yaml`, which the package
loads as a `Recipe`: upstream repository, reference branch and commit, default checkout name,
integration metadata, marker filename, and the ordered patch list with a `sha256` per patch.

| Model | Recipe slug | Pinned commit | Marker | Runtime patcher |
| --- | --- | --- | --- | --- |
| BEVFormer | `bevformer` | `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` | `.mu_space_bevformer.json` | `default_patcher` |
| FlashOcc | `flashocc` | `449dc7947d6549ccb72a99d6f6d8c02f166ec0dd` | `.mu_space_flashocc.json` | `flashocc_patcher` |
| UniAD | `uniad` | `7b5bf15e0e49522b6553ddc48e67833e8f5f0f52` | `.mu_space_uniad.json` | `default_patcher` |
| SparseDrive | `sparsedrive` | `52c4c05b6d446b710c8a12eb9fb19d698b33cb2b` | `.mu_space_sparsedrive.json` | `sparsedrive_patcher` |

The normal flow is identical for all four, with `<Model>` and `<slug>` substituted:

1. `model_examples/<Model>/scripts/prepare_checkout.sh ./<Model>` fetches and verifies the pinned
   upstream commit. It only fetches; it never patches, and it exits non-zero when the origin or the
   resulting `HEAD` does not match the manifest.
2. `mu-space migrate <slug> --checkout ./<Model>` applies the patch recipe in manifest order. A
   patch that no longer applies raises `MigrationError` instead of being skipped.
3. The migration writes the marker file in the checkout with patch names, patch digests, touched
   files, and the checkout diff digest.
4. User code calls the recipe's runtime patcher before importing upstream modules that bind the
   legacy operator helper. BEVFormer configurations with `dcn=dict(type='DCNv2', ...)` also call
   `modulated_deform_conv_patcher.apply()` before model construction.
5. The recipe's validation or launch script exercises the patched model path.

The migration path is idempotent. Re-running it on an already migrated checkout returns
`already-applied`; a partially patched checkout returns `upgraded`.
`mu-space restore <slug> --checkout ./<Model>` reverses the patches and removes the marker when the
checkout is in a known state.

Two recipes deviate from the common shape, and both deviations are recorded in the manifest rather
than in code:

- BEVFormer carries a separate `dependency_patches:` key for the legacy mmcv/mmdet3d sdist patches.
  Those are applied with `patch -p1` by `build_openmmlab_compat_wheels.sh` against extracted
  source distributions, not to the model checkout, so `mu-space migrate` never sees them.
- SparseDrive ships an empty `patches:` list. Its integration is runtime-only, so migration writes
  a marker without modifying the checkout.

Patch files are byte-exact inputs to `git apply`. Whitespace normalization silently breaks context
matching, so `.pre-commit-config.yaml` excludes `*.patch` from the whitespace-fixing hooks.

## Runtime Dispatch and Fallback

MSDA supports three requested backend modes:

| Backend | Behavior |
| --- | --- |
| `auto` | Use native MUSA for MUSA tensors when available; fall back to PyTorch reference if native execution is unavailable or fails. |
| `musa` | Require native MUSA execution and raise on unsupported devices or native failures. |
| `reference` | Use the PyTorch reference implementation explicitly. |

Each call records a `DispatchReport` that includes the requested backend, selected backend, device,
native availability, reason, and fallback information. Users can inspect it with
`get_last_dispatch_report()` or enable `MUSPACE_DISPATCH_LOG=1` to print reports to stderr.

Compatibility-first default behavior is intended for user workflows that should continue running
when a native kernel is unavailable. Strict mode is intended for validation because it turns native
unavailability into an explicit failure.

## Validation Strategy

muSPACE uses layered validation:

- Package tests cover CLI behavior, compatibility reports, migration idempotency across every
  recipe, runtime patching and release, and operator shape/dispatch behavior.
- MUSA tests compare native operators against their PyTorch reference paths when a MUSA device is
  available.
- Patch integrity checks confirm each recipe's patches still apply to a pristine pinned checkout and
  that every patch matches its manifest digest.
- BEVFormer fixture validation runs one small synthetic optimizer step and checks finite gradients,
  optimizer progress, and loss structure.
- FlashOcc and UniAD launch checks run one training iteration on one scene and print a single
  `*_train_step_ok` line with a finite loss. They prove the path launches; they measure nothing.
- Fixture profiling records timing, peak memory, dispatch logs, and optional forward traces for
  triage. It is not a release benchmark.

This separation keeps correctness checks fast while making hardware-dependent validation explicit.
Launch checks and profiling output must never be quoted as performance data.

## Extending muSPACE

When adding a new operator:

1. Define the canonical API under `mu_space/ops/`.
2. Add native extension code under `csrc/` only when needed.
3. Preserve or implement a reference path for correctness comparison and fallback.
4. Record dispatch decisions through `mu_space.runtime`.
5. Add CPU/reference tests and MUSA native comparison tests with `skipif` guards.
6. Update `docs/operator_matrix.md` and source provenance notes.

When adding a new model:

1. Add a `model_examples/<ModelName>/` directory with a manifest, patch recipe, checkout script,
   launch script, and model README.
2. Register the recipe slug in `RECIPE_DIRECTORIES` in `mu_space/patcher/migration.py`. Migration,
   restore, and the CLI model choices are driven by the manifest, so no per-model migration code is
   needed.
3. Keep patches small, ordered, and reversible, and record a `sha256` for each one in the manifest.
4. Add a runtime patcher under `mu_space/patcher/` only when the model needs an operator rebind that
   a checkout patch cannot express.
5. Provide fixture validation or a training launch check before claiming model support, and describe
   in the model README exactly what that check does and does not prove.
6. Keep profiling output diagnostic unless benchmark methodology is documented.

New integrations should follow the same boundary: canonical operators stay reusable, model adapters
stay thin, and upstream checkouts remain external to the source tree.
