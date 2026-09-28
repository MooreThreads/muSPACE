# BEVFormer Fixture Validation

The fixture validation runs one small BEVFormer-tiny optimizer step with synthetic inputs. It is a
correctness smoke test for the muSPACE BEVFormer integration path, not a full training, accuracy, or
performance benchmark.

## Prerequisites

Install muSPACE, prepare the pinned BEVFormer checkout, and apply the migration recipe from
`docs/install.md`. The checkout must be importable and migrated with:

```bash
mu-space migrate bevformer --checkout ./BEVFormer
```

Run inside the validated MUSA environment reported by `mu-space env`.

## One-Command Validation

Default mode is compatibility-first and may fall back to the PyTorch reference path if the native
MUSA operator is unavailable:

```bash
cd muSPACE
model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer
```

Strict mode requires the native MUSA path and fails if it cannot be used:

```bash
model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer --strict
```

To see each MSDA backend decision while the fixture runs, enable dispatch logging:

```bash
MUSPACE_DISPATCH_LOG=1 model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer
```

Results are written to `build/validation/bevformer_fixture/default.json` or
`build/validation/bevformer_fixture/strict.json`. Override the location with `--output-dir`.

## Acceptance Criteria

The validation passes only when the result JSON contains:

- `mode: "fixture"`
- `finite_gradients: true`
- `optimizer_step: true`
- a finite numeric `loss`
- a non-empty `losses` object

Useful diagnostic fields include `device`, `seed`, `elapsed_seconds`, `peak_memory_bytes`,
`backbone_frozen`, and `selected_losses`.

For timing, memory, dispatch logs, and optional forward traces, use
`model_examples/BEVFormer/scripts/profile_fixture.sh`; see `model_examples/BEVFormer/fixture_profiling.md`.

## Troubleshooting

If imports fail, reinstall the legacy OpenMMLab stack and confirm `mu-space env` reports the expected
versions. If BEVFormer files are missing or patched inconsistently, rerun
`mu-space migrate bevformer --checkout ./BEVFormer` and inspect the reported `patch_state`.

If `--device musa` fails before model execution, confirm `torch_musa` is installed and the MUSA
runtime sees the device. If default mode passes but `--strict` fails, the fallback path is working
but the native MSDA path is unavailable or failed; rerun the MSDA operator tests before treating this
as a release issue. If gradients are non-finite, keep the fixture output JSON and rerun with the same
`--seed` for comparison.
