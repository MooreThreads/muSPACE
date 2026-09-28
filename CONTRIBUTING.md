# Contributing to muSPACE

Thank you for improving muSPACE. Keep contributions focused, reproducible, and documented.

## Development Workflow

1. Create a branch for one logical change.
2. Install muSPACE in editable mode:

   ```bash
   export MUSA_ARCH=31
   python -m pip install -e .[dev] --no-build-isolation
   pre-commit install
   ```

3. Install the Git hook once with `pre-commit install`; otherwise `git commit` will not run the
   repository format checks automatically.
4. Run the relevant tests before opening a pull request.
5. Update docs when user-facing behavior, supported stacks, scripts, or operator status changes.

## Testing

Use `python -m pytest` from the repository root:

```bash
python -m pytest -q
python -m pytest tests/ops/test_multi_scale_deformable_attn.py -q
```

Hardware-dependent tests are skipped when MUSA is unavailable. Model-level checks live with each
recipe and must be run in a configured MUSA environment:

```bash
model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer
model_examples/BEVFormer/scripts/validate_fixture.sh --checkout ./BEVFormer --strict
bash model_examples/FlashOCC/scripts/train_1p_occ_smoke.sh --checkout ./FlashOCC
bash model_examples/UniAD/scripts/train_1p_stage1_smoke.sh --checkout ./UniAD
```

The FlashOcc and UniAD launchers run one training iteration on one scene and prove only that the
path launches. Their timings, and profiling script output, must not be presented as release
benchmark results.

## Style

- Use 4-space Python indentation.
- Run `bash format.sh` or `pre-commit run -a` before committing; after `pre-commit install`,
  `git commit` also checks the staged files automatically.
- Python formatting is handled by the Ruff hooks in `.pre-commit-config.yaml`.
- C++, CUDA, and MUSA kernel sources are formatted by the clang-format hook in `.pre-commit-config.yaml`.
- Prefer public package APIs over ad hoc imports from scripts.
- Keep model adapters thin; canonical kernels belong under `mu_space.ops` and `csrc/`.
- Avoid development-phase names in public files and docs. Use descriptive names such as
  `bevformer-02-training-compat.patch` instead of milestone labels.

## Pull Requests

Each pull request should include:

- Summary of the change and motivation
- Affected operators, models, or scripts
- Test commands and results
- Hardware and software stack used for MUSA validation, when applicable
- Documentation updates or an explanation for why none are needed

Do not include generated build outputs, local caches, external model checkouts, or private project
notes in public commits.
