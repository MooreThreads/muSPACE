# BEVFormer Fixture Profiling

Fixture profiling collects diagnostic timing, memory, and dispatch data for the small synthetic
BEVFormer-tiny run. It is useful for comparing code changes, but it is not a release benchmark and
does not replace real-data validation.

## One-Command Profiling

Run the default compatibility-first profile:

```bash
cd muSPACE
model_examples/BEVFormer/scripts/profile_fixture.sh --checkout ./BEVFormer
```

Run strict native-path profiling:

```bash
model_examples/BEVFormer/scripts/profile_fixture.sh --checkout ./BEVFormer --strict
```

Outputs are written under `build/profiling/bevformer_fixture/`:

- `default_train_step.json` or `strict_train_step.json`: loss, elapsed time, peak memory, and optimizer status
- `default_dispatch.log` or `strict_dispatch.log`: MSDA dispatch reports from `MUSPACE_DISPATCH_LOG=1`
- `default_summary.json` or `strict_summary.json`: compact summary for quick comparison

Use `--output-dir` to redirect outputs.

## Optional Forward Trace

Add `--trace` to collect compact tensor summaries from selected BEVFormer modules:

```bash
model_examples/BEVFormer/scripts/profile_fixture.sh --checkout ./BEVFormer --trace
```

This writes `default_trace.json` or `strict_trace.json`. Use `--sample-elements N` to control how
many tensor elements are sampled per record.

## Reading Results

For correctness, require `finite_gradients: true` and `optimizer_step: true`. For dispatch, inspect
the dispatch log: native execution appears as `selected_backend: "musa"`, while auto fallback appears
as `selected_backend: "reference"` with `native_fallback: true`.

Compare `elapsed_seconds` and `peak_memory_bytes` only between runs on the same machine, same device,
same seed, and similar system load. Treat noisy deltas as profiling hints, not performance claims.
