# Operator benchmark standard

This document is normative for everything under `benchmarks/`. It defines what counts as a muSPACE
release benchmark, how latency is measured, how a case is described, how results are recorded, and
what a reviewer checks before a row is published.

The rules elsewhere in the repository apply unchanged: fixture profiling is a diagnostic, not a
release benchmark ([architecture.md](architecture.md)); launch checks such as the FlashOcc and UniAD
one-iteration launchers measure nothing and their timings must never be quoted as performance data
([../README.md](../README.md), [../CONTRIBUTING.md](../CONTRIBUTING.md)); temporary benchmark data
must not be committed ([operator_development.md](operator_development.md)).

## Scope

A release benchmark measures one canonical operator API from `mu_space.ops` or `mu_space`, on one
device, and compares it with a named reference implementation on a reference device. A release
benchmark is:

- reproducible from committed code alone: a fixed seed, a documented command, and no redistributed
  input data;
- comparable: the same case, the same dtype, and the same protocol on both devices;
- scoped: it states exactly which part of the operator boundary is inside the timed region;
- recorded: its published numbers live in the operator README result table, and nowhere else.

A benchmark is not a test. Correctness is established by the operator tests under `tests/ops/`, and a
published row is only valid while the matching test passes.

## Terminology

| Term | Meaning |
| --- | --- |
| Case | One named workload: fixed shapes, dtypes, distributions and seed. Identified by a case id that is stable across releases. |
| Scope | Which work the timed region covers: `kernel-only`, `operator`, or `forward-backward`. |
| Warmup | Launches executed before sampling, never timed, used to reach steady state. |
| Sample | One timing measurement: `iterations` timed launches divided by `iterations`. |
| Repeat | One independent sample. A published row is summarised over `repeats` samples. |
| Published row | One line of an operator README result table: one case, one dtype, one scope. |
| S5000 row | The measured latency of the muSPACE implementation. |
| Reference row | The measured latency of the named reference implementation on the reference device. |
| Baseline | An earlier muSPACE implementation of the same operator, used only for regression claims, never as the reference platform. |
| Normalized latency | The S5000 latency adjusted by the ratio of nominal device bandwidths, as defined below. |

## Layout

```text
benchmarks/
  README.md                    index, coverage table, device constants, comparison rules
  _harness.py                  shared argument parsing, timing, records and table rendering
  check_benchmarks.py          structure, numeric and hygiene checks for this directory
  <operator>/
    README.md                  the published record for this operator
    benchmark_<operator>.py    runner: builds the case, calls the harness, prints and records
    <operator>_case.py         canonical case construction, shared with the runner
    export_<something>.py      optional: writes a case to an artifact for cross-device replay
```

One operator owns one directory. Two entry points of the same operator family (for example a fused
and a non-fused pooling variant) share a directory and select the entry point with `--op`; they do not
get a second directory. A published row is identified by `(case, dtype, scope)`, so two entry points
that would otherwise collide are distinguished by an entry-point-qualified case id.

Case construction belongs in `<operator>_case.py`, so that the runner and any local comparison or
replay script build identical inputs from one place. The published path builds deterministic
synthetic inputs from a fixed seed. Replaying
a captured artifact through `--input-file` is a local investigation path: it may be used to explain a
number, but it is not how a published row is produced, and captured artifacts are never committed.

## Measurement protocol

Every published row uses this protocol. The implementation lives in `benchmarks/_harness.py`; runners
call it instead of timing themselves.

- **Timer.** The device event timer of the measured platform with `enable_timing=True`
  (`torch.musa.Event` on MUSA, `torch.cuda.Event` on CUDA). A wall-clock timer around a synchronised
  loop is not accepted for published rows. The reference device uses the equivalent device event
  timer, so the recorded protocol is the same on both sides of a comparison.
- **Sample.** Warm up with `--warmup` launches, synchronise, record the start event, launch the
  operator `--iterations` times, record the end event, synchronise, and divide the elapsed time by
  `--iterations`.
- **Repeats.** Collect `--repeats` independent samples. The published number is the median of the
  samples; minimum, mean, maximum and standard deviation are reported with it.
- **Executions.** One execution of the protocol produces the published sample set. When the run-to-run
  spread of an operator is larger than the tolerance stated in `## Acceptance`, the row is published
  as the median over at least five full executions of the documented command, the record states the
  execution count in its protocol, and the record lists every per-execution median. The reference row
  for that case uses the same execution count. A row that is stable across executions is published
  from a single execution.
- **Defaults.** `--warmup 20 --iterations 100 --repeats 5`, `--seed 2333`. A deviation is allowed only
  when the protocol would make the run unreasonably long; it must be stated in `## Measurement` and
  visible in the local result record.
- **Scope.** Every case declares `kernel-only`, `operator`, or `forward-backward`. Outputs and
  workspaces are allocated before the timed region. A `kernel-only` measurement must not include host
  synchronisation, allocation, or a device-to-host copy inside the timed region, and must say which
  launcher-side work it excludes.
- **dtype.** Published rows use `float32` unless the operator's production contract requires another
  dtype. Each dtype is a separate row with its own reference measurement.
- **Device.** Measure on an idle device with exclusive access, and state the device model. Do not
  publish a row measured under contention.
- **Correctness first.** The matching `tests/ops/test_<operator>.py` must pass on the same build before
  its numbers are published.

## Local result records

Each run writes one JSON record to a scratch path outside version control (`.tmp/benchmarks/` by
default). The record is the audit trail for the numbers in the README; it is never committed.

```json
{
  "operator": "bev_pool",
  "case": "r50-200k",
  "dtype": "float32",
  "scope": "forward-backward",
  "device": "MTT S5000",
  "seed": 2333,
  "protocol": {"warmup": 20, "iterations": 100, "repeats": 5, "executions": 1, "timer": "device-event"},
  "statistics": {"p50_ms": 0.0, "min_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0, "std_ms": 0.0},
  "derived": {"execution_p50_ms": [0.0]},
  "reference": {"device": "NVIDIA A100 80GB PCIe", "implementation": "BEVFusion CUDA extension"},
  "normalization": {"s5000_bandwidth_gbps": 1600, "reference_bandwidth_gbps": 1935},
  "speedup": 0.0,
  "environment": {"documentation": "see docs/install.md"}
}
```

Records are named `<operator>-<case>[-<dtype>].json`, and that file name is what binds a record to its
benchmark directory; the `operator` field keeps the canonical API or entry-point name that was
measured. The benchmark directory plus `case`, `dtype` and `scope` identify a published row, so two
directories may publish the same case id without being confused with each other. `environment` is
free-form evidence for internal tracking. It is deliberately the only place where build and toolchain
details are recorded, and it is never copied into the README.

Keep reference-device records in a subdirectory such as `<scratch>/a100/`: only the records directly
in the record directory are published candidates, and `--records` audits them against the README
tables.

Executions of one row are combined into the published record with:

```bash
python benchmarks/_harness.py aggregate .tmp/benchmarks/<operator>-<case>-1.json \
  .tmp/benchmarks/<operator>-<case>-2.json --output .tmp/benchmarks/<operator>-<case>.json
```

Verify a README table against fresh records with:

```bash
python benchmarks/check_benchmarks.py --check --records .tmp/benchmarks
```

## Result table

Every operator README ends with the same table, with the same header, the same column order and the
same units:

```text
| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
```

- `Case` — the case id from the runner.
- `dtype` — `float32`, `float16`, and so on.
- `Scope` — `kernel-only`, `operator`, or `forward-backward`.
- `S5000 p50 (ms)` — median S5000 latency, four decimal places.
- `S5000 normalized (ms)` — S5000 latency adjusted by nominal bandwidth (below), four decimal places.
- `Reference p50 (ms)` — median reference latency under the same case and protocol, four decimal
  places.
- `Speedup` — `Reference p50 / S5000 normalized`, two decimal places with an `x` suffix. Values above
  `1.00x` favor S5000.
- `Note` — one short clause when the row needs a caveat. Use `reference unavailable` when no reference
  row is published, and keep the reference and speedup cells empty in that case.

The normalized column depends only on the measured latency and the registered reference bandwidth, so
it is filled for every row, including a row whose reference measurement is still outstanding.

Published comparisons use FP32 latency. S5000 has 1600 GB/s nominal memory bandwidth. The shared
device constants and the reference bandwidth of every reference device are registered in
[../benchmarks/README.md](../benchmarks/README.md).

```text
S5000 normalized latency = S5000 measured latency x 1600 / reference bandwidth
Speedup                  = reference latency / S5000 normalized latency
```

This is a nominal-bandwidth adjustment, not a roofline model. It cannot make a `kernel-only` row
comparable to an `operator` row: scope is part of the comparison, and rows with different scopes are
never merged. When the adjustment is not meaningful for a row, keep the column filled anyway and say
so in `Note`.

## Operator README structure

Every operator README has exactly these sections, in this order:

````markdown
# <Operator> benchmark

## Scope

Which operator entry point is measured, and which parts of the surrounding pipeline are excluded.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
| `<case-id>` | shapes, grids, channel counts, and the workload property the case represents | `float32` | `2333` |

## Test environment

- Measured device: MTT S5000
- Reference device: <reference device model>

## Measurement

Scope, timer, warmup / iterations / repeats, seed, what is preallocated, and any documented deviation
from the default protocol.

## Run

```bash
PYTHONPATH=. python benchmarks/<operator>/benchmark_<operator>.py --case <case-id> \
  --warmup 20 --iterations 100 --repeats 5 --output .tmp/benchmarks/<operator>.json
```

## Results

| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | Reference p50 (ms) | Speedup | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `<case-id>` | `float32` | `kernel-only` | 0.0000 | 0.0000 | 0.0000 | 0.00x | |

## Acceptance

The pass criterion for this operator, and the test that establishes correctness for the measured path.

## Notes

Limitations, model context, and anything a reader must not conclude from the table.
````

## Reference platform

Every published row needs a reference measurement from a named implementation on a reference device.
The README states:

- the reference device model and its nominal bandwidth, as registered in `benchmarks/README.md`;
- the reference implementation by name, for example `the original BEVFusion CUDA extension`;
- the case, dtype, seed, warmup, iterations and repeats used on both devices;
- the exact command that reproduces the measured side, and the upstream call plus the timed boundary
  used on the reference side.

Anything that identifies a development machine, container, account, path, session or build is not
published. Build and toolchain details, reference checkout revisions, and measurement sessions are
tracked in internal release records, not in this repository. A reader reproduces a measured row by
running the documented command with the documented case, and reproduces the reference side by calling
the named upstream implementation at the documented boundary under the same protocol.

A row is published with a reference measurement whenever an upstream implementation of the same
operator exists and can be run. When it cannot be run, publish the row with an empty reference cell and
`reference unavailable` in `Note`, and track the missing reference as outstanding work; the reference
implementation that would be used is named in `## Notes` so the gap is specific. A baseline
implementation of muSPACE itself is not a reference platform; a baseline comparison may be added in
`## Notes` as a regression statement, clearly labelled as such, and never normalised as if it were a
reference row.

## Acceptance criterion

Each operator README states one pass criterion in `## Acceptance`, for example `S5000 is not slower
than the reference on this case (Speedup >= 1.00x)`, or `no regression against the previous
implementation`. The criterion is what a reviewer checks, and what a change to a published number has
to re-establish. Correctness is the precondition: the criterion applies only while
`tests/ops/test_<operator>.py` passes on the measured build.

## Coverage

Every operator in [operator_matrix.md](operator_matrix.md) is covered by exactly one benchmark
directory and one result table. An operator that is not benchmarked yet is listed in
`benchmarks/README.md` with the reason and stays visible until it is covered. Silent gaps are a
defect of this directory, not a neutral state.

## Privacy and hygiene rules

This directory is published as part of the public repository. Benchmark files must not contain:

- user names, account names, e-mail addresses, or `user@host` forms;
- machine names, host names, IP addresses, or network locations;
- container, image, mount or working-directory names;
- absolute paths, including local build, workspace and scratch paths;
- dates, times or measurement timestamps;
- internal service names, internal repository URLs, or internal issue identifiers;
- software stack version numbers, driver versions, toolchain versions, or upstream commit hashes.

Device model names, nominal bandwidth constants, seed values and protocol parameters (warmup,
iterations, repeats) are the permitted information boundary: they are what another reader needs in
order to run the documented command and interpret the table. `docs/install.md` is the single place
where supported software versions are stated; benchmark documentation points there instead of
restating them.

Records that fall outside this boundary — build settings, toolchain versions, reference revisions,
device indices, measurement sessions — belong in internal release records, not in the repository.

`benchmarks/check_benchmarks.py` enforces the mechanical part of these rules. The export tooling
enforces the rest before a release is published.

## Adding or updating a benchmark

Adding a benchmark for a new operator:

1. Add or confirm the operator tests under `tests/ops/` and run them.
2. Add `benchmarks/<operator>/<operator>_case.py` with a fixed-seed case and stable case ids.
3. Add `benchmarks/<operator>/benchmark_<operator>.py` using `benchmarks/_harness.py` for arguments,
   timing, records and table rendering.
4. Write `benchmarks/<operator>/README.md` using the structure above.
5. Measure the S5000 row and the reference row, then fill the result table from the generated rows.
6. Add the operator to the coverage table in `benchmarks/README.md`.
7. Run `python benchmarks/check_benchmarks.py --check --records .tmp/benchmarks`.
8. Remove scratch artifacts from `.tmp/`.

Updating a published number:

1. Re-run the documented command on the documented devices with the default protocol.
2. Update the affected rows in the same change that motivates the re-measurement.
3. Re-check the acceptance criterion and update it only when the criterion itself changes.
4. Keep the diff limited to the rows that actually changed.

## What must not be committed

- Tuning diaries, intermediate measurements, and exploratory shape sweeps.
- Profiler output, traces, counters and dumps.
- Cross-device comparison scripts, exported input artifacts, and comparison logs.
- Local JSON result records.
- Case inputs derived from data sets that the repository does not redistribute.

Only a final, reproducible platform comparison belongs in `benchmarks/`: the runner, the case, the
protocol, and the result table.
