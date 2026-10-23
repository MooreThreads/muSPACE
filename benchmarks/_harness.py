"""Shared measurement harness for muSPACE operator benchmarks.

Every runner under ``benchmarks/`` uses this module for argument parsing, timing, result
records and result-table rendering.  The protocol it implements is normative and documented
in ``docs/benchmarking.md``.

The module imports PyTorch lazily, so the documentation checks that only need the constants
(``benchmarks/check_benchmarks.py``) run without a GPU stack installed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_SEED = 2333
DEFAULT_WARMUP = 20
DEFAULT_ITERATIONS = 100
DEFAULT_REPEATS = 5

SCOPES = ("kernel-only", "operator", "forward-backward")
DEFAULT_SCOPE = "kernel-only"

MEASURED_DEVICE = "MTT S5000"
REFERENCE_BANDWIDTH_GBPS = {"NVIDIA A100 80GB PCIe": 1935.0}
DEVICE_BANDWIDTH_GBPS = {MEASURED_DEVICE: 1600.0, **REFERENCE_BANDWIDTH_GBPS}

SCRATCH_DIR = Path(".tmp/benchmarks")
#: The protocol is identical on both sides of a comparison; only the device behind the event
#: timer differs, so the record keeps a device-neutral name for it.
TIMER_NAME = "device-event"
MS_DECIMALS = 4
SPEEDUP_DECIMALS = 2

CANONICAL_HEADER = (
    "| Case | dtype | Scope | S5000 p50 (ms) | S5000 normalized (ms) | "
    "Reference p50 (ms) | Speedup | Note |"
)
CANONICAL_SEPARATOR = "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"

REQUIRED_SECTIONS = (
    "## Scope",
    "## Cases",
    "## Test environment",
    "## Measurement",
    "## Run",
    "## Results",
    "## Acceptance",
    "## Notes",
)

#: Result tables in the operator READMEs are read back by the checker, so the column set and
#: the unit suffixes are part of the format contract.
RESULT_COLUMNS = (
    "Case",
    "dtype",
    "Scope",
    "S5000 p50 (ms)",
    "S5000 normalized (ms)",
    "Reference p50 (ms)",
    "Speedup",
    "Note",
)


# --------------------------------------------------------------------------------------
# paths and arguments
# --------------------------------------------------------------------------------------


def project_root() -> Path:
    """Return the repository root that contains this file."""

    return Path(__file__).resolve().parents[1]


def check_scratch_path(path: Path | str) -> Path:
    """Resolve ``path`` and refuse to write records into the tracked tree."""

    root = project_root()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return resolved
    if relative.parts and relative.parts[0] == ".tmp":
        return resolved
    raise SystemExit(
        f"{candidate}: benchmark records and scratch output must stay outside the tracked "
        "tree; write them under .tmp/ or outside the repository"
    )


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    cases: Sequence[str] | None = None,
    default_case: str | None = None,
    dtypes: Sequence[str] = ("float32", "float16"),
    devices: Sequence[str] = ("musa",),
) -> argparse.ArgumentParser:
    """Add the benchmark CLI contract shared by every runner."""

    if cases is not None:
        parser.add_argument("--case", choices=tuple(cases), default=default_case)
    parser.add_argument("--dtype", choices=tuple(dtypes), default=dtypes[0])
    parser.add_argument("--device", choices=tuple(devices), default=devices[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="local JSON record path; defaults to .tmp/benchmarks/<operator>-<device>.json",
    )
    return parser


def resolve_output(args: argparse.Namespace, operator: str) -> Path:
    """Return the record path for ``args``, defaulting into the scratch directory."""

    device = getattr(args, "device", "musa")
    requested = getattr(args, "output", None)
    path = Path(requested) if requested else SCRATCH_DIR / f"{operator}-{device}.json"
    return check_scratch_path(path)


# --------------------------------------------------------------------------------------
# device and timing
# --------------------------------------------------------------------------------------


def _torch() -> Any:
    import torch

    return torch


def device_module(device: str) -> Any:
    torch = _torch()
    if device == "musa":
        if not hasattr(torch, "musa") or not torch.musa.is_available():
            raise SystemExit("MUSA is unavailable; run this benchmark on a MUSA device")
        return torch.musa
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable; run the reference side on a CUDA device")
        return torch.cuda
    raise SystemExit(f"unsupported device {device!r}")


def device_name(device: str) -> str:
    return str(device_module(device).get_device_name(0))


def synchronize(device: str) -> None:
    device_module(device).synchronize()


@dataclass
class Measurement:
    """Latency samples collected with the harness protocol.

    ``execution_samples_ms`` holds one list of samples per execution.  A single execution is the
    default; several executions are used for rows whose run-to-run spread is larger than the
    acceptance tolerance, in which case the published latency is the median of the per-execution
    medians.
    """

    execution_samples_ms: list[list[float]]

    @property
    def samples_ms(self) -> list[float]:
        return [sample for execution in self.execution_samples_ms for sample in execution]

    @property
    def execution_p50_ms(self) -> list[float]:
        return [statistics.median(execution) for execution in self.execution_samples_ms]

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.execution_p50_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def std_ms(self) -> float:
        samples = self.samples_ms
        return statistics.pstdev(samples) if len(samples) > 1 else 0.0

    def statistics(self) -> dict[str, float]:
        return {
            "p50_ms": self.p50_ms,
            "min_ms": self.min_ms,
            "mean_ms": self.mean_ms,
            "max_ms": self.max_ms,
            "std_ms": self.std_ms,
        }

    def derived(self) -> dict[str, float]:
        return {
            "execution_p50_ms": self.execution_p50_ms,
            "executions": len(self.execution_samples_ms),
        }


def measure(
    launch: Callable[[], None],
    *,
    device: str = "musa",
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    repeats: int = DEFAULT_REPEATS,
    executions: int = 1,
) -> Measurement:
    """Time ``launch`` with device events, following the published protocol."""

    if iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if executions < 1:
        raise SystemExit("--executions must be >= 1")
    module = device_module(device)
    for _ in range(warmup):
        launch()
    module.synchronize()
    collected: list[list[float]] = []
    for _ in range(executions):
        samples: list[float] = []
        for _ in range(repeats):
            start = module.Event(enable_timing=True)
            end = module.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                launch()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / iterations)
        collected.append(samples)
    return Measurement(execution_samples_ms=collected)


# --------------------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------------------


def environment() -> dict[str, Any]:
    """Collect local build and runtime details for the scratch record only."""

    torch_ = _torch()
    details: dict[str, Any] = {"torch": torch_.__version__}
    try:
        import torch_musa

        details["torch_musa"] = torch_musa.__version__
    except Exception:  # pragma: no cover - CUDA-only environments have no torch_musa
        pass
    musa_arch = os.environ.get("MUSA_ARCH")
    if musa_arch:
        details["musa_arch"] = musa_arch
    return details


def normalized_latency_ms(s5000_p50_ms: float, reference_device: str) -> float:
    measured = DEVICE_BANDWIDTH_GBPS[MEASURED_DEVICE]
    reference = DEVICE_BANDWIDTH_GBPS[reference_device]
    return s5000_p50_ms * measured / reference


def speedup(reference_p50_ms: float, s5000_p50_ms: float, reference_device: str) -> float:
    return reference_p50_ms / normalized_latency_ms(s5000_p50_ms, reference_device)


@dataclass
class ResultRecord:
    """One measurement record; scratch evidence for one table row."""

    operator: str
    case: str
    dtype: str
    scope: str
    device: str
    seed: int
    protocol: dict[str, Any]
    statistics: dict[str, float]
    samples_ms: list[float] = field(default_factory=list)
    derived: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, float] = field(default_factory=dict)
    speedup: float | None = None
    environment: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    def save(self, path: Path | str) -> Path:
        target = check_scratch_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{self.to_json()}\n", encoding="utf-8")
        return target


def make_record(
    *,
    operator: str,
    case: str,
    dtype: str,
    scope: str,
    device: str,
    seed: int,
    measurement: Measurement,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    repeats: int = DEFAULT_REPEATS,
    executions: int = 1,
    derived: dict[str, Any] | None = None,
    reference_device: str | None = None,
    reference_implementation: str | None = None,
    reference_p50_ms: float | None = None,
) -> ResultRecord:
    """Assemble a record from one harness measurement."""

    if scope not in SCOPES:
        raise SystemExit(f"scope {scope!r} is not one of {', '.join(SCOPES)}")
    details = measurement.derived()
    details.update(derived or {})
    record = ResultRecord(
        operator=operator,
        case=case,
        dtype=dtype,
        scope=scope,
        device=_device_label(device),
        seed=seed,
        protocol={
            "warmup": warmup,
            "iterations": iterations,
            "repeats": repeats,
            "executions": executions,
            "timer": TIMER_NAME,
        },
        statistics=measurement.statistics(),
        samples_ms=list(measurement.samples_ms),
        derived=details,
        environment=environment(),
    )
    if reference_device is not None:
        record.reference = {
            "device": reference_device,
            "implementation": reference_implementation or "",
            "p50_ms": reference_p50_ms,
        }
        record.normalization = {
            "s5000_bandwidth_gbps": DEVICE_BANDWIDTH_GBPS[MEASURED_DEVICE],
            "reference_bandwidth_gbps": DEVICE_BANDWIDTH_GBPS[reference_device],
        }
        if reference_p50_ms is not None and record.device == MEASURED_DEVICE:
            record.speedup = speedup(reference_p50_ms, measurement.p50_ms, reference_device)
    return record


def _device_label(device: str) -> str:
    """Return the device model name published in records and tables."""

    if device in DEVICE_BANDWIDTH_GBPS:
        return device
    return device_name(device)


def print_summary(record: ResultRecord) -> None:
    stats = record.statistics
    print(
        f"{record.operator} {record.case} {record.dtype} {record.scope}: "
        f"p50 {stats['p50_ms']:.{MS_DECIMALS}f} ms "
        f"(min {stats['min_ms']:.{MS_DECIMALS}f}, mean {stats['mean_ms']:.{MS_DECIMALS}f}, "
        f"max {stats['max_ms']:.{MS_DECIMALS}f}, std {stats['std_ms']:.{MS_DECIMALS}f}) "
        f"on {record.device}"
    )


# --------------------------------------------------------------------------------------
# result-table rendering
# --------------------------------------------------------------------------------------


def format_ms(value: float) -> str:
    return f"{value:.{MS_DECIMALS}f}"


def format_speedup(value: float) -> str:
    return f"{value:.{SPEEDUP_DECIMALS}f}x"


def render_row(
    *,
    case: str,
    dtype: str,
    scope: str,
    s5000_p50_ms: float,
    reference_p50_ms: float | None = None,
    reference_device: str = next(iter(REFERENCE_BANDWIDTH_GBPS)),
    note: str = "",
) -> str:
    """Render one canonical result-table row."""

    cells = [f"`{case}`", f"`{dtype}`", f"`{scope}`"]
    normalized_value = normalized_latency_ms(s5000_p50_ms, reference_device)
    if reference_p50_ms is None:
        if not note:
            note = "reference unavailable"
        cells.extend([format_ms(s5000_p50_ms), format_ms(normalized_value), "", ""])
    else:
        ratio = speedup(reference_p50_ms, s5000_p50_ms, reference_device)
        cells.extend(
            [
                format_ms(s5000_p50_ms),
                format_ms(normalized_value),
                format_ms(reference_p50_ms),
                format_speedup(ratio),
            ]
        )
    cells.append(note)
    return "| " + " | ".join(cells) + " |"


def render_table(rows: Sequence[str]) -> str:
    """Render the canonical table around already formatted rows."""

    lines = [CANONICAL_HEADER, CANONICAL_SEPARATOR, *rows]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# command line: render a published row from an S5000 record and a reference record
# --------------------------------------------------------------------------------------


def _load_record(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _check_pairing(s5000: dict[str, Any], reference: dict[str, Any]) -> None:
    # The published row is identified by (case, dtype, scope); the operator field may name the
    # entry point on one side and the benchmark directory on the other, so it is not compared.
    for key in ("case", "dtype", "scope"):
        if s5000.get(key) != reference.get(key):
            raise SystemExit(
                f"record mismatch on {key!r}: {s5000.get(key)!r} (measured) vs "
                f"{reference.get(key)!r} (reference)"
            )
    if s5000.get("protocol") != reference.get("protocol"):
        raise SystemExit(
            "protocol mismatch between the measured and reference records: "
            f"{s5000.get('protocol')} vs {reference.get('protocol')}"
        )


def _command_row(args: argparse.Namespace) -> int:
    s5000 = _load_record(args.s5000)
    note = args.note
    reference_p50_ms = None
    reference_device = args.reference_device or next(iter(REFERENCE_BANDWIDTH_GBPS))
    if args.reference is not None:
        reference = _load_record(args.reference)
        _check_pairing(s5000, reference)
        reference_p50_ms = float(reference["statistics"]["p50_ms"])
        reference_device = str(reference["device"])
    row = render_row(
        case=str(s5000["case"]),
        dtype=str(s5000["dtype"]),
        scope=str(s5000["scope"]),
        s5000_p50_ms=float(s5000["statistics"]["p50_ms"]),
        reference_p50_ms=reference_p50_ms,
        reference_device=reference_device,
        note=note,
    )
    published = dict(s5000)
    published["reference"] = {
        "device": reference_device,
        "implementation": (s5000.get("reference") or {}).get("implementation", ""),
        "p50_ms": reference_p50_ms,
    }
    published["normalization"] = {
        "s5000_bandwidth_gbps": DEVICE_BANDWIDTH_GBPS[MEASURED_DEVICE],
        "reference_bandwidth_gbps": DEVICE_BANDWIDTH_GBPS[reference_device],
    }
    published["note"] = note
    published["speedup"] = (
        None
        if reference_p50_ms is None
        else speedup(reference_p50_ms, float(s5000["statistics"]["p50_ms"]), reference_device)
    )
    target = args.output or SCRATCH_DIR / f"{s5000['operator']}-published.json"
    target = check_scratch_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(published, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(row)
    return 0


def _command_aggregate(args: argparse.Namespace) -> int:
    """Combine repeated executions of one row into a single published record."""

    if len(args.records) < 2:
        raise SystemExit("aggregate needs at least two execution records")
    records = [_load_record(path) for path in args.records]
    first = records[0]
    for other in records[1:]:
        _check_pairing(first, other)
    metric = Measurement(
        execution_samples_ms=[
            [float(sample) for sample in record.get("samples_ms") or []]
            or [float(record["statistics"]["p50_ms"])]
            for record in records
        ]
    )
    protocol = dict(first.get("protocol") or {})
    protocol["executions"] = len(records)
    carried = {
        key: value
        for key, value in (first.get("derived") or {}).items()
        if key not in ("execution_p50_ms", "executions")
    }
    carried.update(metric.derived())
    record = ResultRecord(
        operator=str(first["operator"]),
        case=str(first["case"]),
        dtype=str(first["dtype"]),
        scope=str(first["scope"]),
        device=str(first["device"]),
        seed=int(first["seed"]),
        protocol=protocol,
        statistics=metric.statistics(),
        samples_ms=metric.samples_ms,
        derived=carried,
        reference=first.get("reference") or {},
        normalization=first.get("normalization") or {},
        speedup=first.get("speedup"),
        environment=first.get("environment") or {},
    )
    target = args.output or SCRATCH_DIR / f"{record.operator}-{record.case}-aggregate.json"
    saved = record.save(target)
    print_summary(record)
    print(
        f"executions {len(records)}: "
        + ", ".join(f"{value:.{MS_DECIMALS}f}" for value in metric.execution_p50_ms)
    )
    print(f"record: {saved}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks/_harness.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    row = sub.add_parser("row", help="render a published table row from two records")
    row.add_argument("--s5000", type=Path, required=True, help="record from the measured device")
    row.add_argument("--reference", type=Path, help="record from the reference device")
    row.add_argument("--reference-device", default=None)
    row.add_argument("--note", default="")
    row.add_argument("--output", type=Path, default=None)
    row.set_defaults(func=_command_row)
    aggregate = sub.add_parser(
        "aggregate",
        help="combine repeated executions of one row into a single record",
    )
    aggregate.add_argument("records", nargs="+", type=Path, help="one record per execution")
    aggregate.add_argument("--output", type=Path, default=None)
    aggregate.set_defaults(func=_command_aggregate)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
