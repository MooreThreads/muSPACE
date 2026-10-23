#!/usr/bin/env python3
"""Structure, numeric and hygiene checks for the muSPACE operator benchmarks.

The rules this script enforces are documented in ``docs/benchmarking.md``.  It is used by the
repository pre-commit hook and by CI, and it never needs a GPU stack: PyTorch is imported
lazily by ``_harness``, so the constants and the result-table format are available everywhere.

Usage::

    python benchmarks/check_benchmarks.py --check
    python benchmarks/check_benchmarks.py --check --records .tmp/benchmarks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _harness  # noqa: E402  (resolved from the benchmarks directory)

HUB_README = "benchmarks/README.md"
MATRIX_DOC = "docs/operator_matrix.md"
STANDARD_DOC = "docs/benchmarking.md"
PRIVACY_PATHS = (HUB_README, STANDARD_DOC)
SELF = "benchmarks/check_benchmarks.py"
#: Names that belong to the internal dataset sharing workflow and are dropped by the export
#: pipeline before publication, so their contents are intentionally not scanned here.  Only the
#: bare names are spelled out: the export pipeline refuses to publish a file that contains the
#: full internal path, and spelling it here would trip that scan.
INTERNAL_ONLY = (
    "dumped_data.dvc",
    "pull_dumped_data.sh",
    "dumped_data",
)

FORBIDDEN_SUFFIXES = (".json", ".log", ".csv", ".txt", ".png", ".npy", ".npz", ".prof")
ALLOWED_EXTRA_SUFFIXES = (".patch",)
CASE_SUFFIX = "_case.py"

MS_TOLERANCE = 2e-4
SPEEDUP_TOLERANCE = 0.02

#: Patterns that must never appear in a published benchmark file.  They are written generically
#: on purpose: the export pipeline refuses to publish files that contain internal host, service,
#: account or ticket literals, and a detector that spelled those literals out would trip it.
PRIVACY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "absolute filesystem path",
        re.compile(r"(?<![\w/.])/(?:home|data|root|mnt|media|opt|tmp|var|usr)/[\w./-]+"),
    ),
    ("Windows absolute path", re.compile(r"\b[A-Za-z]:\\{1,2}[\w.\\-]+")),
    ("IP address", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
    ("account-qualified host", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("date stamp", re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")),
    ("clock timestamp", re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")),
    (
        "source revision hash",
        re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b"),
    ),
    (
        "toolchain or build version",
        re.compile(
            r"\b(?:musa|cuda|torch|torch_musa|pytorch|python|gcc|g\+\+|driver|toolkit|cmake|"
            r"ninja|cudnn|mcc)\b[\s:=_-]*v?\d+(?:\.\d+)+\b",
            re.IGNORECASE,
        ),
    ),
    ("component version number", re.compile(r"\b\d+\.\d+\.\d+(?:[.\w+-]*)?\b")),
    ("issue identifier", re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{2,}\b")),
    ("internal service URL", re.compile(r"https?://[\w.-]*mthreads[\w.-]*", re.IGNORECASE)),
)


@dataclass
class Finding:
    """One rule violation."""

    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------
# markdown helpers
# --------------------------------------------------------------------------------------


def section_body(text: str, heading: str) -> list[str]:
    """Return the lines of ``heading``'s body, stopping at the next heading."""

    lines = text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return []
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("# ") or line.startswith("## "):
            break
        body.append(line)
    return body


def cells_of(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def first_table(lines: Iterable[str]) -> list[list[str]]:
    """Return the first markdown table in ``lines``, ignoring fenced code blocks."""

    rows: list[list[str]] = []
    in_fence = False
    started = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("|"):
            started = True
            rows.append(cells_of(stripped))
        elif started:
            break
    return rows


def is_separator(row: Sequence[str]) -> bool:
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in row)


# --------------------------------------------------------------------------------------
# per-operator checks
# --------------------------------------------------------------------------------------


def benchmark_dirs(root: Path) -> list[Path]:
    """Return the operator directories: every directory holding a README or Python source."""

    directory = root / "benchmarks"
    if not directory.is_dir():
        return []
    return sorted(
        child
        for child in directory.iterdir()
        if child.is_dir()
        and not child.name.startswith((".", "__"))
        and (child / "README.md").is_file()
        or (child.is_dir() and any(child.glob("*.py")))
    )


def _check_files(root: Path, operator_dir: Path, findings: list[Finding]) -> None:
    operator = operator_dir.name
    names = sorted(child.name for child in operator_dir.iterdir() if child.is_file())
    if "README.md" not in names:
        findings.append(Finding(_rel(root, operator_dir), "missing README.md"))
    if not any(name.startswith("benchmark_") and name.endswith(".py") for name in names):
        findings.append(Finding(_rel(root, operator_dir), "missing benchmark_<operator>.py runner"))
    if not any(name.endswith(CASE_SUFFIX) for name in names):
        findings.append(Finding(_rel(root, operator_dir), f"missing a *{CASE_SUFFIX} case module"))
    runner = f"benchmark_{operator}.py"
    if runner not in names and any(name.startswith("benchmark_") for name in names):
        findings.append(
            Finding(
                _rel(root, operator_dir),
                f"runner is not named {runner} (the directory and runner names must match)",
            )
        )
    for name in names:
        if name.endswith(FORBIDDEN_SUFFIXES):
            findings.append(
                Finding(_rel(root, operator_dir / name), "recorded data must not be committed")
            )
        elif name.endswith(".py") or name.endswith(".md") or name.endswith(ALLOWED_EXTRA_SUFFIXES):
            continue
        elif name in (".gitignore",):
            continue
        else:
            findings.append(
                Finding(_rel(root, operator_dir / name), "unexpected file in a benchmark directory")
            )


def _check_row_math(
    row: Sequence[str], where: str, reference_device: str, findings: list[Finding]
) -> tuple[str, str, str, bool]:
    case, dtype, scope, s5000, normalized, reference, ratio, note = row
    where = f"{where} row {case}"
    if scope.strip("`") not in _harness.SCOPES:
        findings.append(Finding(where, f"unknown scope {scope!r}"))
    try:
        s5000_ms = float(s5000)
    except ValueError:
        findings.append(Finding(where, f"S5000 latency {s5000!r} is not a number"))
        return case, dtype, scope, False
    if not re.fullmatch(r"\d+\.\d{4}", s5000):
        findings.append(Finding(where, "S5000 latency must have exactly four decimals"))
    reference_present = bool(reference)
    expected_normalized = _harness.normalized_latency_ms(s5000_ms, reference_device)
    if not normalized:
        findings.append(Finding(where, "normalized latency is missing"))
    else:
        try:
            actual_normalized = float(normalized)
        except ValueError:
            findings.append(Finding(where, f"normalized latency {normalized!r} is not a number"))
            actual_normalized = None
        else:
            if not re.fullmatch(r"\d+\.\d{4}", normalized):
                findings.append(
                    Finding(where, "normalized latency must have exactly four decimals")
                )
            if abs(actual_normalized - expected_normalized) > MS_TOLERANCE:
                findings.append(
                    Finding(
                        where, f"normalized latency {normalized} does not follow from {s5000} ms"
                    )
                )
    if reference_present:
        try:
            reference_ms = float(reference)
        except ValueError:
            findings.append(Finding(where, f"reference latency {reference!r} is not a number"))
            return case, dtype, scope, True
        if not re.fullmatch(r"\d+\.\d{4}", reference):
            findings.append(Finding(where, "reference latency must have exactly four decimals"))
        expected_ratio = reference_ms / expected_normalized if expected_normalized else 0.0
        if not re.fullmatch(r"\d+\.\d{2}x", ratio):
            findings.append(Finding(where, f"speedup {ratio!r} must have two decimals and an x"))
        else:
            actual_ratio = float(ratio.rstrip("x"))
            if abs(actual_ratio - expected_ratio) > SPEEDUP_TOLERANCE:
                findings.append(
                    Finding(
                        where,
                        f"speedup {ratio} does not follow from the published latencies "
                        f"({expected_ratio:.2f}x expected)",
                    )
                )
        if note.strip().lower() == "reference unavailable":
            findings.append(
                Finding(where, "reference latency is published but the note says unavailable")
            )
    else:
        if ratio:
            findings.append(Finding(where, "speedup requires a reference measurement"))
        if "reference unavailable" not in note:
            findings.append(
                Finding(
                    where,
                    "a row without a reference row must say 'reference unavailable' in Note",
                )
            )
    return case, dtype, scope, reference_present


def reference_device_of(text: str) -> str:
    """Return the registered reference device named in a README's test environment."""

    environment = "\n".join(section_body(text, "## Test environment"))
    for device in _harness.REFERENCE_BANDWIDTH_GBPS:
        if device in environment:
            return device
    return next(iter(_harness.REFERENCE_BANDWIDTH_GBPS))


def check_operator_readme(root: Path, operator_dir: Path, findings: list[Finding]) -> None:
    path = operator_dir / "README.md"
    where = _rel(root, path)
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    if not lines or not lines[0].startswith("# ") or "benchmark" not in lines[0].lower():
        findings.append(Finding(where, "first heading must name the operator benchmark"))

    positions: list[int] = []
    for heading in _harness.REQUIRED_SECTIONS:
        occurrences = [i for i, line in enumerate(lines) if line.strip() == heading]
        if not occurrences:
            findings.append(Finding(where, f"missing section {heading}"))
            continue
        if len(occurrences) > 1:
            findings.append(Finding(where, f"section {heading} appears more than once"))
        positions.append(occurrences[0])
    if positions != sorted(positions):
        findings.append(Finding(where, "sections are not in the documented order"))

    environment = "\n".join(section_body(text, "## Test environment"))
    if _harness.MEASURED_DEVICE not in environment:
        findings.append(
            Finding(
                where,
                f"## Test environment must name the measured device ({_harness.MEASURED_DEVICE})",
            )
        )
    if not any(device in environment for device in _harness.REFERENCE_BANDWIDTH_GBPS):
        findings.append(
            Finding(where, "## Test environment must name a registered reference device model")
        )

    cases_table = first_table(section_body(text, "## Cases"))
    case_ids = {row[0].strip("`") for row in cases_table[1:] if row} if cases_table else set()
    if not cases_table:
        findings.append(Finding(where, "## Cases must contain the case table"))
    elif cases_table[0][:1] != ["Case"]:
        findings.append(Finding(where, "## Cases table must start with a Case column"))

    results_lines = section_body(text, "## Results")
    reference_device = reference_device_of(text)
    table = first_table(results_lines)
    if not table:
        findings.append(Finding(where, "## Results must contain the result table"))
    else:
        header = [cell for cell in table[0]]
        if header != list(_harness.RESULT_COLUMNS):
            findings.append(
                Finding(
                    where,
                    f"## Results header differs from the canonical header: {'| '.join(header)}",
                )
            )
        if len(table) < 2 or not is_separator(table[1]):
            findings.append(Finding(where, "## Results needs a markdown separator row"))
        for row in table[2:]:
            if not row or not any(row):
                continue
            if len(row) != len(_harness.RESULT_COLUMNS):
                findings.append(Finding(where, f"result row has {len(row)} cells, expected 8"))
                continue
            case, _, _, _ = _check_row_math(row, where, reference_device, findings)
            if case.strip("`") not in case_ids:
                findings.append(Finding(where, f"result case {case} is not declared in ## Cases"))

    run_body = "\n".join(section_body(text, "## Run"))
    runner = f"benchmark_{operator_dir.name}.py"
    if runner not in run_body:
        findings.append(Finding(where, f"## Run must invoke {runner}"))
    if "--output" not in run_body or ".tmp/" not in run_body:
        findings.append(Finding(where, "## Run must write its record under .tmp/"))
    if "PYTHONPATH=." not in run_body:
        findings.append(Finding(where, "## Run must set PYTHONPATH=."))

    _check_files(root, operator_dir, findings)


# --------------------------------------------------------------------------------------
# hub, matrix and record checks
# --------------------------------------------------------------------------------------


def check_hub(root: Path, findings: list[Finding]) -> None:
    path = root / HUB_README
    where = _rel(root, path)
    if not path.is_file():
        findings.append(Finding(where, "missing benchmark hub README"))
        return
    text = path.read_text(encoding="utf-8")
    for heading in ("## Coverage", "## Comparison rules", "## Reproducing benchmark inputs"):
        if heading not in text:
            findings.append(Finding(where, f"missing section {heading}"))

    declared = {directory.name for directory in benchmark_dirs(root)}
    linked: set[str] = set()
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "#")):
            continue
        relative = target.split("#")[0]
        if not (path.parent / relative).exists():
            findings.append(Finding(where, f"link target {target} does not exist"))
        segment = relative.split("/")[0]
        if segment in declared:
            linked.add(segment)
    for missing in sorted(declared - linked):
        findings.append(
            Finding(where, f"benchmark directory {missing} is not listed in ## Coverage")
        )

    registered = {
        row[0]: row[2]
        for row in first_table(section_body(text, "## Comparison rules"))[1:]
        if len(row) >= 3
    }
    for device, bandwidth in _harness.DEVICE_BANDWIDTH_GBPS.items():
        value = registered.get(f"`{device}`") or registered.get(device)
        if value is None:
            findings.append(Finding(where, f"device table does not register {device}"))
        elif f"{bandwidth:g}" not in value:
            findings.append(
                Finding(
                    where, f"registered bandwidth for {device} is {value}, expected {bandwidth:g}"
                )
            )


def check_matrix(root: Path, findings: list[Finding]) -> None:
    path = root / MATRIX_DOC
    where = _rel(root, path)
    if not path.is_file():
        findings.append(Finding(where, "missing operator matrix"))
        return
    text = path.read_text(encoding="utf-8")
    table = first_table(text.splitlines())
    if not table or "Benchmark" not in table[0]:
        findings.append(Finding(where, "operator matrix must have a Benchmark column"))
        return
    column = table[0].index("Benchmark")
    linked: set[str] = set()
    for row in table[1:]:
        if len(row) <= column:
            continue
        for target in re.findall(r"benchmarks/([\w.-]+)/README\.md", row[column]):
            linked.add(target)
    for name in sorted(linked):
        if not (root / "benchmarks" / name / "README.md").is_file():
            findings.append(Finding(where, f"Benchmark column points at missing directory {name}"))
    declared = {directory.name for directory in benchmark_dirs(root)}
    for missing in sorted(declared - linked):
        findings.append(Finding(where, f"operator matrix does not cover benchmark {missing}"))


def record_owner(path: Path, record: dict, declared: set[str]) -> str | None:
    """Return the benchmark directory a record belongs to.

    Records are named ``<operator>-<case>[-<dtype>].json``, which is what binds a record to a
    README; the ``operator`` field keeps the canonical API or entry-point name that was measured.
    A record whose file name does not start with a benchmark directory falls back to its
    ``operator`` field when that field is a directory name.
    """

    for directory in sorted(declared, key=len, reverse=True):
        if path.name.startswith(f"{directory}-"):
            return directory
    operator = record.get("operator")
    return operator if operator in declared else None


def collect_records(records_dir: Path) -> dict[tuple, list[tuple[Path, dict]]]:
    """Group local records by case, dtype and scope.

    The operator field is deliberately not part of the key: one benchmark directory may cover
    several entry points of the same operator family, and the published row identifies the
    workload by case, dtype and scope.

    Only records directly inside the directory are published candidates. Records of the reference
    device live in a subdirectory (``<records>/a100``) and are inputs to ``_harness.py row``, not
    published rows; a pair of records becomes one published record in the top level.
    """

    grouped: dict[tuple, list[tuple[Path, dict]]] = {}
    for path in sorted(records_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        key = (data.get("case"), data.get("dtype"), data.get("scope"))
        grouped.setdefault(key, []).append((path, data))
    return grouped


def record_key(record: dict) -> tuple:
    return (record.get("case"), record.get("dtype"), record.get("scope"))


def _p50(record: dict) -> float | None:
    statistics = record.get("statistics") or {}
    value = statistics.get("p50_ms")
    return float(value) if isinstance(value, (int, float)) else None


def check_records(root: Path, records_dir: Path, findings: list[Finding]) -> None:
    where = str(records_dir)
    if not records_dir.is_dir():
        findings.append(Finding(where, "record directory does not exist"))
        return
    grouped = collect_records(records_dir)
    if not grouped:
        findings.append(Finding(where, "no JSON records found"))
        return
    matched: set[tuple] = set()
    declared = {directory.name for directory in benchmark_dirs(root)}
    for operator_dir in benchmark_dirs(root):
        readme = operator_dir / "README.md"
        if not readme.is_file():
            continue
        text = readme.read_text(encoding="utf-8")
        table = first_table(section_body(text, "## Results"))
        owned_records = [
            (key, record)
            for key, entries in grouped.items()
            for path, record in entries
            if record_owner(path, record, declared) == operator_dir.name
        ]
        if not owned_records:
            # A directory without local records cannot be audited; the tree-wide gate is the
            # release checklist, not this optional cross-check.
            continue
        for row in table[2:]:
            if len(row) != len(_harness.RESULT_COLUMNS) or not any(row):
                continue
            case = row[0].strip("`")
            key = (case, row[1].strip("`"), row[2].strip("`"))
            label = f"{_rel(root, readme)} row {case}"
            # Records are bound to their directory by file name, so two directories may publish
            # the same case id without being confused with each other.
            candidates = [
                item
                for item in grouped.get(key, [])
                if record_owner(item[0], item[1], declared) == operator_dir.name
            ]
            if not candidates:
                findings.append(
                    Finding(
                        label,
                        f"no local record found for {key} owned by {operator_dir.name!r}",
                    )
                )
                continue
            matched.add((key, operator_dir.name))
            aggregates = [
                item
                for item in candidates
                if (item[1].get("protocol") or {}).get("executions", 1) > 1
            ]
            published = max(
                aggregates or candidates,
                key=lambda item: item[1].get("reference", {}).get("p50_ms") is not None,
            )
            if not aggregates:
                values = {_p50(record) for _, record in candidates}
                if len(values) > 1:
                    findings.append(
                        Finding(label, f"conflicting p50 values across records: {sorted(values)}")
                    )
            reference_present = bool(row[5])
            record_p50 = _p50(published[1])
            if record_p50 is None:
                findings.append(Finding(label, "record has no statistics.p50_ms"))
                continue
            if f"{record_p50:.4f}" != row[3]:
                findings.append(
                    Finding(label, f"published latency {row[3]} != recorded {record_p50:.4f}")
                )
            record_reference = (published[1].get("reference") or {}).get("p50_ms")
            if reference_present and record_reference is None:
                findings.append(Finding(label, "reference latency published without a record"))
            if not reference_present and record_reference is not None:
                findings.append(
                    Finding(label, "record carries a reference latency that is not published")
                )
            if reference_present and record_reference is not None:
                if f"{float(record_reference):.4f}" != row[5]:
                    findings.append(
                        Finding(
                            label,
                            f"published reference {row[5]} != recorded "
                            f"{float(record_reference):.4f}",
                        )
                    )
    declared = {directory.name for directory in benchmark_dirs(root)}
    for key in sorted(grouped, key=str):
        for path, record in grouped[key]:
            owner = record_owner(path, record, declared)
            if (key, owner) in matched:
                continue
            if owner is not None:
                findings.append(
                    Finding(
                        str(records_dir),
                        f"record {key} of {owner!r} does not correspond to any published row",
                    )
                )


# --------------------------------------------------------------------------------------
# privacy scan
# --------------------------------------------------------------------------------------


def privacy_files(root: Path) -> list[Path]:
    files = [
        path
        for path in sorted((root / "benchmarks").rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and _rel(root, path) != SELF
        and not any(_rel(root, path) == name or name in path.parts for name in INTERNAL_ONLY)
    ]
    files.extend(root / name for name in PRIVACY_PATHS)
    return [path for path in files if path.is_file()]


def check_privacy(root: Path, findings: list[Finding]) -> None:
    for path in privacy_files(root):
        where = _rel(root, path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.startswith("#!"):
                continue
            for label, pattern in PRIVACY_PATTERNS:
                match = pattern.search(line)
                if match:
                    findings.append(
                        Finding(
                            f"{where}:{lineno}", f"{label} {match.group(0)!r} must not be published"
                        )
                    )


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def check_tree(root: Path, records_dir: Path | None = None) -> list[Finding]:
    findings: list[Finding] = []
    directories = benchmark_dirs(root)
    if not directories:
        findings.append(Finding(_rel(root, root / "benchmarks"), "no benchmark directories found"))
    for operator_dir in directories:
        check_operator_readme(root, operator_dir, findings)
    check_hub(root, findings)
    check_matrix(root, findings)
    check_privacy(root, findings)
    if records_dir is not None:
        check_records(root, records_dir, findings)
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the benchmark tree (default action)",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="also verify every published row against the local records in this directory",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root to check; defaults to the root that contains this script",
    )
    args = parser.parse_args(argv)
    root = (args.root or _harness.project_root()).resolve()
    findings = check_tree(root, args.records)
    for finding in findings:
        print(finding)
    if findings:
        print(f"{len(findings)} benchmark standard violation(s)")
        return 1
    print("benchmark standard: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
