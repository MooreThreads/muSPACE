"""Tests for the benchmark documentation and record checker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import _harness  # noqa: E402
import check_benchmarks as checker  # noqa: E402

REFERENCE_DEVICE = next(iter(_harness.REFERENCE_BANDWIDTH_GBPS))


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def operator_readme(rows: list[str], cases: tuple[str, ...] = ("demo-a",)) -> str:
    case_rows = "\n".join(f"| `{case}` | shapes | `float32` | `2333` |" for case in cases)
    return f"""# Demo benchmark

## Scope

Times the demo operator on one case.

## Cases

| Case | Parameters | dtype | Seed |
| --- | --- | --- | ---: |
{case_rows}

## Test environment

- Measured device: {_harness.MEASURED_DEVICE}
- Reference device: {REFERENCE_DEVICE}

## Measurement

Scope `kernel-only`, device events, warmup 20, iterations 100, repeats 5, seed 2333.

## Run

```bash
PYTHONPATH=. python benchmarks/demo/benchmark_demo.py --case demo-a --output .tmp/benchmarks/demo.json
```

## Results

{_harness.render_table(rows)}

## Acceptance

Speedup >= 1.00x on the published case.

## Notes

Synthetic inputs only.
"""


def hub_readme(directories: tuple[str, ...] = ("demo",)) -> str:
    coverage = "\n".join(f"| `{name}` | [`{name}/`]({name}/README.md) |" for name in directories)
    devices = "\n".join(
        f"| {device} | role | {bandwidth:g} GB/s |"
        for device, bandwidth in _harness.DEVICE_BANDWIDTH_GBPS.items()
    )
    return f"""# muSPACE operator benchmarks

## Coverage

| Operator | Benchmark |
| --- | --- |
{coverage}

## Comparison rules

| Device | Role | Nominal memory bandwidth |
| --- | --- | ---: |
{devices}

## Reproducing benchmark inputs

Records stay under `.tmp/`.
"""


def matrix_doc(directories: tuple[str, ...] = ("demo",)) -> str:
    rows = "\n".join(
        f"| operator {name} | `api` | [benchmark](../benchmarks/{name}/README.md) |"
        for name in directories
    )
    return f"""# Operator matrix

| Operator | Canonical API | Benchmark |
| --- | --- | --- |
{rows}
"""


def build_tree(root: Path, rows: list[str] | None = None, cases: tuple[str, ...] = ("demo-a",)):
    rows = rows if rows is not None else [demo_row()]
    write(root / "benchmarks/README.md", hub_readme())
    write(root / "docs/operator_matrix.md", matrix_doc())
    write(root / "docs/benchmarking.md", "# Operator benchmark standard\n")
    write(root / "benchmarks/demo/README.md", operator_readme(rows, cases))
    write(root / "benchmarks/demo/demo_case.py", "def make_case():\n    return None\n")
    write(root / "benchmarks/demo/benchmark_demo.py", "def main():\n    return 0\n")
    return root


def demo_row(p50: float = 1.0, reference: float | None = None, case: str = "demo-a") -> str:
    return _harness.render_row(
        case=case,
        dtype="float32",
        scope="kernel-only",
        s5000_p50_ms=p50,
        reference_p50_ms=reference,
    )


def messages(findings) -> str:
    return "\n".join(str(finding) for finding in findings)


def test_valid_tree_has_no_findings(tmp_path: Path) -> None:
    build_tree(tmp_path)
    assert checker.check_tree(tmp_path) == []


def test_valid_tree_with_reference_row(tmp_path: Path) -> None:
    build_tree(tmp_path, rows=[demo_row(p50=2.0, reference=1.5)])
    assert checker.check_tree(tmp_path) == []


def test_missing_section(tmp_path: Path) -> None:
    build_tree(tmp_path)
    readme = tmp_path / "benchmarks/demo/README.md"
    readme.write_text(readme.read_text(encoding="utf-8").replace("## Notes", "## Extra"), "utf-8")
    findings = checker.check_tree(tmp_path)
    assert "missing section ## Notes" in messages(findings)


def test_header_mismatch(tmp_path: Path) -> None:
    build_tree(tmp_path)
    readme = tmp_path / "benchmarks/demo/README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("| Case | dtype |", "| case | dtype |"),
        "utf-8",
    )
    assert "canonical header" in messages(checker.check_tree(tmp_path))


def test_normalized_latency_math(tmp_path: Path) -> None:
    build_tree(tmp_path)
    readme = tmp_path / "benchmarks/demo/README.md"
    text = readme.read_text(encoding="utf-8")
    row = demo_row(p50=1.0)
    published = _harness.format_ms(_harness.normalized_latency_ms(1.0, REFERENCE_DEVICE))
    broken = row.replace(f"| {published} |", "| 9.9999 |")
    assert broken != row
    readme.write_text(text.replace(row, broken), "utf-8")
    assert "normalized latency" in messages(checker.check_tree(tmp_path))


def test_speedup_math(tmp_path: Path) -> None:
    rows = [demo_row(p50=2.0, reference=1.5)]
    build_tree(tmp_path, rows=rows)
    readme = tmp_path / "benchmarks/demo/README.md"
    text = readme.read_text(encoding="utf-8")
    ratio = _harness.format_speedup(_harness.speedup(1.5, 2.0, REFERENCE_DEVICE))
    assert f"| {ratio} |" in text
    readme.write_text(text.replace(f"| {ratio} |", "| 9.99x |"), "utf-8")
    assert "speedup" in messages(checker.check_tree(tmp_path))


def test_reference_row_must_not_say_unavailable(tmp_path: Path) -> None:
    rows = [demo_row(p50=2.0, reference=1.5)]
    build_tree(tmp_path, rows=rows)
    readme = tmp_path / "benchmarks/demo/README.md"
    text = readme.read_text(encoding="utf-8")
    ratio = _harness.format_speedup(_harness.speedup(1.5, 2.0, REFERENCE_DEVICE))
    readme.write_text(
        text.replace(f"| {ratio} |  |", f"| {ratio} | reference unavailable |"), "utf-8"
    )
    assert "note says unavailable" in messages(checker.check_tree(tmp_path))


def test_row_without_reference_must_say_unavailable(tmp_path: Path) -> None:
    build_tree(tmp_path)
    readme = tmp_path / "benchmarks/demo/README.md"
    text = readme.read_text(encoding="utf-8")
    readme.write_text(text.replace("reference unavailable", ""), "utf-8")
    assert "must say 'reference unavailable'" in messages(checker.check_tree(tmp_path))


def test_undeclared_case(tmp_path: Path) -> None:
    build_tree(tmp_path, rows=[demo_row(case="demo-b")], cases=("demo-a",))
    assert "not declared in ## Cases" in messages(checker.check_tree(tmp_path))


def test_privacy_patterns(tmp_path: Path) -> None:
    build_tree(tmp_path)
    write(
        tmp_path / "benchmarks/demo/demo_case.py",
        "# measured from /data/home/someone/workspace on 203.0.113.7\n"
        "# torch 2.5.0, commit a1b2c3d, owner someone@example.com, 2026-09-30\n",
    )
    text = messages(checker.check_tree(tmp_path))
    for label in (
        "absolute filesystem path",
        "IP address",
        "account-qualified host",
        "date stamp",
        "source revision hash",
        "toolchain or build version",
        "component version number",
    ):
        assert label in text, label


def test_issue_identifier_is_rejected(tmp_path: Path) -> None:
    build_tree(tmp_path)
    write(tmp_path / "benchmarks/demo/notes.py", "# see PROJ-4210 and ABC-12\n")
    assert "issue identifier" in messages(checker.check_tree(tmp_path))


def test_shebang_is_not_a_violation(tmp_path: Path) -> None:
    build_tree(tmp_path)
    write(tmp_path / "benchmarks/demo/notes.py", "#!/usr/bin/env python3\n")
    assert checker.check_tree(tmp_path) == []


def test_committed_records_are_rejected(tmp_path: Path) -> None:
    build_tree(tmp_path)
    write(tmp_path / "benchmarks/demo/results.json", "{}\n")
    assert "must not be committed" in messages(checker.check_tree(tmp_path))


def test_case_module_is_required(tmp_path: Path) -> None:
    build_tree(tmp_path)
    (tmp_path / "benchmarks/demo/demo_case.py").unlink()
    assert "case module" in messages(checker.check_tree(tmp_path))


def test_hub_must_register_devices(tmp_path: Path) -> None:
    build_tree(tmp_path)
    hub = tmp_path / "benchmarks/README.md"
    hub.write_text(hub.read_text(encoding="utf-8").replace("1600 GB/s", "1 GB/s"), "utf-8")
    assert "registered bandwidth" in messages(checker.check_tree(tmp_path))


def test_matrix_must_cover_every_directory(tmp_path: Path) -> None:
    build_tree(tmp_path)
    write(
        tmp_path / "benchmarks/other/README.md",
        operator_readme([demo_row(case="other-a")], ("other-a",)),
    )
    write(tmp_path / "benchmarks/other/other_case.py", "")
    write(tmp_path / "benchmarks/other/benchmark_other.py", "")
    assert "operator matrix does not cover benchmark other" in messages(
        checker.check_tree(tmp_path)
    )


def record(case: str = "demo-a", p50: float = 1.0, reference: float | None = None) -> dict:
    return {
        "operator": "demo",
        "case": case,
        "dtype": "float32",
        "scope": "kernel-only",
        "statistics": {"p50_ms": p50},
        "reference": {"device": REFERENCE_DEVICE, "p50_ms": reference},
    }


def records_dir(tmp_path: Path, payloads: list[dict]) -> Path:
    directory = tmp_path / "records"
    for index, payload in enumerate(payloads):
        write(directory / f"record-{index}.json", json.dumps(payload))
    return directory


def test_records_match_published_rows(tmp_path: Path) -> None:
    build_tree(tmp_path)
    directory = records_dir(tmp_path, [record()])
    assert checker.check_tree(tmp_path, directory) == []


def test_records_with_published_reference(tmp_path: Path) -> None:
    build_tree(tmp_path, rows=[demo_row(p50=2.0, reference=1.5)])
    directory = records_dir(tmp_path, [record(p50=2.0, reference=1.5)])
    assert checker.check_tree(tmp_path, directory) == []


def test_missing_record_is_reported(tmp_path: Path) -> None:
    build_tree(tmp_path)
    directory = records_dir(tmp_path, [record(case="other")])
    assert "no local record found" in messages(checker.check_tree(tmp_path, directory))


def test_record_latency_mismatch(tmp_path: Path) -> None:
    build_tree(tmp_path)
    directory = records_dir(tmp_path, [record(p50=1.5)])
    assert "published latency" in messages(checker.check_tree(tmp_path, directory))


def test_unpublished_reference_is_reported(tmp_path: Path) -> None:
    build_tree(tmp_path)
    directory = records_dir(tmp_path, [record(reference=1.5)])
    assert "not published" in messages(checker.check_tree(tmp_path, directory))


def test_conflicting_records_are_reported(tmp_path: Path) -> None:
    build_tree(tmp_path)
    directory = records_dir(tmp_path, [record(p50=1.0), record(p50=1.2345)])
    assert "conflicting p50 values" in messages(checker.check_tree(tmp_path, directory))


def test_missing_records_directory(tmp_path: Path) -> None:
    build_tree(tmp_path)
    findings = checker.check_tree(tmp_path, tmp_path / "missing")
    assert "record directory does not exist" in messages(findings)


def test_repository_tree_is_clean() -> None:
    """The checked-in benchmark tree must satisfy its own standard."""

    findings = checker.check_tree(REPO_ROOT)
    assert findings == [], messages(findings)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
