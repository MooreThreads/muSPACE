"""Tests for the tracked metadata shim and generated build version file."""

from importlib import import_module
from pathlib import Path

import pytest

import native_build
from build_backend import _create_build_metadata


def test_build_backend_writes_generated_version_file(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "version.txt").write_text("1.2.3\n", encoding="utf-8")
    package_dir = tmp_path / "mu_space"
    package_dir.mkdir()

    assert _create_build_metadata(tmp_path) == "1.2.3"
    generated = (package_dir / "_version.py").read_text(encoding="utf-8")
    assert '__version__ = "1.2.3"' in generated
    assert '__git_version__ = "unknown"' in generated
    assert not (package_dir / "_build_meta.py").exists()


def test_build_backend_preserves_packaged_generated_version_without_git(tmp_path: Path):
    (tmp_path / "version.txt").write_text("1.2.3\n", encoding="utf-8")
    package_dir = tmp_path / "mu_space"
    package_dir.mkdir()
    generated = package_dir / "_version.py"
    original = '__version__ = "1.2.3"\n__git_version__ = "release-revision"\n'
    generated.write_text(original, encoding="utf-8")

    assert _create_build_metadata(tmp_path) == "1.2.3"
    assert generated.read_text(encoding="utf-8") == original


def test_metadata_shim_reads_the_generated_version_module():
    from mu_space import _build_meta

    expected_version = (
        (Path(__file__).parents[1] / "version.txt").read_text(encoding="utf-8").strip()
    )
    assert _build_meta.__version__ == expected_version
    assert _build_meta.__git_version__


def test_metadata_shim_has_a_source_checkout_fallback(tmp_path: Path, monkeypatch):
    package_dir = tmp_path / "metadata_test_package"
    package_dir.mkdir()
    (package_dir / "__init__.py").touch()
    source = Path(__file__).parents[1] / "mu_space" / "_build_meta.py"
    (package_dir / "_build_meta.py").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    metadata = import_module("metadata_test_package._build_meta")
    assert metadata.__version__ == "0.0.0+unknown"
    assert metadata.__git_version__ == "unknown"


@pytest.mark.parametrize(("value", "expected"), [("31", "mp_31"), ("mp_31", "mp_31")])
def test_native_build_maps_musa_arch_to_mcc(value, expected):
    assert native_build._mcc_offload_arch(value) == expected
