"""PEP 517 backend with generated package metadata."""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import build_meta as _setuptools_backend
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel

from build_utils import build_version_string
from native_build import build_aot_library

_ROOT = Path(__file__).parent.resolve()


class _MusaPlatformWheel(bdist_wheel):
    """Keep the Linux platform tag while avoiding a CPython ABI tag.

    ``libmuspace_ops.so`` is loaded through TVM FFI/DLPack and does not link
    against libpython or libtorch.  The wheel is still platform-specific due
    to the MUSA shared library, but it is not CPython-3.10 ABI-specific.
    """

    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


def _create_build_metadata(root: Path = _ROOT) -> str:
    """Create generated package metadata from version.txt and the current checkout."""
    version_file = root / "version.txt"
    base_version = version_file.read_text(encoding="utf-8").strip()
    version, revision = build_version_string(
        base_version=base_version,
        cwd=root,
        dev_suffix=os.environ.get("MUSPACE_DEV_RELEASE_SUFFIX", ""),
        local_version=os.environ.get("MUSPACE_LOCAL_VERSION"),
    )
    package_dir = root / "mu_space"
    target = package_dir / "_version.py"
    if target.exists() and not (root / ".git").exists():
        return version
    target.write_text(
        '"""Generated package version metadata. Do not edit."""\n\n'
        f'__version__ = "{version}"\n'
        f'__git_version__ = "{revision}"\n',
        encoding="utf-8",
    )
    return version


_create_build_metadata()


def _build_platform_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Force a platform wheel tag for the package-local MUSA shared object."""
    original_has_ext_modules = Distribution.has_ext_modules
    original_get_command_class = Distribution.get_command_class

    def get_command_class(distribution, command):
        if command == "bdist_wheel":
            return _MusaPlatformWheel
        return original_get_command_class(distribution, command)

    Distribution.has_ext_modules = lambda self: True  # type: ignore[method-assign]
    Distribution.get_command_class = get_command_class  # type: ignore[method-assign]
    try:
        return _setuptools_backend.build_wheel(wheel_directory, config_settings, metadata_directory)
    finally:
        Distribution.has_ext_modules = original_has_ext_modules
        Distribution.get_command_class = original_get_command_class


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build the AOT FFI library before setuptools collects package data.

    ``libmuspace_ops.so`` is deliberately built without importing torch or
    torch_musa.  It is loaded through ``tvm_ffi`` at runtime.
    """
    build_aot_library(_ROOT)
    return _build_platform_wheel(wheel_directory, config_settings, metadata_directory)


build_sdist = _setuptools_backend.build_sdist


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    build_aot_library(_ROOT)
    return _setuptools_backend.build_editable(wheel_directory, config_settings, metadata_directory)


get_requires_for_build_wheel = _setuptools_backend.get_requires_for_build_wheel
get_requires_for_build_sdist = _setuptools_backend.get_requires_for_build_sdist
get_requires_for_build_editable = getattr(
    _setuptools_backend,
    "get_requires_for_build_editable",
    _setuptools_backend.get_requires_for_build_wheel,
)
prepare_metadata_for_build_wheel = _setuptools_backend.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = getattr(
    _setuptools_backend,
    "prepare_metadata_for_build_editable",
    _setuptools_backend.prepare_metadata_for_build_wheel,
)
