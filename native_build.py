"""AOT build helper for the torch-ABI-independent native module."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from pathlib import Path


def _tvm_ffi_paths() -> tuple[Path, Path, Path]:
    try:
        libinfo = importlib.import_module("tvm_ffi.libinfo")
    except ModuleNotFoundError as exc:  # pragma: no cover - build environment failure
        raise RuntimeError(
            "Building muSPACE requires apache-tvm-ffi>=0.1.9.post3; "
            "install build dependencies with PEP 517."
        ) from exc
    return (
        Path(libinfo.find_include_path()),
        Path(libinfo.find_dlpack_include_path()),
        Path(libinfo.find_libtvm_ffi()),
    )


def build_aot_library(root: Path) -> Path:
    """Compile the package-local FFI library without PyTorch headers or ABI."""
    include_dir, dlpack_dir, tvm_ffi_lib = _tvm_ffi_paths()
    mcc = shutil.which("mcc")
    if mcc is None:
        musa_home = os.environ.get("MUSA_HOME", "/usr/local/musa")
        candidate = Path(musa_home) / "bin" / "mcc"
        if candidate.exists():
            mcc = str(candidate)
    if mcc is None:
        raise RuntimeError("MUSA compiler mcc is required to build the AOT muSPACE wheel")

    sources = [root / source for source in __import__("build_utils").EXTENSION_SOURCES]
    output_dir = root / "mu_space" / "native"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "libmuspace_ops.so"
    command = [
        mcc,
        "-shared",
        "-fPIC",
        "-std=c++17",
        "-O2",
    ]
    musa_arch = os.environ.get("MUSA_ARCH")
    if musa_arch:
        command.append(f"--offload-arch={_mcc_offload_arch(musa_arch)}")
    command.extend([
        *(str(source) for source in sources),
        f"-I{include_dir}",
        f"-I{dlpack_dir}",
        f"-L{tvm_ffi_lib.parent}",
        "-ltvm_ffi",
        "-lmusa",
        "-lmusart",
        "-o",
        str(output),
    ])
    subprocess.run(command, cwd=root, check=True)
    return output


def _mcc_offload_arch(musa_arch: str) -> str:
    """Normalize the public MUSA_ARCH value to mcc's architecture spelling."""
    return musa_arch if musa_arch.startswith("mp_") else f"mp_{musa_arch}"
