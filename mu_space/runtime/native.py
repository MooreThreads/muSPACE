"""Loader for the package AOT TVM FFI native module.

The module intentionally has no torch C++ extension import.  ``tvm_ffi``
adapts PyTorch tensors through DLPack at the Python boundary while the shared
object sees only DLPack tensor views.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any


class NativeModuleUnavailable(RuntimeError):
    """The wheel does not contain a usable AOT native module."""


@lru_cache(maxsize=1)
def load_native_module() -> Any:
    try:
        import tvm_ffi
    except ModuleNotFoundError as exc:
        raise NativeModuleUnavailable(
            "muSPACE native operators require apache-tvm-ffi>=0.1.9.post3"
        ) from exc

    library = Path(files("mu_space").joinpath("native/libmuspace_ops.so"))
    if not library.is_file():
        raise NativeModuleUnavailable(
            "muSPACE AOT native library is missing from this installation; "
            "install a wheel built for the active MUSA toolkit."
        )
    try:
        return tvm_ffi.load_module(str(library))
    except Exception as exc:  # pragma: no cover - platform loader dependent
        raise NativeModuleUnavailable(
            f"unable to load muSPACE AOT native library {library}: {exc}"
        ) from exc


def native_function(name: str) -> Any:
    module = load_native_module()
    try:
        return module.get_function(name)
    except Exception as exc:
        raise NativeModuleUnavailable(f"muSPACE AOT library does not export {name!r}") from exc


class NativeModuleProxy:
    """Attribute-compatible view used by the operator wrappers."""

    def __getattr__(self, name: str) -> Any:
        return native_function(name)


@lru_cache(maxsize=1)
def native_module_proxy() -> NativeModuleProxy:
    load_native_module()
    return NativeModuleProxy()


def native_available() -> bool:
    try:
        load_native_module()
    except NativeModuleUnavailable:
        return False
    return True


def native_error() -> str | None:
    try:
        load_native_module()
    except NativeModuleUnavailable as exc:
        return str(exc)
    return None
