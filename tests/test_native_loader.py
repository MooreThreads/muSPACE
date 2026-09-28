import sys
import types

import pytest

from mu_space.runtime import native


def test_missing_aot_library_is_explained(monkeypatch):
    native.load_native_module.cache_clear()
    monkeypatch.setitem(
        sys.modules, "tvm_ffi", types.SimpleNamespace(load_module=lambda path: None)
    )
    monkeypatch.setattr(native, "files", lambda package: __import__("pathlib").Path("/missing"))
    assert "AOT native library is missing" in native.native_error()
    assert not native.native_available()
    native.load_native_module.cache_clear()


def test_load_failure_is_explained(monkeypatch, tmp_path):
    library = tmp_path / "native" / "libmuspace_ops.so"
    library.parent.mkdir()
    library.touch()
    native.load_native_module.cache_clear()
    monkeypatch.setitem(
        sys.modules,
        "tvm_ffi",
        types.SimpleNamespace(load_module=lambda path: (_ for _ in ()).throw(OSError("bad ABI"))),
    )
    monkeypatch.setattr(native, "files", lambda package: tmp_path)
    assert "unable to load muSPACE AOT native library" in native.native_error()
    native.load_native_module.cache_clear()


def test_missing_export_is_explained(monkeypatch, tmp_path):
    library = tmp_path / "native" / "libmuspace_ops.so"
    library.parent.mkdir()
    library.touch()
    native.load_native_module.cache_clear()

    class Module:
        def get_function(self, name):
            raise KeyError(name)

    monkeypatch.setitem(
        sys.modules, "tvm_ffi", types.SimpleNamespace(load_module=lambda path: Module())
    )
    monkeypatch.setattr(native, "files", lambda package: tmp_path)
    with pytest.raises(native.NativeModuleUnavailable, match="does not export 'missing_op'"):
        native.native_function("missing_op")
    native.load_native_module.cache_clear()
