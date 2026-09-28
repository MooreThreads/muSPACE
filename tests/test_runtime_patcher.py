import sys
import types

import pytest

from mu_space.patcher import DefaultPatcher, PatchError
from mu_space.patcher.runtime import _MissingLegacyExtension


TARGET = "mmcv.ops.multi_scale_deform_attn"
ATTR = "multi_scale_deformable_attn_pytorch"
IMPORTED = "projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention"
UNIAD_IMPORTED = "projects.mmdet3d_plugin.uniad.modules.temporal_self_attention"
EXT_LOADER = "mmcv.utils.ext_loader"


def _install_fake_module(monkeypatch, name: str):
    module = types.ModuleType(name)

    def original(*args, **kwargs):
        return "original"

    setattr(module, ATTR, original)
    monkeypatch.setitem(sys.modules, name, module)
    return module, original


def _install_fake_ext_loader(monkeypatch):
    module = types.ModuleType(EXT_LOADER)

    def load_ext(name, functions):
        raise ImportError(name)

    module.load_ext = load_ext
    monkeypatch.setitem(sys.modules, EXT_LOADER, module)
    return module, load_ext


def test_default_patcher_replaces_mmcv_msda_and_restores(monkeypatch):
    module, original = _install_fake_module(monkeypatch, TARGET)
    patcher = DefaultPatcher()

    report = patcher.apply()

    assert report.action == "applied"
    assert report.strict is False
    assert report.patched_modules == (TARGET,)
    assert getattr(module, ATTR) is not original
    assert getattr(module, ATTR).__name__ == "muspace_multi_scale_deformable_attn_compat"

    restored = patcher.restore()
    assert restored.action == "restored"
    assert getattr(module, ATTR) is original


def test_default_patcher_updates_already_imported_bevformer_modules(monkeypatch):
    module, _ = _install_fake_module(monkeypatch, TARGET)
    imported, imported_original = _install_fake_module(monkeypatch, IMPORTED)
    patcher = DefaultPatcher()

    report = patcher.apply()

    assert report.patched_modules == (TARGET, IMPORTED)
    assert getattr(module, ATTR) is getattr(imported, ATTR)
    assert getattr(imported, ATTR) is not imported_original


def test_default_patcher_updates_already_imported_uniad_modules(monkeypatch):
    module, _ = _install_fake_module(monkeypatch, TARGET)
    imported, imported_original = _install_fake_module(monkeypatch, UNIAD_IMPORTED)
    patcher = DefaultPatcher()

    report = patcher.apply()

    assert report.patched_modules == (TARGET, UNIAD_IMPORTED)
    assert getattr(module, ATTR) is getattr(imported, ATTR)
    assert getattr(imported, ATTR) is not imported_original


def test_default_patcher_strict_mode_from_env(monkeypatch):
    module, _ = _install_fake_module(monkeypatch, TARGET)
    monkeypatch.setenv("MUSPACE_STRICT_PATCH", "1")
    patcher = DefaultPatcher()

    report = patcher.apply()

    assert report.strict is True
    assert getattr(module, ATTR).__name__ == "muspace_multi_scale_deformable_attn_strict"


def test_restore_releases_legacy_extension_guard(monkeypatch):
    _install_fake_module(monkeypatch, TARGET)
    ext_loader, original_load_ext = _install_fake_ext_loader(monkeypatch)
    patcher = DefaultPatcher()

    patcher.apply()
    assert ext_loader.load_ext is not original_load_ext
    assert hasattr(ext_loader, "_muspace_original_load_ext")

    patcher.restore()

    assert ext_loader.load_ext is original_load_ext
    assert not hasattr(ext_loader, "_muspace_original_load_ext")

    # A second restore must not reinstall or corrupt the released state.
    patcher.restore()
    assert ext_loader.load_ext is original_load_ext


def test_legacy_extension_guard_returns_placeholder_for_failed_load(monkeypatch):
    _install_fake_module(monkeypatch, TARGET)
    ext_loader, _ = _install_fake_ext_loader(monkeypatch)
    patcher = DefaultPatcher()
    patcher.apply()

    placeholder = ext_loader.load_ext("_ext", ["nms"])

    assert isinstance(placeholder, _MissingLegacyExtension)
    patcher.restore()


def test_missing_legacy_extension_stays_probe_friendly():
    placeholder = _MissingLegacyExtension("_ext", ["nms"])

    assert placeholder.get_compiler_version() == "unavailable"
    assert hasattr(placeholder, "nms")
    with pytest.raises(AttributeError):
        placeholder.__deepcopy__
    with pytest.raises(PatchError):
        placeholder.nms()
