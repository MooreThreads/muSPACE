import importlib
import sys
import types

import pytest
import torch

from mu_space.patcher import DefaultPatcher, PatchError, QwenDrivePatcher, SparseDrivePatcher
from mu_space.patcher import runtime
from mu_space.patcher.runtime import SPARSEDRIVE_CUDA_SHIM, _MissingLegacyExtension


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


def _install_fake_qwendrive(monkeypatch):
    attention = types.ModuleType("qwen_drive_perception.attention")
    ops = types.ModuleType("qwen_drive_perception.ops")
    view = types.ModuleType("qwen_drive_perception.view_transform")

    def original_msda(*args, **kwargs):
        return ("pytorch", args, kwargs)

    def original_forward(*args, **kwargs):
        return "forward-only"

    def original_pool(*args, **kwargs):
        return "upstream-pool"

    class Uni3DVoxelPoolDepth:
        frustum_range = (0, 0, 1, 4, 4, 3)
        frustum_size = (2, 2, 1)

        def __init__(self):
            self._frustum = None

        @property
        def frustum(self):
            return "upstream-frustum"

    attention.multi_scale_deformable_attn_cuda = original_msda
    ops.ms_deform_attn_bf16_forward = original_forward
    ops.voxel_pool_depth = original_pool
    view.Uni3DVoxelPoolDepth = Uni3DVoxelPoolDepth
    for module in (attention, ops, view):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return attention, ops, view, original_msda, original_forward, original_pool


def test_qwendrive_patcher_rebinds_and_restores(monkeypatch):
    attention, ops, view, msda, forward, pool = _install_fake_qwendrive(monkeypatch)
    frustum = view.Uni3DVoxelPoolDepth.frustum
    patcher = QwenDrivePatcher()

    first = patcher.apply()
    assert first.action == "applied"
    assert first.strict is False
    assert attention.multi_scale_deformable_attn_cuda is not msda
    assert ops.ms_deform_attn_bf16_forward is attention.multi_scale_deformable_attn_cuda
    assert ops.voxel_pool_depth is not pool
    assert view.Uni3DVoxelPoolDepth.frustum is not frustum

    second = patcher.apply()
    assert second.action == "already-applied"
    cpu = types.SimpleNamespace(
        dtype=object(), device=types.SimpleNamespace(type="cpu"), is_musa=False, is_cuda=False
    )
    assert attention.multi_scale_deformable_attn_cuda(cpu, 1, 2, 3, 4, 64)[0] == "pytorch"

    restored = patcher.restore()
    assert restored.action == "restored"
    assert attention.multi_scale_deformable_attn_cuda is msda
    assert ops.ms_deform_attn_bf16_forward is forward
    assert ops.voxel_pool_depth is pool
    assert view.Uni3DVoxelPoolDepth.frustum is frustum

    patcher.restore()
    assert attention.multi_scale_deformable_attn_cuda is msda
    assert view.Uni3DVoxelPoolDepth.frustum is frustum


def test_qwendrive_msda_wrapper_routes_by_device(monkeypatch):
    attention, ops, _view, _msda, _forward, _pool = _install_fake_qwendrive(monkeypatch)
    captured = {}

    def fake_msda(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "native"

    monkeypatch.setattr("mu_space.ops.multi_scale_deformable_attn", fake_msda)
    patcher = QwenDrivePatcher()
    patcher.apply()
    wrapper = attention.multi_scale_deformable_attn_cuda

    musa = types.SimpleNamespace(
        dtype=torch.bfloat16, device=types.SimpleNamespace(type="musa"), is_musa=True, is_cuda=False
    )
    assert wrapper(musa, "shapes", "starts", "locations", "weights", im2col_step=64) == "native"
    assert captured["kwargs"]["im2col_step"] == 64
    assert captured["kwargs"]["backend"] == "auto"
    assert captured["args"][1] == "shapes"

    with pytest.raises(TypeError, match="bfloat16"):
        wrapper(
            types.SimpleNamespace(
                dtype=torch.float32,
                device=types.SimpleNamespace(type="musa"),
                is_musa=True,
                is_cuda=False,
            ),
            None,
            None,
            None,
            None,
            64,
        )

    cpu = types.SimpleNamespace(
        dtype=torch.float32, device=types.SimpleNamespace(type="cpu"), is_musa=False, is_cuda=False
    )
    assert wrapper(cpu, "shapes", "starts", "locations", "weights", 64)[0] == "pytorch"
    assert ops.ms_deform_attn_bf16_forward is wrapper
    patcher.restore()


def test_qwendrive_frustum_property_uses_musa_when_available(monkeypatch):
    _attention, _ops, view, *_rest = _install_fake_qwendrive(monkeypatch)
    patcher = QwenDrivePatcher()
    patcher.apply()
    instance = view.Uni3DVoxelPoolDepth()

    monkeypatch.setattr(runtime, "_qwendrive_frustum_device", lambda: "cpu")
    grid = instance.frustum
    assert grid.shape == (2, 2, 2, 3)
    assert grid.device.type == "cpu"
    assert grid.dtype == torch.float32
    assert instance.frustum is grid
    patcher.restore()
    assert instance.frustum == "upstream-frustum"


def test_qwendrive_frustum_follows_parameter_device(monkeypatch):
    _attention, _ops, view, *_rest = _install_fake_qwendrive(monkeypatch)
    patcher = QwenDrivePatcher()
    patcher.apply()
    instance = view.Uni3DVoxelPoolDepth()
    anchor = torch.zeros(1)

    def parameters():
        yield anchor

    instance.parameters = parameters
    # The helper would ask for MUSA. The grid must follow the parameter instead.
    monkeypatch.setattr(runtime, "_qwendrive_frustum_device", lambda: "musa")
    grid = instance.frustum
    assert grid.device.type == "cpu"
    assert grid.dtype == torch.float32
    patcher.restore()


def test_qwendrive_frustum_cache_distinguishes_device_index():
    assert runtime._qwendrive_same_device(torch.device("cuda:0"), torch.device("cuda:1")) is False
    assert runtime._qwendrive_same_device(torch.device("cuda:0"), torch.device("cuda:0")) is True
    assert runtime._qwendrive_same_device(torch.device("cpu"), torch.device("cpu")) is True
    assert runtime._qwendrive_same_device(None, torch.device("cpu")) is False


def test_qwendrive_patcher_rejects_a_partial_package(monkeypatch):
    attention, ops, view, *_rest = _install_fake_qwendrive(monkeypatch)
    del ops.voxel_pool_depth
    patcher = QwenDrivePatcher()

    with pytest.raises(PatchError, match="voxel_pool_depth"):
        patcher.apply()

    assert callable(attention.multi_scale_deformable_attn_cuda)
    assert attention.multi_scale_deformable_attn_cuda(None)[0] == "pytorch"
    assert isinstance(view.Uni3DVoxelPoolDepth.frustum, property)


def test_missing_legacy_extension_stays_probe_friendly():
    placeholder = _MissingLegacyExtension("_ext", ["nms"])

    assert placeholder.get_compiler_version() == "unavailable"
    assert hasattr(placeholder, "nms")
    with pytest.raises(AttributeError):
        placeholder.__deepcopy__
    with pytest.raises(PatchError):
        placeholder.nms()


SPARSEDRIVE_OPS = "projects.mmdet3d_plugin.ops"
SPARSEDRIVE_EXT = "projects.mmdet3d_plugin.ops.deformable_aggregation_ext"
SPARSEDRIVE_OPS_INIT = (
    "from .deformable_aggregation import DeformableAggregationFunction\n"
    "\n"
    "def deformable_aggregation_function(*args, **kwargs):\n"
    "    return DeformableAggregationFunction.apply(*args)\n"
)


def _install_sparsedrive_checkout(tmp_path, monkeypatch, ops_init: str):
    """Write a minimal SparseDrive package tree and put it on the import path."""

    package = tmp_path / "projects" / "mmdet3d_plugin"
    ops = package / "ops"
    ops.mkdir(parents=True)
    (tmp_path / "projects" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (ops / "deformable_aggregation.py").write_text(
        "from . import deformable_aggregation_ext\n"
        "\n"
        "class DeformableAggregationFunction:\n"
        "    @staticmethod\n"
        "    def apply(*args):\n"
        "        return deformable_aggregation_ext.deformable_aggregation_forward(*args)\n",
        encoding="utf-8",
    )
    (ops / "__init__.py").write_text(ops_init, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in list(sys.modules):
        if name == "projects" or name.startswith("projects."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return ops


def test_sparsedrive_patcher_tolerates_unbuilt_cuda_extension(tmp_path, monkeypatch):
    _install_sparsedrive_checkout(tmp_path, monkeypatch, SPARSEDRIVE_OPS_INIT)
    patcher = SparseDrivePatcher()

    report = patcher.apply(strict=True)

    assert report.action == "applied"
    assert SPARSEDRIVE_OPS in report.patched_modules
    ops = sys.modules[SPARSEDRIVE_OPS]
    assert ops.deformable_aggregation_function.__name__ == "muspace_deformable_aggregation_strict"
    assert SPARSEDRIVE_EXT in sys.modules

    # Re-applying must not install a second placeholder or lose ownership.
    assert patcher.apply(strict=True).action == "already-applied"

    # The placeholder stays probe friendly and fails only on a real call.
    placeholder = sys.modules[SPARSEDRIVE_EXT]
    assert hasattr(placeholder, "deformable_aggregation_forward")
    with pytest.raises(AttributeError):
        placeholder.__deepcopy__
    with pytest.raises(PatchError):
        placeholder.deformable_aggregation_forward()

    patcher.restore()

    assert SPARSEDRIVE_EXT not in sys.modules
    assert ops.deformable_aggregation_function.__name__ == "deformable_aggregation_function"


def test_sparsedrive_patcher_keeps_unrelated_import_errors(tmp_path, monkeypatch):
    _install_sparsedrive_checkout(
        tmp_path, monkeypatch, "import definitely_missing_openmmlab_dependency\n"
    )
    patcher = SparseDrivePatcher()

    with pytest.raises(PatchError, match="SparseDrive ops package import failed"):
        patcher.apply()

    # A missing framework dependency must never be masked by the extension placeholder.
    assert SPARSEDRIVE_EXT not in sys.modules


SPARSEDRIVE_MODELS = "projects.mmdet3d_plugin.models"
SPARSEDRIVE_ATTENTION = "projects.mmdet3d_plugin.models.attention"
SPARSEDRIVE_ATTENTION_SOURCE = """
import torch


class FlashAttention(torch.nn.Module):
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, kv, causal=False, key_padding_mask=None):
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        return q, None
"""


def _install_sparsedrive_attention(
    tmp_path, monkeypatch, source: str = SPARSEDRIVE_ATTENTION_SOURCE
):
    """Add a models package with an attention module that asserts a CUDA device."""

    models = tmp_path / "projects" / "mmdet3d_plugin" / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "__init__.py").write_text("", encoding="utf-8")
    (models / "attention.py").write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in list(sys.modules):
        if name == "projects" or name.startswith("projects."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return models


def _install_fake_mate(monkeypatch):
    """Stand in for MATE's variable-length attention entry point."""

    module = types.ModuleType("mate")
    calls = []

    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=False,
    ):
        calls.append(
            {
                "q": tuple(q.shape),
                "k": tuple(k.shape),
                "cu_seqlens_k": [int(x) for x in cu_seqlens_k],
                "max_seqlen_k": max_seqlen_k,
                "softmax_scale": softmax_scale,
                "causal": causal,
            }
        )
        return q.clone()

    module.flash_attn_varlen_func = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "mate", module)
    return module, calls


def test_sparsedrive_patcher_bridges_flash_attention_through_mate(tmp_path, monkeypatch):
    import torch

    from mu_space.patcher import runtime

    _install_sparsedrive_checkout(tmp_path, monkeypatch, SPARSEDRIVE_OPS_INIT)
    _install_sparsedrive_attention(tmp_path, monkeypatch)
    _, calls = _install_fake_mate(monkeypatch)
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    patcher = SparseDrivePatcher()
    report = patcher.apply()

    assert SPARSEDRIVE_ATTENTION in report.patched_modules
    assert "flash_attn" in sys.modules
    interface = sys.modules["flash_attn.flash_attn_interface"]
    assert interface.flash_attn_unpadded_kvpacked_func is runtime._mate_kvpacked_func
    assert sys.modules["flash_attn.bert_padding"].unpad_input is runtime._unpad_input

    attention = sys.modules[SPARSEDRIVE_ATTENTION]
    module_instance = attention.FlashAttention(softmax_scale=0.125)
    q = torch.randn(2, 5, 4, 8, dtype=torch.float16)
    kv = torch.randn(2, 5, 2, 4, 8, dtype=torch.float16)

    output, weights = module_instance(q, kv, causal=True)

    assert weights is None
    assert output.shape == q.shape
    assert calls[0]["q"] == (10, 4, 8)
    assert calls[0]["k"] == (10, 4, 8)
    assert calls[0]["causal"] is True
    assert calls[0]["softmax_scale"] == 0.125

    patcher.restore()

    assert "flash_attn" not in sys.modules
    assert "flash_attn.flash_attn_interface" not in sys.modules
    assert attention.FlashAttention.forward.__name__ == "forward"


def test_sparsedrive_patcher_leaves_attention_alone_without_mate(tmp_path, monkeypatch):
    _install_sparsedrive_checkout(tmp_path, monkeypatch, SPARSEDRIVE_OPS_INIT)
    _install_sparsedrive_attention(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "mate", None)
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    patcher = SparseDrivePatcher()
    report = patcher.apply()

    assert SPARSEDRIVE_OPS in report.patched_modules
    assert SPARSEDRIVE_ATTENTION not in report.patched_modules
    assert "flash_attn" not in sys.modules
    patcher.restore()


def test_mate_kvpacked_func_rejects_dropout():
    import torch

    from mu_space.patcher import runtime

    with pytest.raises(PatchError, match="does not implement dropout"):
        runtime._mate_kvpacked_func(
            torch.zeros(4, 2, 8, dtype=torch.float16),
            torch.zeros(4, 2, 2, 8, dtype=torch.float16),
            dropout_p=0.1,
        )


def test_sparsedrive_patcher_revalidates_the_cached_ops_probe(tmp_path, monkeypatch):
    ops_init = (
        "from . import feature_maps_format\n"
        "from .deformable_aggregation import DeformableAggregationFunction\n"
        "\n"
        "def deformable_aggregation_function(*args, **kwargs):\n"
        "    return DeformableAggregationFunction.apply(*args)\n"
    )
    ops_dir = _install_sparsedrive_checkout(tmp_path, monkeypatch, ops_init)
    (ops_dir / "feature_maps_format.py").write_text("VALUE = 'formatted'\n", encoding="utf-8")
    models = tmp_path / "projects" / "mmdet3d_plugin" / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "__init__.py").write_text("", encoding="utf-8")
    (models / "probe.py").write_text("DAF_VALID = False\n", encoding="utf-8")
    probe = importlib.import_module("projects.mmdet3d_plugin.models.probe")
    assert probe.DAF_VALID is False
    assert not hasattr(probe, "feature_maps_format")

    patcher = SparseDrivePatcher()
    report = patcher.apply()

    # The checkout caches its import-time probe, so the patch has to clear every trace of it.
    assert probe.DAF_VALID is True
    ops_module = sys.modules[SPARSEDRIVE_OPS]
    assert probe.feature_maps_format is ops_module.feature_maps_format
    assert "projects.mmdet3d_plugin.models.probe" in report.patched_modules

    patcher.restore()
    assert probe.DAF_VALID is False
    assert not hasattr(probe, "feature_maps_format")
    assert ops_module.deformable_aggregation_function.__name__ == "deformable_aggregation_function"


class _FakeMusa:
    """Minimal stand-in for torch.musa on a host without MUSA devices."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 3


def _host_has_musa_device_type() -> bool:
    """True when this torch build knows the musa device type (torch_musa installed)."""

    import torch

    try:
        torch.device("musa", 0)
    except RuntimeError:
        return False
    return True


@pytest.mark.skipif(not _host_has_musa_device_type(), reason="torch build has no musa device type")
def test_sparsedrive_cuda_device_shim_maps_arguments(monkeypatch):
    import torch

    from mu_space.patcher import runtime

    monkeypatch.setattr(torch, "musa", _FakeMusa, raising=False)
    mapped = runtime._musa_device

    assert mapped(torch, None) == torch.device("musa", 3)
    assert mapped(torch, 1) == torch.device("musa", 1)
    assert mapped(torch, "2") == torch.device("musa", 2)
    assert mapped(torch, "cuda") == torch.device("musa", 0)
    assert mapped(torch, "cuda:1") == torch.device("musa", 1)
    assert mapped(torch, torch.device("cuda", 2)) == torch.device("musa", 2)
    assert mapped(torch, torch.device("musa", 1)) == torch.device("musa", 1)


def test_sparsedrive_cuda_device_shim_requires_musa_without_cuda(monkeypatch):
    import torch

    from mu_space.patcher import runtime

    monkeypatch.setattr(torch, "musa", _FakeMusa, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert runtime._cuda_shim_available() is False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delattr(torch, "musa", raising=False)
    assert runtime._cuda_shim_available() is False

    monkeypatch.setattr(torch, "musa", _FakeMusa, raising=False)
    assert runtime._cuda_shim_available() is True


def test_sparsedrive_patcher_installs_and_releases_the_cuda_shim(tmp_path, monkeypatch):
    import torch

    _install_sparsedrive_checkout(tmp_path, monkeypatch, SPARSEDRIVE_OPS_INIT)
    monkeypatch.setattr(torch, "musa", _FakeMusa, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original = torch.Tensor.cuda

    patcher = SparseDrivePatcher()
    report = patcher.apply()

    assert torch.Tensor.cuda is not original
    assert SPARSEDRIVE_CUDA_SHIM in report.patched_modules

    patcher.restore()
    assert torch.Tensor.cuda is original


def test_sparsedrive_attention_returns_float32_for_float32_inputs(tmp_path, monkeypatch):
    import torch

    _install_sparsedrive_checkout(tmp_path, monkeypatch, SPARSEDRIVE_OPS_INIT)
    _install_sparsedrive_attention(tmp_path, monkeypatch)
    _install_fake_mate(monkeypatch)
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    patcher = SparseDrivePatcher()
    patcher.apply()
    attention = sys.modules[SPARSEDRIVE_ATTENTION]
    module_instance = attention.FlashAttention(softmax_scale=0.5)
    q = torch.randn(1, 4, 2, 8, dtype=torch.float32)
    kv = torch.randn(1, 4, 2, 2, 8, dtype=torch.float32)

    output, _ = module_instance(q, kv)

    # auto_fp16(out_fp32=True) upstream keeps a float32 call site in float32.
    assert output.dtype == torch.float32
    patcher.restore()
