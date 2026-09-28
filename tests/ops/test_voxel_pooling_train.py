import sys
import types

import pytest
import torch

from mu_space import voxel_pooling_train, voxel_pooling_train_reference
from mu_space.patcher import BEVDepthPatcher


def _fixture(dtype=torch.float32):
    geom = torch.tensor(
        [[[0, 0, 0], [0, 0, 0], [1, 1, 0], [2, 0, 1], [-1, 0, 0], [0, 0, 2]]],
        dtype=torch.int32,
    )
    features = torch.arange(12, dtype=dtype).reshape(1, 6, 2).requires_grad_()
    return geom, features, [2, 2, 1]


def test_reference_forward_and_backward_filters_invalid_and_accumulates_duplicates():
    geom, features, shape = _fixture()
    output = voxel_pooling_train(geom, features, shape, backend="reference")
    expected = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]], [[4.0, 0.0], [0.0, 5.0]]]])
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    torch.testing.assert_close(
        features.grad,
        torch.tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_reference_minimal_training_backward(dtype):
    geom, features, shape = _fixture(dtype)
    loss = voxel_pooling_train(geom, features, shape, backend="reference").square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(features.grad).all()


def test_reference_empty_and_all_invalid_keep_gradient_connection():
    geom = torch.empty((1, 0, 3), dtype=torch.int32)
    features = torch.empty((1, 0, 4), requires_grad=True)
    output = voxel_pooling_train_reference(geom, features, [3, 4, 1])
    assert output.shape == (1, 4, 4, 3)
    output.sum().backward()
    assert features.grad.shape == features.shape

    geom = torch.full((1, 2, 3), -1, dtype=torch.int32)
    features = torch.ones((1, 2, 4), requires_grad=True)
    voxel_pooling_train(geom, features, [3, 4, 1], backend="reference").sum().backward()
    assert features.grad.eq(0).all()


@pytest.mark.parametrize(
    ("geom", "features", "voxel_num", "error"),
    [
        (torch.zeros((1, 2, 4), dtype=torch.int32), torch.ones((1, 2, 3)), [2, 2, 1], "shape"),
        (torch.zeros((1, 2, 3), dtype=torch.int64), torch.ones((1, 2, 3)), [2, 2, 1], "int32"),
        (torch.zeros((1, 2, 3), dtype=torch.int32), torch.ones((1, 3, 3)), [2, 2, 1], "matching"),
        (torch.zeros((1, 2, 3), dtype=torch.int32), torch.ones((1, 2, 3)), [2, 2], "voxel_num"),
    ],
)
def test_invalid_inputs_fail_explicitly(geom, features, voxel_num, error):
    with pytest.raises((ValueError, TypeError), match=error):
        voxel_pooling_train(geom, features, voxel_num, backend="reference")


def test_bevdepth_patcher_rebinds_target_and_imported_bindings(monkeypatch):
    target_name = "bevdepth.ops.voxel_pooling_train"
    imported_name = "bevdepth.layers.backbones.base_lss_fpn"
    target = types.ModuleType(target_name)
    imported = types.ModuleType(imported_name)

    def original(*args, **kwargs):
        return "original"

    target.voxel_pooling_train = original
    imported.voxel_pooling_train = original
    monkeypatch.setitem(sys.modules, target_name, target)
    monkeypatch.setitem(sys.modules, imported_name, imported)
    patcher = BEVDepthPatcher()
    report = patcher.apply(strict=False)
    assert report.patched_modules == (target_name, imported_name)
    assert target.voxel_pooling_train is imported.voxel_pooling_train
    assert target.voxel_pooling_train is not original
    restored = patcher.restore()
    assert restored.action == "restored"
    assert target.voxel_pooling_train is original


def test_bevdepth_patcher_installs_and_restores_missing_cuda_extension_shim(monkeypatch):
    target_name = "bevdepth.ops.voxel_pooling_train"
    extension_name = f"{target_name}.voxel_pooling_train_ext"
    parent_name = "bevdepth.ops"
    parent = types.ModuleType(parent_name)
    monkeypatch.setitem(sys.modules, parent_name, parent)
    real_import = __import__("importlib").import_module

    def import_module(name, package=None):
        if name == target_name:
            raise ModuleNotFoundError("missing CUDA extension", name=extension_name)
        return real_import(name, package)

    monkeypatch.setattr("mu_space.patcher.runtime.importlib.import_module", import_module)
    patcher = BEVDepthPatcher()
    report = patcher.apply(strict=False)
    assert report.patched_modules == (target_name,)
    target = sys.modules[target_name]
    assert target.voxel_pooling_train is not None
    patcher.restore()
    assert target_name not in sys.modules
    assert not hasattr(parent, "voxel_pooling_train")


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_musa_matches_reference_forward_backward():
    geom, features, shape = _fixture()
    geom = geom.to("musa")
    features = features.detach().to("musa").requires_grad_()
    ref_features = features.detach().cpu().requires_grad_()
    actual = voxel_pooling_train(geom, features, shape, backend="musa")
    expected = voxel_pooling_train(geom.cpu(), ref_features, shape, backend="reference")
    torch.testing.assert_close(actual.cpu(), expected)
    grad = torch.arange(actual.numel(), dtype=actual.dtype, device="musa").reshape_as(actual)
    actual.backward(grad)
    expected.backward(grad.cpu())
    torch.testing.assert_close(features.grad.cpu(), ref_features.grad)
