from importlib import metadata

import pytest
import torch

from mu_space.compatibility import (
    BEVFORMER_EXPECTED,
    compatibility_report,
    enable_eval_batch_norm_reference,
)


@pytest.mark.parametrize(
    ("sdk", "torch_version"),
    (("4.3.8", "2.7.1"), ("5.2.0", "2.9.1"), ("5.2.0", "2.11.0")),
)
def test_core_matrix_is_validated(monkeypatch, sdk, torch_version):
    versions = {"torch": torch_version, "torch_musa": torch_version}
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    report = compatibility_report(sdk_version=sdk, musa_available=True)
    assert report.state == "validated"
    assert report.profile == "core"
    assert report.expected_stack == {"torch": torch_version, "torch_musa": torch_version}


def test_core_matrix_accepts_post_release_tags(monkeypatch):
    versions = {"torch": "2.11.0.post1", "torch_musa": "2.11.0.post1+b431668"}
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    report = compatibility_report(sdk_version="5.2.0", musa_available=True)
    assert report.state == "validated"
    assert report.expected_stack == {"torch": "2.11.0", "torch_musa": "2.11.0"}


@pytest.mark.parametrize(
    ("sdk", "torch", "torch_musa", "musa_available", "reason"),
    (
        ("4.3.8", "2.9.1", "2.9.1", True, "requires"),
        ("5.2.0", "2.9.1", "2.7.1", True, "same version"),
        ("5.2.0", "2.11.0", "2.9.1", True, "same version"),
        ("5.2.0", "2.9.1", "2.9.1", False, "device"),
    ),
)
def test_invalid_core_stack_is_unsupported(
    monkeypatch, sdk, torch, torch_musa, musa_available, reason
):
    versions = {"torch": torch, "torch_musa": torch_musa}
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    report = compatibility_report(sdk_version=sdk, musa_available=musa_available)
    assert report.state == "unsupported"
    assert reason in report.reasons[0]


def test_unknown_sdk_is_reported(monkeypatch):
    versions = {"torch": "2.9.1", "torch_musa": "2.9.1"}
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    monkeypatch.setattr("mu_space.compatibility._detect_musa_sdk_version", lambda: None)
    report = compatibility_report(sdk_version=None, musa_available=True)
    assert report.state == "unknown"
    assert report.musa_sdk is None


def test_missing_package_is_unsupported(monkeypatch):
    def fake_version(name):
        if name == "torch_musa":
            raise metadata.PackageNotFoundError(name)
        return "2.9.1"

    monkeypatch.setattr(metadata, "version", fake_version)
    report = compatibility_report()
    assert report.state == "unsupported"
    assert "torch_musa" in report.reasons[0]


def test_bevformer_profile_remains_pinned(monkeypatch):
    monkeypatch.setattr(metadata, "version", lambda name: BEVFORMER_EXPECTED[name])
    report = compatibility_report(profile="bevformer", sdk_version="5.2.0", musa_available=True)
    assert report.state == "validated"
    assert report.profile == "bevformer"


def test_eval_batch_norm_reference_matches_native_cpu():
    native = torch.nn.BatchNorm2d(4).eval()
    reference = torch.nn.BatchNorm2d(4).eval()
    with torch.no_grad():
        native.running_mean.copy_(torch.randn(4))
        native.running_var.copy_(torch.rand(4) + 0.5)
        native.weight.copy_(torch.tensor([0.0, 0.5, 1.0, -1.5]))
        native.bias.copy_(torch.randn(4))
        reference.load_state_dict(native.state_dict())

    native_input = torch.randn(2, 4, 3, 5, requires_grad=True)
    reference_input = native_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(native_input)
    native_output = native(native_input)
    native_output.backward(grad_output)

    assert enable_eval_batch_norm_reference(reference) == 1
    assert enable_eval_batch_norm_reference(reference) == 0
    reference_output = reference(reference_input)
    reference_output.backward(grad_output)

    torch.testing.assert_close(reference_output, native_output)
    torch.testing.assert_close(reference_input.grad, native_input.grad)


def test_eval_batch_norm_reference_skips_training_modules():
    assert enable_eval_batch_norm_reference(torch.nn.BatchNorm2d(2).train()) == 0
