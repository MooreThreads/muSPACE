from __future__ import annotations

import importlib
import math

import pytest
import torch
from torch.nn import functional as F

from mu_space.ops import SigmoidFocalLoss, sigmoid_focal_loss, sigmoid_focal_loss_reference


def upstream_golden(inputs, targets, alpha=-1.0, gamma=2.0, reduction="mean"):
    inputs = inputs.float()
    targets = targets.float()
    probability = inputs.sigmoid()
    pt = probability * targets + (1 - probability) * (1 - targets)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss * (1 - pt).pow(gamma)
    if alpha >= 0:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean() if loss.numel() else loss.sum()


def case(dtype=torch.float32, shape=(4, 3, 8, 8), *, extreme=False):
    if extreme:
        values = torch.tensor([-40.0, -20.0, -8.0, 0.0, 8.0, 20.0, 40.0], dtype=dtype)
        numel = math.prod(shape)
        inputs = values.repeat((numel + values.numel() - 1) // values.numel())[:numel]
        inputs = inputs.reshape(shape).requires_grad_()
        targets = torch.arange(inputs.numel()).reshape(shape).remainder(2).to(dtype)
        return inputs, targets
    torch.manual_seed(17)
    inputs = torch.randn(*shape, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, 2, shape, dtype=torch.int64).to(dtype)
    return inputs, targets


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("alpha,gamma", [(-1.0, 2.0), (0.25, 2.0), (0.37, 1.7)])
@pytest.mark.parametrize("extreme", [False, True])
def test_reference_matches_upstream_golden_and_backward(reduction, alpha, gamma, extreme):
    inputs, targets = case(extreme=extreme)
    actual = sigmoid_focal_loss_reference(inputs, targets, alpha, gamma, reduction)
    golden_inputs = inputs.detach().clone().requires_grad_()
    expected = upstream_golden(golden_inputs, targets, alpha, gamma, reduction)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(inputs.grad, golden_inputs.grad, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_half_reference_and_module_surface(reduction):
    inputs, targets = case(torch.float16, (2, 4, 16, 16))
    module = SigmoidFocalLoss(alpha=0.37, gamma=1.7, reduction=reduction)
    actual = module(inputs, targets)
    expected = upstream_golden(inputs, targets, 0.37, 1.7, reduction)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_empty_input(reduction):
    inputs = torch.empty(0, 3, dtype=torch.float32, requires_grad=True)
    targets = torch.empty_like(inputs)
    actual = sigmoid_focal_loss(inputs, targets, reduction=reduction, backend="reference")
    expected = upstream_golden(inputs, targets, reduction=reduction)
    torch.testing.assert_close(actual, expected)
    if reduction != "none":
        actual.backward()
        assert inputs.grad is not None and inputs.grad.numel() == 0


@pytest.mark.parametrize(
    ("logit", "target", "expected_loss", "expected_grad"),
    [
        (float("inf"), 0.0, float("inf"), 0.75),
        (float("inf"), 1.0, 0.0, 0.0),
        (float("-inf"), 0.0, 0.0, 0.0),
        (float("-inf"), 1.0, float("inf"), -0.25),
    ],
)
def test_reference_infinite_binary_logits_have_stable_limits(logit, target, expected_loss, expected_grad):
    inputs = torch.tensor([logit], requires_grad=True)
    actual = sigmoid_focal_loss_reference(inputs, torch.tensor([target]), 0.25, 2.0, "none")
    if math.isinf(expected_loss):
        assert torch.isinf(actual).item() and actual.item() > 0
    else:
        torch.testing.assert_close(actual, torch.tensor([expected_loss]))
    actual.backward()
    torch.testing.assert_close(inputs.grad, torch.tensor([expected_grad]))


def test_validation_and_forced_musa_error_on_cpu():
    with pytest.raises(ValueError, match="same shape"):
        sigmoid_focal_loss(torch.randn(2, 3), torch.ones(2), backend="reference")
    with pytest.raises(TypeError, match="dtype"):
        sigmoid_focal_loss(torch.randn(2, 3), torch.ones(2, 3, dtype=torch.float64), backend="reference")
    with pytest.raises(ValueError, match="reduction"):
        sigmoid_focal_loss(torch.randn(2, 3), torch.ones(2, 3), reduction="bad", backend="reference")
    with pytest.raises(RuntimeError):
        sigmoid_focal_loss(torch.randn(2, 3), torch.ones(2, 3), backend="musa")


def test_auto_fallback_when_native_module_is_unavailable(monkeypatch):
    focal = importlib.import_module("mu_space.ops.sigmoid_focal_loss")

    inputs, targets = case()
    monkeypatch.setattr(focal, "_native", lambda: None)
    actual = focal.sigmoid_focal_loss(inputs, targets, backend="auto")
    expected = upstream_golden(inputs, targets)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()), reason="MUSA required"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("extreme", [False, True])
def test_native_matches_upstream_golden(dtype, reduction, extreme):
    inputs, targets = case(dtype, (2, 3, 16, 16), extreme=extreme)
    musa_inputs = inputs.detach().to("musa").requires_grad_()
    musa_targets = targets.to("musa")
    actual = sigmoid_focal_loss(musa_inputs, musa_targets, 0.37, 1.7, reduction, backend="musa")
    golden_inputs = inputs.detach().float().requires_grad_()
    expected = upstream_golden(golden_inputs, targets, 0.37, 1.7, reduction)
    torch.testing.assert_close(actual.cpu(), expected.to(dtype), rtol=3e-3, atol=3e-3)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        musa_inputs.grad.cpu(), golden_inputs.grad.to(dtype), rtol=4e-3, atol=4e-3
    )


@pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()), reason="MUSA required"
)
def test_native_empty_input():
    inputs = torch.empty(0, 3, device="musa", requires_grad=True)
    targets = torch.empty_like(inputs)
    for reduction in ("none", "mean", "sum"):
        actual = sigmoid_focal_loss(inputs, targets, reduction=reduction, backend="musa")
        expected = upstream_golden(inputs.cpu(), targets.cpu(), reduction=reduction)
        torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()), reason="MUSA required"
)
def test_native_infinite_binary_logits_match_stable_reference():
    inputs = torch.tensor([float("inf"), float("inf"), float("-inf"), float("-inf")], device="musa").requires_grad_()
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0], device="musa")
    actual = sigmoid_focal_loss(inputs, targets, 0.25, 2.0, "none", backend="musa")
    expected_inputs = inputs.detach().cpu().requires_grad_()
    expected = sigmoid_focal_loss_reference(expected_inputs, targets.cpu(), 0.25, 2.0, "none")
    assert torch.equal(torch.isinf(actual.cpu()), torch.isinf(expected))
    torch.testing.assert_close(actual.cpu().nan_to_num(), expected.nan_to_num())
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(inputs.grad.cpu(), expected_inputs.grad)
