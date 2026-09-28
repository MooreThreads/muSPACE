"""MUSA implementation of mit-han-lab/BEVFusion sigmoid focal loss."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ..runtime.kernel_dispatch import (
    DispatchReport,
    explain_dispatch,
    record_dispatch_report,
)
from ..runtime.native import NativeModuleUnavailable, native_module_proxy

OPERATOR_NAME = "sigmoid_focal_loss"
BackendName = Literal["auto", "musa", "reference"]


def _native():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _check(
    inputs: torch.Tensor, targets: torch.Tensor, reduction: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.dtype not in (torch.float16, torch.float32):
        raise TypeError("inputs must use float16 or float32")
    if inputs.shape != targets.shape:
        raise ValueError("targets must have the same shape as inputs")
    if targets.dtype != inputs.dtype:
        raise TypeError("targets dtype must match inputs")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")
    return inputs.contiguous(), targets.contiguous()


def sigmoid_focal_loss_reference(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs, targets = _check(inputs, targets, reduction)
    inputs_fp32 = inputs.float()
    targets_fp32 = targets.float()
    binary_targets = (targets_fp32 == 0) | (targets_fp32 == 1)
    infinite_binary = torch.isinf(inputs_fp32) & binary_targets
    # Evaluate the ordinary expression on finite placeholders.  ``where`` alone
    # is insufficient because its unselected branch can still contribute a NaN
    # gradient through 0 * inf in autograd.
    safe_inputs = torch.where(infinite_binary, torch.zeros_like(inputs_fp32), inputs_fp32)
    p = safe_inputs.sigmoid()
    pt = p * targets_fp32 + (1 - p) * (1 - targets_fp32)
    # BCEWithLogits' algebraic expression can evaluate ``0 * inf`` for binary
    # targets at infinite logits.  Select the equivalent softplus form for the
    # binary focal-loss contract so special values have their mathematical
    # limits instead of an implementation-dependent NaN.
    binary_loss = torch.where(
        targets_fp32 == 0,
        F.softplus(safe_inputs),
        F.softplus(-safe_inputs),
    )
    loss = torch.where(
        (targets_fp32 == 0) | (targets_fp32 == 1),
        binary_loss,
        F.binary_cross_entropy_with_logits(safe_inputs, targets_fp32, reduction="none"),
    )
    loss = loss * (1 - pt).pow(gamma)
    if alpha >= 0:
        at = alpha * targets_fp32 + (1 - alpha) * (1 - targets_fp32)
        loss = loss * at
    else:
        at = torch.ones_like(targets_fp32)
    positive_limit = torch.where(targets_fp32 == 0, at * inputs_fp32, torch.zeros_like(inputs_fp32))
    negative_limit = torch.where(targets_fp32 == 1, -at * inputs_fp32, torch.zeros_like(inputs_fp32))
    loss = torch.where(infinite_binary, torch.where(inputs_fp32 > 0, positive_limit, negative_limit), loss)
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean() if loss.numel() else loss.sum()
    return loss


class _NativeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, alpha, gamma, reduction):
        ext = _native()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        output = torch.empty_like(inputs)
        ext.sigmoid_focal_loss_forward(inputs, targets, output, float(gamma), float(alpha))
        ctx.save_for_backward(inputs, targets)
        ctx.alpha, ctx.gamma, ctx.reduction = float(alpha), float(gamma), reduction
        if reduction == "sum":
            return output.sum()
        if reduction == "mean":
            return output.mean() if output.numel() else output.sum()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, targets = ctx.saved_tensors
        grad = torch.empty_like(inputs)
        _native().sigmoid_focal_loss_backward(
            inputs, targets, grad, ctx.gamma, ctx.alpha
        )
        if ctx.reduction == "sum":
            grad *= grad_output
        elif ctx.reduction == "mean":
            if inputs.numel():
                grad *= grad_output / inputs.numel()
            else:
                grad.zero_()
        else:
            grad *= grad_output
        return grad, None, None, None, None


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    inputs, targets = _check(inputs, targets, reduction)
    native_available = _native() is not None
    decision = explain_dispatch(inputs, backend=backend, native_available=native_available)
    if decision.backend == "error":
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        record_dispatch_report(DispatchReport(OPERATOR_NAME, backend, "reference", decision.device,
                                              decision.reason, native_available))
        return sigmoid_focal_loss_reference(inputs, targets, alpha, gamma, reduction)
    try:
        result = _NativeFunction.apply(inputs, targets, alpha, gamma, reduction)
    except RuntimeError as exc:
        if backend != "auto":
            raise
        record_dispatch_report(DispatchReport(OPERATOR_NAME, backend, "reference", decision.device,
                                              "native backend failed; falling back to reference",
                                              native_available, True, str(exc)))
        return sigmoid_focal_loss_reference(inputs, targets, alpha, gamma, reduction)
    record_dispatch_report(DispatchReport(OPERATOR_NAME, backend, "musa", decision.device,
                                          decision.reason, native_available))
    return result


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = -1, gamma: float = 2, reduction: str = "mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)


__all__ = ["SigmoidFocalLoss", "sigmoid_focal_loss", "sigmoid_focal_loss_reference"]
