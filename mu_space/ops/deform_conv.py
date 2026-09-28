"""Standard deformable convolution compatibility API for legacy OpenMMLab models.

The MUSA native implementation is shared with DCNv2 by passing an all-ones mask.
That preserves the standard DCN equation while keeping the existing native kernel
and its backward implementation as the single source of numerical behavior.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair

from .modulated_deform_conv import modulated_deform_conv2d
from ..runtime.kernel_dispatch import DispatchReport, record_dispatch_report
from ..runtime.native import native_available


class DeformConv2d(nn.Module):
    """Standard DCN layer with the mmcv constructor and weight layout."""

    _version = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = False,
        im2col_step: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.im2col_step = im2col_step
        self.transposed = False
        self.output_padding = (0, 0)
        if bias:
            raise ValueError("standard mmcv DeformConv2d does not support bias")
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        taps = self.deform_groups * self.kernel_size[0] * self.kernel_size[1]
        expected = 2 * taps
        if offset.ndim != 4 or offset.shape[1] != expected:
            raise ValueError(f"offset must have shape [B, {expected}, H, W]")
        input_pad = x.shape[2] < self.kernel_size[0] or x.shape[3] < self.kernel_size[1]
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.shape[2], 0)
            pad_w = max(self.kernel_size[1] - x.shape[3], 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), "constant", 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), "constant", 0).contiguous()
        mask = x.new_ones((x.shape[0], taps, offset.shape[2], offset.shape[3]))
        output = modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
            backend="auto",
        )
        is_musa = x.device.type in {"musa", "privateuseone"} or bool(getattr(x, "is_musa", False))
        record_dispatch_report(
            DispatchReport(
                operator="deform_conv2d",
                requested_backend="auto",
                selected_backend="musa" if is_musa and native_available() else "reference",
                device=x.device.type,
                reason="standard DCN reuses the native modulated kernel with mask=1",
                native_available=native_available(),
                native_fallback=False,
            )
        )
        if input_pad:
            output = output[:, :, : output.shape[2] - pad_h, : output.shape[3] - pad_w].contiguous()
        return output


class DeformConv2dPack(DeformConv2d):
    """mmcv-compatible packed standard DCN layer."""

    _version = 2

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, self.conv_offset(x))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            old_weight, old_bias = prefix[:-1] + "_offset.weight", prefix[:-1] + "_offset.bias"
            new_weight, new_bias = prefix + "conv_offset.weight", prefix + "conv_offset.bias"
            if new_weight not in state_dict and old_weight in state_dict:
                state_dict[new_weight] = state_dict.pop(old_weight)
            if new_bias not in state_dict and old_bias in state_dict:
                state_dict[new_bias] = state_dict.pop(old_bias)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


__all__ = ["DeformConv2d", "DeformConv2dPack"]
