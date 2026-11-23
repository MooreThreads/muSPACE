"""Flash-attention compatible facade backed by PyTorch SDPA on MUSA.

Orion's vendored mmcv imports ``flash_attn_varlen_kvpacked_func`` and the padding
helpers at module import time, so the package cannot load without flash-attn.
The shim reproduces the semantics those call sites need -- variable-length
batches, query/key-value packed along one axis, and optional padding masks -- on
top of ``torch.nn.functional.scaled_dot_product_attention``, which runs natively
on MUSA.

Install it before importing the vendored ``mmcv``:

    from mu_space.patcher.orion_flash_attn import install_flash_attn_shim
    install_flash_attn_shim()
    import mmcv
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types

import torch
import torch.nn.functional as F

_SHIM_VERSION = "musa-sdpa-shim"


def _cumulative_lengths(lengths: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros(1, dtype=torch.int32, device=lengths.device)
    return torch.cat((zero, torch.cumsum(lengths.to(torch.int32), dim=0)))


def _unpad(inputs: torch.Tensor, key_padding_mask: torch.Tensor):
    """Mirror ``flash_attn.bert_padding.unpad_input`` for ``[B, S, ...]`` inputs."""
    lengths = key_padding_mask.logical_not().to(torch.int32).sum(dim=1)
    indices = key_padding_mask.logical_not().reshape(-1).nonzero(as_tuple=False).squeeze(1)
    unpadded = inputs.reshape(-1, *inputs.shape[2:]).index_select(0, indices)
    return unpadded, indices, _cumulative_lengths(lengths), int(lengths.max().item())


def _pad(inputs: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Mirror ``flash_attn.bert_padding.pad_input`` for ``[total, ...]`` inputs."""
    output = inputs.new_zeros((batch * seqlen, *inputs.shape[1:]))
    output = output.index_copy(0, indices, inputs)
    return output.reshape(batch, seqlen, *inputs.shape[1:])


def _index_first_axis(inputs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Mirror ``flash_attn.bert_padding.index_first_axis`` for ``[B, S, ...]`` inputs."""
    return inputs.reshape(-1, *inputs.shape[2:]).index_select(0, indices)


def _varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    """Evaluate a padded batch with SDPA and return ``[total_q, H, D]``."""
    batch = int(cu_seqlens_q.numel()) - 1
    nheads = q.shape[1]
    head_dim = q.shape[2]
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5

    q_pad = torch.zeros(
        (batch, max_seqlen_q, nheads, head_dim), dtype=q.dtype, device=q.device
    )
    k_pad = torch.zeros(
        (batch, max_seqlen_k, nheads, head_dim), dtype=k.dtype, device=k.device
    )
    v_pad = torch.zeros(
        (batch, max_seqlen_k, nheads, head_dim), dtype=v.dtype, device=v.device
    )
    q_mask = torch.zeros((batch, max_seqlen_q), dtype=torch.bool, device=q.device)
    k_mask = torch.zeros((batch, max_seqlen_k), dtype=torch.bool, device=q.device)

    q_lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    k_lengths = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
    q_offset = 0
    k_offset = 0
    for row, (q_len, k_len) in enumerate(zip(q_lengths, k_lengths)):
        q_pad[row, :q_len] = q[q_offset : q_offset + q_len]
        k_pad[row, :k_len] = k[k_offset : k_offset + k_len]
        v_pad[row, :k_len] = v[k_offset : k_offset + k_len]
        q_mask[row, :q_len] = True
        k_mask[row, :k_len] = True
        q_offset += q_len
        k_offset += k_len

    # SDPA takes [B, H, S, D]. A boolean mask marks positions that MAY attend, so
    # the padded key tail is set to False and can never contribute.
    sdpa_mask = k_mask[:, None, None, :].expand(batch, 1, max_seqlen_q, max_seqlen_k)
    output = F.scaled_dot_product_attention(
        q_pad.transpose(1, 2),
        k_pad.transpose(1, 2),
        v_pad.transpose(1, 2),
        attn_mask=sdpa_mask,
        dropout_p=dropout_p,
        scale=scale,
        is_causal=causal,
    )
    output = output.transpose(1, 2)
    # Match the varlen contract: only the query positions that exist are returned.
    return output.reshape(-1, nheads, head_dim).index_select(0, _valid_positions(q_mask))


def _valid_positions(mask: torch.Tensor) -> torch.Tensor:
    return mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)


def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Flash-attn compatible ``varlen_kvpacked`` attention.

    ``q`` is ``[total_q, nheads, headdim]`` and ``kv`` is
    ``[total_k, 2, nheads, headdim]``; the return value is ``[total_q, nheads, headdim]``.
    """
    k, v = kv.unbind(dim=1)
    return _varlen_attention(
        q,
        k,
        v,
        cu_seqlens_q.to(q.device),
        cu_seqlens_k.to(q.device),
        int(max_seqlen_q),
        int(max_seqlen_k),
        dropout_p,
        softmax_scale,
        causal,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Flash-attn compatible ``varlen`` attention for separate q/k/v."""
    return _varlen_attention(
        q,
        k,
        v,
        cu_seqlens_q.to(q.device),
        cu_seqlens_k.to(q.device),
        int(max_seqlen_q),
        int(max_seqlen_k),
        dropout_p,
        softmax_scale,
        causal,
    )


def _build_modules() -> dict[str, types.ModuleType]:
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.__spec__ = importlib.machinery.ModuleSpec("flash_attn.flash_attn_interface", None)
    interface.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
    interface.flash_attn_varlen_func = flash_attn_varlen_func
    interface.flash_attn_unpadded_kvpacked_func = flash_attn_varlen_kvpacked_func

    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", None)
    bert_padding.unpad_input = _unpad
    bert_padding.pad_input = _pad
    bert_padding.index_first_axis = _index_first_axis

    package = types.ModuleType("flash_attn")
    # A real spec keeps importlib.util.find_spec() and transformers' package probes
    # working; the empty path marks the module as a package.
    package.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None, is_package=True)
    package.__spec__.submodule_search_locations = []
    package.__path__ = []
    package.flash_attn_interface = interface
    package.bert_padding = bert_padding
    package.__version__ = _SHIM_VERSION
    return {
        "flash_attn": package,
        "flash_attn.flash_attn_interface": interface,
        "flash_attn.bert_padding": bert_padding,
    }


def install_flash_attn_shim(*, force: bool = False) -> bool:
    """Register the SDPA shim as ``flash_attn`` unless the real package exists.

    Returns ``True`` when the shim is the module the process will import.
    """
    existing = sys.modules.get("flash_attn")
    if existing is not None and not force:
        return getattr(existing, "__version__", "") == _SHIM_VERSION
    if not force:
        try:
            importlib.import_module("flash_attn")
        except Exception:
            pass
        else:
            return False
    for name, module in _build_modules().items():
        sys.modules[name] = module
    return True


__all__ = [
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "install_flash_attn_shim",
]
