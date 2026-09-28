"""BEVFusion-compatible 3-D sparse convolution primitives.

The public functions intentionally mirror the legacy ``sparse_conv_ext``
entrypoints.  ``reference`` is kept small and deterministic so the same test
vectors can validate the ABI-independent MUSA implementation.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch

from ..runtime.native import NativeModuleUnavailable, native_module_proxy
from ..runtime.kernel_dispatch import _is_musa_tensor


def _as_tuple(value, ndim: int) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        if len(value) != ndim:
            raise ValueError(f"expected {ndim} values, got {len(value)}")
        return tuple(int(x) for x in value)
    return (int(value),) * ndim


def _conv_shape(shape, kernel, stride, padding, dilation):
    return tuple(
        (int(n) + 2 * p - d * (k - 1) - 1) // s + 1
        for n, k, s, p, d in zip(shape, kernel, stride, padding, dilation)
    )


def _deconv_shape(shape, kernel, stride, padding, output_padding):
    return tuple(
        (int(n) - 1) * s - 2 * p + k + op
        for n, k, s, p, op in zip(shape, kernel, stride, padding, output_padding)
    )


def _iter_kernel(kernel: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    return __import__("itertools").product(*(range(k) for k in kernel))


def _pair_position(pair: tuple[int, ...], kernel, dilation, padding):
    """Map a non-transposed kernel slot to its input-minus-output offset.

    Legacy spconv stores an indice pair at slot ``k`` as
    ``input = output + padding - k * dilation``.  The opposite direction
    preserves counts and merely reverses the 3-D filter, which can stay hidden
    in symmetric fixtures while producing incorrect BEVFusion features.
    """
    return tuple(p - x * d for x, d, p in zip(pair, dilation, padding))


def _vectorized_indice_pairs(
    indices, batch_size, out_shape, spatial_shape, kernel, stride, padding, dilation, subm
):
    """MUSA-friendly pair construction using tensor operations only.

    This keeps coordinate generation on the accelerator; the native FFI
    convolution consumes the resulting compact pair tensors.
    """
    device = indices.device
    kernel_volume = int(kernel[0]) * int(kernel[1]) * int(kernel[2])
    if indices.shape[0] == 0:
        # No active voxel can produce a pair.  Return the empty-input contract
        # directly: the key membership below would index empty tensors.
        return (
            indices.new_empty((0, 4), dtype=torch.int32),
            indices.new_full((kernel_volume, 2, 0), -1, dtype=torch.int32),
            indices.new_zeros((kernel_volume,), dtype=torch.int32),
        )
    kernel_pos = torch.cartesian_prod(
        *[torch.arange(k, device=device, dtype=torch.int32) for k in kernel]
    ).reshape(-1, 3)
    offsets = indices.new_tensor(padding) - kernel_pos * indices.new_tensor(dilation)
    coords = indices[:, 1:].to(torch.int32)
    numerators = coords[:, None, :] + offsets[None, :, :]
    stride_tensor = indices.new_tensor(stride)
    valid = (numerators.remainder(stride_tensor) == 0).all(dim=-1)
    output_coords = torch.div(numerators, stride_tensor, rounding_mode="floor")
    shape_tensor = indices.new_tensor(out_shape)
    valid &= (output_coords >= 0).all(dim=-1) & (output_coords < shape_tensor).all(dim=-1)
    output_coords = output_coords.reshape(-1, 3)
    valid_flat = valid.reshape(-1)
    batch_flat = indices[:, 0:1].expand(-1, kernel_pos.shape[0]).reshape(-1)

    volume = int(out_shape[0]) * int(out_shape[1]) * int(out_shape[2])
    output_keys = (
        batch_flat.to(torch.int64) * volume
        + output_coords[:, 0].to(torch.int64) * int(out_shape[1]) * int(out_shape[2])
        + output_coords[:, 1].to(torch.int64) * int(out_shape[2])
        + output_coords[:, 2].to(torch.int64)
    )
    if subm:
        input_keys = (
            indices[:, 0].to(torch.int64) * volume
            + indices[:, 1].to(torch.int64) * int(spatial_shape[1]) * int(spatial_shape[2])
            + indices[:, 2].to(torch.int64) * int(spatial_shape[2])
            + indices[:, 3].to(torch.int64)
        )
        sorted_keys, sort_order = input_keys.sort()
        membership = torch.searchsorted(sorted_keys, output_keys.clamp_min(0))
        membership = membership.clamp_max(max(0, input_keys.numel() - 1))
        valid_flat &= torch.isin(output_keys, input_keys)
        valid_flat &= membership < sorted_keys.numel()
        valid_flat &= sorted_keys[membership] == output_keys
        output_ids = sort_order[membership]
        outids = indices
    else:
        # The legacy implementation traverses kernel offsets first and input
        # rows second.  Build first-occurrence order in that same traversal,
        # while the final pair tensor below remains input-major.
        key_matrix = output_keys.reshape(indices.shape[0], kernel_volume)
        valid_matrix = valid_flat.reshape(indices.shape[0], kernel_volume)
        traversal_keys = key_matrix.transpose(0, 1).reshape(-1)
        traversal_valid = valid_matrix.transpose(0, 1).reshape(-1)
        unique_keys, inverse = torch.unique(traversal_keys[traversal_valid], sorted=True, return_inverse=True)
        # ``torch.unique`` returns key-sorted values on accelerator backends,
        # while the legacy host rulebook exposes each output on its first
        # encounter in kernel-major/input-major traversal.  Restore that public
        # row order and remap every pair's output index to the reordered rows.
        valid_positions = torch.arange(inverse.numel(), device=device, dtype=torch.int64)
        first_positions = torch.full_like(unique_keys, valid_positions.numel(), dtype=torch.int64)
        first_positions.scatter_reduce_(0, inverse, valid_positions, reduce="amin", include_self=True)
        first_order = first_positions.argsort(stable=True)
        remap = torch.empty_like(first_order)
        remap[first_order] = torch.arange(first_order.numel(), device=device, dtype=first_order.dtype)
        output_ids = torch.zeros_like(output_keys)
        output_ids[valid_flat] = remap[
            torch.searchsorted(unique_keys, output_keys[valid_flat])
        ].to(output_ids.dtype)
        unique_keys = unique_keys[first_order]
        outids = indices.new_empty((unique_keys.numel(), 4))
        outids[:, 0] = torch.div(unique_keys, volume, rounding_mode="floor").to(torch.int32)
        remainder = unique_keys.remainder(volume)
        outids[:, 1] = torch.div(
            remainder, int(out_shape[1]) * int(out_shape[2]), rounding_mode="floor"
        ).to(torch.int32)
        remainder = remainder.remainder(int(out_shape[1]) * int(out_shape[2]))
        outids[:, 2] = torch.div(remainder, int(out_shape[2]), rounding_mode="floor").to(
            torch.int32
        )
        outids[:, 3] = remainder.remainder(int(out_shape[2])).to(torch.int32)

    valid = valid_flat.reshape(indices.shape[0], -1)
    pair_num = valid.sum(dim=0).to(torch.int32)
    capacity = int(pair_num.max().item()) if pair_num.numel() else 0
    pairs = indices.new_full((kernel_volume, 2, capacity), -1, dtype=torch.int32)
    if capacity:
        # Slot index of every live pair inside its own kernel offset, so the whole
        # rulebook can be filled with two scatters instead of one pass per offset.
        slots = valid.cumsum(dim=0) - 1
        voxel, offset = torch.nonzero(valid, as_tuple=True)
        slot = slots[voxel, offset]
        pairs[offset, 0, slot] = voxel.to(torch.int32)
        pairs[offset, 1, slot] = output_ids[voxel * kernel_volume + offset].to(torch.int32)
    return outids.contiguous(), pairs.contiguous(), pair_num.contiguous()


def get_indice_pairs_3d(
    indices: torch.Tensor,
    batch_size: int,
    out_shape,
    spatial_shape,
    kernel_size=3,
    stride=1,
    padding=0,
    dilation=1,
    out_padding=0,
    subm=0,
    transpose=0,
):
    """Reference implementation of the legacy 3-D indice-pair contract.

    The returned layout is ``outids[M_out, 4]``, ``pairs[K, 2, P]`` and
    ``pair_num[K]``.  Invalid capacity slots are filled with ``-1``.
    """
    if indices.ndim != 2 or indices.shape[1] != 4:
        raise ValueError("indices must have shape [M, 4] in [batch,z,y,x] order")
    kernel = _as_tuple(kernel_size, 3)
    stride = _as_tuple(stride, 3)
    padding = _as_tuple(padding, 3)
    dilation = _as_tuple(dilation, 3)
    out_padding = _as_tuple(out_padding, 3)
    if subm and any(value != 1 for value in stride):
        raise ValueError("subm convolution requires stride=1")
    if _is_musa_tensor(indices) and not transpose:
        return _vectorized_indice_pairs(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            kernel,
            stride,
            padding,
            dilation,
            bool(subm),
        )
    del batch_size
    if subm:
        output_shape = tuple(int(x) for x in spatial_shape)
        output_coords = indices
        coord_to_output = {tuple(row): i for i, row in enumerate(indices.tolist())}
    else:
        if transpose:
            output_shape = _deconv_shape(spatial_shape, kernel, stride, padding, out_padding)
        else:
            output_shape = tuple(int(x) for x in out_shape)
        output_coords_list = []
        coord_to_output = {}

    pairs_by_kernel: list[list[tuple[int, int]]] = []
    for kernel_pos in _iter_kernel(kernel):
        # Transposed convolution uses the opposite (output-minus-input)
        # relationship; preserve its established public contract.
        offset = (
            tuple(x * d - p for x, d, p in zip(kernel_pos, dilation, padding))
            if transpose
            else _pair_position(kernel_pos, kernel, dilation, padding)
        )
        pair_list: list[tuple[int, int]] = []
        for input_index, row in enumerate(indices.tolist()):
            batch, z, y, x = row
            if transpose:
                out_xyz = tuple(
                    int(coord * s + offset[d])
                    for d, (coord, s) in enumerate(zip((z, y, x), stride))
                )
            else:
                numerators = tuple(coord + offset[d] for d, coord in enumerate((z, y, x)))
                if any(n % s != 0 for n, s in zip(numerators, stride)):
                    continue
                out_xyz = tuple(n // s for n, s in zip(numerators, stride))
            if any(v < 0 or v >= output_shape[d] for d, v in enumerate(out_xyz)):
                continue
            out_row = (batch, *out_xyz)
            if subm:
                output_index = coord_to_output.get(out_row)
                if output_index is None:
                    continue
            else:
                output_index = coord_to_output.get(out_row)
                if output_index is None:
                    output_index = len(output_coords_list)
                    coord_to_output[out_row] = output_index
                    output_coords_list.append(out_row)
            pair_list.append((input_index, output_index))
        pairs_by_kernel.append(pair_list)

    if subm:
        outids = output_coords
    else:
        outids = indices.new_tensor(output_coords_list, dtype=torch.int32).reshape(-1, 4)
    capacity = max((len(x) for x in pairs_by_kernel), default=0)
    pairs = indices.new_full((len(pairs_by_kernel), 2, capacity), -1, dtype=torch.int32)
    pair_num = indices.new_zeros((len(pairs_by_kernel),), dtype=torch.int32)
    for kernel_index, pair_list in enumerate(pairs_by_kernel):
        if pair_list:
            pair_tensor = indices.new_tensor(pair_list, dtype=torch.int32).t().contiguous()
            pairs[kernel_index, :, : pair_tensor.shape[1]] = pair_tensor
            pair_num[kernel_index] = pair_tensor.shape[1]
    return outids.contiguous(), pairs.contiguous(), pair_num.contiguous()


def get_indice_pairs_grid_3d(*args, **kwargs):
    # The legacy grid is an allocation hint; coordinate semantics are identical.
    args = list(args)
    if len(args) >= 2 and torch.is_tensor(args[1]):
        del args[1]
    elif "grid" in kwargs:
        kwargs = dict(kwargs)
        kwargs.pop("grid")
    return get_indice_pairs_3d(*args, **kwargs)


def _native():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _flatten_filters(filters: torch.Tensor) -> torch.Tensor:
    if filters.ndim == 5:
        return filters.reshape(-1, filters.shape[-2], filters.shape[-1]).contiguous()
    if filters.ndim == 3:
        return filters.contiguous()
    raise ValueError("filters must have shape [Kz,Ky,Kx,Cin,Cout] or [K,Cin,Cout]")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class _tf32_matmul:
    """Enable TF32 matmul for float32 GEMMs, restored on exit.

    Strict float32 GEMM has no tensor-core path on MTT S5000. Matching the
    reduced-precision mode used by muDNN / A100 torch.mm keeps the sparse path
    on the same performance footing as the reference gather-GEMM-scatter
    implementation.  ``MUSPACE_SPCONV_FP32_TF32=0`` forces strict IEEE float32.

    ``allow_tf32`` is a process-global backend flag; this context manager is
    therefore not safe to enter concurrently with other float32 GEMM work.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        self._dtype = dtype
        self._saved: list[tuple[object, bool]] = []

    def __enter__(self) -> "_tf32_matmul":
        if self._dtype == torch.float32 and _env_flag("MUSPACE_SPCONV_FP32_TF32", True):
            for backend_name in ("musa", "cuda"):
                backend = getattr(torch.backends, backend_name, None)
                matmul = getattr(backend, "matmul", None) if backend is not None else None
                if matmul is not None and hasattr(matmul, "allow_tf32"):
                    self._saved.append((matmul, matmul.allow_tf32))
                    matmul.allow_tf32 = True
        return self

    def __exit__(self, *exc) -> None:
        for matmul, previous in self._saved:
            matmul.allow_tf32 = previous


def _pair_counts(indice_pair_num: torch.Tensor) -> list[int]:
    return [int(x) for x in indice_pair_num.detach().to("cpu").reshape(-1).tolist()]


def _gather_gemm_scatter_conv(
    features: torch.Tensor,
    weights: torch.Tensor,
    indice_pairs: torch.Tensor,
    pair_counts: list[int],
    num_activate_out: int,
    inverse: bool = False,
    subm: bool = False,
) -> torch.Tensor:
    """BEVFusion-style sparse conv: gather → GEMM → scatter-add.

    Mirrors ``spconv_ops.h::indiceConv``. SubM center offset (the kernel index
    with the largest pair count) is a direct ``features @ filters[center]``
    and skips gather/scatter for that offset.
    """
    output = features.new_zeros((int(num_activate_out), weights.shape[-1]))
    if not pair_counts or max(pair_counts) <= 0:
        return output

    kernel_volume = weights.shape[0]
    gather_plane = 0 if not inverse else 1
    scatter_plane = 1 if not inverse else 0
    center = max(range(kernel_volume), key=lambda i: pair_counts[i])

    with _tf32_matmul(features.dtype):
        if subm and pair_counts[center] > 0:
            # Not the ``out=`` form: it would be an in-place write on a leaf of
            # the autograd graph when this path is called with requires_grad.
            output = output + torch.mm(features, weights[center])
        for kernel_index in range(kernel_volume):
            count = pair_counts[kernel_index]
            if count <= 0 or (subm and kernel_index == center):
                continue
            input_ids = indice_pairs[kernel_index, gather_plane, :count].long()
            output_ids = indice_pairs[kernel_index, scatter_plane, :count].long()
            values = features.index_select(0, input_ids).matmul(weights[kernel_index])
            output.index_add_(0, output_ids, values)
    return output


def _gather_gemm_scatter_conv_backward(
    features: torch.Tensor,
    weights: torch.Tensor,
    grad_output: torch.Tensor,
    indice_pairs: torch.Tensor,
    pair_counts: list[int],
    inverse: bool = False,
    subm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of :func:`_gather_gemm_scatter_conv` (same BEVFusion layout)."""
    grad_features = torch.zeros_like(features)
    grad_weights = torch.zeros_like(weights)
    if not pair_counts or max(pair_counts) <= 0:
        return grad_features, grad_weights

    kernel_volume = weights.shape[0]
    gather_plane = 0 if not inverse else 1
    scatter_plane = 1 if not inverse else 0
    center = max(range(kernel_volume), key=lambda i: pair_counts[i])
    grad_output = grad_output.contiguous()

    with _tf32_matmul(features.dtype):
        if subm and pair_counts[center] > 0:
            grad_weights[center].add_(features.transpose(0, 1).matmul(grad_output))
            grad_features.add_(grad_output.matmul(weights[center].transpose(0, 1)))
        for kernel_index in range(kernel_volume):
            count = pair_counts[kernel_index]
            if count <= 0 or (subm and kernel_index == center):
                continue
            input_ids = indice_pairs[kernel_index, gather_plane, :count].long()
            output_ids = indice_pairs[kernel_index, scatter_plane, :count].long()
            x = features.index_select(0, input_ids)
            dy = grad_output.index_select(0, output_ids)
            grad_features.index_add_(0, input_ids, dy.matmul(weights[kernel_index].transpose(0, 1)))
            grad_weights[kernel_index].add_(x.transpose(0, 1).matmul(dy))
    return grad_features, grad_weights


# Rows of one gather buffer span ``chunk * capacity`` capacity-padded pair
# slots.  The cap keeps the temporary within a predictable footprint; offsets
# are processed in as few chunks as that allows.
_NATIVE_BUFFER_BYTES = 256 * 1024 * 1024


def _native_pair_ops(features: torch.Tensor):
    """Return ``(gather, scatter_add)`` native callables, or ``None``."""
    if features.dtype == torch.float32:
        suffix = "fp32"
    elif features.dtype == torch.float16:
        suffix = "half"
    else:
        return None
    if not _is_musa_tensor(features):
        return None
    ext = _native()
    if ext is None:
        return None
    try:
        return (
            getattr(ext, f"spconv_sparse_gather_{suffix}"),
            getattr(ext, f"spconv_sparse_scatter_add_{suffix}"),
        )
    except NativeModuleUnavailable:
        return None


def _offset_chunk(kernel_volume: int, capacity: int, channels: int, itemsize: int) -> int:
    per_offset = max(1, capacity * channels * itemsize)
    return max(1, min(kernel_volume, _NATIVE_BUFFER_BYTES // per_offset))


def _native_gather_gemm_scatter_conv(
    features, weights, indice_pairs, indice_pair_num, num_activate_out, inverse=False
):
    """Native gather → batched GEMM → native scatter-add.

    The gather zero-fills capacity slots past ``indice_pair_num`` and the
    scatter skips them, so a run of kernel offsets is covered by one gather,
    one ``bmm`` and one scatter instead of three launches per offset.
    Returns ``None`` when the native kernels are unusable for these tensors.
    """
    ops = _native_pair_ops(features)
    if ops is None:
        return None
    gather, scatter_add = ops
    kernel_volume = int(weights.shape[0])
    capacity = int(indice_pairs.shape[2])
    channels_in = int(features.shape[1])
    channels_out = int(weights.shape[-1])
    output = features.new_zeros((int(num_activate_out), channels_out))
    if kernel_volume == 0 or capacity == 0:
        return output

    gather_plane = 1 if inverse else 0
    scatter_plane = 0 if inverse else 1
    chunk = _offset_chunk(
        kernel_volume, capacity, max(channels_in, channels_out), features.element_size()
    )
    pairs = indice_pairs.contiguous()
    pair_num = indice_pair_num.contiguous()
    features = features.contiguous()
    with _tf32_matmul(features.dtype):
        for start in range(0, kernel_volume, chunk):
            stop = min(start + chunk, kernel_volume)
            span = stop - start
            pairs_chunk = pairs[start:stop]
            num_chunk = pair_num[start:stop]
            buffer = features.new_empty((span * capacity, channels_in))
            gather(features, pairs_chunk, num_chunk, buffer, gather_plane)
            # Autocast can downcast GEMM independently of the sparse feature
            # buffer.  Native scatter is selected from ``features.dtype`` and
            # therefore requires its input buffer to use that same dtype.
            values = torch.bmm(
                buffer.view(span, capacity, channels_in), weights[start:stop]
            ).to(features.dtype)
            scatter_add(
                values.reshape(span * capacity, channels_out),
                pairs_chunk,
                num_chunk,
                output,
                scatter_plane,
            )
    return output


def _native_gather_gemm_scatter_conv_backward(
    features, weights, grad_output, indice_pairs, indice_pair_num, inverse=False
):
    """Backward of :func:`_native_gather_gemm_scatter_conv`.

    Padded slots gather as zero in both buffers, so they contribute nothing to
    the filter gradient and can share the same batched GEMM.
    """
    ops = _native_pair_ops(features)
    if ops is None or grad_output.dtype != features.dtype:
        return None
    gather, scatter_add = ops
    kernel_volume = int(weights.shape[0])
    capacity = int(indice_pairs.shape[2])
    channels_in = int(features.shape[1])
    channels_out = int(weights.shape[-1])
    grad_features = torch.zeros_like(features)
    grad_weights = torch.zeros_like(weights)
    if kernel_volume == 0 or capacity == 0:
        return grad_features, grad_weights

    gather_plane = 1 if inverse else 0
    scatter_plane = 0 if inverse else 1
    chunk = _offset_chunk(
        kernel_volume, capacity, channels_in + channels_out, features.element_size()
    )
    pairs = indice_pairs.contiguous()
    pair_num = indice_pair_num.contiguous()
    features = features.contiguous()
    grad_output = grad_output.contiguous()
    with _tf32_matmul(features.dtype):
        for start in range(0, kernel_volume, chunk):
            stop = min(start + chunk, kernel_volume)
            span = stop - start
            pairs_chunk = pairs[start:stop]
            num_chunk = pair_num[start:stop]
            gathered = features.new_empty((span * capacity, channels_in))
            gather(features, pairs_chunk, num_chunk, gathered, gather_plane)
            grad_values = features.new_empty((span * capacity, channels_out))
            gather(grad_output, pairs_chunk, num_chunk, grad_values, scatter_plane)
            gathered = gathered.view(span, capacity, channels_in)
            grad_values = grad_values.view(span, capacity, channels_out)
            grad_gathered = torch.bmm(grad_values, weights[start:stop].transpose(1, 2))
            scatter_add(
                grad_gathered.reshape(span * capacity, channels_in),
                pairs_chunk,
                num_chunk,
                grad_features,
                gather_plane,
            )
            grad_weights[start:stop].add_(torch.bmm(gathered.transpose(1, 2), grad_values))
    return grad_features, grad_weights


def _reference_conv(features, filters, indice_pairs, indice_pair_num, num_activate_out):
    """Strict gather-GEMM-scatter used by tests (no TF32; matches pure arithmetic)."""
    weights = _flatten_filters(filters)
    output = features.new_zeros((int(num_activate_out), weights.shape[-1]))
    for kernel_index, count in enumerate(_pair_counts(indice_pair_num)):
        if count <= 0:
            continue
        input_ids = indice_pairs[kernel_index, 0, :count].long()
        output_ids = indice_pairs[kernel_index, 1, :count].long()
        values = features.index_select(0, input_ids).matmul(weights[kernel_index])
        output.index_add_(0, output_ids, values)
    return output


# Host-only dispatch heuristic, evaluated from shapes alone (no device sync).
# The native pair path costs a near-constant launch floor, while the compact
# atomic matvec scales with the padded MAC count `K * capacity * Cin * Cout`.
# They cross around 3e7 MACs on S5000; below that the matvec is cheaper.
_NATIVE_PAIR_MAC_FLOOR = 30_000_000


def _padded_macs(weights: torch.Tensor, indice_pairs: torch.Tensor) -> int:
    return (
        int(indice_pairs.shape[0])
        * int(indice_pairs.shape[2])
        * int(weights.shape[1])
        * int(weights.shape[-1])
    )


def _prefer_native_pair_path(features, weights, indice_pairs) -> bool:
    if not _is_musa_tensor(features):
        return False
    return _padded_macs(weights, indice_pairs) >= _NATIVE_PAIR_MAC_FLOOR


def _has_no_pair_slot(weights: torch.Tensor, indice_pairs: torch.Tensor) -> bool:
    # Shape-only test, so it costs no device sync.  A zero-capacity rulebook has
    # no work for any path, and its degenerate strides are rejected by the FFI
    # contiguity check.
    return int(weights.shape[0]) == 0 or int(indice_pairs.shape[2]) == 0


def indice_conv(
    features, filters, indice_pairs, indice_pair_num, num_activate_out, inverse=False, subm=False
):
    weights = _flatten_filters(filters)
    if _has_no_pair_slot(weights, indice_pairs):
        return features.new_zeros((int(num_activate_out), weights.shape[-1]))
    # Wide shapes: native gather → batched GEMM → native scatter-add (BEVFusion
    # algorithm, vendor GEMM). Narrow shapes stay on the compact atomic matvec,
    # which beats the pair path's fixed launch floor there.
    if _prefer_native_pair_path(features, weights, indice_pairs):
        output = _native_gather_gemm_scatter_conv(
            features,
            weights,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            inverse=bool(inverse),
        )
        if output is not None:
            return output
    ext = _native()
    if ext is not None and _is_musa_tensor(features) and not inverse:
        # Native kernel has no inverse/subm special-cases; results remain
        # correct for the default forward layout used by SparseEncoder.
        output = features.new_zeros((int(num_activate_out), weights.shape[-1]))
        name = (
            "spconv_indice_conv_half"
            if features.dtype == torch.float16
            else "spconv_indice_conv_fp32"
        )
        getattr(ext, name)(features, weights, indice_pairs, indice_pair_num, output)
        return output
    return _gather_gemm_scatter_conv(
        features,
        weights,
        indice_pairs,
        _pair_counts(indice_pair_num),
        num_activate_out,
        inverse=bool(inverse),
        subm=bool(subm),
    )


def indice_conv_backward(
    features, filters, grad_output, indice_pairs, indice_pair_num, inverse=False, subm=False
):
    weights = _flatten_filters(filters)
    if _has_no_pair_slot(weights, indice_pairs):
        return torch.zeros_like(features), torch.zeros_like(filters)
    if _prefer_native_pair_path(features, weights, indice_pairs):
        native = _native_gather_gemm_scatter_conv_backward(
            features,
            weights,
            grad_output,
            indice_pairs,
            indice_pair_num,
            inverse=bool(inverse),
        )
        if native is not None:
            grad_features, grad_weights = native
            return grad_features, grad_weights.reshape_as(filters)
    ext = _native()
    if ext is not None and _is_musa_tensor(features) and not inverse:
        grad_features = torch.zeros_like(features)
        grad_weights = torch.zeros_like(weights)
        name = (
            "spconv_indice_conv_backward_half"
            if features.dtype == torch.float16
            else "spconv_indice_conv_backward_fp32"
        )
        getattr(ext, name)(
            features,
            weights,
            grad_output.contiguous(),
            indice_pairs,
            indice_pair_num,
            grad_features,
            grad_weights,
        )
        return grad_features, grad_weights.reshape_as(filters)
    grad_features, grad_weights = _gather_gemm_scatter_conv_backward(
        features,
        weights,
        grad_output,
        indice_pairs,
        _pair_counts(indice_pair_num),
        inverse=bool(inverse),
        subm=bool(subm),
    )
    return grad_features, grad_weights.reshape_as(filters)


def indice_conv_fp32(*args):
    return indice_conv(*args)


def indice_conv_half(*args):
    return indice_conv(*args)


def indice_conv_backward_fp32(*args):
    return indice_conv_backward(*args)


def indice_conv_backward_half(*args):
    return indice_conv_backward(*args)


__all__ = [
    "get_indice_pairs_3d",
    "get_indice_pairs_grid_3d",
    "indice_conv_fp32",
    "indice_conv_half",
    "indice_conv_backward_fp32",
    "indice_conv_backward_half",
]
