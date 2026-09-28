import pytest
import torch

from mu_space.ops.spconv import (
    _flatten_filters,
    _native_gather_gemm_scatter_conv,
    _native_gather_gemm_scatter_conv_backward,
    _reference_conv,
    get_indice_pairs_3d,
    indice_conv,
    indice_conv_backward,
)
from mu_space.runtime.native import native_available


_MUSA = hasattr(torch, "musa") and torch.musa.is_available()
_NATIVE = _MUSA and native_available()


def _fixture(dtype=torch.float32):
    # Two batches with a deliberately non-dense, asymmetric active set.
    indices = torch.tensor(
        [[0, 1, 1, 1], [0, 1, 1, 2], [0, 2, 1, 1], [1, 2, 2, 2]], dtype=torch.int32
    )
    features = torch.arange(1, 1 + indices.shape[0] * 2, dtype=dtype).view(-1, 2) / 7
    filters = torch.arange(1, 1 + 3 * 2 * 2, dtype=dtype).view(3, 2, 2) / 11
    return indices, features, filters


def test_subm_indices_preserve_active_coordinates_and_are_reusable():
    indices, _, _ = _fixture()
    first = get_indice_pairs_3d(indices, 2, (4, 4, 4), (4, 4, 4), 3, 1, 1, 1, 0, 1, 0)
    second = get_indice_pairs_3d(indices, 2, (4, 4, 4), (4, 4, 4), 3, 1, 1, 1, 0, 1, 0)
    torch.testing.assert_close(first[0], indices)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(first[2], second[2])
    assert int(first[2].max()) > 0


@pytest.mark.parametrize("stride", [1, 2])
def test_sparse_conv_indices_have_valid_output_coordinates(stride):
    indices, _, _ = _fixture()
    result = get_indice_pairs_3d(indices, 2, (4, 4, 4), (4, 4, 4), 3, stride, 1, 1, 0, 0, 0)
    outids, pairs, pair_num = result
    out_shape = (2, 2, 2) if stride == 2 else (4, 4, 4)
    assert outids.ndim == 2 and outids.shape[1] == 4
    assert pairs.shape[:2] == (27, 2)
    assert pair_num.shape == (27,)
    assert all(0 <= int(outids[:, axis].min()) for axis in range(1, 4))
    assert all(int(outids[:, axis].max()) < out_shape[axis - 1] for axis in range(1, 4))


def test_forward_and_backward_match_manual_reference():
    indices, features, filters = _fixture()
    outids, pairs, pair_num = get_indice_pairs_3d(
        indices, 2, (4, 4, 4), (4, 4, 4), 1, 1, 0, 1, 0, 1, 0
    )
    features = features.requires_grad_()
    filters = filters[:1].contiguous().requires_grad_()
    output = indice_conv(features, filters, pairs, pair_num, outids.shape[0])
    expected = torch.zeros_like(output)
    for kernel_index in range(filters.shape[0]):
        count = int(pair_num[kernel_index])
        for slot in range(count):
            input_index = int(pairs[kernel_index, 0, slot])
            output_index = int(pairs[kernel_index, 1, slot])
            expected[output_index] += features[input_index] @ filters[kernel_index]
    torch.testing.assert_close(output, expected)

    grad_output = torch.randn_like(output)
    grad_features, grad_filters = indice_conv_backward(
        features.detach(), filters.detach(), grad_output, pairs, pair_num
    )
    output.backward(grad_output)
    torch.testing.assert_close(features.grad, grad_features)
    torch.testing.assert_close(filters.grad, grad_filters)


def test_dilation_two_rulebook_matches_reference():
    indices = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 2, 2, 2],
            [0, 3, 3, 3],
            [1, 2, 1, 3],
        ],
        dtype=torch.int32,
    )
    outids, pairs, pair_num = get_indice_pairs_3d(
        indices,
        2,
        (6, 6, 6),
        (6, 6, 6),
        3,
        1,
        2,
        2,
        0,
        0,
        0,
    )
    assert int(pair_num.sum()) > 0
    assert pairs.shape[0] == 27
    assert outids.shape[1] == 4
    assert all(0 <= int(outids[:, axis].min()) for axis in range(1, 4))
    assert all(int(outids[:, axis].max()) < 6 for axis in range(1, 4))


def test_empty_input_is_supported():
    indices = torch.empty((0, 4), dtype=torch.int32)
    outids, pairs, pair_num = get_indice_pairs_3d(
        indices, 1, (4, 4, 4), (4, 4, 4), 3, 1, 1, 1, 0, 1, 0
    )
    assert outids.shape == (0, 4)
    assert pairs.shape == (27, 2, 0)
    assert pair_num.sum() == 0


@pytest.mark.skipif(not _MUSA, reason="MUSA required")
def test_empty_input_is_supported_on_musa():
    indices = torch.empty((0, 4), dtype=torch.int32).musa()
    outids, pairs, pair_num = get_indice_pairs_3d(
        indices, 1, (4, 4, 4), (4, 4, 4), 3, 1, 1, 1, 0, 1, 0
    )
    assert outids.shape == (0, 4)
    assert pairs.shape == (27, 2, 0)
    assert int(pair_num.sum()) == 0
    features = torch.empty((0, 2), dtype=torch.float32).musa()
    filters = torch.arange(1, 1 + 27 * 2 * 2, dtype=torch.float32).view(27, 2, 2).musa() / 11
    output = indice_conv(features, filters, pairs, pair_num, outids.shape[0])
    assert output.shape == (0, 2)
    grad_features, grad_filters = indice_conv_backward(features, filters, output, pairs, pair_num)
    assert grad_features.shape == features.shape
    torch.testing.assert_close(grad_filters, torch.zeros_like(grad_filters))


def _coordinate_rows(outids):
    return [tuple(int(v) for v in row) for row in outids.detach().to("cpu").tolist()]


def _pair_sets(outids, pairs, pair_num):
    """Per-offset ``{(input index, output coordinate)}``, insensitive to slot order.

    Pairs are keyed by output coordinate rather than by output index: the two
    implementations number the sparse-conv output voxels differently (sorted key
    order against first-encounter order), and only the coordinates are contract.
    """
    coordinates = _coordinate_rows(outids)
    pairs = pairs.detach().to("cpu")
    sets = []
    for kernel_index, count in enumerate(pair_num.detach().to("cpu").tolist()):
        sets.append(
            {
                (
                    int(pairs[kernel_index, 0, slot]),
                    coordinates[int(pairs[kernel_index, 1, slot])],
                )
                for slot in range(int(count))
            }
        )
    return sets


def _canonical_order(outids):
    coordinates = _coordinate_rows(outids)
    return sorted(range(len(coordinates)), key=lambda i: coordinates[i])


@pytest.mark.skipif(not _MUSA, reason="MUSA required")
def test_musa_dilation_two_rulebook_matches_host():
    indices = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 2, 2, 2],
            [0, 3, 3, 3],
            [1, 2, 1, 3],
        ],
        dtype=torch.int32,
    )
    arguments = (2, (6, 6, 6), (6, 6, 6), 3, 1, 2, 2, 0, 0, 0)
    host = get_indice_pairs_3d(indices, *arguments)
    device = get_indice_pairs_3d(indices.musa(), *arguments)
    torch.testing.assert_close(device[0].cpu(), host[0])
    torch.testing.assert_close(device[2].cpu(), host[2])
    torch.testing.assert_close(device[1].cpu(), host[1])


@pytest.mark.skipif(not _MUSA, reason="MUSA required")
@pytest.mark.parametrize("subm,stride", [(1, 1), (0, 1), (0, 2)])
def test_musa_pairs_match_host_reference(subm, stride):
    # The vectorized MUSA rulebook and the host dict reference must agree item
    # by item: same output voxel set, same live count per offset, same pairs.
    indices, _, _ = _fixture()
    out_shape = (2, 2, 2) if stride == 2 else (4, 4, 4)
    arguments = (2, out_shape, (4, 4, 4), 3, stride, 1, 1, 0, subm, 0)
    host = get_indice_pairs_3d(indices, *arguments)
    device = get_indice_pairs_3d(indices.musa(), *arguments)
    torch.testing.assert_close(device[0].cpu(), host[0])
    torch.testing.assert_close(device[2].cpu(), host[2])
    torch.testing.assert_close(device[1].cpu(), host[1])
    assert int(host[2].sum()) > 0


@pytest.mark.skipif(not _MUSA, reason="MUSA required")
@pytest.mark.parametrize("subm,stride", [(1, 1), (0, 1), (0, 2)])
def test_musa_conv_matches_host_reference_on_own_pairs(subm, stride):
    # End-to-end cross-implementation regression: both rulebooks and all
    # outputs use the host reference's public row order directly.
    indices, features, filters = _fixture()
    filters = filters.reshape(3, 1, 1, 2, 2).repeat(1, 3, 3, 1, 1).contiguous()
    out_shape = (2, 2, 2) if stride == 2 else (4, 4, 4)
    arguments = (2, out_shape, (4, 4, 4), 3, stride, 1, 1, 0, subm, 0)

    host_ids, host_pairs, host_num = get_indice_pairs_3d(indices, *arguments)
    host_features = features.clone().requires_grad_()
    host_filters = filters.clone().requires_grad_()
    host_output = _reference_conv(
        host_features, host_filters, host_pairs, host_num, host_ids.shape[0]
    )

    device_ids, device_pairs, device_num = get_indice_pairs_3d(indices.musa(), *arguments)
    device_features = features.musa().requires_grad_()
    device_filters = filters.musa().requires_grad_()
    device_output = indice_conv(
        device_features,
        device_filters,
        device_pairs,
        device_num,
        device_ids.shape[0],
        subm=bool(subm),
    )

    torch.testing.assert_close(
        device_output.cpu(), host_output.detach(), rtol=5e-3, atol=5e-3
    )

    grad_canonical = torch.randn(host_output.shape, dtype=features.dtype)
    host_grad = grad_canonical
    device_grad = grad_canonical.musa()

    host_output.backward(host_grad)
    device_grad_features, device_grad_filters = indice_conv_backward(
        device_features.detach(), device_filters.detach(), device_grad, device_pairs, device_num
    )
    torch.musa.synchronize()
    torch.testing.assert_close(device_grad_features.cpu(), host_features.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(device_grad_filters.cpu(), host_filters.grad, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not _MUSA, reason="MUSA required")
def test_sparseconv_indice_key_cache_reuses_rulebook(monkeypatch):
    try:
        from bevfusion.mmdet3d.ops.spconv import conv as spconv_conv
        from bevfusion.mmdet3d.ops.spconv.conv import SubMConv3d
        from bevfusion.mmdet3d.ops.spconv.structure import SparseConvTensor
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"BEVFusion spconv dependency unavailable: {exc}")

    calls = {"pairs": 0}

    def fake_get_indice_pairs(
        indices,
        batch_size,
        spatial_shape,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        out_padding=0,
        subm=False,
        transpose=False,
        grid=None,
    ):
        del batch_size, spatial_shape, ksize, stride, padding, dilation, out_padding, subm, transpose, grid
        calls["pairs"] += 1
        outids = indices.clone()
        pair_num = torch.zeros((27,), dtype=torch.int32)
        pair_num[13] = indices.shape[0]
        pairs = torch.full((27, 2, int(indices.shape[0])), -1, dtype=torch.int32)
        if indices.shape[0] > 0:
            pair_ids = torch.arange(indices.shape[0], dtype=torch.int32)
            pairs[13, 0, : indices.shape[0]] = pair_ids
            pairs[13, 1, : indices.shape[0]] = pair_ids
        return outids, pairs, pair_num

    def fake_subm_conv(features, weight, indice_pairs, indice_pair_num, num_out):
        del indice_pairs, indice_pair_num
        return features.new_full((num_out, weight.shape[-1]), float(features.sum()))

    monkeypatch.setattr(spconv_conv.ops, "get_indice_pairs", fake_get_indice_pairs)
    monkeypatch.setattr(spconv_conv.Fsp, "indice_subm_conv", fake_subm_conv)

    indices = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int32)
    features = torch.arange(1, 1 + indices.shape[0] * 2, dtype=torch.float32).view(-1, 2)
    tensor = SparseConvTensor(features, indices, (2, 2, 2), 1)
    layer = SubMConv3d(2, 2, 3, indice_key="reuse", bias=False)

    first = layer(tensor)
    second = layer(tensor)

    assert calls["pairs"] == 1
    assert "reuse" in tensor.indice_dict
    torch.testing.assert_close(first.features, second.features)
    torch.testing.assert_close(first.indices, second.indices)


@pytest.mark.skipif(not _MUSA, reason="MUSA required")
def test_host_subm_fallback_supports_autograd():
    # The SubM center-offset shortcut must not write in place into the output.
    indices, features, filters = _fixture()
    outids, pairs, pair_num = get_indice_pairs_3d(
        indices, 2, (4, 4, 4), (4, 4, 4), 3, 1, 1, 1, 0, 1, 0
    )
    filters = filters.reshape(3, 1, 1, 2, 2).repeat(1, 3, 3, 1, 1).contiguous()
    features = features.requires_grad_()
    filters = filters.requires_grad_()
    output = indice_conv(features, filters, pairs, pair_num, outids.shape[0], subm=True)
    expected = _reference_conv(
        features.detach(), filters.detach(), pairs, pair_num, outids.shape[0]
    )
    torch.testing.assert_close(output.detach(), expected)
    output.sum().backward()
    assert features.grad is not None and filters.grad is not None


@pytest.mark.parametrize("device", ["cpu", "musa"])
def test_subm_stride_other_than_one_is_rejected(device):
    if device == "musa" and not _MUSA:
        pytest.skip("MUSA required")
    indices, _, _ = _fixture()
    with pytest.raises(ValueError, match="subm convolution requires stride=1"):
        get_indice_pairs_3d(
            indices.to(device), 2, (2, 2, 2), (4, 4, 4), 3, 2, 1, 1, 0, 1, 0
        )


@pytest.mark.skipif(not _MUSA, reason="MUSA required")
def test_native_musa_forward_and_backward_match_reference():
    # Production MUSA path enables TF32 for GEMMs; compare against a strict
    # reference with tolerances that cover reduced-precision matmul.
    indices, features, filters = _fixture()
    indices = indices.musa()
    features = features.musa().requires_grad_()
    filters = filters.musa().reshape(3, 1, 1, 2, 2).repeat(1, 3, 3, 1, 1).requires_grad_()
    outids, pairs, pair_num = get_indice_pairs_3d(
        indices, 2, (4, 4, 4), (4, 4, 4), 3, 1, 1, 1, 0, 1, 0
    )
    native = indice_conv(features, filters, pairs, pair_num, outids.shape[0])
    reference = _reference_conv(features, filters, pairs, pair_num, outids.shape[0])
    grad_output = torch.randn_like(native)
    reference.backward(grad_output)
    grad_features, grad_filters = indice_conv_backward(
        features.detach(), filters.detach(), grad_output, pairs, pair_num
    )
    torch.musa.synchronize()
    torch.testing.assert_close(native, reference.detach(), rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(grad_features, features.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(grad_filters, filters.grad, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not _NATIVE, reason="MUSA AOT native library required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_native_pair_path_matches_reference(dtype):
    # The fixture is far below the padded-MAC dispatch floor, so the gather →
    # batched GEMM → scatter-add path has to be invoked directly to cover it.
    indices, features, filters = _fixture(dtype)
    indices = indices.musa()
    features = features.musa().requires_grad_()
    filters = filters.musa().reshape(3, 1, 1, 2, 2).repeat(1, 3, 3, 1, 1).requires_grad_()
    outids, pairs, pair_num = get_indice_pairs_3d(
        indices, 2, (4, 4, 4), (4, 4, 4), 3, 1, 1, 1, 0, 1, 0
    )
    weights = _flatten_filters(filters.detach())
    output = _native_gather_gemm_scatter_conv(
        features.detach(), weights, pairs, pair_num, outids.shape[0]
    )
    reference = _reference_conv(features, filters, pairs, pair_num, outids.shape[0])
    grad_output = torch.randn_like(reference)
    reference.backward(grad_output)
    grad_features, grad_weights = _native_gather_gemm_scatter_conv_backward(
        features.detach(), weights, grad_output, pairs, pair_num
    )
    torch.musa.synchronize()
    tolerance = 5e-3 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(output, reference.detach(), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(grad_features, features.grad, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        grad_weights.reshape_as(filters), filters.grad, rtol=tolerance, atol=tolerance
    )
