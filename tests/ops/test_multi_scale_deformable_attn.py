from importlib import import_module

import pytest
import torch

from mu_space import (
    clear_dispatch_report,
    explain_dispatch,
    get_last_dispatch_report,
    multi_scale_deformable_attn,
)

msda_module = import_module("mu_space.ops.multi_scale_deformable_attn")


def _noncontiguous_rand(shape, *, generator, dtype, device):
    return torch.randn(*shape, 2, generator=generator, dtype=dtype).to(device)[..., 0]


def _fixture(
    device="cpu",
    dtype=torch.float32,
    *,
    batch=1,
    queries=5,
    heads=2,
    embed_dims=3,
    levels=((2, 2),),
    points=4,
    noncontiguous=False,
):
    generator = torch.Generator(device="cpu").manual_seed(20260714)
    value_length = sum(height * width for height, width in levels)
    shape = (batch, value_length, heads, embed_dims)
    if noncontiguous:
        value = _noncontiguous_rand(shape, generator=generator, dtype=dtype, device=device)
    else:
        value = torch.randn(*shape, generator=generator, dtype=dtype).to(device)
    value.requires_grad_(True)
    spatial_shapes = torch.tensor(levels, dtype=torch.int64, device=device)
    level_sizes = spatial_shapes[:, 0] * spatial_shapes[:, 1]
    level_start_index = torch.cat([level_sizes.new_zeros(1), level_sizes.cumsum(0)[:-1]])
    sampling_shape = (batch, queries, heads, len(levels), points, 2)
    if noncontiguous:
        sampling_locations = torch.rand(*sampling_shape, 2, generator=generator, dtype=dtype).to(
            device
        )[..., 0]
    else:
        sampling_locations = torch.rand(*sampling_shape, generator=generator, dtype=dtype).to(
            device
        )
    sampling_locations.requires_grad_(True)
    attention_shape = (batch, queries, heads, len(levels), points)
    if noncontiguous:
        attention_weights = torch.rand(*attention_shape, 2, generator=generator, dtype=dtype).to(
            device
        )[..., 0]
    else:
        attention_weights = torch.rand(*attention_shape, generator=generator, dtype=dtype).to(
            device
        )
    attention_weights = torch.softmax(attention_weights, dim=-1)
    attention_weights.requires_grad_(True)
    return value, spatial_shapes, level_start_index, sampling_locations, attention_weights


def test_reference_forward_backward_on_cpu():
    clear_dispatch_report()
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = _fixture()
    output = multi_scale_deformable_attn(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        backend="reference",
    )
    assert output.shape == (1, 5, 6)

    output.sum().backward()
    assert value.grad is not None
    assert sampling_locations.grad is not None
    assert attention_weights.grad is not None
    assert torch.isfinite(value.grad).all()
    assert torch.isfinite(sampling_locations.grad).all()
    assert torch.isfinite(attention_weights.grad).all()
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "multi_scale_deformable_attn"
    assert report.requested_backend == "reference"
    assert report.selected_backend == "reference"
    assert report.native_fallback is False


def test_auto_cpu_dispatch_uses_reference():
    clear_dispatch_report()
    value, *_ = _fixture()
    decision = explain_dispatch(value, backend="auto")
    assert decision.backend == "reference"
    assert decision.device == "cpu"


@pytest.mark.parametrize(
    ("batch", "queries", "heads", "embed_dims", "levels", "points"),
    [
        (1, 5, 2, 3, ((2, 2),), 4),
        (2, 3, 4, 2, ((1, 3), (2, 1)), 2),
        (1, 7, 1, 8, ((1, 1), (1, 2), (2, 2)), 3),
    ],
)
def test_reference_shape_coverage_on_cpu(batch, queries, heads, embed_dims, levels, points):
    tensors = _fixture(
        batch=batch,
        queries=queries,
        heads=heads,
        embed_dims=embed_dims,
        levels=levels,
        points=points,
    )
    output = multi_scale_deformable_attn(*tensors, backend="reference")
    assert output.shape == (batch, queries, heads * embed_dims)
    output.square().mean().backward()
    value, _, _, sampling_locations, attention_weights = tensors
    assert torch.isfinite(value.grad).all()
    assert torch.isfinite(sampling_locations.grad).all()
    assert torch.isfinite(attention_weights.grad).all()


def test_reference_accepts_noncontiguous_inputs_on_cpu():
    tensors = _fixture(
        batch=2,
        queries=3,
        heads=2,
        embed_dims=4,
        levels=((1, 2), (2, 2)),
        points=2,
        noncontiguous=True,
    )
    value, _, _, sampling_locations, _ = tensors
    assert not value.is_contiguous()
    assert not sampling_locations.is_contiguous()
    output = multi_scale_deformable_attn(*tensors, backend="reference")
    assert output.shape == (2, 3, 8)


def test_sampling_and_attention_dtypes_follow_value_dtype():
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = _fixture()
    output = multi_scale_deformable_attn(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations.double(),
        attention_weights.double(),
        backend="reference",
    )
    assert output.dtype == value.dtype


def test_invalid_backend_fails_explicitly():
    clear_dispatch_report()
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = _fixture()
    with pytest.raises(RuntimeError, match="unknown backend"):
        multi_scale_deformable_attn(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            backend="unknown",
        )
    report = get_last_dispatch_report()
    assert report is not None
    assert report.selected_backend == "error"
    assert report.reason == "unknown backend: unknown"


def test_dispatch_log_env_writes_stderr(monkeypatch, capsys):
    clear_dispatch_report()
    monkeypatch.setenv("MUSPACE_DISPATCH_LOG", "1")
    tensors = _fixture()
    multi_scale_deformable_attn(*tensors, backend="reference")
    captured = capsys.readouterr()
    assert "muSPACE dispatch:" in captured.err
    assert '"selected_backend": "reference"' in captured.err


def test_auto_native_failure_records_reference_fallback(monkeypatch):
    clear_dispatch_report()
    tensors = _fixture()
    monkeypatch.setattr(msda_module, "_native_extension", lambda: object())
    monkeypatch.setattr(msda_module, "_requires_native_backend", lambda *args: True)
    monkeypatch.setattr(
        msda_module,
        "_explain_dispatch",
        lambda value, *, backend, native_available: msda_module.DispatchDecision(
            "musa", value.device.type, "simulated MUSA tensor selects native backend"
        ),
    )

    def fail_native(*args, **kwargs):
        raise RuntimeError("simulated native failure")

    monkeypatch.setattr(msda_module._NativeMSDAFunction, "apply", fail_native)
    output = multi_scale_deformable_attn(*tensors, backend="auto")
    assert output.shape == (1, 5, 6)
    report = get_last_dispatch_report()
    assert report is not None
    assert report.requested_backend == "auto"
    assert report.selected_backend == "reference"
    assert report.native_available is True
    assert report.native_fallback is True
    assert report.fallback_reason == "simulated native failure"


def test_spatial_shape_dtype_is_validated():
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = _fixture()
    with pytest.raises(TypeError, match="spatial_shapes must be int64"):
        multi_scale_deformable_attn(
            value,
            spatial_shapes.to(torch.int32),
            level_start_index,
            sampling_locations,
            attention_weights,
            backend="reference",
        )


def test_reference_out_of_range_samples_are_zero_with_zero_gradients():
    tensors = list(_fixture(batch=1, queries=2, heads=1, embed_dims=2, levels=((2, 2),), points=2))
    with torch.no_grad():
        tensors[3].fill_(-1.0)
    output = multi_scale_deformable_attn(*tensors, backend="reference")
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    for tensor in (tensors[0], tensors[3], tensors[4]):
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor.grad))


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_musa_matches_reference_forward():
    tensors = _fixture(device="musa")
    reference = multi_scale_deformable_attn(*tensors, backend="reference")
    actual = multi_scale_deformable_attn(*tensors, backend="musa")
    torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_zero_queries_returns_empty_output_and_zero_gradients():
    value, spatial_shapes, level_start_index, locations, weights = _fixture(device="musa", queries=0)
    assert locations.is_contiguous() and weights.is_contiguous()
    output = multi_scale_deformable_attn(
        value, spatial_shapes, level_start_index, locations, weights, backend="musa"
    )
    assert output.shape == (1, 0, 6)
    output.sum().backward()
    for tensor in (value, locations, weights):
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor.grad))


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_musa_matches_reference_backward_on_multiple_levels():
    native = _fixture(
        device="musa", batch=1, queries=3, heads=2, embed_dims=2, levels=((1, 2), (2, 1)), points=2
    )
    reference = tuple(
        tensor.detach().cpu().clone().requires_grad_(tensor.requires_grad)
        if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point
        else tensor.detach().cpu().clone()
        for tensor in native
    )
    expected = multi_scale_deformable_attn(*reference, backend="reference")
    actual = multi_scale_deformable_attn(*native, backend="musa")
    grad = torch.linspace(0.1, 1.0, actual.numel(), device="musa").view_as(actual)
    (actual * grad).sum().backward()
    (expected * grad.cpu()).sum().backward()
    for actual_tensor, expected_tensor in zip(
        (native[0], native[3], native[4]), (reference[0], reference[3], reference[4])
    ):
        torch.testing.assert_close(
            actual_tensor.grad.cpu(), expected_tensor.grad, rtol=1e-4, atol=1e-4
        )


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_native_musa_shape_and_dtype_coverage(dtype):
    tensors = _fixture(
        device="musa",
        dtype=dtype,
        batch=1,
        queries=4,
        heads=2,
        embed_dims=4,
        levels=((1, 2), (2, 1)),
        points=2,
    )
    reference = multi_scale_deformable_attn(*tensors, backend="reference")
    actual = multi_scale_deformable_attn(*tensors, backend="musa")
    tolerance = 1e-2 if dtype is torch.float16 else 1e-4
    torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=tolerance, atol=tolerance)
