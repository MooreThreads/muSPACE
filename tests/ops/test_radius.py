import pytest
import torch

from mu_space import (
    clear_dispatch_report,
    get_last_dispatch_report,
    radius,
    radius_graph,
    radius_graph_reference,
    radius_reference,
)


def test_reference_radius_preserves_y_x_order_and_truncates_by_x_index():
    x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    actual = radius(x, y, 1.5, max_num_neighbors=1, backend="reference")
    expected = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    torch.testing.assert_close(actual, expected)


def test_reference_radius_batch_isolation_and_empty_segments():
    x = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
    y = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    batch_x = torch.tensor([0, 0, 2], dtype=torch.int64)
    batch_y = torch.tensor([0, 2], dtype=torch.int64)
    actual = radius(x, y, 0.5, batch_x, batch_y, backend="reference")
    expected = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.int64)
    torch.testing.assert_close(actual, expected)


def test_reference_radius_strict_boundary_and_empty_output():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    y = torch.tensor([[0.0, 0.0]])
    assert radius(x, y, 1.0, backend="reference").shape == (2, 1)
    assert radius(x, y, 0.0, backend="reference").numel() == 0
    assert radius(torch.empty((0, 2)), y, 1.0, backend="reference").shape == (2, 0)
    torch.testing.assert_close(radius_reference(x, y, 1.0), radius(x, y, 1.0, backend="reference"))
    torch.testing.assert_close(
        radius_graph_reference(x, 1.0), radius_graph(x, 1.0, backend="reference")
    )


def test_radius_graph_direction_loop_and_neighbor_budget():
    x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    actual = radius_graph(x, 2.5, loop=False, max_num_neighbors=2, backend="reference")
    expected = torch.tensor([[1, 2, 0, 3, 0, 3, 1, 2], [0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.int64)
    torch.testing.assert_close(actual, expected)
    loops = radius_graph(x, 0.1, loop=True, backend="reference")
    torch.testing.assert_close(loops, torch.arange(4, dtype=torch.int64).repeat(2, 1))


def test_validation():
    with pytest.raises(ValueError, match="same coordinate"):
        radius(torch.zeros(2, 2), torch.zeros(3, 3), 1.0, backend="reference")
    with pytest.raises(TypeError, match="float32"):
        radius(torch.zeros(2, 2, dtype=torch.float64), torch.zeros(3, 2), 1.0, backend="reference")
    with pytest.raises(ValueError, match="sorted"):
        radius(
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            1.0,
            torch.tensor([1, 0]),
            torch.tensor([0, 1]),
            backend="reference",
        )
    with pytest.raises(ValueError, match="flow"):
        radius_graph(torch.zeros(2, 2), 1.0, flow="bad", backend="reference")
    with pytest.raises(ValueError, match="max_num_neighbors"):
        radius(torch.zeros(2, 2), torch.zeros(2, 2), 1.0, max_num_neighbors=0, backend="reference")


def test_numpy_integer_neighbor_limit_is_accepted():
    np = pytest.importorskip("numpy")
    actual = radius(
        torch.zeros((1, 2)),
        torch.zeros((1, 2)),
        1.0,
        max_num_neighbors=np.int64(3),
        backend="reference",
    )
    torch.testing.assert_close(actual, torch.tensor([[0], [0]], dtype=torch.int64))


def test_numpy_and_scalar_tensor_radius_are_accepted():
    np = pytest.importorskip("numpy")
    x = torch.zeros((1, 2))
    expected = torch.tensor([[0], [0]], dtype=torch.int64)
    torch.testing.assert_close(radius(x, x, np.float32(1.0), backend="reference"), expected)
    torch.testing.assert_close(radius(x, x, torch.tensor(1.0), backend="reference"), expected)


def test_reference_sparse_large_batch_labels_and_empty_segments():
    maximum = 2**63 - 1
    x = torch.tensor([[0.0], [3.0]])
    y = torch.tensor([[0.0], [3.0]])
    bx = torch.tensor([0, maximum], dtype=torch.int64)
    by = torch.tensor([0, maximum], dtype=torch.int64)
    expected = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)
    torch.testing.assert_close(radius(x, y, 1.0, bx, by, backend="reference"), expected)


def test_reference_negative_and_unsorted_batch_labels():
    x, y = torch.zeros((2, 2)), torch.zeros((2, 2))
    valid = torch.tensor([0, 1], dtype=torch.int64)
    for invalid in (torch.tensor([-1, 0]), torch.tensor([1, 0])):
        for bx, by in ((invalid, valid), (valid, invalid)):
            with pytest.raises(ValueError, match="sorted and non-negative"):
                radius(x, y, 1.0, bx, by, backend="reference")


def test_graph_reference_dispatch_report_is_not_inner_radius():
    clear_dispatch_report()
    radius_graph(torch.zeros((3, 2)), 1.0, backend="reference")
    assert get_last_dispatch_report().operator == "radius_graph"


def test_reference_no_batch_and_batches_are_contiguous_int64():
    x = torch.zeros((2, 2))
    labels = torch.zeros(2, dtype=torch.int64)
    for result in (
        radius(x, x, 1.0, backend="reference"),
        radius(x, x, 1.0, labels, labels, backend="reference"),
    ):
        assert result.dtype == torch.int64 and result.is_contiguous()


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_forced_native_exception_records_error_and_auto_falls_back(monkeypatch):
    import importlib

    radius_module = importlib.import_module("mu_space.ops.radius")

    class FailingNative:
        def radius_forward(self, *args):
            raise RuntimeError("native test failure")

    monkeypatch.setattr(radius_module, "_native", lambda: FailingNative())
    x = torch.zeros((1, 2), device="musa")
    with pytest.raises(RuntimeError, match="native test failure"):
        radius_module.radius(x, x, 1.0, backend="musa")
    assert get_last_dispatch_report().selected_backend == "error"
    expected = radius_module.radius(x, x, 1.0, backend="auto")
    assert expected.shape == (2, 1)
    assert get_last_dispatch_report().native_fallback


def test_cpu_musa_backend_error_records_dispatch():
    clear_dispatch_report()
    with pytest.raises(RuntimeError, match="requires a MUSA tensor"):
        radius(torch.zeros((1, 2)), torch.zeros((1, 2)), 1.0, backend="musa")
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "radius"
    assert report.selected_backend == "error"


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_radius_matches_reference():
    x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], device="musa")
    y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], device="musa")
    batch_x = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device="musa")
    batch_y = torch.tensor([0, 0], dtype=torch.int32, device="musa")
    actual = radius(x, y, 1.5, batch_x, batch_y, max_num_neighbors=3, backend="musa").cpu()
    expected = radius(x.cpu(), y.cpu(), 1.5, max_num_neighbors=3, backend="reference")
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_radius_zero_is_strictly_empty():
    x = torch.zeros((3, 2), dtype=torch.float32, device="musa")
    y = torch.zeros((2, 2), dtype=torch.float32, device="musa")
    actual = radius(x, y, 0.0, max_num_neighbors=4, backend="musa")
    assert actual.shape == (2, 0)
    assert actual.is_contiguous()


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_batch_int64_max_label_matches_reference():
    maximum = 2**63 - 1
    x = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device="musa")
    y = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device="musa")
    batch_x = torch.tensor([maximum], dtype=torch.int64, device="musa")
    batch_y = torch.tensor([maximum], dtype=torch.int64, device="musa")
    actual = radius(x, y, 1.0, batch_x, batch_y, backend="musa").cpu()
    expected = radius(x.cpu(), y.cpu(), 1.0, batch_x.cpu(), batch_y.cpu(), backend="reference")
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
@pytest.mark.parametrize(
    ("coordinate_dim", "query_radius", "max_num_neighbors"),
    [(1, 0.4, 7), (2, 0.5, 37), (8, 2.0, 65)],
)
def test_native_warp_parallel_sparse_batches_match_reference(
    coordinate_dim, query_radius, max_num_neighbors
):
    generator = torch.Generator(device="cpu").manual_seed(20260914 + coordinate_dim)
    x = torch.randn((202, coordinate_dim), generator=generator, dtype=torch.float32)
    y = torch.randn((91, coordinate_dim), generator=generator, dtype=torch.float32)
    batch_x = torch.tensor([0] * 73 + [2] * 129, dtype=torch.int64)
    batch_y = torch.tensor([0] * 19 + [1] * 7 + [2] * 65, dtype=torch.int64)
    expected = radius(
        x,
        y,
        query_radius,
        batch_x,
        batch_y,
        max_num_neighbors,
        backend="reference",
    )
    actual = radius(
        x.musa(),
        y.musa(),
        query_radius,
        batch_x.musa(),
        batch_y.musa(),
        max_num_neighbors,
        backend="musa",
    ).cpu()
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_warp_parallel_preserves_index_order_across_tiles():
    x = torch.zeros((96, 2), dtype=torch.float32, device="musa")
    y = torch.zeros((3, 2), dtype=torch.float32, device="musa")
    batch_x = torch.zeros(96, dtype=torch.int64, device="musa")
    batch_y = torch.zeros(3, dtype=torch.int64, device="musa")
    actual = radius(x, y, 1.0, batch_x, batch_y, 37, backend="musa").cpu()
    expected_y = torch.arange(3, dtype=torch.int64).repeat_interleave(37)
    expected_x = torch.arange(37, dtype=torch.int64).repeat(3)
    torch.testing.assert_close(actual, torch.stack([expected_y, expected_x]))
    assert actual.is_contiguous()


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_compaction_returns_contiguous_empty_edge_index():
    x = torch.zeros((64, 2), dtype=torch.float32, device="musa")
    y = torch.full((17, 2), 100.0, dtype=torch.float32, device="musa")
    batch_x = torch.zeros(64, dtype=torch.int64, device="musa")
    batch_y = torch.zeros(17, dtype=torch.int64, device="musa")
    actual = radius(x, y, 0.5, batch_x, batch_y, 13, backend="musa")
    assert actual.shape == (2, 0)
    assert actual.is_contiguous()


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_no_batch_warp_parallel_path_matches_reference():
    generator = torch.Generator(device="cpu").manual_seed(314159)
    x = torch.randn((129, 2), generator=generator, dtype=torch.float32)
    y = torch.randn((35, 2), generator=generator, dtype=torch.float32)
    expected = radius(x, y, 0.75, max_num_neighbors=41, backend="reference")
    actual = radius(x.musa(), y.musa(), 0.75, max_num_neighbors=41, backend="musa").cpu()
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_musa_batch_validation_matches_reference_contract():
    x = torch.zeros((3, 2), dtype=torch.float32, device="musa")
    y = torch.zeros((2, 2), dtype=torch.float32, device="musa")
    valid_x = torch.tensor([0, 0, 1], dtype=torch.int64, device="musa")
    valid_y = torch.tensor([0, 1], dtype=torch.int64, device="musa")
    for invalid_x, invalid_y in (
        (torch.tensor([0, 1, 0], dtype=torch.int64, device="musa"), valid_y),
        (torch.tensor([-1, 0, 0], dtype=torch.int64, device="musa"), valid_y),
        (valid_x, torch.tensor([1, 0], dtype=torch.int64, device="musa")),
    ):
        with pytest.raises(ValueError, match="sorted and non-negative"):
            radius(x, y, 1.0, invalid_x, invalid_y, backend="reference")
        with pytest.raises(ValueError, match="sorted and non-negative"):
            radius(x, y, 1.0, invalid_x, invalid_y, backend="musa")


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_empty_musa_dispatch_is_musa():
    clear_dispatch_report()
    x = torch.empty((0, 2), dtype=torch.float32, device="musa")
    y = torch.empty((0, 2), dtype=torch.float32, device="musa")
    actual = radius(x, y, 1.0, backend="musa")
    assert actual.shape == (2, 0)
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "radius"
    assert report.selected_backend == "musa"
    assert "empty input" in report.reason


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
@pytest.mark.parametrize(
    "loop,flow",
    [(False, "source_to_target"), (False, "target_to_source"), (True, "source_to_target")],
)
def test_native_radius_graph_matches_reference_and_reports_graph(loop, flow):
    x = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=torch.float32,
        device="musa",
    )
    batch = torch.zeros(4, dtype=torch.int64, device="musa")
    kwargs = {"loop": loop, "flow": flow, "max_num_neighbors": 2}
    query_radius = 2.5 if not loop else 0.1
    actual = radius_graph(x, query_radius, batch, backend="musa", **kwargs).cpu()
    expected = radius_graph(x.cpu(), query_radius, batch.cpu(), backend="reference", **kwargs)
    torch.testing.assert_close(actual, expected)
    report = get_last_dispatch_report()
    assert report is not None
    assert report.operator == "radius_graph"


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_no_batch_zero_neighbor_output_is_contiguous():
    x = torch.zeros((1, 2), dtype=torch.float32, device="musa")
    y = torch.full((1, 2), 100.0, dtype=torch.float32, device="musa")
    actual = radius(x, y, 1.0, backend="musa")
    assert actual.shape == (2, 0)
    assert actual.is_contiguous()


@pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(), reason="MUSA required"
)
def test_native_ptr_and_batch_paths_match_exactly():
    from mu_space.runtime.native import native_module_proxy

    x = torch.tensor([[0.0, 0.0], [0.2, 0.0], [10.0, 0.0], [10.2, 0.0]], device="musa")
    y = torch.tensor([[0.0, 0.0], [10.0, 0.0]], device="musa")
    batch_x = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device="musa")
    batch_y = torch.tensor([0, 1], dtype=torch.int64, device="musa")
    expected = radius(x, y, 0.5, batch_x, batch_y, 3, backend="musa").cpu()
    ptr_x = torch.tensor([0, 2, 4], dtype=torch.int32, device="musa")
    ptr_y = torch.tensor([0, 1, 2], dtype=torch.int32, device="musa")
    fixed_x = torch.empty(6, dtype=torch.int64, device="musa")
    counts = torch.empty(2, dtype=torch.int32, device="musa")
    native = native_module_proxy()
    native.radius_forward(x, y, ptr_x, ptr_y, fixed_x, counts, 0.5, 3)
    inclusive = torch.cumsum(counts, dim=0, dtype=torch.int64)
    actual = torch.empty((2, int(inclusive[-1].item())), dtype=torch.int64, device="musa")
    native.radius_compact_forward(fixed_x, counts, inclusive, actual, 3)
    torch.testing.assert_close(actual.cpu(), expected)
