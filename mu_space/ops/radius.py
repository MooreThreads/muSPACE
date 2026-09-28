"""Deterministic radius and radius-graph neighborhood queries."""

from __future__ import annotations

import math
import numbers
import operator
from typing import Optional

import torch

from ..runtime.kernel_dispatch import (
    BackendName,
    DispatchReport,
    explain_dispatch,
    record_dispatch_report,
    get_last_dispatch_report,
)
from ..runtime.native import NativeModuleUnavailable, native_module_proxy

OPERATOR_NAME = "radius"
GRAPH_OPERATOR_NAME = "radius_graph"


def _native():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _validate_points(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x and y must be torch.Tensor instances")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise TypeError("x and y must use float32")
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must have shape [N, D] and [M, D]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same coordinate dimension")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if x.shape[1] >= 9:
        raise ValueError("radius supports fewer than 9 coordinate dimensions")
    if x.shape[1] == 0:
        raise ValueError("radius requires at least one coordinate dimension")
    return x.contiguous(), y.contiguous()


def _validate_batch(batch: Optional[torch.Tensor], size: int, name: str) -> Optional[torch.Tensor]:
    if batch is None:
        return None
    if not isinstance(batch, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if batch.ndim != 1 or batch.numel() != size:
        raise ValueError(f"{name} must have shape [{size}]")
    if batch.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64")
    batch = batch.contiguous()
    if batch.numel():
        valid = batch[0] >= 0
        if batch.numel() > 1:
            valid = valid & (batch[1:] >= batch[:-1]).all()
        if not bool(valid.item()):
            raise ValueError(f"{name} must be sorted and non-negative")
    return batch


def _pointers(
    batch_x: Optional[torch.Tensor],
    batch_y: Optional[torch.Tensor],
    x_size: int,
    y_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_x is None and batch_y is None:
        return (
            torch.tensor([0, x_size], dtype=torch.int32, device=device),
            torch.tensor([0, y_size], dtype=torch.int32, device=device),
        )
    if batch_x is None or batch_y is None:
        raise ValueError("batch_x and batch_y must be provided together")
    # Only present labels need a segment.  Sparse labels may be as large as
    # int64 max, so expanding arange(max_label + 1) is neither safe nor useful.
    ids = torch.unique(torch.cat([batch_x.to(torch.int64), batch_y.to(torch.int64)]))
    ptr_x = torch.searchsorted(batch_x.to(torch.int64), ids)
    ptr_y = torch.searchsorted(batch_y.to(torch.int64), ids)
    ptr_x = torch.cat([ptr_x, torch.tensor([x_size], device=device, dtype=torch.int64)])
    ptr_y = torch.cat([ptr_y, torch.tensor([y_size], device=device, dtype=torch.int64)])
    return ptr_x.to(torch.int32), ptr_y.to(torch.int32)


def _reference(
    x: torch.Tensor,
    y: torch.Tensor,
    ptr_x: torch.Tensor,
    ptr_y: torch.Tensor,
    radius: float,
    max_num_neighbors: int,
) -> torch.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty((2, 0), dtype=torch.int64, device=x.device)
    r2 = float(radius) * float(radius)
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for batch in range(ptr_x.numel() - 1):
        x_left, x_right = int(ptr_x[batch]), int(ptr_x[batch + 1])
        y_left, y_right = int(ptr_y[batch]), int(ptr_y[batch + 1])
        if x_right <= x_left or y_right <= y_left:
            continue
        xb = x[x_left:x_right]
        for y_index in range(y_left, y_right):
            selected = torch.nonzero(
                (xb - y[y_index]).square().sum(dim=1) < r2, as_tuple=False
            ).flatten()[:max_num_neighbors]
            if selected.numel():
                cols.append(selected.to(torch.int64) + x_left)
                rows.append(torch.full_like(selected, y_index, dtype=torch.int64))
    if not rows:
        return torch.empty((2, 0), dtype=torch.int64, device=x.device)
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


def radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    """Return ``[y, x]`` pairs whose points are strictly within radius ``r``.

    ``num_workers`` is accepted for torch-cluster API compatibility and is
    ignored by the MUSA and PyTorch implementations.
    """
    del num_workers
    x, y = _validate_points(x, y)
    is_musa = x.device.type in {"musa", "privateuseone"} or bool(getattr(x, "is_musa", False))
    batch_x = _validate_batch(batch_x, x.shape[0], "batch_x")
    batch_y = _validate_batch(batch_y, y.shape[0], "batch_y")
    if batch_x is not None and batch_x.device != x.device:
        raise ValueError("batch_x must be on the same device as x")
    if batch_y is not None and batch_y.device != y.device:
        raise ValueError("batch_y must be on the same device as y")
    if (batch_x is None) != (batch_y is None):
        raise ValueError("batch_x and batch_y must be provided together")
    if not isinstance(r, numbers.Real) and not (isinstance(r, torch.Tensor) and r.ndim == 0):
        raise TypeError("r must be a scalar number")
    try:
        radius_value = float(r)
    except (TypeError, ValueError) as exc:
        raise TypeError("r must be a scalar number") from exc
    if not math.isfinite(radius_value) or radius_value < 0:
        raise ValueError("r must be a non-negative number")
    try:
        max_num_neighbors = operator.index(max_num_neighbors)
    except TypeError as exc:
        raise TypeError("max_num_neighbors must be an integer") from exc
    if max_num_neighbors <= 0:
        raise ValueError("max_num_neighbors must be a positive integer")
    native = _native() if is_musa and backend != "reference" else None
    decision = explain_dispatch(x, backend=backend, native_available=native is not None)
    if decision.backend == "error":
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME,
                backend,
                "error",
                decision.device,
                decision.reason,
                native is not None,
            )
        )
        raise RuntimeError(decision.reason)
    if decision.backend == "reference" or x.numel() == 0 or y.numel() == 0:
        ptr_x, ptr_y = _pointers(batch_x, batch_y, x.shape[0], y.shape[0], x.device)
        selected = decision.backend
        reason = decision.reason
        if selected == "musa":
            reason = "MUSA backend selected; empty input, no kernel launch"
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME, backend, selected, decision.device, reason, native is not None
            )
        )
        return _reference(x, y, ptr_x, ptr_y, radius_value, max_num_neighbors)
    if x.shape[0] > 2**31 - 1 or y.shape[0] > 2**31 - 1 or max_num_neighbors > 2**31 - 1:
        message = "radius native path requires point counts within int32 indexing range"
        if backend == "musa":
            record_dispatch_report(
                DispatchReport(OPERATOR_NAME, backend, "error", decision.device, message, True)
            )
            raise ValueError(message)
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME, backend, "reference", decision.device, message, True, True, message
            )
        )
        ptr_x, ptr_y = _pointers(batch_x, batch_y, x.shape[0], y.shape[0], x.device)
        return _reference(x, y, ptr_x, ptr_y, radius_value, max_num_neighbors)
    capacity = y.shape[0] * max_num_neighbors
    if capacity > 2**31 - 1:
        message = "radius native path requires output capacity within int32 indexing range"
        if backend == "musa":
            record_dispatch_report(
                DispatchReport(OPERATOR_NAME, backend, "error", decision.device, message, True)
            )
            raise ValueError(message)
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME, backend, "reference", decision.device, message, True, True, message
            )
        )
        ptr_x, ptr_y = _pointers(batch_x, batch_y, x.shape[0], y.shape[0], x.device)
        return _reference(x, y, ptr_x, ptr_y, radius_value, max_num_neighbors)
    out_x = torch.empty(capacity, dtype=torch.int64, device=x.device)
    counts = torch.empty(y.shape[0], dtype=torch.int32, device=x.device)
    try:
        if batch_x is None:
            ptr_x, ptr_y = _pointers(None, None, x.shape[0], y.shape[0], x.device)
            native.radius_forward(
                x, y, ptr_x, ptr_y, out_x, counts, radius_value, max_num_neighbors
            )
        else:
            native_batch_x = batch_x if batch_x.dtype == torch.int64 else batch_x.to(torch.int64)
            native_batch_y = batch_y if batch_y.dtype == torch.int64 else batch_y.to(torch.int64)
            native.radius_batch_x_forward(
                x,
                y,
                native_batch_x,
                native_batch_y,
                out_x,
                counts,
                radius_value,
                max_num_neighbors,
            )
        inclusive_counts = torch.cumsum(counts, dim=0, dtype=torch.int64)
        edge_count = int(inclusive_counts[-1].item())
        result = torch.empty((2, edge_count), dtype=torch.int64, device=x.device)
        if edge_count:
            native.radius_compact_forward(
                out_x,
                counts,
                inclusive_counts,
                result,
                max_num_neighbors,
            )
    except RuntimeError as exc:
        if backend != "auto":
            record_dispatch_report(
                DispatchReport(OPERATOR_NAME, backend, "error", decision.device, str(exc), True)
            )
            raise
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME,
                backend,
                "reference",
                decision.device,
                "native backend failed; falling back",
                True,
                True,
                str(exc),
            )
        )
        ptr_x, ptr_y = _pointers(batch_x, batch_y, x.shape[0], y.shape[0], x.device)
        return _reference(x, y, ptr_x, ptr_y, radius_value, max_num_neighbors)
    record_dispatch_report(
        DispatchReport(OPERATOR_NAME, backend, "musa", decision.device, decision.reason, True)
    )
    return result


def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
    *,
    backend: BackendName = "auto",
) -> torch.Tensor:
    """Build a radius graph with the same edge direction as torch-cluster."""
    if flow not in {"source_to_target", "target_to_source"}:
        raise ValueError("flow must be 'source_to_target' or 'target_to_source'")
    if not isinstance(loop, bool):
        raise TypeError("loop must be a bool")
    try:
        max_num_neighbors = operator.index(max_num_neighbors)
    except TypeError as exc:
        raise TypeError("max_num_neighbors must be an integer") from exc
    if max_num_neighbors <= 0:
        raise ValueError("max_num_neighbors must be a positive integer")
    x, _ = _validate_points(x, x)
    effective = max_num_neighbors if loop else max_num_neighbors + 1
    try:
        edges = radius(x, x, r, batch, batch, effective, num_workers, backend=backend)
    except RuntimeError:
        report = get_last_dispatch_report()
        if report is not None:
            record_dispatch_report(
                DispatchReport(
                    GRAPH_OPERATOR_NAME,
                    report.requested_backend,
                    report.selected_backend,
                    report.device,
                    report.reason,
                    report.native_available,
                    report.native_fallback,
                    report.fallback_reason,
                )
            )
        raise
    report = get_last_dispatch_report()
    if report is not None:
        record_dispatch_report(
            DispatchReport(
                GRAPH_OPERATOR_NAME,
                report.requested_backend,
                report.selected_backend,
                report.device,
                report.reason,
                report.native_available,
                report.native_fallback,
                report.fallback_reason,
            )
        )
    if flow == "source_to_target":
        row, col = edges[1], edges[0]
    else:
        row, col = edges[0], edges[1]
    if not loop:
        keep = row != col
        row, col = row[keep], col[keep]
    return torch.stack([row, col], dim=0)


def radius_reference(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
) -> torch.Tensor:
    """Explicit PyTorch reference entry point for :func:`radius`."""
    return radius(x, y, r, batch_x, batch_y, max_num_neighbors, num_workers, backend="reference")


def radius_graph_reference(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
) -> torch.Tensor:
    """Explicit PyTorch reference entry point for :func:`radius_graph`."""
    return radius_graph(
        x, r, batch, loop, max_num_neighbors, flow, num_workers, backend="reference"
    )


__all__ = ["radius", "radius_graph", "radius_reference", "radius_graph_reference"]
