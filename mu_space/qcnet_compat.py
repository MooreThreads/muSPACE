"""Device-aware torch-cluster compatibility functions for the QCNet recipe."""

from __future__ import annotations

from typing import Optional

import torch

from .ops.radius import radius as _muspace_radius
from .ops.radius import radius_graph as _muspace_radius_graph


def _is_musa(value: torch.Tensor) -> bool:
    return value.device.type in {"musa", "privateuseone"} or bool(getattr(value, "is_musa", False))


def radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    **kwargs,
):
    if _is_musa(x):
        return _muspace_radius(
            x,
            y,
            r,
            batch_x,
            batch_y,
            max_num_neighbors,
            num_workers,
            backend=kwargs.pop("backend", "musa"),
        )
    from torch_cluster import radius as upstream_radius

    return upstream_radius(x, y, r, batch_x, batch_y, max_num_neighbors, num_workers, **kwargs)


def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
    **kwargs,
):
    if _is_musa(x):
        return _muspace_radius_graph(
            x,
            r,
            batch,
            loop,
            max_num_neighbors,
            flow,
            num_workers,
            backend=kwargs.pop("backend", "musa"),
        )
    from torch_cluster import radius_graph as upstream_radius_graph

    return upstream_radius_graph(x, r, batch, loop, max_num_neighbors, flow, num_workers, **kwargs)


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    """Small torch-only fallback for the CSR reductions used by QCNet."""
    if src.device.type not in {"musa", "privateuseone"}:
        from torch_scatter import segment_csr as upstream_segment_csr

        return upstream_segment_csr(src, indptr, reduce=reduce)
    if reduce not in {"sum", "mean"}:
        raise ValueError(f"unsupported QCNet CSR reduction: {reduce}")
    values = []
    for start, end in zip(indptr[:-1].tolist(), indptr[1:].tolist()):
        part = src[int(start) : int(end)]
        value = part.sum(dim=0)
        if reduce == "mean" and end > start:
            value = value / (int(end) - int(start))
        values.append(value)
    return torch.stack(values, dim=0) if values else src.new_empty((0,) + src.shape[1:])


def gather_csr(src: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
    if src.device.type not in {"musa", "privateuseone"}:
        from torch_scatter import gather_csr as upstream_gather_csr

        return upstream_gather_csr(src, indptr)
    return torch.repeat_interleave(src, (indptr[1:] - indptr[:-1]).to(torch.long), dim=0)


__all__ = ["radius", "radius_graph", "segment_csr", "gather_csr"]
