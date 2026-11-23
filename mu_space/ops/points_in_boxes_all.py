"""Point-in-rotated-3D-box operators for BEVFusion and mmdet3d call sites.

``points_in_boxes_all`` takes ``boxes`` first and returns the full ``[B, N, M]``
mask. The adapters below keep the upstream ``(points, boxes)`` argument order
and return contracts: the first-hit ``[B, N]`` index, the ``[B, N, M]`` mask,
and the unbatched ``[M, npoints]`` CPU mask.

Local ``x``/``y`` bounds are strict and the ``z`` bound is inclusive, matching
the upstream kernels.
"""

from __future__ import annotations

import math

import torch

from ..runtime.kernel_dispatch import (
    BackendName,
    DispatchDecision,
    DispatchReport,
    explain_dispatch,
    record_dispatch_report,
)
from ..runtime.native import NativeModuleUnavailable, native_module_proxy

OPERATOR_NAME = "points_in_boxes_all"
GPU_OPERATOR_NAME = "points_in_boxes_gpu"
BATCH_OPERATOR_NAME = "points_in_boxes_batch"
CPU_OPERATOR_NAME = "points_in_boxes_cpu"

_BACKGROUND_INDEX = -1


def _native_extension():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _validate(boxes: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(boxes, torch.Tensor) or not isinstance(points, torch.Tensor):
        raise TypeError("boxes and points must be torch.Tensor")
    if boxes.dtype != torch.float32 or points.dtype != torch.float32:
        raise TypeError("boxes and points must use float32")
    if boxes.ndim != 3 or boxes.shape[-1] != 7:
        raise ValueError("boxes must have shape [B, M, 7]")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, N, 3]")
    if boxes.shape[0] != points.shape[0]:
        raise ValueError("boxes and points must have the same batch size")
    if boxes.device != points.device:
        raise ValueError("boxes and points must be on the same device")
    return boxes.contiguous(), points.contiguous()


def _validate_unbatched(
    points: torch.Tensor, boxes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(boxes, torch.Tensor) or not isinstance(points, torch.Tensor):
        raise TypeError("boxes and points must be torch.Tensor")
    if boxes.dtype != torch.float32 or points.dtype != torch.float32:
        raise TypeError("boxes and points must use float32")
    if boxes.ndim != 2 or boxes.shape[-1] != 7:
        raise ValueError("boxes must have shape [M, 7]")
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if boxes.device != points.device:
        raise ValueError("boxes and points must be on the same device")
    return points.contiguous(), boxes.contiguous()


def points_in_boxes_all_reference(boxes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    boxes, points = _validate(boxes, points)
    b, m, _ = boxes.shape
    n = points.shape[1]
    if b == 0 or n == 0 or m == 0:
        return torch.zeros((b, n, m), dtype=torch.int32, device=boxes.device)
    # CUDA/OpenMMLab convention: z is the bottom center; x/y are strict.
    center = boxes[:, None, :, :3]
    size = boxes[:, None, :, 3:6]
    angle = boxes[:, None, :, 6]
    delta = points[:, :, None, :] - center
    cz = delta[..., 2] - size[..., 2] * 0.5
    rot_angle = angle + math.pi / 2.0
    cosa, sina = torch.cos(rot_angle), torch.sin(rot_angle)
    local_x = delta[..., 0] * cosa - delta[..., 1] * sina
    local_y = delta[..., 0] * sina + delta[..., 1] * cosa
    inside = (
        (cz.abs() <= size[..., 2] * 0.5)
        & (local_x > -size[..., 1] * 0.5)
        & (local_x < size[..., 1] * 0.5)
        & (local_y > -size[..., 0] * 0.5)
        & (local_y < size[..., 0] * 0.5)
    )
    return inside.to(torch.int32)


def points_in_boxes_batch_reference(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Upstream ``points_in_boxes_batch(points, boxes)`` reference: ``[B, N, M]`` mask."""
    boxes, points = _validate(boxes, points)
    return points_in_boxes_all_reference(boxes, points)


def _first_hit_index(mask: torch.Tensor) -> torch.Tensor:
    """Reduce a ``[B, N, M]`` mask to the upstream ``[B, N]`` first-hit index."""
    if mask.shape[2] == 0:
        return torch.full(mask.shape[:2], _BACKGROUND_INDEX, dtype=torch.int32, device=mask.device)
    # ``argmax`` on a 0/1 mask yields the first hit; points outside every box are
    # overwritten with the upstream background index.
    hits = mask.to(torch.float32).argmax(dim=2).to(torch.int32)
    return torch.where(mask.any(dim=2), hits, torch.full_like(hits, _BACKGROUND_INDEX))


def points_in_boxes_gpu_reference(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Upstream ``points_in_boxes_gpu(points, boxes)`` reference: ``[B, N]`` index."""
    return _first_hit_index(points_in_boxes_batch_reference(points, boxes))


def points_in_boxes_cpu_reference(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Upstream ``points_in_boxes_cpu(points, boxes)`` reference: ``[M, npoints]`` mask."""
    points, boxes = _validate_unbatched(points, boxes)
    mask = points_in_boxes_all_reference(boxes.unsqueeze(0), points.unsqueeze(0))
    return mask[0].transpose(0, 1).contiguous()


def _run_mask(
    boxes: torch.Tensor, points: torch.Tensor, backend: BackendName
) -> tuple[torch.Tensor | None, DispatchDecision, object | None, str | None]:
    """Run the mask kernel without recording a dispatch report."""
    native = (
        None
        if backend == "reference" or boxes.device.type not in {"musa", "privateuseone"}
        else _native_extension()
    )
    decision = explain_dispatch(boxes, backend=backend, native_available=native is not None)
    if decision.backend == "error":
        return None, decision, native, None
    if decision.backend == "reference" or boxes.numel() == 0 or points.numel() == 0:
        return points_in_boxes_all_reference(boxes, points), decision, native, None
    output = torch.empty(
        (boxes.shape[0], points.shape[1], boxes.shape[1]), dtype=torch.int32, device=boxes.device
    )
    try:
        native.points_in_boxes_all_forward(boxes, points, output)
    except RuntimeError as exc:
        if backend != "auto":
            return None, decision, native, str(exc)
        fallback = DispatchDecision(
            "reference",
            decision.device,
            "native backend failed; falling back to PyTorch reference",
        )
        return points_in_boxes_all_reference(boxes, points), fallback, native, str(exc)
    return output, decision, native, None


def _record(
    operator: str,
    backend: BackendName,
    decision: DispatchDecision,
    native: object | None,
    failure: str | None = None,
) -> None:
    record_dispatch_report(
        DispatchReport(
            operator,
            backend,
            "error" if failure is not None else decision.backend,
            decision.device,
            failure or decision.reason,
            native is not None,
            failure is not None,
            failure,
        )
    )


def _check_failure(backend: BackendName, decision: DispatchDecision, failure: str | None) -> None:
    if decision.backend == "error" or (failure is not None and backend != "auto"):
        raise RuntimeError(failure or decision.reason)


def points_in_boxes_all(
    boxes: torch.Tensor, points: torch.Tensor, *, backend: BackendName = "auto"
) -> torch.Tensor:
    """Mask every ``[B, N, 3]`` point against every ``[B, M, 7]`` box."""
    boxes, points = _validate(boxes, points)
    mask, decision, native, failure = _run_mask(boxes, points, backend)
    _record(OPERATOR_NAME, backend, decision, native, failure)
    _check_failure(backend, decision, failure)
    return mask


def points_in_boxes_batch(
    points: torch.Tensor, boxes: torch.Tensor, *, backend: BackendName = "auto"
) -> torch.Tensor:
    """Upstream ``points_in_boxes_batch``: ``(points, boxes)`` in, ``[B, N, M]`` mask out."""
    boxes, points = _validate(boxes, points)
    mask, decision, native, failure = _run_mask(boxes, points, backend)
    _record(BATCH_OPERATOR_NAME, backend, decision, native, failure)
    _check_failure(backend, decision, failure)
    return mask


def points_in_boxes_gpu(
    points: torch.Tensor, boxes: torch.Tensor, *, backend: BackendName = "auto"
) -> torch.Tensor:
    """Upstream ``points_in_boxes_gpu``: ``(points, boxes)`` in, ``[B, N]`` index out.

    The index is the first containing box; points outside every box keep ``-1``.
    """
    boxes, points = _validate(boxes, points)
    mask, decision, native, failure = _run_mask(boxes, points, backend)
    _record(GPU_OPERATOR_NAME, backend, decision, native, failure)
    _check_failure(backend, decision, failure)
    return _first_hit_index(mask)


def points_in_boxes_cpu(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Upstream ``points_in_boxes_cpu``: unbatched ``(points, boxes)`` -> ``[M, npoints]``.

    The upstream CPU entry point has no batch dimension and transposes its mask;
    it stays reference-only.
    """
    points, boxes = _validate_unbatched(points, boxes)
    mask = points_in_boxes_cpu_reference(points, boxes)
    record_dispatch_report(
        DispatchReport(
            CPU_OPERATOR_NAME,
            "reference",
            "reference",
            points.device.type,
            "unbatched CPU point-in-box contract; reference implementation",
            False,
        )
    )
    return mask


__all__ = [
    "points_in_boxes_all",
    "points_in_boxes_all_reference",
    "points_in_boxes_batch",
    "points_in_boxes_batch_reference",
    "points_in_boxes_cpu",
    "points_in_boxes_cpu_reference",
    "points_in_boxes_gpu",
    "points_in_boxes_gpu_reference",
]
