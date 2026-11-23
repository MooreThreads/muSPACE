"""BEV temporal alignment: per-sample gather of the previous BEV feature plane."""

from __future__ import annotations

from functools import lru_cache

import math
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

from ..runtime.kernel_dispatch import DispatchReport, explain_dispatch, record_dispatch_report
from ..runtime.native import NativeModuleUnavailable, native_module_proxy

OPERATOR_NAME = "temporal_align"
BackendName = Literal["auto", "musa", "reference"]
_MODES = {"bilinear": 0, "nearest": 1}
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _native():
    try:
        return native_module_proxy()
    except NativeModuleUnavailable:
        return None


def _empty_ref(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(0, dtype=dtype, device=tensor.device)


def _prepare_trig(bev: torch.Tensor, angle: torch.Tensor | None) -> torch.Tensor:
    """Return the per-sample ``(cosine, sine)`` table the native kernels read.

    The table is built once per call by a one-block kernel that evaluates the
    trigonometry in double precision, which is what the reference does with Python
    scalars.  Keeping that fp64 code out of the sampling kernels matters: inlined
    there it reduced occupancy and cost about 3x on forward and 1.5x on backward.
    """
    ext = _native()
    if ext is None:
        raise RuntimeError("muSPACE AOT native module is unavailable")
    trig = torch.empty((bev.shape[0], 2), dtype=torch.float32, device=bev.device)
    ext.temporal_align_prepare_trig(
        _empty_ref(bev, torch.float32) if angle is None else angle.contiguous(), trig
    )
    return trig


def _resolve_plane(bev: torch.Tensor, bev_h: int | None, bev_w: int | None) -> tuple[int, int, int]:
    """Return ``(batch, bev_h, bev_w)`` after checking the accepted layouts."""
    if bev.dim() == 3:
        if bev_h is None or bev_w is None:
            raise ValueError("bev_h and bev_w are required for a 3-D bev tensor")
        batch = bev.shape[0]
        if bev_h * bev_w != bev.shape[1]:
            raise ValueError(
                f"bev_h*bev_w ({bev_h * bev_w}) must equal the bev plane size ({bev.shape[1]})"
            )
        return batch, int(bev_h), int(bev_w)
    if bev.dim() == 4:
        batch, _, height, width = bev.shape
        if bev_h is not None and bev_h != height:
            raise ValueError(f"bev_h ({bev_h}) must match the bev height ({height})")
        if bev_w is not None and bev_w != width:
            raise ValueError(f"bev_w ({bev_w}) must match the bev width ({width})")
        return batch, height, width
    raise ValueError(
        "bev must be 3-D (batch, bev_h*bev_w, channels) or 4-D (batch, channels, height, width)"
    )


def _validate(
    bev: torch.Tensor,
    angle: torch.Tensor | None,
    shift: torch.Tensor | None,
    bev_h: int | None,
    bev_w: int | None,
    center: Sequence[float] | None,
    mode: str,
):
    if not isinstance(bev, torch.Tensor):
        raise TypeError("bev must be a torch.Tensor")
    batch, height, width = _resolve_plane(bev, bev_h, bev_w)
    if bev.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("bev must use float32, float16 or bfloat16")
    channels = bev.shape[2] if bev.dim() == 3 else bev.shape[1]
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("bev channels and spatial dimensions must be positive")
    if mode not in _MODES:
        raise ValueError("mode must be one of: bilinear, nearest")
    if angle is not None:
        if not isinstance(angle, torch.Tensor):
            raise TypeError("angle must be a torch.Tensor")
        if angle.dtype != torch.float32:
            raise TypeError("angle must use float32")
        if angle.dim() != 1 or angle.shape[0] != batch:
            raise ValueError("angle must be a 1-D tensor with one value per batch sample")
        if angle.device != bev.device:
            raise ValueError("angle must be on the same device as bev")
        if angle.requires_grad:
            raise ValueError("angle and shift are non-trainable inputs")
        if angle.device.type == "cpu" and not bool(torch.isfinite(angle).all()):
            # Only scanned on the host: a finiteness reduction on a device tensor
            # costs a device synchronization plus a tiny kernel per call, which is
            # about as expensive as the whole forward pass.  The native path
            # documents non-finite geometry as a NaN-propagating input instead.
            raise ValueError("angle must contain only finite values")
    if shift is not None:
        if not isinstance(shift, torch.Tensor):
            raise TypeError("shift must be a torch.Tensor")
        if shift.dtype not in (torch.int32, torch.float32):
            raise TypeError("shift must use int32 or float32")
        if shift.dim() != 2 or shift.shape != (batch, 2):
            raise ValueError("shift must be a (batch, 2) tensor")
        if shift.device != bev.device:
            raise ValueError("shift must be on the same device as bev")
        if shift.requires_grad:
            raise ValueError("angle and shift are non-trainable inputs")
        if shift.dtype == torch.int32 and angle is not None:
            raise ValueError("an integer shift cannot be combined with an angle")
        if shift.dtype == torch.float32 and shift.device.type == "cpu" and not bool(
            torch.isfinite(shift).all()
        ):
            raise ValueError("a float shift must contain only finite values")
    if center is not None:
        if len(center) != 2:
            raise ValueError("center must be a pair of finite coordinates")
        if not all(math.isfinite(float(value)) for value in center):
            raise ValueError("center must be a pair of finite coordinates")


def _centre_pixels(center: Sequence[float] | None, width: int, height: int) -> tuple[float, float]:
    """Return the rotation centre in pixel-index coordinates.

    ``center`` follows ``torchvision.transforms.functional.rotate``: a corner
    based pixel coordinate, so ``(width / 2, height / 2)`` is the geometric
    centre. ``None`` selects that geometric centre of the plane.
    """
    if center is None:
        return (width - 1) * 0.5, (height - 1) * 0.5
    return float(center[0]) - 0.5, float(center[1]) - 0.5


def _channels_last(bev: torch.Tensor, bev_h: int, bev_w: int) -> torch.Tensor:
    """Return a contiguous ``(batch, bev_h*bev_w, channels)`` view of ``bev``."""
    if bev.dim() == 3:
        return bev.contiguous()
    return bev.permute(0, 2, 3, 1).contiguous().view(bev.shape[0], bev_h * bev_w, bev.shape[1])


#: Pixels a planar block spans; the kernel launch grid is derived from it.
_PLANAR_SPAN = 32
#: Grid dimension limit shared with the kernel launch configuration.
_GRID_LIMIT = 65535


@lru_cache(maxsize=1)
def _planar_ready() -> bool:
    """True when the loaded native library exposes the channels-first entries.

    An older library answers a missing symbol by raising rather than by an
    ``AttributeError``, so the probe has to catch broadly and remember the answer.
    """
    ext = _native()
    if ext is None:
        return False
    try:
        return ext.temporal_align_forward_planar is not None
    except Exception:
        return False


def _planar_eligible(bev: torch.Tensor, shift: torch.Tensor | None) -> bool:
    """True when the native channels-first kernels can serve ``bev`` unchanged.

    The planar path is the same sampler reading the caller's layout, so it is
    preferred whenever it applies: it skips the two full-plane layout conversions
    the packed path needs. Anything it cannot serve falls back to the packed
    path, which covers every dtype and layout.
    """
    if bev.dim() != 4 or bev.dtype != torch.float32 or not bev.is_contiguous():
        return False
    if shift is not None and shift.dtype == torch.int32:
        # An integer translation is an exact plane copy and keeps the packed path.
        return False
    if not _planar_ready():
        return False
    batch, channels, height, width = bev.shape
    if batch <= 0 or channels <= 0 or height <= 0 or width <= 0:
        return False
    return (
        height <= _GRID_LIMIT
        and batch <= _GRID_LIMIT
        and (width + _PLANAR_SPAN - 1) // _PLANAR_SPAN <= _GRID_LIMIT
    )


def _restore_layout(work: torch.Tensor, bev: torch.Tensor) -> torch.Tensor:
    if bev.dim() == 3:
        return work
    batch, _, height, width = bev.shape
    return work.view(batch, height, width, bev.shape[1]).permute(0, 3, 1, 2)


def _reference_plane(
    plane: torch.Tensor,
    angle: torch.Tensor | None,
    shift: torch.Tensor | None,
    sample: int,
    centre: tuple[float, float],
    mode: str,
) -> torch.Tensor:
    """Resample one ``(channels, height, width)`` plane with PyTorch itself."""
    channels, height, width = plane.shape
    cosine = 1.0
    sine = 0.0
    if angle is not None:
        degrees = float(angle[sample].item())
        if not math.isfinite(degrees):
            raise ValueError("angle must contain only finite values")
        radians = math.radians(degrees)
        cosine = math.cos(radians)
        sine = math.sin(radians)
    offset_x = 0.0
    offset_y = 0.0
    if shift is not None:
        offset_x = float(shift[sample, 0].item())
        offset_y = float(shift[sample, 1].item())
    centre_x, centre_y = centre
    xs = torch.arange(width, dtype=torch.float32, device=plane.device) - centre_x
    ys = torch.arange(height, dtype=torch.float32, device=plane.device) - centre_y
    source_x = cosine * xs[None, :] - sine * ys[:, None] + centre_x + offset_x
    source_y = sine * xs[None, :] + cosine * ys[:, None] + centre_y + offset_y
    grid = torch.stack(
        ((2.0 * source_x + 1.0 - width) / width, (2.0 * source_y + 1.0 - height) / height), dim=-1
    ).unsqueeze(0)
    return F.grid_sample(
        plane.unsqueeze(0), grid, mode=mode, padding_mode="zeros", align_corners=False
    )[0]


def _reference_shift(plane: torch.Tensor, offset: tuple[int, int]) -> torch.Tensor:
    """Integer translation reference: an exact slice copy with zero padding."""
    channels, height, width = plane.shape
    shift_x, shift_y = offset
    result = torch.zeros_like(plane)
    start_y = max(0, -shift_y)
    stop_y = min(height, height - shift_y)
    start_x = max(0, -shift_x)
    stop_x = min(width, width - shift_x)
    if stop_y > start_y and stop_x > start_x:
        result[:, start_y:stop_y, start_x:stop_x] = plane[
            :, start_y + shift_y : stop_y + shift_y, start_x + shift_x : stop_x + shift_x
        ]
    else:
        # A window that falls outside the plane is legal and its gradient is
        # zero, but a bare ``zeros_like`` detaches the output from ``plane`` and
        # the backward pass then fails with "does not require grad and does not
        # have a grad_fn".  Adding the sum of an empty slice keeps the graph at
        # exactly zero cost; ``plane.sum() * 0`` would turn an infinite input
        # into NaN.
        result = result + plane[:0].sum()
    return result


def temporal_align_reference(
    bev: torch.Tensor,
    angle: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    *,
    bev_h: int | None = None,
    bev_w: int | None = None,
    center: Sequence[float] | None = None,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Run the PyTorch reference implementation after muSPACE validation."""
    _validate(bev, angle, shift, bev_h, bev_w, center, mode)
    _, height, width = _resolve_plane(bev, bev_h, bev_w)
    centre = _centre_pixels(center, width, height)
    work = _channels_last(bev, height, width)
    planes = work.view(bev.shape[0], height, width, -1).permute(0, 3, 1, 2)
    if planes.dtype != torch.float32:
        # CPU grid sampling has no half or bfloat16 kernel on every supported
        # stack; run the reference in float32 and keep the public dtype.
        compute = planes.float()
    else:
        compute = planes
    integer = shift is not None and shift.dtype == torch.int32
    results = []
    for sample in range(bev.shape[0]):
        if integer:
            results.append(
                _reference_shift(
                    compute[sample],
                    (int(shift[sample, 0].item()), int(shift[sample, 1].item())),
                )
            )
        else:
            results.append(_reference_plane(compute[sample], angle, shift, sample, centre, mode))
    stacked = torch.stack(results, dim=0)
    restored = _restore_layout(
        stacked.to(bev.dtype).permute(0, 2, 3, 1).reshape(bev.shape[0], height * width, -1),
        bev,
    )
    return restored.contiguous() if bev.dim() == 4 else restored


class _NativeTemporalAlignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, work, trig, shift, bev_h, bev_w, centre_x, centre_y, mode):
        ext = _native()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        output = torch.empty_like(work)
        ext.temporal_align_forward(
            work, trig, shift, output, bev_h, bev_w, centre_x, centre_y, mode
        )
        ctx.save_for_backward(work, trig, shift)
        ctx.spec = (bev_h, bev_w, centre_x, centre_y, mode)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        work, trig, shift = ctx.saved_tensors
        bev_h, bev_w, centre_x, centre_y, mode = ctx.spec
        ext = _native()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        grad_bev = torch.empty(work.shape, dtype=torch.float32, device=work.device)
        ext.temporal_align_backward(
            work,
            trig,
            shift,
            grad_output.contiguous(),
            grad_bev,
            bev_h,
            bev_w,
            centre_x,
            centre_y,
            mode,
        )
        return grad_bev.to(work.dtype), None, None, None, None, None, None, None


class _NativeTemporalAlignPlanarFunction(torch.autograd.Function):
    """The same sampler reading and writing the caller's channels-first layout."""

    @staticmethod
    def forward(ctx, work, trig, shift, bev_h, bev_w, centre_x, centre_y, mode):
        ext = _native()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        output = torch.empty_like(work)
        ext.temporal_align_forward_planar(
            work, trig, shift, output, bev_h, bev_w, centre_x, centre_y, mode
        )
        ctx.save_for_backward(work, trig, shift)
        ctx.spec = (bev_h, bev_w, centre_x, centre_y, mode)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        work, trig, shift = ctx.saved_tensors
        bev_h, bev_w, centre_x, centre_y, mode = ctx.spec
        ext = _native()
        if ext is None:
            raise RuntimeError("muSPACE AOT native module is unavailable")
        grad_bev = torch.empty(work.shape, dtype=torch.float32, device=work.device)
        ext.temporal_align_backward_planar(
            work,
            trig,
            shift,
            grad_output.contiguous(),
            grad_bev,
            bev_h,
            bev_w,
            centre_x,
            centre_y,
            mode,
        )
        return grad_bev.to(work.dtype), None, None, None, None, None, None, None


def temporal_align(
    bev: torch.Tensor,
    angle: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    *,
    bev_h: int | None = None,
    bev_w: int | None = None,
    center: Sequence[float] | None = None,
    mode: str = "bilinear",
    backend: BackendName = "auto",
) -> torch.Tensor:
    """Gather a previous BEV plane into the current frame.

    ``bev`` is either ``(batch, bev_h*bev_w, channels)`` or ``(batch, channels,
    height, width)``. For an output pixel ``p`` and sample ``b`` the source pixel
    is ``R(angle[b]) * (p - c) + c + shift[b]``, where ``R`` is a
    counter-clockwise rotation, ``angle`` is in degrees with the
    ``torchvision.transforms.functional.rotate`` direction, and ``shift`` is a
    translation in source pixels. Sampling outside the plane returns zeros.

    ``center`` is a corner-based pixel coordinate like torchvision's
    ``rotate(center=...)``, so ``(width / 2, height / 2)`` is the geometric
    centre; ``None`` selects that geometric centre of the plane.

    An ``int32`` ``shift`` without an ``angle`` selects a pure integer
    translation that never interpolates.
    """
    _validate(bev, angle, shift, bev_h, bev_w, center, mode)
    batch, height, width = _resolve_plane(bev, bev_h, bev_w)
    centre_x, centre_y = _centre_pixels(center, width, height)
    native_available = _native() is not None
    decision = explain_dispatch(bev, backend=backend, native_available=native_available)
    if decision.backend == "error":
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME, backend, "error", decision.device, decision.reason, native_available
            )
        )
        raise RuntimeError(decision.reason)
    if decision.backend == "reference":
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME,
                backend,
                "reference",
                decision.device,
                decision.reason,
                native_available,
            )
        )
        return temporal_align_reference(
            bev, angle, shift, bev_h=height, bev_w=width, center=center, mode=mode
        )
    channels = bev.shape[1] if bev.dim() == 4 else bev.shape[2]
    if batch == 0 or channels == 0:
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME, backend, "musa", decision.device, decision.reason, native_available
            )
        )
        return torch.zeros_like(bev)
    planar = _planar_eligible(bev, shift)
    work = bev if planar else _channels_last(bev, height, width)
    shift_arg = _empty_ref(bev, torch.int32) if shift is None else shift.contiguous()
    kernel = _NativeTemporalAlignPlanarFunction if planar else _NativeTemporalAlignFunction
    try:
        trig = _prepare_trig(bev, angle)
        output = kernel.apply(
            work,
            trig,
            shift_arg,
            height,
            width,
            centre_x,
            centre_y,
            _MODES[mode],
        )
    except RuntimeError as exc:
        if backend != "auto":
            raise
        record_dispatch_report(
            DispatchReport(
                OPERATOR_NAME,
                backend,
                "reference",
                decision.device,
                "native backend failed; falling back to reference",
                native_available,
                True,
                str(exc),
            )
        )
        return temporal_align_reference(
            bev, angle, shift, bev_h=height, bev_w=width, center=center, mode=mode
        )
    record_dispatch_report(
        DispatchReport(
            OPERATOR_NAME,
            backend,
            "musa",
            decision.device,
            f"{decision.reason} ({'planar' if planar else 'packed'} kernel)",
            native_available,
        )
    )
    restored = output if planar else _restore_layout(output, bev)
    return restored.contiguous() if bev.dim() == 4 else restored


__all__ = ["temporal_align", "temporal_align_reference"]
