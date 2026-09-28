"""Backend selection helpers for canonical kernels."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Literal

import torch

BackendName = Literal["auto", "musa", "reference"]
SelectedBackend = Literal["musa", "reference", "error"]


@dataclass(frozen=True)
class DispatchDecision:
    backend: SelectedBackend
    device: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DispatchReport:
    operator: str
    requested_backend: str
    selected_backend: SelectedBackend
    device: str
    reason: str
    native_available: bool
    native_fallback: bool = False
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_last_dispatch_report: DispatchReport | None = None
_dispatch_reports: list[DispatchReport] = []


def record_dispatch_report(report: DispatchReport) -> DispatchReport:
    global _last_dispatch_report
    _last_dispatch_report = report
    _dispatch_reports.append(report)
    if os.environ.get("MUSPACE_DISPATCH_LOG", "").strip().lower() in {"1", "true", "yes", "on"}:
        print("muSPACE dispatch: " + json.dumps(report.as_dict(), sort_keys=True), file=sys.stderr)
    return report


def get_last_dispatch_report() -> DispatchReport | None:
    return _last_dispatch_report


def get_dispatch_reports() -> tuple[DispatchReport, ...]:
    return tuple(_dispatch_reports)


def clear_dispatch_report() -> None:
    global _last_dispatch_report, _dispatch_reports
    _last_dispatch_report = None
    _dispatch_reports = []


def _is_musa_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type in {"musa", "privateuseone"} or bool(
        getattr(tensor, "is_musa", False)
    )


def explain_dispatch(
    value: torch.Tensor,
    *,
    backend: BackendName = "auto",
    native_available: bool = True,
) -> DispatchDecision:
    if backend not in {"auto", "musa", "reference"}:
        return DispatchDecision("error", value.device.type, f"unknown backend: {backend}")
    device = value.device.type
    if backend == "reference":
        return DispatchDecision("reference", device, "reference backend requested")

    if backend == "musa":
        if not _is_musa_tensor(value):
            return DispatchDecision("error", device, "backend=musa requires a MUSA tensor")
        if not native_available:
            return DispatchDecision("error", device, "muSPACE AOT native module is unavailable")
        return DispatchDecision("musa", device, "forced MUSA backend")

    if _is_musa_tensor(value):
        if not native_available:
            return DispatchDecision(
                "reference",
                device,
                "MUSA backend is unavailable; falling back to PyTorch reference",
            )
        return DispatchDecision("musa", device, "MUSA tensor selects native backend")

    return DispatchDecision("reference", device, "non-MUSA tensors use the PyTorch reference")
