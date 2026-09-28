"""Target-stack compatibility checks with explicit support states."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import MethodType
from typing import Literal

import torch
from packaging.version import Version

SupportState = Literal["validated", "experimental", "unsupported", "unknown"]
CompatibilityProfile = Literal["core", "bevformer"]

BEVFORMER_EXPECTED = {
    "torch": "2.9.1",
    "torch_musa": "2.9.1",
    "mmcv-full": "1.4.0",
    "mmdet": "2.14.0",
    "mmsegmentation": "0.14.1",
    "mmdet3d": "0.17.1",
}
SUPPORTED_CORE_STACKS: dict[str, tuple[dict[str, str], ...]] = {
    "4.3.8": (
        {"torch": "2.7.1", "torch_musa": "2.7.1"},
    ),
    "5.2.0": (
        {"torch": "2.9.1", "torch_musa": "2.9.1"},
        {"torch": "2.11.0", "torch_musa": "2.11.0"},
    ),
}


@dataclass(frozen=True)
class CompatibilityReport:
    profile: CompatibilityProfile
    state: SupportState
    reasons: tuple[str, ...]
    versions: dict[str, str | None]
    musa_sdk: str | None = None
    musa_available: bool | None = None
    expected_stack: dict[str, str] | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _detect_musa_sdk_version() -> str | None:
    """Detect the installed SDK, with an explicit CI/test override.

    MUSA installations differ between development images, so the override is
    the stable interface for CI.  The lightweight filesystem and ``mcc``
    probes make the command useful in standard SDK images without importing
    any vendor-specific Python package.
    """
    override = os.environ.get("MUSPACE_MUSA_SDK_VERSION")
    if override:
        return override.strip()

    roots = [os.environ.get("MUSA_HOME"), os.environ.get("MUSA_PATH")]
    for root in filter(None, roots):
        root_path = Path(root)
        for name in ("version.txt", "VERSION", "version"):
            candidate = root_path / name
            try:
                text = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            match = re.search(r"\b(\d+\.\d+\.\d+)\b", text)
            if match:
                return match.group(1)

    mcc = shutil.which("mcc")
    if mcc:
        try:
            output = subprocess.check_output(
                [mcc, "--version"], stderr=subprocess.STDOUT, text=True, timeout=2
            )
        except (OSError, subprocess.SubprocessError):
            output = ""
        match = re.search(r"\bmcc\s+version\s+(\d+\.\d+\.\d+)\b", output, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _musa_available() -> bool:
    try:
        return bool(hasattr(torch, "musa") and torch.musa.is_available())
    except (AttributeError, RuntimeError):
        return False


def _normalized_version(value: str) -> Version:
    parsed = Version(value.split("+")[0])
    return Version(parsed.base_version)


def _stack_pairs_text(rows: tuple[dict[str, str], ...]) -> str:
    return " or ".join(f"{row['torch']}/{row['torch_musa']}" for row in rows)


def _matching_core_stack(sdk: str, versions: dict[str, str | None]) -> dict[str, str] | None:
    allowed = SUPPORTED_CORE_STACKS.get(sdk)
    if allowed is None:
        return None
    for row in allowed:
        if all(
            _normalized_version(versions[name]) == Version(value)  # type: ignore[arg-type]
            for name, value in row.items()
        ):
            return row
    return None


def _bevformer_stack_for_sdk(sdk: str) -> dict[str, str] | None:
    allowed = SUPPORTED_CORE_STACKS.get(sdk)
    if allowed is None:
        return None
    for row in allowed:
        if (
            row["torch"] == BEVFORMER_EXPECTED["torch"]
            and row["torch_musa"] == BEVFORMER_EXPECTED["torch_musa"]
        ):
            return row
    return None


def compatibility_report(
    profile: CompatibilityProfile = "core",
    *,
    sdk_version: str | None = None,
    musa_available: bool | None = None,
) -> CompatibilityReport:
    """Report core ABI compatibility or the pinned BEVFormer recipe profile."""
    if profile not in {"core", "bevformer"}:
        raise ValueError("profile must be 'core' or 'bevformer'")
    expected = {"torch": "", "torch_musa": ""} if profile == "core" else BEVFORMER_EXPECTED
    versions = {name: _version(name) for name in expected}
    sdk = sdk_version if sdk_version is not None else _detect_musa_sdk_version()
    available = _musa_available() if musa_available is None else musa_available
    missing = [name for name, version in versions.items() if version is None]
    if missing:
        return CompatibilityReport(
            profile,
            "unsupported",
            ("missing required distributions: " + ", ".join(missing),),
            versions,
            musa_sdk=sdk,
            musa_available=available,
        )

    if profile == "core":
        torch_version = _normalized_version(versions["torch"])  # type: ignore[arg-type]
        musa_version = _normalized_version(versions["torch_musa"])  # type: ignore[arg-type]
        if torch_version != musa_version:
            return CompatibilityReport(
                profile,
                "unsupported",
                ("torch and torch_musa must use the same version",),
                versions,
                musa_sdk=sdk,
                musa_available=available,
            )
        if sdk is None:
            return CompatibilityReport(
                profile,
                "unknown",
                ("MUSA SDK version could not be detected",),
                versions,
                musa_sdk=sdk,
                musa_available=available,
            )
        allowed = SUPPORTED_CORE_STACKS.get(sdk)
        if allowed is None:
            return CompatibilityReport(
                profile,
                "unsupported",
                (f"MUSA SDK {sdk} is not in the supported stack matrix",),
                versions,
                musa_sdk=sdk,
                musa_available=available,
            )
        expected_stack = _matching_core_stack(sdk, versions)
        if expected_stack is None:
            return CompatibilityReport(
                profile,
                "unsupported",
                (
                    f"MUSA SDK {sdk} requires torch/torch_musa "
                    f"{_stack_pairs_text(allowed)}",
                ),
                versions,
                musa_sdk=sdk,
                musa_available=available,
            )
        if not available:
            return CompatibilityReport(
                profile,
                "unsupported",
                ("MUSA device is not available",),
                versions,
                musa_sdk=sdk,
                musa_available=available,
                expected_stack=expected_stack,
            )
        return CompatibilityReport(
            profile,
            "validated",
            ("validated MUSA SDK/PyTorch/torch_musa stack",),
            versions,
            musa_sdk=sdk,
            musa_available=available,
            expected_stack=expected_stack,
        )

    mismatches = []
    for name, expected_version in expected.items():
        actual = versions[name]
        assert actual is not None
        if _normalized_version(actual) != Version(expected_version):
            mismatches.append(f"{name}={actual}, expected {expected_version}")

    if mismatches:
        return CompatibilityReport(
            profile,
            "experimental",
            tuple(mismatches),
            versions,
            musa_sdk=sdk,
            musa_available=available,
        )
    if sdk is None:
        return CompatibilityReport(
            profile,
            "unknown",
            ("MUSA SDK version could not be detected",),
            versions,
            musa_sdk=sdk,
            musa_available=available,
        )
    expected_core = _bevformer_stack_for_sdk(sdk)
    if expected_core is None:
        return CompatibilityReport(
            profile,
            "unsupported",
            (f"BEVFormer requires the SDK 5.2.0 / torch 2.9.1 stack, got SDK {sdk}",),
            versions,
            musa_sdk=sdk,
            musa_available=available,
        )
    if not available:
        return CompatibilityReport(
            profile,
            "unsupported",
            ("MUSA device is not available",),
            versions,
            musa_sdk=sdk,
            musa_available=available,
            expected_stack=expected_core,
        )
    return CompatibilityReport(
        profile,
        "validated",
        ("exact BEVFormer compatibility stack",),
        versions,
        musa_sdk=sdk,
        musa_available=available,
        expected_stack=expected_core,
    )


def _eval_batch_norm_reference(module: torch.nn.BatchNorm2d, inputs: torch.Tensor) -> torch.Tensor:
    if module.training or module.running_mean is None or module.running_var is None:
        return module._muspace_native_forward(inputs)
    scale = torch.rsqrt(module.running_var + module.eps)
    if module.weight is not None:
        scale = scale * module.weight
    offset = -module.running_mean * scale
    if module.bias is not None:
        offset = offset + module.bias
    scale = scale.to(dtype=inputs.dtype)
    offset = offset.to(dtype=inputs.dtype)
    return inputs * scale[None, :, None, None] + offset[None, :, None, None]


def enable_eval_batch_norm_reference(root: torch.nn.Module) -> int:
    """Replace eval BatchNorm2d with its explicit affine equivalent.

    The target torch_musa stack can leave parts of the input-gradient
    output unwritten in the native frozen BatchNorm backward kernel. This
    opt-in replacement preserves eval-mode semantics without that kernel.
    """
    replaced = 0
    for module in root.modules():
        if not isinstance(module, torch.nn.BatchNorm2d) or module.training:
            continue
        if getattr(module, "_muspace_eval_bn_reference", False):
            continue
        module._muspace_native_forward = module.forward
        module.forward = MethodType(_eval_batch_norm_reference, module)
        module._muspace_eval_bn_reference = True
        replaced += 1
    return replaced
