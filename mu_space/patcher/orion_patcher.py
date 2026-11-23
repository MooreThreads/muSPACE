"""Route Orion's vendored mmcv operators to muSPACE on MUSA.

Orion ships a vendored copy of mmcv whose operator modules call the compiled
``mmcv._ext`` / ``iou3d_cuda`` / ``roiaware_pool3d_ext`` extensions. This patcher
rebinds the entry points those modules expose to muSPACE equivalents, so the
model graph stays unchanged while the kernels run on MUSA:

* ``mmcv.ops.iou3d_det.iou3d_utils`` -- ``boxes_iou_bev``/``nms_gpu``/``nms_normal_gpu``
* ``mmcv.core.post_processing.{box3d_nms,merge_augs}`` -- the same NMS names
* ``mmcv.ops.roiaware_pool3d`` -- ``points_in_boxes_gpu``/``batch``/``cpu``
* ``mmcv.ops.sigmoid_focal_loss`` -- ``sigmoid_focal_loss``

Extension entry points without a muSPACE replacement stay as ``MissingExtension``
stand-ins and raise a descriptive error if the model reaches them, which keeps a
silently wrong result out of the inference path.

Install the flash-attn shim before importing the vendored mmcv; Orion's
``mmcv.models.utils.attention`` imports ``flash_attn`` at module import time.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from types import ModuleType

from .orion_flash_attn import install_flash_attn_shim
from .runtime import PatchError, _patch_attr, _restore_attrs

IOU3D_UTILS_MODULE = "mmcv.ops.iou3d_det.iou3d_utils"
IOU3D_DET_MODULE = "mmcv.ops.iou3d_det"
BOX3D_NMS_MODULE = "mmcv.core.post_processing.box3d_nms"
MERGE_AUGS_MODULE = "mmcv.core.post_processing.merge_augs"
POINTS_IN_BOXES_MODULE = "mmcv.ops.roiaware_pool3d.points_in_boxes"
ROIAWARE_MODULE = "mmcv.ops.roiaware_pool3d"
FOCAL_LOSS_MODULE = "mmcv.ops.focal_loss"
# Modules that pull the operators in with ``from ... import nms_gpu``; the bound
# name lives in their own namespace and has to be rebound there as well.
IMPORTING_MODULES = ("mmcv.models.dense_heads.orion_head",)

BACKENDS = ("auto", "musa", "reference")


@dataclass(frozen=True)
class OrionPatchReport:
    target: str
    backend: str
    action: str
    flash_attn_shim: bool
    patched_modules: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _make_iou3d_wrappers(backend: str):
    def boxes_iou_bev(boxes_a, boxes_b):
        from mu_space.ops import boxes_iou_bev as operator

        return operator(boxes_a, boxes_b, backend=backend)

    def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
        from mu_space.ops import nms_bev as operator

        return operator(
            boxes,
            scores,
            thresh,
            pre_maxsize=pre_maxsize,
            post_max_size=post_max_size,
            backend=backend,
        )

    def nms_normal_gpu(boxes, scores, thresh):
        from mu_space.ops import nms_bev_normal as operator

        return operator(boxes, scores, thresh, backend=backend)

    return boxes_iou_bev, nms_gpu, nms_normal_gpu


def _make_points_in_boxes_wrappers(backend: str):
    def points_in_boxes_gpu(points, boxes):
        from mu_space.ops import points_in_boxes_gpu as operator

        return operator(points, boxes, backend=backend)

    def points_in_boxes_batch(points, boxes):
        from mu_space.ops import points_in_boxes_batch as operator

        return operator(points, boxes, backend=backend)

    def points_in_boxes_cpu(points, boxes):
        from mu_space.ops import points_in_boxes_cpu as operator

        return operator(points, boxes)

    return points_in_boxes_gpu, points_in_boxes_batch, points_in_boxes_cpu


def _make_focal_loss_wrapper(backend: str):
    def sigmoid_focal_loss(
        inputs,
        targets,
        alpha=-1,
        gamma=2,
        reduction="mean",
    ):
        from mu_space.ops import sigmoid_focal_loss as operator

        return operator(
            inputs, targets, alpha=alpha, gamma=gamma, reduction=reduction, backend=backend
        )

    return sigmoid_focal_loss


class OrionPatcher:
    """Rebind Orion's vendored mmcv operator entry points to muSPACE."""

    def __init__(self) -> None:
        self._originals: dict[tuple[str, str], object] = {}
        self._backend: str | None = None

    def _module(self, name: str) -> ModuleType | None:
        try:
            return importlib.import_module(name)
        except ImportError:
            return None

    def apply(self, *, backend: str = "musa") -> OrionPatchReport:
        """Install the muSPACE operator replacements.

        ``backend`` follows the canonical dispatch names; the default ``musa``
        makes a missing native kernel an error instead of a silent fallback.
        """
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of: {', '.join(BACKENDS)}")
        shimmed = install_flash_attn_shim()
        boxes_iou_bev, nms_gpu, nms_normal_gpu = _make_iou3d_wrappers(backend)
        points_gpu, points_batch, points_cpu = _make_points_in_boxes_wrappers(backend)
        focal_loss = _make_focal_loss_wrapper(backend)

        replacements: list[tuple[str, str, object]] = [
            (IOU3D_UTILS_MODULE, "boxes_iou_bev", boxes_iou_bev),
            (IOU3D_UTILS_MODULE, "nms_gpu", nms_gpu),
            (IOU3D_UTILS_MODULE, "nms_normal_gpu", nms_normal_gpu),
            (IOU3D_DET_MODULE, "boxes_iou_bev", boxes_iou_bev),
            (IOU3D_DET_MODULE, "nms_gpu", nms_gpu),
            (IOU3D_DET_MODULE, "nms_normal_gpu", nms_normal_gpu),
            (BOX3D_NMS_MODULE, "nms_gpu", nms_gpu),
            (BOX3D_NMS_MODULE, "nms_normal_gpu", nms_normal_gpu),
            (MERGE_AUGS_MODULE, "nms_gpu", nms_gpu),
            (MERGE_AUGS_MODULE, "nms_normal_gpu", nms_normal_gpu),
            (POINTS_IN_BOXES_MODULE, "points_in_boxes_gpu", points_gpu),
            (POINTS_IN_BOXES_MODULE, "points_in_boxes_batch", points_batch),
            (POINTS_IN_BOXES_MODULE, "points_in_boxes_cpu", points_cpu),
            (ROIAWARE_MODULE, "points_in_boxes_gpu", points_gpu),
            (ROIAWARE_MODULE, "points_in_boxes_batch", points_batch),
            (ROIAWARE_MODULE, "points_in_boxes_cpu", points_cpu),
            (FOCAL_LOSS_MODULE, "sigmoid_focal_loss", focal_loss),
        ]
        for module_name in IMPORTING_MODULES:
            replacements.extend(
                [
                    (module_name, "nms_gpu", nms_gpu),
                    (module_name, "nms_normal_gpu", nms_normal_gpu),
                ]
            )

        modules: dict[str, ModuleType] = {}
        for module_name, _, _ in replacements:
            if module_name not in modules:
                module = self._module(module_name)
                if module is None:
                    if module_name in IMPORTING_MODULES:
                        # The importing module is loaded by the model build; a missing
                        # one simply leaves those bindings alone.
                        continue
                    raise PatchError(
                        f"{module_name} is unavailable; put the pinned Orion checkout "
                        f"on PYTHONPATH before applying the patcher"
                    )
                modules[module_name] = module

        patched: list[str] = []
        for module_name, attr, replacement in replacements:
            module = modules.get(module_name)
            if module is None:
                continue
            if _patch_attr(self._originals, module, attr, replacement):
                patched.append(f"{module_name}.{attr}")
        self._backend = backend
        return OrionPatchReport(
            target="orion.mmcv",
            backend=backend,
            action="applied",
            flash_attn_shim=shimmed,
            patched_modules=tuple(patched),
        )

    def restore(self) -> OrionPatchReport:
        restored = _restore_attrs(self._originals)
        backend = self._backend or "musa"
        self._backend = None
        return OrionPatchReport(
            target="orion.mmcv",
            backend=backend,
            action="restored",
            flash_attn_shim=False,
            patched_modules=tuple(restored),
        )


_orion_patcher: OrionPatcher | None = None


def orion_patcher() -> OrionPatcher:
    """Return the process-wide Orion patcher, installing it on first use."""
    global _orion_patcher
    if _orion_patcher is None:
        _orion_patcher = OrionPatcher()
        _orion_patcher.apply()
    return _orion_patcher


__all__ = [
    "OrionPatchReport",
    "OrionPatcher",
    "orion_patcher",
]
