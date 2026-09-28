"""Runtime patch entry points for framework compatibility shims."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from types import ModuleType

PatchCallable = Callable[..., object]

STRICT_ENV = "MUSPACE_STRICT_PATCH"
MSDA_TARGET_MODULE = "mmcv.ops.multi_scale_deform_attn"
MSDA_TARGET_ATTR = "multi_scale_deformable_attn_pytorch"
FLASHOCC_BEV_POOL_TARGET_MODULE = "projects.mmdet3d_plugin.ops.bev_pool_v2.bev_pool"
FLASHOCC_BEV_POOL_TARGET_ATTR = "bev_pool_v2"
BEVFUSION_BEV_POOL_TARGET_MODULE = "mmdet3d.ops.bev_pool.bev_pool"
BEVFUSION_BEV_POOL_TARGET_ATTR = "bev_pool"
MODULATED_DEFORM_CONV_TARGET_MODULE = "mmcv.ops.modulated_deform_conv"
MODULATED_DEFORM_CONV_TARGET_ATTR = "ModulatedDeformConv2dPack"
MODULATED_DEFORM_CONV_REGISTRY_MODULE = "mmcv.cnn.bricks.registry"
MODULATED_DEFORM_CONV_REGISTRY_ATTR = "CONV_LAYERS"
MODULATED_DEFORM_CONV_REGISTRY_KEY = "DCNv2"
MODULATED_DEFORM_CONV_IMPORTED_MODULES = ("mmcv.ops",)
DEFORM_CONV_TARGET_MODULE = "mmcv.ops.deform_conv"
DEFORM_CONV_TARGET_ATTR = "DeformConv2dPack"
DEFORM_CONV_REGISTRY_KEY = "DCN"
BEVFORMER_IMPORTED_MODULES = (
    "projects.mmdet3d_plugin.bevformer.modules.decoder",
    "projects.mmdet3d_plugin.bevformer.modules.encoder",
    "projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention",
    "projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention",
)
UNIAD_IMPORTED_MODULES = (
    "projects.mmdet3d_plugin.uniad.modules.decoder",
    "projects.mmdet3d_plugin.uniad.modules.spatial_cross_attention",
    "projects.mmdet3d_plugin.uniad.modules.temporal_self_attention",
    "projects.mmdet3d_plugin.uniad.dense_heads.motion_head_plugin.motion_deformable_attn",
)
FLASHOCC_IMPORTED_MODULES = (
    "projects.mmdet3d_plugin.models.necks.view_transformer",
    "projects.mmdet3d_plugin.ops.bev_pool_v2",
)
BEVFUSION_IMPORTED_MODULES = (
    "mmdet3d.ops.bev_pool",
    "mmdet3d.models.vtransforms.base",
)
BEVDEPTH_TARGET_MODULE = "bevdepth.ops.voxel_pooling_train"
BEVDEPTH_TARGET_ATTR = "voxel_pooling_train"
BEVDEPTH_IMPORTED_MODULES = (
    "bevdepth.layers.backbones.base_lss_fpn",
    "bevdepth.layers.backbones.bevstereo_lss_fpn",
    "bevdepth.layers.backbones.fusion_lss_fpn",
)
BEVFUSION_POINTS_MODULE = "mmdet3d.ops"
BEVFUSION_POINTS_ATTR = "points_in_boxes_batch"
BEVFUSION_POINTS_IMPORTED_MODULES = ("mmdet3d.core.bbox.structures.depth_box3d",)
_MISSING = object()
SPARSEDRIVE_OPS_MODULE = "projects.mmdet3d_plugin.ops"
SPARSEDRIVE_DAF_ATTR = "deformable_aggregation_function"
SPARSEDRIVE_IMPORTED_MODULES = ("projects.mmdet3d_plugin.models.blocks",)
SPARSEDRIVE_IMPORTED_ATTR = "DAF"
BEVFUSION_IOU_MODULE = "mmdet3d.ops.iou3d.iou3d_utils"
BEVFUSION_NMS_MODULE = "mmdet3d.core.post_processing.box3d_nms"


class PatchError(RuntimeError):
    """Raised when runtime patching cannot be installed safely."""


class _MissingLegacyExtension:
    """Placeholder for disabled framework extensions during model imports."""

    def __init__(self, name: str, functions: object) -> None:
        self.name = name
        self.functions = functions

    def __getattr__(self, attr: str) -> object:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        if attr in {"get_compiler_version", "get_compiling_cuda_version"}:
            return lambda: "unavailable"

        # Stay visible to hasattr() probes and raise the descriptive error only
        # when the framework actually calls into the disabled extension.
        def unavailable(*args: object, **kwargs: object) -> object:
            raise PatchError(
                f"Legacy extension mmcv.{self.name} is unavailable; "
                f"the muSPACE replacement must handle this path before {attr} is called"
            )

        unavailable.__name__ = attr
        return unavailable


def _guard_legacy_mmcv_extensions() -> bool:
    """Let compatibility model modules import when mmcv._ext is disabled.

    Returns True when this call installed the replacement, so the caller owns
    the process-wide state and is responsible for releasing it again.
    """

    try:
        ext_loader = importlib.import_module("mmcv.utils.ext_loader")
    except ImportError:
        return False
    if hasattr(ext_loader, "_muspace_original_load_ext"):
        return False
    original = ext_loader.load_ext

    def guarded_load_ext(name: str, functions: object):
        try:
            return original(name, functions)
        except (ImportError, OSError):
            return _MissingLegacyExtension(name, functions)

    ext_loader._muspace_original_load_ext = original
    ext_loader.load_ext = guarded_load_ext
    return True


def _release_legacy_mmcv_extensions() -> bool:
    """Undo _guard_legacy_mmcv_extensions so no global state outlives restore()."""

    ext_loader = sys.modules.get("mmcv.utils.ext_loader")
    if ext_loader is None:
        return False
    original = getattr(ext_loader, "_muspace_original_load_ext", None)
    if original is None:
        return False
    ext_loader.load_ext = original
    del ext_loader._muspace_original_load_ext
    return True


def _patch_attr(
    originals: dict[tuple[str, str], object],
    module: ModuleType,
    attr: str,
    replacement: object,
) -> bool:
    """Rebind module.attr, recording the first value seen for restore()."""

    if not hasattr(module, attr):
        return False
    key = (module.__name__, attr)
    if key not in originals:
        originals[key] = getattr(module, attr)
    setattr(module, attr, replacement)
    return True


def _restore_attrs(originals: dict[tuple[str, str], object]) -> list[str]:
    """Undo every rebind recorded by _patch_attr and clear the record."""

    restored = []
    for (module_name, attr), original in list(originals.items()):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, attr, original)
            restored.append(module_name)
        del originals[(module_name, attr)]
    return restored


@dataclass(frozen=True)
class PatchReport:
    target: str
    strict: bool
    action: str
    patched_modules: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _env_strict() -> bool:
    return os.environ.get(STRICT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _level_start_index(value_spatial_shapes):
    import torch

    level_sizes = value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]
    if level_sizes.numel() == 0:
        return level_sizes.new_empty(0)
    return torch.cat([level_sizes.new_zeros(1), level_sizes.cumsum(0)[:-1]])


def _make_msda_wrapper(strict: bool) -> PatchCallable:
    if strict:

        def strict_msda(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
        ):
            from mu_space.ops import multi_scale_deformable_attn

            return multi_scale_deformable_attn(
                value,
                value_spatial_shapes,
                _level_start_index(value_spatial_shapes),
                sampling_locations,
                attention_weights,
                backend="musa",
            )

        strict_msda.__name__ = "muspace_multi_scale_deformable_attn_strict"
        strict_msda.__module__ = __name__
        return strict_msda

    def compat_msda(
        value,
        value_spatial_shapes,
        sampling_locations,
        attention_weights,
    ):
        from mu_space.ops import multi_scale_deformable_attn_pytorch_compat

        return multi_scale_deformable_attn_pytorch_compat(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
        )

    compat_msda.__name__ = "muspace_multi_scale_deformable_attn_compat"
    compat_msda.__module__ = __name__
    return compat_msda


def _make_bev_pool_v2_wrapper(strict: bool) -> PatchCallable:
    backend = "musa" if strict else "auto"

    def muspace_bev_pool_v2(
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        bev_feat_shape,
        interval_starts,
        interval_lengths,
    ):
        from mu_space.ops import bev_pool_v2

        return bev_pool_v2(
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            bev_feat_shape,
            interval_starts,
            interval_lengths,
            backend=backend,
        )

    muspace_bev_pool_v2.__name__ = (
        "muspace_bev_pool_v2_strict" if strict else "muspace_bev_pool_v2_compat"
    )
    muspace_bev_pool_v2.__module__ = __name__
    return muspace_bev_pool_v2


def _make_bev_pool_wrapper(strict: bool) -> PatchCallable:
    backend = "musa" if strict else "auto"

    def muspace_bev_pool(feats, coords, B, D, H, W):
        from mu_space.ops import bev_pool

        return bev_pool(feats, coords, B, D, H, W, backend=backend)

    muspace_bev_pool.__name__ = "muspace_bev_pool_strict" if strict else "muspace_bev_pool_compat"
    muspace_bev_pool.__module__ = __name__
    return muspace_bev_pool


def _make_voxel_pooling_train_wrapper(strict: bool) -> PatchCallable:
    backend = "musa" if strict else "auto"

    def muspace_voxel_pooling_train(geom_xyz, input_features, voxel_num):
        from mu_space.ops import voxel_pooling_train

        return voxel_pooling_train(geom_xyz, input_features, voxel_num, backend=backend)

    muspace_voxel_pooling_train.__name__ = (
        "muspace_voxel_pooling_train_strict" if strict else "muspace_voxel_pooling_train_compat"
    )
    muspace_voxel_pooling_train.__module__ = __name__
    return muspace_voxel_pooling_train


def _make_points_in_boxes_wrapper(strict: bool) -> PatchCallable:
    backend = "musa" if strict else "auto"

    def muspace_points_in_boxes_batch(points, boxes):
        from mu_space.ops import points_in_boxes_all

        return points_in_boxes_all(boxes, points, backend=backend)

    muspace_points_in_boxes_batch.__name__ = (
        "muspace_points_in_boxes_batch_strict" if strict else "muspace_points_in_boxes_batch_compat"
    )
    muspace_points_in_boxes_batch.__module__ = __name__
    return muspace_points_in_boxes_batch


def _make_deformable_aggregation_wrapper(strict: bool) -> PatchCallable:
    backend = "musa" if strict else "auto"

    def muspace_deformable_aggregation(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        from mu_space.ops import deformable_aggregation_function

        return deformable_aggregation_function(
            feature_maps,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            backend=backend,
        )

    muspace_deformable_aggregation.__name__ = (
        "muspace_deformable_aggregation_strict"
        if strict
        else "muspace_deformable_aggregation_compat"
    )
    muspace_deformable_aggregation.__module__ = __name__
    return muspace_deformable_aggregation


def _make_bevfusion_iou_wrapper(
    backend: str,
) -> tuple[PatchCallable, PatchCallable, PatchCallable]:
    def boxes_iou_bev(boxes_a, boxes_b):
        from mu_space.ops import boxes_iou_bev as operator

        return operator(boxes_a, boxes_b, backend=backend)

    def nms_bev(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
        from mu_space.ops import nms_bev as operator

        return operator(
            boxes,
            scores,
            thresh,
            pre_maxsize=pre_maxsize,
            post_max_size=post_max_size,
            backend=backend,
        )

    def nms_bev_normal(boxes, scores, thresh):
        from mu_space.ops import nms_bev_normal as operator

        return operator(boxes, scores, thresh, backend=backend)

    return boxes_iou_bev, nms_bev, nms_bev_normal


class DefaultPatcher:
    """Install runtime compatibility patches without editing user model code."""

    def __init__(self) -> None:
        self._originals: dict[tuple[str, str], object] = {}
        self._strict: bool | None = None
        self._guarded = False

    def apply(self, *, strict: bool | None = None) -> PatchReport:
        strict = _env_strict() if strict is None else strict
        wrapper = _make_msda_wrapper(strict)
        patched_modules = self._patch_mmcv_msda(wrapper)
        action = "already-applied" if self._strict == strict else "applied"
        self._strict = strict
        return PatchReport(
            target=f"{MSDA_TARGET_MODULE}.{MSDA_TARGET_ATTR}",
            strict=strict,
            action=action,
            patched_modules=tuple(patched_modules),
        )

    def restore(self) -> PatchReport:
        restored = _restore_attrs(self._originals)
        if self._guarded:
            _release_legacy_mmcv_extensions()
            self._guarded = False
        self._strict = None
        return PatchReport(
            target=f"{MSDA_TARGET_MODULE}.{MSDA_TARGET_ATTR}",
            strict=False,
            action="restored",
            patched_modules=tuple(restored),
        )

    def _patch_attr(self, module: ModuleType, attr: str, replacement: object) -> bool:
        return _patch_attr(self._originals, module, attr, replacement)

    def _patch_mmcv_msda(self, replacement: PatchCallable) -> list[str]:
        self._guarded = _guard_legacy_mmcv_extensions() or self._guarded
        try:
            target = importlib.import_module(MSDA_TARGET_MODULE)
        except ImportError as exc:
            raise PatchError(
                "mmcv MSDeformAttn helper is unavailable; install the BEVFormer "
                "OpenMMLab stack before calling default_patcher.apply()"
            ) from exc

        patched = []
        if self._patch_attr(target, MSDA_TARGET_ATTR, replacement):
            patched.append(target.__name__)
        else:
            raise PatchError(f"{MSDA_TARGET_MODULE}.{MSDA_TARGET_ATTR} is unavailable")

        # Models bind this helper at import time. Update loaded globals as well
        # so MUSA tensors use muSPACE instead of the framework fallback.
        for module_name in (*BEVFORMER_IMPORTED_MODULES, *UNIAD_IMPORTED_MODULES):
            module = sys.modules.get(module_name)
            if module is not None and self._patch_attr(module, MSDA_TARGET_ATTR, replacement):
                patched.append(module_name)
        return patched


class FlashOCCPatcher:
    """Install FlashOCC BEVPoolV2 runtime replacement without editing model code."""

    def __init__(self) -> None:
        self._originals: dict[tuple[str, str], object] = {}
        self._strict: bool | None = None
        self._guarded = False

    def apply(self, *, strict: bool | None = None) -> PatchReport:
        strict = _env_strict() if strict is None else strict
        wrapper = _make_bev_pool_v2_wrapper(strict)
        patched_modules = self._patch_flashocc_bev_pool_v2(wrapper)
        action = "already-applied" if self._strict == strict else "applied"
        self._strict = strict
        return PatchReport(
            target=f"{FLASHOCC_BEV_POOL_TARGET_MODULE}.{FLASHOCC_BEV_POOL_TARGET_ATTR}",
            strict=strict,
            action=action,
            patched_modules=tuple(patched_modules),
        )

    def restore(self) -> PatchReport:
        restored = _restore_attrs(self._originals)
        if self._guarded:
            _release_legacy_mmcv_extensions()
            self._guarded = False
        self._strict = None
        return PatchReport(
            target=f"{FLASHOCC_BEV_POOL_TARGET_MODULE}.{FLASHOCC_BEV_POOL_TARGET_ATTR}",
            strict=False,
            action="restored",
            patched_modules=tuple(restored),
        )

    def _patch_attr(self, module: ModuleType, attr: str, replacement: object) -> bool:
        return _patch_attr(self._originals, module, attr, replacement)

    def _patch_flashocc_bev_pool_v2(self, replacement: PatchCallable) -> list[str]:
        self._guarded = _guard_legacy_mmcv_extensions() or self._guarded
        try:
            target = importlib.import_module(FLASHOCC_BEV_POOL_TARGET_MODULE)
        except ImportError as exc:
            raise PatchError(
                "FlashOCC BEVPoolV2 helper import failed. If the environment lacks FlashOCC's "
                "OpenMMLab dependencies or CUDA extension, apply the source patch in "
                "model_examples/FlashOCC/patches/flashocc-01-bev-pool-v2-import.patch instead."
            ) from exc

        patched = []
        if self._patch_attr(target, FLASHOCC_BEV_POOL_TARGET_ATTR, replacement):
            patched.append(target.__name__)
        else:
            raise PatchError(
                f"{FLASHOCC_BEV_POOL_TARGET_MODULE}.{FLASHOCC_BEV_POOL_TARGET_ATTR} is unavailable"
            )

        for module_name in FLASHOCC_IMPORTED_MODULES:
            module = sys.modules.get(module_name)
            if module is not None and self._patch_attr(
                module, FLASHOCC_BEV_POOL_TARGET_ATTR, replacement
            ):
                patched.append(module_name)
        return patched


class BEVFusionPatcher:
    """Install BEVFusion's original ``bev_pool`` replacement at runtime."""

    def __init__(self) -> None:
        self._originals: dict[tuple[str, str], object] = {}
        self._strict: bool | None = None

    def apply(self, *, strict: bool | None = None) -> PatchReport:
        strict = _env_strict() if strict is None else strict
        wrapper = _make_bev_pool_wrapper(strict)
        patched_modules = self._patch_bevfusion_bev_pool(wrapper, strict)
        action = "already-applied" if self._strict == strict else "applied"
        self._strict = strict
        return PatchReport(
            target=f"{BEVFUSION_BEV_POOL_TARGET_MODULE}.{BEVFUSION_BEV_POOL_TARGET_ATTR}",
            strict=strict,
            action=action,
            patched_modules=tuple(patched_modules),
        )

    def restore(self) -> PatchReport:
        restored = []
        for (module_name, attr), original in list(self._originals.items()):
            module = sys.modules.get(module_name)
            if module is not None:
                setattr(module, attr, original)
                restored.append(module_name)
            del self._originals[(module_name, attr)]
        self._strict = None
        return PatchReport(
            target=f"{BEVFUSION_BEV_POOL_TARGET_MODULE}.{BEVFUSION_BEV_POOL_TARGET_ATTR}",
            strict=False,
            action="restored",
            patched_modules=tuple(restored),
        )

    def _patch_attr(self, module: ModuleType, attr: str, replacement: object) -> bool:
        if not hasattr(module, attr):
            return False
        key = (module.__name__, attr)
        if key not in self._originals:
            self._originals[key] = getattr(module, attr)
        setattr(module, attr, replacement)
        return True

    def _patch_bevfusion_bev_pool(self, replacement: PatchCallable, strict: bool) -> list[str]:
        try:
            target = importlib.import_module(BEVFUSION_BEV_POOL_TARGET_MODULE)
        except ImportError as exc:
            raise PatchError(
                "BEVFusion bev_pool helper is unavailable; install the mit-han-lab/BEVFusion "
                "checkout before calling bevfusion_patcher.apply()"
            ) from exc
        patched = []
        if self._patch_attr(target, BEVFUSION_BEV_POOL_TARGET_ATTR, replacement):
            patched.append(target.__name__)
        else:
            raise PatchError(
                f"{BEVFUSION_BEV_POOL_TARGET_MODULE}.{BEVFUSION_BEV_POOL_TARGET_ATTR} is unavailable"
            )
        for module_name in BEVFUSION_IMPORTED_MODULES:
            module = sys.modules.get(module_name)
            if module is not None and self._patch_attr(
                module, BEVFUSION_BEV_POOL_TARGET_ATTR, replacement
            ):
                patched.append(module_name)
        # DepthInstance3DBoxes imports this helper directly at module import time.
        # The symbol is optional in camera-only fixtures, so patch it when present.
        points_wrapper = _make_points_in_boxes_wrapper(strict)
        ops_module = sys.modules.get(BEVFUSION_POINTS_MODULE)
        if ops_module is None:
            try:
                ops_module = importlib.import_module(BEVFUSION_POINTS_MODULE)
            except ImportError:
                ops_module = None
        if ops_module is not None and hasattr(ops_module, BEVFUSION_POINTS_ATTR):
            self._patch_attr(ops_module, BEVFUSION_POINTS_ATTR, points_wrapper)
            patched.append(f"{ops_module.__name__}.{BEVFUSION_POINTS_ATTR}")
            for module_name in BEVFUSION_POINTS_IMPORTED_MODULES:
                module = sys.modules.get(module_name)
                if module is not None and self._patch_attr(
                    module, BEVFUSION_POINTS_ATTR, points_wrapper
                ):
                    patched.append(f"{module_name}.{BEVFUSION_POINTS_ATTR}")
        return patched


class BEVDepthPatcher:
    """Install muSPACE's BEVDepth training voxel-pooling replacement."""

    def __init__(self) -> None:
        self._originals: dict[tuple[str, str], object] = {}
        self._strict: bool | None = None
        self._shim_target: ModuleType | None = None
        self._shim_parent: ModuleType | None = None
        self._shim_parent_attr: object = _MISSING
        self._shim_inference: ModuleType | None = None
        self._torch_device_originals: dict[tuple[object, str], object] = {}
        self._tensor_cuda: object = _MISSING

    def _target_or_shim(self, replacement: PatchCallable) -> ModuleType:
        """Return BEVDepth's target module, creating a MUSA import shim when needed.

        The upstream package imports its CUDA extension eagerly.  That is useful for
        CUDA, but prevents the Python module from being imported on MUSA before the
        runtime replacement has a chance to bind it.  Only an absent extension below
        the known voxel-pooling package is shimmed; unrelated import failures remain
        actionable errors.
        """

        try:
            return importlib.import_module(BEVDEPTH_TARGET_MODULE)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if not missing.startswith(f"{BEVDEPTH_TARGET_MODULE}."):
                raise PatchError(
                    "BEVDepth voxel_pooling_train module is unavailable; install the pinned "
                    "BEVDepth checkout"
                ) from exc
        except ImportError as exc:
            # A package-style upstream binding imports its extension from the
            # partially initialized package, which raises ImportError with no
            # useful ``name`` attribute.  Limit the shim to this known missing
            # CUDA extension; unrelated import failures must remain visible.
            if "voxel_pooling_train_ext" not in str(exc):
                raise PatchError(
                    "BEVDepth voxel_pooling_train module is unavailable; install the pinned "
                    "BEVDepth checkout"
                ) from exc
        except Exception as exc:
            raise PatchError(
                "BEVDepth voxel_pooling_train module is unavailable; install the pinned "
                "BEVDepth checkout"
            ) from exc

        try:
            parent = importlib.import_module("bevdepth.ops")
        except ImportError as exc:
            raise PatchError("BEVDepth ops package is unavailable") from exc
        target = ModuleType(BEVDEPTH_TARGET_MODULE)
        target.__package__ = BEVDEPTH_TARGET_MODULE
        target.voxel_pooling_train = replacement
        self._shim_target = target
        self._shim_parent = parent
        self._shim_parent_attr = getattr(parent, "voxel_pooling_train", _MISSING)
        sys.modules[BEVDEPTH_TARGET_MODULE] = target
        setattr(parent, "voxel_pooling_train", target)
        # base_lss_fpn imports inference and train in one try block.  Provide a
        # loud placeholder for inference so the supported training import can
        # proceed without pretending inference is implemented by muSPACE.
        inference_name = "bevdepth.ops.voxel_pooling_inference"
        inference = ModuleType(inference_name)

        def unavailable_inference(*args, **kwargs):
            raise RuntimeError("BEVDepth voxel_pooling_inference is not covered by this patcher")

        inference.voxel_pooling_inference = unavailable_inference
        self._shim_inference = inference
        sys.modules[inference_name] = inference
        return target

    def _install_tensor_cuda_bridge(self) -> bool:
        """Bridge BEVDepth's legacy ``tensor.cuda()`` call to the active MUSA device."""

        if self._tensor_cuda is not _MISSING:
            return False
        try:
            import torch
        except ImportError:
            return False
        if not hasattr(torch, "musa") or not torch.musa.is_available() or torch.cuda.is_available():
            return False

        original = torch.Tensor.cuda

        def musa_cuda(tensor, device=None, non_blocking=False, memory_format=torch.preserve_format):
            if device is not None and isinstance(device, str) and not device.startswith("cuda"):
                raise RuntimeError(
                    f"BEVDepth legacy cuda bridge received unsupported device {device!r}"
                )
            index = 0
            if isinstance(device, str) and ":" in device:
                index = int(device.rsplit(":", 1)[1])
            elif hasattr(device, "index") and device.index is not None:
                index = int(device.index)
            return tensor.to(
                device=torch.device(f"musa:{index}"),
                non_blocking=non_blocking,
                memory_format=memory_format,
            )

        torch.Tensor.cuda = musa_cuda
        self._tensor_cuda = original
        return True

    def _install_cuda_device_bridge(self) -> bool:
        """Redirect legacy ``device='cuda'`` factories to the active MUSA device."""

        try:
            import torch
        except ImportError:
            return False
        if not hasattr(torch, "musa") or not torch.musa.is_available() or torch.cuda.is_available():
            return False

        def redirect(function):
            def wrapped(*args, **kwargs):
                device = kwargs.get("device")
                if device == "cuda" or (isinstance(device, str) and device.startswith("cuda:")):
                    kwargs["device"] = "musa" + device[4:]
                return function(*args, **kwargs)

            return wrapped

        for name in (
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "zeros_like",
            "ones_like",
            "full_like",
            "tensor",
            "arange",
        ):
            original = getattr(torch, name, None)
            if original is not None and (torch, name) not in self._torch_device_originals:
                setattr(torch, name, redirect(original))
                self._torch_device_originals[(torch, name)] = original
        for name in ("new_zeros", "new_empty", "new_full", "new_tensor"):
            original = getattr(torch.Tensor, name, None)
            if original is not None and (torch.Tensor, name) not in self._torch_device_originals:
                setattr(torch.Tensor, name, redirect(original))
                self._torch_device_originals[(torch.Tensor, name)] = original
        amp = getattr(getattr(torch, "cuda", None), "amp", None)
        original_amp = getattr(amp, "autocast", None)
        if original_amp is not None and (amp, "autocast") not in self._torch_device_originals:

            def musa_autocast(*args, **kwargs):
                enabled = kwargs.pop("enabled", True)
                dtype = kwargs.pop("dtype", torch.float16)
                kwargs.pop("cache_enabled", None)
                return torch.autocast(device_type="musa", dtype=dtype, enabled=enabled, **kwargs)

            amp.autocast = musa_autocast
            self._torch_device_originals[(amp, "autocast")] = original_amp
        functional = torch.nn.functional
        original_bce = functional.binary_cross_entropy
        if (functional, "binary_cross_entropy") not in self._torch_device_originals:

            def musa_bce(*args, **kwargs):
                with torch.autocast(device_type="musa", enabled=False):
                    if args and torch.is_tensor(args[0]):
                        args = (args[0].float(),) + tuple(args[1:])
                    if len(args) > 1 and torch.is_tensor(args[1]):
                        args = (args[0], args[1].float()) + tuple(args[2:])
                    return original_bce(*args, **kwargs)

            functional.binary_cross_entropy = musa_bce
            self._torch_device_originals[(functional, "binary_cross_entropy")] = original_bce
        return True

    def apply(self, *, strict: bool | None = None) -> PatchReport:
        strict = _env_strict() if strict is None else strict
        replacement = _make_voxel_pooling_train_wrapper(strict)
        target = self._target_or_shim(replacement)
        patched = []
        if self._shim_target is target:
            patched.append(target.__name__)
        elif self._patch_attr(target, BEVDEPTH_TARGET_ATTR, replacement):
            patched.append(target.__name__)
        else:
            raise PatchError(f"{BEVDEPTH_TARGET_MODULE}.{BEVDEPTH_TARGET_ATTR} is unavailable")
        self._install_tensor_cuda_bridge()
        self._install_cuda_device_bridge()
        for module_name in BEVDEPTH_IMPORTED_MODULES:
            module = sys.modules.get(module_name)
            if module is not None and self._patch_attr(module, BEVDEPTH_TARGET_ATTR, replacement):
                patched.append(module_name)
        action = "already-applied" if self._strict == strict else "applied"
        self._strict = strict
        return PatchReport(
            target=f"{BEVDEPTH_TARGET_MODULE}.{BEVDEPTH_TARGET_ATTR}",
            strict=strict,
            action=action,
            patched_modules=tuple(patched),
        )

    def restore(self) -> PatchReport:
        restored = _restore_attrs(self._originals)
        if self._shim_target is not None:
            if sys.modules.get(BEVDEPTH_TARGET_MODULE) is self._shim_target:
                del sys.modules[BEVDEPTH_TARGET_MODULE]
            if self._shim_parent is not None:
                if self._shim_parent_attr is _MISSING:
                    if getattr(self._shim_parent, "voxel_pooling_train", None) is self._shim_target:
                        delattr(self._shim_parent, "voxel_pooling_train")
                else:
                    setattr(self._shim_parent, "voxel_pooling_train", self._shim_parent_attr)
            restored.append((BEVDEPTH_TARGET_MODULE, BEVDEPTH_TARGET_ATTR))
        if self._shim_inference is not None:
            inference_name = "bevdepth.ops.voxel_pooling_inference"
            if sys.modules.get(inference_name) is self._shim_inference:
                del sys.modules[inference_name]
            self._shim_inference = None
        if self._tensor_cuda is not _MISSING:
            try:
                import torch

                torch.Tensor.cuda = self._tensor_cuda
            finally:
                self._tensor_cuda = _MISSING
        for (owner, name), original in self._torch_device_originals.items():
            setattr(owner, name, original)
        self._torch_device_originals.clear()
        self._shim_target = None
        self._shim_parent = None
        self._shim_parent_attr = _MISSING
        self._strict = None
        return PatchReport(
            target=f"{BEVDEPTH_TARGET_MODULE}.{BEVDEPTH_TARGET_ATTR}",
            strict=False,
            action="restored",
            patched_modules=tuple(restored),
        )

    def _patch_attr(self, module: ModuleType, attr: str, replacement: object) -> bool:
        return _patch_attr(self._originals, module, attr, replacement)


class SparseDrivePatcher:
    """Install SparseDrive deformable aggregation runtime replacement."""

    def __init__(self) -> None:
        self._originals: dict[tuple[str, str], object] = {}
        self._strict: bool | None = None

    def apply(self, *, strict: bool | None = None) -> PatchReport:
        strict = _env_strict() if strict is None else strict
        wrapper = _make_deformable_aggregation_wrapper(strict)
        patched_modules = self._patch_sparsedrive_daf(wrapper)
        action = "already-applied" if self._strict == strict else "applied"
        self._strict = strict
        return PatchReport(
            target=f"{SPARSEDRIVE_OPS_MODULE}.{SPARSEDRIVE_DAF_ATTR}",
            strict=strict,
            action=action,
            patched_modules=tuple(patched_modules),
        )

    def restore(self) -> PatchReport:
        restored = _restore_attrs(self._originals)
        self._strict = None
        return PatchReport(
            target=f"{SPARSEDRIVE_OPS_MODULE}.{SPARSEDRIVE_DAF_ATTR}",
            strict=False,
            action="restored",
            patched_modules=tuple(restored),
        )

    def _patch_attr(self, module: ModuleType, attr: str, replacement: object) -> bool:
        return _patch_attr(self._originals, module, attr, replacement)

    def _patch_sparsedrive_daf(self, replacement: PatchCallable) -> list[str]:
        try:
            target = importlib.import_module(SPARSEDRIVE_OPS_MODULE)
        except ImportError as exc:
            raise PatchError(
                "SparseDrive ops package import failed. Ensure the SparseDrive checkout is on "
                "PYTHONPATH, or apply a source patch under model_examples/SparseDrive/."
            ) from exc

        patched = []
        if self._patch_attr(target, SPARSEDRIVE_DAF_ATTR, replacement):
            patched.append(target.__name__)
        else:
            raise PatchError(f"{SPARSEDRIVE_OPS_MODULE}.{SPARSEDRIVE_DAF_ATTR} is unavailable")

        for module_name in SPARSEDRIVE_IMPORTED_MODULES:
            module = sys.modules.get(module_name)
            if module is not None and self._patch_attr(
                module, SPARSEDRIVE_IMPORTED_ATTR, replacement
            ):
                patched.append(module_name)
        return patched


class BEVFusionIou3dPatcher:
    """Replace BEVFusion's iou3d BEV IoU and NMS entry points at runtime."""

    def __init__(self) -> None:
        self._originals: dict[tuple[str, str], object] = {}
        self._strict: bool | None = None

    def apply(
        self,
        *,
        strict: bool | None = None,
        backend: str | None = None,
    ) -> PatchReport:
        """Install the BEVFusion iou3d replacement.

        ``strict=True`` remains the compatibility spelling for ``backend="musa"``.
        Passing ``backend`` makes fixture validation deterministic and accepts the
        canonical ``auto``, ``musa``, and ``reference`` backends.
        """
        strict = _env_strict() if strict is None else strict
        if backend is None:
            backend = "musa" if strict else "auto"
        if backend not in {"auto", "musa", "reference"}:
            raise ValueError("backend must be one of: auto, musa, reference")
        if strict and backend != "musa":
            raise ValueError("strict=True requires backend='musa'")
        boxes_iou_bev, nms_bev, nms_bev_normal = _make_bevfusion_iou_wrapper(backend)
        patched = []
        try:
            iou_module = importlib.import_module(BEVFUSION_IOU_MODULE)
            nms_module = importlib.import_module(BEVFUSION_NMS_MODULE)
        except ImportError as exc:
            raise PatchError(
                "BEVFusion mmdet3d iou3d modules are unavailable; put the pinned "
                "BEVFusion checkout on PYTHONPATH before applying the patcher"
            ) from exc
        replacements = (
            (iou_module, "boxes_iou_bev", boxes_iou_bev),
            (iou_module, "nms_gpu", nms_bev),
            (iou_module, "nms_normal_gpu", nms_bev_normal),
            (nms_module, "nms_gpu", nms_bev),
            (nms_module, "nms_normal_gpu", nms_bev_normal),
        )
        missing = [
            f"{module.__name__}.{attr}"
            for module, attr, _ in replacements
            if not hasattr(module, attr)
        ]
        if missing:
            raise PatchError(f"BEVFusion iou3d symbols are unavailable: {', '.join(missing)}")
        originals_before_apply = set(self._originals)
        try:
            for module, attr, replacement in replacements:
                self._patch_attr(module, attr, replacement)
                patched.append(f"{module.__name__}.{attr}")
        except Exception:
            for key in set(self._originals) - originals_before_apply:
                module = sys.modules.get(key[0])
                if module is not None:
                    setattr(module, key[1], self._originals[key])
                del self._originals[key]
            raise
        self._strict = backend == "musa"
        return PatchReport(
            target=BEVFUSION_IOU_MODULE,
            strict=backend == "musa",
            action="applied",
            patched_modules=tuple(patched),
        )

    def restore(self) -> PatchReport:
        restored = []
        for (module_name, attr), original in list(self._originals.items()):
            module = sys.modules.get(module_name)
            if module is not None:
                setattr(module, attr, original)
                restored.append(f"{module_name}.{attr}")
            del self._originals[(module_name, attr)]
        self._strict = None
        return PatchReport(
            target=BEVFUSION_IOU_MODULE,
            strict=False,
            action="restored",
            patched_modules=tuple(restored),
        )

    def _patch_attr(self, module: ModuleType, attr: str, replacement: object) -> bool:
        if not hasattr(module, attr):
            raise PatchError(f"{module.__name__}.{attr} is unavailable")
        key = (module.__name__, attr)
        if key not in self._originals:
            self._originals[key] = getattr(module, attr)
        setattr(module, attr, replacement)
        return True


class ModulatedDeformConv2dPatcher:
    """Install the muSPACE DCNv2 layer in place of mmcv's, without editing model code.

    mmdet-based backbones do not import the layer class; they build it from a config
    entry, ``dcn=dict(type='DCNv2', ...)``, which ``build_conv_layer`` resolves through
    mmcv's ``CONV_LAYERS`` registry. The replacement therefore has to reach the registry
    as well as the module attributes that a direct import would bind. muSPACE's module
    keeps mmcv's constructor signature, parameter names, and state dict layout including
    the legacy ``*_offset`` key rename, so checkpoints load unchanged.

    ``strict`` is recorded for consistency with the other patchers but does not change
    execution: DCNv2 is native-only, so there is no compatibility path to select and a
    non-MUSA tensor raises either way.
    """

    def __init__(self) -> None:
        self._originals: dict[tuple[str, str], object] = {}
        self._registry_original: object | None = None
        self._strict: bool | None = None

    def apply(self, *, strict: bool | None = None) -> PatchReport:
        strict = _env_strict() if strict is None else strict
        from mu_space.ops import ModulatedDeformConv2dPack

        patched_modules = self._patch_mmcv_modulated_deform_conv(ModulatedDeformConv2dPack)
        action = "already-applied" if self._strict == strict else "applied"
        self._strict = strict
        return PatchReport(
            target=(
                f"{MODULATED_DEFORM_CONV_REGISTRY_MODULE}."
                f"{MODULATED_DEFORM_CONV_REGISTRY_ATTR}['{MODULATED_DEFORM_CONV_REGISTRY_KEY}']"
            ),
            strict=strict,
            action=action,
            patched_modules=tuple(patched_modules),
        )

    def restore(self) -> PatchReport:
        restored = []
        if self._registry_original is not None:
            registry = self._registry()
            self._set_registry_entry(registry, self._registry_original)
            restored.append(MODULATED_DEFORM_CONV_REGISTRY_MODULE)
            self._registry_original = None
        for (module_name, attr), original in list(self._originals.items()):
            module = sys.modules.get(module_name)
            if module is not None:
                setattr(module, attr, original)
                restored.append(module_name)
            del self._originals[(module_name, attr)]
        self._strict = None
        return PatchReport(
            target=(
                f"{MODULATED_DEFORM_CONV_REGISTRY_MODULE}."
                f"{MODULATED_DEFORM_CONV_REGISTRY_ATTR}['{MODULATED_DEFORM_CONV_REGISTRY_KEY}']"
            ),
            strict=False,
            action="restored",
            patched_modules=tuple(restored),
        )

    def _patch_attr(self, module: ModuleType, attr: str, replacement: object) -> bool:
        if not hasattr(module, attr):
            return False
        key = (module.__name__, attr)
        if key not in self._originals:
            self._originals[key] = getattr(module, attr)
        setattr(module, attr, replacement)
        return True

    def _registry(self) -> object:
        try:
            registry_module = importlib.import_module(MODULATED_DEFORM_CONV_REGISTRY_MODULE)
        except ImportError as exc:
            raise PatchError(
                "mmcv's convolution registry is unavailable; install the OpenMMLab stack "
                "before calling modulated_deform_conv_patcher.apply()"
            ) from exc
        registry = getattr(registry_module, MODULATED_DEFORM_CONV_REGISTRY_ATTR, None)
        if registry is None:
            raise PatchError(
                f"{MODULATED_DEFORM_CONV_REGISTRY_MODULE}."
                f"{MODULATED_DEFORM_CONV_REGISTRY_ATTR} is unavailable"
            )
        return registry

    @staticmethod
    def _set_registry_entry(
        registry: object, layer: object, key: str = MODULATED_DEFORM_CONV_REGISTRY_KEY
    ) -> None:
        register = getattr(registry, "register_module", None)
        if register is not None:
            register(name=key, force=True, module=layer)
            return
        module_dict = getattr(registry, "_module_dict", None)
        if not isinstance(module_dict, dict):
            raise PatchError("mmcv's convolution registry has an unexpected layout")
        module_dict[key] = layer

    def _patch_mmcv_modulated_deform_conv(self, replacement: object) -> list[str]:
        # mmcv registers 'DCNv2' as a side effect of defining the class, so importing
        # the module first makes apply() independent of the caller's import order.
        try:
            importlib.import_module(MODULATED_DEFORM_CONV_TARGET_MODULE)
        except ImportError as exc:
            raise PatchError(
                "mmcv's modulated deformable convolution is unavailable; install the "
                "OpenMMLab stack before calling modulated_deform_conv_patcher.apply()"
            ) from exc

        registry = self._registry()
        original = registry.get(MODULATED_DEFORM_CONV_REGISTRY_KEY)
        if original is None:
            raise PatchError(
                f"mmcv's convolution registry does not provide "
                f"'{MODULATED_DEFORM_CONV_REGISTRY_KEY}'"
            )
        if self._registry_original is None:
            self._registry_original = original
        self._set_registry_entry(registry, replacement, MODULATED_DEFORM_CONV_REGISTRY_KEY)
        patched = [MODULATED_DEFORM_CONV_REGISTRY_MODULE]

        # Also rebind the class itself, for code that imported it directly.
        for module_name in (
            MODULATED_DEFORM_CONV_TARGET_MODULE,
            *MODULATED_DEFORM_CONV_IMPORTED_MODULES,
        ):
            module = sys.modules.get(module_name)
            if module is not None and self._patch_attr(
                module, MODULATED_DEFORM_CONV_TARGET_ATTR, replacement
            ):
                patched.append(module_name)
        return patched


class DeformConv2dPatcher(ModulatedDeformConv2dPatcher):
    """Install the standard mmcv ``DCN`` layer using muSPACE's native backend."""

    def apply(self, *, strict: bool | None = None) -> PatchReport:
        strict = _env_strict() if strict is None else strict
        if strict:
            from mu_space.runtime.native import native_available, native_error

            if not native_available():
                raise PatchError(
                    "strict standard DCN patching requires muSPACE native operators: "
                    f"{native_error()}"
                )
        from mu_space.ops import DeformConv2dPack

        patched_modules = self._patch_mmcv_standard_deform_conv(DeformConv2dPack)
        action = "already-applied" if self._strict == strict else "applied"
        self._strict = strict
        return PatchReport(
            target=(
                f"{MODULATED_DEFORM_CONV_REGISTRY_MODULE}."
                f"{MODULATED_DEFORM_CONV_REGISTRY_ATTR}['{DEFORM_CONV_REGISTRY_KEY}']"
            ),
            strict=strict,
            action=action,
            patched_modules=tuple(patched_modules),
        )

    def _patch_mmcv_standard_deform_conv(self, replacement: object) -> list[str]:
        try:
            importlib.import_module(DEFORM_CONV_TARGET_MODULE)
        except ImportError as exc:
            raise PatchError(
                "mmcv's deformable convolution is unavailable; install the OpenMMLab stack"
            ) from exc
        registry = self._registry()
        original = registry.get(DEFORM_CONV_REGISTRY_KEY)
        if original is None:
            raise PatchError(
                f"mmcv's convolution registry does not provide '{DEFORM_CONV_REGISTRY_KEY}'"
            )
        if self._registry_original is None:
            self._registry_original = original
        self._set_registry_entry(registry, replacement, DEFORM_CONV_REGISTRY_KEY)
        patched = [MODULATED_DEFORM_CONV_REGISTRY_MODULE]
        for module_name in (DEFORM_CONV_TARGET_MODULE, "mmcv.ops"):
            module = sys.modules.get(module_name)
            if module is not None and self._patch_attr(
                module, DEFORM_CONV_TARGET_ATTR, replacement
            ):
                patched.append(module_name)
        return patched

    def restore(self) -> PatchReport:
        restored = []
        if self._registry_original is not None:
            registry = self._registry()
            self._set_registry_entry(registry, self._registry_original, DEFORM_CONV_REGISTRY_KEY)
            restored.append(MODULATED_DEFORM_CONV_REGISTRY_MODULE)
            self._registry_original = None
        for (module_name, attr), original in list(self._originals.items()):
            module = sys.modules.get(module_name)
            if module is not None:
                setattr(module, attr, original)
                restored.append(module_name)
            del self._originals[(module_name, attr)]
        self._strict = None
        return PatchReport(
            target=(
                f"{MODULATED_DEFORM_CONV_REGISTRY_MODULE}."
                f"{MODULATED_DEFORM_CONV_REGISTRY_ATTR}['{DEFORM_CONV_REGISTRY_KEY}']"
            ),
            strict=False,
            action="restored",
            patched_modules=tuple(restored),
        )


default_patcher = DefaultPatcher()
bevfusion_patcher = BEVFusionPatcher()
flashocc_patcher = FlashOCCPatcher()
sparsedrive_patcher = SparseDrivePatcher()
modulated_deform_conv_patcher = ModulatedDeformConv2dPatcher()
deform_conv_patcher = DeformConv2dPatcher()
bevfusion_iou3d_patcher = BEVFusionIou3dPatcher()
bevdepth_patcher = BEVDepthPatcher()

__all__ = [
    "BEVFusionPatcher",
    "DefaultPatcher",
    "BEVFusionIou3dPatcher",
    "BEVDepthPatcher",
    "FlashOCCPatcher",
    "ModulatedDeformConv2dPatcher",
    "DeformConv2dPatcher",
    "PatchError",
    "PatchReport",
    "SparseDrivePatcher",
    "bevfusion_patcher",
    "default_patcher",
    "bevfusion_iou3d_patcher",
    "bevdepth_patcher",
    "flashocc_patcher",
    "modulated_deform_conv_patcher",
    "deform_conv_patcher",
    "sparsedrive_patcher",
]
