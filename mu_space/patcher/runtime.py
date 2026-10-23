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
SPARSEDRIVE_PACKAGE = "projects"
SPARSEDRIVE_DAF_VALID_ATTR = "DAF_VALID"
SPARSEDRIVE_CUDA_SHIM = "torch.Tensor.cuda"
SPARSEDRIVE_FEATURE_MAPS_ATTR = "feature_maps_format"
SPARSEDRIVE_EXT_MODULE = "projects.mmdet3d_plugin.ops.deformable_aggregation_ext"
SPARSEDRIVE_EXT_ATTR = "deformable_aggregation_ext"
SPARSEDRIVE_IMPORT_FAILURE = (
    "SparseDrive ops package import failed. Ensure the SparseDrive checkout is on "
    "PYTHONPATH, or apply a source patch under model_examples/SparseDrive/."
)
BEVFUSION_IOU_MODULE = "mmdet3d.ops.iou3d.iou3d_utils"
BEVFUSION_NMS_MODULE = "mmdet3d.core.post_processing.box3d_nms"
QWENDRIVE_ATTENTION_MODULE = "qwen_drive_perception.attention"
QWENDRIVE_MSDA_ATTR = "multi_scale_deformable_attn_cuda"
QWENDRIVE_OPS_MODULE = "qwen_drive_perception.ops"
QWENDRIVE_MSDA_FORWARD_ATTR = "ms_deform_attn_bf16_forward"
QWENDRIVE_VOXEL_POOL_ATTR = "voxel_pool_depth"
QWENDRIVE_VIEW_MODULE = "qwen_drive_perception.view_transform"
QWENDRIVE_VIEW_CLASS = "Uni3DVoxelPoolDepth"
QWENDRIVE_FRUSTUM_ATTR = "frustum"


SPARSEDRIVE_ATTENTION_MODULE = "projects.mmdet3d_plugin.models.attention"
SPARSEDRIVE_ATTENTION_CLASS = "FlashAttention"
SPARSEDRIVE_MATE_MODULE = "mate"
SPARSEDRIVE_FLASH_ATTN_MODULE = "flash_attn"


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


def _missing_sparsedrive_extension(error: ImportError) -> bool:
    """Report whether an ImportError is the unbuilt SparseDrive CUDA extension.

    Only that one cause is tolerated. A checkout that is missing mmcv, mmdet, or
    mmdet3d must still fail with the actionable PatchError instead of being masked.
    """

    name = getattr(error, "name", None)
    if isinstance(name, str) and name in {SPARSEDRIVE_EXT_MODULE, SPARSEDRIVE_EXT_ATTR}:
        return True
    return SPARSEDRIVE_EXT_ATTR in str(error)


def _unavailable_sparsedrive_extension_attr(attr: str) -> object:
    """Fail loudly when something really calls into the unbuilt extension."""

    if attr.startswith("__") and attr.endswith("__"):
        raise AttributeError(attr)

    def unavailable(*args: object, **kwargs: object) -> object:
        raise PatchError(
            f"SparseDrive extension {SPARSEDRIVE_EXT_MODULE} is unavailable; the muSPACE "
            f"deformable aggregation replacement must handle this path before {attr} is called"
        )

    unavailable.__name__ = attr
    return unavailable


def _install_sparsedrive_extension_placeholder() -> bool:
    """Register a placeholder for the SparseDrive extension this recipe never builds.

    ``projects.mmdet3d_plugin.ops`` imports ``deformable_aggregation_ext`` at module
    scope even though the muSPACE runtime patch replaces the only binding that reaches
    it. Returns True when this call installed the placeholder, so the caller owns the
    process-wide state and is responsible for releasing it again.
    """

    if SPARSEDRIVE_EXT_MODULE in sys.modules:
        return False
    module = ModuleType(SPARSEDRIVE_EXT_MODULE)
    module.__doc__ = (
        "Placeholder for SparseDrive's deformable_aggregation CUDA extension, which this "
        "recipe never builds; the muSPACE runtime patch replaces the binding that uses it."
    )
    module.__getattr__ = _unavailable_sparsedrive_extension_attr
    sys.modules[SPARSEDRIVE_EXT_MODULE] = module
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


def _bind_attr(
    originals: dict[tuple[str, str], object],
    module: ModuleType,
    attr: str,
    value: object,
) -> bool:
    """Bind module.attr, creating it when absent, and record what was there before."""

    key = (module.__name__, attr)
    if key not in originals:
        originals[key] = getattr(module, attr, _MISSING)
    setattr(module, attr, value)
    return True


def _restore_attrs(originals: dict[tuple[str, str], object]) -> list[str]:
    """Undo every rebind recorded by _patch_attr and clear the record."""

    restored = []
    for (module_name, attr), original in list(originals.items()):
        module = sys.modules.get(module_name)
        if module is not None:
            if original is _MISSING:
                if hasattr(module, attr):
                    delattr(module, attr)
            else:
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


def _accelerated_tensor(value) -> bool:
    """True for a tensor on MUSA or CUDA, including the private-use device spelling."""

    device_type = getattr(getattr(value, "device", None), "type", None)
    return device_type in {"musa", "privateuseone", "cuda"} or bool(
        getattr(value, "is_musa", False) or getattr(value, "is_cuda", False)
    )


def _make_qwendrive_msda_wrapper(strict: bool, original: PatchCallable) -> PatchCallable:
    """Six-argument Qwen-Drive MSDA entry with the upstream ``im2col_step``.

    The published gate rejects anything that is not a CUDA bfloat16 tensor before
    it ever calls the forward-only kernel. A MUSA tensor reports ``is_cuda`` as
    false, so replacing only the ops symbol never runs on MUSA. This wrapper is
    installed in its place: bfloat16 on MUSA or CUDA goes through the autograd
    operator, and every other device keeps the original PyTorch fallback.
    """

    backend = "musa" if strict else "auto"

    def muspace_qwendrive_msda(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        if not _accelerated_tensor(value):
            return original(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step,
            )
        import torch

        if value.dtype != torch.bfloat16:
            raise TypeError(
                "the ms_deform_attn kernel requires bfloat16 tensors on MUSA or CUDA, "
                f"got {value.dtype}; run the model in bfloat16"
            )
        from mu_space.ops import multi_scale_deformable_attn

        return multi_scale_deformable_attn(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=int(im2col_step),
            backend=backend,
        )

    muspace_qwendrive_msda.__name__ = (
        "muspace_qwendrive_msda_strict" if strict else "muspace_qwendrive_msda"
    )
    muspace_qwendrive_msda.__module__ = __name__
    return muspace_qwendrive_msda


def _make_qwendrive_voxel_pool_wrapper(strict: bool) -> PatchCallable:
    """Forward the view-transform call into the muSPACE autograd operator."""

    backend = "musa" if strict else "auto"

    def muspace_qwendrive_voxel_pool_depth(
        img_feats, img_depth, voxel_coords, mask, batch, size_x, size_y, size_z
    ):
        from mu_space.ops import voxel_pool_depth

        return voxel_pool_depth(
            img_feats,
            img_depth,
            voxel_coords,
            mask,
            batch,
            size_x,
            size_y,
            size_z,
            backend=backend,
        )

    muspace_qwendrive_voxel_pool_depth.__name__ = (
        "muspace_qwendrive_voxel_pool_depth_strict"
        if strict
        else "muspace_qwendrive_voxel_pool_depth"
    )
    muspace_qwendrive_voxel_pool_depth.__module__ = __name__
    return muspace_qwendrive_voxel_pool_depth


def _qwendrive_frustum_device() -> str:
    """Pick the device the frustum grid is built on.

    The published property chooses CUDA whenever it is available and CPU
    otherwise. On a MUSA runtime CUDA is not available, so the grid would be
    built on CPU while the features live on MUSA and voxel pooling cannot run.
    """

    import torch

    musa = getattr(torch, "musa", None)
    if musa is not None and musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


def mate_flash_attention_available() -> bool:
    """True when MATE can stand in for the flash_attn package SparseDrive imports.

    A real ``flash_attn`` already in ``sys.modules`` keeps ownership: the checkout
    prefers it, and this recipe never shadows an installed accelerator package.
    """

    if SPARSEDRIVE_FLASH_ATTN_MODULE in sys.modules:
        return False
    try:
        mate = importlib.import_module(SPARSEDRIVE_MATE_MODULE)
    except ImportError:
        return False
    return hasattr(mate, "flash_attn_varlen_func")


def _flash_attention_dtype(tensor: object) -> object:
    """Return the tensor in a dtype MATE's attention kernels accept.

    Upstream asserts fp16/bf16 and SparseDrive runs its attention blocks under fp16
    autocast; a float32 call is cast the same way flash_attn would have been.
    """

    import torch

    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor
    return tensor.to(torch.float16)


def _unpad_input(hidden_states: object, attention_mask: object):  # noqa: ANN201
    """``flash_attn.bert_padding.unpad_input`` for the mask branch of the checkout."""

    import torch

    mask = attention_mask.to(torch.bool)
    batch, seqlen = int(mask.shape[0]), int(mask.shape[1])
    indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1).to(torch.int32)
    lengths = mask.reshape(batch, seqlen).sum(dim=1)
    cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device=mask.device)
    if batch:
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0).to(torch.int32)
    unpadded = hidden_states.reshape(batch * seqlen, *hidden_states.shape[2:])
    return (
        unpadded.index_select(0, indices.to(torch.long)),
        indices,
        cu_seqlens,
        int(lengths.max().item()) if batch else 0,
    )


def _pad_input(hidden_states: object, indices: object, batch: int, seqlen: int) -> object:
    """``flash_attn.bert_padding.pad_input`` for the mask branch of the checkout."""

    import torch

    padded = hidden_states.new_zeros(batch * seqlen, *hidden_states.shape[1:])
    padded.index_copy_(0, indices.to(torch.long), hidden_states)
    return padded.reshape(batch, seqlen, *hidden_states.shape[1:])


def _index_first_axis(tensor: object, indices: object) -> object:
    """``flash_attn.bert_padding.index_first_axis`` for the checkout's import."""

    import torch

    return tensor.index_select(0, indices.to(tensor.device).to(torch.long))


def _musa_device(torch: object, device: object) -> object:
    """Map a ``.cuda()`` device argument onto the equivalent MUSA device."""

    if device is None:
        return torch.device("musa", torch.musa.current_device())
    if isinstance(device, int):
        return torch.device("musa", device)
    if isinstance(device, str):
        if device.isdigit():
            return torch.device("musa", int(device))
        if device.startswith("cuda"):
            index = device[5:].lstrip(":")
            return torch.device("musa", int(index) if index else 0)
        return torch.device(device)
    if isinstance(device, torch.device) and device.type == "cuda":
        return torch.device("musa", device.index or 0)
    return device


def _cuda_device_shim(
    tensor: object, device: object = None, non_blocking: bool = False, memory_format: object = None
) -> object:
    """``Tensor.cuda`` for a MUSA-only process.

    The checkout hardcodes ``.cuda()`` in the grid mask and the detection head. Without
    CUDA devices that call cannot work, so it places the tensor on the current MUSA
    device instead; the kernel that consumes it still fails loudly if it is CUDA-only.
    """

    import torch

    kwargs = {"non_blocking": non_blocking}
    if memory_format is not None:
        kwargs["memory_format"] = memory_format
    return tensor.to(_musa_device(torch, device), **kwargs)


def _cuda_shim_available() -> bool:
    """True when this process has MUSA devices and no CUDA devices."""

    try:
        import torch
    except ImportError:
        return False
    musa = getattr(torch, "musa", None)
    return not torch.cuda.is_available() and bool(musa is not None and musa.is_available())


def _mate_kvpacked_func(  # noqa: PLR0913
    q: object,
    kv: object,
    cu_seqlens_q: object = None,
    cu_seqlens_k: object = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **kwargs: object,
):
    """``flash_attn_unpadded_kvpacked_func`` backed by MATE's variable-length attention.

    SparseDrive packs key and value on the third-last axis; MATE splits them, so this
    adapter keeps the upstream call site untouched.
    """

    if dropout_p:
        raise PatchError("MATE attention does not implement dropout; disable attention dropout")
    if q.dim() == 4:
        if cu_seqlens_q is not None:
            raise PatchError("batched queries require cu_seqlens_q=None")
        q = q.flatten(0, 1)
    if kv.dim() == 5:
        if cu_seqlens_k is not None:
            raise PatchError("batched key/value require cu_seqlens_k=None")
        kv = kv.flatten(0, 1)
    if kv.dim() != 4 or kv.shape[-3] != 2:
        raise PatchError(f"kv must be (tokens, 2, heads, dim), got {tuple(kv.shape)}")
    mate = importlib.import_module(SPARSEDRIVE_MATE_MODULE)
    key, value = kv.unbind(dim=-3)
    return mate.flash_attn_varlen_func(
        q,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=bool(causal),
    )


def _mate_flash_attention_forward(  # noqa: PLR0913
    self: object, q: object, kv: object, causal: bool = False, key_padding_mask: object = None
):
    """``FlashAttention.forward`` on MATE.

    The upstream method asserts ``q.is_cuda``, which every MUSA tensor fails, and calls
    an extension that this recipe never builds for MUSA. The signature and the
    ``(output, None)`` return shape are unchanged.
    """

    import torch

    # Upstream wraps this method in ``auto_fp16(apply_to=('q', 'kv'), out_fp32=True)``:
    # float32 inputs are cast for the kernel and the result comes back as float32.
    widened = q.dtype == torch.float32 or kv.dtype == torch.float32
    q = _flash_attention_dtype(q)
    kv = _flash_attention_dtype(kv)
    batch, seqlen_q = int(q.shape[0]), int(q.shape[1])
    seqlen_k = int(kv.shape[1])
    heads, head_dim = int(q.shape[-2]), int(q.shape[-1])
    q_flat = q.reshape(batch * seqlen_q, heads, head_dim)
    cu_seqlens_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device
    )
    if key_padding_mask is None:
        kv_flat = kv.reshape(batch * seqlen_k, 2, heads, head_dim)
        cu_seqlens_k = torch.arange(
            0, (batch + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=kv.device
        )
        max_seqlen_k = seqlen_k
    else:
        kv_flat, cu_seqlens_k, max_seqlen_k = _unpad_kvpacked(
            kv.reshape(batch * seqlen_k, 2, heads, head_dim), key_padding_mask, batch, seqlen_k
        )
    output = _mate_kvpacked_func(
        q_flat,
        kv_flat,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale=getattr(self, "softmax_scale", None),
        causal=causal,
    )
    output = output.reshape(batch, seqlen_q, heads, head_dim)
    if widened and output.dtype != torch.float32:
        output = output.float()
    return output, None


def _unpad_kvpacked(kv_flat: object, key_padding_mask: object, batch: int, seqlen: int):  # noqa: ANN201
    """Drop padded key/value positions and return the matching ``cu_seqlens``."""

    import torch

    keep = key_padding_mask.reshape(batch * seqlen).to(torch.bool)
    lengths = keep.reshape(batch, seqlen).sum(dim=1)
    cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device=key_padding_mask.device)
    if batch:
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0).to(torch.int32)
    packed = kv_flat.index_select(0, keep.nonzero(as_tuple=False).squeeze(1).to(torch.long))
    return packed, cu_seqlens, int(lengths.max().item()) if batch else 0


def _install_mate_flash_attention_module() -> list[ModuleType]:
    """Register a MATE-backed ``flash_attn`` package for the checkout to import.

    ``projects.mmdet3d_plugin.models.attention`` imports ``flash_attn`` and its
    ``bert_padding`` helpers at module scope. Returns the modules this call created so
    the patcher owns them and can drop them again.
    """

    if SPARSEDRIVE_FLASH_ATTN_MODULE in sys.modules:
        return []
    package = ModuleType(SPARSEDRIVE_FLASH_ATTN_MODULE)
    package.__doc__ = "MATE-backed stand-in for the flash_attn package SparseDrive imports."
    package.__path__ = []  # mark as a package so the submodule imports resolve
    interface = ModuleType(f"{SPARSEDRIVE_FLASH_ATTN_MODULE}.flash_attn_interface")
    interface.__doc__ = "MATE-backed flash_attn entry points."
    interface.flash_attn_unpadded_kvpacked_func = _mate_kvpacked_func
    interface.flash_attn_varlen_kvpacked_func = _mate_kvpacked_func
    padding = ModuleType(f"{SPARSEDRIVE_FLASH_ATTN_MODULE}.bert_padding")
    padding.__doc__ = "Padding helpers backed by torch indexing."
    padding.unpad_input = _unpad_input
    padding.pad_input = _pad_input
    padding.index_first_axis = _index_first_axis
    package.flash_attn_interface = interface
    package.bert_padding = padding
    modules = [package, interface, padding]
    for module in modules:
        sys.modules[module.__name__] = module
    return modules


class SparseDrivePatcher:
    """Install SparseDrive deformable aggregation runtime replacement."""

    def __init__(self) -> None:
        self._originals: dict[tuple[str, str], object] = {}
        self._strict: bool | None = None
        self._extension_placeholder: ModuleType | None = None
        self._attention_originals: list[tuple[object, str, object]] = []
        self._flash_attention_modules: list[ModuleType] = []
        self._cuda_shim: object | None = None

    def apply(self, *, strict: bool | None = None) -> PatchReport:
        strict = _env_strict() if strict is None else strict
        wrapper = _make_deformable_aggregation_wrapper(strict)
        # The checkout's package __init__ imports ops, and ops pulls the attention module
        # in, so the MATE bridge has to be registered before that import runs.
        self._install_attention_bridge()
        cuda_shim = self._install_cuda_device_shim()
        try:
            patched_modules = self._patch_sparsedrive_daf(wrapper)
            attention_module = self._patch_sparsedrive_attention()
        except BaseException:
            self._release_attention_bridge()
            self._release_cuda_device_shim()
            raise
        if attention_module is not None and attention_module not in patched_modules:
            patched_modules = [*patched_modules, attention_module]
        if cuda_shim:
            patched_modules = [*patched_modules, SPARSEDRIVE_CUDA_SHIM]
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
        self._release_cuda_device_shim()
        self._release_attention_patch()
        self._release_extension_placeholder()
        self._strict = None
        return PatchReport(
            target=f"{SPARSEDRIVE_OPS_MODULE}.{SPARSEDRIVE_DAF_ATTR}",
            strict=False,
            action="restored",
            patched_modules=tuple(restored),
        )

    def _patch_attr(self, module: ModuleType, attr: str, replacement: object) -> bool:
        return _patch_attr(self._originals, module, attr, replacement)

    def _release_extension_placeholder(self) -> None:
        """Drop the placeholder this patcher installed, if it still owns it."""

        placeholder = self._extension_placeholder
        self._extension_placeholder = None
        if placeholder is None:
            return
        if sys.modules.get(SPARSEDRIVE_EXT_MODULE) is placeholder:
            del sys.modules[SPARSEDRIVE_EXT_MODULE]

    def _import_sparsedrive_ops(self) -> ModuleType:
        """Import the upstream ops package, tolerating its unbuilt CUDA extension.

        ``projects.mmdet3d_plugin.ops`` imports ``deformable_aggregation_ext`` at module
        scope, and this recipe never builds that CUDA extension: the runtime patch
        replaces the Python binding that reaches it. When the extension is the only
        missing piece a placeholder keeps the import working; any other import failure
        keeps the actionable checkout hint.
        """

        try:
            return importlib.import_module(SPARSEDRIVE_OPS_MODULE)
        except ImportError as exc:
            if not _missing_sparsedrive_extension(exc):
                raise PatchError(f"{SPARSEDRIVE_IMPORT_FAILURE} ({exc})") from exc
            installed = _install_sparsedrive_extension_placeholder()
            try:
                target = importlib.import_module(SPARSEDRIVE_OPS_MODULE)
            except ImportError as retry_exc:
                if installed:
                    sys.modules.pop(SPARSEDRIVE_EXT_MODULE, None)
                raise PatchError(f"{SPARSEDRIVE_IMPORT_FAILURE} ({retry_exc})") from retry_exc
            if installed:
                self._extension_placeholder = sys.modules.get(SPARSEDRIVE_EXT_MODULE)
            return target

    def _install_cuda_device_shim(self) -> bool:
        """Route the checkout's hardcoded ``.cuda()`` calls to the current MUSA device."""

        if self._cuda_shim is not None or not _cuda_shim_available():
            return False
        import torch

        self._cuda_shim = torch.Tensor.cuda
        torch.Tensor.cuda = _cuda_device_shim
        return True

    def _release_cuda_device_shim(self) -> None:
        """Undo the ``.cuda()`` shim if this patcher installed it."""

        if self._cuda_shim is None:
            return
        import torch

        torch.Tensor.cuda = self._cuda_shim
        self._cuda_shim = None

    def _install_attention_bridge(self) -> None:
        """Register MATE's flash_attn before the checkout imports its attention module.

        The attention module imports ``flash_attn`` and its ``bert_padding`` helpers at
        module scope, and the checkout's package __init__ reaches that module while it
        imports ops, so the bridge has to exist first. Without MATE nothing is registered
        and the documented import failure stands.
        """

        if self._flash_attention_modules or not mate_flash_attention_available():
            return
        self._flash_attention_modules.extend(_install_mate_flash_attention_module())

    def _patch_sparsedrive_attention(self) -> str | None:
        """Rebind ``FlashAttention.forward`` onto the MATE bridge.

        Upstream asserts ``q.is_cuda``, which every MUSA tensor fails, and calls an
        extension this recipe never builds for MUSA. The rest of the method keeps the
        upstream signature and its ``(output, None)`` return shape.
        """

        if self._attention_originals:
            return SPARSEDRIVE_ATTENTION_MODULE
        if not self._flash_attention_modules:
            return None
        try:
            module = importlib.import_module(SPARSEDRIVE_ATTENTION_MODULE)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "") or ""
            if missing and SPARSEDRIVE_ATTENTION_MODULE.startswith(missing):
                # This checkout carries no attention module, so there is nothing to bridge.
                self._release_attention_bridge()
                return None
            self._release_attention_bridge()
            raise PatchError(
                f"SparseDrive attention module import failed with MATE available ({exc}). "
                "Install the recipe dependencies from requirements.lock."
            ) from exc
        except ImportError as exc:
            self._release_attention_bridge()
            raise PatchError(
                f"SparseDrive attention module import failed with MATE available ({exc}). "
                "Install the recipe dependencies from requirements.lock."
            ) from exc
        flash_attention = getattr(module, SPARSEDRIVE_ATTENTION_CLASS, None)
        original = getattr(flash_attention, "forward", None)
        if flash_attention is None or original is None:
            return None
        self._attention_originals.append((flash_attention, "forward", original))
        flash_attention.forward = _mate_flash_attention_forward
        return module.__name__

    def _release_attention_bridge(self) -> None:
        """Drop the MATE flash_attn modules this patcher registered."""

        for module in self._flash_attention_modules:
            if sys.modules.get(module.__name__) is module:
                del sys.modules[module.__name__]
        self._flash_attention_modules.clear()

    def _release_attention_patch(self) -> None:
        """Undo the attention rebind and the MATE flash_attn modules this patcher owns."""

        for owner, attr, original in reversed(self._attention_originals):
            setattr(owner, attr, original)
        self._attention_originals.clear()
        self._release_attention_bridge()

    def _patch_sparsedrive_daf(self, replacement: PatchCallable) -> list[str]:
        target = self._import_sparsedrive_ops()
        try:
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

            # The checkout probes the ops import at module scope and caches the answer, so
            # a module imported before the extension placeholder existed keeps
            # DAF_VALID=False even though the patched entry point is now available.
            prefix = f"{SPARSEDRIVE_PACKAGE}."
            for name, module in list(sys.modules.items()):
                if module is None or not (name == SPARSEDRIVE_PACKAGE or name.startswith(prefix)):
                    continue
                if getattr(module, SPARSEDRIVE_DAF_VALID_ATTR, None) is not False:
                    continue
                _bind_attr(self._originals, module, SPARSEDRIVE_DAF_VALID_ATTR, True)
                patched.append(name)
                # The same guarded import left the name it tried to bind undefined.
                ops_module = sys.modules.get(SPARSEDRIVE_OPS_MODULE)
                feature_maps_format = getattr(ops_module, SPARSEDRIVE_FEATURE_MAPS_ATTR, None)
                if feature_maps_format is not None and not hasattr(
                    module, SPARSEDRIVE_FEATURE_MAPS_ATTR
                ):
                    _bind_attr(
                        self._originals, module, SPARSEDRIVE_FEATURE_MAPS_ATTR, feature_maps_format
                    )
            return patched
        except BaseException:
            # A failed apply must not leave a placeholder or a partial rebind behind.
            _restore_attrs(self._originals)
            self._release_extension_placeholder()
            raise


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


def _qwendrive_frustum_target(module):
    """Device the frustum grid should occupy.

    ``module.to(device)`` does not move the plain ``_frustum`` attribute, and a
    type-only check would keep a ``musa:0`` grid after the head moves to
    ``musa:1``. Follow the module parameters, including the device index, and
    otherwise use the published CUDA-or-CPU choice with MUSA preferred.
    """

    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        try:
            iterator = parameters()
        except TypeError:
            iterator = None
        if iterator is not None:
            for parameter in iterator:
                device = getattr(parameter, "device", None)
                if device is not None and getattr(device, "type", None) not in (None, "meta"):
                    return device
    return _qwendrive_frustum_device()


def _qwendrive_same_device(stored, target) -> bool:
    """True when ``stored`` already lives on ``target``, including its index."""

    if stored is None:
        return False
    import torch

    target_device = target if isinstance(target, torch.device) else torch.device(target)
    if stored.type != target_device.type:
        return False
    if stored.type == "cpu":
        return True
    stored_index = 0 if stored.index is None else stored.index
    target_index = 0 if target_device.index is None else target_device.index
    return stored_index == target_index


def _qwendrive_frustum_getter(self):
    """Rebuild of the published frustum grid, on the module's device.

    The body matches the upstream property: ``arange`` over the frustum range,
    ``meshgrid(..., indexing="ij")``, and a plain ``_frustum`` attribute so
    ``model.to(bfloat16)`` never casts it. The dtype is fixed to fp32. Integer
    bounds would make ``arange`` return int64, and ``meshgrid`` would reject the
    mixed grid. Moving the head to another card rebuilds the grid.
    """

    import torch

    device = _qwendrive_frustum_target(self)
    if not _qwendrive_same_device(None if self._frustum is None else self._frustum.device, device):
        self._frustum = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(
                        self.frustum_range[index],
                        self.frustum_range[index + 3],
                        self.frustum_size[index],
                        dtype=torch.float32,
                        device=device,
                    )
                    for index in range(3)
                ],
                indexing="ij",
            ),
            dim=-1,
        )
    return self._frustum


class QwenDrivePatcher:
    """Rebind the Qwen-Drive perception head onto the muSPACE operators.

    The checkout is left untouched. The perception package routes multi-scale
    deformable attention through a CUDA-only forward kernel and builds its
    view-transform frustum on CUDA or CPU, neither of which runs on MUSA. This
    patcher replaces those four entry points after the package is imported and
    before the head is constructed.
    """

    def __init__(self) -> None:
        self._originals: dict[tuple[str, str], object] = {}
        self._descriptor: tuple[str, str, property] | None = None
        self._strict: bool | None = None

    def apply(self, *, strict: bool | None = None) -> PatchReport:
        strict = _env_strict() if strict is None else strict
        attention, ops, view = self._import_package()
        self._require(attention, ops, view)
        # A second apply must keep the published PyTorch fallback. The attribute
        # currently installed may already be our wrapper.
        original = self._originals.get(
            (attention.__name__, QWENDRIVE_MSDA_ATTR),
            getattr(attention, QWENDRIVE_MSDA_ATTR),
        )
        msda_wrapper = _make_qwendrive_msda_wrapper(strict, original)
        voxel_wrapper = _make_qwendrive_voxel_pool_wrapper(strict)
        patched = self._install(attention, ops, view, msda_wrapper, voxel_wrapper)
        action = "already-applied" if self._strict == strict else "applied"
        self._strict = strict
        return PatchReport(
            target=f"{QWENDRIVE_ATTENTION_MODULE}.{QWENDRIVE_MSDA_ATTR}",
            strict=strict,
            action=action,
            patched_modules=tuple(patched),
        )

    def restore(self) -> PatchReport:
        restored = _restore_attrs(self._originals)
        if self._descriptor is not None:
            module_name, class_name, original = self._descriptor
            module = sys.modules.get(module_name)
            owner = getattr(module, class_name, None) if module is not None else None
            if owner is not None:
                setattr(owner, QWENDRIVE_FRUSTUM_ATTR, original)
                restored.append(module_name)
            self._descriptor = None
        self._strict = None
        return PatchReport(
            target=f"{QWENDRIVE_ATTENTION_MODULE}.{QWENDRIVE_MSDA_ATTR}",
            strict=False,
            action="restored",
            patched_modules=tuple(restored),
        )

    def _patch_attr(self, module: ModuleType, attr: str, replacement: object) -> bool:
        return _patch_attr(self._originals, module, attr, replacement)

    def _import_package(self) -> tuple[ModuleType, ModuleType, ModuleType]:
        try:
            attention = importlib.import_module(QWENDRIVE_ATTENTION_MODULE)
            ops = importlib.import_module(QWENDRIVE_OPS_MODULE)
            view = importlib.import_module(QWENDRIVE_VIEW_MODULE)
        except ImportError as exc:
            raise PatchError(
                "qwen_drive_perception import failed. Add the Qwen-Drive checkout src "
                "directory to PYTHONPATH before qwendrive_patcher.apply()."
            ) from exc
        return attention, ops, view

    def _require(self, attention: ModuleType, ops: ModuleType, view: ModuleType) -> None:
        missing = []
        if not callable(getattr(attention, QWENDRIVE_MSDA_ATTR, None)):
            missing.append(f"{QWENDRIVE_ATTENTION_MODULE}.{QWENDRIVE_MSDA_ATTR}")
        for module, module_name, attr in (
            (ops, QWENDRIVE_OPS_MODULE, QWENDRIVE_MSDA_FORWARD_ATTR),
            (ops, QWENDRIVE_OPS_MODULE, QWENDRIVE_VOXEL_POOL_ATTR),
        ):
            if not callable(getattr(module, attr, None)):
                missing.append(f"{module_name}.{attr}")
        owner = getattr(view, QWENDRIVE_VIEW_CLASS, None)
        frustum = getattr(owner, QWENDRIVE_FRUSTUM_ATTR, None) if owner is not None else None
        if not isinstance(frustum, property):
            missing.append(
                f"{QWENDRIVE_VIEW_MODULE}.{QWENDRIVE_VIEW_CLASS}.{QWENDRIVE_FRUSTUM_ATTR}"
            )
        if missing:
            raise PatchError(
                "qwen_drive_perception is missing the symbols this patcher replaces: "
                + ", ".join(missing)
            )

    def _install(
        self,
        attention: ModuleType,
        ops: ModuleType,
        view: ModuleType,
        msda_wrapper: PatchCallable,
        voxel_wrapper: PatchCallable,
    ) -> list[str]:
        patched = []
        if self._patch_attr(attention, QWENDRIVE_MSDA_ATTR, msda_wrapper):
            patched.append(attention.__name__)
        if self._patch_attr(ops, QWENDRIVE_MSDA_FORWARD_ATTR, msda_wrapper):
            patched.append(ops.__name__)
        if self._patch_attr(ops, QWENDRIVE_VOXEL_POOL_ATTR, voxel_wrapper):
            patched.append(ops.__name__)
        owner = getattr(view, QWENDRIVE_VIEW_CLASS)
        if self._descriptor is None:
            self._descriptor = (
                view.__name__,
                QWENDRIVE_VIEW_CLASS,
                getattr(owner, QWENDRIVE_FRUSTUM_ATTR),
            )
        setattr(
            owner,
            QWENDRIVE_FRUSTUM_ATTR,
            property(_qwendrive_frustum_getter),
        )
        patched.append(view.__name__)
        return patched


default_patcher = DefaultPatcher()
bevfusion_patcher = BEVFusionPatcher()
flashocc_patcher = FlashOCCPatcher()
sparsedrive_patcher = SparseDrivePatcher()
modulated_deform_conv_patcher = ModulatedDeformConv2dPatcher()
deform_conv_patcher = DeformConv2dPatcher()
bevfusion_iou3d_patcher = BEVFusionIou3dPatcher()
bevdepth_patcher = BEVDepthPatcher()
qwendrive_patcher = QwenDrivePatcher()

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
    "QwenDrivePatcher",
    "bevfusion_patcher",
    "default_patcher",
    "bevfusion_iou3d_patcher",
    "bevdepth_patcher",
    "flashocc_patcher",
    "modulated_deform_conv_patcher",
    "deform_conv_patcher",
    "sparsedrive_patcher",
    "qwendrive_patcher",
]
