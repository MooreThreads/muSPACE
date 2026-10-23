"""Public API for muSPACE."""

from importlib.metadata import PackageNotFoundError, version as package_version

from .compatibility import (
    CompatibilityReport,
    compatibility_report,
    enable_eval_batch_norm_reference,
)
from .ops import (
    DeformConv2d,
    DeformConv2dPack,
    ModulatedDeformConv2d,
    ModulatedDeformConv2dPack,
    bev_pool,
    bev_pool_reference,
    bev_pool_v2,
    bev_pool_v2_reference,
    boxes_iou_bev,
    boxes_iou_bev_reference,
    centerpoint_target_assign,
    centerpoint_target_assign_reference,
    nms_bev,
    nms_bev_normal,
    deformable_aggregation,
    deformable_aggregation_function,
    deformable_aggregation_reference,
    explain_dispatch,
    flash_bevpool,
    flash_bevpool_reference,
    flash_bevpool_sca,
    flash_bevpool_sca_reference,
    feature_maps_format,
    modulated_deform_conv2d,
    modulated_deform_conv2d_pack,
    multi_scale_deformable_attn,
    Voxelization,
    dynamic_voxelize,
    hard_voxelize,
    dynamic_voxelize_reference,
    hard_voxelize_reference,
    voxelization,
    voxel_pooling_train,
    voxel_pooling_train_reference,
    voxel_pool_depth,
    voxel_pool_depth_reference,
    points_in_boxes_all,
    points_in_boxes_all_reference,
    radius,
    radius_graph,
    radius_graph_reference,
    radius_reference,
)
from .runtime.kernel_dispatch import (
    DispatchReport,
    clear_dispatch_report,
    get_last_dispatch_report,
    get_dispatch_reports,
)


def _load_version() -> str:
    try:
        return package_version("mu-space")
    except PackageNotFoundError:
        try:
            from ._build_meta import __version__ as build_version

            return build_version
        except ModuleNotFoundError:
            return "0.0.0+unknown"


__version__ = _load_version()

try:
    from ._build_meta import __git_version__
except ModuleNotFoundError:
    __git_version__ = "unknown"

__all__ = [
    "CompatibilityReport",
    "DispatchReport",
    "DeformConv2d",
    "DeformConv2dPack",
    "ModulatedDeformConv2d",
    "ModulatedDeformConv2dPack",
    "__git_version__",
    "__version__",
    "bev_pool",
    "bev_pool_reference",
    "bev_pool_v2",
    "bev_pool_v2_reference",
    "boxes_iou_bev",
    "boxes_iou_bev_reference",
    "centerpoint_target_assign",
    "centerpoint_target_assign_reference",
    "nms_bev",
    "nms_bev_normal",
    "clear_dispatch_report",
    "compatibility_report",
    "deformable_aggregation",
    "deformable_aggregation_function",
    "deformable_aggregation_reference",
    "enable_eval_batch_norm_reference",
    "explain_dispatch",
    "flash_bevpool",
    "flash_bevpool_reference",
    "flash_bevpool_sca",
    "flash_bevpool_sca_reference",
    "feature_maps_format",
    "get_last_dispatch_report",
    "get_dispatch_reports",
    "modulated_deform_conv2d",
    "modulated_deform_conv2d_pack",
    "multi_scale_deformable_attn",
    "Voxelization",
    "dynamic_voxelize",
    "hard_voxelize",
    "dynamic_voxelize_reference",
    "hard_voxelize_reference",
    "voxelization",
    "voxel_pooling_train",
    "voxel_pooling_train_reference",
    "voxel_pool_depth",
    "voxel_pool_depth_reference",
    "points_in_boxes_all",
    "points_in_boxes_all_reference",
    "radius",
    "radius_graph",
    "radius_reference",
    "radius_graph_reference",
]
