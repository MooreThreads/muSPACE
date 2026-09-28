"""Legacy ``mmdet3d.ops.spconv.sparse_conv_ext`` compatibility exports."""

from .spconv import (
    get_indice_pairs_3d,
    get_indice_pairs_grid_3d,
    indice_conv_backward_fp32,
    indice_conv_backward_half,
    indice_conv_fp32,
    indice_conv_half,
)

__all__ = [
    "get_indice_pairs_3d",
    "get_indice_pairs_grid_3d",
    "indice_conv_fp32",
    "indice_conv_half",
    "indice_conv_backward_fp32",
    "indice_conv_backward_half",
]
