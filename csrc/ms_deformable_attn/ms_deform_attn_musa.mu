/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <musa.h>
#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include <vector>

#include "../ffi/tensor_utils.hpp"
#include "ms_deform_attn_musa_kernel.muh"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::dim;
using muspace::ffi_utils::stream;

namespace {
constexpr int kMsDeformAttnNumThreads              = 1024;
constexpr int kMsDeformAttnChannels32WarpsPerBlock = 8;
constexpr int kMsDeformAttnWarpSize                = 32;

inline int GetMsDeformAttnBlocks(const int n, const int num_threads) {
  return (n + num_threads - 1) / num_threads;
}

// bfloat16 is a storage dtype only: the kernels evaluate sampling geometry,
// the bilinear weights, the forward accumulator and the gradient accumulators
// in ms_deform_attn_acc_t (fp32).
inline DLDataType ms_deform_attn_data_dtype(const ffi::TensorView &value) {
  const DLDataType dtype = value.dtype();
  TVM_FFI_ICHECK(same_dtype(dtype, kDLFloat32) || same_dtype(dtype, kDLBFloat16))
      << "value must be float32 or bfloat16";
  return dtype;
}
}  // namespace

template <typename scalar_t>
void ms_deformable_im2col_musa(musaStream_t    stream,
                               const scalar_t *data_value,
                               const int64_t  *data_spatial_shapes,
                               const int64_t  *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int       batch_size,
                               const int       spatial_size,
                               const int       num_heads,
                               const int       channels,
                               const int       num_levels,
                               const int       num_query,
                               const int       num_point,
                               scalar_t       *data_col) {
  const int num_kernels        = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads        = kMsDeformAttnNumThreads;
  if (channels == 32 && num_levels == 1 && num_point == 4) {
    ms_deformable_im2col_gpu_kernel_channels32<scalar_t, 1, 4>
        <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                             data_value,
                                                                                             data_spatial_shapes,
                                                                                             data_level_start_index,
                                                                                             data_sampling_loc,
                                                                                             data_attn_weight,
                                                                                             batch_size,
                                                                                             spatial_size,
                                                                                             num_heads,
                                                                                             num_query,
                                                                                             data_col);
  } else if (channels == 32 && num_levels == 4 && num_point == 8) {
    ms_deformable_im2col_gpu_kernel_channels32_runtime<scalar_t>
        <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                             data_value,
                                                                                             data_spatial_shapes,
                                                                                             data_level_start_index,
                                                                                             data_sampling_loc,
                                                                                             data_attn_weight,
                                                                                             batch_size,
                                                                                             spatial_size,
                                                                                             num_heads,
                                                                                             num_levels,
                                                                                             num_query,
                                                                                             num_point,
                                                                                             data_col);
  } else {
    ms_deformable_im2col_gpu_kernel<scalar_t>
        <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                             data_value,
                                                                                             data_spatial_shapes,
                                                                                             data_level_start_index,
                                                                                             data_sampling_loc,
                                                                                             data_attn_weight,
                                                                                             batch_size,
                                                                                             spatial_size,
                                                                                             num_heads,
                                                                                             channels,
                                                                                             num_levels,
                                                                                             num_query,
                                                                                             num_point,
                                                                                             data_col);
  }

  check_last_error("ms_deformable_im2col_musa");
}

template <typename scalar_t, typename acc_t = ms_deform_attn_acc_t>
void ms_deformable_col2im_musa(musaStream_t    stream,
                               const scalar_t *grad_col,
                               const scalar_t *data_value,
                               const int64_t  *data_spatial_shapes,
                               const int64_t  *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int       batch_size,
                               const int       spatial_size,
                               const int       num_heads,
                               const int       channels,
                               const int       num_levels,
                               const int       num_query,
                               const int       num_point,
                               acc_t          *grad_value,
                               acc_t          *grad_sampling_loc,
                               acc_t          *grad_attn_weight) {
  const int num_threads        = (channels > kMsDeformAttnNumThreads) ? kMsDeformAttnNumThreads : channels;
  const int num_kernels        = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  if (channels > kMsDeformAttnNumThreads) {
    if ((channels & (kMsDeformAttnNumThreads - 1)) == 0) {
      ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
             num_threads,
             num_threads * 3 * sizeof(acc_t),
             stream>>>(num_kernels,
                       grad_col,
                       data_value,
                       data_spatial_shapes,
                       data_level_start_index,
                       data_sampling_loc,
                       data_attn_weight,
                       batch_size,
                       spatial_size,
                       num_heads,
                       channels,
                       num_levels,
                       num_query,
                       num_point,
                       grad_value,
                       grad_sampling_loc,
                       grad_attn_weight);
    } else {
      ms_deformable_col2im_gpu_kernel_gm<scalar_t>
          <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                               grad_col,
                                                                                               data_value,
                                                                                               data_spatial_shapes,
                                                                                               data_level_start_index,
                                                                                               data_sampling_loc,
                                                                                               data_attn_weight,
                                                                                               batch_size,
                                                                                               spatial_size,
                                                                                               num_heads,
                                                                                               channels,
                                                                                               num_levels,
                                                                                               num_query,
                                                                                               num_point,
                                                                                               grad_value,
                                                                                               grad_sampling_loc,
                                                                                               grad_attn_weight);
    }
  } else {
    switch (channels) {
      case 1:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                                 grad_col,
                                                                                                 data_value,
                                                                                                 data_spatial_shapes,
                                                                                                 data_level_start_index,
                                                                                                 data_sampling_loc,
                                                                                                 data_attn_weight,
                                                                                                 batch_size,
                                                                                                 spatial_size,
                                                                                                 num_heads,
                                                                                                 channels,
                                                                                                 num_levels,
                                                                                                 num_query,
                                                                                                 num_point,
                                                                                                 grad_value,
                                                                                                 grad_sampling_loc,
                                                                                                 grad_attn_weight);
        break;
      case 2:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                                 grad_col,
                                                                                                 data_value,
                                                                                                 data_spatial_shapes,
                                                                                                 data_level_start_index,
                                                                                                 data_sampling_loc,
                                                                                                 data_attn_weight,
                                                                                                 batch_size,
                                                                                                 spatial_size,
                                                                                                 num_heads,
                                                                                                 channels,
                                                                                                 num_levels,
                                                                                                 num_query,
                                                                                                 num_point,
                                                                                                 grad_value,
                                                                                                 grad_sampling_loc,
                                                                                                 grad_attn_weight);
        break;
      case 4:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                                 grad_col,
                                                                                                 data_value,
                                                                                                 data_spatial_shapes,
                                                                                                 data_level_start_index,
                                                                                                 data_sampling_loc,
                                                                                                 data_attn_weight,
                                                                                                 batch_size,
                                                                                                 spatial_size,
                                                                                                 num_heads,
                                                                                                 channels,
                                                                                                 num_levels,
                                                                                                 num_query,
                                                                                                 num_point,
                                                                                                 grad_value,
                                                                                                 grad_sampling_loc,
                                                                                                 grad_attn_weight);
        break;
      case 8:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                                 grad_col,
                                                                                                 data_value,
                                                                                                 data_spatial_shapes,
                                                                                                 data_level_start_index,
                                                                                                 data_sampling_loc,
                                                                                                 data_attn_weight,
                                                                                                 batch_size,
                                                                                                 spatial_size,
                                                                                                 num_heads,
                                                                                                 channels,
                                                                                                 num_levels,
                                                                                                 num_query,
                                                                                                 num_point,
                                                                                                 grad_value,
                                                                                                 grad_sampling_loc,
                                                                                                 grad_attn_weight);
        break;
      case 16:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                                 grad_col,
                                                                                                 data_value,
                                                                                                 data_spatial_shapes,
                                                                                                 data_level_start_index,
                                                                                                 data_sampling_loc,
                                                                                                 data_attn_weight,
                                                                                                 batch_size,
                                                                                                 spatial_size,
                                                                                                 num_heads,
                                                                                                 channels,
                                                                                                 num_levels,
                                                                                                 num_query,
                                                                                                 num_point,
                                                                                                 grad_value,
                                                                                                 grad_sampling_loc,
                                                                                                 grad_attn_weight);
        break;
      case 32:
        ms_deformable_col2im_gpu_kernel_warp_broadcast_multi_warp_channels32<scalar_t,
                                                                             kMsDeformAttnChannels32WarpsPerBlock>
            <<<GetMsDeformAttnBlocks(num_actual_kernels / kMsDeformAttnWarpSize, kMsDeformAttnChannels32WarpsPerBlock),
               kMsDeformAttnChannels32WarpsPerBlock * kMsDeformAttnWarpSize,
               0,
               stream>>>(num_kernels,
                         grad_col,
                         data_value,
                         data_spatial_shapes,
                         data_level_start_index,
                         data_sampling_loc,
                         data_attn_weight,
                         batch_size,
                         spatial_size,
                         num_heads,
                         channels,
                         num_levels,
                         num_query,
                         num_point,
                         grad_value,
                         grad_sampling_loc,
                         grad_attn_weight);
        break;
      case 64:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                                 grad_col,
                                                                                                 data_value,
                                                                                                 data_spatial_shapes,
                                                                                                 data_level_start_index,
                                                                                                 data_sampling_loc,
                                                                                                 data_attn_weight,
                                                                                                 batch_size,
                                                                                                 spatial_size,
                                                                                                 num_heads,
                                                                                                 channels,
                                                                                                 num_levels,
                                                                                                 num_query,
                                                                                                 num_point,
                                                                                                 grad_value,
                                                                                                 grad_sampling_loc,
                                                                                                 grad_attn_weight);
        break;
      case 128:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                                 grad_col,
                                                                                                 data_value,
                                                                                                 data_spatial_shapes,
                                                                                                 data_level_start_index,
                                                                                                 data_sampling_loc,
                                                                                                 data_attn_weight,
                                                                                                 batch_size,
                                                                                                 spatial_size,
                                                                                                 num_heads,
                                                                                                 channels,
                                                                                                 num_levels,
                                                                                                 num_query,
                                                                                                 num_point,
                                                                                                 grad_value,
                                                                                                 grad_sampling_loc,
                                                                                                 grad_attn_weight);
        break;
      case 256:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                                 grad_col,
                                                                                                 data_value,
                                                                                                 data_spatial_shapes,
                                                                                                 data_level_start_index,
                                                                                                 data_sampling_loc,
                                                                                                 data_attn_weight,
                                                                                                 batch_size,
                                                                                                 spatial_size,
                                                                                                 num_heads,
                                                                                                 channels,
                                                                                                 num_levels,
                                                                                                 num_query,
                                                                                                 num_point,
                                                                                                 grad_value,
                                                                                                 grad_sampling_loc,
                                                                                                 grad_attn_weight);
        break;
      case 512:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                                 grad_col,
                                                                                                 data_value,
                                                                                                 data_spatial_shapes,
                                                                                                 data_level_start_index,
                                                                                                 data_sampling_loc,
                                                                                                 data_attn_weight,
                                                                                                 batch_size,
                                                                                                 spatial_size,
                                                                                                 num_heads,
                                                                                                 channels,
                                                                                                 num_levels,
                                                                                                 num_query,
                                                                                                 num_point,
                                                                                                 grad_value,
                                                                                                 grad_sampling_loc,
                                                                                                 grad_attn_weight);
        break;
      case 1024:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                                 grad_col,
                                                                                                 data_value,
                                                                                                 data_spatial_shapes,
                                                                                                 data_level_start_index,
                                                                                                 data_sampling_loc,
                                                                                                 data_attn_weight,
                                                                                                 batch_size,
                                                                                                 spatial_size,
                                                                                                 num_heads,
                                                                                                 channels,
                                                                                                 num_levels,
                                                                                                 num_query,
                                                                                                 num_point,
                                                                                                 grad_value,
                                                                                                 grad_sampling_loc,
                                                                                                 grad_attn_weight);
        break;
      default:
        if (channels < 64) {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
              <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
                 num_threads,
                 num_threads * 3 * sizeof(acc_t),
                 stream>>>(num_kernels,
                           grad_col,
                           data_value,
                           data_spatial_shapes,
                           data_level_start_index,
                           data_sampling_loc,
                           data_attn_weight,
                           batch_size,
                           spatial_size,
                           num_heads,
                           channels,
                           num_levels,
                           num_query,
                           num_point,
                           grad_value,
                           grad_sampling_loc,
                           grad_attn_weight);
        } else {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
              <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
                 num_threads,
                 num_threads * 3 * sizeof(acc_t),
                 stream>>>(num_kernels,
                           grad_col,
                           data_value,
                           data_spatial_shapes,
                           data_level_start_index,
                           data_sampling_loc,
                           data_attn_weight,
                           batch_size,
                           spatial_size,
                           num_heads,
                           channels,
                           num_levels,
                           num_query,
                           num_point,
                           grad_value,
                           grad_sampling_loc,
                           grad_attn_weight);
        }
    }
  }
  check_last_error("ms_deformable_col2im_musa");
}

template <typename scalar_t>
void ms_deformable_im2col_scatter_div_musa(musaStream_t    stream,
                                           const scalar_t *data_value,
                                           const int64_t  *data_spatial_shapes,
                                           const int64_t  *data_level_start_index,
                                           const scalar_t *data_sampling_loc,
                                           const scalar_t *data_attn_weight,
                                           const int64_t  *data_index,
                                           const scalar_t *data_count,
                                           const int       batch_size,
                                           const int       num_cams,
                                           const int       cam_id,
                                           const int       valid_len,
                                           const int       spatial_size,
                                           const int       num_heads,
                                           const int       channels,
                                           const int       num_levels,
                                           const int       max_len,
                                           const int       num_point,
                                           const int       num_query,
                                           scalar_t       *data_slots) {
  if (valid_len == 0) {
    return;
  }
  const int num_kernels = batch_size * valid_len * num_heads * channels;
  const int num_threads = kMsDeformAttnNumThreads;
  if (channels == 32) {
    ms_deformable_im2col_scatter_div_gpu_kernel_channels32_runtime<scalar_t>
        <<<GetMsDeformAttnBlocks(num_kernels, num_threads), num_threads, 0, stream>>>(num_kernels,
                                                                                      data_value,
                                                                                      data_spatial_shapes,
                                                                                      data_level_start_index,
                                                                                      data_sampling_loc,
                                                                                      data_attn_weight,
                                                                                      data_index,
                                                                                      data_count,
                                                                                      batch_size,
                                                                                      num_cams,
                                                                                      cam_id,
                                                                                      spatial_size,
                                                                                      num_heads,
                                                                                      num_levels,
                                                                                      valid_len,
                                                                                      max_len,
                                                                                      num_point,
                                                                                      num_query,
                                                                                      data_slots);
  } else {
    printf("unsupported channels in ms_deformable_im2col_scatter_div_musa: %d\n", channels);
  }
}

void ms_deform_attn_forward(ffi::TensorView value,
                            ffi::TensorView spatial,
                            ffi::TensorView start,
                            ffi::TensorView loc,
                            ffi::TensorView weight,
                            ffi::TensorView out,
                            int             im2col_step) {
  const DLDataType data_dtype = ms_deform_attn_data_dtype(value);
  check_musa_contiguous(value, "value", data_dtype);
  check_musa_contiguous(spatial, "spatial_shapes", kDLInt64);
  check_musa_contiguous(start, "level_start_index", kDLInt64);
  check_musa_contiguous(loc, "sampling_locations", data_dtype);
  check_musa_contiguous(weight, "attention_weights", data_dtype);
  check_musa_contiguous(out, "out", data_dtype);
  const int batch = dim(value, 0), spatial_size = dim(value, 1), heads = dim(value, 2), channels = dim(value, 3),
            levels = dim(spatial, 0), queries = dim(loc, 1), points = dim(loc, 4), step = std::min(batch, im2col_step);
  TVM_FFI_ICHECK_GT(step, 0);
  TVM_FFI_ICHECK_EQ(batch % step, 0);
  const int64_t vs = spatial_size * heads * channels, ls = queries * heads * levels * points * 2,
                ws = queries * heads * levels * points, os = queries * heads * channels;
  for (int n = 0; n < batch / step; ++n) {
    if (same_dtype(data_dtype, kDLBFloat16)) {
      ms_deformable_im2col_musa<bf16_t>(stream(value),
                                        static_cast<const bf16_t *>(value.data_ptr()) + n * step * vs,
                                        static_cast<const int64_t *>(spatial.data_ptr()),
                                        static_cast<const int64_t *>(start.data_ptr()),
                                        static_cast<const bf16_t *>(loc.data_ptr()) + n * step * ls,
                                        static_cast<const bf16_t *>(weight.data_ptr()) + n * step * ws,
                                        step,
                                        spatial_size,
                                        heads,
                                        channels,
                                        levels,
                                        queries,
                                        points,
                                        static_cast<bf16_t *>(out.data_ptr()) + n * step * os);
    } else {
      ms_deformable_im2col_musa(stream(value),
                                (float *)value.data_ptr() + n * step * vs,
                                (int64_t *)spatial.data_ptr(),
                                (int64_t *)start.data_ptr(),
                                (float *)loc.data_ptr() + n * step * ls,
                                (float *)weight.data_ptr() + n * step * ws,
                                step,
                                spatial_size,
                                heads,
                                channels,
                                levels,
                                queries,
                                points,
                                (float *)out.data_ptr() + n * step * os);
    }
  }
}
void ms_deform_attn_backward(ffi::TensorView value,
                             ffi::TensorView spatial,
                             ffi::TensorView start,
                             ffi::TensorView loc,
                             ffi::TensorView weight,
                             ffi::TensorView grad,
                             ffi::TensorView gv,
                             ffi::TensorView gl,
                             ffi::TensorView gw,
                             int             im2col_step) {
  const DLDataType data_dtype = ms_deform_attn_data_dtype(value);
  check_musa_contiguous(value, "value", data_dtype);
  check_musa_contiguous(spatial, "spatial_shapes", kDLInt64);
  check_musa_contiguous(start, "level_start_index", kDLInt64);
  check_musa_contiguous(loc, "sampling_locations", data_dtype);
  check_musa_contiguous(weight, "attention_weights", data_dtype);
  check_musa_contiguous(grad, "grad", data_dtype);
  check_musa_contiguous(gv, "grad_value", kDLFloat32);
  check_musa_contiguous(gl, "grad_sampling_locations", kDLFloat32);
  check_musa_contiguous(gw, "grad_attention_weights", kDLFloat32);
  const int batch = dim(value, 0), spatial_size = dim(value, 1), heads = dim(value, 2), channels = dim(value, 3),
            levels = dim(spatial, 0), queries = dim(loc, 1), points = dim(loc, 4), step = std::min(batch, im2col_step);
  TVM_FFI_ICHECK_GT(step, 0);
  TVM_FFI_ICHECK_EQ(batch % step, 0);
  const int64_t vs = spatial_size * heads * channels, ls = queries * heads * levels * points * 2,
                ws = queries * heads * levels * points, os = queries * heads * channels;
  for (int n = 0; n < batch / step; ++n) {
    if (same_dtype(data_dtype, kDLBFloat16)) {
      ms_deformable_col2im_musa<bf16_t>(stream(value),
                                        static_cast<const bf16_t *>(grad.data_ptr()) + n * step * os,
                                        static_cast<const bf16_t *>(value.data_ptr()) + n * step * vs,
                                        static_cast<const int64_t *>(spatial.data_ptr()),
                                        static_cast<const int64_t *>(start.data_ptr()),
                                        static_cast<const bf16_t *>(loc.data_ptr()) + n * step * ls,
                                        static_cast<const bf16_t *>(weight.data_ptr()) + n * step * ws,
                                        step,
                                        spatial_size,
                                        heads,
                                        channels,
                                        levels,
                                        queries,
                                        points,
                                        (float *)gv.data_ptr() + n * step * vs,
                                        (float *)gl.data_ptr() + n * step * ls,
                                        (float *)gw.data_ptr() + n * step * ws);
    } else {
      ms_deformable_col2im_musa(stream(value),
                                (float *)grad.data_ptr() + n * step * os,
                                (float *)value.data_ptr() + n * step * vs,
                                (int64_t *)spatial.data_ptr(),
                                (int64_t *)start.data_ptr(),
                                (float *)loc.data_ptr() + n * step * ls,
                                (float *)weight.data_ptr() + n * step * ws,
                                step,
                                spatial_size,
                                heads,
                                channels,
                                levels,
                                queries,
                                points,
                                (float *)gv.data_ptr() + n * step * vs,
                                (float *)gl.data_ptr() + n * step * ls,
                                (float *)gw.data_ptr() + n * step * ws);
    }
  }
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ms_deform_attn_forward, ms_deform_attn_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ms_deform_attn_backward, ms_deform_attn_backward);
