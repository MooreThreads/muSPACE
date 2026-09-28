#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::dim;
using muspace::ffi_utils::stream;

#include <cmath>

namespace {
constexpr int kThreads = 256;

__device__ inline bool project(
    const float* p, float x, float y, float z, float eps, float image_h, float image_w, float* u, float* v) {
  const float d = p[8] * x + p[9] * y + p[10] * z + p[11];
  if (d <= 0.0f) return false;
  const float safe_d = fmaxf(d, eps);
  *u                 = (p[0] * x + p[1] * y + p[2] * z + p[3]) / safe_d;
  *v                 = (p[4] * x + p[5] * y + p[6] * z + p[7]) / safe_d;
  return *u >= 0.0f && *u < image_w && *v >= 0.0f && *v < image_h;
}

__device__ inline float sample_bilinear(
    const float* value, int base, int h, int w, int c, int channel, float u, float v) {
  const int   u0  = static_cast<int>(floorf(u));
  const int   v0  = static_cast<int>(floorf(v));
  const int   u1  = u0 + 1;
  const int   v1  = v0 + 1;
  const float du  = u - u0;
  const float dv  = v - v0;
  const float f00 = (u0 >= 0 && u0 < w && v0 >= 0 && v0 < h) ? value[base + (v0 * w + u0) * c + channel] : 0.0f;
  const float f01 = (u0 >= 0 && u0 < w && v1 >= 0 && v1 < h) ? value[base + (v1 * w + u0) * c + channel] : 0.0f;
  const float f10 = (u1 >= 0 && u1 < w && v0 >= 0 && v0 < h) ? value[base + (v0 * w + u1) * c + channel] : 0.0f;
  const float f11 = (u1 >= 0 && u1 < w && v1 >= 0 && v1 < h) ? value[base + (v1 * w + u1) * c + channel] : 0.0f;
  return (1.0f - du) * (1.0f - dv) * f00 + (1.0f - du) * dv * f01 + du * (1.0f - dv) * f10 + du * dv * f11;
}

__global__ void flash_bevpool_sca_forward_kernel(int   channels,
                                                 int   cameras,
                                                 int   batch_size,
                                                 int   levels,
                                                 int   heads,
                                                 int   points,
                                                 float epsilon,
                                                 const float* __restrict__ value,
                                                 const float* __restrict__ projections,
                                                 const int* __restrict__ spatial_shapes,
                                                 const int* __restrict__ level_start,
                                                 const int* __restrict__ image_size,
                                                 const float* __restrict__ offsets,
                                                 const float* __restrict__ weights,
                                                 int   grid_x,
                                                 int   grid_y,
                                                 int   grid_z,
                                                 float voxel_size_x,
                                                 float voxel_size_y,
                                                 float voxel_size_z,
                                                 float voxel_x_min,
                                                 float voxel_y_min,
                                                 float voxel_z_min,
                                                 int   value_stride_n,
                                                 float* __restrict__ output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * grid_x * grid_y * channels;
  if (index >= total) return;
  const int   channel           = index % channels;
  const int   bev               = index / channels;
  const int   batch             = bev / (grid_x * grid_y);
  const int   cell              = bev % (grid_x * grid_y);
  const int   x_index           = cell % grid_x;
  const int   y_index           = cell / grid_x;
  const int   channels_per_head = channels / heads;
  const int   head              = channel / channels_per_head;
  const float x                 = voxel_x_min + (x_index + 0.5f) * voxel_size_x;
  const float y                 = voxel_y_min + (y_index + 0.5f) * voxel_size_y;
  const float image_h           = static_cast<float>(image_size[0]);
  const float image_w           = static_cast<float>(image_size[1]);
  const int   projection_base   = batch * cameras * 12;
  const int   off_hlpz2         = heads * levels * points * grid_z * 2;
  const int   off_lpz2          = levels * points * grid_z * 2;
  const int   off_pz2           = points * grid_z * 2;
  const int   off_z2            = grid_z * 2;
  const int   wgt_hlpz          = heads * levels * points * grid_z;
  const int   wgt_lpz           = levels * points * grid_z;
  const int   wgt_pz            = points * grid_z;
  const int   off_base =
      batch * grid_y * grid_x * off_hlpz2 + y_index * grid_x * off_hlpz2 + x_index * off_hlpz2 + head * off_lpz2;
  const int wgt_base =
      batch * grid_y * grid_x * wgt_hlpz + y_index * grid_x * wgt_hlpz + x_index * wgt_hlpz + head * wgt_lpz;

  int valid_cameras = 0;
  for (int camera = 0; camera < cameras; ++camera) {
    const float* p       = projections + projection_base + camera * 12;
    bool         visible = false;
    for (int z_index = 0; z_index < grid_z; ++z_index) {
      float       u = 0.0f, v = 0.0f;
      const float z = voxel_z_min + (z_index + 0.5f) * voxel_size_z;
      if (project(p, x, y, z, epsilon, image_h, image_w, &u, &v)) {
        visible = true;
        break;
      }
    }
    valid_cameras += visible;
  }
  const float inv_count = valid_cameras > 0 ? 1.0f / valid_cameras : 0.0f;
  float       result    = 0.0f;
  for (int camera = 0; camera < cameras; ++camera) {
    const float* p = projections + projection_base + camera * 12;
    for (int z_index = 0; z_index < grid_z; ++z_index) {
      float       image_u = 0.0f, image_v = 0.0f;
      const float z = voxel_z_min + (z_index + 0.5f) * voxel_size_z;
      if (!project(p, x, y, z, epsilon, image_h, image_w, &image_u, &image_v)) continue;
      for (int level = 0; level < levels; ++level) {
        const int   h          = spatial_shapes[level * 2];
        const int   w          = spatial_shapes[level * 2 + 1];
        const int   level_base = (batch * cameras + camera) * value_stride_n * channels + level_start[level] * channels;
        const float base_u     = image_u / image_w * w - 0.5f;
        const float base_v     = image_v / image_h * h - 0.5f;
        for (int point = 0; point < points; ++point) {
          const int   offset_index = off_base + level * off_pz2 + point * off_z2 + z_index * 2;
          const float u            = base_u + offsets[offset_index];
          const float v            = base_v + offsets[offset_index + 1];
          const float sampled      = sample_bilinear(value, level_base, h, w, channels, channel, u, v);
          const int   weight_index = wgt_base + level * wgt_pz + point * grid_z + z_index;
          result += weights[weight_index] * sampled;
        }
      }
    }
  }
  output[index] = result * inv_count;
}

__global__ void flash_bevpool_sca_backward_kernel(int   channels,
                                                  int   cameras,
                                                  int   batch_size,
                                                  int   levels,
                                                  int   heads,
                                                  int   points,
                                                  float epsilon,
                                                  const float* __restrict__ output_grad,
                                                  const float* __restrict__ value,
                                                  const float* __restrict__ projections,
                                                  const int* __restrict__ spatial_shapes,
                                                  const int* __restrict__ level_start,
                                                  const int* __restrict__ image_size,
                                                  const float* __restrict__ roi_range,
                                                  const float* __restrict__ offsets,
                                                  const float* __restrict__ weights,
                                                  int grid_x,
                                                  int grid_y,
                                                  int grid_z,
                                                  int value_stride_n,
                                                  float* __restrict__ value_grad,
                                                  float* __restrict__ offsets_grad,
                                                  float* __restrict__ weights_grad) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * grid_x * grid_y * channels;
  if (index >= total) return;
  const int   channel           = index % channels;
  const int   bev               = index / channels;
  const int   batch             = bev / (grid_x * grid_y);
  const int   cell              = bev % (grid_x * grid_y);
  const int   x_index           = cell % grid_x;
  const int   y_index           = cell / grid_x;
  const int   channels_per_head = channels / heads;
  const int   head              = channel / channels_per_head;
  const float size_x            = (roi_range[1] - roi_range[0]) / grid_x;
  const float size_y            = (roi_range[3] - roi_range[2]) / grid_y;
  const float size_z            = (roi_range[5] - roi_range[4]) / grid_z;
  const float x                 = roi_range[0] + (x_index + 0.5f) * size_x;
  const float y                 = roi_range[2] + (y_index + 0.5f) * size_y;
  const float image_h           = static_cast<float>(image_size[0]);
  const float image_w           = static_cast<float>(image_size[1]);
  const int   projection_base   = batch * cameras * 12;
  const float grad              = output_grad[index];
  const int   off_hlpz2         = heads * levels * points * grid_z * 2;
  const int   off_lpz2          = levels * points * grid_z * 2;
  const int   off_pz2           = points * grid_z * 2;
  const int   off_z2            = grid_z * 2;
  const int   wgt_hlpz          = heads * levels * points * grid_z;
  const int   wgt_lpz           = levels * points * grid_z;
  const int   wgt_pz            = points * grid_z;
  const int   off_base =
      batch * grid_y * grid_x * off_hlpz2 + y_index * grid_x * off_hlpz2 + x_index * off_hlpz2 + head * off_lpz2;
  const int wgt_base =
      batch * grid_y * grid_x * wgt_hlpz + y_index * grid_x * wgt_hlpz + x_index * wgt_hlpz + head * wgt_lpz;
  int valid_cameras = 0;
  for (int camera = 0; camera < cameras; ++camera) {
    const float* p       = projections + projection_base + camera * 12;
    bool         visible = false;
    for (int z_index = 0; z_index < grid_z; ++z_index) {
      float       u = 0.0f, v = 0.0f;
      const float z = roi_range[4] + (z_index + 0.5f) * size_z;
      if (project(p, x, y, z, epsilon, image_h, image_w, &u, &v)) {
        visible = true;
        break;
      }
    }
    valid_cameras += visible;
  }
  const float inv_count = valid_cameras > 0 ? 1.0f / valid_cameras : 0.0f;
  for (int camera = 0; camera < cameras; ++camera) {
    const float* p            = projections + projection_base + camera * 12;
    const int    batch_camera = batch * cameras + camera;
    for (int z_index = 0; z_index < grid_z; ++z_index) {
      float       image_u = 0.0f, image_v = 0.0f;
      const float z = roi_range[4] + (z_index + 0.5f) * size_z;
      if (!project(p, x, y, z, epsilon, image_h, image_w, &image_u, &image_v)) continue;
      for (int level = 0; level < levels; ++level) {
        const int   h          = spatial_shapes[level * 2];
        const int   w          = spatial_shapes[level * 2 + 1];
        const int   level_base = batch_camera * value_stride_n * channels + level_start[level] * channels;
        const float base_u     = image_u / image_w * w - 0.5f;
        const float base_v     = image_v / image_h * h - 0.5f;
        for (int point = 0; point < points; ++point) {
          const int   offset_index = off_base + level * off_pz2 + point * off_z2 + z_index * 2;
          const float sample_u     = base_u + offsets[offset_index];
          const float sample_v     = base_v + offsets[offset_index + 1];
          const int   u0           = static_cast<int>(floorf(sample_u));
          const int   v0           = static_cast<int>(floorf(sample_v));
          const int   u1           = u0 + 1;
          const int   v1           = v0 + 1;
          const float du           = sample_u - u0;
          const float dv           = sample_v - v0;
          const bool  b00          = u0 >= 0 && u0 < w && v0 >= 0 && v0 < h;
          const bool  b01          = u0 >= 0 && u0 < w && v1 >= 0 && v1 < h;
          const bool  b10          = u1 >= 0 && u1 < w && v0 >= 0 && v0 < h;
          const bool  b11          = u1 >= 0 && u1 < w && v1 >= 0 && v1 < h;
          const int   i00          = level_base + (v0 * w + u0) * channels + channel;
          const int   i01          = level_base + (v1 * w + u0) * channels + channel;
          const int   i10          = level_base + (v0 * w + u1) * channels + channel;
          const int   i11          = level_base + (v1 * w + u1) * channels + channel;
          const float f00          = b00 ? value[i00] : 0.0f;
          const float f01          = b01 ? value[i01] : 0.0f;
          const float f10          = b10 ? value[i10] : 0.0f;
          const float f11          = b11 ? value[i11] : 0.0f;
          const float sampled =
              (1.0f - du) * (1.0f - dv) * f00 + (1.0f - du) * dv * f01 + du * (1.0f - dv) * f10 + du * dv * f11;
          const int   weight_index = wgt_base + level * wgt_pz + point * grid_z + z_index;
          const float weight       = weights[weight_index];
          const float coeff        = grad * inv_count * weight;
          if (b00) atomicAdd(value_grad + i00, coeff * (1.0f - du) * (1.0f - dv));
          if (b01) atomicAdd(value_grad + i01, coeff * (1.0f - du) * dv);
          if (b10) atomicAdd(value_grad + i10, coeff * du * (1.0f - dv));
          if (b11) atomicAdd(value_grad + i11, coeff * du * dv);
          const float ds_du = (1.0f - dv) * (f10 - f00) + dv * (f11 - f01);
          const float ds_dv = (1.0f - du) * (f01 - f00) + du * (f11 - f10);
          atomicAdd(offsets_grad + offset_index, coeff * ds_du);
          atomicAdd(offsets_grad + offset_index + 1, coeff * ds_dv);
          atomicAdd(weights_grad + weight_index, grad * inv_count * sampled);
        }
      }
    }
  }
}

void check_float_sca(ffi::TensorView tensor, const char* name) {
  check_musa_contiguous(tensor, name, kDLFloat32);
}
void check_int_sca(ffi::TensorView tensor, const char* name) {
  check_musa_contiguous(tensor, name, kDLInt32);
}

}  // namespace

void flash_bevpool_sca_forward(ffi::TensorView value,
                               ffi::TensorView projection_matrices,
                               ffi::TensorView spatial_shapes,
                               ffi::TensorView level_start_index,
                               ffi::TensorView image_size,
                               ffi::TensorView sampling_offsets,
                               ffi::TensorView attention_weights,
                               ffi::TensorView out,
                               int             num_heads,
                               int             num_points,
                               int             grid_x,
                               int             grid_y,
                               int             grid_z,
                               float           voxel_size_x,
                               float           voxel_size_y,
                               float           voxel_size_z,
                               float           voxel_x_min,
                               float           voxel_y_min,
                               float           voxel_z_min,
                               int             value_stride_n,
                               float           epsilon) {
  check_float_sca(value, "value");
  check_float_sca(projection_matrices, "projection_matrices");
  check_float_sca(sampling_offsets, "sampling_offsets");
  check_float_sca(attention_weights, "attention_weights");
  check_float_sca(out, "out");
  check_int_sca(spatial_shapes, "spatial_shapes");
  check_int_sca(level_start_index, "level_start_index");
  check_int_sca(image_size, "image_size");
  TVM_FFI_ICHECK_EQ(value.ndim(), 4) << "value must have shape [B,N,total_HW,C]";
  TVM_FFI_ICHECK(num_heads > 0 && num_points > 0) << "num_heads and num_points must be positive";
  TVM_FFI_ICHECK_EQ(dim(value, 3) % num_heads, 0) << "channels must be divisible by num_heads";
  TVM_FFI_ICHECK(grid_x > 0 && grid_y > 0 && grid_z > 0) << "grid sizes must be positive";
  const int total = static_cast<int>(dim(value, 0) * grid_x * grid_y * dim(value, 3));
  if (total > 0) {
    flash_bevpool_sca_forward_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream(value)>>>(
        static_cast<int>(dim(value, 3)),
        static_cast<int>(dim(value, 1)),
        static_cast<int>(dim(value, 0)),
        static_cast<int>(dim(spatial_shapes, 0)),
        num_heads,
        num_points,
        epsilon,
        static_cast<float*>(value.data_ptr()),
        static_cast<float*>(projection_matrices.data_ptr()),
        static_cast<int*>(spatial_shapes.data_ptr()),
        static_cast<int*>(level_start_index.data_ptr()),
        static_cast<int*>(image_size.data_ptr()),
        static_cast<float*>(sampling_offsets.data_ptr()),
        static_cast<float*>(attention_weights.data_ptr()),
        grid_x,
        grid_y,
        grid_z,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        voxel_x_min,
        voxel_y_min,
        voxel_z_min,
        value_stride_n,
        static_cast<float*>(out.data_ptr()));
  }
  check_last_error("flash_bevpool_sca_forward");
}

void flash_bevpool_sca_backward(ffi::TensorView out_grad,
                                ffi::TensorView value,
                                ffi::TensorView projection_matrices,
                                ffi::TensorView spatial_shapes,
                                ffi::TensorView level_start_index,
                                ffi::TensorView image_size,
                                ffi::TensorView roi_range,
                                ffi::TensorView sampling_offsets,
                                ffi::TensorView attention_weights,
                                ffi::TensorView value_grad,
                                ffi::TensorView sampling_offsets_grad,
                                ffi::TensorView attention_weights_grad,
                                int             num_heads,
                                int             num_points,
                                int             grid_x,
                                int             grid_y,
                                int             grid_z,
                                int             value_stride_n,
                                float           epsilon) {
  check_float_sca(out_grad, "out_grad");
  check_float_sca(value, "value");
  check_float_sca(projection_matrices, "projection_matrices");
  check_float_sca(roi_range, "roi_range");
  check_float_sca(sampling_offsets, "sampling_offsets");
  check_float_sca(attention_weights, "attention_weights");
  check_float_sca(value_grad, "value_grad");
  check_float_sca(sampling_offsets_grad, "sampling_offsets_grad");
  check_float_sca(attention_weights_grad, "attention_weights_grad");
  check_int_sca(spatial_shapes, "spatial_shapes");
  check_int_sca(level_start_index, "level_start_index");
  check_int_sca(image_size, "image_size");
  TVM_FFI_ICHECK_EQ(roi_range.numel(), 6) << "roi_range must have shape [6]";
  TVM_FFI_ICHECK_EQ(dim(value, 3) % num_heads, 0) << "channels must be divisible by num_heads";
  const int total = static_cast<int>(dim(value, 0) * grid_x * grid_y * dim(value, 3));
  if (total > 0) {
    flash_bevpool_sca_backward_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream(out_grad)>>>(
        static_cast<int>(dim(value, 3)),
        static_cast<int>(dim(value, 1)),
        static_cast<int>(dim(value, 0)),
        static_cast<int>(dim(spatial_shapes, 0)),
        num_heads,
        num_points,
        epsilon,
        static_cast<float*>(out_grad.data_ptr()),
        static_cast<float*>(value.data_ptr()),
        static_cast<float*>(projection_matrices.data_ptr()),
        static_cast<int*>(spatial_shapes.data_ptr()),
        static_cast<int*>(level_start_index.data_ptr()),
        static_cast<int*>(image_size.data_ptr()),
        static_cast<float*>(roi_range.data_ptr()),
        static_cast<float*>(sampling_offsets.data_ptr()),
        static_cast<float*>(attention_weights.data_ptr()),
        grid_x,
        grid_y,
        grid_z,
        value_stride_n,
        static_cast<float*>(value_grad.data_ptr()),
        static_cast<float*>(sampling_offsets_grad.data_ptr()),
        static_cast<float*>(attention_weights_grad.data_ptr()));
  }
  check_last_error("flash_bevpool_sca_backward");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flash_bevpool_sca_forward, flash_bevpool_sca_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flash_bevpool_sca_backward, flash_bevpool_sca_backward);
