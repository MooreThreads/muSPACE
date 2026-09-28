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

constexpr int kThreads               = 256;
constexpr int kBackwardThreads       = 256;
constexpr int kBackwardCachedCameras = 8;

__device__ inline int clamp_int(const int value, const int low, const int high) {
  return value < low ? low : (value > high ? high : value);
}

__device__ inline bool project_point(const float* projection,
                                     const float  x,
                                     const float  y,
                                     const float  z,
                                     const float  epsilon,
                                     const float  image_h,
                                     const float  image_w,
                                     float*       u,
                                     float*       v) {
  const float depth = projection[8] * x + projection[9] * y + projection[10] * z + projection[11];
  if (depth <= 0.0f) {
    return false;
  }
  const float numerator_u = projection[0] * x + projection[1] * y + projection[2] * z + projection[3];
  const float numerator_v = projection[4] * x + projection[5] * y + projection[6] * z + projection[7];
  const float safe_depth  = fmaxf(depth, epsilon);
  *u                      = numerator_u / safe_depth;
  *v                      = numerator_v / safe_depth;
  return *u >= 0.0f && *u < image_w && *v >= 0.0f && *v < image_h;
}

__global__ void flash_bevpool_forward_kernel(int   channels,
                                             int   cameras,
                                             int   batch_size,
                                             float epsilon,
                                             const float* __restrict__ image_feats,
                                             const float* __restrict__ projections,
                                             const int* __restrict__ feature_size,
                                             const int* __restrict__ image_size,
                                             int   grid_x,
                                             int   grid_y,
                                             int   grid_z,
                                             float voxel_size_x,
                                             float voxel_size_y,
                                             float voxel_size_z,
                                             float voxel_x_min,
                                             float voxel_y_min,
                                             float voxel_z_min,
                                             float* __restrict__ output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * grid_x * grid_y * channels;
  if (index >= total) return;
  const int   channel         = index % channels;
  const int   bev             = index / channels;
  const int   batch           = bev / (grid_x * grid_y);
  const int   cell            = bev % (grid_x * grid_y);
  const int   x_index         = cell % grid_x;
  const int   y_index         = cell / grid_x;
  const float x               = voxel_x_min + (x_index + 0.5f) * voxel_size_x;
  const float y               = voxel_y_min + (y_index + 0.5f) * voxel_size_y;
  const float image_h         = static_cast<float>(image_size[0]);
  const float image_w         = static_cast<float>(image_size[1]);
  const float inv_image_h     = 1.0f / image_h;
  const float inv_image_w     = 1.0f / image_w;
  const int   projection_base = batch * cameras * 12;

  float result = 0.0f;
  for (int z_index = 0; z_index < grid_z; ++z_index) {
    const float z             = voxel_z_min + (z_index + 0.5f) * voxel_size_z;
    float       sum           = 0.0f;
    int         valid_cameras = 0;
    for (int camera = 0; camera < cameras; ++camera) {
      const float* projection = projections + projection_base + camera * 12;
      float        image_u = 0.0f, image_v = 0.0f;
      if (!project_point(projection, x, y, z, epsilon, image_h, image_w, &image_u, &image_v)) continue;
      const int   feature_h = feature_size[(batch * cameras + camera) * 2];
      const int   feature_w = feature_size[(batch * cameras + camera) * 2 + 1];
      const float feature_u = image_u * inv_image_w * feature_w;
      const float feature_v = image_v * inv_image_h * feature_h;
      if (feature_u < 0.0f || feature_u >= feature_w || feature_v < 0.0f || feature_v >= feature_h) continue;
      const int   u0_raw       = static_cast<int>(floorf(feature_u));
      const int   v0_raw       = static_cast<int>(floorf(feature_v));
      const float du           = feature_u - u0_raw;
      const float dv           = feature_v - v0_raw;
      const int   u0           = clamp_int(u0_raw, 0, feature_w - 1);
      const int   u1           = clamp_int(u0_raw + 1, 0, feature_w - 1);
      const int   v0           = clamp_int(v0_raw, 0, feature_h - 1);
      const int   v1           = clamp_int(v0_raw + 1, 0, feature_h - 1);
      const float w00          = (1.0f - du) * (1.0f - dv);
      const float w01          = (1.0f - du) * dv;
      const float w10          = du * (1.0f - dv);
      const float w11          = du * dv;
      const int   feature_base = (batch * cameras + camera) * feature_h * feature_w * channels;
      const int   row0         = feature_base + v0 * feature_w * channels;
      const int   row1         = feature_base + v1 * feature_w * channels;
      const float sampled =
          w00 * image_feats[row0 + u0 * channels + channel] + w01 * image_feats[row1 + u0 * channels + channel] +
          w10 * image_feats[row0 + u1 * channels + channel] + w11 * image_feats[row1 + u1 * channels + channel];
      sum += sampled;
      ++valid_cameras;
    }
    if (valid_cameras > 0) result += sum / static_cast<float>(valid_cameras);
  }

  output[index] = result;
}

__global__ void flash_bevpool_geometry_kernel(int   channels,
                                              int   cameras,
                                              int   batch_size,
                                              float epsilon,
                                              const float* __restrict__ projections,
                                              const int* __restrict__ feature_size,
                                              const int* __restrict__ image_size,
                                              int   grid_x,
                                              int   grid_y,
                                              int   grid_z,
                                              float voxel_size_x,
                                              float voxel_size_y,
                                              float voxel_size_z,
                                              float voxel_x_min,
                                              float voxel_y_min,
                                              float voxel_z_min,
                                              int* __restrict__ indices,
                                              float* __restrict__ weights) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int total  = batch_size * grid_x * grid_y * grid_z * cameras;
  if (sample >= total) return;
  const int    camera     = sample % cameras;
  const int    z_index    = (sample / cameras) % grid_z;
  const int    bev        = sample / (cameras * grid_z);
  const int    batch      = bev / (grid_x * grid_y);
  const int    cell       = bev % (grid_x * grid_y);
  const int    x_index    = cell % grid_x;
  const int    y_index    = cell / grid_x;
  const float  x          = voxel_x_min + (x_index + 0.5f) * voxel_size_x;
  const float  y          = voxel_y_min + (y_index + 0.5f) * voxel_size_y;
  const float  z          = voxel_z_min + (z_index + 0.5f) * voxel_size_z;
  const float  image_h    = static_cast<float>(image_size[0]);
  const float  image_w    = static_cast<float>(image_size[1]);
  const float* projection = projections + (batch * cameras + camera) * 12;
  float        image_u = 0.0f, image_v = 0.0f;
  int*         sample_indices = indices + sample * 4;
  if (!project_point(projection, x, y, z, epsilon, image_h, image_w, &image_u, &image_v)) {
    sample_indices[0] = -1;
    return;
  }
  const int   feature_h = feature_size[(batch * cameras + camera) * 2];
  const int   feature_w = feature_size[(batch * cameras + camera) * 2 + 1];
  const float feature_u = image_u / image_w * feature_w;
  const float feature_v = image_v / image_h * feature_h;
  if (feature_u < 0.0f || feature_u >= feature_w || feature_v < 0.0f || feature_v >= feature_h) {
    sample_indices[0] = -1;
    return;
  }
  const int   u0_raw    = static_cast<int>(floorf(feature_u));
  const int   v0_raw    = static_cast<int>(floorf(feature_v));
  const float du        = feature_u - u0_raw;
  const float dv        = feature_v - v0_raw;
  const int   u0        = clamp_int(u0_raw, 0, feature_w - 1);
  const int   u1        = clamp_int(u0_raw + 1, 0, feature_w - 1);
  const int   v0        = clamp_int(v0_raw, 0, feature_h - 1);
  const int   v1        = clamp_int(v0_raw + 1, 0, feature_h - 1);
  const int   base      = (batch * cameras + camera) * feature_h * feature_w * channels;
  sample_indices[0]     = base + (v0 * feature_w + u0) * channels;
  sample_indices[1]     = base + (v1 * feature_w + u0) * channels;
  sample_indices[2]     = base + (v0 * feature_w + u1) * channels;
  sample_indices[3]     = base + (v1 * feature_w + u1) * channels;
  float* sample_weights = weights + sample * 4;
  sample_weights[0]     = (1.0f - du) * (1.0f - dv);
  sample_weights[1]     = (1.0f - du) * dv;
  sample_weights[2]     = du * (1.0f - dv);
  sample_weights[3]     = du * dv;
}

__global__ void flash_bevpool_cached_forward_kernel(int channels,
                                                    int cameras,
                                                    int batch_size,
                                                    const float* __restrict__ image_feats,
                                                    const int* __restrict__ indices,
                                                    const float* __restrict__ weights,
                                                    int grid_x,
                                                    int grid_y,
                                                    int grid_z,
                                                    float* __restrict__ output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * grid_x * grid_y * channels;
  if (index >= total) return;
  const int channel = index % channels;
  const int bev     = index / channels;
  float     result  = 0.0f;
  for (int z_index = 0; z_index < grid_z; ++z_index) {
    float sum           = 0.0f;
    int   valid_cameras = 0;
    for (int camera = 0; camera < cameras; ++camera) {
      const int  sample         = (bev * grid_z + z_index) * cameras + camera;
      const int* sample_indices = indices + sample * 4;
      if (sample_indices[0] < 0) continue;
      const float* sample_weights = weights + sample * 4;
      sum += sample_weights[0] * image_feats[sample_indices[0] + channel] +
             sample_weights[1] * image_feats[sample_indices[1] + channel] +
             sample_weights[2] * image_feats[sample_indices[2] + channel] +
             sample_weights[3] * image_feats[sample_indices[3] + channel];
      ++valid_cameras;
    }
    if (valid_cameras > 0) result += sum / static_cast<float>(valid_cameras);
  }
  output[index] = result;
}

template <int kCachedCameras>
__global__ void flash_bevpool_backward_kernel(int   channels,
                                              int   cameras,
                                              int   batch_size,
                                              float epsilon,
                                              const float* __restrict__ output_grad,
                                              const float* __restrict__ projections,
                                              const int* __restrict__ feature_size,
                                              const int* __restrict__ image_size,
                                              const float* __restrict__ roi_range,
                                              int grid_x,
                                              int grid_y,
                                              int grid_z,
                                              float* __restrict__ image_feats_grad) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * grid_x * grid_y * channels;
  if (index >= total) {
    return;
  }

  const int   channel         = index % channels;
  const int   bev             = index / channels;
  const int   batch           = bev / (grid_x * grid_y);
  const int   cell            = bev % (grid_x * grid_y);
  const int   x_index         = cell % grid_x;
  const int   y_index         = cell / grid_x;
  const float size_x          = (roi_range[1] - roi_range[0]) / grid_x;
  const float size_y          = (roi_range[3] - roi_range[2]) / grid_y;
  const float size_z          = (roi_range[5] - roi_range[4]) / grid_z;
  const float x               = roi_range[0] + (x_index + 0.5f) * size_x;
  const float y               = roi_range[2] + (y_index + 0.5f) * size_y;
  const float image_h         = static_cast<float>(image_size[0]);
  const float image_w         = static_cast<float>(image_size[1]);
  const float inv_image_h     = 1.0f / image_h;
  const float inv_image_w     = 1.0f / image_w;
  const int   projection_base = batch * cameras * 12;
  const float grad            = output_grad[index];
  const bool  cache_geometry  = cameras <= kCachedCameras;

  for (int z_index = 0; z_index < grid_z; ++z_index) {
    const float  z             = roi_range[4] + (z_index + 0.5f) * size_z;
    int          valid_cameras = 0;
    unsigned int valid_mask    = 0;
    float        feature_u_cache[kCachedCameras];
    float        feature_v_cache[kCachedCameras];
    for (int count_camera = 0; count_camera < cameras; ++count_camera) {
      float        count_u          = 0.0f;
      float        count_v          = 0.0f;
      const float* count_projection = projections + projection_base + count_camera * 12;
      if (project_point(count_projection, x, y, z, epsilon, image_h, image_w, &count_u, &count_v)) {
        const int   count_h = feature_size[(batch * cameras + count_camera) * 2];
        const int   count_w = feature_size[(batch * cameras + count_camera) * 2 + 1];
        const float count_x = count_u * inv_image_w * count_w;
        const float count_y = count_v * inv_image_h * count_h;
        const bool  valid   = count_x >= 0.0f && count_x < count_w && count_y >= 0.0f && count_y < count_h;
        valid_cameras += valid;
        if (cache_geometry && valid) {
          valid_mask |= 1u << count_camera;
          feature_u_cache[count_camera] = count_x;
          feature_v_cache[count_camera] = count_y;
        }
      }
    }
    if (valid_cameras == 0) continue;
    const float scale = grad / static_cast<float>(valid_cameras);
    for (int camera = 0; camera < cameras; ++camera) {
      const int feature_h = feature_size[(batch * cameras + camera) * 2];
      const int feature_w = feature_size[(batch * cameras + camera) * 2 + 1];
      float     feature_u = 0.0f;
      float     feature_v = 0.0f;
      if (cache_geometry) {
        if ((valid_mask & (1u << camera)) == 0) continue;
        feature_u = feature_u_cache[camera];
        feature_v = feature_v_cache[camera];
      } else {
        const float* projection = projections + projection_base + camera * 12;
        float        image_u    = 0.0f;
        float        image_v    = 0.0f;
        if (!project_point(projection, x, y, z, epsilon, image_h, image_w, &image_u, &image_v)) continue;
        feature_u = image_u * inv_image_w * feature_w;
        feature_v = image_v * inv_image_h * feature_h;
        if (feature_u < 0.0f || feature_u >= feature_w || feature_v < 0.0f || feature_v >= feature_h) continue;
      }
      const int   u0_raw       = static_cast<int>(floorf(feature_u));
      const int   v0_raw       = static_cast<int>(floorf(feature_v));
      const int   u0           = clamp_int(u0_raw, 0, feature_w - 1);
      const int   v0           = clamp_int(v0_raw, 0, feature_h - 1);
      const int   u1           = clamp_int(u0 + 1, 0, feature_w - 1);
      const int   v1           = clamp_int(v0 + 1, 0, feature_h - 1);
      const float du           = feature_u - u0_raw;
      const float dv           = feature_v - v0_raw;
      const int   feature_base = (batch * cameras + camera) * feature_h * feature_w * channels;
      const int   row0         = feature_base + v0 * feature_w * channels;
      const int   row1         = feature_base + v1 * feature_w * channels;
      atomicAdd(image_feats_grad + row0 + u0 * channels + channel, scale * (1.0f - du) * (1.0f - dv));
      atomicAdd(image_feats_grad + row1 + u0 * channels + channel, scale * (1.0f - du) * dv);
      atomicAdd(image_feats_grad + row0 + u1 * channels + channel, scale * du * (1.0f - dv));
      atomicAdd(image_feats_grad + row1 + u1 * channels + channel, scale * du * dv);
    }
  }
}

__global__ void flash_bevpool_backward_uv_kernel(int   cameras,
                                                 int   batch_size,
                                                 float epsilon,
                                                 const float* __restrict__ projections,
                                                 const int* __restrict__ feature_size,
                                                 const int* __restrict__ image_size,
                                                 const float* __restrict__ roi_range,
                                                 int grid_x,
                                                 int grid_y,
                                                 int grid_z,
                                                 float2* __restrict__ coordinates) {
  const int sample = blockIdx.x * blockDim.x + threadIdx.x;
  const int total  = batch_size * grid_x * grid_y * grid_z * cameras;
  if (sample >= total) return;
  const int    camera     = sample % cameras;
  const int    z_index    = (sample / cameras) % grid_z;
  const int    bev        = sample / (cameras * grid_z);
  const int    batch      = bev / (grid_x * grid_y);
  const int    cell       = bev % (grid_x * grid_y);
  const float  x          = roi_range[0] + (cell % grid_x + 0.5f) * (roi_range[1] - roi_range[0]) / grid_x;
  const float  y          = roi_range[2] + (cell / grid_x + 0.5f) * (roi_range[3] - roi_range[2]) / grid_y;
  const float  z          = roi_range[4] + (z_index + 0.5f) * (roi_range[5] - roi_range[4]) / grid_z;
  const float  image_h    = static_cast<float>(image_size[0]);
  const float  image_w    = static_cast<float>(image_size[1]);
  const float* projection = projections + (batch * cameras + camera) * 12;
  float        image_u = 0.0f, image_v = 0.0f;
  if (!project_point(projection, x, y, z, epsilon, image_h, image_w, &image_u, &image_v)) {
    coordinates[sample] = make_float2(-1.0f, -1.0f);
    return;
  }
  const int   feature_h = feature_size[(batch * cameras + camera) * 2];
  const int   feature_w = feature_size[(batch * cameras + camera) * 2 + 1];
  const float feature_u = image_u / image_w * feature_w;
  const float feature_v = image_v / image_h * feature_h;
  coordinates[sample]   = (feature_u >= 0.0f && feature_u < feature_w && feature_v >= 0.0f && feature_v < feature_h)
                              ? make_float2(feature_u, feature_v)
                              : make_float2(-1.0f, -1.0f);
}

__global__ void flash_bevpool_backward_uv_cached_kernel(int channels,
                                                        int cameras,
                                                        int batch_size,
                                                        const float* __restrict__ out_grad,
                                                        const int* __restrict__ feature_size,
                                                        const float2* __restrict__ coordinates,
                                                        const int* __restrict__ valid_counts,
                                                        int grid_x,
                                                        int grid_y,
                                                        int grid_z,
                                                        float* __restrict__ image_feats_grad) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * grid_x * grid_y * channels;
  if (index >= total) return;
  const int   channel = index % channels;
  const int   bev     = index / channels;
  const int   batch   = bev / (grid_x * grid_y);
  const float grad    = out_grad[index];
  for (int z_index = 0; z_index < grid_z; ++z_index) {
    const int valid_cameras = valid_counts[bev * grid_z + z_index];
    if (valid_cameras == 0) continue;
    const float scale = grad / static_cast<float>(valid_cameras);
    for (int camera = 0; camera < cameras; ++camera) {
      const float2 coordinate = coordinates[(bev * grid_z + z_index) * cameras + camera];
      if (coordinate.x < 0.0f) continue;
      const int   feature_h = feature_size[(batch * cameras + camera) * 2];
      const int   feature_w = feature_size[(batch * cameras + camera) * 2 + 1];
      const int   u0_raw    = static_cast<int>(floorf(coordinate.x));
      const int   v0_raw    = static_cast<int>(floorf(coordinate.y));
      const int   u0        = clamp_int(u0_raw, 0, feature_w - 1);
      const int   u1        = clamp_int(u0_raw + 1, 0, feature_w - 1);
      const int   v0        = clamp_int(v0_raw, 0, feature_h - 1);
      const int   v1        = clamp_int(v0_raw + 1, 0, feature_h - 1);
      const float du        = coordinate.x - u0_raw;
      const float dv        = coordinate.y - v0_raw;
      const int   base      = (batch * cameras + camera) * feature_h * feature_w * channels;
      atomicAdd(image_feats_grad + base + (v0 * feature_w + u0) * channels + channel,
                scale * (1.0f - du) * (1.0f - dv));
      atomicAdd(image_feats_grad + base + (v1 * feature_w + u0) * channels + channel, scale * (1.0f - du) * dv);
      atomicAdd(image_feats_grad + base + (v0 * feature_w + u1) * channels + channel, scale * du * (1.0f - dv));
      atomicAdd(image_feats_grad + base + (v1 * feature_w + u1) * channels + channel, scale * du * dv);
    }
  }
}

__global__ void flash_bevpool_valid_count_kernel(int cameras,
                                                 int total_bev_z,
                                                 const float2* __restrict__ coordinates,
                                                 int* __restrict__ valid_counts) {
  const int bev_z = blockIdx.x * blockDim.x + threadIdx.x;
  if (bev_z >= total_bev_z) return;
  int count = 0;
  for (int camera = 0; camera < cameras; ++camera) count += coordinates[bev_z * cameras + camera].x >= 0.0f;
  valid_counts[bev_z] = count;
}

__global__ void flash_bevpool_backward_uv_pair_kernel(int channels,
                                                      int cameras,
                                                      int batch_size,
                                                      const float* __restrict__ out_grad,
                                                      const int* __restrict__ feature_size,
                                                      const float2* __restrict__ coordinates,
                                                      int grid_x,
                                                      int grid_y,
                                                      int grid_z,
                                                      float* __restrict__ image_feats_grad) {
  const int index   = blockIdx.x * blockDim.x + threadIdx.x;
  const int pairs_x = (grid_x + 1) / 2;
  const int total   = batch_size * grid_y * pairs_x * channels;
  if (index >= total) return;
  const int   channel         = index % channels;
  const int   pair            = index / channels;
  const int   pairs_per_batch = grid_y * pairs_x;
  const int   batch           = pair / pairs_per_batch;
  const int   local_pair      = pair % pairs_per_batch;
  const int   y               = local_pair / pairs_x;
  const int   x0              = (local_pair % pairs_x) * 2;
  const int   bev0            = batch * grid_x * grid_y + y * grid_x + x0;
  const bool  has_second      = x0 + 1 < grid_x;
  const int   bev1            = bev0 + 1;
  const float grad0           = out_grad[bev0 * channels + channel];
  const float grad1           = has_second ? out_grad[bev1 * channels + channel] : 0.0f;
  for (int z_index = 0; z_index < grid_z; ++z_index) {
    int valid0 = 0, valid1 = 0;
    for (int camera = 0; camera < cameras; ++camera) {
      valid0 += coordinates[(bev0 * grid_z + z_index) * cameras + camera].x >= 0.0f;
      valid1 += has_second && coordinates[(bev1 * grid_z + z_index) * cameras + camera].x >= 0.0f;
    }
    const float scale0 = valid0 > 0 ? grad0 / static_cast<float>(valid0) : 0.0f;
    const float scale1 = valid1 > 0 ? grad1 / static_cast<float>(valid1) : 0.0f;
    for (int camera = 0; camera < cameras; ++camera) {
      const float2 a = coordinates[(bev0 * grid_z + z_index) * cameras + camera];
      const float2 b =
          has_second ? coordinates[(bev1 * grid_z + z_index) * cameras + camera] : make_float2(-1.0f, -1.0f);
      if (a.x < 0.0f && b.x < 0.0f) continue;
      const int feature_h = feature_size[(batch * cameras + camera) * 2];
      const int feature_w = feature_size[(batch * cameras + camera) * 2 + 1];
      int       a00 = -1, a01 = -1, a10 = -1, a11 = -1, b00 = -2, b01 = -2, b10 = -2, b11 = -2;
      float     aw00 = 0.0f, aw01 = 0.0f, aw10 = 0.0f, aw11 = 0.0f, bw00 = 0.0f, bw01 = 0.0f, bw10 = 0.0f, bw11 = 0.0f;
      const int base = (batch * cameras + camera) * feature_h * feature_w * channels;
      if (a.x >= 0.0f) {
        const int   u = static_cast<int>(floorf(a.x)), v = static_cast<int>(floorf(a.y));
        const int   u0 = clamp_int(u, 0, feature_w - 1), u1 = clamp_int(u + 1, 0, feature_w - 1);
        const int   v0 = clamp_int(v, 0, feature_h - 1), v1 = clamp_int(v + 1, 0, feature_h - 1);
        const float du = a.x - u, dv = a.y - v;
        a00  = base + (v0 * feature_w + u0) * channels;
        a01  = base + (v1 * feature_w + u0) * channels;
        a10  = base + (v0 * feature_w + u1) * channels;
        a11  = base + (v1 * feature_w + u1) * channels;
        aw00 = scale0 * (1.0f - du) * (1.0f - dv);
        aw01 = scale0 * (1.0f - du) * dv;
        aw10 = scale0 * du * (1.0f - dv);
        aw11 = scale0 * du * dv;
      }
      if (b.x >= 0.0f) {
        const int   u = static_cast<int>(floorf(b.x)), v = static_cast<int>(floorf(b.y));
        const int   u0 = clamp_int(u, 0, feature_w - 1), u1 = clamp_int(u + 1, 0, feature_w - 1);
        const int   v0 = clamp_int(v, 0, feature_h - 1), v1 = clamp_int(v + 1, 0, feature_h - 1);
        const float du = b.x - u, dv = b.y - v;
        b00  = base + (v0 * feature_w + u0) * channels;
        b01  = base + (v1 * feature_w + u0) * channels;
        b10  = base + (v0 * feature_w + u1) * channels;
        b11  = base + (v1 * feature_w + u1) * channels;
        bw00 = scale1 * (1.0f - du) * (1.0f - dv);
        bw01 = scale1 * (1.0f - du) * dv;
        bw10 = scale1 * du * (1.0f - dv);
        bw11 = scale1 * du * dv;
      }
      if (a00 == b00)
        atomicAdd(image_feats_grad + a00 + channel, aw00 + bw00);
      else {
        if (a00 >= 0) atomicAdd(image_feats_grad + a00 + channel, aw00);
        if (b00 >= 0) atomicAdd(image_feats_grad + b00 + channel, bw00);
      }
      if (a01 == b01)
        atomicAdd(image_feats_grad + a01 + channel, aw01 + bw01);
      else {
        if (a01 >= 0) atomicAdd(image_feats_grad + a01 + channel, aw01);
        if (b01 >= 0) atomicAdd(image_feats_grad + b01 + channel, bw01);
      }
      if (a10 == b10)
        atomicAdd(image_feats_grad + a10 + channel, aw10 + bw10);
      else {
        if (a10 >= 0) atomicAdd(image_feats_grad + a10 + channel, aw10);
        if (b10 >= 0) atomicAdd(image_feats_grad + b10 + channel, bw10);
      }
      if (a11 == b11)
        atomicAdd(image_feats_grad + a11 + channel, aw11 + bw11);
      else {
        if (a11 >= 0) atomicAdd(image_feats_grad + a11 + channel, aw11);
        if (b11 >= 0) atomicAdd(image_feats_grad + b11 + channel, bw11);
      }
    }
  }
}

void check_float(ffi::TensorView tensor, const char* name) {
  check_musa_contiguous(tensor, name, kDLFloat32);
}

void check_int(ffi::TensorView tensor, const char* name) {
  check_musa_contiguous(tensor, name, kDLInt32);
}

}  // namespace

void flash_bevpool_forward(ffi::TensorView image_feats,
                           ffi::TensorView projection_matrices,
                           ffi::TensorView feature_size,
                           ffi::TensorView image_size,
                           ffi::TensorView out,
                           int             grid_x,
                           int             grid_y,
                           int             grid_z,
                           float           voxel_size_x,
                           float           voxel_size_y,
                           float           voxel_size_z,
                           float           voxel_x_min,
                           float           voxel_y_min,
                           float           voxel_z_min,
                           float           epsilon) {
  check_float(image_feats, "image_feats");
  check_float(projection_matrices, "projection_matrices");
  check_float(out, "out");
  check_int(feature_size, "feature_size");
  check_int(image_size, "image_size");
  TVM_FFI_ICHECK_EQ(image_feats.ndim(), 5) << "image_feats must have shape [B,N,H,W,C]";
  TVM_FFI_ICHECK_EQ(projection_matrices.numel(), dim(image_feats, 0) * dim(image_feats, 1) * 12)
      << "projection_matrices must have shape [B,N,3,4]";
  TVM_FFI_ICHECK_EQ(feature_size.numel(), dim(image_feats, 0) * dim(image_feats, 1) * 2)
      << "feature_size must have shape [B,N,2]";
  TVM_FFI_ICHECK_EQ(image_size.numel(), 2) << "image_size must have shape [2]";
  TVM_FFI_ICHECK(grid_x > 0 && grid_y > 0 && grid_z > 0) << "grid sizes must be positive";
  const int total = static_cast<int>(dim(image_feats, 0) * grid_x * grid_y * dim(image_feats, 4));
  if (total > 0) {
    flash_bevpool_forward_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, stream(image_feats)>>>(
        static_cast<int>(dim(image_feats, 4)),
        static_cast<int>(dim(image_feats, 1)),
        static_cast<int>(dim(image_feats, 0)),
        epsilon,
        static_cast<float*>(image_feats.data_ptr()),
        static_cast<float*>(projection_matrices.data_ptr()),
        static_cast<int*>(feature_size.data_ptr()),
        static_cast<int*>(image_size.data_ptr()),
        grid_x,
        grid_y,
        grid_z,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        voxel_x_min,
        voxel_y_min,
        voxel_z_min,
        static_cast<float*>(out.data_ptr()));
  }
  check_last_error("flash_bevpool_forward");
}

void flash_bevpool_backward(ffi::TensorView out_grad,
                            ffi::TensorView image_feats,
                            ffi::TensorView projection_matrices,
                            ffi::TensorView feature_size,
                            ffi::TensorView image_size,
                            ffi::TensorView roi_range,
                            ffi::TensorView image_feats_grad,
                            int             grid_x,
                            int             grid_y,
                            int             grid_z,
                            float           epsilon) {
  check_float(out_grad, "out_grad");
  check_float(image_feats, "image_feats");
  check_float(projection_matrices, "projection_matrices");
  check_float(roi_range, "roi_range");
  check_float(image_feats_grad, "image_feats_grad");
  check_int(feature_size, "feature_size");
  check_int(image_size, "image_size");
  TVM_FFI_ICHECK_EQ(roi_range.numel(), 6) << "roi_range must have shape [6]";
  TVM_FFI_ICHECK(grid_x > 0 && grid_y > 0 && grid_z > 0) << "grid sizes must be positive";
  const int total = static_cast<int>(dim(image_feats, 0) * grid_x * grid_y * dim(image_feats, 4));
  if (total > 0) {
    const int cameras = static_cast<int>(dim(image_feats, 1));
    if (cameras <= 6) {
      flash_bevpool_backward_kernel<6>
          <<<(total + kBackwardThreads - 1) / kBackwardThreads, kBackwardThreads, 0, stream(out_grad)>>>(
              static_cast<int>(dim(image_feats, 4)),
              cameras,
              static_cast<int>(dim(image_feats, 0)),
              epsilon,
              static_cast<float*>(out_grad.data_ptr()),
              static_cast<float*>(projection_matrices.data_ptr()),
              static_cast<int*>(feature_size.data_ptr()),
              static_cast<int*>(image_size.data_ptr()),
              static_cast<float*>(roi_range.data_ptr()),
              grid_x,
              grid_y,
              grid_z,
              static_cast<float*>(image_feats_grad.data_ptr()));
    } else {
      flash_bevpool_backward_kernel<kBackwardCachedCameras>
          <<<(total + kBackwardThreads - 1) / kBackwardThreads, kBackwardThreads, 0, stream(out_grad)>>>(
              static_cast<int>(dim(image_feats, 4)),
              cameras,
              static_cast<int>(dim(image_feats, 0)),
              epsilon,
              static_cast<float*>(out_grad.data_ptr()),
              static_cast<float*>(projection_matrices.data_ptr()),
              static_cast<int*>(feature_size.data_ptr()),
              static_cast<int*>(image_size.data_ptr()),
              static_cast<float*>(roi_range.data_ptr()),
              grid_x,
              grid_y,
              grid_z,
              static_cast<float*>(image_feats_grad.data_ptr()));
    }
  }
  check_last_error("flash_bevpool_backward");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flash_bevpool_forward, flash_bevpool_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flash_bevpool_backward, flash_bevpool_backward);
