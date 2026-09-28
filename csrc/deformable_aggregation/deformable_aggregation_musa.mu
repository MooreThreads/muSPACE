#include <musa.h>
#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include <cmath>

#include "../ffi/fast_divmod.hpp"
#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::dim;
using muspace::ffi_utils::FastDivmod;
using muspace::ffi_utils::stream;

namespace {

constexpr int kThreads   = 128;
constexpr int kMaxGroups = 8;

inline int blocks_for(const int64_t size) {
  if (size <= 0) {
    return 0;
  }
  return static_cast<int>((size + kThreads - 1) / kThreads);
}

__device__ float bilinear_sampling(const float*& bottom_data,
                                   const int&    height,
                                   const int&    width,
                                   const int&    num_embeds,
                                   const float&  h_im,
                                   const float&  w_im,
                                   const int&    base_ptr) {
  const int h_low  = floorf(h_im);
  const int w_low  = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride          = num_embeds;
  const int h_stride          = width * w_stride;
  const int h_low_ptr_offset  = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset  = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1             = bottom_data[ptr1];
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2             = bottom_data[ptr2];
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3             = bottom_data[ptr3];
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4             = bottom_data[ptr4];
  }

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__device__ void bilinear_sampling_grad(const float*& bottom_data,
                                       const float&  weight,
                                       const int&    height,
                                       const int&    width,
                                       const int&    num_embeds,
                                       const float&  h_im,
                                       const float&  w_im,
                                       const int&    base_ptr,
                                       const float&  grad_output,
                                       float*&       grad_mc_ms_feat,
                                       float*        grad_sampling_location,
                                       float*        grad_weights) {
  const int h_low  = floorf(h_im);
  const int w_low  = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride          = num_embeds;
  const int h_stride          = width * w_stride;
  const int h_low_ptr_offset  = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset  = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float top_grad_mc_ms_feat = grad_output * weight;
  float       grad_h_weight = 0, grad_w_weight = 0;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1             = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2             = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3             = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4             = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
  }

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_weights, grad_output * val);
  atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
  atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
}

__global__ void deformable_aggregation_kernel(const int    num_kernels,
                                              float*       output,
                                              const float* mc_ms_feat,
                                              const int*   spatial_shape,
                                              const int*   scale_start_index,
                                              const float* sample_location,
                                              const float* weights,
                                              int          batch_size,
                                              int          num_cams,
                                              int          num_feat,
                                              int          num_embeds,
                                              int          num_scale,
                                              int          num_anchors,
                                              int          num_pts,
                                              int          num_groups) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_kernels) return;

  const float weight        = *(weights + idx / (num_embeds / num_groups));
  const int   channel_index = idx % num_embeds;
  idx /= num_embeds;
  const int scale_index = idx % num_scale;
  idx /= num_scale;

  const int cam_index = idx % num_cams;
  idx /= num_cams;
  const int pts_index = idx % num_pts;
  idx /= num_pts;

  int anchor_index = idx % num_anchors;
  idx /= num_anchors;
  const int batch_index = idx % batch_size;
  idx /= batch_size;

  anchor_index         = batch_index * num_anchors + anchor_index;
  const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

  const float loc_w = sample_location[loc_offset];
  if (loc_w <= 0 || loc_w >= 1) return;
  const float loc_h = sample_location[loc_offset + 1];
  if (loc_h <= 0 || loc_h >= 1) return;

  int       cam_scale_index = cam_index * num_scale + scale_index;
  const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

  cam_scale_index = cam_scale_index << 1;
  const int h     = spatial_shape[cam_scale_index];
  const int w     = spatial_shape[cam_scale_index + 1];

  const float h_im = loc_h * h - 0.5;
  const float w_im = loc_w * w - 0.5;

  atomicAdd(output + anchor_index * num_embeds + channel_index,
            bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight);
}

__global__ void deformable_aggregation_grad_kernel(const int    num_kernels,
                                                   const float* mc_ms_feat,
                                                   const int*   spatial_shape,
                                                   const int*   scale_start_index,
                                                   const float* sample_location,
                                                   const float* weights,
                                                   const float* grad_output,
                                                   float*       grad_mc_ms_feat,
                                                   float*       grad_sampling_location,
                                                   float*       grad_weights,
                                                   int          batch_size,
                                                   int          num_cams,
                                                   int          num_feat,
                                                   int          num_embeds,
                                                   int          num_scale,
                                                   int          num_anchors,
                                                   int          num_pts,
                                                   int          num_groups) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_kernels) return;

  const int weights_ptr   = idx / (num_embeds / num_groups);
  const int channel_index = idx % num_embeds;
  idx /= num_embeds;
  const int scale_index = idx % num_scale;
  idx /= num_scale;

  const int cam_index = idx % num_cams;
  idx /= num_cams;
  const int pts_index = idx % num_pts;
  idx /= num_pts;

  int anchor_index = idx % num_anchors;
  idx /= num_anchors;
  const int batch_index = idx % batch_size;
  idx /= batch_size;

  anchor_index         = batch_index * num_anchors + anchor_index;
  const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

  const float loc_w = sample_location[loc_offset];
  if (loc_w <= 0 || loc_w >= 1) return;
  const float loc_h = sample_location[loc_offset + 1];
  if (loc_h <= 0 || loc_h >= 1) return;

  const float grad = grad_output[anchor_index * num_embeds + channel_index];

  int       cam_scale_index = cam_index * num_scale + scale_index;
  const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

  cam_scale_index = cam_scale_index << 1;
  const int h     = spatial_shape[cam_scale_index];
  const int w     = spatial_shape[cam_scale_index + 1];

  const float h_im = loc_h * h - 0.5;
  const float w_im = loc_w * w - 0.5;

  /* atomicAdd( */
  /*     output + anchor_index * num_embeds + channel_index, */
  /*     bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight */
  /* ); */
  const float weight            = weights[weights_ptr];
  float*      grad_weights_ptr  = grad_weights + weights_ptr;
  float*      grad_location_ptr = grad_sampling_location + loc_offset;
  bilinear_sampling_grad(mc_ms_feat,
                         weight,
                         h,
                         w,
                         num_embeds,
                         h_im,
                         w_im,
                         value_offset,
                         grad,
                         grad_mc_ms_feat,
                         grad_location_ptr,
                         grad_weights_ptr);
}

__global__ void deformable_aggregation_kernel_opt(int          num_kernels,
                                                  float*       output,
                                                  const float* mc_ms_feat,
                                                  const int*   spatial_shape,
                                                  const int*   scale_start_index,
                                                  const float* sample_location,
                                                  const float* weights,
                                                  int          batch_size,
                                                  int          num_cams,
                                                  int          num_feat,
                                                  int          num_embeds,
                                                  int          num_scale,
                                                  int          num_anchors,
                                                  int          num_pts,
                                                  int          num_groups,
                                                  FastDivmod   fastdiv1,
                                                  FastDivmod   fastdiv2,
                                                  FastDivmod   fastdiv3,
                                                  FastDivmod   fastdiv4,
                                                  FastDivmod   fastdiv5,
                                                  FastDivmod   fastdiv6) {
  const uint32_t index = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= static_cast<uint32_t>(num_kernels)) return;
  uint32_t    i1, i2, i3, i4, i5, ignored;
  uint32_t    channel, scale, camera, point, anchor, batch;
  const float weight = weights[index / (num_embeds / num_groups)];
  fastdiv1(i1, channel, index);
  fastdiv2(i2, scale, i1);
  fastdiv3(i3, camera, i2);
  fastdiv4(i4, point, i3);
  fastdiv5(i5, anchor, i4);
  fastdiv6(ignored, batch, i5);
  const int   flat_anchor = static_cast<int>(batch) * num_anchors + static_cast<int>(anchor);
  const int   location = ((flat_anchor * num_pts + static_cast<int>(point)) * num_cams + static_cast<int>(camera)) << 1;
  const float loc_w    = sample_location[location];
  const float loc_h    = sample_location[location + 1];
  if (loc_w <= 0.0F || loc_w >= 1.0F || loc_h <= 0.0F || loc_h >= 1.0F) return;
  int       camera_scale = static_cast<int>(camera) * num_scale + static_cast<int>(scale);
  const int value_offset =
      (static_cast<int>(batch) * num_feat + scale_start_index[camera_scale]) * num_embeds + static_cast<int>(channel);
  camera_scale <<= 1;
  const int   height  = spatial_shape[camera_scale];
  const int   width   = spatial_shape[camera_scale + 1];
  const float sampled = bilinear_sampling(
      mc_ms_feat, height, width, num_embeds, loc_h * height - 0.5F, loc_w * width - 0.5F, value_offset);
  atomicAdd(output + flat_anchor * num_embeds + static_cast<int>(channel), sampled * weight);
}

__global__ void deformable_aggregation_grad_kernel_opt(int          num_kernels,
                                                       const float* mc_ms_feat,
                                                       const int*   spatial_shape,
                                                       const int*   scale_start_index,
                                                       const float* sample_location,
                                                       const float* weights,
                                                       const float* grad_output,
                                                       float*       grad_mc_ms_feat,
                                                       float*       grad_sampling_location,
                                                       float*       grad_weights,
                                                       int          batch_size,
                                                       int          num_cams,
                                                       int          num_feat,
                                                       int          num_embeds,
                                                       int          num_scale,
                                                       int          num_anchors,
                                                       int          num_pts,
                                                       int          num_groups,
                                                       FastDivmod   fastdiv1,
                                                       FastDivmod   fastdiv2,
                                                       FastDivmod   fastdiv3,
                                                       FastDivmod   fastdiv4,
                                                       FastDivmod   fastdiv5,
                                                       FastDivmod   fastdiv6) {
  const uint32_t index = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= static_cast<uint32_t>(num_kernels)) return;
  uint32_t i3, i4, i5, ignored;
  uint32_t camera, point, anchor, batch;
  fastdiv3(i3, camera, index);
  fastdiv4(i4, point, i3);
  fastdiv5(i5, anchor, i4);
  fastdiv6(ignored, batch, i5);
  const int   flat_anchor = static_cast<int>(batch) * num_anchors + static_cast<int>(anchor);
  const int   location = ((flat_anchor * num_pts + static_cast<int>(point)) * num_cams + static_cast<int>(camera)) << 1;
  const float loc_w    = sample_location[location];
  const float loc_h    = sample_location[location + 1];
  if (!(loc_w > 0.0F && loc_w < 1.0F && loc_h > 0.0F && loc_h < 1.0F)) return;
  const float* grad_output_ptr    = grad_output + flat_anchor * num_embeds;
  const int    channels_per_group = num_embeds / num_groups;
  float        grad_location_x    = 0.0F;
  float        grad_location_y    = 0.0F;
  for (int scale = 0; scale < num_scale; ++scale) {
    float       grad_weights_local[kMaxGroups] = {0.0F};
    const int   camera_scale                   = static_cast<int>(camera) * num_scale + scale;
    const int   height                         = spatial_shape[camera_scale * 2];
    const int   width                          = spatial_shape[camera_scale * 2 + 1];
    const float h_im                           = loc_h * static_cast<float>(height) - 0.5F;
    const float w_im                           = loc_w * static_cast<float>(width) - 0.5F;
    const int   h_low = floorf(h_im), w_low = floorf(w_im);
    const int   h_high = h_low + 1, w_high = w_low + 1;
    const float lh = h_im - h_low, lw = w_im - w_low;
    const float hh = 1.0F - lh, hw = 1.0F - lw;
    const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    const int   row_stride = width * num_embeds;
    const int   value_base = (static_cast<int>(batch) * num_feat + scale_start_index[camera_scale]) * num_embeds;
    const int   weights_base =
        (((((flat_anchor * num_pts + static_cast<int>(point)) * num_cams + static_cast<int>(camera)) * num_scale) +
          scale) *
         num_groups);
    for (int channel = 0; channel < num_embeds; ++channel) {
      const int   group    = channel / channels_per_group;
      const float top_grad = grad_output_ptr[channel] * weights[weights_base + group];
      const int   value    = value_base + channel;
      const int   p1       = (h_low >= 0 && w_low >= 0) ? value + h_low * row_stride + w_low * num_embeds : -1;
      const int   p2       = (h_low >= 0 && w_high < width) ? value + h_low * row_stride + w_high * num_embeds : -1;
      const int   p3       = (h_high < height && w_low >= 0) ? value + h_high * row_stride + w_low * num_embeds : -1;
      const int   p4 = (h_high < height && w_high < width) ? value + h_high * row_stride + w_high * num_embeds : -1;
      const float v1 = p1 >= 0 ? mc_ms_feat[p1] : 0.0F;
      const float v2 = p2 >= 0 ? mc_ms_feat[p2] : 0.0F;
      const float v3 = p3 >= 0 ? mc_ms_feat[p3] : 0.0F;
      const float v4 = p4 >= 0 ? mc_ms_feat[p4] : 0.0F;
      if (p1 >= 0) atomicAdd(grad_mc_ms_feat + p1, w1 * top_grad);
      if (p2 >= 0) atomicAdd(grad_mc_ms_feat + p2, w2 * top_grad);
      if (p3 >= 0) atomicAdd(grad_mc_ms_feat + p3, w3 * top_grad);
      if (p4 >= 0) atomicAdd(grad_mc_ms_feat + p4, w4 * top_grad);
      grad_weights_local[group] += grad_output_ptr[channel] * (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
      const float grad_h = -(p1 >= 0 ? hw * v1 : 0.0F) - (p2 >= 0 ? lw * v2 : 0.0F) + (p3 >= 0 ? hw * v3 : 0.0F) +
                           (p4 >= 0 ? lw * v4 : 0.0F);
      const float grad_w = -(p1 >= 0 ? hh * v1 : 0.0F) + (p2 >= 0 ? hh * v2 : 0.0F) - (p3 >= 0 ? lh * v3 : 0.0F) +
                           (p4 >= 0 ? lh * v4 : 0.0F);
      grad_location_x += static_cast<float>(width) * grad_w * top_grad;
      grad_location_y += static_cast<float>(height) * grad_h * top_grad;
    }
    const int grad_weights_base = weights_base;
    for (int group = 0; group < num_groups; ++group)
      grad_weights[grad_weights_base + group] = grad_weights_local[group];
  }
  grad_sampling_location[location]     = grad_location_x;
  grad_sampling_location[location + 1] = grad_location_y;
}

void launch_deformable_aggregation_forward_base(float*       output,
                                                const float* mc_ms_feat,
                                                const int*   spatial_shape,
                                                const int*   scale_start_index,
                                                const float* sample_location,
                                                const float* weights,
                                                int          batch_size,
                                                int          num_cams,
                                                int          num_feat,
                                                int          num_embeds,
                                                int          num_scale,
                                                int          num_anchors,
                                                int          num_pts,
                                                int          num_groups,
                                                musaStream_t current_stream) {
  const int64_t num_kernels =
      static_cast<int64_t>(batch_size) * num_pts * num_embeds * num_anchors * num_cams * num_scale;
  const int blocks = blocks_for(num_kernels);
  if (blocks == 0) {
    return;
  }
  deformable_aggregation_kernel<<<blocks, kThreads, 0, current_stream>>>(static_cast<int>(num_kernels),
                                                                         output,
                                                                         mc_ms_feat,
                                                                         spatial_shape,
                                                                         scale_start_index,
                                                                         sample_location,
                                                                         weights,
                                                                         batch_size,
                                                                         num_cams,
                                                                         num_feat,
                                                                         num_embeds,
                                                                         num_scale,
                                                                         num_anchors,
                                                                         num_pts,
                                                                         num_groups);
}

void launch_deformable_aggregation_backward_base(const float* mc_ms_feat,
                                                 const int*   spatial_shape,
                                                 const int*   scale_start_index,
                                                 const float* sample_location,
                                                 const float* weights,
                                                 const float* grad_output,
                                                 float*       grad_mc_ms_feat,
                                                 float*       grad_sampling_location,
                                                 float*       grad_weights,
                                                 int          batch_size,
                                                 int          num_cams,
                                                 int          num_feat,
                                                 int          num_embeds,
                                                 int          num_scale,
                                                 int          num_anchors,
                                                 int          num_pts,
                                                 int          num_groups,
                                                 musaStream_t current_stream) {
  const int64_t num_kernels =
      static_cast<int64_t>(batch_size) * num_pts * num_embeds * num_anchors * num_cams * num_scale;
  const int blocks = blocks_for(num_kernels);
  if (blocks == 0) {
    return;
  }
  deformable_aggregation_grad_kernel<<<blocks, kThreads, 0, current_stream>>>(static_cast<int>(num_kernels),
                                                                              mc_ms_feat,
                                                                              spatial_shape,
                                                                              scale_start_index,
                                                                              sample_location,
                                                                              weights,
                                                                              grad_output,
                                                                              grad_mc_ms_feat,
                                                                              grad_sampling_location,
                                                                              grad_weights,
                                                                              batch_size,
                                                                              num_cams,
                                                                              num_feat,
                                                                              num_embeds,
                                                                              num_scale,
                                                                              num_anchors,
                                                                              num_pts,
                                                                              num_groups);
}

void launch_deformable_aggregation_forward_opt(float*       output,
                                               const float* mc_ms_feat,
                                               const int*   spatial_shape,
                                               const int*   scale_start_index,
                                               const float* sample_location,
                                               const float* weights,
                                               int          batch_size,
                                               int          num_cams,
                                               int          num_feat,
                                               int          num_embeds,
                                               int          num_scale,
                                               int          num_anchors,
                                               int          num_pts,
                                               int          num_groups,
                                               musaStream_t current_stream) {
  const int64_t num_kernels =
      static_cast<int64_t>(batch_size) * num_pts * num_embeds * num_anchors * num_cams * num_scale;
  const int blocks = blocks_for(num_kernels);
  if (blocks == 0) return;
  deformable_aggregation_kernel_opt<<<blocks, kThreads, 0, current_stream>>>(static_cast<int>(num_kernels),
                                                                             output,
                                                                             mc_ms_feat,
                                                                             spatial_shape,
                                                                             scale_start_index,
                                                                             sample_location,
                                                                             weights,
                                                                             batch_size,
                                                                             num_cams,
                                                                             num_feat,
                                                                             num_embeds,
                                                                             num_scale,
                                                                             num_anchors,
                                                                             num_pts,
                                                                             num_groups,
                                                                             FastDivmod(num_embeds),
                                                                             FastDivmod(num_scale),
                                                                             FastDivmod(num_cams),
                                                                             FastDivmod(num_pts),
                                                                             FastDivmod(num_anchors),
                                                                             FastDivmod(batch_size));
}

void launch_deformable_aggregation_backward_opt(const float* mc_ms_feat,
                                                const int*   spatial_shape,
                                                const int*   scale_start_index,
                                                const float* sample_location,
                                                const float* weights,
                                                const float* grad_output,
                                                float*       grad_mc_ms_feat,
                                                float*       grad_sampling_location,
                                                float*       grad_weights,
                                                int          batch_size,
                                                int          num_cams,
                                                int          num_feat,
                                                int          num_embeds,
                                                int          num_scale,
                                                int          num_anchors,
                                                int          num_pts,
                                                int          num_groups,
                                                musaStream_t current_stream) {
  const int64_t num_kernels = static_cast<int64_t>(batch_size) * num_pts * num_anchors * num_cams;
  const int     blocks      = blocks_for(num_kernels);
  if (blocks == 0) return;
  deformable_aggregation_grad_kernel_opt<<<blocks, kThreads, 0, current_stream>>>(static_cast<int>(num_kernels),
                                                                                  mc_ms_feat,
                                                                                  spatial_shape,
                                                                                  scale_start_index,
                                                                                  sample_location,
                                                                                  weights,
                                                                                  grad_output,
                                                                                  grad_mc_ms_feat,
                                                                                  grad_sampling_location,
                                                                                  grad_weights,
                                                                                  batch_size,
                                                                                  num_cams,
                                                                                  num_feat,
                                                                                  num_embeds,
                                                                                  num_scale,
                                                                                  num_anchors,
                                                                                  num_pts,
                                                                                  num_groups,
                                                                                  FastDivmod(num_embeds),
                                                                                  FastDivmod(num_scale),
                                                                                  FastDivmod(num_cams),
                                                                                  FastDivmod(num_pts),
                                                                                  FastDivmod(num_anchors),
                                                                                  FastDivmod(batch_size));
}

}  // namespace

static void validate(
    ffi::TensorView mc, ffi::TensorView sp, ffi::TensorView st, ffi::TensorView loc, ffi::TensorView w) {
  check_musa_contiguous(mc, "mc_ms_feat", kDLFloat32);
  check_musa_contiguous(sp, "spatial_shape", kDLInt32);
  check_musa_contiguous(st, "scale_start_index", kDLInt32);
  check_musa_contiguous(loc, "sampling_location", kDLFloat32);
  check_musa_contiguous(w, "weights", kDLFloat32);
  TVM_FFI_ICHECK_EQ(mc.ndim(), 3);
  TVM_FFI_ICHECK_EQ(sp.ndim(), 3);
  TVM_FFI_ICHECK_EQ(loc.ndim(), 5);
  TVM_FFI_ICHECK_EQ(w.ndim(), 6);
}
static void forward(ffi::TensorView mc,
                    ffi::TensorView sp,
                    ffi::TensorView st,
                    ffi::TensorView loc,
                    ffi::TensorView weights,
                    ffi::TensorView out,
                    bool            optimized) {
  validate(mc, sp, st, loc, weights);
  check_musa_contiguous(out, "out", kDLFloat32);
  const auto launch =
      optimized ? launch_deformable_aggregation_forward_opt : launch_deformable_aggregation_forward_base;
  launch(static_cast<float*>(out.data_ptr()),
         static_cast<float*>(mc.data_ptr()),
         static_cast<int*>(sp.data_ptr()),
         static_cast<int*>(st.data_ptr()),
         static_cast<float*>(loc.data_ptr()),
         static_cast<float*>(weights.data_ptr()),
         dim(mc, 0),
         dim(sp, 0),
         dim(mc, 1),
         dim(mc, 2),
         dim(sp, 1),
         dim(loc, 1),
         dim(loc, 2),
         dim(weights, 5),
         stream(mc));
  check_last_error(optimized ? "deformable_aggregation_forward_opt" : "deformable_aggregation_forward_base");
}

static void backward(ffi::TensorView mc,
                     ffi::TensorView sp,
                     ffi::TensorView st,
                     ffi::TensorView loc,
                     ffi::TensorView weights,
                     ffi::TensorView grad,
                     ffi::TensorView grad_mc,
                     ffi::TensorView grad_loc,
                     ffi::TensorView grad_weights,
                     bool            optimized) {
  validate(mc, sp, st, loc, weights);
  const auto launch =
      optimized ? launch_deformable_aggregation_backward_opt : launch_deformable_aggregation_backward_base;
  launch(static_cast<float*>(mc.data_ptr()),
         static_cast<int*>(sp.data_ptr()),
         static_cast<int*>(st.data_ptr()),
         static_cast<float*>(loc.data_ptr()),
         static_cast<float*>(weights.data_ptr()),
         static_cast<float*>(grad.data_ptr()),
         static_cast<float*>(grad_mc.data_ptr()),
         static_cast<float*>(grad_loc.data_ptr()),
         static_cast<float*>(grad_weights.data_ptr()),
         dim(mc, 0),
         dim(sp, 0),
         dim(mc, 1),
         dim(mc, 2),
         dim(sp, 1),
         dim(loc, 1),
         dim(loc, 2),
         dim(weights, 5),
         stream(mc));
  check_last_error(optimized ? "deformable_aggregation_backward_opt" : "deformable_aggregation_backward_base");
}

void deformable_aggregation_forward_opt(
    ffi::TensorView a, ffi::TensorView b, ffi::TensorView c, ffi::TensorView d, ffi::TensorView e, ffi::TensorView f) {
  forward(a, b, c, d, e, f, true);
}
void deformable_aggregation_forward_base(
    ffi::TensorView a, ffi::TensorView b, ffi::TensorView c, ffi::TensorView d, ffi::TensorView e, ffi::TensorView f) {
  forward(a, b, c, d, e, f, false);
}
void deformable_aggregation_backward_opt(ffi::TensorView a,
                                         ffi::TensorView b,
                                         ffi::TensorView c,
                                         ffi::TensorView d,
                                         ffi::TensorView e,
                                         ffi::TensorView f,
                                         ffi::TensorView g,
                                         ffi::TensorView h,
                                         ffi::TensorView i) {
  backward(a, b, c, d, e, f, g, h, i, true);
}
void deformable_aggregation_backward_base(ffi::TensorView a,
                                          ffi::TensorView b,
                                          ffi::TensorView c,
                                          ffi::TensorView d,
                                          ffi::TensorView e,
                                          ffi::TensorView f,
                                          ffi::TensorView g,
                                          ffi::TensorView h,
                                          ffi::TensorView i) {
  backward(a, b, c, d, e, f, g, h, i, false);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deformable_aggregation_forward_opt, deformable_aggregation_forward_opt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deformable_aggregation_forward_base, deformable_aggregation_forward_base);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deformable_aggregation_backward_opt, deformable_aggregation_backward_opt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(deformable_aggregation_backward_base, deformable_aggregation_backward_base);
