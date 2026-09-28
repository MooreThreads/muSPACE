#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include <cstdlib>
#include <string_view>

#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::dim;
using muspace::ffi_utils::stream;

namespace {

constexpr int kThreads                        = 256;
constexpr int kTwoDimensionalMaxAutoIntervals = 4096;

inline int blocks_for(const int64_t size) {
  return static_cast<int>((size + kThreads - 1) / kThreads);
}

enum class LaunchStrategy {
  kAuto,
  kLinear,
  kTwoDimensional,
  kPacked4,
};

LaunchStrategy launch_strategy_from_env() {
  const char* value = std::getenv("MUSPACE_BEV_POOL_V2_LAUNCH");
  if (value == nullptr) {
    return LaunchStrategy::kAuto;
  }
  const std::string_view strategy(value);
  if (strategy == "linear") {
    return LaunchStrategy::kLinear;
  }
  if (strategy == "2d") {
    return LaunchStrategy::kTwoDimensional;
  }
  if (strategy == "packed4") {
    return LaunchStrategy::kPacked4;
  }
  return LaunchStrategy::kAuto;
}

bool use_two_dimensional_launch(const int channels, const int num_intervals) {
  const LaunchStrategy strategy = launch_strategy_from_env();
  if (strategy == LaunchStrategy::kLinear) {
    return false;
  }
  if (strategy == LaunchStrategy::kTwoDimensional) {
    return true;
  }
  return channels <= 32 && num_intervals <= kTwoDimensionalMaxAutoIntervals;
}

bool use_packed4_launch(const int channels) {
  const LaunchStrategy strategy = launch_strategy_from_env();
  if (strategy == LaunchStrategy::kLinear || strategy == LaunchStrategy::kTwoDimensional) {
    return false;
  }
  if (strategy == LaunchStrategy::kPacked4) {
    return channels % 4 == 0;
  }
  return channels >= 64 && channels % 4 == 0;
}

__global__ void bev_pool_v2_forward_kernel(const float* depth,
                                           const float* feat,
                                           const int*   ranks_depth,
                                           const int*   ranks_feat,
                                           const int*   ranks_bev,
                                           const int*   interval_starts,
                                           const int*   interval_lengths,
                                           float*       out,
                                           int          num_intervals,
                                           int          channels) {
  const int interval_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int channel        = blockIdx.y;
  if (interval_index >= num_intervals || channel >= channels) {
    return;
  }
  const int interval_start  = interval_starts[interval_index];
  const int interval_length = interval_lengths[interval_index];

  for (int i = 0; i < interval_length; ++i) {
    const int point   = interval_start + i;
    const int bev_idx = ranks_bev[point];
    float     val     = depth[ranks_depth[point]] * feat[ranks_feat[point] * channels + channel];
    atomicAdd(&out[bev_idx * channels + channel], val);
  }
}

__global__ void bev_pool_v2_forward_linear_kernel(const float* depth,
                                                  const float* feat,
                                                  const int*   ranks_depth,
                                                  const int*   ranks_feat,
                                                  const int*   ranks_bev,
                                                  const int*   interval_starts,
                                                  const int*   interval_lengths,
                                                  float*       out,
                                                  int          num_intervals,
                                                  int          channels) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_intervals * channels;
  if (index >= total) {
    return;
  }
  const int interval_index  = index / channels;
  const int channel         = index % channels;
  const int interval_start  = interval_starts[interval_index];
  const int interval_length = interval_lengths[interval_index];

  for (int i = 0; i < interval_length; ++i) {
    const int point   = interval_start + i;
    const int bev_idx = ranks_bev[point];
    float     val     = depth[ranks_depth[point]] * feat[ranks_feat[point] * channels + channel];
    atomicAdd(&out[bev_idx * channels + channel], val);
  }
}

__global__ void bev_pool_v2_forward_packed4_kernel(const float* depth,
                                                   const float* feat,
                                                   const int*   ranks_depth,
                                                   const int*   ranks_feat,
                                                   const int*   ranks_bev,
                                                   const int*   interval_starts,
                                                   const int*   interval_lengths,
                                                   float*       out,
                                                   int          num_intervals,
                                                   int          packed_channels) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_intervals * packed_channels;
  if (index >= total) {
    return;
  }
  const int interval_index  = index / packed_channels;
  const int packed_channel  = index % packed_channels;
  const int interval_start  = interval_starts[interval_index];
  const int interval_length = interval_lengths[interval_index];

  const float4* const packed_feat = reinterpret_cast<const float4*>(feat);
  for (int i = 0; i < interval_length; ++i) {
    const int    point   = interval_start + i;
    const float  weight  = depth[ranks_depth[point]];
    const int    bev_idx = ranks_bev[point];
    const float4 value   = packed_feat[ranks_feat[point] * packed_channels + packed_channel];
    const int    base    = bev_idx * packed_channels * 4 + packed_channel * 4;
    atomicAdd(&out[base + 0], weight * value.x);
    atomicAdd(&out[base + 1], weight * value.y);
    atomicAdd(&out[base + 2], weight * value.z);
    atomicAdd(&out[base + 3], weight * value.w);
  }
}

__global__ void bev_pool_v2_grad_depth_kernel(const float* grad_out,
                                              const float* feat,
                                              const int*   ranks_depth,
                                              const int*   ranks_feat,
                                              const int*   ranks_bev,
                                              float*       grad_depth,
                                              int          num_points,
                                              int          channels) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= num_points) {
    return;
  }

  float sum = 0.0f;
  for (int channel = 0; channel < channels; ++channel) {
    sum += grad_out[ranks_bev[point] * channels + channel] * feat[ranks_feat[point] * channels + channel];
  }
  grad_depth[ranks_depth[point]] = sum;
}

__global__ void bev_pool_v2_grad_feat_kernel(const float* grad_out,
                                             const float* depth,
                                             const int*   ranks_depth,
                                             const int*   ranks_feat,
                                             const int*   ranks_bev,
                                             const int*   interval_starts,
                                             const int*   interval_lengths,
                                             float*       grad_feat,
                                             int          num_intervals,
                                             int          channels) {
  const int interval_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int channel        = blockIdx.y;
  if (interval_index >= num_intervals || channel >= channels) {
    return;
  }

  const int interval_start  = interval_starts[interval_index];
  const int interval_length = interval_lengths[interval_index];

  for (int i = 0; i < interval_length; ++i) {
    const int point    = interval_start + i;
    const int feat_idx = ranks_feat[point];
    float     val      = grad_out[ranks_bev[point] * channels + channel] * depth[ranks_depth[point]];
    atomicAdd(&grad_feat[feat_idx * channels + channel], val);
  }
}

__global__ void bev_pool_v2_grad_feat_linear_kernel(const float* grad_out,
                                                    const float* depth,
                                                    const int*   ranks_depth,
                                                    const int*   ranks_feat,
                                                    const int*   ranks_bev,
                                                    const int*   interval_starts,
                                                    const int*   interval_lengths,
                                                    float*       grad_feat,
                                                    int          num_intervals,
                                                    int          channels) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_intervals * channels;
  if (index >= total) {
    return;
  }

  const int interval_index  = index / channels;
  const int channel         = index % channels;
  const int interval_start  = interval_starts[interval_index];
  const int interval_length = interval_lengths[interval_index];

  for (int i = 0; i < interval_length; ++i) {
    const int point    = interval_start + i;
    const int feat_idx = ranks_feat[point];
    float     val      = grad_out[ranks_bev[point] * channels + channel] * depth[ranks_depth[point]];
    atomicAdd(&grad_feat[feat_idx * channels + channel], val);
  }
}

}  // namespace

void bev_pool_v2_forward(ffi::TensorView depth,
                         ffi::TensorView feat,
                         ffi::TensorView out,
                         ffi::TensorView ranks_depth,
                         ffi::TensorView ranks_feat,
                         ffi::TensorView ranks_bev,
                         ffi::TensorView interval_starts,
                         ffi::TensorView interval_lengths) {
  check_musa_contiguous(depth, "depth", kDLFloat32);
  check_musa_contiguous(feat, "feat", kDLFloat32);
  check_musa_contiguous(out, "out", kDLFloat32);
  check_musa_contiguous(ranks_depth, "ranks_depth", kDLInt32);
  check_musa_contiguous(ranks_feat, "ranks_feat", kDLInt32);
  check_musa_contiguous(ranks_bev, "ranks_bev", kDLInt32);
  check_musa_contiguous(interval_starts, "interval_starts", kDLInt32);
  check_musa_contiguous(interval_lengths, "interval_lengths", kDLInt32);
  TVM_FFI_ICHECK_EQ(feat.ndim(), 5) << "feat must have shape [B, N, H, W, C]";
  const int channels      = static_cast<int>(dim(feat, 4));
  const int num_intervals = static_cast<int>(interval_starts.numel());
  if (num_intervals == 0 || channels == 0) {
    return;
  }
  // Packed float4 threads reduce per-channel indexing for the common C>=64 path. The 2D
  // launch remains useful for small C<=32 workloads with few intervals.
  if (use_packed4_launch(channels)) {
    const int packed_channels = channels / 4;
    bev_pool_v2_forward_packed4_kernel<<<blocks_for(static_cast<int64_t>(num_intervals) * packed_channels),
                                         kThreads,
                                         0,
                                         stream(depth)>>>(static_cast<float*>(depth.data_ptr()),
                                                          static_cast<float*>(feat.data_ptr()),
                                                          static_cast<int*>(ranks_depth.data_ptr()),
                                                          static_cast<int*>(ranks_feat.data_ptr()),
                                                          static_cast<int*>(ranks_bev.data_ptr()),
                                                          static_cast<int*>(interval_starts.data_ptr()),
                                                          static_cast<int*>(interval_lengths.data_ptr()),
                                                          static_cast<float*>(out.data_ptr()),
                                                          num_intervals,
                                                          packed_channels);
  } else if (use_two_dimensional_launch(channels, num_intervals)) {
    const dim3 blocks(blocks_for(num_intervals), channels);
    bev_pool_v2_forward_kernel<<<blocks, kThreads, 0, stream(depth)>>>(static_cast<float*>(depth.data_ptr()),
                                                                       static_cast<float*>(feat.data_ptr()),
                                                                       static_cast<int*>(ranks_depth.data_ptr()),
                                                                       static_cast<int*>(ranks_feat.data_ptr()),
                                                                       static_cast<int*>(ranks_bev.data_ptr()),
                                                                       static_cast<int*>(interval_starts.data_ptr()),
                                                                       static_cast<int*>(interval_lengths.data_ptr()),
                                                                       static_cast<float*>(out.data_ptr()),
                                                                       num_intervals,
                                                                       channels);
  } else {
    bev_pool_v2_forward_linear_kernel<<<blocks_for(static_cast<int64_t>(num_intervals) * channels),
                                        kThreads,
                                        0,
                                        stream(depth)>>>(static_cast<float*>(depth.data_ptr()),
                                                         static_cast<float*>(feat.data_ptr()),
                                                         static_cast<int*>(ranks_depth.data_ptr()),
                                                         static_cast<int*>(ranks_feat.data_ptr()),
                                                         static_cast<int*>(ranks_bev.data_ptr()),
                                                         static_cast<int*>(interval_starts.data_ptr()),
                                                         static_cast<int*>(interval_lengths.data_ptr()),
                                                         static_cast<float*>(out.data_ptr()),
                                                         num_intervals,
                                                         channels);
  }
  check_last_error("bev_pool_v2_forward");
}

void bev_pool_v2_backward(ffi::TensorView grad_out,
                          ffi::TensorView depth,
                          ffi::TensorView feat,
                          ffi::TensorView ranks_depth,
                          ffi::TensorView ranks_feat,
                          ffi::TensorView ranks_bev,
                          ffi::TensorView interval_starts_bp,
                          ffi::TensorView interval_lengths_bp,
                          ffi::TensorView grad_depth,
                          ffi::TensorView grad_feat) {
  check_musa_contiguous(grad_out, "grad_out", kDLFloat32);
  check_musa_contiguous(depth, "depth", kDLFloat32);
  check_musa_contiguous(feat, "feat", kDLFloat32);
  check_musa_contiguous(grad_depth, "grad_depth", kDLFloat32);
  check_musa_contiguous(grad_feat, "grad_feat", kDLFloat32);
  check_musa_contiguous(ranks_depth, "ranks_depth", kDLInt32);
  check_musa_contiguous(ranks_feat, "ranks_feat", kDLInt32);
  check_musa_contiguous(ranks_bev, "ranks_bev", kDLInt32);
  check_musa_contiguous(interval_starts_bp, "interval_starts_bp", kDLInt32);
  check_musa_contiguous(interval_lengths_bp, "interval_lengths_bp", kDLInt32);
  TVM_FFI_ICHECK_EQ(feat.ndim(), 5) << "feat must have shape [B, N, H, W, C]";
  const int channels   = static_cast<int>(dim(feat, 4));
  const int num_points = static_cast<int>(ranks_depth.numel());
  if (num_points > 0 && channels > 0) {
    bev_pool_v2_grad_depth_kernel<<<blocks_for(num_points), kThreads, 0, stream(grad_out)>>>(
        static_cast<float*>(grad_out.data_ptr()),
        static_cast<float*>(feat.data_ptr()),
        static_cast<int*>(ranks_depth.data_ptr()),
        static_cast<int*>(ranks_feat.data_ptr()),
        static_cast<int*>(ranks_bev.data_ptr()),
        static_cast<float*>(grad_depth.data_ptr()),
        num_points,
        channels);
  }

  const int num_intervals = static_cast<int>(interval_starts_bp.numel());
  if (num_intervals > 0 && channels > 0) {
    if (use_two_dimensional_launch(channels, num_intervals)) {
      const dim3 blocks(blocks_for(num_intervals), channels);
      bev_pool_v2_grad_feat_kernel<<<blocks, kThreads, 0, stream(grad_out)>>>(
          static_cast<float*>(grad_out.data_ptr()),
          static_cast<float*>(depth.data_ptr()),
          static_cast<int*>(ranks_depth.data_ptr()),
          static_cast<int*>(ranks_feat.data_ptr()),
          static_cast<int*>(ranks_bev.data_ptr()),
          static_cast<int*>(interval_starts_bp.data_ptr()),
          static_cast<int*>(interval_lengths_bp.data_ptr()),
          static_cast<float*>(grad_feat.data_ptr()),
          num_intervals,
          channels);
    } else {
      bev_pool_v2_grad_feat_linear_kernel<<<blocks_for(static_cast<int64_t>(num_intervals) * channels),
                                            kThreads,
                                            0,
                                            stream(grad_out)>>>(static_cast<float*>(grad_out.data_ptr()),
                                                                static_cast<float*>(depth.data_ptr()),
                                                                static_cast<int*>(ranks_depth.data_ptr()),
                                                                static_cast<int*>(ranks_feat.data_ptr()),
                                                                static_cast<int*>(ranks_bev.data_ptr()),
                                                                static_cast<int*>(interval_starts_bp.data_ptr()),
                                                                static_cast<int*>(interval_lengths_bp.data_ptr()),
                                                                static_cast<float*>(grad_feat.data_ptr()),
                                                                num_intervals,
                                                                channels);
    }
  }
  check_last_error("bev_pool_v2_backward");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bev_pool_v2_forward, bev_pool_v2_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bev_pool_v2_backward, bev_pool_v2_backward);
