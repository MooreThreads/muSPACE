#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::dim;
using muspace::ffi_utils::stream;

namespace {
constexpr int kThreads = 256;

int blocks_for(int64_t size) {
  return static_cast<int>((size + kThreads - 1) / kThreads);
}

__global__ void bev_pool_forward_kernel(const float* feats,
                                        const int*   coords,
                                        const int*   interval_starts,
                                        const int*   interval_lengths,
                                        float*       out,
                                        int          intervals,
                                        int          channels,
                                        int          depth,
                                        int          height,
                                        int          width) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = intervals * channels;
  if (index >= total) return;
  const int  interval  = index / channels;
  const int  channel   = index % channels;
  const int  start     = interval_starts[interval];
  const int  length    = interval_lengths[interval];
  const int* coord     = coords + start * 4;
  const int  out_index = (((coord[3] * depth + coord[2]) * height + coord[0]) * width + coord[1]) * channels + channel;
  float      sum       = 0.0f;
  for (int i = 0; i < length; ++i) sum += feats[(start + i) * channels + channel];
  out[out_index] = sum;
}

__global__ void bev_pool_backward_kernel(const float* grad_out,
                                         const int*   coords,
                                         const int*   interval_starts,
                                         const int*   interval_lengths,
                                         float*       grad_feats,
                                         int          intervals,
                                         int          channels,
                                         int          depth,
                                         int          height,
                                         int          width) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = intervals * channels;
  if (index >= total) return;
  const int  interval  = index / channels;
  const int  channel   = index % channels;
  const int  start     = interval_starts[interval];
  const int  length    = interval_lengths[interval];
  const int* coord     = coords + start * 4;
  const int  out_index = (((coord[3] * depth + coord[2]) * height + coord[0]) * width + coord[1]) * channels + channel;
  for (int i = 0; i < length; ++i) grad_feats[(start + i) * channels + channel] = grad_out[out_index];
}

__global__ void bev_pool_forward_packed4_kernel(const float4* feats,
                                                const int*    coords,
                                                const int*    interval_starts,
                                                const int*    interval_lengths,
                                                float4*       out,
                                                int           intervals,
                                                int           packed_channels,
                                                int           depth,
                                                int           height,
                                                int           width) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= intervals * packed_channels) return;
  const int  interval = index / packed_channels;
  const int  channel  = index % packed_channels;
  const int  start    = interval_starts[interval];
  const int  length   = interval_lengths[interval];
  const int* coord    = coords + start * 4;
  float4     sum      = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int i = 0; i < length; ++i) {
    const float4 value = feats[(start + i) * packed_channels + channel];
    sum.x += value.x;
    sum.y += value.y;
    sum.z += value.z;
    sum.w += value.w;
  }
  const int out_index =
      (((coord[3] * depth + coord[2]) * height + coord[0]) * width + coord[1]) * packed_channels + channel;
  out[out_index] = sum;
}

__global__ void bev_pool_backward_packed4_kernel(const float4* grad_out,
                                                 const int*    coords,
                                                 const int*    interval_starts,
                                                 const int*    interval_lengths,
                                                 float4*       grad_feats,
                                                 int           intervals,
                                                 int           packed_channels,
                                                 int           depth,
                                                 int           height,
                                                 int           width) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= intervals * packed_channels) return;
  const int  interval = index / packed_channels;
  const int  channel  = index % packed_channels;
  const int  start    = interval_starts[interval];
  const int  length   = interval_lengths[interval];
  const int* coord    = coords + start * 4;
  const int  out_index =
      (((coord[3] * depth + coord[2]) * height + coord[0]) * width + coord[1]) * packed_channels + channel;
  const float4 value = grad_out[out_index];
  for (int i = 0; i < length; ++i) grad_feats[(start + i) * packed_channels + channel] = value;
}
}  // namespace

void bev_pool_forward(ffi::TensorView feats,
                      ffi::TensorView coords,
                      ffi::TensorView interval_starts,
                      ffi::TensorView interval_lengths,
                      ffi::TensorView out) {
  check_musa_contiguous(feats, "feats", kDLFloat32);
  check_musa_contiguous(coords, "coords", kDLInt32);
  check_musa_contiguous(interval_starts, "interval_starts", kDLInt32);
  check_musa_contiguous(interval_lengths, "interval_lengths", kDLInt32);
  check_musa_contiguous(out, "out", kDLFloat32);
  TVM_FFI_ICHECK_EQ(feats.ndim(), 2) << "feats must have shape [N, C]";
  TVM_FFI_ICHECK_EQ(coords.ndim(), 2) << "coords must have shape [N, 4]";
  TVM_FFI_ICHECK_EQ(dim(coords, 0), dim(feats, 0)) << "coords and feats point counts differ";
  TVM_FFI_ICHECK_EQ(dim(coords, 1), 4) << "coords must have four columns";
  TVM_FFI_ICHECK_EQ(out.ndim(), 5) << "out must have shape [B, D, H, W, C]";
  TVM_FFI_ICHECK_EQ(dim(out, 4), dim(feats, 1)) << "out channel count differs from feats";
  TVM_FFI_ICHECK_EQ(interval_starts.numel(), interval_lengths.numel()) << "interval sizes differ";
  const int64_t total = interval_starts.numel() * dim(feats, 1);
  if (total == 0) return;
  const int channels = static_cast<int>(dim(feats, 1));
  if (dim(feats, 1) % 4 == 0) {
    const int packed_channels = channels / 4;
    bev_pool_forward_packed4_kernel<<<blocks_for(total / 4), kThreads, 0, stream(feats)>>>(
        reinterpret_cast<const float4*>(feats.data_ptr()),
        static_cast<int*>(coords.data_ptr()),
        static_cast<int*>(interval_starts.data_ptr()),
        static_cast<int*>(interval_lengths.data_ptr()),
        reinterpret_cast<float4*>(out.data_ptr()),
        static_cast<int>(interval_starts.numel()),
        packed_channels,
        static_cast<int>(dim(out, 1)),
        static_cast<int>(dim(out, 2)),
        static_cast<int>(dim(out, 3)));
  } else {
    bev_pool_forward_kernel<<<blocks_for(total), kThreads, 0, stream(feats)>>>(
        static_cast<float*>(feats.data_ptr()),
        static_cast<int*>(coords.data_ptr()),
        static_cast<int*>(interval_starts.data_ptr()),
        static_cast<int*>(interval_lengths.data_ptr()),
        static_cast<float*>(out.data_ptr()),
        static_cast<int>(interval_starts.numel()),
        channels,
        static_cast<int>(dim(out, 1)),
        static_cast<int>(dim(out, 2)),
        static_cast<int>(dim(out, 3)));
  }
  check_last_error("bev_pool_forward");
}

void bev_pool_backward(ffi::TensorView grad_out,
                       ffi::TensorView coords,
                       ffi::TensorView interval_starts,
                       ffi::TensorView interval_lengths,
                       ffi::TensorView grad_feats) {
  check_musa_contiguous(grad_out, "grad_out", kDLFloat32);
  check_musa_contiguous(coords, "coords", kDLInt32);
  check_musa_contiguous(interval_starts, "interval_starts", kDLInt32);
  check_musa_contiguous(interval_lengths, "interval_lengths", kDLInt32);
  check_musa_contiguous(grad_feats, "grad_feats", kDLFloat32);
  TVM_FFI_ICHECK_EQ(grad_out.ndim(), 5) << "grad_out must have shape [B, D, H, W, C]";
  TVM_FFI_ICHECK_EQ(coords.ndim(), 2) << "coords must have shape [N, 4]";
  TVM_FFI_ICHECK_EQ(dim(coords, 0), dim(grad_feats, 0)) << "coords and grad_feats point counts differ";
  TVM_FFI_ICHECK_EQ(dim(coords, 1), 4) << "coords must have four columns";
  TVM_FFI_ICHECK_EQ(dim(grad_out, 4), dim(grad_feats, 1)) << "grad_out channel count differs from grad_feats";
  TVM_FFI_ICHECK_EQ(interval_starts.numel(), interval_lengths.numel()) << "interval sizes differ";
  const int64_t total = interval_starts.numel() * dim(grad_feats, 1);
  if (total == 0) return;
  const int channels = static_cast<int>(dim(grad_feats, 1));
  if (dim(grad_feats, 1) % 4 == 0) {
    const int packed_channels = channels / 4;
    bev_pool_backward_packed4_kernel<<<blocks_for(total / 4), kThreads, 0, stream(grad_out)>>>(
        reinterpret_cast<const float4*>(grad_out.data_ptr()),
        static_cast<int*>(coords.data_ptr()),
        static_cast<int*>(interval_starts.data_ptr()),
        static_cast<int*>(interval_lengths.data_ptr()),
        reinterpret_cast<float4*>(grad_feats.data_ptr()),
        static_cast<int>(interval_starts.numel()),
        packed_channels,
        static_cast<int>(dim(grad_out, 1)),
        static_cast<int>(dim(grad_out, 2)),
        static_cast<int>(dim(grad_out, 3)));
  } else {
    bev_pool_backward_kernel<<<blocks_for(total), kThreads, 0, stream(grad_out)>>>(
        static_cast<float*>(grad_out.data_ptr()),
        static_cast<int*>(coords.data_ptr()),
        static_cast<int*>(interval_starts.data_ptr()),
        static_cast<int*>(interval_lengths.data_ptr()),
        static_cast<float*>(grad_feats.data_ptr()),
        static_cast<int>(interval_starts.numel()),
        channels,
        static_cast<int>(dim(grad_out, 1)),
        static_cast<int>(dim(grad_out, 2)),
        static_cast<int>(dim(grad_out, 3)));
  }
  check_last_error("bev_pool_backward");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bev_pool_forward, bev_pool_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bev_pool_backward, bev_pool_backward);
