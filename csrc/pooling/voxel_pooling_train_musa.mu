#include <musa_fp16.h>
#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using ::same_dtype;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::dim;
using muspace::ffi_utils::stream;

namespace {
constexpr DLDataType kDLFloat16 = DLDataType{kDLFloat, 16, 1};
constexpr int        kThreadsX  = 32;
constexpr int        kThreadsY  = 4;
constexpr int        kThreads   = kThreadsX * kThreadsY;

template <typename T>
__device__ inline void add_value(T* destination, T value) {
  atomicAdd(destination, value);
}

template <typename T>
__global__ void voxel_pooling_train_forward_kernel(int        batch,
                                                   int        points,
                                                   int        channels,
                                                   int        voxel_x,
                                                   int        voxel_y,
                                                   int        voxel_z,
                                                   const int* geom,
                                                   const T*   features,
                                                   T*         output,
                                                   int*       pos_memo) {
  const int sample = blockIdx.x * kThreads + threadIdx.y * kThreadsX + threadIdx.x;
  const int total  = batch * points;
  if (sample < total) {
    const int x = geom[sample * 3];
    const int y = geom[sample * 3 + 1];
    const int z = geom[sample * 3 + 2];
    if (x >= 0 && x < voxel_x && y >= 0 && y < voxel_y && z >= 0 && z < voxel_z) {
      pos_memo[sample * 3]     = sample / points;
      pos_memo[sample * 3 + 1] = y;
      pos_memo[sample * 3 + 2] = x;
    }
  }
  __syncthreads();
  for (int sample_in_block = threadIdx.y; sample_in_block < kThreads && blockIdx.x * kThreads + sample_in_block < total;
       sample_in_block += kThreadsY) {
    const int sample_index = blockIdx.x * kThreads + sample_in_block;
    const int x            = geom[sample_index * 3];
    const int y            = geom[sample_index * 3 + 1];
    const int z            = geom[sample_index * 3 + 2];
    if (x < 0 || x >= voxel_x || y < 0 || y >= voxel_y || z < 0 || z >= voxel_z) continue;
    for (int channel = threadIdx.x; channel < channels; channel += kThreadsX) {
      const int offset = ((sample_index / points * voxel_y + y) * voxel_x + x) * channels + channel;
      add_value(output + offset, features[sample_index * channels + channel]);
    }
  }
}

template <typename T>
void launch_forward(ffi::TensorView geom,
                    ffi::TensorView features,
                    ffi::TensorView output,
                    ffi::TensorView pos_memo,
                    int             voxel_x,
                    int             voxel_y,
                    int             voxel_z) {
  const int batch    = static_cast<int>(dim(features, 0));
  const int points   = static_cast<int>(dim(features, 1));
  const int channels = static_cast<int>(dim(features, 2));
  if (batch == 0 || points == 0 || channels == 0) return;
  const int blocks = (batch * points + kThreads - 1) / kThreads;
  voxel_pooling_train_forward_kernel<T>
      <<<blocks, dim3(kThreadsX, kThreadsY), 0, stream(features)>>>(batch,
                                                                    points,
                                                                    channels,
                                                                    voxel_x,
                                                                    voxel_y,
                                                                    voxel_z,
                                                                    static_cast<const int*>(geom.data_ptr()),
                                                                    static_cast<const T*>(features.data_ptr()),
                                                                    static_cast<T*>(output.data_ptr()),
                                                                    static_cast<int*>(pos_memo.data_ptr()));
}

template <typename T>
__global__ void voxel_pooling_train_backward_typed_kernel(int        batch,
                                                          int        points,
                                                          int        channels,
                                                          int        voxel_x,
                                                          int        voxel_y,
                                                          const int* pos_memo,
                                                          const T*   grad_output,
                                                          T*         grad_features) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * points * channels;
  if (index >= total) return;
  const int  sample  = index / channels;
  const int  channel = index % channels;
  const int* pos     = pos_memo + sample * 3;
  if (pos[0] < 0) return;
  const int output_index = ((pos[0] * voxel_y + pos[1]) * voxel_x + pos[2]) * channels + channel;
  grad_features[index]   = grad_output[output_index];
}

template <typename T>
void launch_backward(
    ffi::TensorView grad_output, ffi::TensorView pos_memo, ffi::TensorView grad_features, int voxel_x, int voxel_y) {
  const int     batch    = static_cast<int>(dim(grad_features, 0));
  const int     points   = static_cast<int>(dim(grad_features, 1));
  const int     channels = static_cast<int>(dim(grad_features, 2));
  const int64_t total    = static_cast<int64_t>(batch) * points * channels;
  if (total == 0) return;
  const int blocks = static_cast<int>((total + 255) / 256);
  voxel_pooling_train_backward_typed_kernel<T>
      <<<blocks, 256, 0, stream(grad_output)>>>(batch,
                                                points,
                                                channels,
                                                voxel_x,
                                                voxel_y,
                                                static_cast<const int*>(pos_memo.data_ptr()),
                                                static_cast<const T*>(grad_output.data_ptr()),
                                                static_cast<T*>(grad_features.data_ptr()));
}
}  // namespace

void voxel_pooling_train_forward(ffi::TensorView geom_xyz,
                                 ffi::TensorView input_features,
                                 ffi::TensorView output_features,
                                 ffi::TensorView pos_memo,
                                 int             voxel_x,
                                 int             voxel_y,
                                 int             voxel_z) {
  check_musa_contiguous(geom_xyz, "geom_xyz", kDLInt32);
  check_musa_contiguous(pos_memo, "pos_memo", kDLInt32);
  TVM_FFI_ICHECK(same_dtype(input_features.dtype(), kDLFloat16) || same_dtype(input_features.dtype(), kDLFloat32))
      << "input_features must use float16 or float32";
  TVM_FFI_ICHECK(input_features.IsContiguous()) << "input_features must be contiguous";
  TVM_FFI_ICHECK(same_dtype(output_features.dtype(), input_features.dtype())) << "output dtype differs from input";
  TVM_FFI_ICHECK(output_features.IsContiguous()) << "output_features must be contiguous";
  TVM_FFI_ICHECK_EQ(geom_xyz.ndim(), 3) << "geom_xyz must have shape [B, N, 3]";
  TVM_FFI_ICHECK_EQ(input_features.ndim(), 3) << "input_features must have shape [B, N, C]";
  TVM_FFI_ICHECK_EQ(dim(geom_xyz, 0), dim(input_features, 0));
  TVM_FFI_ICHECK_EQ(dim(geom_xyz, 1), dim(input_features, 1));
  TVM_FFI_ICHECK_EQ(dim(geom_xyz, 2), 3);
  TVM_FFI_ICHECK_EQ(dim(pos_memo, 0), dim(input_features, 0));
  TVM_FFI_ICHECK_EQ(dim(pos_memo, 1), dim(input_features, 1));
  TVM_FFI_ICHECK_EQ(dim(pos_memo, 2), 3);
  TVM_FFI_ICHECK_GE(voxel_x, 1);
  TVM_FFI_ICHECK_GE(voxel_y, 1);
  TVM_FFI_ICHECK_GE(voxel_z, 1);
  if (same_dtype(input_features.dtype(), kDLFloat32)) {
    check_musa_contiguous(output_features, "output_features", kDLFloat32);
    launch_forward<float>(geom_xyz, input_features, output_features, pos_memo, voxel_x, voxel_y, voxel_z);
  } else {
    check_musa_contiguous(output_features, "output_features", kDLFloat16);
    launch_forward<__half>(geom_xyz, input_features, output_features, pos_memo, voxel_x, voxel_y, voxel_z);
  }
  check_last_error("voxel_pooling_train_forward");
}

void voxel_pooling_train_backward(
    ffi::TensorView grad_output, ffi::TensorView pos_memo, ffi::TensorView grad_features, int voxel_x, int voxel_y) {
  check_musa_contiguous(pos_memo, "pos_memo", kDLInt32);
  TVM_FFI_ICHECK(same_dtype(grad_output.dtype(), kDLFloat16) || same_dtype(grad_output.dtype(), kDLFloat32))
      << "grad_output must use float16 or float32";
  TVM_FFI_ICHECK(same_dtype(grad_features.dtype(), grad_output.dtype())) << "gradient dtype differs";
  TVM_FFI_ICHECK(grad_output.IsContiguous()) << "grad_output must be contiguous";
  TVM_FFI_ICHECK(grad_features.IsContiguous()) << "grad_features must be contiguous";
  TVM_FFI_ICHECK_EQ(grad_output.ndim(), 4) << "grad_output must have shape [B, C, Y, X]";
  TVM_FFI_ICHECK_EQ(grad_features.ndim(), 3) << "grad_features must have shape [B, N, C]";
  TVM_FFI_ICHECK_EQ(dim(pos_memo, 0), dim(grad_features, 0));
  TVM_FFI_ICHECK_EQ(dim(pos_memo, 1), dim(grad_features, 1));
  TVM_FFI_ICHECK_EQ(dim(pos_memo, 2), 3);
  if (same_dtype(grad_output.dtype(), kDLFloat32)) {
    check_musa_contiguous(grad_output, "grad_output", kDLFloat32);
    launch_backward<float>(grad_output, pos_memo, grad_features, voxel_x, voxel_y);
  } else {
    check_musa_contiguous(grad_output, "grad_output", kDLFloat16);
    launch_backward<__half>(grad_output, pos_memo, grad_features, voxel_x, voxel_y);
  }
  check_last_error("voxel_pooling_train_backward");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(voxel_pooling_train_forward, voxel_pooling_train_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(voxel_pooling_train_backward, voxel_pooling_train_backward);
