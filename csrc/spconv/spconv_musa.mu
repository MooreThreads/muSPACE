// Torch-ABI-independent indice convolution kernels for BEVFusion SparseEncoder.

#include <musa_fp16.h>
#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <type_traits>

#include "../ffi/dtype_utils.hpp"
#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::check_same_musa_device;
using muspace::ffi_utils::stream;

namespace {

inline constexpr DLDataType kDLFloat16 = DLDataType{kDLFloat, 16, 1};

template <typename T>
__device__ inline float as_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ inline float as_float<__half>(__half value) {
  return __half2float(value);
}

template <typename T>
__device__ inline T from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ inline __half from_float<__half>(float value) {
  return __float2half(value);
}

template <typename V>
inline bool is_aligned(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % alignof(V) == 0;
}

template <typename T>
__device__ inline T zero_of();

template <>
__device__ inline float zero_of<float>() {
  return 0.0F;
}

template <>
__device__ inline __half zero_of<__half>() {
  return __float2half(0.0F);
}

template <>
__device__ inline float4 zero_of<float4>() {
  return make_float4(0.0F, 0.0F, 0.0F, 0.0F);
}

// One warp per pair slot, one grid row per kernel offset.  Keeping the offset
// on ``blockIdx.y`` avoids the 64-bit division a flattened index would need and
// lets a whole warp share the single ``pairs`` / ``pair_num`` load.
constexpr int kLaneThreads   = 32;
constexpr int kSlotsPerBlock = 8;

// Gather and scatter-add over a run of kernel offsets.  Slots past ``pair_num``
// are zero-filled on gather and skipped on scatter, so one capacity-padded GEMM
// can cover the whole run instead of one GEMM per offset.  ``V`` is the copy
// unit: ``float4`` whenever the row stride is 16-byte aligned.
template <typename V>
__global__ void sparse_gather_kernel(
    const V* features, const int* pairs, const int* pair_num, V* buffer, int capacity, int channels, int plane) {
  const int kernel_index = blockIdx.y;
  const int pair_slot    = blockIdx.x * blockDim.y + threadIdx.y;
  if (pair_slot >= capacity) return;
  V*  destination = buffer + (static_cast<int64_t>(kernel_index) * capacity + pair_slot) * channels;
  int input_index = -1;
  if (pair_slot < pair_num[kernel_index]) {
    input_index = pairs[(static_cast<int64_t>(kernel_index) * 2 + plane) * capacity + pair_slot];
  }
  if (input_index < 0) {
    for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
      destination[channel] = zero_of<V>();
    }
    return;
  }
  const V* source = features + static_cast<int64_t>(input_index) * channels;
  for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    destination[channel] = source[channel];
  }
}

template <typename T>
__global__ void sparse_scatter_add_kernel(
    const T* buffer, const int* pairs, const int* pair_num, T* output, int capacity, int channels, int plane) {
  const int kernel_index = blockIdx.y;
  const int pair_slot    = blockIdx.x * blockDim.y + threadIdx.y;
  if (pair_slot >= capacity || pair_slot >= pair_num[kernel_index]) return;
  const int output_index = pairs[(static_cast<int64_t>(kernel_index) * 2 + plane) * capacity + pair_slot];
  if (output_index < 0) return;
  const T* source      = buffer + (static_cast<int64_t>(kernel_index) * capacity + pair_slot) * channels;
  T*       destination = output + static_cast<int64_t>(output_index) * channels;
  for (int channel = threadIdx.x; channel < channels; channel += blockDim.x) {
    atomicAdd(&destination[channel], source[channel]);
  }
}

template <typename T>
__global__ void indice_conv_kernel(const T*   features,
                                   const T*   filters,
                                   const int* pairs,
                                   const int* pair_num,
                                   T*         output,
                                   int        kernel_volume,
                                   int        capacity,
                                   int        channels_in,
                                   int        channels_out) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kernel_volume * capacity * channels_out;
  if (index >= total) return;
  const int channel_out  = index % channels_out;
  const int pair_slot    = (index / channels_out) % capacity;
  const int kernel_index = index / (capacity * channels_out);
  if (pair_slot >= pair_num[kernel_index]) return;

  const int pair_base    = (kernel_index * 2) * capacity;
  const int input_index  = pairs[pair_base + pair_slot];
  const int output_index = pairs[pair_base + capacity + pair_slot];
  if (input_index < 0 || output_index < 0) return;

  float value = 0.0F;
  for (int channel_in = 0; channel_in < channels_in; ++channel_in) {
    value += as_float(features[input_index * channels_in + channel_in]) *
             as_float(filters[(kernel_index * channels_in + channel_in) * channels_out + channel_out]);
  }
  atomicAdd(&output[output_index * channels_out + channel_out], from_float<T>(value));
}

template <typename T>
__global__ void indice_conv_backward_kernel(const T*   features,
                                            const T*   filters,
                                            const T*   grad_output,
                                            const int* pairs,
                                            const int* pair_num,
                                            T*         grad_features,
                                            T*         grad_filters,
                                            int        kernel_volume,
                                            int        capacity,
                                            int        channels_in,
                                            int        channels_out) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kernel_volume * capacity;
  if (index >= total) return;
  const int pair_slot    = index % capacity;
  const int kernel_index = index / capacity;
  if (pair_slot >= pair_num[kernel_index]) return;

  const int pair_base    = (kernel_index * 2) * capacity;
  const int input_index  = pairs[pair_base + pair_slot];
  const int output_index = pairs[pair_base + capacity + pair_slot];
  if (input_index < 0 || output_index < 0) return;

  for (int channel_in = 0; channel_in < channels_in; ++channel_in) {
    float sum = 0.0F;
    for (int channel_out = 0; channel_out < channels_out; ++channel_out) {
      sum += as_float(grad_output[output_index * channels_out + channel_out]) *
             as_float(filters[(kernel_index * channels_in + channel_in) * channels_out + channel_out]);
    }
    atomicAdd(&grad_features[input_index * channels_in + channel_in], from_float<T>(sum));
  }
  for (int channel_in = 0; channel_in < channels_in; ++channel_in) {
    const float input_value = as_float(features[input_index * channels_in + channel_in]);
    for (int channel_out = 0; channel_out < channels_out; ++channel_out) {
      const float value = input_value * as_float(grad_output[output_index * channels_out + channel_out]);
      atomicAdd(&grad_filters[(kernel_index * channels_in + channel_in) * channels_out + channel_out],
                from_float<T>(value));
    }
  }
}

template <typename T>
void check_inputs(ffi::TensorView features,
                  ffi::TensorView filters,
                  ffi::TensorView pairs,
                  ffi::TensorView pair_num,
                  ffi::TensorView output,
                  const char*     name) {
  check_musa_contiguous(features, name, std::is_same_v<T, float> ? kDLFloat32 : kDLFloat16);
  check_musa_contiguous(filters, "filters", std::is_same_v<T, float> ? kDLFloat32 : kDLFloat16);
  check_musa_contiguous(pairs, "indice_pairs", kDLInt32);
  check_musa_contiguous(pair_num, "indice_pair_num", kDLInt32);
  check_musa_contiguous(output, "output", std::is_same_v<T, float> ? kDLFloat32 : kDLFloat16);
  check_same_musa_device(features, filters, "filters");
  check_same_musa_device(features, pairs, "indice_pairs");
  check_same_musa_device(features, pair_num, "indice_pair_num");
  check_same_musa_device(features, output, "output");
  TVM_FFI_ICHECK_EQ(features.ndim(), 2);
  TVM_FFI_ICHECK_EQ(filters.ndim(), 3);
  TVM_FFI_ICHECK_EQ(pairs.ndim(), 3);
  TVM_FFI_ICHECK_EQ(pair_num.ndim(), 1);
  TVM_FFI_ICHECK_EQ(output.ndim(), 2);
  TVM_FFI_ICHECK_EQ(pairs.size(0), filters.size(0));
  TVM_FFI_ICHECK_EQ(pairs.size(1), 2);
  TVM_FFI_ICHECK_EQ(pair_num.size(0), filters.size(0));
  TVM_FFI_ICHECK_EQ(filters.size(1), features.size(1));
  TVM_FFI_ICHECK_EQ(output.size(1), filters.size(2));
}

template <typename T>
void indice_conv(ffi::TensorView features,
                 ffi::TensorView filters,
                 ffi::TensorView pairs,
                 ffi::TensorView pair_num,
                 ffi::TensorView output) {
  check_inputs<T>(features, filters, pairs, pair_num, output, "features");
  const int     kernel_volume = static_cast<int>(filters.size(0));
  const int     capacity      = static_cast<int>(pairs.size(2));
  const int     channels_in   = static_cast<int>(features.size(1));
  const int     channels_out  = static_cast<int>(filters.size(2));
  const int64_t output_bytes  = output.numel() * sizeof(T);
  musaMemsetAsync(output.data_ptr(), 0, output_bytes, stream(output));
  const int total = kernel_volume * capacity * channels_out;
  if (total > 0) {
    indice_conv_kernel<T><<<(total + 255) / 256, 256, 0, stream(output)>>>(static_cast<const T*>(features.data_ptr()),
                                                                           static_cast<const T*>(filters.data_ptr()),
                                                                           static_cast<const int*>(pairs.data_ptr()),
                                                                           static_cast<const int*>(pair_num.data_ptr()),
                                                                           static_cast<T*>(output.data_ptr()),
                                                                           kernel_volume,
                                                                           capacity,
                                                                           channels_in,
                                                                           channels_out);
  }
  check_last_error("spconv indice convolution");
}

template <typename T>
void indice_conv_backward(ffi::TensorView features,
                          ffi::TensorView filters,
                          ffi::TensorView grad_output,
                          ffi::TensorView pairs,
                          ffi::TensorView pair_num,
                          ffi::TensorView grad_features,
                          ffi::TensorView grad_filters) {
  check_inputs<T>(features, filters, pairs, pair_num, grad_output, "features");
  check_musa_contiguous(grad_features, "grad_features", std::is_same_v<T, float> ? kDLFloat32 : kDLFloat16);
  check_musa_contiguous(grad_filters, "grad_filters", std::is_same_v<T, float> ? kDLFloat32 : kDLFloat16);
  check_same_musa_device(features, grad_features, "grad_features");
  check_same_musa_device(features, grad_filters, "grad_filters");
  TVM_FFI_ICHECK_EQ(grad_features.size(0), features.size(0));
  TVM_FFI_ICHECK_EQ(grad_features.size(1), features.size(1));
  TVM_FFI_ICHECK_EQ(grad_filters.size(0), filters.size(0));
  TVM_FFI_ICHECK_EQ(grad_filters.size(1), filters.size(1));
  TVM_FFI_ICHECK_EQ(grad_filters.size(2), filters.size(2));
  musaMemsetAsync(grad_features.data_ptr(), 0, grad_features.numel() * sizeof(T), stream(grad_features));
  musaMemsetAsync(grad_filters.data_ptr(), 0, grad_filters.numel() * sizeof(T), stream(grad_filters));
  const int total = static_cast<int>(filters.size(0) * pairs.size(2));
  if (total > 0) {
    indice_conv_backward_kernel<T>
        <<<(total + 255) / 256, 256, 0, stream(features)>>>(static_cast<const T*>(features.data_ptr()),
                                                            static_cast<const T*>(filters.data_ptr()),
                                                            static_cast<const T*>(grad_output.data_ptr()),
                                                            static_cast<const int*>(pairs.data_ptr()),
                                                            static_cast<const int*>(pair_num.data_ptr()),
                                                            static_cast<T*>(grad_features.data_ptr()),
                                                            static_cast<T*>(grad_filters.data_ptr()),
                                                            static_cast<int>(filters.size(0)),
                                                            static_cast<int>(pairs.size(2)),
                                                            static_cast<int>(features.size(1)),
                                                            static_cast<int>(filters.size(2)));
  }
  check_last_error("spconv indice convolution backward");
}

template <typename T>
void check_pair_buffer(
    ffi::TensorView pairs, ffi::TensorView pair_num, ffi::TensorView buffer, ffi::TensorView features, int plane) {
  constexpr DLDataType kValueType = std::is_same_v<T, float> ? kDLFloat32 : kDLFloat16;
  check_musa_contiguous(pairs, "indice_pairs", kDLInt32);
  check_musa_contiguous(pair_num, "indice_pair_num", kDLInt32);
  check_musa_contiguous(buffer, "buffer", kValueType);
  check_same_musa_device(features, pairs, "indice_pairs");
  check_same_musa_device(features, pair_num, "indice_pair_num");
  check_same_musa_device(features, buffer, "buffer");
  TVM_FFI_ICHECK_EQ(pairs.ndim(), 3);
  TVM_FFI_ICHECK_EQ(pairs.size(1), 2);
  TVM_FFI_ICHECK_EQ(pair_num.ndim(), 1);
  TVM_FFI_ICHECK_EQ(pair_num.size(0), pairs.size(0));
  TVM_FFI_ICHECK_EQ(buffer.ndim(), 2);
  TVM_FFI_ICHECK_EQ(buffer.size(0), pairs.size(0) * pairs.size(2));
  TVM_FFI_ICHECK_EQ(buffer.size(1), features.size(1));
  TVM_FFI_ICHECK(plane == 0 || plane == 1) << "plane must be 0 or 1";
}

template <typename T>
void sparse_gather(
    ffi::TensorView features, ffi::TensorView pairs, ffi::TensorView pair_num, ffi::TensorView buffer, int plane) {
  check_musa_contiguous(features, "features", std::is_same_v<T, float> ? kDLFloat32 : kDLFloat16);
  TVM_FFI_ICHECK_EQ(features.ndim(), 2);
  check_pair_buffer<T>(pairs, pair_num, buffer, features, plane);
  const int kernel_volume = static_cast<int>(pairs.size(0));
  const int capacity      = static_cast<int>(pairs.size(2));
  const int channels      = static_cast<int>(features.size(1));
  if (kernel_volume > 0 && capacity > 0 && channels > 0) {
    const dim3    block(kLaneThreads, kSlotsPerBlock);
    const dim3    grid((capacity + kSlotsPerBlock - 1) / kSlotsPerBlock, kernel_volume);
    const int64_t row_bytes = static_cast<int64_t>(channels) * sizeof(T);
    if (row_bytes % sizeof(float4) == 0 && is_aligned<float4>(features.data_ptr()) &&
        is_aligned<float4>(buffer.data_ptr())) {
      sparse_gather_kernel<float4><<<grid, block, 0, stream(buffer)>>>(static_cast<const float4*>(features.data_ptr()),
                                                                       static_cast<const int*>(pairs.data_ptr()),
                                                                       static_cast<const int*>(pair_num.data_ptr()),
                                                                       static_cast<float4*>(buffer.data_ptr()),
                                                                       capacity,
                                                                       static_cast<int>(row_bytes / sizeof(float4)),
                                                                       plane);
    } else {
      sparse_gather_kernel<T><<<grid, block, 0, stream(buffer)>>>(static_cast<const T*>(features.data_ptr()),
                                                                  static_cast<const int*>(pairs.data_ptr()),
                                                                  static_cast<const int*>(pair_num.data_ptr()),
                                                                  static_cast<T*>(buffer.data_ptr()),
                                                                  capacity,
                                                                  channels,
                                                                  plane);
    }
  }
  check_last_error("spconv sparse gather");
}

template <typename T>
void sparse_scatter_add(
    ffi::TensorView buffer, ffi::TensorView pairs, ffi::TensorView pair_num, ffi::TensorView output, int plane) {
  check_musa_contiguous(output, "output", std::is_same_v<T, float> ? kDLFloat32 : kDLFloat16);
  TVM_FFI_ICHECK_EQ(output.ndim(), 2);
  check_pair_buffer<T>(pairs, pair_num, buffer, output, plane);
  const int kernel_volume = static_cast<int>(pairs.size(0));
  const int capacity      = static_cast<int>(pairs.size(2));
  const int channels      = static_cast<int>(output.size(1));
  if (kernel_volume > 0 && capacity > 0 && channels > 0) {
    const dim3 block(kLaneThreads, kSlotsPerBlock);
    const dim3 grid((capacity + kSlotsPerBlock - 1) / kSlotsPerBlock, kernel_volume);
    sparse_scatter_add_kernel<T><<<grid, block, 0, stream(buffer)>>>(static_cast<const T*>(buffer.data_ptr()),
                                                                     static_cast<const int*>(pairs.data_ptr()),
                                                                     static_cast<const int*>(pair_num.data_ptr()),
                                                                     static_cast<T*>(output.data_ptr()),
                                                                     capacity,
                                                                     channels,
                                                                     plane);
  }
  check_last_error("spconv sparse scatter add");
}

}  // namespace

void spconv_sparse_gather_fp32(
    ffi::TensorView features, ffi::TensorView pairs, ffi::TensorView pair_num, ffi::TensorView buffer, int64_t plane) {
  sparse_gather<float>(features, pairs, pair_num, buffer, static_cast<int>(plane));
}

void spconv_sparse_gather_half(
    ffi::TensorView features, ffi::TensorView pairs, ffi::TensorView pair_num, ffi::TensorView buffer, int64_t plane) {
  sparse_gather<__half>(features, pairs, pair_num, buffer, static_cast<int>(plane));
}

void spconv_sparse_scatter_add_fp32(
    ffi::TensorView buffer, ffi::TensorView pairs, ffi::TensorView pair_num, ffi::TensorView output, int64_t plane) {
  sparse_scatter_add<float>(buffer, pairs, pair_num, output, static_cast<int>(plane));
}

void spconv_sparse_scatter_add_half(
    ffi::TensorView buffer, ffi::TensorView pairs, ffi::TensorView pair_num, ffi::TensorView output, int64_t plane) {
  sparse_scatter_add<__half>(buffer, pairs, pair_num, output, static_cast<int>(plane));
}

void spconv_indice_conv_fp32(ffi::TensorView features,
                             ffi::TensorView filters,
                             ffi::TensorView pairs,
                             ffi::TensorView pair_num,
                             ffi::TensorView output) {
  indice_conv<float>(features, filters, pairs, pair_num, output);
}

void spconv_indice_conv_half(ffi::TensorView features,
                             ffi::TensorView filters,
                             ffi::TensorView pairs,
                             ffi::TensorView pair_num,
                             ffi::TensorView output) {
  indice_conv<__half>(features, filters, pairs, pair_num, output);
}

void spconv_indice_conv_backward_fp32(ffi::TensorView features,
                                      ffi::TensorView filters,
                                      ffi::TensorView grad_output,
                                      ffi::TensorView pairs,
                                      ffi::TensorView pair_num,
                                      ffi::TensorView grad_features,
                                      ffi::TensorView grad_filters) {
  indice_conv_backward<float>(features, filters, grad_output, pairs, pair_num, grad_features, grad_filters);
}

void spconv_indice_conv_backward_half(ffi::TensorView features,
                                      ffi::TensorView filters,
                                      ffi::TensorView grad_output,
                                      ffi::TensorView pairs,
                                      ffi::TensorView pair_num,
                                      ffi::TensorView grad_features,
                                      ffi::TensorView grad_filters) {
  indice_conv_backward<__half>(features, filters, grad_output, pairs, pair_num, grad_features, grad_filters);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(spconv_indice_conv_fp32, spconv_indice_conv_fp32);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(spconv_indice_conv_half, spconv_indice_conv_half);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(spconv_indice_conv_backward_fp32, spconv_indice_conv_backward_fp32);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(spconv_indice_conv_backward_half, spconv_indice_conv_backward_half);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(spconv_sparse_gather_fp32, spconv_sparse_gather_fp32);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(spconv_sparse_gather_half, spconv_sparse_gather_half);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(spconv_sparse_scatter_add_fp32, spconv_sparse_scatter_add_fp32);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(spconv_sparse_scatter_add_half, spconv_sparse_scatter_add_half);
