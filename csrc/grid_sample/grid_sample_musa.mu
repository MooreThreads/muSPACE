#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include <climits>
#include <cstdint>

#include "../common/sampling.muh"
#include "../ffi/dtype_utils.hpp"
#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::stream;

namespace {
constexpr DLDataType kDLFloat16 = DLDataType{kDLFloat, 16, 1};

template <typename T>
__device__ __forceinline__ float as_float(T value) {
  return static_cast<float>(value);
}
template <>
__device__ __forceinline__ float as_float(__half value) {
  return __half2float(value);
}
template <>
__device__ __forceinline__ float as_float(__mt_bfloat16 value) {
  return __bfloat162float(value);
}
template <typename T>
__device__ __forceinline__ T from_float(float value) {
  return static_cast<T>(value);
}
template <>
__device__ __forceinline__ __half from_float(float value) {
  return __float2half(value);
}
template <>
__device__ __forceinline__ __mt_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}

// Forward splits channels across blocks; the corner loads are issued together so
// several requests stay in flight.
template <typename T, int Dim, bool Nearest>
__global__ void sample_forward_kernel(const T* input,
                                      const T* grid,
                                      T*       output,
                                      int64_t  locations,
                                      int64_t  channels,
                                      int64_t  output_spatial,
                                      int      depth,
                                      int      height,
                                      int      width,
                                      int      padding_mode,
                                      bool     align_corners) {
  const int64_t location = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (location >= locations) return;
  constexpr int kCorners    = 1 << Dim;
  constexpr int kActive     = Nearest ? 1 : kCorners;
  const int64_t batch       = location / output_spatial;
  const int64_t spatial     = location % output_spatial;
  const int     sizes[3]    = {width, height, depth};
  int           lower[3]    = {0, 0, 0};
  float         fraction[3] = {0.0F, 0.0F, 0.0F};
  for (int axis = 0; axis < Dim; ++axis) {
    float       scale;
    const float coord = muspace::sampling::source_coordinate(
        as_float(grid[location * Dim + axis]), sizes[axis], padding_mode, align_corners, &scale);
    lower[axis]    = static_cast<int>(Nearest ? nearbyintf(coord) : floorf(coord));
    fraction[axis] = coord - static_cast<float>(lower[axis]);
  }

  int64_t corner_offset[kCorners];
  float   corner_weight[kCorners];
  bool    all_valid = true;
#pragma unroll
  for (int corner = 0; corner < kActive; ++corner) {
    int   index[3] = {lower[0], lower[1], lower[2]};
    bool  valid    = true;
    float weight   = 1.0F;
#pragma unroll
    for (int axis = 0; axis < Dim; ++axis) {
      const int upper = (corner >> axis) & 1;
      index[axis]     = lower[axis] + upper;
      valid &= index[axis] >= 0 && index[axis] < sizes[axis];
      weight *= Nearest ? 1.0F : (upper ? fraction[axis] : 1.0F - fraction[axis]);
    }
    corner_offset[corner] = valid ? (static_cast<int64_t>(index[2]) * height + index[1]) * width + index[0] : -1;
    corner_weight[corner] = weight;
    all_valid &= valid;
  }

  const int64_t input_spatial = static_cast<int64_t>(depth) * height * width;
  for (int64_t channel = blockIdx.y; channel < channels; channel += gridDim.y) {
    const int64_t base      = (batch * channels + channel) * input_spatial;
    const int64_t out_index = (batch * channels + channel) * output_spatial + spatial;
    float         result    = 0.0F;
    if (all_valid) {
#pragma unroll
      for (int corner = 0; corner < kActive; ++corner) {
        result += as_float(input[base + corner_offset[corner]]) * corner_weight[corner];
      }
    } else {
#pragma unroll
      for (int corner = 0; corner < kActive; ++corner) {
        if (corner_offset[corner] >= 0) {
          result += as_float(input[base + corner_offset[corner]]) * corner_weight[corner];
        }
      }
    }
    output[out_index] = from_float<T>(result);
  }
}

// Backward keeps one thread per location so grid gradients accumulate in channel
// order.
template <typename T, int Dim>
__global__ void sample_backward_kernel(const T* input,
                                       const T* grid,
                                       const T* grad_output,
                                       float*   grad_input,
                                       float*   grad_grid,
                                       int64_t  locations,
                                       int64_t  channels,
                                       int64_t  output_spatial,
                                       int      depth,
                                       int      height,
                                       int      width,
                                       int      mode,
                                       int      padding_mode,
                                       bool     align_corners) {
  const int64_t location = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (location >= locations) return;
  const int64_t batch       = location / output_spatial;
  const int64_t spatial     = location % output_spatial;
  const int     sizes[3]    = {width, height, depth};
  float         coord[3]    = {0.0F, 0.0F, 0.0F};
  float         scale[3]    = {0.0F, 0.0F, 0.0F};
  int           lower[3]    = {0, 0, 0};
  float         fraction[3] = {0.0F, 0.0F, 0.0F};
  for (int axis = 0; axis < Dim; ++axis) {
    coord[axis] = muspace::sampling::source_coordinate(
        as_float(grid[location * Dim + axis]), sizes[axis], padding_mode, align_corners, &scale[axis]);
    lower[axis]    = static_cast<int>(mode == 1 ? nearbyintf(coord[axis]) : floorf(coord[axis]));
    fraction[axis] = coord[axis] - lower[axis];
  }
  float         grid_grad[3]  = {0.0F, 0.0F, 0.0F};
  const int64_t input_spatial = static_cast<int64_t>(depth) * height * width;
  const int     neighbors     = mode == 1 ? 1 : (1 << Dim);
  int64_t       corner_offset[1 << Dim];
  float         corner_weight[1 << Dim];
  float         corner_derivative[1 << Dim][Dim];
  for (int corner = 0; corner < (1 << Dim); ++corner) {
    int   index[3]   = {0, 0, 0};
    float weights[3] = {1.0F, 1.0F, 1.0F};
    bool  valid      = corner < neighbors;
    float weight     = 1.0F;
    for (int axis = 0; axis < Dim; ++axis) {
      const int upper = (corner >> axis) & 1;
      index[axis]     = lower[axis] + upper;
      valid &= index[axis] >= 0 && index[axis] < sizes[axis];
      weights[axis] = mode == 1 ? 1.0F : (upper ? fraction[axis] : 1.0F - fraction[axis]);
      weight *= weights[axis];
    }
    corner_offset[corner] = valid ? (static_cast<int64_t>(index[2]) * height + index[1]) * width + index[0] : -1;
    corner_weight[corner] = weight;
    for (int axis = 0; axis < Dim; ++axis) {
      // Keep per-axis factors; the reference multiplies them in this order.
      corner_derivative[corner][axis] = weights[axis];
    }
  }
  for (int64_t channel = 0; channel < channels; ++channel) {
    const int64_t base      = (batch * channels + channel) * input_spatial;
    const int64_t out_index = (batch * channels + channel) * output_spatial + spatial;
    const float   go        = as_float(grad_output[out_index]);
#pragma unroll
    for (int corner = 0; corner < (1 << Dim); ++corner) {
      if (corner >= neighbors) continue;
      if (corner_offset[corner] < 0) continue;
      const int64_t offset = base + corner_offset[corner];
      atomicAdd(grad_input + offset, go * corner_weight[corner]);
      if (mode == 0) {
        const float value = as_float(input[offset]);
        for (int axis = 0; axis < Dim; ++axis) {
          // Keep the reference multiply order and fused accumulation.
          float contribution = value;
          for (int other = 0; other < Dim; ++other) {
            if (other != axis) contribution *= corner_derivative[corner][other];
          }
          if ((corner >> axis) & 1) {
            grid_grad[axis] = fmaf(contribution, go, grid_grad[axis]);
          } else {
            grid_grad[axis] = fmaf(-contribution, go, grid_grad[axis]);
          }
        }
      }
    }
  }
  for (int axis = 0; axis < Dim; ++axis) {
    grad_grid[location * Dim + axis] = grid_grad[axis] * scale[axis];
  }
}

void check_tensor(ffi::TensorView tensor, ffi::TensorView input, DLDataType dtype, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLExtDev) << name << " must be a MUSA tensor";
  TVM_FFI_ICHECK_EQ(tensor.device().device_id, input.device().device_id) << name << " device mismatch";
  TVM_FFI_ICHECK(tensor.numel() == 0 || tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK(same_dtype(tensor.dtype(), dtype)) << name << " dtype mismatch";
}

void check_inputs(ffi::TensorView input, ffi::TensorView grid, int64_t mode, int64_t padding_mode) {
  TVM_FFI_ICHECK(same_dtype(input.dtype(), kDLFloat32) || same_dtype(input.dtype(), kDLFloat16) ||
                 same_dtype(input.dtype(), kDLBFloat16))
      << "input must use float32, float16 or bfloat16";
  check_tensor(input, input, input.dtype(), "input");
  check_tensor(grid, input, input.dtype(), "grid");
  TVM_FFI_ICHECK(input.ndim() == 4 || input.ndim() == 5) << "input must be 4-D or 5-D";
  TVM_FFI_ICHECK_EQ(grid.ndim(), input.ndim()) << "grid rank must match input rank";
  TVM_FFI_ICHECK_EQ(grid.size(0), input.size(0)) << "grid batch dimension must match input";
  TVM_FFI_ICHECK_EQ(grid.size(grid.ndim() - 1), input.ndim() - 2) << "grid last dimension mismatch";
  for (int axis = 2; axis < input.ndim(); ++axis) {
    TVM_FFI_ICHECK_GT(input.size(axis), 0) << "input spatial dimensions must be positive";
    TVM_FFI_ICHECK_LT(input.size(axis), INT_MAX) << "input spatial dimension exceeds indexing range";
  }
  TVM_FFI_ICHECK_LE(grid.numel() / (input.ndim() - 2), static_cast<int64_t>(INT_MAX) * 256)
      << "grid exceeds launch range";
  TVM_FFI_ICHECK(mode == 0 || mode == 1) << "unknown interpolation mode";
  TVM_FFI_ICHECK(padding_mode >= 0 && padding_mode <= 2) << "unknown padding mode";
}

void check_shape(ffi::TensorView tensor, ffi::TensorView expected, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), expected.ndim()) << name << " rank mismatch";
  for (int i = 0; i < expected.ndim(); ++i) {
    TVM_FFI_ICHECK_EQ(tensor.size(i), expected.size(i)) << name << " shape mismatch";
  }
}

void check_output_shape(ffi::TensorView output, ffi::TensorView input, ffi::TensorView grid) {
  TVM_FFI_ICHECK_EQ(output.ndim(), input.ndim()) << "output rank mismatch";
  TVM_FFI_ICHECK_EQ(output.size(0), input.size(0)) << "output batch mismatch";
  TVM_FFI_ICHECK_EQ(output.size(1), input.size(1)) << "output channels mismatch";
  for (int i = 2; i < input.ndim(); ++i) {
    TVM_FFI_ICHECK_EQ(output.size(i), grid.size(i - 1)) << "output spatial shape mismatch";
  }
}

template <typename T, bool Backward>
void launch(ffi::TensorView input,
            ffi::TensorView grid,
            ffi::TensorView output,
            ffi::TensorView grad_input,
            ffi::TensorView grad_grid,
            int64_t         mode,
            int64_t         padding_mode,
            bool            align_corners) {
  int64_t output_spatial = 1;
  for (int i = 1; i < grid.ndim() - 1; ++i) output_spatial *= grid.size(i);
  const int64_t locations = grid.numel() / (input.ndim() - 2);
  if constexpr (Backward) {
    if (grad_input.numel()) {
      TVM_FFI_ICHECK_EQ(musaMemsetAsync(grad_input.data_ptr(), 0, grad_input.numel() * sizeof(float), stream(input)),
                        musaSuccess);
    }
  }
  if (!locations || (!Backward && !input.size(1))) return;
  const int     depth  = input.ndim() == 4 ? 1 : input.size(2);
  const int     height = input.size(input.ndim() - 2);
  const int     width  = input.size(input.ndim() - 1);
  constexpr int kBlock = 256;
  const int     blocks = static_cast<int>((locations + kBlock - 1) / kBlock);
  const dim3    launch_grid(blocks,
                         Backward ? 1u : static_cast<unsigned int>(input.size(1) < 65535 ? input.size(1) : 65535));
  T*            out = nullptr;
  const T*      go  = nullptr;
  float*        gi  = nullptr;
  float*        gg  = nullptr;
  if constexpr (Backward) {
    go = static_cast<const T*>(output.data_ptr());
    gi = static_cast<float*>(grad_input.data_ptr());
    gg = static_cast<float*>(grad_grid.data_ptr());
  } else {
    out = static_cast<T*>(output.data_ptr());
  }
  const T* in_ptr = static_cast<const T*>(input.data_ptr());
  const T* gr_ptr = static_cast<const T*>(grid.data_ptr());
  if constexpr (Backward) {
    if (input.ndim() == 4) {
      sample_backward_kernel<T, 2><<<launch_grid, kBlock, 0, stream(input)>>>(in_ptr,
                                                                              gr_ptr,
                                                                              go,
                                                                              gi,
                                                                              gg,
                                                                              locations,
                                                                              input.size(1),
                                                                              output_spatial,
                                                                              depth,
                                                                              height,
                                                                              width,
                                                                              mode,
                                                                              padding_mode,
                                                                              align_corners);
    } else {
      sample_backward_kernel<T, 3><<<launch_grid, kBlock, 0, stream(input)>>>(in_ptr,
                                                                              gr_ptr,
                                                                              go,
                                                                              gi,
                                                                              gg,
                                                                              locations,
                                                                              input.size(1),
                                                                              output_spatial,
                                                                              depth,
                                                                              height,
                                                                              width,
                                                                              mode,
                                                                              padding_mode,
                                                                              align_corners);
    }
  } else if (input.ndim() == 4) {
    if (mode == 1) {
      sample_forward_kernel<T, 2, true><<<launch_grid, kBlock, 0, stream(input)>>>(in_ptr,
                                                                                   gr_ptr,
                                                                                   out,
                                                                                   locations,
                                                                                   input.size(1),
                                                                                   output_spatial,
                                                                                   depth,
                                                                                   height,
                                                                                   width,
                                                                                   padding_mode,
                                                                                   align_corners);
    } else {
      sample_forward_kernel<T, 2, false><<<launch_grid, kBlock, 0, stream(input)>>>(in_ptr,
                                                                                    gr_ptr,
                                                                                    out,
                                                                                    locations,
                                                                                    input.size(1),
                                                                                    output_spatial,
                                                                                    depth,
                                                                                    height,
                                                                                    width,
                                                                                    padding_mode,
                                                                                    align_corners);
    }
  } else {
    if (mode == 1) {
      sample_forward_kernel<T, 3, true><<<launch_grid, kBlock, 0, stream(input)>>>(in_ptr,
                                                                                   gr_ptr,
                                                                                   out,
                                                                                   locations,
                                                                                   input.size(1),
                                                                                   output_spatial,
                                                                                   depth,
                                                                                   height,
                                                                                   width,
                                                                                   padding_mode,
                                                                                   align_corners);
    } else {
      sample_forward_kernel<T, 3, false><<<launch_grid, kBlock, 0, stream(input)>>>(in_ptr,
                                                                                    gr_ptr,
                                                                                    out,
                                                                                    locations,
                                                                                    input.size(1),
                                                                                    output_spatial,
                                                                                    depth,
                                                                                    height,
                                                                                    width,
                                                                                    padding_mode,
                                                                                    align_corners);
    }
  }
  TVM_FFI_ICHECK_EQ(musaGetLastError(), musaSuccess) << "grid_sample launch failed";
}

template <bool Backward>
void dispatch(ffi::TensorView input,
              ffi::TensorView grid,
              ffi::TensorView output,
              ffi::TensorView grad_input,
              ffi::TensorView grad_grid,
              int64_t         mode,
              int64_t         padding_mode,
              bool            align_corners) {
  if (same_dtype(input.dtype(), kDLFloat32)) {
    launch<float, Backward>(input, grid, output, grad_input, grad_grid, mode, padding_mode, align_corners);
  } else if (same_dtype(input.dtype(), kDLFloat16)) {
    launch<__half, Backward>(input, grid, output, grad_input, grad_grid, mode, padding_mode, align_corners);
  } else {
    launch<__mt_bfloat16, Backward>(input, grid, output, grad_input, grad_grid, mode, padding_mode, align_corners);
  }
}
}  // namespace

void grid_sample_forward(ffi::TensorView input,
                         ffi::TensorView grid,
                         ffi::TensorView output,
                         int64_t         mode,
                         int64_t         padding_mode,
                         bool            align_corners) {
  check_inputs(input, grid, mode, padding_mode);
  check_tensor(output, input, input.dtype(), "output");
  check_output_shape(output, input, grid);
  dispatch<false>(input, grid, output, input, grid, mode, padding_mode, align_corners);
}

void grid_sample_backward(ffi::TensorView input,
                          ffi::TensorView grid,
                          ffi::TensorView grad_output,
                          ffi::TensorView grad_input,
                          ffi::TensorView grad_grid,
                          int64_t         mode,
                          int64_t         padding_mode,
                          bool            align_corners) {
  check_inputs(input, grid, mode, padding_mode);
  check_tensor(grad_output, input, input.dtype(), "grad_output");
  check_tensor(grad_input, input, kDLFloat32, "grad_input");
  check_tensor(grad_grid, input, kDLFloat32, "grad_grid");
  check_output_shape(grad_output, input, grid);
  check_shape(grad_input, input, "grad_input");
  check_shape(grad_grid, grid, "grad_grid");
  dispatch<true>(input, grid, grad_output, grad_input, grad_grid, mode, padding_mode, align_corners);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(grid_sample_forward, grid_sample_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(grid_sample_backward, grid_sample_backward);
