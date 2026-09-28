#include <cfloat>
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <tvm/ffi/function.h>

#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::stream;

namespace {
constexpr DLDataType kDLFloat16 = DLDataType{kDLFloat, 16, 1};

template <bool Backward>
__device__ float focal_element(float input, float target, float gamma, float alpha) {
  const float at = alpha >= 0.0F ? alpha * target + (1.0F - alpha) * (1.0F - target) : 1.0F;
  // The binary focal-loss API uses targets 0/1.  Handle infinite logits before
  // forming BCE algebra so expressions such as ``input * target`` never create
  // an indeterminate 0 * inf.  These are the analytical forward/backward
  // limits of BCE-with-logits multiplied by the focal factor.
  if (isinf(input) && (target == 0.0F || target == 1.0F)) {
    if constexpr (Backward) {
      if (input > 0.0F) return target == 0.0F ? at : 0.0F;
      return target == 1.0F ? -at : 0.0F;
    }
    if (input > 0.0F) return target == 0.0F ? HUGE_VALF : 0.0F;
    return target == 1.0F ? HUGE_VALF : 0.0F;
  }
  const float p = 1.0F / (1.0F + expf(-input));
  const float pt = target * p + (1.0F - target) * (1.0F - p);
  const float one_minus_pt = 1.0F - pt;
  const float ce = fmaxf(input, 0.0F) - input * target + log1pf(expf(-fabsf(input)));
  const float mod = powf(one_minus_pt, gamma);
  if constexpr (Backward) {
    const float dpt = (2.0F * target - 1.0F) * p * (1.0F - p);
    const float focal_derivative = one_minus_pt > 0.0F ? powf(one_minus_pt, gamma - 1.0F) : 0.0F;
    return at * (mod * (p - target) - gamma * focal_derivative * ce * dpt);
  }
  return at * ce * mod;
}

template <typename T>
__global__ void sigmoid_focal_loss_forward_kernel(const T* input, const T* target, T* output,
                                                  int64_t n, float gamma, float alpha) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  output[i] = (T)focal_element<false>((float)input[i], (float)target[i], gamma, alpha);
}

template <typename T>
__global__ void sigmoid_focal_loss_backward_kernel(const T* input, const T* target, T* output,
                                                   int64_t n, float gamma, float alpha) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  output[i] = (T)focal_element<true>((float)input[i], (float)target[i], gamma, alpha);
}

void check(ffi::TensorView input, ffi::TensorView target, ffi::TensorView output,
           const char* output_name) {
  TVM_FFI_ICHECK_EQ(input.device().device_type, kDLExtDev) << "input must be a MUSA tensor";
  TVM_FFI_ICHECK_EQ(target.device().device_type, kDLExtDev) << "target must be a MUSA tensor";
  TVM_FFI_ICHECK_EQ(output.device().device_type, kDLExtDev) << output_name << " must be a MUSA tensor";
  TVM_FFI_ICHECK_EQ(target.device().device_id, input.device().device_id)
      << "target must be on the same MUSA device as input";
  TVM_FFI_ICHECK_EQ(output.device().device_id, input.device().device_id)
      << output_name << " must be on the same MUSA device as input";
  TVM_FFI_ICHECK(same_dtype(input.dtype(), kDLFloat32) || same_dtype(input.dtype(), kDLFloat16))
      << "input must use float32 or float16";
  TVM_FFI_ICHECK(same_dtype(target.dtype(), input.dtype())) << "target dtype must match input";
  TVM_FFI_ICHECK(same_dtype(output.dtype(), input.dtype())) << output_name << " dtype mismatch";
  TVM_FFI_ICHECK(input.IsContiguous() && target.IsContiguous() && output.IsContiguous())
      << "input, target and output must be contiguous";
  TVM_FFI_ICHECK_EQ(input.ndim(), target.ndim()) << "target must match input shape";
  TVM_FFI_ICHECK_EQ(input.ndim(), output.ndim()) << output_name << " must match input shape";
  for (int32_t i = 0; i < input.ndim(); ++i) {
    TVM_FFI_ICHECK_EQ(input.size(i), target.size(i)) << "target must match input shape";
    TVM_FFI_ICHECK_EQ(input.size(i), output.size(i)) << output_name << " must match input shape";
  }
}

template <typename T, bool Backward>
void launch(ffi::TensorView input, ffi::TensorView target, ffi::TensorView output,
            double gamma, double alpha) {
  constexpr int threads = 256;
  const int64_t n = input.numel();
  if (n == 0) return;
  const int blocks = (int)((n + threads - 1) / threads);
  if constexpr (Backward) {
    sigmoid_focal_loss_backward_kernel<T><<<blocks, threads, 0, stream(input)>>>(
        (const T*)input.data_ptr(), (const T*)target.data_ptr(),
        (T*)output.data_ptr(), n, (float)gamma, (float)alpha);
  } else {
    sigmoid_focal_loss_forward_kernel<T><<<blocks, threads, 0, stream(input)>>>(
        (const T*)input.data_ptr(), (const T*)target.data_ptr(),
        (T*)output.data_ptr(), n, (float)gamma, (float)alpha);
  }
  TVM_FFI_ICHECK_EQ(musaGetLastError(), musaSuccess) << "sigmoid focal loss launch failed";
}

void dispatch(ffi::TensorView input, ffi::TensorView target, ffi::TensorView output,
              double gamma, double alpha, bool backward) {
  if (same_dtype(input.dtype(), kDLFloat32)) {
    if (backward) launch<float, true>(input, target, output, gamma, alpha);
    else launch<float, false>(input, target, output, gamma, alpha);
  } else {
    if (backward) launch<half, true>(input, target, output, gamma, alpha);
    else launch<half, false>(input, target, output, gamma, alpha);
  }
}
}  // namespace

void sigmoid_focal_loss_forward(ffi::TensorView input, ffi::TensorView target,
                                ffi::TensorView output, double gamma, double alpha) {
  check(input, target, output, "output");
  dispatch(input, target, output, gamma, alpha, false);
}

void sigmoid_focal_loss_backward(ffi::TensorView input, ffi::TensorView target,
                                 ffi::TensorView grad_input, double gamma, double alpha) {
  check(input, target, grad_input, "grad_input");
  dispatch(input, target, grad_input, gamma, alpha, true);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sigmoid_focal_loss_forward, sigmoid_focal_loss_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sigmoid_focal_loss_backward, sigmoid_focal_loss_backward);
