// Torch-ABI-independent MUSA entrypoints.
//
// The public Python APIs allocate all outputs and pass them as DLPack tensor
// views.  This translation unit is intentionally independent of PyTorch C++
// headers and ABI.

#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "dtype_utils.hpp"

namespace ffi = tvm::ffi;

namespace {

void check_musa_f32_contiguous(ffi::TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLExtDev) << name << " must be a MUSA tensor";
  TVM_FFI_ICHECK(same_dtype(tensor.dtype(), kDLFloat32)) << name << " must use float32";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

musaStream_t current_stream(ffi::TensorView tensor) {
  return static_cast<musaStream_t>(TVMFFIEnvGetStream(tensor.device().device_type, tensor.device().device_id));
}

__global__ void add_one_kernel(const float* input, float* output, int64_t size) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) output[index] = input[index] + 1.0F;
}

}  // namespace

// Smoke entrypoint and reference implementation for the FFI tensor contract.
// Production operators follow the same input/output convention as they are
// migrated from the legacy PyTorch extension source files.
void add_one(ffi::TensorView input, ffi::TensorView output) {
  check_musa_f32_contiguous(input, "input");
  check_musa_f32_contiguous(output, "output");
  TVM_FFI_ICHECK_EQ(input.device().device_id, output.device().device_id)
      << "input and output must be on the same MUSA device";
  TVM_FFI_ICHECK_EQ(input.numel(), output.numel()) << "output must match input shape";
  const int64_t size = input.numel();
  if (size == 0) return;
  constexpr int kThreads = 256;
  const int     blocks   = static_cast<int>((size + kThreads - 1) / kThreads);
  add_one_kernel<<<blocks, kThreads, 0, current_stream(input)>>>(
      static_cast<const float*>(input.data_ptr()), static_cast<float*>(output.data_ptr()), size);
  TVM_FFI_ICHECK_EQ(musaGetLastError(), musaSuccess) << "add_one launch failed";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, add_one);
