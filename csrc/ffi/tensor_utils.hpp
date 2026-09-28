#pragma once

#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include "dtype_utils.hpp"

namespace muspace::ffi_utils {
namespace ffi = tvm::ffi;

inline void check_musa_contiguous(ffi::TensorView tensor, const char* name, DLDataType dtype) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLExtDev) << name << " must be a MUSA tensor";
  TVM_FFI_ICHECK(same_dtype(tensor.dtype(), dtype)) << name << " has unsupported dtype";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

inline void check_same_musa_device(
    ffi::TensorView reference, ffi::TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, reference.device().device_type)
      << name << " must use the same device type as the reference tensor";
  TVM_FFI_ICHECK_EQ(tensor.device().device_id, reference.device().device_id)
      << name << " must be on the same MUSA device as the reference tensor";
}

inline musaStream_t stream(ffi::TensorView tensor) {
  return static_cast<musaStream_t>(TVMFFIEnvGetStream(tensor.device().device_type, tensor.device().device_id));
}

inline void check_last_error(const char* operation) {
  const musaError_t error = musaGetLastError();
  TVM_FFI_ICHECK_EQ(error, musaSuccess) << operation << " launch failed: " << musaGetErrorString(error);
}

inline int64_t dim(ffi::TensorView tensor, int32_t index) {
  return tensor.size(index);
}
}  // namespace muspace::ffi_utils
