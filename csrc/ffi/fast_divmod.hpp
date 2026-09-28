#pragma once

#include <cstdint>

// Adapted from torch_musa's MUSAMath FastDivmod implementation.  The
// arithmetic originates from CUTLASS fast division / Hacker's Delight and is
// intentionally kept free of ATen and torch_musa headers.
namespace muspace::ffi_utils {

__host__ __device__ __forceinline__ uint32_t count_leading_zeros(uint32_t value) {
  for (int bit = 31; bit >= 0; --bit) {
    if ((uint32_t{1} << bit) & value) {
      return static_cast<uint32_t>(31 - bit);
    }
  }
  return 32;
}

__host__ __device__ __forceinline__ uint32_t find_log2(uint32_t value) {
  uint32_t result = 31 - count_leading_zeros(value);
  result += (value & (value - 1)) != 0;
  return result;
}

__host__ __device__ __forceinline__ void find_divisor(uint32_t& multiplier, uint32_t& shift_right, uint32_t divisor) {
  if ((divisor & (divisor - 1)) == 0) {
    multiplier  = 1;
    shift_right = find_log2(divisor);
  } else {
    const uint32_t precision = 31 + find_log2(divisor);
    multiplier  = static_cast<uint32_t>(((uint64_t{1} << precision) + static_cast<uint64_t>(divisor) - 1) / divisor);
    shift_right = precision;
  }
}

struct FastDivmod {
  uint32_t divisor;
  uint32_t multiplier;
  uint32_t shift_right;

  __host__ __device__ __forceinline__ FastDivmod() : divisor(0), multiplier(0), shift_right(0) {
  }

  __host__ __device__ __forceinline__ explicit FastDivmod(uint32_t divisor_) : divisor(divisor_) {
    find_divisor(multiplier, shift_right, divisor);
  }

  __host__ __device__ __forceinline__ void operator()(uint32_t& quotient,
                                                      uint32_t& remainder,
                                                      uint32_t  dividend) const {
    quotient  = static_cast<uint32_t>((static_cast<uint64_t>(dividend) * multiplier) >> shift_right);
    remainder = dividend - quotient * divisor;
  }
};

}  // namespace muspace::ffi_utils
