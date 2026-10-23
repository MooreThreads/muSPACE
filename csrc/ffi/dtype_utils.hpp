#pragma once

#include <dlpack/dlpack.h>

inline constexpr DLDataType kDLFloat32  = DLDataType{kDLFloat, 32, 1};
inline constexpr DLDataType kDLBFloat16 = DLDataType{kDLBfloat, 16, 1};
inline constexpr DLDataType kDLInt32    = DLDataType{kDLInt, 32, 1};
inline constexpr DLDataType kDLInt64    = DLDataType{kDLInt, 64, 1};

inline constexpr bool same_dtype(DLDataType lhs, DLDataType rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}
