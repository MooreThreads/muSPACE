// Deterministic radius query for the torch-cluster neighborhood contract.

#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

#include <cstdint>

#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::check_same_musa_device;
using muspace::ffi_utils::stream;

namespace {

constexpr int          kThreads       = 256;
constexpr int          kWarpSize      = 32;
constexpr int          kWarpsPerBlock = kThreads / kWarpSize;
constexpr unsigned int kFullWarpMask  = 0xffffffffu;
constexpr int64_t      kInt32Max      = 2147483647;
constexpr int64_t      kInt64Max      = 9223372036854775807LL;

struct PtrResolver {
  const int32_t* ptr_x;
  const int32_t* ptr_y;
  int            batch_size;

  __device__ __forceinline__ void resolve(int y_index, int& x_left, int& x_right) const {
    int left  = 0;
    int right = batch_size;
    for (int iteration = 0; iteration < 32 && left < right; ++iteration) {
      const int middle = (left + right) / 2;
      if (ptr_y[middle + 1] <= y_index)
        left = middle + 1;
      else
        right = middle;
    }
    x_left  = ptr_x[left];
    x_right = ptr_x[left + 1];
  }
};

struct BatchResolver {
  const int64_t* batch_x;
  const int64_t* batch_y;
  int            num_x;

  __device__ __forceinline__ int lower_bound(int64_t value) const {
    int left  = 0;
    int right = num_x;
    for (int iteration = 0; iteration < 32 && left < right; ++iteration) {
      const int middle = (left + right) / 2;
      if (batch_x[middle] < value)
        left = middle + 1;
      else
        right = middle;
    }
    return left;
  }

  __device__ __forceinline__ void resolve(int y_index, int& x_left, int& x_right) const {
    const int64_t batch = batch_y[y_index];
    x_left              = lower_bound(batch);
    x_right             = batch == kInt64Max ? num_x : lower_bound(batch + 1);
  }
};

template <int kCoordinateDim, typename Resolver>
__global__ void radius_query_kernel(const float* x,
                                    const float* y,
                                    Resolver     resolver,
                                    int64_t*     out_x,
                                    int32_t*     counts,
                                    int          coordinate_dim,
                                    int          max_num_neighbors,
                                    float        radius_squared,
                                    int          num_y) {
  const int  lane    = (int)(threadIdx.x) & (kWarpSize - 1);
  const int  warp    = (int)(threadIdx.x) / kWarpSize;
  const int  y_index = (int)(blockIdx.x) * kWarpsPerBlock + warp;
  const bool active  = y_index < num_y;

  int x_left  = 0;
  int x_right = 0;
  if (active && lane == 0) resolver.resolve(y_index, x_left, x_right);
  x_left  = __shfl_sync(kFullWarpMask, x_left, 0);
  x_right = __shfl_sync(kFullWarpMask, x_right, 0);

  float y_coordinates[8];
  if constexpr (kCoordinateDim == 2) {
    float2 y_point = make_float2(0.0f, 0.0f);
    if (active && lane == 0) y_point = ((const float2*)(y))[y_index];
    y_coordinates[0] = __shfl_sync(kFullWarpMask, y_point.x, 0);
    y_coordinates[1] = __shfl_sync(kFullWarpMask, y_point.y, 0);
  } else {
    for (int dim = 0; dim < coordinate_dim; ++dim) {
      float coordinate   = active && lane == 0 ? y[y_index * coordinate_dim + dim] : 0.0f;
      y_coordinates[dim] = __shfl_sync(kFullWarpMask, coordinate, 0);
    }
  }

  int                count           = 0;
  const unsigned int lower_lane_mask = lane == 0 ? 0u : (1u << lane) - 1u;
  for (int x_base = x_left; x_base < x_right && count < max_num_neighbors; x_base += kWarpSize) {
    const int x_index  = x_base + lane;
    float     distance = 0.0f;
    if (active && x_index < x_right) {
      if constexpr (kCoordinateDim == 2) {
        const float2 x_point = ((const float2*)(x))[x_index];
        const float  delta_x = x_point.x - y_coordinates[0];
        const float  delta_y = x_point.y - y_coordinates[1];
        distance             = delta_x * delta_x + delta_y * delta_y;
      } else {
        for (int dim = 0; dim < coordinate_dim; ++dim) {
          const float delta = x[x_index * coordinate_dim + dim] - y_coordinates[dim];
          distance += delta * delta;
        }
      }
    }
    const bool         matched     = active && x_index < x_right && distance < radius_squared;
    const unsigned int matches     = __ballot_sync(kFullWarpMask, matched);
    const int          rank        = __popc(matches & lower_lane_mask);
    const int          match_count = __popc(matches);
    const int          remaining   = max_num_neighbors - count;
    if (matched && rank < remaining) {
      const int64_t slot = (int64_t)(y_index)*max_num_neighbors + count + rank;
      out_x[slot]        = (int64_t)(x_index);
    }
    count += match_count < remaining ? match_count : remaining;
  }
  if (active && lane == 0) counts[y_index] = count;
}

template <typename Resolver>
void launch_query(const float* x,
                  const float* y,
                  Resolver     resolver,
                  int64_t*     out_x,
                  int32_t*     counts,
                  int          coordinate_dim,
                  int          max_num_neighbors,
                  float        radius_squared,
                  int          num_y,
                  musaStream_t op_stream) {
  const unsigned int blocks = (unsigned int)((num_y + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (coordinate_dim == 2) {
    radius_query_kernel<2><<<blocks, kThreads, 0, op_stream>>>(
        x, y, resolver, out_x, counts, coordinate_dim, max_num_neighbors, radius_squared, num_y);
  } else {
    radius_query_kernel<0><<<blocks, kThreads, 0, op_stream>>>(
        x, y, resolver, out_x, counts, coordinate_dim, max_num_neighbors, radius_squared, num_y);
  }
}

__global__ void radius_compact_kernel(const int64_t* fixed_x,
                                      const int32_t* counts,
                                      const int64_t* inclusive_counts,
                                      int64_t*       output,
                                      int            max_num_neighbors,
                                      int            num_y,
                                      int64_t        edge_count) {
  const int lane    = (int)(threadIdx.x) & (kWarpSize - 1);
  const int warp    = (int)(threadIdx.x) / kWarpSize;
  const int y_index = (int)(blockIdx.x) * kWarpsPerBlock + warp;
  if (y_index >= num_y) return;
  const int     count         = counts[y_index];
  const int64_t output_offset = y_index == 0 ? 0 : inclusive_counts[y_index - 1];
  const int64_t input_offset  = (int64_t)(y_index)*max_num_neighbors;
  for (int neighbor = lane; neighbor < count; neighbor += kWarpSize) {
    const int64_t output_index        = output_offset + neighbor;
    output[output_index]              = (int64_t)(y_index);
    output[edge_count + output_index] = fixed_x[input_offset + neighbor];
  }
}

void validate_common(ffi::TensorView x,
                     ffi::TensorView y,
                     ffi::TensorView out_x,
                     ffi::TensorView counts,
                     int64_t         max_num_neighbors,
                     float           radius) {
  check_musa_contiguous(x, "x", kDLFloat32);
  check_musa_contiguous(y, "y", kDLFloat32);
  check_musa_contiguous(out_x, "out_x", kDLInt64);
  check_musa_contiguous(counts, "counts", kDLInt32);
  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must have shape [N, D]";
  TVM_FFI_ICHECK_EQ(y.ndim(), 2) << "y must have shape [M, D]";
  TVM_FFI_ICHECK_EQ(x.size(1), y.size(1)) << "x and y must have the same coordinate dimension";
  TVM_FFI_ICHECK_LT(x.size(1), 9) << "radius supports fewer than 9 coordinate dimensions";
  TVM_FFI_ICHECK_GT(x.size(1), 0) << "radius requires at least one coordinate dimension";
  TVM_FFI_ICHECK_EQ(out_x.numel(), y.size(0) * max_num_neighbors);
  TVM_FFI_ICHECK_EQ(counts.numel(), y.size(0));
  TVM_FFI_ICHECK_GT(max_num_neighbors, 0);
  TVM_FFI_ICHECK_LE(max_num_neighbors, kInt32Max);
  TVM_FFI_ICHECK_GE(radius, 0.0f);
  TVM_FFI_ICHECK_LE(x.size(0), kInt32Max);
  TVM_FFI_ICHECK_LE(y.size(0), kInt32Max);
  check_same_musa_device(x, y, "y");
  check_same_musa_device(x, out_x, "out_x");
  check_same_musa_device(x, counts, "counts");
}

void radius_forward(ffi::TensorView x,
                    ffi::TensorView y,
                    ffi::TensorView ptr_x,
                    ffi::TensorView ptr_y,
                    ffi::TensorView out_x,
                    ffi::TensorView counts,
                    float           radius,
                    int64_t         max_num_neighbors) {
  validate_common(x, y, out_x, counts, max_num_neighbors, radius);
  check_musa_contiguous(ptr_x, "ptr_x", kDLInt32);
  check_musa_contiguous(ptr_y, "ptr_y", kDLInt32);
  TVM_FFI_ICHECK_EQ(ptr_x.ndim(), 1);
  TVM_FFI_ICHECK_EQ(ptr_y.ndim(), 1);
  TVM_FFI_ICHECK_EQ(ptr_x.size(0), ptr_y.size(0));
  TVM_FFI_ICHECK_GE(ptr_x.size(0), 2);
  TVM_FFI_ICHECK_LE(ptr_x.size(0), kInt32Max);
  check_same_musa_device(x, ptr_x, "ptr_x");
  check_same_musa_device(x, ptr_y, "ptr_y");
  if (y.size(0) == 0) return;
  PtrResolver resolver{
      (const int32_t*)(ptr_x.data_ptr()), (const int32_t*)(ptr_y.data_ptr()), (int)(ptr_x.size(0) - 1)};
  launch_query((const float*)(x.data_ptr()),
               (const float*)(y.data_ptr()),
               resolver,
               (int64_t*)(out_x.data_ptr()),
               (int32_t*)(counts.data_ptr()),
               (int)(x.size(1)),
               (int)(max_num_neighbors),
               radius * radius,
               (int)(y.size(0)),
               stream(x));
  check_last_error("radius_forward");
}

void radius_batch_x_forward(ffi::TensorView x,
                            ffi::TensorView y,
                            ffi::TensorView batch_x,
                            ffi::TensorView batch_y,
                            ffi::TensorView out_x,
                            ffi::TensorView counts,
                            float           radius,
                            int64_t         max_num_neighbors) {
  validate_common(x, y, out_x, counts, max_num_neighbors, radius);
  check_musa_contiguous(batch_x, "batch_x", kDLInt64);
  check_musa_contiguous(batch_y, "batch_y", kDLInt64);
  TVM_FFI_ICHECK_EQ(batch_x.ndim(), 1);
  TVM_FFI_ICHECK_EQ(batch_y.ndim(), 1);
  TVM_FFI_ICHECK_EQ(batch_x.size(0), x.size(0));
  TVM_FFI_ICHECK_EQ(batch_y.size(0), y.size(0));
  check_same_musa_device(x, batch_x, "batch_x");
  check_same_musa_device(x, batch_y, "batch_y");
  if (y.size(0) == 0) return;
  BatchResolver resolver{(const int64_t*)(batch_x.data_ptr()), (const int64_t*)(batch_y.data_ptr()), (int)(x.size(0))};
  launch_query((const float*)(x.data_ptr()),
               (const float*)(y.data_ptr()),
               resolver,
               (int64_t*)(out_x.data_ptr()),
               (int32_t*)(counts.data_ptr()),
               (int)(x.size(1)),
               (int)(max_num_neighbors),
               radius * radius,
               (int)(y.size(0)),
               stream(x));
  check_last_error("radius_batch_x_forward");
}

void radius_compact_forward(ffi::TensorView fixed_x,
                            ffi::TensorView counts,
                            ffi::TensorView inclusive_counts,
                            ffi::TensorView output,
                            int64_t         max_num_neighbors) {
  check_musa_contiguous(fixed_x, "fixed_x", kDLInt64);
  check_musa_contiguous(counts, "counts", kDLInt32);
  check_musa_contiguous(inclusive_counts, "inclusive_counts", kDLInt64);
  check_musa_contiguous(output, "output", kDLInt64);
  TVM_FFI_ICHECK_EQ(fixed_x.ndim(), 1);
  TVM_FFI_ICHECK_EQ(counts.ndim(), 1);
  TVM_FFI_ICHECK_EQ(inclusive_counts.ndim(), 1);
  TVM_FFI_ICHECK_EQ(inclusive_counts.size(0), counts.size(0));
  TVM_FFI_ICHECK_EQ(fixed_x.numel(), counts.size(0) * max_num_neighbors);
  TVM_FFI_ICHECK_EQ(output.ndim(), 2);
  TVM_FFI_ICHECK_EQ(output.size(0), 2);
  TVM_FFI_ICHECK_GT(max_num_neighbors, 0);
  check_same_musa_device(fixed_x, counts, "counts");
  check_same_musa_device(fixed_x, inclusive_counts, "inclusive_counts");
  check_same_musa_device(fixed_x, output, "output");
  const int     num_y      = (int)(counts.size(0));
  const int64_t edge_count = output.size(1);
  if (num_y == 0 || edge_count == 0) return;
  const unsigned int blocks = (unsigned int)((num_y + kWarpsPerBlock - 1) / kWarpsPerBlock);
  radius_compact_kernel<<<blocks, kThreads, 0, stream(fixed_x)>>>((const int64_t*)(fixed_x.data_ptr()),
                                                                  (const int32_t*)(counts.data_ptr()),
                                                                  (const int64_t*)(inclusive_counts.data_ptr()),
                                                                  (int64_t*)(output.data_ptr()),
                                                                  (int)(max_num_neighbors),
                                                                  num_y,
                                                                  edge_count);
  check_last_error("radius_compact_forward");
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(radius_forward, radius_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radius_batch_x_forward, radius_batch_x_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(radius_compact_forward, radius_compact_forward);
