// BEVFusion points_in_boxes_batch replacement.

#include <musa_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

#include <cstdint>

#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::stream;

namespace {

constexpr int kThreads        = 512;
constexpr int kWarpSize       = 32;
constexpr int kBoxesPerLane   = 4;
constexpr int kPointsPerBlock = kThreads / kWarpSize;

__global__ void points_in_boxes_all_kernel(
    const float* boxes, const float* points, int32_t* output, int64_t points_num, int64_t boxes_num) {
  // A warp owns one point. Its lanes handle interleaved boxes, so box loads
  // and the corresponding [M] output stores are issued coalesced by the warp.
  // Four boxes per lane cover the BEVFusion fixture's M=100 in one pass.
  const int64_t b     = blockIdx.y;
  const int     lane  = threadIdx.x % kWarpSize;
  const int64_t point = (int64_t)blockIdx.x * kPointsPerBlock + threadIdx.x / kWarpSize;
  if (point >= points_num) return;
  const float* p            = points + (b * points_num + point) * 3;
  const float* batch_boxes  = boxes + b * boxes_num * 7;
  int32_t*     point_output = output + (b * points_num + point) * boxes_num;
  for (int64_t box_start = 0; box_start < boxes_num; box_start += kWarpSize * kBoxesPerLane) {
#pragma unroll
    for (int box_offset = 0; box_offset < kBoxesPerLane; ++box_offset) {
      const int64_t box = box_start + box_offset * kWarpSize + lane;
      if (box < boxes_num) {
        const float* q  = batch_boxes + box * 7;
        const float  dx = p[0] - q[0];
        const float  dy = p[1] - q[1];
        const float  dz = p[2] - (q[2] + q[5] * 0.5f);
        float        s, c;
        __sincosf(q[6] + 1.57079632679489661923f, &s, &c);
        const float local_x = dx * c - dy * s;
        const float local_y = dx * s + dy * c;
        point_output[box]   = (fabsf(dz) <= q[5] * 0.5f && local_x > -q[4] * 0.5f && local_x < q[4] * 0.5f &&
                             local_y > -q[3] * 0.5f && local_y < q[3] * 0.5f)
                                  ? 1
                                  : 0;
      }
    }
  }
}

}  // namespace

void points_in_boxes_all_forward(ffi::TensorView boxes, ffi::TensorView points, ffi::TensorView output) {
  check_musa_contiguous(boxes, "boxes", kDLFloat32);
  check_musa_contiguous(points, "points", kDLFloat32);
  check_musa_contiguous(output, "output", kDLInt32);
  TVM_FFI_ICHECK_EQ(boxes.ndim(), 3) << "boxes must have shape [B, M, 7]";
  TVM_FFI_ICHECK_EQ(points.ndim(), 3) << "points must have shape [B, N, 3]";
  TVM_FFI_ICHECK_EQ(output.ndim(), 3) << "output must have shape [B, N, M]";
  TVM_FFI_ICHECK_EQ(boxes.size(2), 7);
  TVM_FFI_ICHECK_EQ(points.size(2), 3);
  TVM_FFI_ICHECK_EQ(boxes.size(0), points.size(0));
  TVM_FFI_ICHECK_EQ(output.size(0), boxes.size(0));
  TVM_FFI_ICHECK_EQ(output.size(1), points.size(1));
  TVM_FFI_ICHECK_EQ(output.size(2), boxes.size(1));
  muspace::ffi_utils::check_same_musa_device(boxes, points, "points");
  muspace::ffi_utils::check_same_musa_device(boxes, output, "output");
  const int64_t total = boxes.size(0) * points.size(1);
  if (total == 0) return;
  const dim3 blocks((unsigned int)((points.size(1) + kPointsPerBlock - 1) / kPointsPerBlock),
                    (unsigned int)boxes.size(0));
  points_in_boxes_all_kernel<<<blocks, kThreads, 0, stream(boxes)>>>((const float*)boxes.data_ptr(),
                                                                     (const float*)points.data_ptr(),
                                                                     (int32_t*)output.data_ptr(),
                                                                     points.size(1),
                                                                     boxes.size(1));
  check_last_error("points_in_boxes_all");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(points_in_boxes_all_forward, points_in_boxes_all_forward);
