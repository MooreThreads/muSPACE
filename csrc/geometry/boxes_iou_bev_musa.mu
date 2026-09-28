// BEVFusion mmdet3d/ops/iou3d migration for MUSA.
//
// Upstream contract (BEVFusion 326653d): boxes are
// [x1, y1, x2, y2, rotation] and the extension exports BEV IoU plus rotated
// and axis-aligned NMS.  This file intentionally does not implement volume IoU.

#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::stream;

namespace {

constexpr int kIoUBlock = 16;
constexpr int kNmsBlock = 64;
constexpr float kEps = 1e-8f;

class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    TVM_FFI_ICHECK_EQ(musaGetDevice(&previous_), musaSuccess) << "musaGetDevice failed";
    changed_ = previous_ != device;
    if (changed_) {
      TVM_FFI_ICHECK_EQ(musaSetDevice(device), musaSuccess) << "musaSetDevice failed";
    }
  }
  ~DeviceGuard() {
    if (changed_) musaSetDevice(previous_);
  }

 private:
  int previous_ = 0;
  bool changed_ = false;
};

struct Point {
  float x, y;

  __device__ Point() : x(0.0f), y(0.0f) {}
  __device__ Point(float x_, float y_) : x(x_), y(y_) {}
  __device__ Point operator+(const Point& other) const {
    return Point(x + other.x, y + other.y);
  }
  __device__ Point operator-(const Point& other) const {
    return Point(x - other.x, y - other.y);
  }
};

__device__ __forceinline__ float cross(const Point& a, const Point& b) {
  return a.x * b.y - a.y * b.x;
}

__device__ __forceinline__ float cross(const Point& p1, const Point& p2,
                                       const Point& p0) {
  return (p1.x - p0.x) * (p2.y - p0.y) -
         (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ __forceinline__ bool rect_cross(const Point& p1, const Point& p2,
                                           const Point& q1, const Point& q2) {
  return fminf(p1.x, p2.x) <= fmaxf(q1.x, q2.x) &&
         fminf(q1.x, q2.x) <= fmaxf(p1.x, p2.x) &&
         fminf(p1.y, p2.y) <= fmaxf(q1.y, q2.y) &&
         fminf(q1.y, q2.y) <= fmaxf(p1.y, p2.y);
}

__device__ __forceinline__ bool in_box(const float* box, float c, float s,
                                       const Point& point) {
  const float cx = (box[0] + box[2]) * 0.5f;
  const float cy = (box[1] + box[3]) * 0.5f;
  const float dx = point.x - cx;
  const float dy = point.y - cy;
  const float x = dx * c - dy * s + cx;
  const float y = dx * s + dy * c + cy;
  constexpr float margin = 1e-5f;
  return x > box[0] - margin && x < box[2] + margin &&
         y > box[1] - margin && y < box[3] + margin;
}

__device__ __forceinline__ bool intersection(const Point& p1, const Point& p0,
                                              const Point& q1, const Point& q0,
                                              Point& answer) {
  if (!rect_cross(p0, p1, q0, q1)) return false;
  const float s1 = cross(q0, p1, p0);
  const float s2 = cross(p1, q1, p0);
  const float s3 = cross(p0, q1, q0);
  const float s4 = cross(q1, p1, q0);
  if (!(s1 * s2 > 0.0f && s3 * s4 > 0.0f)) return false;
  const float s5 = cross(q1, p1, p0);
  const float denominator = s5 - s1;
  if (fabsf(denominator) <= kEps) return false;
  answer.x = (s5 * q0.x - s1 * q1.x) / denominator;
  answer.y = (s5 * q0.y - s1 * q1.y) / denominator;
  return true;
}

__device__ __forceinline__ void rotate(Point& point, const Point& center,
                                       float c, float s) {
  const float dx = point.x - center.x;
  const float dy = point.y - center.y;
  point.x = dx * c + dy * s + center.x;
  point.y = -dx * s + dy * c + center.y;
}

__device__ float overlap_area(const float* box_a, float ac, float as,
                              const float* box_b, float bc, float bs) {
  const Point center_a((box_a[0] + box_a[2]) * 0.5f,
                       (box_a[1] + box_a[3]) * 0.5f);
  const Point center_b((box_b[0] + box_b[2]) * 0.5f,
                       (box_b[1] + box_b[3]) * 0.5f);
  Point a[5] = {Point(box_a[0], box_a[1]), Point(box_a[2], box_a[1]),
                Point(box_a[2], box_a[3]), Point(box_a[0], box_a[3])};
  Point b[5] = {Point(box_b[0], box_b[1]), Point(box_b[2], box_b[1]),
                Point(box_b[2], box_b[3]), Point(box_b[0], box_b[3])};
  for (int i = 0; i < 4; ++i) {
    rotate(a[i], center_a, ac, as);
    rotate(b[i], center_b, bc, bs);
  }
  a[4] = a[0];
  b[4] = b[0];

  Point points[16];
  Point center(0.0f, 0.0f);
  int count = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      Point point;
      if (intersection(a[i + 1], a[i], b[j + 1], b[j], point)) {
        points[count++] = point;
        center = center + point;
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    if (in_box(box_a, ac, as, b[i])) {
      points[count++] = b[i];
      center = center + b[i];
    }
    if (in_box(box_b, bc, bs, a[i])) {
      points[count++] = a[i];
      center = center + a[i];
    }
  }
  if (count < 3) return 0.0f;
  center.x /= count;
  center.y /= count;
  float angles[16];
  for (int i = 0; i < count; ++i) {
    angles[i] = atan2f(points[i].y - center.y, points[i].x - center.x);
  }
  for (int end = count - 1; end > 0; --end) {
    for (int i = 0; i < end; ++i) {
      if (angles[i] > angles[i + 1]) {
        Point point = points[i];
        points[i] = points[i + 1];
        points[i + 1] = point;
        float angle = angles[i];
        angles[i] = angles[i + 1];
        angles[i + 1] = angle;
      }
    }
  }
  float area = 0.0f;
  for (int i = 0; i < count - 1; ++i) {
    area += cross(points[i] - points[0], points[i + 1] - points[0]);
  }
  return fabsf(area) * 0.5f;
}

__device__ __forceinline__ float iou_bev(const float* a, const float* b) {
  const float ac = cosf(a[4]), as = sinf(a[4]);
  const float bc = cosf(b[4]), bs = sinf(b[4]);
  const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
  const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
  const float overlap = overlap_area(a, ac, as, b, bc, bs);
  return overlap / fmaxf(area_a + area_b - overlap, kEps);
}

__device__ __forceinline__ float iou_bev_precomputed(const float* a, float ac,
                                                      float as, const float* b,
                                                      float bc, float bs) {
  const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
  const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
  const float overlap = overlap_area(a, ac, as, b, bc, bs);
  return overlap / fmaxf(area_a + area_b - overlap, kEps);
}

__device__ __forceinline__ float iou_normal(const float* a, const float* b) {
  const float overlap = fmaxf(fminf(a[2], b[2]) - fmaxf(a[0], b[0]), 0.0f) *
                        fmaxf(fminf(a[3], b[3]) - fmaxf(a[1], b[1]), 0.0f);
  const float area_a = fmaxf(a[2] - a[0], 0.0f) * fmaxf(a[3] - a[1], 0.0f);
  const float area_b = fmaxf(b[2] - b[0], 0.0f) * fmaxf(b[3] - b[1], 0.0f);
  return overlap / fmaxf(area_a + area_b - overlap, kEps);
}

__global__ void boxes_iou_bev_kernel(int num_a, const float* boxes_a, int num_b,
                                     const float* boxes_b, float* output) {
  const int a_idx = blockIdx.y * blockDim.y + threadIdx.y;
  const int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (a_idx >= num_a || b_idx >= num_b) return;
  output[a_idx * num_b + b_idx] = iou_bev(boxes_a + a_idx * 5, boxes_b + b_idx * 5);
}

template <bool Rotated>
__global__ void nms_kernel(int boxes_num, float threshold, const float* boxes,
                           unsigned long long* mask) {
  const int row_block = blockIdx.y;
  const int col_block = blockIdx.x;
  const int row_remaining = boxes_num - row_block * kNmsBlock;
  const int col_remaining = boxes_num - col_block * kNmsBlock;
  const int row_size = row_remaining < kNmsBlock ? row_remaining : kNmsBlock;
  const int col_size = col_remaining < kNmsBlock ? col_remaining : kNmsBlock;
  constexpr int kBoxStride = Rotated ? 7 : 5;
  __shared__ float block_boxes[kNmsBlock * kBoxStride];
  if (threadIdx.x < col_size) {
    for (int k = 0; k < 5; ++k) {
      block_boxes[threadIdx.x * kBoxStride + k] =
          boxes[(col_block * kNmsBlock + threadIdx.x) * 5 + k];
    }
    if (Rotated) {
      const float angle = block_boxes[threadIdx.x * kBoxStride + 4];
      block_boxes[threadIdx.x * kBoxStride + 5] = cosf(angle);
      block_boxes[threadIdx.x * kBoxStride + 6] = sinf(angle);
    }
  }
  __syncthreads();
  if (threadIdx.x >= row_size) return;
  const int current = row_block * kNmsBlock + threadIdx.x;
  const int col_blocks = (boxes_num + kNmsBlock - 1) / kNmsBlock;
  const int start = row_block == col_block ? threadIdx.x + 1 : 0;
  unsigned long long suppressed = 0;
  const float* current_box = boxes + current * 5;
  const float current_c = Rotated ? cosf(current_box[4]) : 0.0f;
  const float current_s = Rotated ? sinf(current_box[4]) : 0.0f;
  for (int i = start; i < col_size; ++i) {
    const float* candidate = block_boxes + i * kBoxStride;
    const float value = Rotated
                            ? iou_bev_precomputed(current_box, current_c, current_s, candidate,
                                                  candidate[5], candidate[6])
                            : iou_normal(current_box, candidate);
    if (value > threshold) suppressed |= 1ULL << i;
  }
  mask[current * col_blocks + col_block] = suppressed;
}

template <bool Rotated>
void launch_nms_mask(ffi::TensorView boxes, float threshold, ffi::TensorView mask) {
  const int boxes_num = static_cast<int>(boxes.size(0));
  const int col_blocks = (boxes_num + kNmsBlock - 1) / kNmsBlock;
  check_musa_contiguous(mask, "mask", kDLInt64);
  TVM_FFI_ICHECK_EQ(mask.ndim(), 2);
  TVM_FFI_ICHECK_EQ(mask.size(0), boxes_num);
  TVM_FFI_ICHECK_EQ(mask.size(1), col_blocks);
  TVM_FFI_ICHECK_EQ(mask.device().device_id, boxes.device().device_id);
  if (boxes_num == 0) return;
  const dim3 grid(col_blocks, col_blocks);
  nms_kernel<Rotated><<<grid, dim3(kNmsBlock), 0, stream(boxes)>>>(
      boxes_num, threshold, static_cast<const float*>(boxes.data_ptr()),
      static_cast<unsigned long long*>(mask.data_ptr()));
  check_last_error("nms kernel");
}

template <bool Rotated>
int nms_forward(ffi::TensorView boxes, float threshold, ffi::TensorView mask,
                ffi::TensorView keep) {
  const int boxes_num = static_cast<int>(boxes.size(0));
  if (boxes_num == 0) return 0;
  const int col_blocks = (boxes_num + kNmsBlock - 1) / kNmsBlock;
  const size_t bytes = static_cast<size_t>(boxes_num) * col_blocks * sizeof(unsigned long long);
  DeviceGuard device_guard(boxes.device().device_id);
  const musaStream_t op_stream = stream(boxes);
  launch_nms_mask<Rotated>(boxes, threshold, mask);
  TVM_FFI_ICHECK_EQ(musaStreamSynchronize(op_stream), musaSuccess)
      << "nms kernel stream synchronization failed";
  std::vector<unsigned long long> host_mask(boxes_num * col_blocks);
  TVM_FFI_ICHECK_EQ(musaMemcpy(host_mask.data(), mask.data_ptr(), bytes, musaMemcpyDeviceToHost), musaSuccess)
      << "nms mask copy failed";

  std::vector<unsigned long long> removed(col_blocks, 0);
  int count = 0;
  std::vector<int64_t> kept(boxes_num);
  for (int i = 0; i < boxes_num; ++i) {
    const int block = i / kNmsBlock;
    const int bit = i % kNmsBlock;
    if (removed[block] & (1ULL << bit)) continue;
    kept[count++] = i;
    const auto* row = host_mask.data() + i * col_blocks;
    for (int j = block; j < col_blocks; ++j) removed[j] |= row[j];
  }
  if (count > 0) {
    TVM_FFI_ICHECK_EQ(musaMemcpy(keep.data_ptr(), kept.data(),
                                 static_cast<size_t>(count) * sizeof(int64_t),
                                 musaMemcpyHostToDevice), musaSuccess)
        << "nms keep copy failed";
  }
  return count;
}

void boxes_iou_bev_forward(ffi::TensorView boxes_a, ffi::TensorView boxes_b,
                           ffi::TensorView output) {
  check_musa_contiguous(boxes_a, "boxes_a", kDLFloat32);
  check_musa_contiguous(boxes_b, "boxes_b", kDLFloat32);
  check_musa_contiguous(output, "output", kDLFloat32);
  TVM_FFI_ICHECK_EQ(boxes_a.ndim(), 2) << "boxes_a must have shape [N, 5]";
  TVM_FFI_ICHECK_EQ(boxes_b.ndim(), 2) << "boxes_b must have shape [M, 5]";
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must have shape [N, M]";
  TVM_FFI_ICHECK_EQ(boxes_a.size(1), 5);
  TVM_FFI_ICHECK_EQ(boxes_b.size(1), 5);
  TVM_FFI_ICHECK_EQ(output.size(0), boxes_a.size(0));
  TVM_FFI_ICHECK_EQ(output.size(1), boxes_b.size(0));
  TVM_FFI_ICHECK_EQ(boxes_a.device().device_id, boxes_b.device().device_id);
  TVM_FFI_ICHECK_EQ(boxes_a.device().device_id, output.device().device_id);
  if (boxes_a.size(0) == 0 || boxes_b.size(0) == 0) return;
  const dim3 threads(kIoUBlock, kIoUBlock);
  const dim3 grid((boxes_b.size(0) + kIoUBlock - 1) / kIoUBlock,
                  (boxes_a.size(0) + kIoUBlock - 1) / kIoUBlock);
  boxes_iou_bev_kernel<<<grid, threads, 0, stream(boxes_a)>>>(
      static_cast<int>(boxes_a.size(0)), static_cast<const float*>(boxes_a.data_ptr()),
      static_cast<int>(boxes_b.size(0)), static_cast<const float*>(boxes_b.data_ptr()),
      static_cast<float*>(output.data_ptr()));
  check_last_error("boxes_iou_bev_forward");
}

int nms_bev_forward(ffi::TensorView boxes, float threshold, ffi::TensorView mask,
                    ffi::TensorView keep) {
  check_musa_contiguous(boxes, "boxes", kDLFloat32);
  check_musa_contiguous(keep, "keep", kDLInt64);
  TVM_FFI_ICHECK_EQ(boxes.ndim(), 2) << "boxes must have shape [N, 5]";
  TVM_FFI_ICHECK_EQ(boxes.size(1), 5);
  TVM_FFI_ICHECK_EQ(keep.ndim(), 1);
  TVM_FFI_ICHECK_EQ(keep.size(0), boxes.size(0));
  TVM_FFI_ICHECK_EQ(boxes.device().device_id, keep.device().device_id);
  return nms_forward<true>(boxes, threshold, mask, keep);
}

int nms_bev_normal_forward(ffi::TensorView boxes, float threshold, ffi::TensorView mask,
                           ffi::TensorView keep) {
  check_musa_contiguous(boxes, "boxes", kDLFloat32);
  check_musa_contiguous(keep, "keep", kDLInt64);
  TVM_FFI_ICHECK_EQ(boxes.ndim(), 2) << "boxes must have shape [N, 5]";
  TVM_FFI_ICHECK_EQ(boxes.size(1), 5);
  TVM_FFI_ICHECK_EQ(keep.ndim(), 1);
  TVM_FFI_ICHECK_EQ(keep.size(0), boxes.size(0));
  TVM_FFI_ICHECK_EQ(boxes.device().device_id, keep.device().device_id);
  return nms_forward<false>(boxes, threshold, mask, keep);
}

void nms_bev_mask_forward(ffi::TensorView boxes, float threshold, ffi::TensorView mask) {
  check_musa_contiguous(boxes, "boxes", kDLFloat32);
  TVM_FFI_ICHECK_EQ(boxes.ndim(), 2) << "boxes must have shape [N, 5]";
  TVM_FFI_ICHECK_EQ(boxes.size(1), 5);
  DeviceGuard device_guard(boxes.device().device_id);
  launch_nms_mask<true>(boxes, threshold, mask);
}

void nms_bev_normal_mask_forward(ffi::TensorView boxes, float threshold, ffi::TensorView mask) {
  check_musa_contiguous(boxes, "boxes", kDLFloat32);
  TVM_FFI_ICHECK_EQ(boxes.ndim(), 2) << "boxes must have shape [N, 5]";
  TVM_FFI_ICHECK_EQ(boxes.size(1), 5);
  DeviceGuard device_guard(boxes.device().device_id);
  launch_nms_mask<false>(boxes, threshold, mask);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(boxes_iou_bev_forward, boxes_iou_bev_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nms_bev_forward, nms_bev_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nms_bev_normal_forward, nms_bev_normal_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nms_bev_mask_forward, nms_bev_mask_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nms_bev_normal_mask_forward, nms_bev_normal_mask_forward);
