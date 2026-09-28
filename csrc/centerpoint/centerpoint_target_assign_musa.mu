#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::check_same_musa_device;
using muspace::ffi_utils::stream;

namespace {

constexpr DLDataType kDLInt64     = DLDataType{kDLInt, 64, 1};
constexpr DLDataType kDLUInt8     = DLDataType{kDLUInt, 8, 1};
constexpr int        kThreads     = 128;
constexpr int        kTile        = 16;
constexpr int        kTileThreads = kTile * kTile;

__device__ int gaussian_radius(float height, float width, float overlap) {
  const float b1 = height + width;
  const float c1 = width * height * (1.0f - overlap) / (1.0f + overlap);
  const float r1 = (b1 + sqrtf(b1 * b1 - 4.0f * c1)) * 0.5f;
  const float b2 = 2.0f * (height + width);
  const float c2 = (1.0f - overlap) * width * height;
  const float r2 = (b2 + sqrtf(b2 * b2 - 16.0f * c2)) * 0.5f;
  const float a3 = 4.0f * overlap;
  const float b3 = -2.0f * overlap * (height + width);
  const float c3 = (overlap - 1.0f) * width * height;
  const float r3 = (b3 + sqrtf(b3 * b3 - 4.0f * a3 * c3)) * 0.5f;
  return static_cast<int>(fminf(r1, fminf(r2, r3)));
}

__device__ void atomic_max_nonnegative(float* address, float value) {
  int* address_as_int = reinterpret_cast<int*>(address);
  int  old            = *address_as_int;
  while (__int_as_float(old) < value) {
    const int assumed = old;
    old               = atomicCAS(address_as_int, assumed, __float_as_int(value));
    if (old == assumed) break;
  }
}

__global__ void prepare_targets_kernel(const float* __restrict__ boxes,
                                       float* __restrict__ anno_box,
                                       int64_t* __restrict__ inds,
                                       uint8_t* __restrict__ mask,
                                       int   n,
                                       int   out_size_factor,
                                       float voxel_x,
                                       float voxel_y,
                                       float range_x,
                                       float range_y,
                                       int   fmap_w,
                                       int   fmap_h) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  const float* box      = boxes + index * 9;
  const float  scaled_w = box[3] / voxel_x / static_cast<float>(out_size_factor);
  const float  scaled_l = box[4] / voxel_y / static_cast<float>(out_size_factor);
  if (!(scaled_w > 0.0f && scaled_l > 0.0f)) return;
  const float x = (box[0] - range_x) / voxel_x / static_cast<float>(out_size_factor);
  const float y = (box[1] - range_y) / voxel_y / static_cast<float>(out_size_factor);
  // This is intentionally C/C++ truncation toward zero, matching the upstream
  // ``np.ndarray.astype(np.int32)`` conversion rather than floor().
  const int ix = static_cast<int>(x);
  const int iy = static_cast<int>(y);
  if (ix < 0 || ix >= fmap_w || iy < 0 || iy >= fmap_h) return;
  inds[index]   = static_cast<int64_t>(iy) * fmap_w + ix;
  mask[index]   = 1;
  float* target = anno_box + index * 10;
  target[0]     = x - static_cast<float>(ix);
  target[1]     = y - static_cast<float>(iy);
  target[2]     = box[2];
  target[3]     = logf(box[3]);
  target[4]     = logf(box[4]);
  target[5]     = logf(box[5]);
  target[6]     = sinf(box[8]);
  target[7]     = cosf(box[8]);
  target[8]     = box[6];
  target[9]     = box[7];
}

__global__ void draw_gaussians_kernel(const int64_t* __restrict__ class_ids,
                                      const float* __restrict__ boxes,
                                      const uint8_t* __restrict__ mask,
                                      float* __restrict__ heatmap,
                                      int   n,
                                      int   out_size_factor,
                                      float overlap,
                                      int   min_radius,
                                      float voxel_x,
                                      float voxel_y,
                                      float range_x,
                                      float range_y,
                                      int   fmap_w,
                                      int   fmap_h) {
  const int object = blockIdx.x;
  if (object >= n || !mask[object]) return;
  const float* box         = boxes + object * 9;
  const int    radius      = max(min_radius,
                         gaussian_radius(box[4] / voxel_y / static_cast<float>(out_size_factor),
                                         box[3] / voxel_x / static_cast<float>(out_size_factor),
                                         overlap));
  const int    x           = static_cast<int>((box[0] - range_x) / voxel_x / static_cast<float>(out_size_factor));
  const int    y           = static_cast<int>((box[1] - range_y) / voxel_y / static_cast<float>(out_size_factor));
  const int    left        = min(x, radius);
  const int    right       = min(fmap_w - x, radius + 1);
  const int    top         = min(y, radius);
  const int    bottom      = min(fmap_h - y, radius + 1);
  const int    patch_w     = left + right;
  const int    patch_h     = top + bottom;
  const float  diameter    = static_cast<float>(radius * 2 + 1);
  const float  denominator = 2.0f * (diameter / 6.0f) * (diameter / 6.0f);
  float*       channel     = heatmap + (class_ids[object] - 1) * fmap_h * fmap_w;
  for (int offset = threadIdx.x; offset < patch_w * patch_h; offset += blockDim.x) {
    const int   local_y = offset / patch_w;
    const int   local_x = offset % patch_w;
    const int   dx      = local_x - left;
    const int   dy      = local_y - top;
    const float value   = expf(-static_cast<float>(dx * dx + dy * dy) / denominator);
    atomic_max_nonnegative(channel + (y + dy) * fmap_w + x + dx, value);
  }
}

// One block owns one (class, 16x16) heatmap tile.  Objects are partitioned
// between threads and their Gaussian patches are reduced in shared memory;
// only the completed tile is written to global memory.  This preserves the
// upstream maximum semantics while removing inter-object global CAS traffic.
__global__ void draw_gaussians_tiled_kernel(const int64_t* __restrict__ class_ids,
                                            const float* __restrict__ boxes,
                                            const uint8_t* __restrict__ mask,
                                            float* __restrict__ heatmap,
                                            int   n,
                                            int   out_size_factor,
                                            float overlap,
                                            int   min_radius,
                                            float voxel_x,
                                            float voxel_y,
                                            float range_x,
                                            float range_y,
                                            int   fmap_w,
                                            int   fmap_h) {
  extern __shared__ float tile[];
  const int               tx         = threadIdx.x & (kTile - 1);
  const int               ty         = threadIdx.x >> 4;
  const int               tile_x     = blockIdx.x * kTile;
  const int               tile_y     = blockIdx.y * kTile;
  const int               channel_id = static_cast<int>(blockIdx.z);
  const int               tile_pos   = ty * kTile + tx;
  tile[tile_pos]                     = 0.0f;
  __syncthreads();

  for (int object = threadIdx.x; object < n; object += blockDim.x) {
    if (!mask[object] || static_cast<int>(class_ids[object]) - 1 != channel_id) continue;
    const float* box         = boxes + object * 9;
    const int    radius      = max(min_radius,
                           gaussian_radius(box[4] / voxel_y / static_cast<float>(out_size_factor),
                                           box[3] / voxel_x / static_cast<float>(out_size_factor),
                                           overlap));
    const int    x           = static_cast<int>((box[0] - range_x) / voxel_x / static_cast<float>(out_size_factor));
    const int    y           = static_cast<int>((box[1] - range_y) / voxel_y / static_cast<float>(out_size_factor));
    const int    left        = min(x, radius);
    const int    right       = min(fmap_w - x, radius + 1);
    const int    top         = min(y, radius);
    const int    bottom      = min(fmap_h - y, radius + 1);
    const int    diameter    = radius * 2 + 1;
    const float  denominator = 2.0f * (static_cast<float>(diameter) / 6.0f) * (static_cast<float>(diameter) / 6.0f);
    const int    tile_right  = min(tile_x + kTile, fmap_w);
    const int    tile_bottom = min(tile_y + kTile, fmap_h);
    for (int local_y = max(tile_y, y - top); local_y < min(tile_bottom, y + bottom); ++local_y) {
      for (int local_x = max(tile_x, x - left); local_x < min(tile_right, x + right); ++local_x) {
        const int   tile_index = (local_y - tile_y) * kTile + (local_x - tile_x);
        const int   dx         = local_x - x;
        const int   dy         = local_y - y;
        const float value      = expf(-static_cast<float>(dx * dx + dy * dy) / denominator);
        atomic_max_nonnegative(tile + tile_index, value);
      }
    }
  }
  __syncthreads();
  if (tile_x + tx < fmap_w && tile_y + ty < fmap_h) {
    float* channel                                = heatmap + channel_id * fmap_h * fmap_w;
    channel[(tile_y + ty) * fmap_w + tile_x + tx] = tile[tile_pos];
  }
}

void check(ffi::TensorView boxes,
           ffi::TensorView class_ids,
           ffi::TensorView heatmap,
           ffi::TensorView anno_box,
           ffi::TensorView inds,
           ffi::TensorView mask,
           int             max_objs,
           int             fmap_w,
           int             fmap_h) {
  check_musa_contiguous(boxes, "boxes", kDLFloat32);
  check_musa_contiguous(class_ids, "class_ids", kDLInt64);
  check_musa_contiguous(heatmap, "heatmap", kDLFloat32);
  check_musa_contiguous(anno_box, "anno_box", kDLFloat32);
  check_musa_contiguous(inds, "inds", kDLInt64);
  check_musa_contiguous(mask, "mask", kDLUInt8);
  check_same_musa_device(boxes, class_ids, "class_ids");
  check_same_musa_device(boxes, heatmap, "heatmap");
  check_same_musa_device(boxes, anno_box, "anno_box");
  check_same_musa_device(boxes, inds, "inds");
  check_same_musa_device(boxes, mask, "mask");
  TVM_FFI_ICHECK_EQ(boxes.ndim(), 2) << "boxes must have rank 2";
  TVM_FFI_ICHECK_EQ(boxes.size(1), 9) << "boxes must have shape [N, 9]";
  TVM_FFI_ICHECK_EQ(class_ids.ndim(), 1) << "class_ids must have rank 1";
  TVM_FFI_ICHECK_EQ(class_ids.size(0), boxes.size(0)) << "class_ids must match boxes";
  TVM_FFI_ICHECK_EQ(heatmap.ndim(), 3) << "heatmap must have rank 3";
  TVM_FFI_ICHECK_EQ(heatmap.size(1), fmap_h) << "heatmap height mismatch";
  TVM_FFI_ICHECK_EQ(heatmap.size(2), fmap_w) << "heatmap width mismatch";
  TVM_FFI_ICHECK_EQ(anno_box.size(0), max_objs) << "anno_box max_objs mismatch";
  TVM_FFI_ICHECK_EQ(anno_box.size(1), 10) << "anno_box must have shape [max_objs, 10]";
  TVM_FFI_ICHECK_EQ(inds.size(0), max_objs) << "inds max_objs mismatch";
  TVM_FFI_ICHECK_EQ(mask.size(0), max_objs) << "mask max_objs mismatch";
}
}  // namespace

void centerpoint_target_assign_forward(ffi::TensorView boxes,
                                       ffi::TensorView class_ids,
                                       ffi::TensorView heatmap,
                                       ffi::TensorView anno_box,
                                       ffi::TensorView inds,
                                       ffi::TensorView mask,
                                       int             out_size_factor,
                                       double          gaussian_overlap,
                                       int             min_radius,
                                       double          voxel_x,
                                       double          voxel_y,
                                       double          range_x,
                                       double          range_y,
                                       int             fmap_w,
                                       int             fmap_h,
                                       int             max_objs) {
  check(boxes, class_ids, heatmap, anno_box, inds, mask, max_objs, fmap_w, fmap_h);
  const int n = static_cast<int>(std::min<int64_t>(boxes.size(0), max_objs));
  if (n == 0) return;
  const int blocks = (n + kThreads - 1) / kThreads;
  prepare_targets_kernel<<<blocks, kThreads, 0, stream(boxes)>>>(static_cast<const float*>(boxes.data_ptr()),
                                                                 static_cast<float*>(anno_box.data_ptr()),
                                                                 static_cast<int64_t*>(inds.data_ptr()),
                                                                 static_cast<uint8_t*>(mask.data_ptr()),
                                                                 n,
                                                                 out_size_factor,
                                                                 static_cast<float>(voxel_x),
                                                                 static_cast<float>(voxel_y),
                                                                 static_cast<float>(range_x),
                                                                 static_cast<float>(range_y),
                                                                 fmap_w,
                                                                 fmap_h);
  // Radius-aware block sizing for the object-parallel path.  Small radii
  // benefit from lower scheduling overhead; larger patches need more
  // threads to expose Gaussian/atomic parallelism.
  const int draw_threads = min_radius <= 2 ? 64 : (min_radius <= 8 ? 128 : 256);
  // For the common single-head case, tile aggregation wins when there are
  // enough objects to overlap.  Very large N/map combinations retain the
  // object path because scanning all objects per tile would dominate.
  const bool use_tiled = (min_radius >= 6 && n <= 1000 && fmap_w <= 256 && fmap_h <= 256) ||
                         (min_radius >= 4 && n >= 128 && n <= 1000 && fmap_w <= 256 && fmap_h <= 256) ||
                         (min_radius >= 8 && n <= 256 && fmap_w <= 512 && fmap_h <= 512);
  if (use_tiled) {
    dim3 grid((fmap_w + kTile - 1) / kTile, (fmap_h + kTile - 1) / kTile, static_cast<unsigned>(heatmap.size(0)));
    draw_gaussians_tiled_kernel<<<grid, kTileThreads, kTile * kTile * sizeof(float), stream(boxes)>>>(
        static_cast<const int64_t*>(class_ids.data_ptr()),
        static_cast<const float*>(boxes.data_ptr()),
        static_cast<const uint8_t*>(mask.data_ptr()),
        static_cast<float*>(heatmap.data_ptr()),
        n,
        out_size_factor,
        static_cast<float>(gaussian_overlap),
        min_radius,
        static_cast<float>(voxel_x),
        static_cast<float>(voxel_y),
        static_cast<float>(range_x),
        static_cast<float>(range_y),
        fmap_w,
        fmap_h);
  } else {
    draw_gaussians_kernel<<<n, draw_threads, 0, stream(boxes)>>>(static_cast<const int64_t*>(class_ids.data_ptr()),
                                                                 static_cast<const float*>(boxes.data_ptr()),
                                                                 static_cast<const uint8_t*>(mask.data_ptr()),
                                                                 static_cast<float*>(heatmap.data_ptr()),
                                                                 n,
                                                                 out_size_factor,
                                                                 static_cast<float>(gaussian_overlap),
                                                                 min_radius,
                                                                 static_cast<float>(voxel_x),
                                                                 static_cast<float>(voxel_y),
                                                                 static_cast<float>(range_x),
                                                                 static_cast<float>(range_y),
                                                                 fmap_w,
                                                                 fmap_h);
  }
  check_last_error("centerpoint_target_assign_forward");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(centerpoint_target_assign_forward, centerpoint_target_assign_forward);
