#include <musa_runtime.h>
#include <mutex>
#include <unordered_map>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::stream;

namespace {

struct ScanWorkspace {
  void* ptr{nullptr};
  size_t bytes{0};
};

void* cached_scan_workspace(size_t required, musaStream_t stream_handle) {
  static std::mutex mutex;
  static std::unordered_map<uintptr_t, ScanWorkspace> workspaces;
  std::lock_guard<std::mutex> guard(mutex);
  ScanWorkspace& workspace = workspaces[reinterpret_cast<uintptr_t>(stream_handle)];
  if (workspace.bytes < required) {
    // Do not free an old buffer here: a different invocation can still have it
    // in flight. Growth is rare and the module-lifetime cache is stream-local.
    void* allocation = nullptr;
    TVM_FFI_ICHECK_EQ(musaMalloc(&allocation, required), musaSuccess);
    workspace.ptr = allocation;
    workspace.bytes = required;
  }
  return workspace.ptr;
}

// Match CUDA's explicit FP32 subtract/divide sequence at voxel boundaries.
// The volatile intermediates prevent target-specific contraction/reciprocal
// lowering from changing which side of an integer boundary a point lands on.
__device__ __forceinline__ int quantize_coord(float point, float lower, float size) {
  // NOTE: MUSA's current FP32 division lowering may use a reciprocal-multiply
  // sequence, while CUDA's division can round a near-integer quotient to the
  // adjacent FP32 value.  Consequently, an exceptionally close voxel-boundary
  // point can land in a neighboring bin compared with the CUDA implementation
  // (for example, CUDA floor(73.99999) == 73 while MUSA can observe 74.0).
  // Keep the native FP32 formula here; a global epsilon would move legitimate
  // points and creates more mismatches.  A bit-exact boundary rule can be added
  // later once a matching MUSA division intrinsic is available.
  volatile float delta = point - lower;
  volatile float scaled = delta / size;
  return (int)floorf(scaled);
}

__global__ void dynamic_kernel(const float* points, int* coors, const float* voxel,
                               const float* bounds, int n, int features) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* p = points + i * features;
  int* c = coors + i * 3;
  if (isnan(p[0])) { c[0] = -1; return; }
  int grid_x = (int)roundf((bounds[3] - bounds[0]) / voxel[0]);
  int grid_y = (int)roundf((bounds[4] - bounds[1]) / voxel[1]);
  int grid_z = (int)roundf((bounds[5] - bounds[2]) / voxel[2]);
  int x = quantize_coord(p[0], bounds[0], voxel[0]);
  if (x < 0 || x >= grid_x) { c[0] = -1; return; }
  if (isnan(p[1])) { c[0] = -1; c[1] = -1; return; }
  int y = quantize_coord(p[1], bounds[1], voxel[1]);
  if (y < 0 || y >= grid_y) { c[0] = -1; c[1] = -1; return; }
  if (isnan(p[2])) { c[0] = -1; c[1] = -1; c[2] = -1; return; }
  int z = quantize_coord(p[2], bounds[2], voxel[2]);
  if (z < 0 || z >= grid_z) { c[0] = -1; c[1] = -1; c[2] = -1; return; }
  c[0] = z;
  c[1] = y;
  c[2] = x;
}

// Deterministic first-seen fallback. Large workloads should use the parallel path
// selected by the native implementation once a workspace is supplied.
__global__ void hard_kernel(const float* points, float* voxels, int* coors, int* counts, int* seen,
                            int* voxel_num, const float* voxel, const float* bounds,
                            int n, int features, int max_points, int max_voxels) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int grid_x = (int)roundf((bounds[3] - bounds[0]) / voxel[0]);
  int grid_y = (int)roundf((bounds[4] - bounds[1]) / voxel[1]);
  int grid_z = (int)roundf((bounds[5] - bounds[2]) / voxel[2]);
  int used = 0;
  for (int i = 0; i < n; ++i) {
    const float* p = points + i * features;
    if (isnan(p[0]) || isnan(p[1]) || isnan(p[2])) continue;
    int x = quantize_coord(p[0], bounds[0], voxel[0]);
    int y = quantize_coord(p[1], bounds[1], voxel[1]);
    int z = quantize_coord(p[2], bounds[2], voxel[2]);
    if (x < 0 || x >= grid_x || y < 0 || y >= grid_y || z < 0 || z >= grid_z) continue;
    int which = -1;
    for (int v = 0; v < used; ++v) {
      if (seen[v * 3] == x && seen[v * 3 + 1] == y && seen[v * 3 + 2] == z) {
        which = v;
        break;
      }
    }
    if (which < 0) {
      if (used >= max_voxels) continue;
      which = used++;
      seen[which * 3] = x;
      seen[which * 3 + 1] = y;
      seen[which * 3 + 2] = z;
      coors[which * 3] = z;
      coors[which * 3 + 1] = y;
      coors[which * 3 + 2] = x;
    }
    int pos = counts[which];
    if (pos < max_points) {
      for (int f = 0; f < features; ++f)
        voxels[(which * max_points + pos) * features + f] = p[f];
      counts[which] = pos + 1;
    }
  }
  *voxel_num = used;
}

__global__ void compact_compute_keys_kernel(const float* points, int* point_keys, int* occupied,
                                            const float* voxel, const float* bounds, int n, int features,
                                            int grid_x, int grid_y, int grid_z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* p = points + i * features;
  if (isnan(p[0]) || isnan(p[1]) || isnan(p[2])) {
    point_keys[i] = -1;
    return;
  }
  int x = quantize_coord(p[0], bounds[0], voxel[0]);
  int y = quantize_coord(p[1], bounds[1], voxel[1]);
  int z = quantize_coord(p[2], bounds[2], voxel[2]);
  if (x < 0 || x >= grid_x || y < 0 || y >= grid_y || z < 0 || z >= grid_z) {
    point_keys[i] = -1;
    return;
  }
  int key = (z * grid_y + y) * grid_x + x;
  point_keys[i] = key;
  atomicCAS(&occupied[key], 0, 1);
}

__global__ void compact_scatter_kernel(const float* points, const int* point_keys, const int* offsets,
                                       float* voxels, int* counts, int n, int features,
                                       int max_points, int max_voxels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int key = point_keys[i];
  if (key < 0) return;
  int output = offsets[key];
  if (output >= max_voxels) return;
  // counts may overshoot max_points here; compact_write_voxel_num_kernel clamps it.
  int position = atomicAdd(&counts[output], 1);
  if (position >= max_points) return;
  const float* src = points + i * features;
  float* dst = voxels + (output * max_points + position) * features;
  for (int f = 0; f < features; ++f) dst[f] = src[f];
}

__global__ void compact_write_voxel_num_kernel(const int* occupied, const int* offsets, int* counts,
                                                int* voxel_num, int grid_total, int max_voxels,
                                                int max_points) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    int total = grid_total ? offsets[grid_total - 1] + occupied[grid_total - 1] : 0;
    *voxel_num = total < max_voxels ? total : max_voxels;
  }
  if (i < max_voxels && counts[i] > max_points) counts[i] = max_points;
}

// Ordered exclusive prefix sum over the occupied-cell flags. Single-pass
// decoupled look-back scans spin on global-memory status words and can hang
// on toolchains with broken fence codegen; kernel boundaries synchronize
// these three passes instead.
constexpr int kScanThreads = 1024;
constexpr int kScanWarps = kScanThreads / 32;
constexpr int kScanIpt = 4;
constexpr int kScanTile = kScanThreads * kScanIpt;

__device__ __forceinline__ int warp_inclusive_scan(int value, int lane) {
  for (int offset = 1; offset < 32; offset <<= 1) {
    int other = __shfl_up_sync(0xffffffffu, value, offset);
    if (lane >= offset) value += other;
  }
  return value;
}

// Each thread owns kScanIpt consecutive cells so a block covers kScanTile cells;
// full tiles use aligned int4 loads.
__global__ void occupied_block_sums_kernel(const int* occupied, int* block_sums, int grid_total) {
  __shared__ int warp_sums[kScanWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int base = blockIdx.x * kScanTile + threadIdx.x * kScanIpt;
  int value = 0;
  if (base + kScanIpt <= grid_total) {
    const int4 packed = *reinterpret_cast<const int4*>(occupied + base);
    value = (packed.x != 0) + (packed.y != 0) + (packed.z != 0) + (packed.w != 0);
  } else {
    for (int j = 0; j < kScanIpt && base + j < grid_total; ++j) value += occupied[base + j] != 0 ? 1 : 0;
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  if (warp == 0) {
    int total = warp_sums[lane];
    for (int offset = 16; offset > 0; offset >>= 1)
      total += __shfl_down_sync(0xffffffffu, total, offset);
    if (lane == 0) block_sums[blockIdx.x] = total;
  }
}

__global__ void block_offsets_scan_kernel(int* block_sums, int num_blocks) {
  __shared__ int warp_offsets[kScanWarps];
  __shared__ int chunk_total;
  __shared__ int carry;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  for (int base = 0; base < num_blocks; base += blockDim.x) {
    int i = base + threadIdx.x;
    int value = (i < num_blocks) ? block_sums[i] : 0;
    int inclusive = warp_inclusive_scan(value, lane);
    if (lane == 31) warp_offsets[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
      int w = warp_offsets[lane];
      int w_inclusive = warp_inclusive_scan(w, lane);
      warp_offsets[lane] = w_inclusive - w;
      if (lane == 31) chunk_total = w_inclusive;
    }
    __syncthreads();
    int base_offset = carry;
    if (i < num_blocks) block_sums[i] = base_offset + warp_offsets[warp] + inclusive - value;
    __syncthreads();
    if (threadIdx.x == 0) carry = base_offset + chunk_total;
    __syncthreads();
  }
}

__global__ void occupied_prefix_kernel(const int* occupied, const int* block_offsets, int* offsets,
                                       int* coors, int grid_total, int max_voxels, int grid_x,
                                       int grid_y) {
  __shared__ int warp_offsets[kScanWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int base = blockIdx.x * kScanTile + threadIdx.x * kScanIpt;
  int flags[kScanIpt] = {};
  if (base + kScanIpt <= grid_total) {
    const int4 packed = *reinterpret_cast<const int4*>(occupied + base);
    flags[0] = packed.x != 0;
    flags[1] = packed.y != 0;
    flags[2] = packed.z != 0;
    flags[3] = packed.w != 0;
  } else {
    for (int j = 0; j < kScanIpt && base + j < grid_total; ++j) flags[j] = occupied[base + j] != 0 ? 1 : 0;
  }
  int local[kScanIpt];
  int thread_sum = 0;
#pragma unroll
  for (int j = 0; j < kScanIpt; ++j) {
    local[j] = thread_sum;
    thread_sum += flags[j];
  }
  int inclusive = warp_inclusive_scan(thread_sum, lane);
  if (lane == 31) warp_offsets[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    int w = warp_offsets[lane];
    warp_offsets[lane] = warp_inclusive_scan(w, lane) - w;
  }
  __syncthreads();
  const int thread_base = block_offsets[blockIdx.x] + warp_offsets[warp] + inclusive - thread_sum;
  if (base + kScanIpt <= grid_total) {
    int4 packed_out;
    packed_out.x = thread_base + local[0];
    packed_out.y = thread_base + local[1];
    packed_out.z = thread_base + local[2];
    packed_out.w = thread_base + local[3];
    *reinterpret_cast<int4*>(offsets + base) = packed_out;
  } else {
    for (int j = 0; j < kScanIpt && base + j < grid_total; ++j) offsets[base + j] = thread_base + local[j];
  }
#pragma unroll
  for (int j = 0; j < kScanIpt; ++j) {
    const int i = base + j;
    if (i < grid_total && flags[j]) {
      const int output = thread_base + local[j];
      if (output < max_voxels) {
        int x = i % grid_x;
        int yz = i / grid_x;
        coors[output * 3] = yz / grid_y;
        coors[output * 3 + 1] = yz % grid_y;
        coors[output * 3 + 2] = x;
      }
    }
  }
}

}  // namespace

void dynamic_voxelize_forward(ffi::TensorView points, ffi::TensorView voxel_size,
                              ffi::TensorView coors_range, ffi::TensorView coors) {
  check_musa_contiguous(points, "points", kDLFloat32);
  check_musa_contiguous(voxel_size, "voxel_size", kDLFloat32);
  check_musa_contiguous(coors_range, "coors_range", kDLFloat32);
  check_musa_contiguous(coors, "coors", kDLInt32);
  TVM_FFI_ICHECK_EQ(points.ndim(), 2) << "points must be [N,F]";
  TVM_FFI_ICHECK_EQ(points.size(0), coors.size(0));
  if (points.size(0) == 0) return;
  dynamic_kernel<<<(static_cast<int>(points.size(0)) + 255) / 256, 256, 0, stream(points)>>>(
      static_cast<const float*>(points.data_ptr()), static_cast<int*>(coors.data_ptr()),
      static_cast<const float*>(voxel_size.data_ptr()), static_cast<const float*>(coors_range.data_ptr()),
      static_cast<int>(points.size(0)), static_cast<int>(points.size(1)));
  check_last_error("dynamic_voxelize_forward");
}

void hard_voxelize_forward(ffi::TensorView points, ffi::TensorView voxel_size,
                           ffi::TensorView coors_range, ffi::TensorView voxels,
                           ffi::TensorView coors, ffi::TensorView counts,
                           ffi::TensorView voxel_num, ffi::TensorView coord_to_voxel,
                           ffi::TensorView coord_offsets, ffi::TensorView point_keys,
                           int max_points, int max_voxels,
                           int grid_x, int grid_y, int grid_z, bool deterministic) {
  check_musa_contiguous(points, "points", kDLFloat32);
  check_musa_contiguous(voxel_size, "voxel_size", kDLFloat32);
  check_musa_contiguous(coors_range, "coors_range", kDLFloat32);
  check_musa_contiguous(voxels, "voxels", kDLFloat32);
  check_musa_contiguous(coors, "coors", kDLInt32);
  check_musa_contiguous(counts, "counts", kDLInt32);
  check_musa_contiguous(voxel_num, "voxel_num", kDLInt32);
  check_musa_contiguous(coord_to_voxel, "coord_to_voxel", kDLInt32);
  check_musa_contiguous(coord_offsets, "coord_offsets", kDLInt32);
  check_musa_contiguous(point_keys, "point_keys", kDLInt32);
  TVM_FFI_ICHECK_EQ(points.ndim(), 2) << "points must be [N,F]";
  TVM_FFI_ICHECK_EQ(voxels.ndim(), 3) << "voxels must be [M,P,F]";
  TVM_FFI_ICHECK_EQ(voxels.size(0), max_voxels);
  TVM_FFI_ICHECK_EQ(voxels.size(1), max_points);
  TVM_FFI_ICHECK_EQ(voxels.size(2), points.size(1));
  if (deterministic) {
    // Reference-order serial path: the toolchain's device pair sort does not
    // preserve the key/index pairing, so a sort-based stable path is not viable.
    int* seen = nullptr;
    TVM_FFI_ICHECK_EQ(musaMalloc(reinterpret_cast<void**>(&seen), static_cast<size_t>(max_voxels) * 3 * sizeof(int)), musaSuccess);
    musaMemsetAsync(voxel_num.data_ptr(), 0, sizeof(int), stream(points));
    musaMemsetAsync(counts.data_ptr(), 0, static_cast<size_t>(max_voxels) * sizeof(int), stream(points));
    musaMemsetAsync(seen, 0xff, static_cast<size_t>(max_voxels) * 3 * sizeof(int), stream(points));
    hard_kernel<<<1, 1, 0, stream(points)>>>(
        static_cast<const float*>(points.data_ptr()), static_cast<float*>(voxels.data_ptr()),
        static_cast<int*>(coors.data_ptr()), static_cast<int*>(counts.data_ptr()), seen,
        static_cast<int*>(voxel_num.data_ptr()), static_cast<const float*>(voxel_size.data_ptr()),
        static_cast<const float*>(coors_range.data_ptr()), static_cast<int>(points.size(0)),
        static_cast<int>(points.size(1)), max_points, max_voxels);
    check_last_error("hard_voxelize_forward");
    TVM_FFI_ICHECK_EQ(musaFree(seen), musaSuccess);
    return;
  }
  if (!deterministic) {
    int grid_total = grid_x * grid_y * grid_z;
    TVM_FFI_ICHECK_GE(coord_to_voxel.size(0), grid_total) << "coord_to_voxel workspace is too small";
    TVM_FFI_ICHECK_GE(coord_offsets.size(0), grid_total) << "coord_offsets workspace is too small";
    TVM_FFI_ICHECK_GE(point_keys.size(0), points.size(0)) << "point_keys workspace is too small";
    int* coord_map = static_cast<int*>(coord_to_voxel.data_ptr());
    int* offsets = static_cast<int*>(coord_offsets.data_ptr());
    int* keys = static_cast<int*>(point_keys.data_ptr());
    musaMemsetAsync(coord_map, 0, static_cast<size_t>(grid_total) * sizeof(int), stream(points));
    musaMemsetAsync(voxel_num.data_ptr(), 0, sizeof(int), stream(points));
    musaMemsetAsync(counts.data_ptr(), 0, static_cast<size_t>(max_voxels) * sizeof(int), stream(points));
    compact_compute_keys_kernel<<<(static_cast<int>(points.size(0)) + 255) / 256, 256, 0, stream(points)>>>(
        static_cast<const float*>(points.data_ptr()), keys, coord_map,
        static_cast<const float*>(voxel_size.data_ptr()), static_cast<const float*>(coors_range.data_ptr()),
        static_cast<int>(points.size(0)), static_cast<int>(points.size(1)), grid_x, grid_y, grid_z);
    const int scan_blocks = (grid_total + kScanTile - 1) / kScanTile;
    int* block_sums = static_cast<int*>(cached_scan_workspace(
        static_cast<size_t>(scan_blocks) * sizeof(int), stream(points)));
    occupied_block_sums_kernel<<<scan_blocks, kScanThreads, 0, stream(points)>>>(coord_map, block_sums, grid_total);
    block_offsets_scan_kernel<<<1, kScanThreads, 0, stream(points)>>>(block_sums, scan_blocks);
    occupied_prefix_kernel<<<scan_blocks, kScanThreads, 0, stream(points)>>>(
        coord_map, block_sums, offsets, static_cast<int*>(coors.data_ptr()), grid_total, max_voxels,
        grid_x, grid_y);
    compact_scatter_kernel<<<(static_cast<int>(points.size(0)) + 255) / 256, 256, 0, stream(points)>>>(
        static_cast<const float*>(points.data_ptr()), keys, offsets, static_cast<float*>(voxels.data_ptr()),
        static_cast<int*>(counts.data_ptr()), static_cast<int>(points.size(0)), static_cast<int>(points.size(1)),
        max_points, max_voxels);
    const int clamp_blocks = (max_voxels + 255) / 256;
    compact_write_voxel_num_kernel<<<clamp_blocks > 0 ? clamp_blocks : 1, 256, 0, stream(points)>>>(
        coord_map, offsets, static_cast<int*>(counts.data_ptr()),
        static_cast<int*>(voxel_num.data_ptr()), grid_total, max_voxels, max_points);
    check_last_error("hard_parallel_voxelize_forward");
    return;
  }
  int* seen = nullptr;
  TVM_FFI_ICHECK_EQ(musaMalloc(reinterpret_cast<void**>(&seen), static_cast<size_t>(max_voxels) * 3 * sizeof(int)),
                    musaSuccess) << "hard_voxelize workspace allocation failed";
  musaMemsetAsync(voxel_num.data_ptr(), 0, sizeof(int), stream(points));
  musaMemsetAsync(counts.data_ptr(), 0, static_cast<size_t>(max_voxels) * sizeof(int), stream(points));
  musaMemsetAsync(seen, 0xff, static_cast<size_t>(max_voxels) * 3 * sizeof(int), stream(points));
  hard_kernel<<<1, 1, 0, stream(points)>>>(
      static_cast<const float*>(points.data_ptr()), static_cast<float*>(voxels.data_ptr()),
      static_cast<int*>(coors.data_ptr()), static_cast<int*>(counts.data_ptr()),
      seen,
      static_cast<int*>(voxel_num.data_ptr()), static_cast<const float*>(voxel_size.data_ptr()),
      static_cast<const float*>(coors_range.data_ptr()), static_cast<int>(points.size(0)),
      static_cast<int>(points.size(1)), max_points, max_voxels);
  check_last_error("hard_voxelize_forward");
  TVM_FFI_ICHECK_EQ(musaFree(seen), musaSuccess) << "hard_voxelize workspace free failed";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dynamic_voxelize_forward, dynamic_voxelize_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(hard_voxelize_forward, hard_voxelize_forward);
