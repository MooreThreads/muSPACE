// Depth-aware voxel pooling for the Qwen-Drive-1.0 view transform.
//
// The Python layer performs the frustum bookkeeping that connects the image
// plane to the voxel grid: it compacts the valid frustum cells with `mask` and
// keeps their linear point indices.  These kernels consume that compacted
// representation, so they only unpack a linear point index into
// `(batch, sweep, cam, d, h, w)` and never build it.
//
// Forward accumulates into the zero-initialised output with `atomicAdd`, so a
// warp no longer has to walk a sorted rank interval.  Backward fuses both
// gradients into one warp-per-cell kernel, accumulates in float32 and casts
// back to the caller's precision in Python.

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "../ffi/fast_divmod.hpp"
#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::dim;
using muspace::ffi_utils::FastDivmod;
using muspace::ffi_utils::stream;

namespace {

constexpr DLDataType kDLFloat16 = DLDataType{kDLFloat, 16, 1};

constexpr int     kThreads  = 256;
constexpr int     kWarpSize = 32;
constexpr int64_t kInt32Max = 2147483647LL;

inline int blocks_for(const int64_t size) {
  return static_cast<int>((size + kThreads - 1) / kThreads);
}

enum class LaunchStrategy {
  kAuto,
  kLinear,
  kTwoDimensional,
  kPacked,
};

LaunchStrategy launch_strategy_from_env() {
  const char* value = std::getenv("MUSPACE_VOXEL_POOL_DEPTH_LAUNCH");
  if (value == nullptr) {
    return LaunchStrategy::kAuto;
  }
  const std::string_view strategy(value);
  if (strategy == "linear") {
    return LaunchStrategy::kLinear;
  }
  if (strategy == "2d") {
    return LaunchStrategy::kTwoDimensional;
  }
  if (strategy == "packed") {
    return LaunchStrategy::kPacked;
  }
  return LaunchStrategy::kAuto;
}

__device__ __forceinline__ float to_float(const float value) {
  return value;
}

__device__ __forceinline__ float to_float(const __half value) {
  return __half2float(value);
}

__device__ __forceinline__ float to_float(const __mt_bfloat16 value) {
  return __bfloat162float(value);
}

// Reciprocal divisors for the index arithmetic the kernels repeat. Unpacking a
// frustum cell costs ten integer divisions in the naive form and it runs once
// per cell per channel, so the inner loop replaces them with five
// multiply-shift pairs.
struct KernelDivmods {
  FastDivmod point_w;
  FastDivmod point_h;
  FastDivmod point_d;
  FastDivmod point_cam;
  FastDivmod point_sweep;
  FastDivmod channels;
  FastDivmod packed_channels;
  FastDivmod depth_chunks;
};

KernelDivmods make_divmods(const int W,
                           const int H,
                           const int D,
                           const int N_cam,
                           const int N_sweep,
                           const int channels,
                           const int depth_chunk = 1) {
  KernelDivmods divmods;
  divmods.point_w         = FastDivmod((uint32_t)W);
  divmods.point_h         = FastDivmod((uint32_t)H);
  divmods.point_d         = FastDivmod((uint32_t)D);
  divmods.point_cam       = FastDivmod((uint32_t)N_cam);
  divmods.point_sweep     = FastDivmod((uint32_t)N_sweep);
  divmods.channels        = FastDivmod((uint32_t)channels);
  divmods.packed_channels = FastDivmod((uint32_t)(channels / 4));
  divmods.depth_chunks    = FastDivmod((uint32_t)((channels + depth_chunk - 1) / depth_chunk));
  return divmods;
}

__device__ __forceinline__ void divmod_u32(const FastDivmod& divmod, const int value, int& quotient, int& remainder) {
  uint32_t q = 0;
  uint32_t r = 0;
  divmod(q, r, (uint32_t)value);
  quotient  = (int)q;
  remainder = (int)r;
}

// Unpacks a linear frustum index laid out as [batch, sweep, cam, d, h, w].
struct PointLocation {
  int image_index;
  int d;
  int h;
  int w;
  int cam;
};

__device__ __forceinline__ PointLocation unpack_point(const int            point_index,
                                                      const KernelDivmods& divmods,
                                                      const int            N_sweep,
                                                      const int            N_cam) {
  PointLocation location;
  int           p = point_index;
  int           q = 0;
  int           r = 0;
  divmod_u32(divmods.point_w, p, q, r);
  location.w = r;
  p          = q;
  divmod_u32(divmods.point_h, p, q, r);
  location.h = r;
  p          = q;
  divmod_u32(divmods.point_d, p, q, r);
  location.d = r;
  p          = q;
  divmod_u32(divmods.point_cam, p, q, r);
  location.cam = r;
  p            = q;
  divmod_u32(divmods.point_sweep, p, q, r);
  location.image_index = ((q * N_sweep + r) * N_cam) + location.cam;
  return location;
}

struct ForwardArgs {
  int           N_sweep;
  int           N_cam;
  int           X;
  int           Y;
  int           Z;
  int           D;
  int           H;
  int           W;
  int           feat_channels;
  int           n_valid;
  KernelDivmods divmods;
  const void*   img_feats;
  const void*   img_depth;
  const int*    coords;
  const int*    point_indices;
  float*        out;
};

struct BackwardArgs {
  int           N_sweep;
  int           N_cam;
  int           X;
  int           Y;
  int           Z;
  int           D;
  int           H;
  int           W;
  int           feat_channels;
  int           n_valid;
  KernelDivmods divmods;
  const void*   out_grad;
  const void*   img_feats;
  const void*   img_depth;
  const int*    coords;
  const int*    point_indices;
  float*        img_feats_grad;
  float*        img_depth_grad;
};

// One thread per (valid cell, channel): the warp-friendly baseline that mirrors
// the upstream Qwen-Drive launch shape.
template <typename feat_t, typename depth_t>
__global__ void voxel_pool_depth_forward_kernel(const ForwardArgs args) {
  const int index       = blockIdx.x * blockDim.x + threadIdx.x;
  int       valid_index = 0;
  int       channel     = 0;
  divmod_u32(args.divmods.channels, index, valid_index, channel);
  if (valid_index >= args.n_valid) {
    return;
  }

  const PointLocation loc       = unpack_point(args.point_indices[valid_index], args.divmods, args.N_sweep, args.N_cam);
  const int*          coords    = args.coords + valid_index * 4;
  const feat_t*       img_feats = (const feat_t*)args.img_feats;
  const depth_t*      img_depth = (const depth_t*)args.img_depth;

  const int feat_offset  = ((loc.image_index * args.feat_channels + channel) * args.H + loc.h) * args.W + loc.w;
  const int depth_offset = ((loc.image_index * args.D + loc.d) * args.H + loc.h) * args.W + loc.w;
  const int out_offset =
      ((((coords[0] * args.N_cam + loc.cam) * args.X + coords[1]) * args.Y + coords[2]) * args.Z + coords[3]) *
          args.feat_channels +
      channel;

  atomicAdd(args.out + out_offset, to_float(img_feats[feat_offset]) * to_float(img_depth[depth_offset]));
}

// Experimental alternative mapping: the channel index moves to `blockIdx.y` so
// the per-thread integer division disappears. It loses to the packed kernel on
// every measured shape because a warp then spans 256 cells, which scatters the
// `coords` loads and the output atomics, so it is reachable only through
// `MUSPACE_VOXEL_POOL_DEPTH_LAUNCH=2d`.
template <typename feat_t, typename depth_t>
__global__ void voxel_pool_depth_forward_2d_kernel(const ForwardArgs args) {
  const int valid_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int channel     = blockIdx.y;
  if (valid_index >= args.n_valid || channel >= args.feat_channels) {
    return;
  }

  const PointLocation loc       = unpack_point(args.point_indices[valid_index], args.divmods, args.N_sweep, args.N_cam);
  const int*          coords    = args.coords + valid_index * 4;
  const feat_t*       img_feats = (const feat_t*)args.img_feats;
  const depth_t*      img_depth = (const depth_t*)args.img_depth;

  const int feat_offset  = ((loc.image_index * args.feat_channels + channel) * args.H + loc.h) * args.W + loc.w;
  const int depth_offset = ((loc.image_index * args.D + loc.d) * args.H + loc.h) * args.W + loc.w;
  const int out_offset =
      ((((coords[0] * args.N_cam + loc.cam) * args.X + coords[1]) * args.Y + coords[2]) * args.Z + coords[3]) *
          args.feat_channels +
      channel;

  atomicAdd(args.out + out_offset, to_float(img_feats[feat_offset]) * to_float(img_depth[depth_offset]));
}

// Four adjacent channels per thread. Feature channels live H*W apart, so the
// loads stay scalar, but the index bookkeeping drops by 4x and the four output
// atomics stay inside one 16-byte span.
template <typename feat_t, typename depth_t>
__global__ void voxel_pool_depth_forward_packed_kernel(const ForwardArgs args) {
  const int index          = blockIdx.x * blockDim.x + threadIdx.x;
  int       valid_index    = 0;
  int       packed_channel = 0;
  divmod_u32(args.divmods.packed_channels, index, valid_index, packed_channel);
  if (valid_index >= args.n_valid) {
    return;
  }

  const PointLocation loc       = unpack_point(args.point_indices[valid_index], args.divmods, args.N_sweep, args.N_cam);
  const int*          coords    = args.coords + valid_index * 4;
  const feat_t*       img_feats = (const feat_t*)args.img_feats;
  const depth_t*      img_depth = (const depth_t*)args.img_depth;
  const int           channel   = packed_channel * 4;

  const int   feat_offset    = ((loc.image_index * args.feat_channels + channel) * args.H + loc.h) * args.W + loc.w;
  const int   depth_offset   = ((loc.image_index * args.D + loc.d) * args.H + loc.h) * args.W + loc.w;
  const int   channel_stride = args.H * args.W;
  const float weight         = to_float(img_depth[depth_offset]);
  const int   out_offset =
      ((((coords[0] * args.N_cam + loc.cam) * args.X + coords[1]) * args.Y + coords[2]) * args.Z + coords[3]) *
          args.feat_channels +
      channel;

  atomicAdd(args.out + out_offset, to_float(img_feats[feat_offset]) * weight);
  atomicAdd(args.out + out_offset + 1, to_float(img_feats[feat_offset + channel_stride]) * weight);
  atomicAdd(args.out + out_offset + 2, to_float(img_feats[feat_offset + 2 * channel_stride]) * weight);
  atomicAdd(args.out + out_offset + 3, to_float(img_feats[feat_offset + 3 * channel_stride]) * weight);
}

template <typename feat_t, typename depth_t, typename grad_t>
__global__ void voxel_pool_depth_backward_feat_kernel(const BackwardArgs args) {
  const int index       = blockIdx.x * blockDim.x + threadIdx.x;
  int       valid_index = 0;
  int       channel     = 0;
  divmod_u32(args.divmods.channels, index, valid_index, channel);
  if (valid_index >= args.n_valid) {
    return;
  }

  const PointLocation loc = unpack_point(args.point_indices[valid_index], args.divmods, args.N_sweep, args.N_cam);
  const int*          cur_coords   = args.coords + valid_index * 4;
  const grad_t*       out_grad     = (const grad_t*)args.out_grad;
  const depth_t*      img_depth    = (const depth_t*)args.img_depth;
  const int           depth_offset = ((loc.image_index * args.D + loc.d) * args.H + loc.h) * args.W + loc.w;
  const int           grad_out_offset =
      ((((cur_coords[0] * args.N_cam + loc.cam) * args.X + cur_coords[1]) * args.Y + cur_coords[2]) * args.Z +
       cur_coords[3]) *
          args.feat_channels +
      channel;
  const int grad_feat_offset = ((loc.image_index * args.feat_channels + channel) * args.H + loc.h) * args.W + loc.w;

  const float grad = to_float(out_grad[grad_out_offset]) * to_float(img_depth[depth_offset]);
  atomicAdd(args.img_feats_grad + grad_feat_offset, grad);
}

// One thread per valid frustum cell reduces the feature gradient over channels,
// mirroring the upstream `[image, C, H, W]` feature layout.
template <typename feat_t, typename depth_t, typename grad_t>
__global__ void voxel_pool_depth_backward_depth_kernel(const BackwardArgs args) {
  const int valid_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (valid_index >= args.n_valid) {
    return;
  }

  const PointLocation loc = unpack_point(args.point_indices[valid_index], args.divmods, args.N_sweep, args.N_cam);
  const int*          cur_coords     = args.coords + valid_index * 4;
  const grad_t*       out_grad       = (const grad_t*)args.out_grad;
  const feat_t*       img_feats      = (const feat_t*)args.img_feats;
  const int           channel_stride = args.H * args.W;
  const int           grad_out_base =
      ((((cur_coords[0] * args.N_cam + loc.cam) * args.X + cur_coords[1]) * args.Y + cur_coords[2]) * args.Z +
       cur_coords[3]) *
      args.feat_channels;
  const int feat_base = (loc.image_index * args.feat_channels * args.H + loc.h) * args.W + loc.w;

  float grad = 0.0F;
  for (int channel = 0; channel < args.feat_channels; ++channel) {
    grad += to_float(out_grad[grad_out_base + channel]) * to_float(img_feats[feat_base + channel * channel_stride]);
  }
  const int depth_offset            = ((loc.image_index * args.D + loc.d) * args.H + loc.h) * args.W + loc.w;
  args.img_depth_grad[depth_offset] = grad;
}

// One warp per valid frustum cell. Splitting the channel dimension over the 32
// lanes exposes 32x more threads than the serial reduction and makes the
// `out_grad` read consecutive within the warp. The reduction is a fixed shuffle
// tree, so it stays deterministic and only differs from the serial order by
// float32 rounding.
template <typename feat_t, typename depth_t, typename grad_t>
__global__ void voxel_pool_depth_backward_depth_warp_kernel(const BackwardArgs args) {
  const int lane    = threadIdx.x & 31;
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp_id >= args.n_valid) {
    return;
  }

  const PointLocation loc        = unpack_point(args.point_indices[warp_id], args.divmods, args.N_sweep, args.N_cam);
  const int*          cur_coords = args.coords + warp_id * 4;
  const grad_t*       out_grad   = (const grad_t*)args.out_grad;
  const feat_t*       img_feats  = (const feat_t*)args.img_feats;
  const int           channel_stride = args.H * args.W;
  const int           grad_out_base =
      ((((cur_coords[0] * args.N_cam + loc.cam) * args.X + cur_coords[1]) * args.Y + cur_coords[2]) * args.Z +
       cur_coords[3]) *
      args.feat_channels;
  const int feat_base = (loc.image_index * args.feat_channels * args.H + loc.h) * args.W + loc.w;

  float grad = 0.0F;
  for (int channel = lane; channel < args.feat_channels; channel += 32) {
    grad += to_float(out_grad[grad_out_base + channel]) * to_float(img_feats[feat_base + channel * channel_stride]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    grad += __shfl_down_sync(0xffffffffu, grad, offset);
  }
  if (lane == 0) {
    const int depth_offset            = ((loc.image_index * args.D + loc.d) * args.H + loc.h) * args.W + loc.w;
    args.img_depth_grad[depth_offset] = grad;
  }
}

// One warp per valid frustum cell for both gradients at once. The warp unpacks
// the cell once, then each lane walks a strided slice of the channels: it
// scatters the feature gradient with the depth weight and folds the feature
// value into the depth-gradient reduction. Merging the two passes halves the
// `out_grad` traffic and removes the per-(cell, channel) index arithmetic that
// the standalone feature-gradient kernel repeated for every channel of a cell.
template <typename feat_t, typename depth_t, typename grad_t>
__global__ void voxel_pool_depth_backward_fused_warp_kernel(const BackwardArgs args) {
  const int lane    = threadIdx.x & 31;
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp_id >= args.n_valid) {
    return;
  }

  const PointLocation loc        = unpack_point(args.point_indices[warp_id], args.divmods, args.N_sweep, args.N_cam);
  const int*          cur_coords = args.coords + warp_id * 4;
  const grad_t*       out_grad   = (const grad_t*)args.out_grad;
  const feat_t*       img_feats  = (const feat_t*)args.img_feats;
  const depth_t*      img_depth  = (const depth_t*)args.img_depth;
  const int           channel_stride = args.H * args.W;
  const int           grad_out_base =
      ((((cur_coords[0] * args.N_cam + loc.cam) * args.X + cur_coords[1]) * args.Y + cur_coords[2]) * args.Z +
       cur_coords[3]) *
      args.feat_channels;
  const int   feat_base    = (loc.image_index * args.feat_channels * args.H + loc.h) * args.W + loc.w;
  const int   depth_offset = ((loc.image_index * args.D + loc.d) * args.H + loc.h) * args.W + loc.w;
  const float weight       = to_float(img_depth[depth_offset]);

  float depth_grad = 0.0F;
  for (int channel = lane; channel < args.feat_channels; channel += kWarpSize) {
    const float grad             = to_float(out_grad[grad_out_base + channel]);
    const int   feat_grad_offset = feat_base + channel * channel_stride;
    atomicAdd(args.img_feats_grad + feat_grad_offset, grad * weight);
    depth_grad += grad * to_float(img_feats[feat_grad_offset]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    depth_grad += __shfl_down_sync(0xffffffffu, depth_grad, offset);
  }
  if (lane == 0) {
    args.img_depth_grad[depth_offset] = depth_grad;
  }
}

bool two_dimensional_requested() {
  return launch_strategy_from_env() == LaunchStrategy::kTwoDimensional;
}

bool use_packed_launch(const int channels, const bool out_aligned) {
  const LaunchStrategy strategy = launch_strategy_from_env();
  if (strategy == LaunchStrategy::kLinear || strategy == LaunchStrategy::kTwoDimensional) {
    return false;
  }
  // Four adjacent channels per thread costs three quarters of the threads, so
  // packing only pays off once the per-thread index work outweighs that loss:
  // on S5000 it draws level with the linear kernel through 32 channels and wins
  // from 64 upward, and is never slower on the measured shapes.
  return channels >= 4 && channels % 4 == 0 && out_aligned;
}

// The warp reduction trades a fixed shuffle tree for 32x more threads and a
// consecutive `out_grad` read. `MUSPACE_VOXEL_POOL_DEPTH_DEPTH_WARP=0` forces
// the serial per-cell reduction.
bool use_warp_depth_launch(const int channels) {
  const char* value = std::getenv("MUSPACE_VOXEL_POOL_DEPTH_DEPTH_WARP");
  if (value != nullptr && std::string_view(value) == "0") {
    return false;
  }
  return channels >= kWarpSize;
}

// The fused warp kernel needs a full warp per cell to cover the channel slice.
// `MUSPACE_VOXEL_POOL_DEPTH_FUSED_BACKWARD=0` restores the two separate passes.
bool use_fused_backward_launch(const int channels) {
  const char* value = std::getenv("MUSPACE_VOXEL_POOL_DEPTH_FUSED_BACKWARD");
  if (value != nullptr && std::string_view(value) == "0") {
    return false;
  }
  return channels >= kWarpSize;
}

template <typename feat_t, typename depth_t>
void launch_forward(const ForwardArgs& args, const musaStream_t stream) {
  const bool out_aligned = reinterpret_cast<uintptr_t>(args.out) % 16 == 0;
  if (use_packed_launch(args.feat_channels, out_aligned)) {
    const int packed_channels = args.feat_channels / 4;
    voxel_pool_depth_forward_packed_kernel<feat_t, depth_t>
        <<<blocks_for(static_cast<int64_t>(args.n_valid) * packed_channels), kThreads, 0, stream>>>(args);
  } else if (two_dimensional_requested()) {
    const dim3 blocks(blocks_for(args.n_valid), args.feat_channels);
    voxel_pool_depth_forward_2d_kernel<feat_t, depth_t><<<blocks, kThreads, 0, stream>>>(args);
  } else {
    voxel_pool_depth_forward_kernel<feat_t, depth_t>
        <<<blocks_for(static_cast<int64_t>(args.n_valid) * args.feat_channels), kThreads, 0, stream>>>(args);
  }
}

template <typename feat_t>
void dispatch_forward_depth(const ForwardArgs& args, const musaStream_t stream, const int depth_code) {
  switch (depth_code) {
    case 0:
      launch_forward<feat_t, float>(args, stream);
      break;
    case 1:
      launch_forward<feat_t, __half>(args, stream);
      break;
    default:
      launch_forward<feat_t, __mt_bfloat16>(args, stream);
      break;
  }
}

void dispatch_forward(const ForwardArgs& args, const musaStream_t stream, const int feat_code, const int depth_code) {
  switch (feat_code) {
    case 0:
      dispatch_forward_depth<float>(args, stream, depth_code);
      break;
    case 1:
      dispatch_forward_depth<__half>(args, stream, depth_code);
      break;
    default:
      dispatch_forward_depth<__mt_bfloat16>(args, stream, depth_code);
      break;
  }
}

template <typename feat_t, typename depth_t>
void launch_backward(const BackwardArgs& args, const musaStream_t stream) {
  if (args.n_valid > 0 && args.feat_channels > 0) {
    if (use_fused_backward_launch(args.feat_channels)) {
      voxel_pool_depth_backward_fused_warp_kernel<feat_t, depth_t, float>
          <<<blocks_for(static_cast<int64_t>(args.n_valid) * kWarpSize), kThreads, 0, stream>>>(args);
      return;
    }
    voxel_pool_depth_backward_feat_kernel<feat_t, depth_t, float>
        <<<blocks_for(static_cast<int64_t>(args.n_valid) * args.feat_channels), kThreads, 0, stream>>>(args);
    if (use_warp_depth_launch(args.feat_channels)) {
      voxel_pool_depth_backward_depth_warp_kernel<feat_t, depth_t, float>
          <<<blocks_for(static_cast<int64_t>(args.n_valid) * kWarpSize), kThreads, 0, stream>>>(args);
    } else {
      voxel_pool_depth_backward_depth_kernel<feat_t, depth_t, float>
          <<<blocks_for(args.n_valid), kThreads, 0, stream>>>(args);
    }
  }
}

template <typename feat_t>
void dispatch_backward_depth(const BackwardArgs& args, const musaStream_t stream, const int depth_code) {
  switch (depth_code) {
    case 0:
      launch_backward<feat_t, float>(args, stream);
      break;
    case 1:
      launch_backward<feat_t, __half>(args, stream);
      break;
    default:
      launch_backward<feat_t, __mt_bfloat16>(args, stream);
      break;
  }
}

void dispatch_backward(const BackwardArgs& args, const musaStream_t stream, const int feat_code, const int depth_code) {
  switch (feat_code) {
    case 0:
      dispatch_backward_depth<float>(args, stream, depth_code);
      break;
    case 1:
      dispatch_backward_depth<__half>(args, stream, depth_code);
      break;
    default:
      dispatch_backward_depth<__mt_bfloat16>(args, stream, depth_code);
      break;
  }
}

// 0 for float32, 1 for float16, 2 for bfloat16, -1 otherwise.
int float_dtype_code(const DLDataType dtype) {
  if (same_dtype(dtype, kDLFloat32)) {
    return 0;
  }
  if (same_dtype(dtype, kDLFloat16)) {
    return 1;
  }
  if (same_dtype(dtype, kDLBFloat16)) {
    return 2;
  }
  return -1;
}

void check_int32_contiguous(ffi::TensorView tensor, const char* name) {
  check_musa_contiguous(tensor, name, kDLInt32);
}

}  // namespace

void voxel_pool_depth_forward(ffi::TensorView img_feats,
                              ffi::TensorView img_depth,
                              ffi::TensorView out,
                              ffi::TensorView coords,
                              ffi::TensorView point_indices,
                              int64_t         B,
                              int64_t         N_sweep,
                              int64_t         N_cam,
                              int64_t         X,
                              int64_t         Y,
                              int64_t         Z,
                              int64_t         D,
                              int64_t         H,
                              int64_t         W) {
  TVM_FFI_ICHECK_EQ(img_feats.ndim(), 5) << "img_feats must have shape [B, N_sweep * N_cam, C, H, W]";
  const int feat_code = float_dtype_code(img_feats.dtype());
  TVM_FFI_ICHECK_GE(feat_code, 0) << "img_feats must use float32, float16 or bfloat16";
  TVM_FFI_ICHECK(img_feats.IsContiguous()) << "img_feats must be contiguous";
  TVM_FFI_ICHECK_EQ(img_feats.device().device_type, kDLExtDev) << "img_feats must be a MUSA tensor";

  const int depth_code = float_dtype_code(img_depth.dtype());
  TVM_FFI_ICHECK_GE(depth_code, 0) << "img_depth must use float32, float16 or bfloat16";
  TVM_FFI_ICHECK_EQ(img_depth.ndim(), 4) << "img_depth must have shape [B * N_sweep * N_cam, D, H, W]";
  TVM_FFI_ICHECK(img_depth.IsContiguous()) << "img_depth must be contiguous";
  TVM_FFI_ICHECK_EQ(img_depth.device().device_type, kDLExtDev) << "img_depth must be a MUSA tensor";

  check_musa_contiguous(out, "out", kDLFloat32);
  TVM_FFI_ICHECK_EQ(out.ndim(), 6) << "out must have shape [B, N_cam, X, Y, Z, C]";
  check_int32_contiguous(coords, "coords");
  check_int32_contiguous(point_indices, "point_indices");
  TVM_FFI_ICHECK_EQ(coords.ndim(), 2) << "coords must have shape [N_valid, 4]";
  TVM_FFI_ICHECK_EQ(dim(coords, 1), 4) << "coords must have shape [N_valid, 4]";

  const int64_t n_valid = dim(coords, 0);
  TVM_FFI_ICHECK_EQ(dim(point_indices, 0), n_valid) << "point_indices must match coords rows";

  TVM_FFI_ICHECK_GE(B, 1);
  TVM_FFI_ICHECK_GE(N_sweep, 1);
  TVM_FFI_ICHECK_GE(N_cam, 1);
  TVM_FFI_ICHECK_GE(X, 1);
  TVM_FFI_ICHECK_GE(Y, 1);
  TVM_FFI_ICHECK_GE(Z, 1);
  TVM_FFI_ICHECK_GE(D, 1);
  TVM_FFI_ICHECK_GE(H, 1);
  TVM_FFI_ICHECK_GE(W, 1);
  TVM_FFI_ICHECK_EQ(dim(img_feats, 0), B) << "img_feats batch must match B";
  TVM_FFI_ICHECK_EQ(dim(img_feats, 1), N_sweep * N_cam) << "img_feats must hold N_sweep * N_cam images";
  TVM_FFI_ICHECK_EQ(dim(img_feats, 3), H) << "img_feats height must match H";
  TVM_FFI_ICHECK_EQ(dim(img_feats, 4), W) << "img_feats width must match W";
  TVM_FFI_ICHECK_EQ(dim(img_depth, 0), B * N_sweep * N_cam) << "img_depth must hold B * N_sweep * N_cam images";
  TVM_FFI_ICHECK_EQ(dim(img_depth, 1), D) << "img_depth depth bins must match D";
  TVM_FFI_ICHECK_EQ(dim(img_depth, 2), H) << "img_depth height must match H";
  TVM_FFI_ICHECK_EQ(dim(img_depth, 3), W) << "img_depth width must match W";
  TVM_FFI_ICHECK_EQ(dim(out, 0), B) << "out batch must match B";
  TVM_FFI_ICHECK_EQ(dim(out, 1), N_cam) << "out must hold N_cam cameras";
  TVM_FFI_ICHECK_EQ(dim(out, 2), X) << "out X must match";
  TVM_FFI_ICHECK_EQ(dim(out, 3), Y) << "out Y must match";
  TVM_FFI_ICHECK_EQ(dim(out, 4), Z) << "out Z must match";
  TVM_FFI_ICHECK_EQ(dim(out, 5), dim(img_feats, 2)) << "out channels must match img_feats";

  const int64_t n_points = B * N_sweep * N_cam * D * H * W;
  TVM_FFI_ICHECK_LT(n_points, kInt32Max) << "point index space exceeds int32";
  TVM_FFI_ICHECK_LT(img_feats.numel(), kInt32Max) << "img_feats numel exceeds int32";
  TVM_FFI_ICHECK_LT(img_depth.numel(), kInt32Max) << "img_depth numel exceeds int32";
  TVM_FFI_ICHECK_LT(out.numel(), kInt32Max) << "out numel exceeds int32";

  if (n_valid == 0) {
    return;
  }

  ForwardArgs args{};
  args.N_sweep          = static_cast<int>(N_sweep);
  args.N_cam            = static_cast<int>(N_cam);
  args.X                = static_cast<int>(X);
  args.Y                = static_cast<int>(Y);
  args.Z                = static_cast<int>(Z);
  args.D                = static_cast<int>(D);
  args.H                = static_cast<int>(H);
  args.W                = static_cast<int>(W);
  args.feat_channels    = static_cast<int>(dim(img_feats, 2));
  args.n_valid          = static_cast<int>(n_valid);
  args.divmods          = make_divmods(args.W, args.H, args.D, args.N_cam, args.N_sweep, args.feat_channels);
  args.img_feats        = img_feats.data_ptr();
  args.img_depth        = img_depth.data_ptr();
  args.coords           = static_cast<const int*>(coords.data_ptr());
  args.point_indices    = static_cast<const int*>(point_indices.data_ptr());
  args.out              = static_cast<float*>(out.data_ptr());

  dispatch_forward(args, stream(img_feats), feat_code, depth_code);
  check_last_error("voxel_pool_depth_forward");
}

void voxel_pool_depth_backward(ffi::TensorView out_grad,
                               ffi::TensorView img_feats,
                               ffi::TensorView img_depth,
                               ffi::TensorView coords,
                               ffi::TensorView point_indices,
                               ffi::TensorView img_feats_grad,
                               ffi::TensorView img_depth_grad,
                               int64_t         B,
                               int64_t         N_sweep,
                               int64_t         N_cam,
                               int64_t         X,
                               int64_t         Y,
                               int64_t         Z,
                               int64_t         D,
                               int64_t         H,
                               int64_t         W) {
  check_musa_contiguous(out_grad, "out_grad", kDLFloat32);
  TVM_FFI_ICHECK_EQ(out_grad.ndim(), 6) << "out_grad must have shape [B, N_cam, X, Y, Z, C]";

  const int feat_code = float_dtype_code(img_feats.dtype());
  TVM_FFI_ICHECK_GE(feat_code, 0) << "img_feats must use float32, float16 or bfloat16";
  TVM_FFI_ICHECK_EQ(img_feats.ndim(), 5) << "img_feats must have shape [B, N_sweep * N_cam, C, H, W]";
  TVM_FFI_ICHECK(img_feats.IsContiguous()) << "img_feats must be contiguous";
  const int depth_code = float_dtype_code(img_depth.dtype());
  TVM_FFI_ICHECK_GE(depth_code, 0) << "img_depth must use float32, float16 or bfloat16";
  TVM_FFI_ICHECK_EQ(img_depth.ndim(), 4) << "img_depth must have shape [B * N_sweep * N_cam, D, H, W]";
  TVM_FFI_ICHECK(img_depth.IsContiguous()) << "img_depth must be contiguous";

  check_musa_contiguous(img_feats_grad, "img_feats_grad", kDLFloat32);
  check_musa_contiguous(img_depth_grad, "img_depth_grad", kDLFloat32);
  check_int32_contiguous(coords, "coords");
  check_int32_contiguous(point_indices, "point_indices");
  TVM_FFI_ICHECK_EQ(coords.ndim(), 2) << "coords must have shape [N_valid, 4]";
  TVM_FFI_ICHECK_EQ(dim(coords, 1), 4) << "coords must have shape [N_valid, 4]";
  const int64_t n_valid = dim(coords, 0);
  TVM_FFI_ICHECK_EQ(dim(point_indices, 0), n_valid) << "point_indices must match coords rows";
  TVM_FFI_ICHECK_EQ(img_feats_grad.numel(), img_feats.numel()) << "img_feats_grad shape mismatch";
  TVM_FFI_ICHECK_EQ(img_depth_grad.numel(), img_depth.numel()) << "img_depth_grad shape mismatch";
  TVM_FFI_ICHECK_EQ(dim(out_grad, 0), B) << "out_grad batch must match B";
  TVM_FFI_ICHECK_EQ(dim(out_grad, 1), N_cam) << "out_grad must hold N_cam cameras";
  TVM_FFI_ICHECK_EQ(dim(out_grad, 5), dim(img_feats, 2)) << "out_grad channels must match img_feats";

  const int64_t n_points = B * N_sweep * N_cam * D * H * W;
  TVM_FFI_ICHECK_LT(n_points, kInt32Max) << "point index space exceeds int32";
  TVM_FFI_ICHECK_LT(out_grad.numel(), kInt32Max) << "out_grad numel exceeds int32";

  if (n_valid == 0 || dim(img_feats, 2) == 0) {
    return;
  }

  BackwardArgs args{};
  args.N_sweep        = static_cast<int>(N_sweep);
  args.N_cam          = static_cast<int>(N_cam);
  args.X              = static_cast<int>(X);
  args.Y              = static_cast<int>(Y);
  args.Z              = static_cast<int>(Z);
  args.D              = static_cast<int>(D);
  args.H              = static_cast<int>(H);
  args.W              = static_cast<int>(W);
  args.feat_channels  = static_cast<int>(dim(img_feats, 2));
  args.n_valid        = static_cast<int>(n_valid);
  args.divmods        = make_divmods(args.W, args.H, args.D, args.N_cam, args.N_sweep, args.feat_channels);
  args.out_grad       = out_grad.data_ptr();
  args.img_feats      = img_feats.data_ptr();
  args.img_depth      = img_depth.data_ptr();
  args.coords         = static_cast<const int*>(coords.data_ptr());
  args.point_indices  = static_cast<const int*>(point_indices.data_ptr());
  args.img_feats_grad = static_cast<float*>(img_feats_grad.data_ptr());
  args.img_depth_grad = static_cast<float*>(img_depth_grad.data_ptr());

  dispatch_backward(args, stream(out_grad), feat_code, depth_code);
  check_last_error("voxel_pool_depth_backward");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(voxel_pool_depth_forward, voxel_pool_depth_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(voxel_pool_depth_backward, voxel_pool_depth_backward);
