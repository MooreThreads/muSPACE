// BEV temporal gather/scatter (temporal_align) kernels.
//
// The forward pass resamples a per-sample BEV feature plane around a per-sample
// centre; the backward pass scatters output gradients back onto the source
// plane. The geometry follows torchvision.transforms.functional.rotate: angles
// are degrees, the rotation is applied in pixel space and positive angles match
// the torchvision direction, `center` is a corner-based pixel coordinate, and
// `shift` is a source-pixel translation applied after the rotation.
//
// Tensors use the channels-last BEV layout `(batch, bev_h*bev_w, channels)`; the
// Python wrapper converts the channels-first form before calling these entries.
// The `*_planar` entries take the channels-first form `(batch, channels, bev_h,
// bev_w)` directly, which is the layout the model pipeline holds, so a float32
// caller pays no layout conversion at all.
//
// Three shapes of kernel exist. The float32 tiled fast path packs the plane over
// `(x tile, row, sample)`, so a block never divides indices and every thread
// issues 128-bit accesses on consecutive channels. The planar fast path keeps the
// channels-first layout and therefore vectorises along x instead of along
// channels. The generic path gives each block one output location and lets thread
// 0 evaluate that location's geometry into shared memory, which keeps the 64-bit
// plane arithmetic and the trigonometry off the per-element path for the
// low-precision and odd-channel cases.

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "../ffi/dtype_utils.hpp"
#include "../ffi/tensor_utils.hpp"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::check_last_error;
using muspace::ffi_utils::check_musa_contiguous;
using muspace::ffi_utils::check_same_musa_device;
using muspace::ffi_utils::stream;

namespace {

constexpr double kPi            = 3.14159265358979323846;
constexpr int   kMaxThreads     = 256;
constexpr int   kMaxGroupWidth  = 128;
constexpr int   kWarpSize       = 32;
constexpr int   kMaxTaps        = 4;
constexpr int   kVector         = 4;
constexpr int   kXTile          = 4;
constexpr int   kPlanarWidth    = 32;
constexpr int   kPlanarChannels = 8;
// The backward gather tests every output pixel that could reach a source pixel:
// the integers within one and a half pixels of the inverse position.
constexpr int     kGatherSpan       = 4;
constexpr int     kGatherCandidates = kGatherSpan * kGatherSpan;
constexpr int64_t kGridLimit        = 65535;
constexpr int64_t kShiftInteger     = 1;
constexpr int64_t kShiftFloat       = 2;
constexpr int64_t kBilinear         = 0;
constexpr int64_t kNearest          = 1;

constexpr DLDataType kDLFloat16 = DLDataType{kDLFloat, 16, 1};

template <typename T>
__device__ __forceinline__ float as_float(T value) {
  return static_cast<float>(value);
}
template <>
__device__ __forceinline__ float as_float(__half value) {
  return __half2float(value);
}
template <>
__device__ __forceinline__ float as_float(__mt_bfloat16 value) {
  return __bfloat162float(value);
}
template <typename T>
__device__ __forceinline__ T from_float(float value) {
  return static_cast<T>(value);
}
template <>
__device__ __forceinline__ __half from_float(float value) {
  return __float2half(value);
}
template <>
__device__ __forceinline__ __mt_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}

// Trigonometry for one sample. Quarter turns snap to their exact values: float32
// argument reduction moves cos(90 deg) by about 4e-8, which is enough to send a
// nearest-neighbour sample sitting on an exact half pixel to the other side.
//
// Every other angle is evaluated in double and only then rounded to float32,
// because that is what the reference does: it takes cos/sin from Python's
// double-precision libm and lets the float32 tensor op cast the scalar.  A
// float32 sin/cos can be one ulp away, and one ulp of sin times a 183-pixel
// offset is enough to flip a nearest-neighbour tie, so the extra precision is
// what makes the tap choice match ``F.grid_sample`` bit for bit.
__device__ __forceinline__ void sample_trig(float degrees, float* sine, float* cosine) {
  const float quarter = nearbyintf(degrees * (1.0F / 90.0F));
  if (fabsf(quarter) < 1.0e6F && fabsf(degrees - quarter * 90.0F) <= 1.0e-4F) {
    const int turn = ((static_cast<int>(quarter) % 4) + 4) % 4;
    *cosine        = (turn == 0) ? 1.0F : ((turn == 2) ? -1.0F : 0.0F);
    *sine          = (turn == 1) ? 1.0F : ((turn == 3) ? -1.0F : 0.0F);
    return;
  }
  const double radians = static_cast<double>(degrees) * (kPi / 180.0);
  double       s       = 0.0;
  double       c       = 0.0;
  sincos(radians, &s, &c);
  *sine   = static_cast<float>(s);
  *cosine = static_cast<float>(c);
}

// Source position of an output pixel, in source pixel-index space.
__device__ __forceinline__ void source_position(int    x,
                                                int    y,
                                                float  centre_x,
                                                float  centre_y,
                                                float  cosine,
                                                float  sine,
                                                float  shift_x,
                                                float  shift_y,
                                                float* sample_x,
                                                float* sample_y) {
  const float delta_x = static_cast<float>(x) - centre_x;
  const float delta_y = static_cast<float>(y) - centre_y;
  *sample_x           = cosine * delta_x - sine * delta_y + centre_x + shift_x;
  *sample_y           = sine * delta_x + cosine * delta_y + centre_y + shift_y;
}

// The source position expressed the way ``F.grid_sample`` sees it.
//
// ``temporal_align_reference`` feeds a normalised grid to ``F.grid_sample``, which
// unnormalises it again with ``((coord + 1) * size - 1) / 2`` for
// ``align_corners=False``.  Both roundings are float32, and on an exact half
// pixel they are what decides a nearest-neighbour tie: the round trip can land
// just below the half, where ``nearbyintf`` then rounds down while the exact
// coordinate rounds to even.  Reproducing the round trip keeps the kernel bit
// identical to the reference at those ties.
__device__ __forceinline__ float pytorch_source_index(float sample, int size) {
  const float extent = static_cast<float>(size);
  const float grid   = (2.0F * sample + 1.0F - extent) / extent;
  return ((grid + 1.0F) * extent - 1.0F) * 0.5F;
}

// Collects the in-range source taps of one output pixel and returns the tap
// count. A zero count means the pixel samples padding and yields zeros. Offsets
// are element offsets into one channel plane, scaled by `stride`: a packed
// channels-last plane passes its channel count so a tap is reached with
// `base + offset + channel`, a planar channels-first plane passes one so a tap
// is reached with `plane + offset`.
__device__ __forceinline__ int collect_taps(float    sample_x,
                                            float    sample_y,
                                            int      width,
                                            int      height,
                                            int64_t  stride,
                                            bool     nearest,
                                            int64_t* offsets,
                                            float*   weights) {
  if (nearest) {
    // The tie-break has to follow the reference, so the index is chosen from the
    // grid-normalised position rather than from the exact source coordinate.
    const float normalised_x = pytorch_source_index(sample_x, width);
    const float normalised_y = pytorch_source_index(sample_y, height);
    // A non-finite rotation reaches this point instead of raising: the native path
    // does not scan the caller's angle (that costs a device synchronization per
    // call).  It lands on no tap rather than rounding an undefined value.
    if (!(normalised_x == normalised_x) || !(normalised_y == normalised_y)) return 0;
    const int index_x = static_cast<int>(nearbyintf(normalised_x));
    const int index_y = static_cast<int>(nearbyintf(normalised_y));
    if (index_x < 0 || index_x >= width || index_y < 0 || index_y >= height) return 0;
    offsets[0] = (static_cast<int64_t>(index_y) * width + index_x) * stride;
    weights[0] = 1.0F;
    return 1;
  }
  if (!(sample_x > -1.0F && sample_x < static_cast<float>(width) && sample_y > -1.0F &&
        sample_y < static_cast<float>(height))) {
    return 0;
  }
  const int   lower_x = static_cast<int>(floorf(sample_x));
  const int   lower_y = static_cast<int>(floorf(sample_y));
  const float frac_x  = sample_x - static_cast<float>(lower_x);
  const float frac_y  = sample_y - static_cast<float>(lower_y);
  int         taps    = 0;
#pragma unroll
  for (int corner = 0; corner < 4; ++corner) {
    const int step_x  = corner & 1;
    const int step_y  = corner >> 1;
    const int index_x = lower_x + step_x;
    const int index_y = lower_y + step_y;
    if (index_x < 0 || index_x >= width || index_y < 0 || index_y >= height) continue;
    offsets[taps] = (static_cast<int64_t>(index_y) * width + index_x) * stride;
    weights[taps] = (step_x ? frac_x : 1.0F - frac_x) * (step_y ? frac_y : 1.0F - frac_y);
    ++taps;
  }
  return taps;
}

// Sample geometry of one batch entry.
//
// ``trig`` holds ``(cosine, sine)`` per sample, prepared once per call by
// ``prepare_trig_kernel``.  Keeping the double-precision trigonometry out of the
// sampling kernels matters: inlining it cost about 3x on the forward pass and 1.5x
// on the backward pass, because the extra 64-bit registers reduced occupancy of
// these bandwidth-bound kernels even on runs with no rotation at all.
struct SampleGeometry {
  float cosine;
  float sine;
  float shift_x;
  float shift_y;
};

__device__ __forceinline__ SampleGeometry sample_geometry(const float* trig,
                                                          const void*  shift,
                                                          int64_t      shift_kind,
                                                          int          batch) {
  SampleGeometry geometry{trig[2 * batch], trig[2 * batch + 1], 0.0F, 0.0F};
  if (shift_kind == kShiftFloat) {
    const float* values = static_cast<const float*>(shift);
    geometry.shift_x    = values[2 * batch];
    geometry.shift_y    = values[2 * batch + 1];
  }
  return geometry;
}

// Fills the per-sample ``(cosine, sine)`` table once per call.
__global__ void prepare_trig_kernel(const float* __restrict__ angle,
                                    int                            batch,
                                    float* __restrict__            trig) {
  const int sample = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                     static_cast<int>(threadIdx.x);
  if (sample >= batch) return;
  float sine   = 0.0F;
  float cosine = 1.0F;
  if (angle != nullptr) sample_trig(angle[sample], &sine, &cosine);
  trig[2 * sample]     = cosine;
  trig[2 * sample + 1] = sine;
}

// --------------------------------------------------------------------------------------
// generic path: one block per output location
// --------------------------------------------------------------------------------------

// Decodes the output location owned by a block into plane coordinates.
__device__ __forceinline__ void decode_location(
    int64_t location, int64_t spatial, int width, int* batch, int* x, int* y) {
  const int64_t sample_batch = location / spatial;
  const int64_t plane        = location - sample_batch * spatial;
  *batch                     = static_cast<int>(sample_batch);
  *x                         = static_cast<int>(plane % width);
  *y                         = static_cast<int>(plane / width);
}

template <typename T, bool Nearest>
__global__ void affine_forward_kernel(const T* __restrict__ bev,
                                      const float* __restrict__ angle,
                                      const void* __restrict__ shift,
                                      int64_t shift_kind,
                                      T* __restrict__ out,
                                      int64_t channels,
                                      int64_t spatial,
                                      int     width,
                                      int     height,
                                      float   centre_x,
                                      float   centre_y) {
  __shared__ int64_t offsets[kMaxTaps];
  __shared__ float   weights[kMaxTaps];
  __shared__ int     taps;
  __shared__ int64_t base;
  __shared__ int64_t out_base;
  if (threadIdx.x == 0) {
    int batch;
    int x;
    int y;
    decode_location(blockIdx.x, spatial, width, &batch, &x, &y);
    base                          = static_cast<int64_t>(batch) * spatial * channels;
    out_base                      = blockIdx.x * channels;
    const SampleGeometry geometry = sample_geometry(angle, shift, shift_kind, batch);
    float                sample_x;
    float                sample_y;
    source_position(x,
                    y,
                    centre_x,
                    centre_y,
                    geometry.cosine,
                    geometry.sine,
                    geometry.shift_x,
                    geometry.shift_y,
                    &sample_x,
                    &sample_y);
    taps = collect_taps(sample_x, sample_y, width, height, channels, Nearest, offsets, weights);
  }
  __syncthreads();

  const int     count  = taps;
  const int     total  = static_cast<int>(channels);
  const int64_t source = base;
  const int64_t target = out_base;
  for (int channel = threadIdx.x; channel < total; channel += blockDim.x) {
    float result = 0.0F;
    for (int tap = 0; tap < count; ++tap) {
      result += as_float(bev[source + offsets[tap] + channel]) * weights[tap];
    }
    out[target + channel] = from_float<T>(result);
  }
}

// Backward scatters every output gradient into the source plane. The
// accumulation is atomic and therefore not bit-reproducible.
template <typename T, bool Nearest>
__global__ void affine_backward_kernel(const T* __restrict__ grad_out,
                                       const float* __restrict__ angle,
                                       const void* __restrict__ shift,
                                       int64_t shift_kind,
                                       float* __restrict__ grad_bev,
                                       int64_t channels,
                                       int64_t spatial,
                                       int     width,
                                       int     height,
                                       float   centre_x,
                                       float   centre_y) {
  __shared__ int64_t offsets[kMaxTaps];
  __shared__ float   weights[kMaxTaps];
  __shared__ int     taps;
  __shared__ int64_t base;
  __shared__ int64_t out_base;
  if (threadIdx.x == 0) {
    int batch;
    int x;
    int y;
    decode_location(blockIdx.x, spatial, width, &batch, &x, &y);
    base                          = static_cast<int64_t>(batch) * spatial * channels;
    out_base                      = blockIdx.x * channels;
    const SampleGeometry geometry = sample_geometry(angle, shift, shift_kind, batch);
    float                sample_x;
    float                sample_y;
    source_position(x,
                    y,
                    centre_x,
                    centre_y,
                    geometry.cosine,
                    geometry.sine,
                    geometry.shift_x,
                    geometry.shift_y,
                    &sample_x,
                    &sample_y);
    taps = collect_taps(sample_x, sample_y, width, height, channels, Nearest, offsets, weights);
  }
  __syncthreads();
  if (taps == 0) return;

  const int     count  = taps;
  const int     total  = static_cast<int>(channels);
  const int64_t source = base;
  const int64_t target = out_base;
  for (int channel = threadIdx.x; channel < total; channel += blockDim.x) {
    const float gradient = as_float(grad_out[target + channel]);
    for (int tap = 0; tap < count; ++tap) {
      atomicAdd(grad_bev + source + offsets[tap] + channel, gradient * weights[tap]);
    }
  }
}

// Integer translation path: a pure plane shift, no interpolation and no atomic
// accumulation because the mapping is one-to-one.
template <typename T>
__global__ void shift_forward_kernel(const T* __restrict__ bev,
                                     const int32_t* __restrict__ shift,
                                     T* __restrict__ out,
                                     int64_t channels,
                                     int64_t spatial,
                                     int     width,
                                     int     height) {
  __shared__ int64_t offsets[kMaxTaps];
  __shared__ int     taps;
  __shared__ int64_t base;
  __shared__ int64_t out_base;
  if (threadIdx.x == 0) {
    int batch;
    int x;
    int y;
    decode_location(blockIdx.x, spatial, width, &batch, &x, &y);
    base                   = static_cast<int64_t>(batch) * spatial * channels;
    out_base               = blockIdx.x * channels;
    const int64_t source_x = static_cast<int64_t>(x) + shift[2 * batch];
    const int64_t source_y = static_cast<int64_t>(y) + shift[2 * batch + 1];
    if (source_x < 0 || source_x >= width || source_y < 0 || source_y >= height) {
      taps       = 0;
      offsets[0] = 0;
    } else {
      offsets[0] = (source_y * width + source_x) * channels;
      taps       = 1;
    }
  }
  __syncthreads();

  const int     total    = static_cast<int>(channels);
  const int64_t source   = base + offsets[0];
  const int64_t target   = out_base;
  const bool    in_plane = taps != 0;
  for (int channel = threadIdx.x; channel < total; channel += blockDim.x) {
    out[target + channel] = in_plane ? bev[source + channel] : from_float<T>(0.0F);
  }
}

template <typename T>
__global__ void shift_backward_kernel(const T* __restrict__ grad_out,
                                      const int32_t* __restrict__ shift,
                                      float* __restrict__ grad_bev,
                                      int64_t channels,
                                      int64_t spatial,
                                      int     width,
                                      int     height) {
  __shared__ int64_t offsets[kMaxTaps];
  __shared__ int     taps;
  __shared__ int64_t base;
  __shared__ int64_t out_base;
  if (threadIdx.x == 0) {
    int batch;
    int x;
    int y;
    decode_location(blockIdx.x, spatial, width, &batch, &x, &y);
    base                   = static_cast<int64_t>(batch) * spatial * channels;
    out_base               = blockIdx.x * channels;
    const int64_t source_x = static_cast<int64_t>(x) + shift[2 * batch];
    const int64_t source_y = static_cast<int64_t>(y) + shift[2 * batch + 1];
    if (source_x < 0 || source_x >= width || source_y < 0 || source_y >= height) {
      taps       = 0;
      offsets[0] = 0;
    } else {
      offsets[0] = (source_y * width + source_x) * channels;
      taps       = 1;
    }
  }
  __syncthreads();
  if (taps == 0) return;

  const int     total  = static_cast<int>(channels);
  const int64_t source = base + offsets[0];
  const int64_t target = out_base;
  for (int channel = threadIdx.x; channel < total; channel += blockDim.x) {
    grad_bev[source + channel] = as_float(grad_out[target + channel]);
  }
}

// --------------------------------------------------------------------------------------
// float32 fast path: one block covers kXTile output pixels and their channels, four
// channels per thread through 128-bit accesses. Indices come from the grid, so a block
// performs no division; the shared-memory plan is filled once per pixel.
// --------------------------------------------------------------------------------------

struct TilePlan {
  int64_t offsets[kXTile][kMaxTaps];
  float   weights[kXTile][kMaxTaps];
  int     taps[kXTile];
};

// Fills the per-pixel geometry of the tile. Only the first thread of each pixel row
// runs it; the caller synchronises afterwards.
__device__ __forceinline__ void plan_tile(TilePlan* __restrict__ plan,
                                          int     row,
                                          int     x,
                                          int     y,
                                          bool    active,
                                          float   cosine,
                                          float   sine,
                                          float   shift_x,
                                          float   shift_y,
                                          int     width,
                                          int     height,
                                          int64_t channels,
                                          float   centre_x,
                                          float   centre_y,
                                          bool    nearest) {
  if (threadIdx.x != 0) return;
  if (!active) {
    plan->taps[row] = 0;
    return;
  }
  float sample_x;
  float sample_y;
  source_position(x, y, centre_x, centre_y, cosine, sine, shift_x, shift_y, &sample_x, &sample_y);
  const int taps =
      collect_taps(sample_x, sample_y, width, height, channels, nearest, plan->offsets[row], plan->weights[row]);
  plan->taps[row] = taps;
  for (int tap = 0; tap < taps; ++tap) plan->offsets[row][tap] /= kVector;
}

__device__ __forceinline__ float4 accumulate(float4 accumulator, const float4 value, const float weight) {
  accumulator.x += value.x * weight;
  accumulator.y += value.y * weight;
  accumulator.z += value.z * weight;
  accumulator.w += value.w * weight;
  return accumulator;
}

// --------------------------------------------------------------------------------------
// planar (channels-first) kernels
// --------------------------------------------------------------------------------------

// A channels-first plane is contiguous along x, so x is the fast axis: one
// thread owns one output pixel and a warp owns 32 consecutive pixels of one
// channel, which turns the tap gather and the gradient scatter into coalesced
// accesses. The per-pixel tap plan is shared by every channel of the block.
// This keeps the caller's layout, which removes the two full-plane layout
// conversions a packed plane would need.
struct PlanarPlan {
  int64_t offsets[kPlanarWidth][kMaxTaps];
  float   weights[kPlanarWidth][kMaxTaps];
  int     taps[kPlanarWidth];
};

__device__ __forceinline__ void plan_planar(PlanarPlan* __restrict__ plan,
                                            int   slot,
                                            int   x,
                                            int   y,
                                            bool  active,
                                            float cosine,
                                            float sine,
                                            float shift_x,
                                            float shift_y,
                                            int   width,
                                            int   height,
                                            float centre_x,
                                            float centre_y,
                                            bool  nearest) {
  if (slot >= kPlanarWidth) return;
  // A pixel past the right edge has no output, so it must carry no tap: the
  // backward pass scatters whatever the plan holds.
  if (!active) {
    plan->taps[slot] = 0;
    return;
  }
  float sample_x;
  float sample_y;
  source_position(x, y, centre_x, centre_y, cosine, sine, shift_x, shift_y, &sample_x, &sample_y);
  plan->taps[slot] =
      collect_taps(sample_x, sample_y, width, height, 1, nearest, plan->offsets[slot], plan->weights[slot]);
}

// Gathers the taps of one output pixel for every channel the thread owns.
template <bool Nearest>
__global__ void planar_forward_tiled_kernel(const float* __restrict__ bev,
                                            const float* __restrict__ angle,
                                            const void* __restrict__ shift,
                                            int64_t shift_kind,
                                            float* __restrict__ out,
                                            int64_t channels,
                                            int64_t spatial,
                                            int     width,
                                            int     height,
                                            float   centre_x,
                                            float   centre_y) {
  __shared__ PlanarPlan plan;
  const int             slot   = threadIdx.x;
  const int             batch  = blockIdx.z;
  const int             row    = blockIdx.y;
  const int             x      = static_cast<int>(blockIdx.x) * kPlanarWidth + slot;
  const bool            active = x < width;
  if (threadIdx.y == 0) {
    const SampleGeometry geometry = sample_geometry(angle, shift, shift_kind, batch);
    plan_planar(&plan,
                slot,
                x,
                row,
                active,
                geometry.cosine,
                geometry.sine,
                geometry.shift_x,
                geometry.shift_y,
                width,
                height,
                centre_x,
                centre_y,
                Nearest);
  }
  __syncthreads();
  if (!active) return;

  const int64_t batch0 = static_cast<int64_t>(batch) * channels * spatial;
  const int64_t index  = static_cast<int64_t>(row) * width + x;
  const int     taps   = plan.taps[slot];
  for (int channel = static_cast<int>(threadIdx.y); channel < static_cast<int>(channels);
       channel += static_cast<int>(blockDim.y)) {
    const float* source = bev + batch0 + static_cast<int64_t>(channel) * spatial;
    float        result = 0.0F;
    for (int tap = 0; tap < taps; ++tap) {
      result += source[plan.offsets[slot][tap]] * plan.weights[slot][tap];
    }
    out[batch0 + static_cast<int64_t>(channel) * spatial + index] = result;
  }
}

// Scatters the output gradient of every channel back onto the source plane.
//
// The adjoint is taken as a gather: every source pixel asks which output pixels
// sampled it. The sampling map is an affine rotation, so the inverse map sends
// the source pixel back to a continuous output position, and the output pixels
// that can reach it are the integers around that position. One output pixel
// touches a source pixel only when the latter is a corner of its footprint, so
// at most four of them qualify. The gradient is written once per source pixel in
// a coalesced store: no atomics, no accumulation order, and the buffer does not
// have to be cleared first.
template <bool Nearest>
__global__ void planar_backward_gather_kernel(const float* __restrict__ grad_out,
                                              const float* __restrict__ angle,
                                              const void* __restrict__ shift,
                                              int64_t shift_kind,
                                              float* __restrict__ grad_bev,
                                              int64_t channels,
                                              int64_t spatial,
                                              int     width,
                                              int     height,
                                              float   centre_x,
                                              float   centre_y) {
  const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (x >= width) return;
  const int            y        = static_cast<int>(blockIdx.y);
  const int            batch    = static_cast<int>(blockIdx.z);
  const SampleGeometry geometry = sample_geometry(angle, shift, shift_kind, batch);
  // Inverse of source_position: the output position whose sample lands on (x, y).
  const float relative_x   = static_cast<float>(x) - centre_x - geometry.shift_x;
  const float relative_y   = static_cast<float>(y) - centre_y - geometry.shift_y;
  const float centre_out_x = centre_x + geometry.cosine * relative_x + geometry.sine * relative_y;
  const float centre_out_y = centre_y - geometry.sine * relative_x + geometry.cosine * relative_y;
  const int   base_x       = static_cast<int>(floorf(centre_out_x));
  const int   base_y       = static_cast<int>(floorf(centre_out_y));
  // One weight per candidate, zero when the candidate does not reach this pixel.
  // The grid is indexed by compile-time slots, so it stays in registers; a list of
  // (index, weight) pairs would be indexed at run time and spill to local memory.
  // Zero is safe as an "absent" marker: a real corner only has weight zero when one
  // of its fractions is exactly zero, and then it contributes exactly zero anyway.
  float weight[kGatherCandidates];
#pragma unroll
  for (int slot = 0; slot < kGatherCandidates; ++slot) weight[slot] = 0.0F;
#pragma unroll
  for (int step_y = -1; step_y < kGatherSpan - 1; ++step_y) {
#pragma unroll
    for (int step_x = -1; step_x < kGatherSpan - 1; ++step_x) {
      const int slot  = (step_y + 1) * kGatherSpan + step_x + 1;
      const int out_x = base_x + step_x;
      const int out_y = base_y + step_y;
      if (out_x < 0 || out_x >= width || out_y < 0 || out_y >= height) continue;
      float sample_x;
      float sample_y;
      source_position(out_x,
                      out_y,
                      centre_x,
                      centre_y,
                      geometry.cosine,
                      geometry.sine,
                      geometry.shift_x,
                      geometry.shift_y,
                      &sample_x,
                      &sample_y);
      if (Nearest) {
        const float normalised_x = pytorch_source_index(sample_x, width);
        const float normalised_y = pytorch_source_index(sample_y, height);
        if (!(normalised_x == normalised_x) || !(normalised_y == normalised_y)) continue;
        if (static_cast<int>(nearbyintf(normalised_x)) == x &&
            static_cast<int>(nearbyintf(normalised_y)) == y) {
          weight[slot] = 1.0F;
        }
        continue;
      }
      // The same padding rule as the forward pass: a sample outside the plane
      // contributes nothing at all.
      if (!(sample_x > -1.0F && sample_x < static_cast<float>(width) && sample_y > -1.0F &&
            sample_y < static_cast<float>(height))) {
        continue;
      }
      const int floor_x  = static_cast<int>(floorf(sample_x));
      const int floor_y  = static_cast<int>(floorf(sample_y));
      const int corner_x = x - floor_x;
      const int corner_y = y - floor_y;
      if (corner_x < 0 || corner_x > 1 || corner_y < 0 || corner_y > 1) continue;
      const float frac_x = sample_x - static_cast<float>(floor_x);
      const float frac_y = sample_y - static_cast<float>(floor_y);
      weight[slot]       = (corner_x ? frac_x : 1.0F - frac_x) * (corner_y ? frac_y : 1.0F - frac_y);
    }
  }

  const int64_t batch0 = static_cast<int64_t>(batch) * channels * spatial;
  const int64_t index  = static_cast<int64_t>(y) * width + x;
  // Two channels per iteration: the candidate test is the same for both, so half
  // of it is saved and the two loads overlap.
  const int stride  = static_cast<int>(blockDim.y) * 2;
  int       channel = static_cast<int>(threadIdx.y) * 2;
  for (; channel + 1 < static_cast<int>(channels); channel += stride) {
    const int64_t offset_near = batch0 + static_cast<int64_t>(channel) * spatial;
    const int64_t offset_far  = offset_near + spatial;
    float         sum_near    = 0.0F;
    float         sum_far     = 0.0F;
#pragma unroll
    for (int slot = 0; slot < kGatherCandidates; ++slot) {
      if (weight[slot] != 0.0F) {
        const int source =
            static_cast<int64_t>(base_y + slot / kGatherSpan - 1) * width + base_x + slot % kGatherSpan - 1;
        sum_near += weight[slot] * grad_out[offset_near + source];
        sum_far += weight[slot] * grad_out[offset_far + source];
      }
    }
    grad_bev[offset_near + index] = sum_near;
    grad_bev[offset_far + index]  = sum_far;
  }
  if (channel < static_cast<int>(channels)) {
    const int64_t offset = batch0 + static_cast<int64_t>(channel) * spatial;
    float         sum    = 0.0F;
#pragma unroll
    for (int slot = 0; slot < kGatherCandidates; ++slot) {
      if (weight[slot] != 0.0F) {
        const int source =
            static_cast<int64_t>(base_y + slot / kGatherSpan - 1) * width + base_x + slot % kGatherSpan - 1;
        sum += weight[slot] * grad_out[offset + source];
      }
    }
    grad_bev[offset + index] = sum;
  }
}

template <bool Nearest>
__global__ void affine_forward_tiled_kernel(const float* __restrict__ bev,
                                            const float* __restrict__ angle,
                                            const void* __restrict__ shift,
                                            int64_t shift_kind,
                                            float* __restrict__ out,
                                            int64_t channels,
                                            int64_t spatial,
                                            int     width,
                                            int     height,
                                            float   centre_x,
                                            float   centre_y) {
  __shared__ TilePlan plan;
  const int           row    = threadIdx.y;
  const int           batch  = blockIdx.z;
  const int           x      = static_cast<int>(blockIdx.x) * kXTile + row;
  const int           y      = blockIdx.y;
  const bool          active = x < width;
  if (threadIdx.x == 0) {
    const SampleGeometry geometry = sample_geometry(angle, shift, shift_kind, batch);
    plan_tile(&plan,
              row,
              x,
              y,
              active,
              geometry.cosine,
              geometry.sine,
              geometry.shift_x,
              geometry.shift_y,
              width,
              height,
              channels,
              centre_x,
              centre_y,
              Nearest);
  }
  __syncthreads();
  if (!active) return;

  const int count  = plan.taps[row];
  const int groups = static_cast<int>(channels) / kVector;
  // The tap offsets are relative to the sample's plane, the output index to the
  // whole tensor, so the two bases are kept apart.
  const int64_t plane  = static_cast<int64_t>(batch) * spatial * groups;
  const int64_t base   = (static_cast<int64_t>(batch) * spatial + static_cast<int64_t>(y) * width + x) * groups;
  const float4* source = reinterpret_cast<const float4*>(bev) + plane;
  float4*       target = reinterpret_cast<float4*>(out) + base;
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    float4 accumulator = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (count == kMaxTaps) {
      accumulator = accumulate(accumulator, source[plan.offsets[row][0] + group], plan.weights[row][0]);
      accumulator = accumulate(accumulator, source[plan.offsets[row][1] + group], plan.weights[row][1]);
      accumulator = accumulate(accumulator, source[plan.offsets[row][2] + group], plan.weights[row][2]);
      accumulator = accumulate(accumulator, source[plan.offsets[row][3] + group], plan.weights[row][3]);
    } else {
      for (int tap = 0; tap < count; ++tap) {
        accumulator = accumulate(accumulator, source[plan.offsets[row][tap] + group], plan.weights[row][tap]);
      }
    }
    target[group] = accumulator;
  }
}

template <bool Nearest>
__global__ void affine_backward_tiled_kernel(const float* __restrict__ grad_out,
                                             const float* __restrict__ angle,
                                             const void* __restrict__ shift,
                                             int64_t shift_kind,
                                             float* __restrict__ grad_bev,
                                             int64_t channels,
                                             int64_t spatial,
                                             int     width,
                                             int     height,
                                             float   centre_x,
                                             float   centre_y) {
  __shared__ TilePlan plan;
  const int           row    = threadIdx.y;
  const int           batch  = blockIdx.z;
  const int           x      = static_cast<int>(blockIdx.x) * kXTile + row;
  const int           y      = blockIdx.y;
  const bool          active = x < width;
  if (threadIdx.x == 0) {
    const SampleGeometry geometry = sample_geometry(angle, shift, shift_kind, batch);
    plan_tile(&plan,
              row,
              x,
              y,
              active,
              geometry.cosine,
              geometry.sine,
              geometry.shift_x,
              geometry.shift_y,
              width,
              height,
              channels,
              centre_x,
              centre_y,
              Nearest);
  }
  __syncthreads();
  if (!active) return;

  const int count = plan.taps[row];
  if (count == 0) return;
  const int     groups = static_cast<int>(channels) / kVector;
  const int64_t plane  = static_cast<int64_t>(batch) * spatial * groups;
  const int64_t base   = (static_cast<int64_t>(batch) * spatial + static_cast<int64_t>(y) * width + x) * groups;
  const float4* source = reinterpret_cast<const float4*>(grad_out) + base;
  float4*       target = reinterpret_cast<float4*>(grad_bev) + plane;
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    const float4 gradient = source[group];
    for (int tap = 0; tap < count; ++tap) {
      const float weight = plan.weights[row][tap];
      const int   offset = static_cast<int>(plan.offsets[row][tap]) + group;
      atomicAdd(&target[offset].x, gradient.x * weight);
      atomicAdd(&target[offset].y, gradient.y * weight);
      atomicAdd(&target[offset].z, gradient.z * weight);
      atomicAdd(&target[offset].w, gradient.w * weight);
    }
  }
}

__global__ void shift_forward_tiled_kernel(const float* __restrict__ bev,
                                           const int32_t* __restrict__ shift,
                                           float* __restrict__ out,
                                           int64_t channels,
                                           int64_t spatial,
                                           int     width,
                                           int     height) {
  __shared__ int64_t offsets[kXTile];
  __shared__ int     taps[kXTile];
  const int          row   = threadIdx.y;
  const int          batch = blockIdx.z;
  const int          x     = static_cast<int>(blockIdx.x) * kXTile + row;
  const int          y     = blockIdx.y;
  if (threadIdx.x == 0) {
    const int64_t source_x = static_cast<int64_t>(x) + shift[2 * batch];
    const int64_t source_y = static_cast<int64_t>(y) + shift[2 * batch + 1];
    if (x >= width || source_x < 0 || source_x >= width || source_y < 0 || source_y >= height) {
      taps[row]    = 0;
      offsets[row] = 0;
    } else {
      offsets[row] = ((source_y * width + source_x) * channels) / kVector;
      taps[row]    = 1;
    }
  }
  __syncthreads();
  if (x >= width) return;

  const int     groups = static_cast<int>(channels) / kVector;
  const int64_t plane  = static_cast<int64_t>(batch) * spatial * groups;
  const int64_t base =
      ((static_cast<int64_t>(batch) * spatial + static_cast<int64_t>(y) * width + x) * channels) / kVector;
  const float4* source   = reinterpret_cast<const float4*>(bev) + plane + offsets[row];
  float4*       target   = reinterpret_cast<float4*>(out) + base;
  const bool    in_plane = taps[row] != 0;
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    target[group] = in_plane ? source[group] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  }
}

__global__ void shift_backward_tiled_kernel(const float* __restrict__ grad_out,
                                            const int32_t* __restrict__ shift,
                                            float* __restrict__ grad_bev,
                                            int64_t channels,
                                            int64_t spatial,
                                            int     width,
                                            int     height) {
  __shared__ int64_t offsets[kXTile];
  __shared__ int     taps[kXTile];
  const int          row   = threadIdx.y;
  const int          batch = blockIdx.z;
  const int          x     = static_cast<int>(blockIdx.x) * kXTile + row;
  const int          y     = blockIdx.y;
  if (threadIdx.x == 0) {
    const int64_t source_x = static_cast<int64_t>(x) + shift[2 * batch];
    const int64_t source_y = static_cast<int64_t>(y) + shift[2 * batch + 1];
    if (x >= width || source_x < 0 || source_x >= width || source_y < 0 || source_y >= height) {
      taps[row]    = 0;
      offsets[row] = 0;
    } else {
      offsets[row] = ((source_y * width + source_x) * channels) / kVector;
      taps[row]    = 1;
    }
  }
  __syncthreads();
  if (x >= width || taps[row] == 0) return;

  const int     groups = static_cast<int>(channels) / kVector;
  const int64_t plane  = static_cast<int64_t>(batch) * spatial * groups;
  const int64_t base =
      ((static_cast<int64_t>(batch) * spatial + static_cast<int64_t>(y) * width + x) * channels) / kVector;
  const float4* source = reinterpret_cast<const float4*>(grad_out) + base;
  float4*       target = reinterpret_cast<float4*>(grad_bev) + plane + offsets[row];
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    target[group] = source[group];
  }
}

// --------------------------------------------------------------------------------------
// launch configuration and validation
// --------------------------------------------------------------------------------------

int threads_per_block(int64_t work) {
  if (work >= kMaxThreads) return kMaxThreads;
  const int64_t rounded = ((work + kWarpSize - 1) / kWarpSize) * kWarpSize;
  return static_cast<int>(rounded < kWarpSize ? kWarpSize : rounded);
}

int group_threads(int64_t groups) {
  if (groups >= kMaxGroupWidth) return kMaxGroupWidth;
  const int64_t rounded = ((groups + kWarpSize - 1) / kWarpSize) * kWarpSize;
  return static_cast<int>(rounded < kWarpSize ? kWarpSize : rounded);
}

void check_geometry(
    ffi::TensorView bev, ffi::TensorView angle, ffi::TensorView shift, int64_t bev_h, int64_t bev_w, int64_t mode) {
  TVM_FFI_ICHECK(same_dtype(bev.dtype(), kDLFloat32) || same_dtype(bev.dtype(), kDLFloat16) ||
                 same_dtype(bev.dtype(), kDLBFloat16))
      << "bev must use float32, float16 or bfloat16";
  check_musa_contiguous(bev, "bev", bev.dtype());
  TVM_FFI_ICHECK_EQ(bev.ndim(), 3) << "bev must be (batch, bev_h*bev_w, channels)";
  TVM_FFI_ICHECK_GT(bev_h, 0) << "bev_h must be positive";
  TVM_FFI_ICHECK_GT(bev_w, 0) << "bev_w must be positive";
  TVM_FFI_ICHECK_LT(bev_h, INT_MAX) << "bev_h exceeds the indexing range";
  TVM_FFI_ICHECK_LT(bev_w, INT_MAX) << "bev_w exceeds the indexing range";
  TVM_FFI_ICHECK_EQ(bev.size(1), bev_h * bev_w) << "bev_h*bev_w must equal the bev plane size";
  TVM_FFI_ICHECK_GT(bev.size(2), 0) << "bev channels must be positive";
  TVM_FFI_ICHECK_LT(bev.size(2), static_cast<int64_t>(INT_MAX)) << "bev channels exceed the indexing range";
  TVM_FFI_ICHECK_LE(bev.size(0) * bev_h * bev_w, static_cast<int64_t>(INT_MAX)) << "bev plane exceeds the launch range";
  TVM_FFI_ICHECK(mode == kBilinear || mode == kNearest) << "unknown interpolation mode";
  check_musa_contiguous(angle, "trig", kDLFloat32);
  check_same_musa_device(bev, angle, "trig");
  TVM_FFI_ICHECK_EQ(angle.ndim(), 2) << "trig must be (batch, 2)";
  TVM_FFI_ICHECK_EQ(angle.size(0), bev.size(0)) << "trig length must match the batch";
  TVM_FFI_ICHECK_EQ(angle.size(1), 2) << "trig must be (batch, 2)";
  if (shift.numel() != 0) {
    TVM_FFI_ICHECK(same_dtype(shift.dtype(), kDLInt32) || same_dtype(shift.dtype(), kDLFloat32))
        << "shift must use int32 or float32";
    check_musa_contiguous(shift, "shift", shift.dtype());
    check_same_musa_device(bev, shift, "shift");
    TVM_FFI_ICHECK_EQ(shift.ndim(), 2) << "shift must be (batch, 2)";
    TVM_FFI_ICHECK_EQ(shift.size(0), bev.size(0)) << "shift length must match the batch";
    TVM_FFI_ICHECK_EQ(shift.size(1), 2) << "shift must be (batch, 2)";
  }
}

void check_plane(ffi::TensorView tensor, ffi::TensorView bev, const char* name) {
  check_musa_contiguous(tensor, name, bev.dtype());
  check_same_musa_device(bev, tensor, name);
  TVM_FFI_ICHECK_EQ(tensor.ndim(), bev.ndim()) << name << " rank mismatch";
  for (int axis = 0; axis < bev.ndim(); ++axis) {
    TVM_FFI_ICHECK_EQ(tensor.size(axis), bev.size(axis)) << name << " shape mismatch";
  }
}

// The tiled fast path needs float32 channels divisible by four and grid dimensions
// that fit the hardware launch limits.
bool tiled(ffi::TensorView bev, int64_t bev_h) {
  return same_dtype(bev.dtype(), kDLFloat32) && bev.size(2) % kVector == 0 && bev_h <= kGridLimit &&
         bev.size(0) <= kGridLimit;
}

template <typename T>
void launch_forward(ffi::TensorView bev,
                    const float*    angle,
                    const void*     shift,
                    int64_t         shift_kind,
                    ffi::TensorView out,
                    int64_t         bev_h,
                    int64_t         bev_w,
                    float           centre_x,
                    float           centre_y,
                    int64_t         mode) {
  const int64_t channels  = bev.size(2);
  const int64_t spatial   = bev_h * bev_w;
  const int64_t locations = bev.size(0) * spatial;
  if (locations == 0) return;
  const unsigned int blocks  = static_cast<unsigned int>(locations);
  const int          width   = static_cast<int>(bev_w);
  const int          height  = static_cast<int>(bev_h);
  const int          threads = threads_per_block(channels);
  const T*           source  = static_cast<const T*>(bev.data_ptr());
  T*                 target  = static_cast<T*>(out.data_ptr());
  if (shift_kind == kShiftInteger) {
    shift_forward_kernel<T><<<blocks, threads, 0, stream(bev)>>>(
        source, static_cast<const int32_t*>(shift), target, channels, spatial, width, height);
    return;
  }
  if (mode == kNearest) {
    affine_forward_kernel<T, true><<<blocks, threads, 0, stream(bev)>>>(
        source, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
    return;
  }
  affine_forward_kernel<T, false><<<blocks, threads, 0, stream(bev)>>>(
      source, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
}

void launch_forward_tiled(ffi::TensorView bev,
                          const float*    angle,
                          const void*     shift,
                          int64_t         shift_kind,
                          ffi::TensorView out,
                          int64_t         bev_h,
                          int64_t         bev_w,
                          float           centre_x,
                          float           centre_y,
                          int64_t         mode) {
  const int64_t channels = bev.size(2);
  const int64_t spatial  = bev_h * bev_w;
  if (bev.size(0) * spatial == 0) return;
  const int    width  = static_cast<int>(bev_w);
  const int    height = static_cast<int>(bev_h);
  const dim3   block(group_threads(channels / kVector), kXTile);
  const dim3   grid(static_cast<unsigned int>((width + kXTile - 1) / kXTile),
                  static_cast<unsigned int>(height),
                  static_cast<unsigned int>(bev.size(0)));
  const float* source = static_cast<const float*>(bev.data_ptr());
  float*       target = static_cast<float*>(out.data_ptr());
  if (shift_kind == kShiftInteger) {
    shift_forward_tiled_kernel<<<grid, block, 0, stream(bev)>>>(
        source, static_cast<const int32_t*>(shift), target, channels, spatial, width, height);
    return;
  }
  if (mode == kNearest) {
    affine_forward_tiled_kernel<true><<<grid, block, 0, stream(bev)>>>(
        source, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
    return;
  }
  affine_forward_tiled_kernel<false><<<grid, block, 0, stream(bev)>>>(
      source, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
}

template <typename T>
void launch_backward(ffi::TensorView bev,
                     const float*    angle,
                     const void*     shift,
                     int64_t         shift_kind,
                     ffi::TensorView grad_out,
                     ffi::TensorView grad_bev,
                     int64_t         bev_h,
                     int64_t         bev_w,
                     float           centre_x,
                     float           centre_y,
                     int64_t         mode) {
  const int64_t channels  = bev.size(2);
  const int64_t spatial   = bev_h * bev_w;
  const int64_t locations = bev.size(0) * spatial;
  if (locations == 0 || channels == 0) return;
  if (musaMemsetAsync(grad_bev.data_ptr(), 0, grad_bev.numel() * sizeof(float), stream(bev)) != musaSuccess) {
    TVM_FFI_ICHECK(false) << "temporal_align could not clear the gradient buffer";
  }
  const unsigned int blocks   = static_cast<unsigned int>(locations);
  const int          width    = static_cast<int>(bev_w);
  const int          height   = static_cast<int>(bev_h);
  const int          threads  = threads_per_block(channels);
  const T*           gradient = static_cast<const T*>(grad_out.data_ptr());
  float*             target   = static_cast<float*>(grad_bev.data_ptr());
  if (shift_kind == kShiftInteger) {
    shift_backward_kernel<T><<<blocks, threads, 0, stream(bev)>>>(
        gradient, static_cast<const int32_t*>(shift), target, channels, spatial, width, height);
    return;
  }
  if (mode == kNearest) {
    affine_backward_kernel<T, true><<<blocks, threads, 0, stream(bev)>>>(
        gradient, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
    return;
  }
  affine_backward_kernel<T, false><<<blocks, threads, 0, stream(bev)>>>(
      gradient, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
}

void launch_backward_tiled(ffi::TensorView bev,
                           const float*    angle,
                           const void*     shift,
                           int64_t         shift_kind,
                           ffi::TensorView grad_out,
                           ffi::TensorView grad_bev,
                           int64_t         bev_h,
                           int64_t         bev_w,
                           float           centre_x,
                           float           centre_y,
                           int64_t         mode) {
  const int64_t channels = bev.size(2);
  const int64_t spatial  = bev_h * bev_w;
  if (bev.size(0) * spatial == 0 || channels == 0) return;
  if (musaMemsetAsync(grad_bev.data_ptr(), 0, grad_bev.numel() * sizeof(float), stream(bev)) != musaSuccess) {
    TVM_FFI_ICHECK(false) << "temporal_align could not clear the gradient buffer";
  }
  const int    width  = static_cast<int>(bev_w);
  const int    height = static_cast<int>(bev_h);
  const dim3   block(group_threads(channels / kVector), kXTile);
  const dim3   grid(static_cast<unsigned int>((width + kXTile - 1) / kXTile),
                  static_cast<unsigned int>(height),
                  static_cast<unsigned int>(bev.size(0)));
  const float* gradient = static_cast<const float*>(grad_out.data_ptr());
  float*       target   = static_cast<float*>(grad_bev.data_ptr());
  if (shift_kind == kShiftInteger) {
    shift_backward_tiled_kernel<<<grid, block, 0, stream(bev)>>>(
        gradient, static_cast<const int32_t*>(shift), target, channels, spatial, width, height);
    return;
  }
  if (mode == kNearest) {
    affine_backward_tiled_kernel<true><<<grid, block, 0, stream(bev)>>>(
        gradient, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
    return;
  }
  affine_backward_tiled_kernel<false><<<grid, block, 0, stream(bev)>>>(
      gradient, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
}

// --------------------------------------------------------------------------------------
// planar launch configuration
// --------------------------------------------------------------------------------------

// A planar plane is (batch, channels, bev_h, bev_w) float32, contiguous along x.
bool planar_eligible(ffi::TensorView bev, int64_t bev_h, int64_t bev_w) {
  const int64_t span = kPlanarWidth;
  return same_dtype(bev.dtype(), kDLFloat32) && bev.ndim() == 4 && bev.size(1) > 0 && bev_h > 0 &&
         bev_h <= kGridLimit && bev_w > 0 && (bev_w + span - 1) / span <= kGridLimit && bev.size(0) > 0 &&
         bev.size(0) <= kGridLimit;
}

dim3 planar_block() {
  return dim3(kPlanarWidth, kPlanarChannels);
}

dim3 planar_grid(int64_t bev_w, int64_t bev_h, int64_t batch) {
  const int64_t span = kPlanarWidth;
  return dim3(static_cast<unsigned int>((bev_w + span - 1) / span),
              static_cast<unsigned int>(bev_h),
              static_cast<unsigned int>(batch));
}

void launch_forward_planar(ffi::TensorView bev,
                           const float*    angle,
                           const void*     shift,
                           int64_t         shift_kind,
                           ffi::TensorView out,
                           int64_t         bev_h,
                           int64_t         bev_w,
                           float           centre_x,
                           float           centre_y,
                           int64_t         mode) {
  const int64_t channels = bev.size(1);
  const int64_t spatial  = bev_h * bev_w;
  const int     width    = static_cast<int>(bev_w);
  const int     height   = static_cast<int>(bev_h);
  const float*  source   = static_cast<const float*>(bev.data_ptr());
  float*        target   = static_cast<float*>(out.data_ptr());
  const dim3    block    = planar_block();
  const dim3    grid     = planar_grid(bev_w, bev_h, bev.size(0));
  if (mode == kNearest) {
    planar_forward_tiled_kernel<true><<<grid, block, 0, stream(bev)>>>(
        source, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
    return;
  }
  planar_forward_tiled_kernel<false><<<grid, block, 0, stream(bev)>>>(
      source, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
}

void launch_backward_planar(ffi::TensorView bev,
                            const float*    angle,
                            const void*     shift,
                            int64_t         shift_kind,
                            ffi::TensorView grad_out,
                            ffi::TensorView grad_bev,
                            int64_t         bev_h,
                            int64_t         bev_w,
                            float           centre_x,
                            float           centre_y,
                            int64_t         mode) {
  const int64_t channels = bev.size(1);
  const int64_t spatial  = bev_h * bev_w;
  const int     width    = static_cast<int>(bev_w);
  const int     height   = static_cast<int>(bev_h);
  const float*  gradient = static_cast<const float*>(grad_out.data_ptr());
  float*        target   = static_cast<float*>(grad_bev.data_ptr());
  // The gather writes every source pixel, so the buffer does not need clearing.
  const dim3 block = planar_block();
  const dim3 grid(static_cast<unsigned int>((bev_w + kPlanarWidth - 1) / kPlanarWidth),
                  static_cast<unsigned int>(bev_h),
                  static_cast<unsigned int>(bev.size(0)));
  if (mode == kNearest) {
    planar_backward_gather_kernel<true><<<grid, block, 0, stream(bev)>>>(
        gradient, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
    return;
  }
  planar_backward_gather_kernel<false><<<grid, block, 0, stream(bev)>>>(
      gradient, angle, shift, shift_kind, target, channels, spatial, width, height, centre_x, centre_y);
}

}  // namespace

void temporal_align_forward(ffi::TensorView bev,
                            ffi::TensorView angle,
                            ffi::TensorView shift,
                            ffi::TensorView out,
                            int64_t         bev_h,
                            int64_t         bev_w,
                            double          centre_x,
                            double          centre_y,
                            int64_t         mode) {
  check_geometry(bev, angle, shift, bev_h, bev_w, mode);
  check_plane(out, bev, "out");
  const int64_t shift_kind =
      shift.numel() == 0 ? 0 : (same_dtype(shift.dtype(), kDLInt32) ? kShiftInteger : kShiftFloat);
  const float* angle_ptr  = angle.numel() == 0 ? nullptr : static_cast<const float*>(angle.data_ptr());
  const void*  shift_ptr  = shift.numel() == 0 ? nullptr : shift.data_ptr();
  const float  centre_x_f = static_cast<float>(centre_x);
  const float  centre_y_f = static_cast<float>(centre_y);
  if (tiled(bev, bev_h)) {
    launch_forward_tiled(bev, angle_ptr, shift_ptr, shift_kind, out, bev_h, bev_w, centre_x_f, centre_y_f, mode);
  } else if (same_dtype(bev.dtype(), kDLFloat32)) {
    launch_forward<float>(bev, angle_ptr, shift_ptr, shift_kind, out, bev_h, bev_w, centre_x_f, centre_y_f, mode);
  } else if (same_dtype(bev.dtype(), kDLFloat16)) {
    launch_forward<__half>(bev, angle_ptr, shift_ptr, shift_kind, out, bev_h, bev_w, centre_x_f, centre_y_f, mode);
  } else {
    launch_forward<__mt_bfloat16>(
        bev, angle_ptr, shift_ptr, shift_kind, out, bev_h, bev_w, centre_x_f, centre_y_f, mode);
  }
  check_last_error("temporal_align forward");
}

void temporal_align_backward(ffi::TensorView bev,
                             ffi::TensorView angle,
                             ffi::TensorView shift,
                             ffi::TensorView grad_out,
                             ffi::TensorView grad_bev,
                             int64_t         bev_h,
                             int64_t         bev_w,
                             double          centre_x,
                             double          centre_y,
                             int64_t         mode) {
  check_geometry(bev, angle, shift, bev_h, bev_w, mode);
  check_plane(grad_out, bev, "grad_output");
  check_musa_contiguous(grad_bev, "grad_bev", kDLFloat32);
  check_same_musa_device(bev, grad_bev, "grad_bev");
  TVM_FFI_ICHECK_EQ(grad_bev.ndim(), bev.ndim()) << "grad_bev rank mismatch";
  for (int axis = 0; axis < bev.ndim(); ++axis) {
    TVM_FFI_ICHECK_EQ(grad_bev.size(axis), bev.size(axis)) << "grad_bev shape mismatch";
  }
  const int64_t shift_kind =
      shift.numel() == 0 ? 0 : (same_dtype(shift.dtype(), kDLInt32) ? kShiftInteger : kShiftFloat);
  const float* angle_ptr  = angle.numel() == 0 ? nullptr : static_cast<const float*>(angle.data_ptr());
  const void*  shift_ptr  = shift.numel() == 0 ? nullptr : shift.data_ptr();
  const float  centre_x_f = static_cast<float>(centre_x);
  const float  centre_y_f = static_cast<float>(centre_y);
  if (tiled(bev, bev_h)) {
    launch_backward_tiled(
        bev, angle_ptr, shift_ptr, shift_kind, grad_out, grad_bev, bev_h, bev_w, centre_x_f, centre_y_f, mode);
  } else if (same_dtype(bev.dtype(), kDLFloat32)) {
    launch_backward<float>(
        bev, angle_ptr, shift_ptr, shift_kind, grad_out, grad_bev, bev_h, bev_w, centre_x_f, centre_y_f, mode);
  } else if (same_dtype(bev.dtype(), kDLFloat16)) {
    launch_backward<__half>(
        bev, angle_ptr, shift_ptr, shift_kind, grad_out, grad_bev, bev_h, bev_w, centre_x_f, centre_y_f, mode);
  } else {
    launch_backward<__mt_bfloat16>(
        bev, angle_ptr, shift_ptr, shift_kind, grad_out, grad_bev, bev_h, bev_w, centre_x_f, centre_y_f, mode);
  }
  check_last_error("temporal_align backward");
}

// --------------------------------------------------------------------------------------
// planar entries: the caller's channels-first layout, unchanged
// --------------------------------------------------------------------------------------

void check_planar_geometry(
    ffi::TensorView bev, ffi::TensorView angle, ffi::TensorView shift, int64_t bev_h, int64_t bev_w, int64_t mode) {
  TVM_FFI_ICHECK(same_dtype(bev.dtype(), kDLFloat32)) << "planar bev must use float32";
  check_musa_contiguous(bev, "bev", bev.dtype());
  TVM_FFI_ICHECK_EQ(bev.ndim(), 4) << "bev must be (batch, channels, bev_h, bev_w)";
  TVM_FFI_ICHECK_GT(bev_h, 0) << "bev_h must be positive";
  TVM_FFI_ICHECK_GT(bev_w, 0) << "bev_w must be positive";
  TVM_FFI_ICHECK_EQ(bev.size(2), bev_h) << "bev_h must equal the bev plane height";
  TVM_FFI_ICHECK_EQ(bev.size(3), bev_w) << "bev_w must equal the bev plane width";
  TVM_FFI_ICHECK_GT(bev.size(1), 0) << "bev channels must be positive";
  TVM_FFI_ICHECK_LT(bev.size(1), static_cast<int64_t>(INT_MAX)) << "bev channels exceed the indexing range";
  TVM_FFI_ICHECK(mode == kBilinear || mode == kNearest) << "unknown interpolation mode";
  check_musa_contiguous(angle, "trig", kDLFloat32);
  check_same_musa_device(bev, angle, "trig");
  TVM_FFI_ICHECK_EQ(angle.ndim(), 2) << "trig must be (batch, 2)";
  TVM_FFI_ICHECK_EQ(angle.size(0), bev.size(0)) << "trig length must match the batch";
  TVM_FFI_ICHECK_EQ(angle.size(1), 2) << "trig must be (batch, 2)";
  if (shift.numel() != 0) {
    TVM_FFI_ICHECK(same_dtype(shift.dtype(), kDLFloat32)) << "a planar call accepts only a float32 shift";
    check_musa_contiguous(shift, "shift", shift.dtype());
    check_same_musa_device(bev, shift, "shift");
    TVM_FFI_ICHECK_EQ(shift.ndim(), 2) << "shift must be (batch, 2)";
    TVM_FFI_ICHECK_EQ(shift.size(0), bev.size(0)) << "shift length must match the batch";
    TVM_FFI_ICHECK_EQ(shift.size(1), 2) << "shift must be (batch, 2)";
  }
}

void temporal_align_forward_planar(ffi::TensorView bev,
                                   ffi::TensorView angle,
                                   ffi::TensorView shift,
                                   ffi::TensorView out,
                                   int64_t         bev_h,
                                   int64_t         bev_w,
                                   double          centre_x,
                                   double          centre_y,
                                   int64_t         mode) {
  check_planar_geometry(bev, angle, shift, bev_h, bev_w, mode);
  TVM_FFI_ICHECK(planar_eligible(bev, bev_h, bev_w)) << "the planar path cannot serve this shape";
  check_plane(out, bev, "out");
  const int64_t shift_kind = shift.numel() == 0 ? 0 : kShiftFloat;
  const float*  angle_ptr  = angle.numel() == 0 ? nullptr : static_cast<const float*>(angle.data_ptr());
  const void*   shift_ptr  = shift.numel() == 0 ? nullptr : shift.data_ptr();
  launch_forward_planar(bev,
                        angle_ptr,
                        shift_ptr,
                        shift_kind,
                        out,
                        bev_h,
                        bev_w,
                        static_cast<float>(centre_x),
                        static_cast<float>(centre_y),
                        mode);
  check_last_error("temporal_align forward planar");
}

void temporal_align_backward_planar(ffi::TensorView bev,
                                    ffi::TensorView angle,
                                    ffi::TensorView shift,
                                    ffi::TensorView grad_out,
                                    ffi::TensorView grad_bev,
                                    int64_t         bev_h,
                                    int64_t         bev_w,
                                    double          centre_x,
                                    double          centre_y,
                                    int64_t         mode) {
  check_planar_geometry(bev, angle, shift, bev_h, bev_w, mode);
  TVM_FFI_ICHECK(planar_eligible(bev, bev_h, bev_w)) << "the planar path cannot serve this shape";
  check_plane(grad_out, bev, "grad_output");
  check_musa_contiguous(grad_bev, "grad_bev", kDLFloat32);
  check_same_musa_device(bev, grad_bev, "grad_bev");
  TVM_FFI_ICHECK_EQ(grad_bev.ndim(), bev.ndim()) << "grad_bev rank mismatch";
  for (int axis = 0; axis < bev.ndim(); ++axis) {
    TVM_FFI_ICHECK_EQ(grad_bev.size(axis), bev.size(axis)) << "grad_bev shape mismatch";
  }
  const int64_t shift_kind = shift.numel() == 0 ? 0 : kShiftFloat;
  const float*  angle_ptr  = angle.numel() == 0 ? nullptr : static_cast<const float*>(angle.data_ptr());
  const void*   shift_ptr  = shift.numel() == 0 ? nullptr : shift.data_ptr();
  launch_backward_planar(bev,
                         angle_ptr,
                         shift_ptr,
                         shift_kind,
                         grad_out,
                         grad_bev,
                         bev_h,
                         bev_w,
                         static_cast<float>(centre_x),
                         static_cast<float>(centre_y),
                         mode);
  check_last_error("temporal_align backward planar");
}

// Prepares the per-sample ``(cosine, sine)`` table the sampling kernels read.
// ``angle`` is empty when the call has no rotation; the table then holds the
// identity rotation.  Doing the trigonometry once per call keeps the fp64 work out
// of the bandwidth-bound sampling kernels.
void temporal_align_prepare_trig(ffi::TensorView angle, ffi::TensorView trig) {
  check_musa_contiguous(trig, "trig", kDLFloat32);
  TVM_FFI_ICHECK_EQ(trig.ndim(), 2) << "trig must be (batch, 2)";
  TVM_FFI_ICHECK_EQ(trig.size(1), 2) << "trig must be (batch, 2)";
  const int64_t batch = trig.size(0);
  const float*  angle_ptr = nullptr;
  if (angle.numel() != 0) {
    check_musa_contiguous(angle, "angle", kDLFloat32);
    check_same_musa_device(trig, angle, "angle");
    TVM_FFI_ICHECK_EQ(angle.ndim(), 1) << "angle must be (batch,)";
    TVM_FFI_ICHECK_EQ(angle.size(0), batch) << "angle length must match the batch";
    angle_ptr = static_cast<const float*>(angle.data_ptr());
  }
  if (batch > 0) {
    constexpr int threads = 128;
    const int     blocks  = static_cast<int>((batch + threads - 1) / threads);
    prepare_trig_kernel<<<blocks, threads, 0, stream(trig)>>>(
        angle_ptr, static_cast<int>(batch), static_cast<float*>(trig.data_ptr()));
  }
  check_last_error("temporal_align prepare trig");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(temporal_align_prepare_trig, temporal_align_prepare_trig);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(temporal_align_forward, temporal_align_forward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(temporal_align_backward, temporal_align_backward);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(temporal_align_forward_planar, temporal_align_forward_planar);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(temporal_align_backward_planar, temporal_align_backward_planar);
