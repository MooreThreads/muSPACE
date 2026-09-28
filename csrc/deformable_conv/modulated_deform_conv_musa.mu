/* Torch-ABI-independent DCNv2 MUSA launchers.
 *
 * All outputs are supplied by Python as DLPack tensor views.  This file is
 * intentionally independent of the PyTorch C++ ABI.
 */

#include <musa_runtime.h>
#include <tvm/ffi/function.h>

#include <cstring>
#include <mutex>

#include "../ffi/tensor_utils.hpp"
#include "modulated_deform_conv_musa_kernel.muh"

namespace ffi = tvm::ffi;
using muspace::ffi_utils::dim;
using muspace::ffi_utils::stream;

namespace {

using muspace_dcn::kCol2ImMargin;
using muspace_dcn::kCol2ImTileH;
using muspace_dcn::kCol2ImTileW;
using muspace_dcn::kWarpSize;

constexpr int        kThreadsPerBlock = 512;
constexpr int        kMaxInt32        = 2147483647;
constexpr DLDataType kDLFloat16       = DLDataType{kDLFloat, 16, 1};

inline int GetBlocks(int n, int threads) {
  return (n + threads - 1) / threads;
}

inline void CheckLastError(const char* name) {
  const musaError_t error = musaGetLastError();
  TVM_FFI_ICHECK_EQ(error, musaSuccess) << name << " launch failed: " << musaGetErrorString(error);
}

int MultiProcessorCount(int device_id) {
  static std::once_flag once;
  static int            counts[64] = {0};
  std::call_once(once, [] { std::memset(counts, 0, sizeof(counts)); });
  if (device_id < 0 || device_id >= 64) return 1;
  if (counts[device_id] == 0) {
    musaDeviceProp prop;
    counts[device_id] = musaGetDeviceProperties(&prop, device_id) == musaSuccess ? prop.multiProcessorCount : 1;
  }
  return counts[device_id];
}

inline bool DcnDtype(DLDataType dtype) {
  return same_dtype(dtype, kDLFloat32) || same_dtype(dtype, kDLFloat16);
}

void CheckDcnTensor(ffi::TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLExtDev) << name << " must be a MUSA tensor";
  TVM_FFI_ICHECK(DcnDtype(tensor.dtype())) << name << " must use float32 or float16";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

void CheckSameDtype(ffi::TensorView first, const char* first_name, ffi::TensorView second, const char* second_name) {
  TVM_FFI_ICHECK(same_dtype(first.dtype(), second.dtype()))
      << second_name << " must have the same dtype as " << first_name;
  TVM_FFI_ICHECK_EQ(first.device().device_id, second.device().device_id)
      << second_name << " must be on the same MUSA device as " << first_name;
}

int64_t OutputSize(int64_t size, int64_t pad, int64_t dilation, int64_t kernel, int64_t stride) {
  return (size + 2 * pad - (dilation * (kernel - 1) + 1)) / stride + 1;
}

struct NchwGeometry {
  int64_t batch, channels, height, width, height_col, width_col;
};

NchwGeometry CheckNchwShapes(ffi::TensorView input,
                             ffi::TensorView offset,
                             ffi::TensorView mask,
                             int64_t         kernel_h,
                             int64_t         kernel_w,
                             int64_t         pad_h,
                             int64_t         pad_w,
                             int64_t         stride_h,
                             int64_t         stride_w,
                             int64_t         dilation_h,
                             int64_t         dilation_w,
                             int64_t         deform_groups) {
  TVM_FFI_ICHECK_EQ(input.ndim(), 4) << "input must have shape [B, C, H, W]";
  TVM_FFI_ICHECK_EQ(offset.ndim(), 4) << "offset must have shape [B, dg*2*kH*kW, Ho, Wo]";
  TVM_FFI_ICHECK_EQ(mask.ndim(), 4) << "mask must have shape [B, dg*kH*kW, Ho, Wo]";
  TVM_FFI_ICHECK(kernel_h > 0 && kernel_w > 0 && stride_h > 0 && stride_w > 0 && dilation_h > 0 && dilation_w > 0 &&
                 deform_groups > 0)
      << "kernel, stride, dilation and deform_groups must be positive";
  const int64_t batch = dim(input, 0), channels = dim(input, 1);
  const int64_t height = dim(input, 2), width = dim(input, 3);
  TVM_FFI_ICHECK_EQ(channels % deform_groups, 0) << "channels must divide deform_groups";
  const int64_t height_col = OutputSize(height, pad_h, dilation_h, kernel_h, stride_h);
  const int64_t width_col  = OutputSize(width, pad_w, dilation_w, kernel_w, stride_w);
  TVM_FFI_ICHECK(height_col > 0 && width_col > 0) << "the convolution output would be empty";
  TVM_FFI_ICHECK_EQ(dim(offset, 0), batch);
  TVM_FFI_ICHECK_EQ(dim(offset, 1), deform_groups * 2 * kernel_h * kernel_w);
  TVM_FFI_ICHECK_EQ(dim(offset, 2), height_col);
  TVM_FFI_ICHECK_EQ(dim(offset, 3), width_col);
  TVM_FFI_ICHECK_EQ(dim(mask, 0), batch);
  TVM_FFI_ICHECK_EQ(dim(mask, 1), deform_groups * kernel_h * kernel_w);
  TVM_FFI_ICHECK_EQ(dim(mask, 2), height_col);
  TVM_FFI_ICHECK_EQ(dim(mask, 3), width_col);
  TVM_FFI_ICHECK(channels * kernel_h * kernel_w * batch * height_col * width_col < kMaxInt32)
      << "the column buffer exceeds int32 indexing range";
  return {batch, channels, height, width, height_col, width_col};
}

struct NhwcGeometry {
  int64_t batch, channels, height, width, height_col, width_col;
};

NhwcGeometry CheckNhwcShapes(ffi::TensorView input,
                             ffi::TensorView offset,
                             int64_t         kernel_h,
                             int64_t         kernel_w,
                             int64_t         pad_h,
                             int64_t         pad_w,
                             int64_t         stride_h,
                             int64_t         stride_w,
                             int64_t         dilation_h,
                             int64_t         dilation_w) {
  TVM_FFI_ICHECK_EQ(input.ndim(), 4) << "input must have shape [B, H, W, C]";
  TVM_FFI_ICHECK_EQ(offset.ndim(), 4) << "offset must have shape [B, Ho, Wo, 3*kH*kW]";
  TVM_FFI_ICHECK(kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1)
      << "the NHWC kernels require 3x3, stride 1, dilation 1";
  const int64_t batch = dim(input, 0), height = dim(input, 1), width = dim(input, 2), channels = dim(input, 3);
  TVM_FFI_ICHECK(channels % kWarpSize == 0 && channels <= 1024)
      << "NHWC kernels require channels divisible by 32 and <= 1024";
  const int64_t height_col = OutputSize(height, pad_h, dilation_h, kernel_h, stride_h);
  const int64_t width_col  = OutputSize(width, pad_w, dilation_w, kernel_w, stride_w);
  TVM_FFI_ICHECK(height_col > 0 && width_col > 0) << "the convolution output would be empty";
  TVM_FFI_ICHECK_EQ(dim(offset, 0), batch);
  TVM_FFI_ICHECK_EQ(dim(offset, 1), height_col);
  TVM_FFI_ICHECK_EQ(dim(offset, 2), width_col);
  TVM_FFI_ICHECK_EQ(dim(offset, 3), 3 * kernel_h * kernel_w);
  TVM_FFI_ICHECK(batch * height_col * width_col * kernel_h * kernel_w * channels < kMaxInt32)
      << "the column buffer exceeds int32 indexing range";
  return {batch, channels, height, width, height_col, width_col};
}

template <typename T>
void LaunchNchwIm2Col(ffi::TensorView     input,
                      ffi::TensorView     offset,
                      ffi::TensorView     mask,
                      ffi::TensorView     columns,
                      const NchwGeometry& geo,
                      int64_t             kh,
                      int64_t             kw,
                      int64_t             ph,
                      int64_t             pw,
                      int64_t             sh,
                      int64_t             sw,
                      int64_t             dh,
                      int64_t             dw,
                      int64_t             dg) {
  const int hw = static_cast<int>(geo.height_col * geo.width_col), threads = 256;
  const int spatial_blocks = GetBlocks(hw, threads), cpg = static_cast<int>(geo.channels / dg);
  const int target_blocks = MultiProcessorCount(input.device().device_id) * 4;
  int       chunks        = 1;
  if (spatial_blocks * geo.batch * kh < target_blocks) {
    for (int cc = 2; cc <= cpg / 32; cc *= 2) {
      if (cpg % cc == 0) {
        chunks = cc;
        if (spatial_blocks * geo.batch * kh * cc >= target_blocks) break;
      }
    }
  }
  dim3 grid(spatial_blocks, static_cast<int>(geo.batch), static_cast<int>(kh) * chunks);
  muspace_dcn::modulated_deform_conv_im2col_nchw_kernel<T>
      <<<grid, threads, 0, stream(input)>>>(static_cast<const T*>(input.data_ptr()),
                                            static_cast<const T*>(offset.data_ptr()),
                                            static_cast<const T*>(mask.data_ptr()),
                                            geo.height,
                                            geo.width,
                                            kh,
                                            kw,
                                            ph,
                                            pw,
                                            sh,
                                            sw,
                                            dh,
                                            dw,
                                            geo.batch,
                                            geo.channels,
                                            dg,
                                            geo.height_col,
                                            geo.width_col,
                                            chunks,
                                            static_cast<T*>(columns.data_ptr()));
}

template <typename T>
void LaunchNchwCol2Im(ffi::TensorView     columns,
                      ffi::TensorView     input,
                      ffi::TensorView     offset,
                      ffi::TensorView     mask,
                      ffi::TensorView     grad_input,
                      const NchwGeometry& geo,
                      int64_t             kh,
                      int64_t             kw,
                      int64_t             ph,
                      int64_t             pw,
                      int64_t             sh,
                      int64_t             sw,
                      int64_t             dh,
                      int64_t             dw,
                      int64_t             dg) {
  const int smem_h = kCol2ImTileH + 2 * kCol2ImMargin;
  const int smem_w = kCol2ImTileW + 2 * kCol2ImMargin + 1;
  dim3      grid(GetBlocks(static_cast<int>(geo.width_col), kCol2ImTileW),
            GetBlocks(static_cast<int>(geo.height_col), kCol2ImTileH),
            static_cast<int>(geo.batch * geo.channels));
  muspace_dcn::modulated_deform_conv_col2im_nchw_kernel<T>
      <<<grid, 128, smem_h * smem_w * sizeof(float), stream(input)>>>(static_cast<const T*>(columns.data_ptr()),
                                                                      static_cast<const T*>(offset.data_ptr()),
                                                                      static_cast<const T*>(mask.data_ptr()),
                                                                      geo.channels,
                                                                      geo.height,
                                                                      geo.width,
                                                                      kh,
                                                                      kw,
                                                                      ph,
                                                                      pw,
                                                                      sh,
                                                                      sw,
                                                                      dh,
                                                                      dw,
                                                                      geo.batch,
                                                                      dg,
                                                                      geo.height_col,
                                                                      geo.width_col,
                                                                      static_cast<T*>(grad_input.data_ptr()));
}

template <typename T>
void LaunchNchwCoord(ffi::TensorView     columns,
                     ffi::TensorView     input,
                     ffi::TensorView     offset,
                     ffi::TensorView     mask,
                     ffi::TensorView     grad_offset,
                     ffi::TensorView     grad_mask,
                     const NchwGeometry& geo,
                     int64_t             kh,
                     int64_t             kw,
                     int64_t             ph,
                     int64_t             pw,
                     int64_t             sh,
                     int64_t             sw,
                     int64_t             dh,
                     int64_t             dw,
                     int64_t             dg) {
  const int n = static_cast<int>(geo.batch * geo.height_col * geo.width_col * 2 * kh * kw * dg);
  muspace_dcn::modulated_deform_conv_col2im_coord_nchw_kernel<T>
      <<<GetBlocks(n, kThreadsPerBlock), kThreadsPerBlock, 0, stream(input)>>>(
          n,
          static_cast<const T*>(columns.data_ptr()),
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(offset.data_ptr()),
          static_cast<const T*>(mask.data_ptr()),
          geo.channels,
          geo.height,
          geo.width,
          kh,
          kw,
          ph,
          pw,
          sh,
          sw,
          dh,
          dw,
          geo.channels * kh * kw / dg,
          geo.batch,
          2 * kh * kw * dg,
          dg,
          geo.height_col,
          geo.width_col,
          static_cast<T*>(grad_offset.data_ptr()),
          static_cast<T*>(grad_mask.data_ptr()));
}

template <typename T>
void LaunchNhwcIm2Col(ffi::TensorView     input,
                      ffi::TensorView     offset,
                      ffi::TensorView     columns,
                      const NhwcGeometry& geo,
                      int64_t             ph,
                      int64_t             pw,
                      int64_t             sh,
                      int64_t             sw,
                      int64_t             dh,
                      int64_t             dw) {
  constexpr int threads = 256;
  dim3          grid(GetBlocks(static_cast<int>(geo.width_col), threads / kWarpSize),
            static_cast<int>(geo.height_col),
            static_cast<int>(geo.batch * 9));
  muspace_dcn::modulated_deform_conv_im2col_nhwc_kernel<T, 3, 3, 4>
      <<<grid, threads, 0, stream(input)>>>(static_cast<const T*>(input.data_ptr()),
                                            static_cast<const T*>(offset.data_ptr()),
                                            geo.batch,
                                            geo.channels,
                                            geo.height,
                                            geo.width,
                                            geo.height_col,
                                            geo.width_col,
                                            ph,
                                            pw,
                                            sh,
                                            sw,
                                            dh,
                                            dw,
                                            static_cast<T*>(columns.data_ptr()));
}

template <typename T>
void LaunchNhwcCol2Im(ffi::TensorView     grad_columns,
                      ffi::TensorView     input,
                      ffi::TensorView     offset,
                      ffi::TensorView     grad_input,
                      ffi::TensorView     grad_offset,
                      ffi::TensorView     grad_mask,
                      const NhwcGeometry& geo,
                      int64_t             ph,
                      int64_t             pw) {
  dim3 grid(static_cast<int>(geo.width_col), static_cast<int>(geo.height_col), static_cast<int>(geo.batch * 9));
  muspace_dcn::modulated_deform_conv_col2im_nhwc_kernel<T, 3, 3>
      <<<grid, static_cast<int>(geo.channels), 0, stream(input)>>>(static_cast<const T*>(grad_columns.data_ptr()),
                                                                   static_cast<const T*>(input.data_ptr()),
                                                                   static_cast<const T*>(offset.data_ptr()),
                                                                   geo.batch,
                                                                   geo.channels,
                                                                   geo.height,
                                                                   geo.width,
                                                                   geo.height_col,
                                                                   geo.width_col,
                                                                   ph,
                                                                   pw,
                                                                   static_cast<float*>(grad_input.data_ptr()),
                                                                   static_cast<float*>(grad_offset.data_ptr()),
                                                                   static_cast<float*>(grad_mask.data_ptr()));
}

#define MUSPACE_DCN_DISPATCH(tensor, code)                           \
  do {                                                               \
    if (same_dtype((tensor).dtype(), kDLFloat32)) {                  \
      using scalar_t = float;                                        \
      code;                                                          \
    } else if (same_dtype((tensor).dtype(), kDLFloat16)) {           \
      using scalar_t = half;                                         \
      code;                                                          \
    } else {                                                         \
      TVM_FFI_ICHECK(false) << "DCNv2 supports float32 and float16"; \
    }                                                                \
  } while (false)

}  // namespace

void modulated_deform_conv_im2col(ffi::TensorView input,
                                  ffi::TensorView offset,
                                  ffi::TensorView mask,
                                  int64_t         kh,
                                  int64_t         kw,
                                  int64_t         ph,
                                  int64_t         pw,
                                  int64_t         sh,
                                  int64_t         sw,
                                  int64_t         dh,
                                  int64_t         dw,
                                  int64_t         dg,
                                  ffi::TensorView columns) {
  CheckDcnTensor(input, "input");
  CheckDcnTensor(offset, "offset");
  CheckDcnTensor(mask, "mask");
  CheckDcnTensor(columns, "columns");
  CheckSameDtype(input, "input", offset, "offset");
  CheckSameDtype(input, "input", mask, "mask");
  CheckSameDtype(input, "input", columns, "columns");
  const auto geo = CheckNchwShapes(input, offset, mask, kh, kw, ph, pw, sh, sw, dh, dw, dg);
  TVM_FFI_ICHECK_EQ(columns.numel(), geo.channels * kh * kw * geo.batch * geo.height_col * geo.width_col);
  if (columns.numel() == 0) return;
  MUSPACE_DCN_DISPATCH(
      input, LaunchNchwIm2Col<scalar_t>(input, offset, mask, columns, geo, kh, kw, ph, pw, sh, sw, dh, dw, dg));
  CheckLastError("modulated_deform_conv_im2col");
}

void modulated_deform_conv_col2im(ffi::TensorView columns,
                                  ffi::TensorView input,
                                  ffi::TensorView offset,
                                  ffi::TensorView mask,
                                  int64_t         kh,
                                  int64_t         kw,
                                  int64_t         ph,
                                  int64_t         pw,
                                  int64_t         sh,
                                  int64_t         sw,
                                  int64_t         dh,
                                  int64_t         dw,
                                  int64_t         dg,
                                  ffi::TensorView grad_input) {
  CheckDcnTensor(columns, "columns");
  CheckDcnTensor(input, "input");
  CheckDcnTensor(offset, "offset");
  CheckDcnTensor(mask, "mask");
  CheckDcnTensor(grad_input, "grad_input");
  CheckSameDtype(input, "input", columns, "columns");
  CheckSameDtype(input, "input", offset, "offset");
  CheckSameDtype(input, "input", mask, "mask");
  CheckSameDtype(input, "input", grad_input, "grad_input");
  const auto geo = CheckNchwShapes(input, offset, mask, kh, kw, ph, pw, sh, sw, dh, dw, dg);
  TVM_FFI_ICHECK_EQ(columns.numel(), geo.channels * kh * kw * geo.batch * geo.height_col * geo.width_col);
  TVM_FFI_ICHECK_EQ(grad_input.numel(), input.numel());
  if (grad_input.numel() == 0) return;
  MUSPACE_DCN_DISPATCH(
      input,
      LaunchNchwCol2Im<scalar_t>(columns, input, offset, mask, grad_input, geo, kh, kw, ph, pw, sh, sw, dh, dw, dg));
  CheckLastError("modulated_deform_conv_col2im");
}

void modulated_deform_conv_col2im_coord(ffi::TensorView columns,
                                        ffi::TensorView input,
                                        ffi::TensorView offset,
                                        ffi::TensorView mask,
                                        int64_t         kh,
                                        int64_t         kw,
                                        int64_t         ph,
                                        int64_t         pw,
                                        int64_t         sh,
                                        int64_t         sw,
                                        int64_t         dh,
                                        int64_t         dw,
                                        int64_t         dg,
                                        ffi::TensorView grad_offset,
                                        ffi::TensorView grad_mask) {
  CheckDcnTensor(columns, "columns");
  CheckDcnTensor(input, "input");
  CheckDcnTensor(offset, "offset");
  CheckDcnTensor(mask, "mask");
  CheckDcnTensor(grad_offset, "grad_offset");
  CheckDcnTensor(grad_mask, "grad_mask");
  CheckSameDtype(input, "input", columns, "columns");
  CheckSameDtype(input, "input", offset, "offset");
  CheckSameDtype(input, "input", mask, "mask");
  CheckSameDtype(input, "input", grad_offset, "grad_offset");
  CheckSameDtype(input, "input", grad_mask, "grad_mask");
  const auto geo = CheckNchwShapes(input, offset, mask, kh, kw, ph, pw, sh, sw, dh, dw, dg);
  TVM_FFI_ICHECK_EQ(columns.numel(), geo.channels * kh * kw * geo.batch * geo.height_col * geo.width_col);
  TVM_FFI_ICHECK_EQ(grad_offset.numel(), offset.numel());
  TVM_FFI_ICHECK_EQ(grad_mask.numel(), mask.numel());
  if (grad_offset.numel() == 0) return;
  MUSPACE_DCN_DISPATCH(
      input,
      LaunchNchwCoord<scalar_t>(
          columns, input, offset, mask, grad_offset, grad_mask, geo, kh, kw, ph, pw, sh, sw, dh, dw, dg));
  CheckLastError("modulated_deform_conv_col2im_coord");
}

void modulated_deform_conv_im2col_nhwc(ffi::TensorView input,
                                       ffi::TensorView offset,
                                       int64_t         kh,
                                       int64_t         kw,
                                       int64_t         ph,
                                       int64_t         pw,
                                       int64_t         sh,
                                       int64_t         sw,
                                       int64_t         dh,
                                       int64_t         dw,
                                       ffi::TensorView columns) {
  CheckDcnTensor(input, "input");
  CheckDcnTensor(offset, "offset");
  CheckDcnTensor(columns, "columns");
  CheckSameDtype(input, "input", offset, "offset");
  CheckSameDtype(input, "input", columns, "columns");
  const auto geo = CheckNhwcShapes(input, offset, kh, kw, ph, pw, sh, sw, dh, dw);
  TVM_FFI_ICHECK_EQ(columns.numel(), geo.batch * geo.height_col * geo.width_col * kh * kw * geo.channels);
  if (columns.numel() == 0) return;
  MUSPACE_DCN_DISPATCH(input, LaunchNhwcIm2Col<scalar_t>(input, offset, columns, geo, ph, pw, sh, sw, dh, dw));
  CheckLastError("modulated_deform_conv_im2col_nhwc");
}

void modulated_deform_conv_col2im_nhwc(ffi::TensorView grad_columns,
                                       ffi::TensorView input,
                                       ffi::TensorView offset,
                                       int64_t         kh,
                                       int64_t         kw,
                                       int64_t         ph,
                                       int64_t         pw,
                                       int64_t         sh,
                                       int64_t         sw,
                                       int64_t         dh,
                                       int64_t         dw,
                                       ffi::TensorView grad_input,
                                       ffi::TensorView grad_offset,
                                       ffi::TensorView grad_mask) {
  CheckDcnTensor(grad_columns, "grad_columns");
  CheckDcnTensor(input, "input");
  CheckDcnTensor(offset, "offset");
  CheckSameDtype(input, "input", grad_columns, "grad_columns");
  CheckSameDtype(input, "input", offset, "offset");
  const auto geo = CheckNhwcShapes(input, offset, kh, kw, ph, pw, sh, sw, dh, dw);
  TVM_FFI_ICHECK(same_dtype(grad_input.dtype(), kDLFloat32) && same_dtype(grad_offset.dtype(), kDLFloat32) &&
                 same_dtype(grad_mask.dtype(), kDLFloat32))
      << "NHWC gradients must use float32";
  TVM_FFI_ICHECK(grad_input.IsContiguous() && grad_offset.IsContiguous() && grad_mask.IsContiguous())
      << "NHWC gradients must be contiguous";
  TVM_FFI_ICHECK_EQ(grad_columns.numel(), geo.batch * geo.height_col * geo.width_col * kh * kw * geo.channels);
  TVM_FFI_ICHECK_EQ(grad_input.numel(), input.numel());
  TVM_FFI_ICHECK_EQ(grad_offset.numel(), geo.batch * geo.height_col * geo.width_col * 2 * kh * kw);
  TVM_FFI_ICHECK_EQ(grad_mask.numel(), geo.batch * geo.height_col * geo.width_col * kh * kw);
  if (grad_columns.numel() == 0) return;
  MUSPACE_DCN_DISPATCH(
      input, LaunchNhwcCol2Im<scalar_t>(grad_columns, input, offset, grad_input, grad_offset, grad_mask, geo, ph, pw));
  CheckLastError("modulated_deform_conv_col2im_nhwc");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(modulated_deform_conv_im2col, modulated_deform_conv_im2col);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(modulated_deform_conv_col2im, modulated_deform_conv_col2im);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(modulated_deform_conv_col2im_coord, modulated_deform_conv_col2im_coord);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(modulated_deform_conv_im2col_nhwc, modulated_deform_conv_im2col_nhwc);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(modulated_deform_conv_col2im_nhwc, modulated_deform_conv_col2im_nhwc);
