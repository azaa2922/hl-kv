#pragma once

// hl_quant GPU entry points. Available only when the library is built with the
// CUDA toolchain (target hl::quant_cuda; CMake option HLKV_CUDA).
//
// FP8 -> FP16 dequantization, element i of group g = i / group_size:
//
//   out[i] = fp16_rne( decode_fp8(in[i]) * params[g].scale )
//
// i.e. the binary32 value of dequantize_reference() rounded to nearest-even
// binary16 — exactly dequantize_reference_to_fp16() (cpu_dequant.hpp), bit for
// bit, except that NaN payloads are not specified (NaN stays NaN). The FP8
// bias is 0 by definition (quant_types.hpp) and is not read on the device.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>

#include "hl_quant/quant_types.hpp"

namespace hl::quant::cuda {

struct DeviceStatus {
  QuantStatus status = QuantStatus::kOk;  ///< argument errors, or kCudaError
  cudaError_t cuda_error = cudaSuccess;   ///< set when status == kCudaError
  [[nodiscard]] bool ok() const noexcept { return status == QuantStatus::kOk; }
};

/// Enqueues FP8 -> FP16 dequantization of `n` elements on `stream` (asynchronous;
/// errors of the running kernel surface at the next synchronization).
///
///   d_in      n FP8 codes (device memory)
///   d_params  num_groups(n, group_size) AffineParams (device memory; .scale only)
///   d_out     n binary16 values (device memory, raw bit patterns)
///
/// Requirements (otherwise nothing is launched):
///   format is kFp8E4M3 or kFp8E5M2                  else kUnsupportedFormat
///   group_size >= 1, n <= kMaxElements              else kInvalidArgument
///   non-null pointers when n > 0                    else kInvalidArgument
/// n == 0 is a no-op. Any alignment is accepted: 8-byte-aligned input with
/// 16-byte-aligned output and (group_size % 8 == 0 or group_size >= n) take the
/// vectorized path (8 elements per thread); everything else, including the
/// tail, takes the scalar path. Both paths compute identical values.
/// A launch failure returns kCudaError with the CUDA error code.
[[nodiscard]] DeviceStatus dequantize_fp8_to_fp16(QuantFormat format, const std::byte* d_in,
                                                  std::size_t n, const AffineParams* d_params,
                                                  std::size_t group_size, std::uint16_t* d_out,
                                                  cudaStream_t stream) noexcept;

/// Which code path of the kernel to run on the host.
enum class HostPath : std::uint8_t {
  kScalar,  ///< per-element conversion (the kernel's scalar/tail path)
  kVector,  ///< 8-element chunks + scalar tail, as the vectorized kernel does
};

/// Runs the kernel's per-element arithmetic on the CPU: the same __host__
/// __device__ functions, compiled for the host by nvcc (cuda_fp8.h /
/// cuda_fp16.h host implementations). Validates the kernel's arithmetic where
/// no GPU is available; it cannot validate device instructions, indexing or
/// launch geometry. n = in.size(). Same requirements as dequantize_fp8_to_fp16,
/// plus params.size() == num_groups(n, group_size) and out.size() >= n
/// (else kInvalidArgument / kBufferTooSmall). kVector falls back to kScalar
/// when the group condition of the vectorized path does not hold.
[[nodiscard]] QuantStatus emulate_fp8_to_fp16_on_host(QuantFormat format,
                                                      std::span<const std::byte> in,
                                                      std::span<const AffineParams> params,
                                                      std::size_t group_size,
                                                      std::span<std::uint16_t> out,
                                                      HostPath path) noexcept;

}  // namespace hl::quant::cuda
