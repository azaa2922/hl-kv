#pragma once

// CPU reference dequantization — the numerical ground truth for hl_quant GPU
// kernels. Correctness over speed: scalar code, no SIMD, no fast-math, every
// value decoded from its bit pattern with exact integer arithmetic.

#include <cstddef>
#include <cstdint>
#include <span>

#include "hl_quant/quant_types.hpp"

namespace hl::quant {

/// Exact binary16 -> binary32 (every FP16 value is representable in FP32).
/// Inf maps to Inf; every NaN maps to a quiet NaN with the same sign.
[[nodiscard]] float decode_fp16(std::uint16_t bits) noexcept;

/// Exact OCP E4M3 (E4M3FN) -> binary32. S.1111.111 decodes to NaN.
[[nodiscard]] float decode_fp8_e4m3(std::uint8_t bits) noexcept;

/// Exact OCP E5M2 -> binary32. S.11111.00 decodes to Inf, S.11111.{01,10,11} to NaN.
[[nodiscard]] float decode_fp8_e5m2(std::uint8_t bits) noexcept;

/// Dequantizes `num_elements` values of `format` from `packed` into `out`
/// using the semantics in quant_types.hpp.
///
/// Requirements (otherwise the listed status is returned and `out` is untouched):
///   is_valid(format)                                         else kUnsupportedFormat
///   num_elements <= kMaxElements                             else kInvalidArgument
///   is_valid_group_size(format, group_size)                  else kInvalidArgument
///   params.size() == num_groups(num_elements, group_size)    else kInvalidArgument
///   every scale/bias finite; FP8 bias == 0                   else kInvalidArgument
///   packed.size() >= packed_bytes(format, num_elements)      else kBufferTooSmall
///   out.size() >= num_elements                               else kBufferTooSmall
[[nodiscard]] QuantStatus dequantize_reference(QuantFormat format,
                                               std::span<const std::byte> packed,
                                               std::size_t num_elements,
                                               std::span<const AffineParams> params,
                                               std::size_t group_size,
                                               std::span<float> out) noexcept;

/// Reference for kernels that emit binary16: the binary32 value of
/// dequantize_reference(), rounded to nearest-even binary16 (encode_fp16:
/// overflow -> ±Inf, NaN -> quiet NaN). This is two roundings by definition
/// (binary32 product, then binary16), which is what a GPU computing in fp32 and
/// converting with __float2half_rn produces. Same requirements and errors as
/// dequantize_reference(); `out` holds raw binary16 bit patterns.
[[nodiscard]] QuantStatus dequantize_reference_to_fp16(QuantFormat format,
                                                       std::span<const std::byte> packed,
                                                       std::size_t num_elements,
                                                       std::span<const AffineParams> params,
                                                       std::size_t group_size,
                                                       std::span<std::uint16_t> out) noexcept;

}  // namespace hl::quant
