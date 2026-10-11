#pragma once

// CPU reference quantization. Produces packed data + per-group parameters that
// dequantize_reference() consumes. Used to generate test vectors from raw FP32
// targets and, later, as the reference for GPU quantization kernels.
//
// Parameter policy (one reasonable choice, not the only valid one):
//   FP8:  scale = max(amax / max_finite, FLT_MIN) (1 if amax == 0), bias = 0,
//         q = encode(x / scale)  — IEEE division, RNE, saturating.
//   INT:  scale = (max - min) / max_code, bias = min,
//         q = clamp(round_half_even((x - min) / scale), 0, max_code);
//         a constant group (max == min) gets scale = 0, q = 0 (exact).

#include <cstddef>
#include <cstdint>
#include <span>

#include "hl_quant/quant_types.hpp"

namespace hl::quant {

/// binary32 -> binary16, round-to-nearest-even. Overflow -> ±Inf, NaN -> quiet NaN (sign kept).
[[nodiscard]] std::uint16_t encode_fp16(float x) noexcept;

/// binary32 -> E4M3, round-to-nearest-even, saturating: |x| beyond 448 and ±Inf
/// map to ±448; NaN maps to S.1111.111.
[[nodiscard]] std::uint8_t encode_fp8_e4m3(float x) noexcept;

/// binary32 -> E5M2, round-to-nearest-even, saturating: |x| beyond 57344 and ±Inf
/// map to ±57344; NaN maps to S.11111.11.
[[nodiscard]] std::uint8_t encode_fp8_e5m2(float x) noexcept;

/// Quantizes `in` into `packed_out` and `params_out`.
///
/// Requirements (otherwise the listed status is returned and outputs are untouched):
///   is_valid(format)                                         else kUnsupportedFormat
///   in.size() <= kMaxElements                                else kInvalidArgument
///   is_valid_group_size(format, group_size)                  else kInvalidArgument
///   params_out.size() == num_groups(in.size(), group_size)   else kInvalidArgument
///   FP8/INT: every input finite, every INT group range finite else kInvalidArgument
///   packed_out.size() >= packed_bytes(format, in.size())     else kBufferTooSmall
/// FP16 accepts non-finite inputs (they are representable).
[[nodiscard]] QuantStatus quantize_reference(QuantFormat format, std::span<const float> in,
                                             std::size_t group_size,
                                             std::span<AffineParams> params_out,
                                             std::span<std::byte> packed_out) noexcept;

}  // namespace hl::quant
