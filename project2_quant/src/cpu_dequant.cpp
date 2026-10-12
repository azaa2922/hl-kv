#include "hl_quant/cpu_dequant.hpp"

#include <bit>
#include <cmath>

#include "hl_quant/cpu_quant.hpp"
#include "minifloat_spec.hpp"

namespace hl::quant {
namespace {

using detail::MiniFloatSpec;

constexpr std::uint32_t kF32QuietNan = 0x7FC0'0000u;
constexpr std::uint32_t kF32Inf = 0x7F80'0000u;

// Builds the binary32 bit pattern directly (no libm), so the decode is exact
// by construction: every value of the supported formats has at most 11
// significant bits and an exponent inside the binary32 normal range.
float decode_minifloat(std::uint32_t code, const MiniFloatSpec& s) noexcept {
  const std::uint32_t width = s.exp_bits + s.man_bits;
  const std::uint32_t sign_bit = ((code >> width) & 1u) << 31;
  const std::uint32_t mag = code & ((1u << width) - 1u);
  const std::uint32_t exp_field = mag >> s.man_bits;
  const std::uint32_t man_field = mag & ((1u << s.man_bits) - 1u);
  const std::uint32_t exp_all_ones = (1u << s.exp_bits) - 1u;

  if (s.ieee_special && exp_field == exp_all_ones) {
    return std::bit_cast<float>(sign_bit | (man_field == 0 ? kF32Inf : kF32QuietNan));
  }
  if (!s.ieee_special && mag > s.max_finite) {  // E4M3FN: only S.1111.111
    return std::bit_cast<float>(sign_bit | kF32QuietNan);
  }
  if (exp_field == 0) {
    if (man_field == 0) return std::bit_cast<float>(sign_bit);  // ±0
    // Subnormal: man_field * 2^(1 - bias - man_bits). Normalize around the
    // leading one at bit position p.
    const auto p = static_cast<std::int32_t>(std::bit_width(man_field)) - 1;
    const std::int32_t unbiased = 1 - s.bias - static_cast<std::int32_t>(s.man_bits) + p;
    const std::uint32_t frac = (man_field - (1u << p)) << (23 - p);
    return std::bit_cast<float>(sign_bit | (static_cast<std::uint32_t>(unbiased + 127) << 23) |
                                frac);
  }
  const std::int32_t unbiased = static_cast<std::int32_t>(exp_field) - s.bias;
  return std::bit_cast<float>(sign_bit | (static_cast<std::uint32_t>(unbiased + 127) << 23) |
                              (man_field << (23 - s.man_bits)));
}

std::uint32_t read_code(QuantFormat f, std::span<const std::byte> packed, std::size_t i) noexcept {
  switch (f) {
    case QuantFormat::kFp16:
      return std::to_integer<std::uint32_t>(packed[2 * i]) |
             (std::to_integer<std::uint32_t>(packed[2 * i + 1]) << 8);
    case QuantFormat::kFp8E4M3:
    case QuantFormat::kFp8E5M2:
      return std::to_integer<std::uint32_t>(packed[i]);
    case QuantFormat::kInt4Packed:
      return (std::to_integer<std::uint32_t>(packed[i / 2]) >> (4 * (i % 2))) & 0xFu;
  }
  return 0;
}

}  // namespace

float decode_fp16(std::uint16_t bits) noexcept { return decode_minifloat(bits, detail::kFp16Spec); }

float decode_fp8_e4m3(std::uint8_t bits) noexcept {
  return decode_minifloat(bits, detail::kE4M3Spec);
}

float decode_fp8_e5m2(std::uint8_t bits) noexcept {
  return decode_minifloat(bits, detail::kE5M2Spec);
}

namespace {

QuantStatus check_args(QuantFormat format, std::size_t packed_size, std::size_t num_elements,
                       std::span<const AffineParams> params, std::size_t group_size) noexcept {
  if (!is_valid(format)) return QuantStatus::kUnsupportedFormat;
  if (num_elements > kMaxElements) return QuantStatus::kInvalidArgument;
  if (!is_valid_group_size(format, group_size)) return QuantStatus::kInvalidArgument;
  if (params.size() != num_groups(num_elements, group_size)) return QuantStatus::kInvalidArgument;
  const FormatTraits traits = format_traits(format);
  for (const AffineParams& p : params) {
    if (!std::isfinite(p.scale) || !std::isfinite(p.bias)) return QuantStatus::kInvalidArgument;
    if (!traits.uses_bias && p.bias != 0.0f) return QuantStatus::kInvalidArgument;
  }
  if (packed_size < packed_bytes(format, num_elements)) return QuantStatus::kBufferTooSmall;
  return QuantStatus::kOk;
}

// The reference binary32 value of element i (arguments already validated).
float reference_value(QuantFormat format, std::span<const std::byte> packed, std::size_t i,
                      std::span<const AffineParams> params, std::size_t group_size) noexcept {
  const std::uint32_t code = read_code(format, packed, i);
  switch (format) {
    case QuantFormat::kFp16:
      return decode_minifloat(code, detail::kFp16Spec);
    case QuantFormat::kFp8E4M3:
      return decode_minifloat(code, detail::kE4M3Spec) * params[i / group_size].scale;
    case QuantFormat::kFp8E5M2:
      return decode_minifloat(code, detail::kE5M2Spec) * params[i / group_size].scale;
    case QuantFormat::kInt4Packed: {
      const AffineParams& p = params[i / group_size];
      return std::fma(static_cast<float>(code), p.scale, p.bias);
    }
  }
  return 0.0f;
}

}  // namespace

QuantStatus dequantize_reference(QuantFormat format, std::span<const std::byte> packed,
                                 std::size_t num_elements, std::span<const AffineParams> params,
                                 std::size_t group_size, std::span<float> out) noexcept {
  if (const QuantStatus s = check_args(format, packed.size(), num_elements, params, group_size);
      s != QuantStatus::kOk) {
    return s;
  }
  if (out.size() < num_elements) return QuantStatus::kBufferTooSmall;
  for (std::size_t i = 0; i < num_elements; ++i) {
    out[i] = reference_value(format, packed, i, params, group_size);
  }
  return QuantStatus::kOk;
}

QuantStatus dequantize_reference_to_fp16(QuantFormat format, std::span<const std::byte> packed,
                                         std::size_t num_elements,
                                         std::span<const AffineParams> params,
                                         std::size_t group_size,
                                         std::span<std::uint16_t> out) noexcept {
  if (const QuantStatus s = check_args(format, packed.size(), num_elements, params, group_size);
      s != QuantStatus::kOk) {
    return s;
  }
  if (out.size() < num_elements) return QuantStatus::kBufferTooSmall;
  for (std::size_t i = 0; i < num_elements; ++i) {
    out[i] = encode_fp16(reference_value(format, packed, i, params, group_size));
  }
  return QuantStatus::kOk;
}

}  // namespace hl::quant
