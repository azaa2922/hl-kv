#include "hl_quant/cpu_dequant.hpp"

#include <bit>
#include <cmath>

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
    case QuantFormat::kInt2Packed:
      return (std::to_integer<std::uint32_t>(packed[i / 4]) >> (2 * (i % 4))) & 0x3u;
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

QuantStatus dequantize_reference(QuantFormat format, std::span<const std::byte> packed,
                                 std::size_t num_elements, std::span<const AffineParams> params,
                                 std::size_t group_size, std::span<float> out) noexcept {
  if (!is_valid(format)) return QuantStatus::kUnsupportedFormat;
  if (num_elements > kMaxElements) return QuantStatus::kInvalidArgument;
  if (!is_valid_group_size(format, group_size)) return QuantStatus::kInvalidArgument;
  if (params.size() != num_groups(num_elements, group_size)) return QuantStatus::kInvalidArgument;

  const FormatTraits traits = format_traits(format);
  for (const AffineParams& p : params) {
    if (!std::isfinite(p.scale) || !std::isfinite(p.bias)) return QuantStatus::kInvalidArgument;
    if (!traits.uses_bias && p.bias != 0.0f) return QuantStatus::kInvalidArgument;
  }
  if (packed.size() < packed_bytes(format, num_elements)) return QuantStatus::kBufferTooSmall;
  if (out.size() < num_elements) return QuantStatus::kBufferTooSmall;

  switch (format) {
    case QuantFormat::kFp16:
      for (std::size_t i = 0; i < num_elements; ++i) {
        out[i] = decode_minifloat(read_code(format, packed, i), detail::kFp16Spec);
      }
      break;
    case QuantFormat::kFp8E4M3:
    case QuantFormat::kFp8E5M2: {
      const MiniFloatSpec& spec =
          format == QuantFormat::kFp8E4M3 ? detail::kE4M3Spec : detail::kE5M2Spec;
      for (std::size_t i = 0; i < num_elements; ++i) {
        out[i] = decode_minifloat(read_code(format, packed, i), spec) * params[i / group_size].scale;
      }
      break;
    }
    case QuantFormat::kInt4Packed:
    case QuantFormat::kInt2Packed:
      for (std::size_t i = 0; i < num_elements; ++i) {
        const AffineParams& p = params[i / group_size];
        out[i] = std::fma(static_cast<float>(read_code(format, packed, i)), p.scale, p.bias);
      }
      break;
  }
  return QuantStatus::kOk;
}

}  // namespace hl::quant
