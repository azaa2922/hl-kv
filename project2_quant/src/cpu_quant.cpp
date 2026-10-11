#include "hl_quant/cpu_quant.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

#include "minifloat_spec.hpp"

namespace hl::quant {
namespace {

using detail::MiniFloatSpec;

// Exact binary32 -> small-float rounding with integer arithmetic.
//
// The input magnitude is written as sig * 2^exp2 (sig < 2^24). The target
// spacing ("quantum") at the input's binade is 2^(E - man_bits), where E is
// the input exponent clamped below at the target's minimum normal exponent
// (this makes subnormal targets fall out of the same formula). sig is shifted
// down to that quantum with round-half-to-even on the discarded bits.
//
// The code is then ((E - emin) << man_bits) + rounded: for normals `rounded`
// carries the implicit leading one into the exponent field, for subnormals it
// is < 2^man_bits and lands in exponent field 0, and a rounding carry
// (rounded == 2^(man_bits+1)) increments the exponent automatically.
std::uint32_t encode_minifloat(float x, const MiniFloatSpec& s) noexcept {
  const std::uint32_t f = std::bit_cast<std::uint32_t>(x);
  const std::uint32_t sign_out = (f >> 31) << (s.exp_bits + s.man_bits);
  const std::uint32_t abs = f & 0x7FFF'FFFFu;

  if (abs > 0x7F80'0000u) return sign_out | s.canonical_nan;
  if (abs == 0x7F80'0000u) return sign_out | (s.saturate ? s.max_finite : s.inf);

  const std::int32_t emin = 1 - s.bias;
  std::int32_t e = 0;
  std::int32_t exp2 = 0;
  std::uint64_t sig = 0;
  if (abs >= 0x0080'0000u) {  // binary32 normal
    e = static_cast<std::int32_t>(abs >> 23) - 127;
    sig = (abs & 0x007F'FFFFu) | 0x0080'0000u;
    exp2 = e - 23;
  } else {  // binary32 subnormal or zero: below every target's emin
    e = emin;
    sig = abs;
    exp2 = -149;
  }
  const std::int32_t binade = std::max(e, emin);
  const std::int32_t quantum = binade - static_cast<std::int32_t>(s.man_bits);
  const std::int32_t shift = quantum - exp2;  // >= 1 because man_bits < 23

  std::uint64_t rounded = 0;
  if (shift < 40) {  // otherwise sig / 2^shift < 2^-16 and rounds to zero
    const auto sh = static_cast<std::uint32_t>(shift);
    rounded = sig >> sh;
    const std::uint64_t rem = sig & ((std::uint64_t{1} << sh) - 1);
    const std::uint64_t half = std::uint64_t{1} << (sh - 1);
    if (rem > half || (rem == half && (rounded & 1u) != 0)) ++rounded;
  }
  const std::uint64_t code =
      (static_cast<std::uint64_t>(binade - emin) << s.man_bits) + rounded;
  if (code > s.max_finite) return sign_out | (s.saturate ? s.max_finite : s.inf);
  return sign_out | static_cast<std::uint32_t>(code);
}

void write_code(QuantFormat f, std::span<std::byte> packed, std::size_t i,
                std::uint32_t code) noexcept {
  switch (f) {
    case QuantFormat::kFp16:
      packed[2 * i] = static_cast<std::byte>(code & 0xFFu);
      packed[2 * i + 1] = static_cast<std::byte>((code >> 8) & 0xFFu);
      break;
    case QuantFormat::kFp8E4M3:
    case QuantFormat::kFp8E5M2:
      packed[i] = static_cast<std::byte>(code & 0xFFu);
      break;
    case QuantFormat::kInt4Packed:
      packed[i / 2] |= static_cast<std::byte>((code & 0xFu) << (4 * (i % 2)));
      break;
    case QuantFormat::kInt2Packed:
      packed[i / 4] |= static_cast<std::byte>((code & 0x3u) << (2 * (i % 4)));
      break;
  }
}

}  // namespace

std::uint16_t encode_fp16(float x) noexcept {
  return static_cast<std::uint16_t>(encode_minifloat(x, detail::kFp16Spec));
}

std::uint8_t encode_fp8_e4m3(float x) noexcept {
  return static_cast<std::uint8_t>(encode_minifloat(x, detail::kE4M3Spec));
}

std::uint8_t encode_fp8_e5m2(float x) noexcept {
  return static_cast<std::uint8_t>(encode_minifloat(x, detail::kE5M2Spec));
}

QuantStatus quantize_reference(QuantFormat format, std::span<const float> in,
                               std::size_t group_size, std::span<AffineParams> params_out,
                               std::span<std::byte> packed_out) noexcept {
  if (!is_valid(format)) return QuantStatus::kUnsupportedFormat;
  const std::size_t n = in.size();
  if (n > kMaxElements) return QuantStatus::kInvalidArgument;
  if (!is_valid_group_size(format, group_size)) return QuantStatus::kInvalidArgument;
  const std::size_t groups = num_groups(n, group_size);
  if (params_out.size() != groups) return QuantStatus::kInvalidArgument;

  const FormatTraits traits = format_traits(format);
  // Validate everything before writing so failures leave outputs untouched.
  if (traits.uses_params) {
    for (const float x : in) {
      if (!std::isfinite(x)) return QuantStatus::kInvalidArgument;
    }
    if (traits.kind == ElementKind::kUnsignedInt) {
      for (std::size_t g = 0; g < groups; ++g) {
        const auto first = in.begin() + static_cast<std::ptrdiff_t>(g * group_size);
        const auto last = in.begin() + static_cast<std::ptrdiff_t>(std::min(n, (g + 1) * group_size));
        const auto [lo, hi] = std::minmax_element(first, last);
        if (!std::isfinite(*hi - *lo)) return QuantStatus::kInvalidArgument;
      }
    }
  }
  const std::size_t bytes = packed_bytes(format, n);
  if (packed_out.size() < bytes) return QuantStatus::kBufferTooSmall;

  std::fill_n(packed_out.begin(), bytes, std::byte{0});

  if (format == QuantFormat::kFp16) {
    for (std::size_t i = 0; i < n; ++i) write_code(format, packed_out, i, encode_fp16(in[i]));
    return QuantStatus::kOk;
  }

  for (std::size_t g = 0; g < groups; ++g) {
    const std::size_t begin = g * group_size;
    const std::size_t end = std::min(n, begin + group_size);
    const auto first = in.begin() + static_cast<std::ptrdiff_t>(begin);
    const auto last = in.begin() + static_cast<std::ptrdiff_t>(end);

    if (traits.kind == ElementKind::kFloat) {  // FP8
      float amax = 0.0f;
      for (auto it = first; it != last; ++it) amax = std::max(amax, std::fabs(*it));
      float scale = 1.0f;
      if (amax > 0.0f) {
        scale = std::max(amax / traits.max_finite, std::numeric_limits<float>::min());
      }
      params_out[g] = AffineParams{scale, 0.0f};
      const bool e4m3 = format == QuantFormat::kFp8E4M3;
      for (std::size_t i = begin; i < end; ++i) {
        const float y = in[i] / scale;
        write_code(format, packed_out, i, e4m3 ? encode_fp8_e4m3(y) : encode_fp8_e5m2(y));
      }
    } else {  // INT4 / INT2
      const auto [lo_it, hi_it] = std::minmax_element(first, last);
      const float lo = *lo_it;
      const float qmax = static_cast<float>(traits.max_code);
      const float scale = (*hi_it - lo) / qmax;
      params_out[g] = AffineParams{scale, lo};
      for (std::size_t i = begin; i < end; ++i) {
        std::uint32_t q = 0;
        if (scale > 0.0f) {
          // nearbyint uses the current rounding mode; the default
          // (round-to-nearest-even) is assumed and documented.
          const float r = std::nearbyint((in[i] - lo) / scale);
          q = static_cast<std::uint32_t>(std::clamp(r, 0.0f, qmax));
        }
        write_code(format, packed_out, i, q);
      }
    }
  }
  return QuantStatus::kOk;
}

}  // namespace hl::quant
