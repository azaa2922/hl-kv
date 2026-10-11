#pragma once

// hl_quant — precise definitions of the KV storage formats under evaluation.
//
// The numeric code of each QuantFormat is the value stored in the 4-bit
// format field of hl_allocator's BlockExecDescriptor (hl::alloc::FormatId).
//
// Decoding semantics (the reference every CPU/GPU implementation must match):
//
//   FP16        x = decode_fp16(q)                       no parameters
//   FP8_E4M3    x = decode_e4m3(q) * scale               bias must be 0
//   FP8_E5M2    x = decode_e5m2(q) * scale               bias must be 0
//   INT4_PACKED x = fma(float(q), scale, bias)           q in [0, 15]
//   INT2_PACKED x = fma(float(q), scale, bias)           q in [0, 3]
//
// All arithmetic is binary32 with round-to-nearest-even; the INT formats use a
// single-rounding fused multiply-add so CPU and GPU (__fmaf_rn) results can be
// compared bit-exactly. (scale, bias) are per group of `group_size`
// consecutive elements. Symmetric signed INT schemes are expressed with
// bias = -2^(bits-1) * scale; min/max asymmetric schemes with bias = min.
//
// Packing (sub-byte formats), element i of a buffer:
//   INT4: byte i/2, bits [4*(i%2), 4*(i%2)+3]   (element 2k in the low nibble)
//   INT2: byte i/4, bits [2*(i%4), 2*(i%4)+1]   (element 4k in bits 0..1)
// Groups must start on a byte boundary (group_size a multiple of the elements
// per byte), so one packed byte never needs two parameter sets. Padding bits
// in a trailing partial byte are written as zero and ignored on decode.
//
// FP16 is stored little-endian (2 bytes/element). FP8 is one byte/element.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace hl::quant {

enum class QuantFormat : std::uint8_t {
  kFp16 = 0,        ///< IEEE 754 binary16
  kFp8E4M3 = 1,     ///< OCP OFP8 E4M3 ("E4M3FN"): bias 7, no Inf, NaN = S.1111.111, max 448
  kFp8E5M2 = 2,     ///< OCP OFP8 E5M2: bias 15, IEEE-style Inf/NaN, max 57344
  kInt4Packed = 3,  ///< unsigned 4-bit codes, 2 per byte
  kInt2Packed = 4,  ///< unsigned 2-bit codes, 4 per byte
};

inline constexpr std::size_t kNumQuantFormats = 5;
static_assert(kNumQuantFormats <= 15, "format codes must fit hl_allocator's 4-bit FormatId");

enum class ElementKind : std::uint8_t { kFloat, kUnsignedInt };

enum class QuantStatus : std::uint8_t {
  kOk = 0,
  kInvalidArgument,   ///< bad group size, parameter count, non-finite input/param, ...
  kBufferTooSmall,    ///< packed or output span shorter than required
  kUnsupportedFormat, ///< value outside QuantFormat
};

[[nodiscard]] const char* to_string(QuantStatus status) noexcept;

/// Per-group post-decode transform; see the table at the top of this file.
struct AffineParams {
  float scale;
  float bias;
};
static_assert(sizeof(AffineParams) == 8);

struct FormatTraits {
  std::string_view name;
  ElementKind kind;
  std::uint8_t bits_per_element;
  std::uint8_t exponent_bits;  ///< float formats only
  std::uint8_t mantissa_bits;  ///< float formats only (explicit bits)
  std::int8_t exponent_bias;   ///< float formats only
  bool has_infinity;           ///< float formats only
  bool uses_params;            ///< needs AffineParams per group
  bool uses_bias;              ///< AffineParams::bias participates in decode
  std::uint32_t max_code;      ///< INT: largest code; float: magnitude code of max finite
  float max_finite;            ///< largest finite decoded magnitude, before scaling
};

[[nodiscard]] constexpr bool is_valid(QuantFormat f) noexcept {
  return static_cast<std::size_t>(f) < kNumQuantFormats;
}

/// Precondition: is_valid(f).
[[nodiscard]] constexpr FormatTraits format_traits(QuantFormat f) noexcept {
  switch (f) {
    case QuantFormat::kFp16:
      return {"FP16", ElementKind::kFloat, 16, 5, 10, 15, true, false, false, 0x7BFFu, 65504.0f};
    case QuantFormat::kFp8E4M3:
      return {"FP8_E4M3", ElementKind::kFloat, 8, 4, 3, 7, false, true, false, 0x7Eu, 448.0f};
    case QuantFormat::kFp8E5M2:
      return {"FP8_E5M2", ElementKind::kFloat, 8, 5, 2, 15, true, true, false, 0x7Bu, 57344.0f};
    case QuantFormat::kInt4Packed:
      return {"INT4_PACKED", ElementKind::kUnsignedInt, 4, 0, 0, 0, false, true, true, 15u, 15.0f};
    case QuantFormat::kInt2Packed:
      return {"INT2_PACKED", ElementKind::kUnsignedInt, 2, 0, 0, 0, false, true, true, 3u, 3.0f};
  }
  return {"INVALID", ElementKind::kFloat, 0, 0, 0, 0, false, false, false, 0u, 0.0f};
}

/// Largest element count accepted by the reference APIs (keeps byte math in range).
inline constexpr std::size_t kMaxElements = std::numeric_limits<std::size_t>::max() / 2;

/// Elements per packed byte for sub-byte formats; 1 for FP8; 0 for FP16.
[[nodiscard]] constexpr std::size_t elements_per_byte(QuantFormat f) noexcept {
  const std::size_t bits = format_traits(f).bits_per_element;
  return bits == 0 || bits > 8 ? 0 : 8 / bits;
}

/// Bytes needed to store n elements. Precondition: n <= kMaxElements.
[[nodiscard]] constexpr std::size_t packed_bytes(QuantFormat f, std::size_t n) noexcept {
  const std::size_t bits = format_traits(f).bits_per_element;
  if (bits >= 8) return n * (bits / 8);
  const std::size_t per = 8 / bits;
  return n / per + (n % per != 0 ? 1 : 0);
}

/// Number of parameter groups for n elements (0 when group_size == 0).
[[nodiscard]] constexpr std::size_t num_groups(std::size_t n, std::size_t group_size) noexcept {
  return group_size == 0 ? 0 : n / group_size + (n % group_size != 0 ? 1 : 0);
}

/// FP16 takes no parameters (group_size must be 0); every other format needs
/// group_size >= 1 and a multiple of elements_per_byte().
[[nodiscard]] constexpr bool is_valid_group_size(QuantFormat f, std::size_t group_size) noexcept {
  if (!is_valid(f)) return false;
  if (!format_traits(f).uses_params) return group_size == 0;
  return group_size > 0 && group_size % elements_per_byte(f) == 0;
}

}  // namespace hl::quant
