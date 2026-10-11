#pragma once

// Internal: bit-level description of the small floating-point formats. Shared
// by the reference encoder (cpu_quant.cpp) and decoder (cpu_dequant.cpp).
// Not installed / not part of the public API.

#include <cstdint>

namespace hl::quant::detail {

struct MiniFloatSpec {
  std::uint32_t exp_bits;
  std::uint32_t man_bits;
  std::int32_t bias;
  bool ieee_special;           ///< all-ones exponent reserved for Inf/NaN (FP16, E5M2)
  bool saturate;               ///< encode: overflow and ±Inf -> ±max_finite (else ±Inf)
  std::uint32_t max_finite;    ///< magnitude code of the largest finite value
  std::uint32_t canonical_nan; ///< magnitude code written for NaN
  std::uint32_t inf;           ///< magnitude code of Inf (ieee_special only)
};

inline constexpr MiniFloatSpec kFp16Spec{5, 10, 15, true, false, 0x7BFFu, 0x7E00u, 0x7C00u};
inline constexpr MiniFloatSpec kE4M3Spec{4, 3, 7, false, true, 0x7Eu, 0x7Fu, 0u};
inline constexpr MiniFloatSpec kE5M2Spec{5, 2, 15, true, true, 0x7Bu, 0x7Fu, 0x7Cu};

}  // namespace hl::quant::detail
