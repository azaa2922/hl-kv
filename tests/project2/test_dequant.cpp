// CPU reference dequantization tests.
//
//  1. Exhaustive decode of every FP16 / E4M3 / E5M2 code against an independent
//     oracle evaluated in double precision with std::ldexp (the library builds
//     float bit patterns directly; the oracle uses the textbook formula).
//  2. Packing / parameter semantics on hand-constructed buffers.
//  3. Accuracy against raw FP32 targets: quantize -> dequantize must stay within
//     the analytical error bound of each format; error statistics are printed.
//  4. Argument validation (explicit boundaries, untouched outputs on error).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

#include "hl_quant/cpu_dequant.hpp"
#include "hl_quant/cpu_quant.hpp"
#include "hl_quant/quant_types.hpp"

namespace hl::quant {
namespace {

// ---- independent oracle -------------------------------------------------------------------

struct OracleSpec {
  int exp_bits;
  int man_bits;
  int bias;
  bool ieee_special;
};

// Returns the exact value of `code` as a double; NaN for NaN encodings.
double oracle_decode(std::uint32_t code, const OracleSpec& s) {
  const std::uint32_t sign = (code >> (s.exp_bits + s.man_bits)) & 1u;
  const auto e = static_cast<int>((code >> s.man_bits) & ((1u << s.exp_bits) - 1u));
  const auto m = static_cast<int>(code & ((1u << s.man_bits) - 1u));
  const int e_max = (1 << s.exp_bits) - 1;
  double v = 0.0;
  if (s.ieee_special && e == e_max) {
    v = m == 0 ? std::numeric_limits<double>::infinity() : std::numeric_limits<double>::quiet_NaN();
  } else if (!s.ieee_special && e == e_max && m == (1 << s.man_bits) - 1) {
    v = std::numeric_limits<double>::quiet_NaN();  // E4M3FN: S.1111.111
  } else if (e == 0) {
    v = std::ldexp(static_cast<double>(m), 1 - s.bias - s.man_bits);
  } else {
    v = std::ldexp(static_cast<double>((1 << s.man_bits) + m), e - s.bias - s.man_bits);
  }
  return sign != 0 ? -v : v;
}

constexpr OracleSpec kOracleFp16{5, 10, 15, true};
constexpr OracleSpec kOracleE4M3{4, 3, 7, false};
constexpr OracleSpec kOracleE5M2{5, 2, 15, true};

void expect_same_value(float got, double want, std::uint32_t code) {
  if (std::isnan(want)) {
    EXPECT_TRUE(std::isnan(got)) << "code 0x" << std::hex << code;
    return;
  }
  EXPECT_EQ(static_cast<double>(got), want) << "code 0x" << std::hex << code;
  EXPECT_EQ(std::signbit(got), std::signbit(want)) << "code 0x" << std::hex << code;  // ±0
}

// ---- 1. exhaustive decode ----------------------------------------------------------------

TEST(DecodeExhaustive, Fp16AllCodes) {
  for (std::uint32_t c = 0; c <= 0xFFFFu; ++c) {
    expect_same_value(decode_fp16(static_cast<std::uint16_t>(c)), oracle_decode(c, kOracleFp16), c);
  }
}

TEST(DecodeExhaustive, Fp8E4M3AllCodes) {
  for (std::uint32_t c = 0; c <= 0xFFu; ++c) {
    expect_same_value(decode_fp8_e4m3(static_cast<std::uint8_t>(c)), oracle_decode(c, kOracleE4M3),
                      c);
  }
}

TEST(DecodeExhaustive, Fp8E5M2AllCodes) {
  for (std::uint32_t c = 0; c <= 0xFFu; ++c) {
    expect_same_value(decode_fp8_e5m2(static_cast<std::uint8_t>(c)), oracle_decode(c, kOracleE5M2),
                      c);
  }
}

TEST(DecodeKnownValues, SpecConstants) {
  // FP16
  EXPECT_EQ(decode_fp16(0x3C00), 1.0f);
  EXPECT_EQ(decode_fp16(0x7BFF), 65504.0f);
  EXPECT_EQ(decode_fp16(0x0001), std::ldexp(1.0f, -24));
  EXPECT_EQ(decode_fp16(0x0400), std::ldexp(1.0f, -14));
  EXPECT_TRUE(std::isinf(decode_fp16(0xFC00)) && decode_fp16(0xFC00) < 0);
  // E4M3 (OCP E4M3FN)
  EXPECT_EQ(decode_fp8_e4m3(0x7E), 448.0f);
  EXPECT_EQ(decode_fp8_e4m3(0xFE), -448.0f);
  EXPECT_EQ(decode_fp8_e4m3(0x38), 1.0f);
  EXPECT_EQ(decode_fp8_e4m3(0x08), std::ldexp(1.0f, -6));  // min normal
  EXPECT_EQ(decode_fp8_e4m3(0x01), std::ldexp(1.0f, -9));  // min subnormal
  EXPECT_EQ(decode_fp8_e4m3(0x78), 256.0f);                // exponent 15 is finite in E4M3FN
  EXPECT_TRUE(std::isnan(decode_fp8_e4m3(0x7F)));
  EXPECT_TRUE(std::isnan(decode_fp8_e4m3(0xFF)));
  // E5M2
  EXPECT_EQ(decode_fp8_e5m2(0x7B), 57344.0f);
  EXPECT_EQ(decode_fp8_e5m2(0x3C), 1.0f);
  EXPECT_EQ(decode_fp8_e5m2(0x01), std::ldexp(1.0f, -16));
  EXPECT_TRUE(std::isinf(decode_fp8_e5m2(0x7C)));
  EXPECT_TRUE(std::isnan(decode_fp8_e5m2(0x7D)));
  EXPECT_TRUE(std::isnan(decode_fp8_e5m2(0x7E)));
  EXPECT_TRUE(std::isnan(decode_fp8_e5m2(0x7F)));
}

// ---- 2. packing and parameter semantics ---------------------------------------------------------

std::vector<std::byte> bytes(std::initializer_list<unsigned> v) {
  std::vector<std::byte> out;
  for (const unsigned b : v) out.push_back(static_cast<std::byte>(b));
  return out;
}

TEST(DequantPacking, Int4LowNibbleFirst) {
  const std::vector<std::byte> packed = bytes({0x21, 0xF0, 0x0E});
  const std::array<AffineParams, 1> identity{{{1.0f, 0.0f}}};
  std::vector<float> out(5, -1.0f);
  ASSERT_EQ(dequantize_reference(QuantFormat::kInt4Packed, packed, 5, identity, 6, out),
            QuantStatus::kOk);
  EXPECT_EQ(out, (std::vector<float>{1, 2, 0, 15, 14}));  // high nibble of byte 2 is padding
}

TEST(DequantPacking, Int2LsbFirst) {
  const std::vector<std::byte> packed = bytes({0xE4, 0x1B, 0x03});  // 0b11100100, 0b00011011
  const std::array<AffineParams, 1> identity{{{1.0f, 0.0f}}};
  std::vector<float> out(9);
  ASSERT_EQ(dequantize_reference(QuantFormat::kInt2Packed, packed, 9, identity, 12, out),
            QuantStatus::kOk);
  EXPECT_EQ(out, (std::vector<float>{0, 1, 2, 3, 3, 2, 1, 0, 3}));
}

TEST(DequantPacking, Fp16IsLittleEndian) {
  const std::vector<std::byte> packed = bytes({0x00, 0x3C, 0x00, 0xC0});  // 1.0, -2.0
  std::vector<float> out(2);
  ASSERT_EQ(dequantize_reference(QuantFormat::kFp16, packed, 2, {}, 0, out), QuantStatus::kOk);
  EXPECT_EQ(out, (std::vector<float>{1.0f, -2.0f}));
}

TEST(DequantParams, IntUsesFusedAffinePerGroup) {
  // Two groups of 2 elements: codes {3, 5 | 7, 15}.
  const std::vector<std::byte> packed = bytes({0x53, 0xF7});
  const std::array<AffineParams, 2> params{{{0.5f, -1.0f}, {-0.25f, 2.0f}}};
  std::vector<float> out(4);
  ASSERT_EQ(dequantize_reference(QuantFormat::kInt4Packed, packed, 4, params, 2, out),
            QuantStatus::kOk);
  EXPECT_EQ(out, (std::vector<float>{0.5f, 1.5f, 0.25f, -1.75f}));

  // fma is a single rounding: pick q*scale whose separate rounding would differ.
  const float scale = 1.0f + std::ldexp(1.0f, -23);
  const float bias = -15.0f;
  const std::vector<std::byte> fifteen = bytes({0x0F});
  const std::array<AffineParams, 1> p{{{scale, bias}}};
  std::vector<float> one(1);
  ASSERT_EQ(dequantize_reference(QuantFormat::kInt4Packed, fifteen, 1, p, 2, one),
            QuantStatus::kOk);
  EXPECT_EQ(one[0], std::fma(15.0f, scale, bias));
  EXPECT_EQ(one[0], 15.0f * std::ldexp(1.0f, -23));  // exact; a two-step q*s+b would lose it
}

TEST(DequantParams, Fp8MultipliesByScale) {
  const std::vector<std::byte> packed = bytes({0x38, 0xBC, 0x01, 0x80});  // 1, -1.5, 2^-9, -0
  const std::array<AffineParams, 2> params{{{2.0f, 0.0f}, {0.5f, 0.0f}}};
  std::vector<float> out(4);
  ASSERT_EQ(dequantize_reference(QuantFormat::kFp8E4M3, packed, 4, params, 2, out),
            QuantStatus::kOk);
  EXPECT_EQ(out[0], 2.0f);
  EXPECT_EQ(out[1], -3.0f);
  EXPECT_EQ(out[2], std::ldexp(1.0f, -10));
  EXPECT_EQ(out[3], 0.0f);
  EXPECT_TRUE(std::signbit(out[3]));  // multiply-only keeps -0 (no bias add)
}

// ---- 3. accuracy against raw FP32 targets ---------------------------------------------------------

// Deterministic, platform-independent generator (std distributions are not).
struct SplitMix64 {
  std::uint64_t state;
  std::uint64_t next() {
    std::uint64_t z = (state += 0x9E37'79B9'7F4A'7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58'476D'1CE4'E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D0'49BB'1331'11EBull;
    return z ^ (z >> 31);
  }
  double uniform01() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
  float normal() {  // Box–Muller
    const double u1 = 1.0 - uniform01();
    const double u2 = uniform01();
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
  }
};

std::vector<float> gaussian(std::size_t n, std::uint64_t seed, float sigma = 1.0f) {
  SplitMix64 rng{seed};
  std::vector<float> v(n);
  for (float& x : v) x = sigma * rng.normal();
  return v;
}

struct ErrorStats {
  double max_abs = 0, mean_abs = 0, rmse = 0, max_rel = 0, signal_rms = 0;
  std::size_t over_bound = 0;
  double worst_bound_ratio = 0;  // max |err| / bound
};

// bound(i) is the per-element analytical error bound.
template <typename Bound>
ErrorStats measure(std::span<const float> ref, std::span<const float> got, Bound bound) {
  ErrorStats s;
  double sum_abs = 0, sum_sq = 0, sum_sig = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double r = ref[i];
    const double e = std::fabs(static_cast<double>(got[i]) - r);
    s.max_abs = std::max(s.max_abs, e);
    sum_abs += e;
    sum_sq += e * e;
    sum_sig += r * r;
    if (std::fabs(r) > 1e-3) s.max_rel = std::max(s.max_rel, e / std::fabs(r));
    const double b = bound(i);
    if (e > b) ++s.over_bound;
    if (b > 0) s.worst_bound_ratio = std::max(s.worst_bound_ratio, e / b);
  }
  const auto n = static_cast<double>(ref.size());
  s.mean_abs = sum_abs / n;
  s.rmse = std::sqrt(sum_sq / n);
  s.signal_rms = std::sqrt(sum_sig / n);
  return s;
}

void report(const char* name, std::size_t group, const ErrorStats& s) {
  std::printf("[ error ] %-12s group=%-5zu max_abs=%.3e mean_abs=%.3e rmse=%.3e "
              "rmse/rms=%.3e max_rel(|x|>1e-3)=%.3e worst_err/bound=%.3f\n",
              name, group, s.max_abs, s.mean_abs, s.rmse, s.rmse / s.signal_rms, s.max_rel,
              s.worst_bound_ratio);
}

struct RoundTrip {
  std::vector<AffineParams> params;
  std::vector<float> out;
};

RoundTrip round_trip(QuantFormat f, std::span<const float> x, std::size_t group) {
  RoundTrip rt;
  rt.params.resize(num_groups(x.size(), group));
  std::vector<std::byte> packed(packed_bytes(f, x.size()));
  EXPECT_EQ(quantize_reference(f, x, group, rt.params, packed), QuantStatus::kOk);
  rt.out.assign(x.size(), std::numeric_limits<float>::quiet_NaN());
  EXPECT_EQ(dequantize_reference(f, packed, x.size(), rt.params, group, rt.out), QuantStatus::kOk);
  return rt;
}

// A few ulps of binary32 at magnitude |v| to absorb the scale division/multiply.
double slack(double v) { return 4.0 * std::ldexp(std::fabs(v) + 1e-30, -23); }

constexpr std::size_t kN = 1 << 16;

TEST(DequantAccuracy, Fp16WithinHalfUlp) {
  const std::vector<float> x = gaussian(kN, 11, 100.0f);
  const RoundTrip rt = round_trip(QuantFormat::kFp16, x, 0);
  // |x| < 65504: normal half-ulp is 2^-11 relative; subnormal half-spacing 2^-25 absolute.
  const ErrorStats s = measure(x, rt.out, [&](std::size_t i) {
    return std::max(std::ldexp(std::fabs(static_cast<double>(x[i])), -11), std::ldexp(1.0, -25));
  });
  report("FP16", 0, s);
  EXPECT_EQ(s.over_bound, 0u);
}

template <QuantFormat F>
void check_fp8(std::size_t group, int rel_exp, int sub_exp) {
  const std::vector<float> x = gaussian(kN, 12 + group);
  const RoundTrip rt = round_trip(F, x, group);
  // In the scaled domain y = x/scale the encoder's error is at most half a
  // spacing: 2^rel_exp * |y| for normals, 2^sub_exp absolute in the subnormal range.
  const ErrorStats s = measure(x, rt.out, [&](std::size_t i) {
    const double scale = rt.params[i / group].scale;
    const double xi = x[i];
    return std::max(std::ldexp(std::fabs(xi), rel_exp), std::ldexp(scale, sub_exp)) + slack(xi);
  });
  report(format_traits(F).name.data(), group, s);
  EXPECT_EQ(s.over_bound, 0u);
}

TEST(DequantAccuracy, Fp8E4M3PerGroupAndPerTensor) {
  for (const std::size_t g : {std::size_t{16}, std::size_t{128}, kN}) {
    check_fp8<QuantFormat::kFp8E4M3>(g, -4, -10);
  }
}

TEST(DequantAccuracy, Fp8E5M2PerGroupAndPerTensor) {
  for (const std::size_t g : {std::size_t{16}, std::size_t{128}, kN}) {
    check_fp8<QuantFormat::kFp8E5M2>(g, -3, -17);
  }
}

template <QuantFormat F>
void check_int(std::size_t group) {
  const std::vector<float> x = gaussian(kN, 13 + group);
  const RoundTrip rt = round_trip(F, x, group);
  // Round-to-nearest in code space: |err| <= scale / 2.
  const ErrorStats s = measure(x, rt.out, [&](std::size_t i) {
    return 0.5 * static_cast<double>(rt.params[i / group].scale) * (1.0 + 1e-5) + slack(x[i]);
  });
  report(format_traits(F).name.data(), group, s);
  EXPECT_EQ(s.over_bound, 0u);
}

TEST(DequantAccuracy, Int4MinMaxGroups) {
  for (const std::size_t g : {std::size_t{16}, std::size_t{32}, std::size_t{64}, std::size_t{128}}) {
    check_int<QuantFormat::kInt4Packed>(g);
  }
}

TEST(DequantAccuracy, Int2MinMaxGroups) {
  for (const std::size_t g : {std::size_t{16}, std::size_t{32}, std::size_t{64}, std::size_t{128}}) {
    check_int<QuantFormat::kInt2Packed>(g);
  }
}

TEST(DequantAccuracy, GroupMinAndMaxAreExactForInt) {
  // bias = min is exact; max = fma(qmax, (max-min)/qmax, min) is within scale/2.
  const std::vector<float> x{-3.0f, 0.1f, 2.5f, 7.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  const RoundTrip rt = round_trip(QuantFormat::kInt4Packed, x, 4);
  EXPECT_EQ(rt.out[0], -3.0f);
  EXPECT_NEAR(rt.out[3], 7.0, 0.5 * rt.params[0].scale);
  // Constant group: scale 0, every element exact.
  EXPECT_EQ(rt.params[1].scale, 0.0f);
  for (std::size_t i = 4; i < 8; ++i) EXPECT_EQ(rt.out[i], 1.0f);
}

TEST(DequantAccuracy, EdgeShapes) {
  // Odd element counts, partial last group and partial last byte.
  for (const std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{5}, std::size_t{7},
                              std::size_t{33}, std::size_t{129}}) {
    const std::vector<float> x = gaussian(n, 100 + n);
    for (const QuantFormat f : {QuantFormat::kInt4Packed, QuantFormat::kInt2Packed}) {
      const std::size_t g = f == QuantFormat::kInt4Packed ? 32 : 64;
      const RoundTrip rt = round_trip(f, x, g);
      for (std::size_t i = 0; i < n; ++i) {
        EXPECT_LE(std::fabs(static_cast<double>(rt.out[i]) - x[i]),
                  0.5 * rt.params[i / g].scale * (1.0 + 1e-5) + slack(x[i]));
      }
    }
    const RoundTrip e4 = round_trip(QuantFormat::kFp8E4M3, x, 1);  // per-element scale
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_LE(std::fabs(static_cast<double>(e4.out[i]) - x[i]),
                std::ldexp(std::fabs(static_cast<double>(x[i])), -4) + slack(x[i]));
    }
  }
}

TEST(DequantAccuracy, ZeroLengthIsANoOp) {
  std::vector<float> out;
  EXPECT_EQ(dequantize_reference(QuantFormat::kInt4Packed, {}, 0, {}, 32, out), QuantStatus::kOk);
  EXPECT_EQ(dequantize_reference(QuantFormat::kFp16, {}, 0, {}, 0, out), QuantStatus::kOk);
}

// ---- 4. argument validation ------------------------------------------------------------------

TEST(DequantValidation, RejectsBadArguments) {
  const std::vector<std::byte> packed(64);
  const std::vector<AffineParams> one{{1.0f, 0.0f}};
  std::vector<float> out(64, 7.0f);

  EXPECT_EQ(dequantize_reference(static_cast<QuantFormat>(9), packed, 4, one, 4, out),
            QuantStatus::kUnsupportedFormat);
  // Group size rules.
  EXPECT_EQ(dequantize_reference(QuantFormat::kFp16, packed, 4, {}, 4, out),
            QuantStatus::kInvalidArgument);
  EXPECT_EQ(dequantize_reference(QuantFormat::kFp8E4M3, packed, 4, one, 0, out),
            QuantStatus::kInvalidArgument);
  EXPECT_EQ(dequantize_reference(QuantFormat::kInt4Packed, packed, 4, one, 3, out),
            QuantStatus::kInvalidArgument);
  EXPECT_EQ(dequantize_reference(QuantFormat::kInt2Packed, packed, 4, one, 6, out),
            QuantStatus::kInvalidArgument);
  // Parameter count must match exactly.
  EXPECT_EQ(dequantize_reference(QuantFormat::kInt4Packed, packed, 8, one, 4, out),
            QuantStatus::kInvalidArgument);
  // Non-finite parameters; non-zero bias for FP8.
  const std::vector<AffineParams> nan_scale{{std::numeric_limits<float>::quiet_NaN(), 0.0f}};
  const std::vector<AffineParams> inf_bias{{1.0f, std::numeric_limits<float>::infinity()}};
  const std::vector<AffineParams> fp8_bias{{1.0f, 0.5f}};
  EXPECT_EQ(dequantize_reference(QuantFormat::kInt4Packed, packed, 4, nan_scale, 4, out),
            QuantStatus::kInvalidArgument);
  EXPECT_EQ(dequantize_reference(QuantFormat::kInt4Packed, packed, 4, inf_bias, 4, out),
            QuantStatus::kInvalidArgument);
  EXPECT_EQ(dequantize_reference(QuantFormat::kFp8E5M2, packed, 4, fp8_bias, 4, out),
            QuantStatus::kInvalidArgument);
  // Buffer sizes.
  EXPECT_EQ(dequantize_reference(QuantFormat::kInt4Packed, std::span(packed).first(1), 4, one, 4,
                                 out),
            QuantStatus::kBufferTooSmall);
  EXPECT_EQ(dequantize_reference(QuantFormat::kFp16, std::span(packed).first(7), 4, {}, 0, out),
            QuantStatus::kBufferTooSmall);
  EXPECT_EQ(dequantize_reference(QuantFormat::kInt4Packed, packed, 4, one, 4,
                                 std::span(out).first(3)),
            QuantStatus::kBufferTooSmall);
  // Outputs untouched on every failure above.
  EXPECT_TRUE(std::all_of(out.begin(), out.end(), [](float v) { return v == 7.0f; }));
}

TEST(QuantTypes, TraitsAndSizes) {
  EXPECT_EQ(packed_bytes(QuantFormat::kFp16, 3), 6u);
  EXPECT_EQ(packed_bytes(QuantFormat::kFp8E4M3, 3), 3u);
  EXPECT_EQ(packed_bytes(QuantFormat::kInt4Packed, 3), 2u);
  EXPECT_EQ(packed_bytes(QuantFormat::kInt2Packed, 5), 2u);
  EXPECT_EQ(packed_bytes(QuantFormat::kInt2Packed, 8), 2u);
  EXPECT_EQ(num_groups(129, 64), 3u);
  EXPECT_EQ(num_groups(0, 64), 0u);
  EXPECT_EQ(elements_per_byte(QuantFormat::kInt4Packed), 2u);
  EXPECT_EQ(elements_per_byte(QuantFormat::kFp16), 0u);
  EXPECT_TRUE(is_valid_group_size(QuantFormat::kFp16, 0));
  EXPECT_FALSE(is_valid_group_size(QuantFormat::kFp8E4M3, 0));
  EXPECT_TRUE(is_valid_group_size(QuantFormat::kFp8E4M3, 1));
  EXPECT_FALSE(is_valid_group_size(static_cast<QuantFormat>(5), 4));
  EXPECT_EQ(format_traits(QuantFormat::kFp8E4M3).max_finite, decode_fp8_e4m3(0x7E));
  EXPECT_EQ(format_traits(QuantFormat::kFp8E5M2).max_finite, decode_fp8_e5m2(0x7B));
  EXPECT_EQ(format_traits(QuantFormat::kFp16).max_finite, decode_fp16(0x7BFF));
  EXPECT_STREQ(to_string(QuantStatus::kBufferTooSmall), "buffer too small");
}

}  // namespace
}  // namespace hl::quant
