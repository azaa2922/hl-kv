// CPU reference encoder tests (FP32 -> FP16 / E4M3 / E5M2, INT packing).
//
// The encoders generate the test vectors for dequantization and will be the
// reference for GPU quantization kernels, so their rounding is verified
// exhaustively: every finite code round-trips, and every midpoint between two
// adjacent codes rounds to the even code (round-to-nearest-even), with the
// neighbouring floats on either side rounding away from the tie.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "hl_quant/cpu_dequant.hpp"
#include "hl_quant/cpu_quant.hpp"
#include "hl_quant/quant_types.hpp"

namespace hl::quant {
namespace {

struct Codec {
  const char* name;
  std::uint32_t (*encode)(float);
  float (*decode)(std::uint32_t);
  std::uint32_t sign_bit;
  std::uint32_t max_finite;  // magnitude code
};

const Codec kFp16{"FP16", [](float x) -> std::uint32_t { return encode_fp16(x); },
                  [](std::uint32_t c) { return decode_fp16(static_cast<std::uint16_t>(c)); },
                  0x8000u, 0x7BFFu};
const Codec kE4M3{"E4M3", [](float x) -> std::uint32_t { return encode_fp8_e4m3(x); },
                  [](std::uint32_t c) { return decode_fp8_e4m3(static_cast<std::uint8_t>(c)); },
                  0x80u, 0x7Eu};
const Codec kE5M2{"E5M2", [](float x) -> std::uint32_t { return encode_fp8_e5m2(x); },
                  [](std::uint32_t c) { return decode_fp8_e5m2(static_cast<std::uint8_t>(c)); },
                  0x80u, 0x7Bu};

class EncodeExhaustive : public ::testing::TestWithParam<Codec> {};

TEST_P(EncodeExhaustive, EveryCodeRoundTrips) {
  const Codec& c = GetParam();
  for (std::uint32_t code = 0; code < 2 * c.sign_bit; ++code) {
    const float v = c.decode(code);
    const std::uint32_t back = c.encode(v);
    if (std::isnan(v)) {
      EXPECT_TRUE(std::isnan(c.decode(back))) << c.name << " code 0x" << std::hex << code;
      EXPECT_EQ(back & c.sign_bit, code & c.sign_bit);
    } else if (std::isinf(v)) {
      // Only FP16 keeps Inf; the FP8 encoders saturate.
      const std::uint32_t expect = c.sign_bit == 0x8000u ? code : (code & c.sign_bit) | c.max_finite;
      EXPECT_EQ(back, expect) << c.name << " code 0x" << std::hex << code;
    } else {
      EXPECT_EQ(back, code) << c.name << " code 0x" << std::hex << code;
    }
  }
}

TEST_P(EncodeExhaustive, MidpointsRoundToEven) {
  const Codec& c = GetParam();
  for (std::uint32_t lo = 0; lo < c.max_finite; ++lo) {
    const std::uint32_t hi = lo + 1;
    const double a = c.decode(lo);
    const double b = c.decode(hi);
    // The midpoint needs one more significant bit than the format: exact in binary32.
    const auto mid = static_cast<float>((a + b) / 2);
    ASSERT_EQ(static_cast<double>(mid), (a + b) / 2);
    const std::uint32_t even = (lo & 1u) == 0 ? lo : hi;
    for (const std::uint32_t s : {0u, c.sign_bit}) {
      const float m = s != 0 ? -mid : mid;
      EXPECT_EQ(c.encode(m), s | even) << c.name << " tie between 0x" << std::hex << lo;
      EXPECT_EQ(c.encode(std::nextafter(m, 0.0f)), s | lo) << c.name << " below 0x" << std::hex << lo;
      const float away = std::nextafter(m, s != 0 ? -INFINITY : INFINITY);
      EXPECT_EQ(c.encode(away), s | hi) << c.name << " above 0x" << std::hex << lo;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Formats, EncodeExhaustive, ::testing::Values(kFp16, kE4M3, kE5M2),
                         [](const ::testing::TestParamInfo<Codec>& param_info) {
                           return std::string(param_info.param.name);
                         });

// Widen to unsigned so every comparison below is unsigned vs unsigned.
unsigned h16(float x) { return encode_fp16(x); }
unsigned e4(float x) { return encode_fp8_e4m3(x); }
unsigned e5(float x) { return encode_fp8_e5m2(x); }

TEST(EncodeSpecialValues, OverflowSaturationAndNan) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  // FP16: IEEE overflow to Inf (65520 is the tie between 65504 and 2^16 -> even = Inf).
  EXPECT_EQ(h16(65504.0f), 0x7BFFu);
  EXPECT_EQ(h16(std::nextafter(65520.0f, 0.0f)), 0x7BFFu);
  EXPECT_EQ(h16(65520.0f), 0x7C00u);
  EXPECT_EQ(h16(1e10f), 0x7C00u);
  EXPECT_EQ(h16(-inf), 0xFC00u);
  EXPECT_EQ(h16(nan), 0x7E00u);
  // E4M3: saturating; there is no Inf encoding.
  EXPECT_EQ(e4(448.0f), 0x7Eu);
  EXPECT_EQ(e4(464.0f), 0x7Eu);  // tie with the (absent) 480 -> even code
  EXPECT_EQ(e4(465.0f), 0x7Eu);
  EXPECT_EQ(e4(3e38f), 0x7Eu);
  EXPECT_EQ(e4(inf), 0x7Eu);
  EXPECT_EQ(e4(-inf), 0xFEu);
  EXPECT_EQ(e4(nan), 0x7Fu);
  EXPECT_EQ(e4(-nan), 0xFFu);
  // E5M2: saturating.
  EXPECT_EQ(e5(57344.0f), 0x7Bu);
  EXPECT_EQ(e5(61440.0f), 0x7Bu);
  EXPECT_EQ(e5(inf), 0x7Bu);
  EXPECT_EQ(e5(-1e30f), 0xFBu);
  EXPECT_EQ(e5(nan), 0x7Fu);
}

TEST(EncodeSpecialValues, UnderflowAndSignedZero) {
  EXPECT_EQ(h16(0.0f), 0x0000u);
  EXPECT_EQ(h16(-0.0f), 0x8000u);
  EXPECT_EQ(h16(1e-40f), 0x0000u);   // binary32 subnormal input
  EXPECT_EQ(h16(-1e-40f), 0x8000u);  // keeps the sign
  EXPECT_EQ(e4(std::ldexp(1.0f, -10)), 0x00u);  // tie with 2^-9 -> even (0)
  EXPECT_EQ(e4(std::ldexp(1.5f, -10)), 0x01u);
  EXPECT_EQ(e4(-std::ldexp(1.0f, -30)), 0x80u);
  EXPECT_EQ(e5(std::ldexp(1.0f, -17)), 0x00u);
  EXPECT_EQ(e5(std::ldexp(1.0f, -16)), 0x01u);
}

#if defined(__FLT16_MANT_DIG__)
// Cross-check against the compiler's _Float16 conversion (GCC/Clang on x86-64;
// hardware F16C or correctly rounded soft-float). Not available on MSVC.
TEST(EncodeCrossCheck, Fp16MatchesCompilerFloat16) {
  std::uint32_t state = 12345;
  for (int i = 0; i < 2'000'000; ++i) {
    state = state * 1664525u + 1013904223u;
    const float x = std::bit_cast<float>(state);
    if (std::isnan(x)) continue;
    const auto h = static_cast<_Float16>(x);
    EXPECT_EQ(h16(x), static_cast<unsigned>(std::bit_cast<std::uint16_t>(h))) << "x=" << x;
  }
}
#endif

// ---- quantize_reference ------------------------------------------------------------------

TEST(QuantizeReference, PaddingBitsAreZero) {
  const std::vector<float> x{0.0f, 1.0f, 2.0f};
  std::vector<AffineParams> p(1);
  std::vector<std::byte> packed(2, std::byte{0xFF});
  ASSERT_EQ(quantize_reference(QuantFormat::kInt4Packed, x, 4, p, packed), QuantStatus::kOk);
  EXPECT_EQ(std::to_integer<unsigned>(packed[1]) & 0xF0u, 0u);

  const std::vector<float> y{0, 1, 2, 3, 3};
  std::vector<std::byte> packed2(2, std::byte{0xFF});
  ASSERT_EQ(quantize_reference(QuantFormat::kInt2Packed, y, 8, p, packed2), QuantStatus::kOk);
  EXPECT_EQ(std::to_integer<unsigned>(packed2[0]), 0xE4u);  // codes 0,1,2,3
  EXPECT_EQ(std::to_integer<unsigned>(packed2[1]), 0x03u);  // code 3 + zero padding
  EXPECT_EQ(p[0].scale, 1.0f);
  EXPECT_EQ(p[0].bias, 0.0f);
}

TEST(QuantizeReference, Fp8ScaleIsAmaxOverMaxFinite) {
  const std::vector<float> x{0.5f, -3.0f, 1.0f, 0.0f};
  std::vector<AffineParams> p(2);
  std::vector<std::byte> packed(4);
  ASSERT_EQ(quantize_reference(QuantFormat::kFp8E4M3, x, 2, p, packed), QuantStatus::kOk);
  EXPECT_EQ(p[0].scale, 3.0f / 448.0f);
  EXPECT_EQ(p[1].scale, 1.0f / 448.0f);
  EXPECT_EQ(p[0].bias, 0.0f);
  // The group maximum maps to (or saturates at) the largest finite code.
  EXPECT_EQ(std::to_integer<unsigned>(packed[1]), 0xFEu);
  EXPECT_EQ(std::to_integer<unsigned>(packed[2]), 0x7Eu);

  const std::vector<float> zeros(4, 0.0f);
  ASSERT_EQ(quantize_reference(QuantFormat::kFp8E5M2, zeros, 4, std::span(p).first(1), packed),
            QuantStatus::kOk);
  EXPECT_EQ(p[0].scale, 1.0f);
}

TEST(QuantizeReference, IsDeterministic) {
  std::vector<float> x(1000);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i)) * 7.0f;
  for (const QuantFormat f : {QuantFormat::kFp8E4M3, QuantFormat::kInt4Packed,
                              QuantFormat::kInt2Packed}) {
    std::vector<AffineParams> p1(num_groups(x.size(), 8)), p2(p1.size());
    std::vector<std::byte> b1(packed_bytes(f, x.size())), b2(b1.size());
    ASSERT_EQ(quantize_reference(f, x, 8, p1, b1), QuantStatus::kOk);
    ASSERT_EQ(quantize_reference(f, x, 8, p2, b2), QuantStatus::kOk);
    EXPECT_EQ(b1, b2);
    EXPECT_TRUE(std::equal(p1.begin(), p1.end(), p2.begin(), [](AffineParams a, AffineParams b) {
      return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
    }));
  }
}

TEST(QuantizeReference, RejectsBadArgumentsWithoutWriting) {
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<AffineParams> p(1, AffineParams{9.0f, 9.0f});
  std::vector<std::byte> packed(8, std::byte{0xAB});
  auto untouched = [&] {
    return p[0].scale == 9.0f && p[0].bias == 9.0f &&
           std::all_of(packed.begin(), packed.end(), [](std::byte b) { return b == std::byte{0xAB}; });
  };

  const std::vector<float> with_inf{1.0f, inf, 0.0f, 2.0f};
  EXPECT_EQ(quantize_reference(QuantFormat::kInt4Packed, with_inf, 4, p, packed),
            QuantStatus::kInvalidArgument);
  EXPECT_EQ(quantize_reference(QuantFormat::kFp8E4M3, with_inf, 4, p, packed),
            QuantStatus::kInvalidArgument);
  const std::vector<float> huge_range{-3e38f, 3e38f, 0.0f, 0.0f};  // max - min overflows
  EXPECT_EQ(quantize_reference(QuantFormat::kInt2Packed, huge_range, 4, p, packed),
            QuantStatus::kInvalidArgument);
  const std::vector<float> ok{1.0f, 2.0f, 3.0f, 4.0f};
  EXPECT_EQ(quantize_reference(QuantFormat::kInt4Packed, ok, 2, p, packed),
            QuantStatus::kInvalidArgument);  // needs 2 param groups
  EXPECT_EQ(quantize_reference(QuantFormat::kInt4Packed, ok, 4, p, std::span(packed).first(1)),
            QuantStatus::kBufferTooSmall);
  EXPECT_EQ(quantize_reference(static_cast<QuantFormat>(7), ok, 4, p, packed),
            QuantStatus::kUnsupportedFormat);
  EXPECT_TRUE(untouched());

  // FP16 represents Inf, so it is accepted there.
  std::vector<std::byte> fp16(8);
  ASSERT_EQ(quantize_reference(QuantFormat::kFp16, with_inf, 0, {}, fp16), QuantStatus::kOk);
  EXPECT_EQ(std::to_integer<unsigned>(fp16[2]) | (std::to_integer<unsigned>(fp16[3]) << 8),
            0x7C00u);
}

}  // namespace
}  // namespace hl::quant
