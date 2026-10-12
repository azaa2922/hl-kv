// Tests for the FP8 -> FP16 CUDA kernel (hl::quant_cuda).
//
//   HostEmulation.*  run the kernel's per-element code (the same __host__
//                    __device__ functions, compiled for the host by nvcc) on the
//                    CPU and compare with dequantize_reference_to_fp16 bit for
//                    bit. They need no GPU and always run in CUDA builds.
//   Device.*         launch the real kernels; skipped when no CUDA device is
//                    available. They cover what host emulation cannot: device
//                    conversion instructions, indexing, launch geometry,
//                    alignment-dependent path selection and grid-stride loops.
//
// NaN payloads are unspecified by the contract: NaN must map to NaN.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "hl_quant/cpu_dequant.hpp"
#include "hl_quant/cuda_dequant.hpp"
#include "hl_quant/quant_types.hpp"

namespace hl::quant {
namespace {

using cuda::HostPath;

constexpr std::array<QuantFormat, 2> kFp8Formats{QuantFormat::kFp8E4M3, QuantFormat::kFp8E5M2};

struct SplitMix64 {
  std::uint64_t state;
  std::uint64_t next() {
    std::uint64_t z = (state += 0x9E37'79B9'7F4A'7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58'476D'1CE4'E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D0'49BB'1331'11EBull;
    return z ^ (z >> 31);
  }
};

std::vector<std::byte> all_codes() {
  std::vector<std::byte> v;
  for (unsigned c = 0; c < 256; ++c) v.push_back(static_cast<std::byte>(c));
  return v;
}

std::vector<std::byte> random_codes(std::size_t n, SplitMix64& rng) {
  std::vector<std::byte> v(n);
  for (std::byte& b : v) b = static_cast<std::byte>(rng.next() & 0xFFu);
  return v;
}

// Log-uniform magnitudes in [2^-24, 2^12] with random sign: exercises FP16
// subnormal results, overflow to Inf and ordinary values.
std::vector<AffineParams> random_params(std::size_t groups, SplitMix64& rng) {
  std::vector<AffineParams> p(groups);
  for (AffineParams& a : p) {
    const int e = static_cast<int>(rng.next() % 37) - 24;
    const float mantissa = 1.0f + static_cast<float>(rng.next() % 1024) / 1024.0f;
    const float sign = (rng.next() & 1u) != 0 ? -1.0f : 1.0f;
    a = AffineParams{sign * std::ldexp(mantissa, e), 0.0f};
  }
  return p;
}

// Identity, powers of two, non-dyadic values, FP16-subnormal and overflowing
// products, binary32-subnormal products (FLT_MIN), zero and negative scales.
const std::vector<float> kScales{1.0f,   0.5f,     2.0f,   1.0f / 448.0f, 3.7e-3f,
                                 12.5f,  -1.0f,    0.0f,   0x1p-20f,      0x1p-30f,
                                 1e-6f,  150.0f,   1e4f,   0.1f,          65504.0f / 448.0f,
                                 std::numeric_limits<float>::min()};

bool is_fp16_nan(std::uint16_t h) { return (h & 0x7C00u) == 0x7C00u && (h & 0x03FFu) != 0; }

bool same_fp16(std::uint16_t got, std::uint16_t want) {
  if (is_fp16_nan(got) || is_fp16_nan(want)) return is_fp16_nan(got) && is_fp16_nan(want);
  return got == want;
}

::testing::AssertionResult matches_reference(QuantFormat f, std::span<const std::byte> in,
                                             std::span<const AffineParams> params,
                                             std::size_t group_size,
                                             std::span<const std::uint16_t> got) {
  std::vector<std::uint16_t> want(in.size());
  const QuantStatus s = dequantize_reference_to_fp16(f, in, in.size(), params, group_size, want);
  if (s != QuantStatus::kOk) {
    return ::testing::AssertionFailure() << "reference rejected the arguments: " << to_string(s);
  }
  std::size_t mismatches = 0;
  std::string first;
  for (std::size_t i = 0; i < in.size(); ++i) {
    if (same_fp16(got[i], want[i])) continue;
    if (mismatches++ == 0) {
      first = "i=" + std::to_string(i) + " code=" +
              std::to_string(std::to_integer<unsigned>(in[i])) + " got=" + std::to_string(got[i]) +
              " want=" + std::to_string(want[i]);
    }
  }
  if (mismatches == 0) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << mismatches << " mismatches, first: " << first;
}

// ---- host emulation (no GPU needed) ------------------------------------------------------

TEST(HostEmulation, AllCodesManyScalesBothPaths) {
  const std::vector<std::byte> in = all_codes();
  for (const QuantFormat f : kFp8Formats) {
    for (const float scale : kScales) {
      const std::vector<AffineParams> params{{scale, 0.0f}};
      for (const HostPath path : {HostPath::kScalar, HostPath::kVector}) {
        std::vector<std::uint16_t> out(in.size());
        ASSERT_EQ(cuda::emulate_fp8_to_fp16_on_host(f, in, params, in.size(), out, path),
                  QuantStatus::kOk);
        EXPECT_TRUE(matches_reference(f, in, params, in.size(), out))
            << format_traits(f).name << " scale=" << scale << " path=" << static_cast<int>(path);
      }
    }
  }
}

TEST(HostEmulation, RandomShapesAndGroups) {
  SplitMix64 rng{7};
  for (const QuantFormat f : kFp8Formats) {
    for (const std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{9},
                                std::size_t{17}, std::size_t{64}, std::size_t{65},
                                std::size_t{1000}, std::size_t{4099}}) {
      for (const std::size_t g : {std::size_t{1}, std::size_t{3}, std::size_t{8}, std::size_t{24},
                                  std::size_t{128}, n}) {
        const std::vector<std::byte> in = random_codes(n, rng);
        const std::vector<AffineParams> params = random_params(num_groups(n, g), rng);
        for (const HostPath path : {HostPath::kScalar, HostPath::kVector}) {
          std::vector<std::uint16_t> out(n);
          ASSERT_EQ(cuda::emulate_fp8_to_fp16_on_host(f, in, params, g, out, path),
                    QuantStatus::kOk);
          EXPECT_TRUE(matches_reference(f, in, params, g, out)) << "n=" << n << " g=" << g;
        }
      }
    }
  }
}

TEST(HostEmulation, RejectsBadArguments) {
  const std::vector<std::byte> in(16);
  const std::vector<AffineParams> one{{1.0f, 0.0f}};
  std::vector<std::uint16_t> out(16);
  for (const QuantFormat f : {QuantFormat::kFp16, QuantFormat::kInt4Packed,
                              static_cast<QuantFormat>(kRetiredInt2Code)}) {
    EXPECT_EQ(cuda::emulate_fp8_to_fp16_on_host(f, in, one, 16, out, HostPath::kScalar),
              QuantStatus::kUnsupportedFormat);
  }
  EXPECT_EQ(cuda::emulate_fp8_to_fp16_on_host(QuantFormat::kFp8E4M3, in, one, 0, out,
                                               HostPath::kScalar),
            QuantStatus::kInvalidArgument);
  EXPECT_EQ(cuda::emulate_fp8_to_fp16_on_host(QuantFormat::kFp8E4M3, in, one, 8, out,
                                               HostPath::kScalar),
            QuantStatus::kInvalidArgument);  // 2 groups need 2 params
  EXPECT_EQ(cuda::emulate_fp8_to_fp16_on_host(QuantFormat::kFp8E4M3, in, one, 16,
                                               std::span(out).first(15), HostPath::kScalar),
            QuantStatus::kBufferTooSmall);
}

// ---- device -------------------------------------------------------------------------------

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0 && cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)) != cudaSuccess) {
      ptr_ = nullptr;
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (ptr_ != nullptr) (void)cudaFree(ptr_);
  }
  [[nodiscard]] T* get() const noexcept { return ptr_; }
  [[nodiscard]] bool ok() const noexcept { return count_ == 0 || ptr_ != nullptr; }

 private:
  T* ptr_ = nullptr;
  std::size_t count_;
};

class Device : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    const cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
      GTEST_SKIP() << "no CUDA device available (" << cudaGetErrorString(e) << ")";
    }
  }

  // Runs the kernel with the input/output placed at the given element offsets
  // inside larger allocations (offsets control which path the launcher takes).
  static std::vector<std::uint16_t> run(QuantFormat f, std::span<const std::byte> in,
                                        std::span<const AffineParams> params,
                                        std::size_t group_size, std::size_t in_offset = 0,
                                        std::size_t out_offset = 0,
                                        cudaStream_t stream = nullptr) {
    const std::size_t n = in.size();
    DeviceBuffer<std::byte> d_in(n + in_offset);
    DeviceBuffer<AffineParams> d_params(params.size());
    DeviceBuffer<std::uint16_t> d_out(n + out_offset);
    EXPECT_TRUE(d_in.ok() && d_params.ok() && d_out.ok());
    EXPECT_EQ(cudaMemcpy(d_in.get() + in_offset, in.data(), n, cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_params.get(), params.data(), params.size_bytes(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    const cuda::DeviceStatus s = cuda::dequantize_fp8_to_fp16(
        f, d_in.get() + in_offset, n, d_params.get(), group_size, d_out.get() + out_offset, stream);
    EXPECT_TRUE(s.ok()) << to_string(s.status) << " / " << cudaGetErrorString(s.cuda_error);
    EXPECT_EQ(stream == nullptr ? cudaDeviceSynchronize() : cudaStreamSynchronize(stream),
              cudaSuccess);
    std::vector<std::uint16_t> out(n, 0xFFFFu);
    EXPECT_EQ(cudaMemcpy(out.data(), d_out.get() + out_offset, n * sizeof(std::uint16_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return out;
  }
};

TEST_F(Device, AllCodesManyScalesVectorAndScalarPaths) {
  const std::vector<std::byte> in = all_codes();
  for (const QuantFormat f : kFp8Formats) {
    for (const float scale : kScales) {
      const std::vector<AffineParams> params{{scale, 0.0f}};
      const std::vector<std::uint16_t> vec = run(f, in, params, in.size());        // aligned
      const std::vector<std::uint16_t> sca = run(f, in, params, in.size(), 1, 1);  // misaligned
      EXPECT_TRUE(matches_reference(f, in, params, in.size(), vec)) << "vector, scale=" << scale;
      EXPECT_TRUE(matches_reference(f, in, params, in.size(), sca)) << "scalar, scale=" << scale;
    }
  }
}

TEST_F(Device, RandomShapesGroupsAndAlignments) {
  SplitMix64 rng{11};
  for (const QuantFormat f : kFp8Formats) {
    for (const std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{9},
                                std::size_t{255}, std::size_t{4099}, std::size_t{65543}}) {
      for (const std::size_t g : {std::size_t{1}, std::size_t{24}, std::size_t{128}, n}) {
        const std::vector<std::byte> in = random_codes(n, rng);
        const std::vector<AffineParams> params = random_params(num_groups(n, g), rng);
        for (const auto& [in_off, out_off] : {std::pair<std::size_t, std::size_t>{0, 0},
                                              {8, 8}, {3, 0}, {0, 1}, {5, 3}}) {
          const std::vector<std::uint16_t> out = run(f, in, params, g, in_off, out_off);
          EXPECT_TRUE(matches_reference(f, in, params, g, out))
              << "n=" << n << " g=" << g << " offsets=" << in_off << "/" << out_off;
        }
      }
    }
  }
}

TEST_F(Device, LargeInputExercisesGridStride) {
  SplitMix64 rng{13};
  const std::size_t n = (std::size_t{1} << 25) + 13;  // more chunks than one grid covers
  const std::vector<std::byte> in = random_codes(n, rng);
  const std::vector<AffineParams> params = random_params(num_groups(n, 128), rng);
  const std::vector<std::uint16_t> out = run(QuantFormat::kFp8E4M3, in, params, 128);
  EXPECT_TRUE(matches_reference(QuantFormat::kFp8E4M3, in, params, 128, out));
}

TEST_F(Device, RunsOnTheGivenStream) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  SplitMix64 rng{17};
  const std::vector<std::byte> in = random_codes(10000, rng);
  const std::vector<AffineParams> params = random_params(num_groups(in.size(), 64), rng);
  const std::vector<std::uint16_t> out = run(QuantFormat::kFp8E5M2, in, params, 64, 0, 0, stream);
  EXPECT_TRUE(matches_reference(QuantFormat::kFp8E5M2, in, params, 64, out));
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(Device, ArgumentErrorsLaunchNothing) {
  DeviceBuffer<std::byte> in(16);
  DeviceBuffer<AffineParams> params(1);
  DeviceBuffer<std::uint16_t> out(16);
  auto call = [&](QuantFormat f, std::size_t n, std::size_t g, bool null_out) {
    return cuda::dequantize_fp8_to_fp16(f, in.get(), n, params.get(), g,
                                        null_out ? nullptr : out.get(), nullptr)
        .status;
  };
  EXPECT_EQ(call(QuantFormat::kFp16, 16, 16, false), QuantStatus::kUnsupportedFormat);
  EXPECT_EQ(call(QuantFormat::kInt4Packed, 16, 16, false), QuantStatus::kUnsupportedFormat);
  EXPECT_EQ(call(QuantFormat::kFp8E4M3, 16, 0, false), QuantStatus::kInvalidArgument);
  EXPECT_EQ(call(QuantFormat::kFp8E4M3, 16, 16, true), QuantStatus::kInvalidArgument);
  EXPECT_EQ(call(QuantFormat::kFp8E4M3, 0, 16, true), QuantStatus::kOk);  // n == 0: no-op
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

}  // namespace
}  // namespace hl::quant
