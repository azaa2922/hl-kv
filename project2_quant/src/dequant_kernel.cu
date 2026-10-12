// FP8 -> FP16 dequantization kernels (hl_quant, Project 2).
//
// Numerical contract (cuda_dequant.hpp), per element:
//   1. FP8 -> FP16 with cuda_fp8.h. Every E4M3 and E5M2 value is exactly
//      representable in binary16, so this step is exact on every architecture
//      (sm_89+ use the hardware cvt instruction; older architectures and the
//      host use the header's software path).
//   2. FP16 -> FP32 (exact), multiply by the group scale in FP32 (RNE).
//   3. FP32 -> FP16, round-to-nearest-even (__float2half_rn / __float22half2_rn).
// Built without fast-math: subnormal products must not be flushed to zero.
//
// Memory access: the vectorized kernel converts 8 elements per thread — one
// 8-byte load and one 16-byte store, contiguous across a warp (256 B in, 512 B
// out per warp instruction). The group scale is read once per 8 elements;
// neighbouring threads usually share a group, so it is served from cache.
// The scalar kernel handles any alignment, group size and the tail.
//
// Group index arithmetic uses 32-bit division whenever n < 2^32 (64-bit integer
// division is several times more expensive on the GPU).

#include "hl_quant/cuda_dequant.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace hl::quant::cuda {
namespace {

constexpr unsigned kThreadsPerBlock = 256;
constexpr std::uint64_t kMaxBlocks = 65535;  // grid-stride loops cover any n
constexpr std::size_t kVec = 8;              // elements per thread, vectorized kernel

// ---- per-element arithmetic, shared by the kernels and the host emulation ------------

template <__nv_fp8_interpretation_t Kind>
__host__ __device__ __forceinline__ std::uint16_t convert1(std::uint8_t code, float scale) {
  const __half exact(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code), Kind));
  const __half_raw rounded = __float2half_rn(__half2float(exact) * scale);
  return rounded.x;
}

// Eight codes (little-endian in a uint2) -> eight binary16 values (in a uint4).
// Pair conversions keep element order: the low byte of an fp8x2 maps to .x,
// which is the low half of the output word.
template <__nv_fp8_interpretation_t Kind>
__host__ __device__ __forceinline__ uint4 convert8(uint2 codes, float scale) {
  const std::uint32_t words[2] = {codes.x, codes.y};
  std::uint32_t halves[4];
  for (int w = 0; w < 2; ++w) {
    for (int p = 0; p < 2; ++p) {
      const auto pair = static_cast<__nv_fp8x2_storage_t>((words[w] >> (16 * p)) & 0xFFFFu);
      const __half2 exact(__nv_cvt_fp8x2_to_halfraw2(pair, Kind));
      float2 f = __half22float2(exact);
      f.x *= scale;
      f.y *= scale;
      const __half2_raw rounded = __float22half2_rn(f);
      halves[2 * w + p] =
          static_cast<std::uint32_t>(rounded.x) | (static_cast<std::uint32_t>(rounded.y) << 16);
    }
  }
  return make_uint4(halves[0], halves[1], halves[2], halves[3]);
}

// ---- kernels ---------------------------------------------------------------------------

// Index is uint32_t when every element index fits in 32 bits, else uint64_t;
// it is only used for the group division.
template <__nv_fp8_interpretation_t Kind, typename Index>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fp8_to_fp16_vec8(const uint2* __restrict__ in, const AffineParams* __restrict__ params,
                     Index chunks_per_group, std::uint64_t num_chunks, uint4* __restrict__ out) {
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t c = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       c < num_chunks; c += stride) {
    const float scale = params[static_cast<Index>(c) / chunks_per_group].scale;
    out[c] = convert8<Kind>(in[c], scale);
  }
}

template <__nv_fp8_interpretation_t Kind, typename Index>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fp8_to_fp16_scalar(const std::uint8_t* __restrict__ in,
                       const AffineParams* __restrict__ params, Index group_size,
                       std::uint64_t begin, std::uint64_t end, std::uint16_t* __restrict__ out) {
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t i = begin + static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < end; i += stride) {
    out[i] = convert1<Kind>(in[i], params[static_cast<Index>(i) / group_size].scale);
  }
}

// ---- launch ----------------------------------------------------------------------------

unsigned grid_for(std::uint64_t work) {
  const std::uint64_t blocks = (work + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return static_cast<unsigned>(std::min(blocks, kMaxBlocks));
}

bool vector_path_applies(std::size_t n, std::size_t group_size) {
  // Every 8-element chunk must lie inside one parameter group.
  return group_size % kVec == 0 || group_size >= n;
}

template <__nv_fp8_interpretation_t Kind, typename Index>
void launch(const std::byte* d_in, std::size_t n, const AffineParams* d_params,
            std::size_t group_size, std::uint16_t* d_out, cudaStream_t stream) {
  // With a single group the division must always yield 0.
  const bool single_group = group_size >= n;
  std::uint64_t done = 0;
  const bool aligned = reinterpret_cast<std::uintptr_t>(d_in) % alignof(uint2) == 0 &&
                       reinterpret_cast<std::uintptr_t>(d_out) % alignof(uint4) == 0;
  if (aligned && vector_path_applies(n, group_size) && n >= kVec) {
    const std::uint64_t chunks = n / kVec;
    const Index chunks_per_group =
        single_group ? std::numeric_limits<Index>::max() : static_cast<Index>(group_size / kVec);
    fp8_to_fp16_vec8<Kind, Index><<<grid_for(chunks), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const uint2*>(d_in), d_params, chunks_per_group, chunks,
        reinterpret_cast<uint4*>(d_out));
    done = chunks * kVec;
  }
  if (done < n) {
    const Index groups_of =
        single_group ? std::numeric_limits<Index>::max() : static_cast<Index>(group_size);
    fp8_to_fp16_scalar<Kind, Index><<<grid_for(n - done), kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const std::uint8_t*>(d_in), d_params, groups_of, done, n, d_out);
  }
}

template <__nv_fp8_interpretation_t Kind>
void launch_any(const std::byte* d_in, std::size_t n, const AffineParams* d_params,
                std::size_t group_size, std::uint16_t* d_out, cudaStream_t stream) {
  if (n <= std::numeric_limits<std::uint32_t>::max()) {
    launch<Kind, std::uint32_t>(d_in, n, d_params, group_size, d_out, stream);
  } else {
    launch<Kind, std::uint64_t>(d_in, n, d_params, group_size, d_out, stream);
  }
}

// ---- host emulation --------------------------------------------------------------------

template <__nv_fp8_interpretation_t Kind>
void emulate(std::span<const std::byte> in, std::span<const AffineParams> params,
             std::size_t group_size, std::span<std::uint16_t> out, HostPath path) {
  const std::size_t n = in.size();
  std::size_t done = 0;
  if (path == HostPath::kVector && vector_path_applies(n, group_size)) {
    for (std::size_t c = 0; c < n / kVec; ++c) {
      uint2 codes;
      std::memcpy(&codes, in.data() + c * kVec, sizeof(codes));
      const uint4 halves = convert8<Kind>(codes, params[(c * kVec) / group_size].scale);
      std::memcpy(out.data() + c * kVec, &halves, sizeof(halves));
    }
    done = (n / kVec) * kVec;
  }
  for (std::size_t i = done; i < n; ++i) {
    out[i] = convert1<Kind>(std::to_integer<std::uint8_t>(in[i]), params[i / group_size].scale);
  }
}

bool is_fp8(QuantFormat format) {
  return format == QuantFormat::kFp8E4M3 || format == QuantFormat::kFp8E5M2;
}

}  // namespace

DeviceStatus dequantize_fp8_to_fp16(QuantFormat format, const std::byte* d_in, std::size_t n,
                                    const AffineParams* d_params, std::size_t group_size,
                                    std::uint16_t* d_out, cudaStream_t stream) noexcept {
  if (!is_fp8(format)) return {QuantStatus::kUnsupportedFormat, cudaSuccess};
  if (group_size == 0 || n > kMaxElements) return {QuantStatus::kInvalidArgument, cudaSuccess};
  if (n == 0) return {};
  if (d_in == nullptr || d_params == nullptr || d_out == nullptr) {
    return {QuantStatus::kInvalidArgument, cudaSuccess};
  }
  if (format == QuantFormat::kFp8E4M3) {
    launch_any<__NV_E4M3>(d_in, n, d_params, group_size, d_out, stream);
  } else {
    launch_any<__NV_E5M2>(d_in, n, d_params, group_size, d_out, stream);
  }
  // Reports launch errors (and, as with any cudaGetLastError, an earlier
  // pending asynchronous error).
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
    return {QuantStatus::kCudaError, e};
  }
  return {};
}

QuantStatus emulate_fp8_to_fp16_on_host(QuantFormat format, std::span<const std::byte> in,
                                        std::span<const AffineParams> params,
                                        std::size_t group_size, std::span<std::uint16_t> out,
                                        HostPath path) noexcept {
  if (!is_fp8(format)) return QuantStatus::kUnsupportedFormat;
  const std::size_t n = in.size();
  if (group_size == 0 || n > kMaxElements) return QuantStatus::kInvalidArgument;
  if (params.size() != num_groups(n, group_size)) return QuantStatus::kInvalidArgument;
  if (out.size() < n) return QuantStatus::kBufferTooSmall;
  if (format == QuantFormat::kFp8E4M3) {
    emulate<__NV_E4M3>(in, params, group_size, out, path);
  } else {
    emulate<__NV_E5M2>(in, params, group_size, out, path);
  }
  return QuantStatus::kOk;
}

}  // namespace hl::quant::cuda
