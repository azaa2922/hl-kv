// Benchmark harness for the FP8 -> FP16 CUDA kernel (requires a CUDA device;
// every case reports an error and is skipped otherwise).
//
// Timing: cudaEvent pair around one launch on a dedicated stream, reported as
// manual time per iteration (kernel time only; no host<->device copies).
// bytes_per_second counts DRAM traffic the kernel must do at minimum:
// n bytes of FP8 read + 2n bytes of FP16 written + the parameter array.
//
// Arguments: {format (1 = E4M3, 2 = E5M2), n, group_size (0 = per-tensor),
//             input offset in bytes (0 = aligned / vectorized, 1 = scalar path)}
//
// Run:  hl_quant_cuda_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true

#include <benchmark/benchmark.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "hl_quant/cuda_dequant.hpp"
#include "hl_quant/quant_types.hpp"

namespace {

using hl::quant::AffineParams;
using hl::quant::QuantFormat;

bool device_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void BM_Fp8ToFp16(benchmark::State& state) {
  if (!device_available()) {
    state.SkipWithError("no CUDA device");
    return;
  }
  const auto format = static_cast<QuantFormat>(state.range(0));
  const auto n = static_cast<std::size_t>(state.range(1));
  const std::size_t group = state.range(2) == 0 ? n : static_cast<std::size_t>(state.range(2));
  const auto in_offset = static_cast<std::size_t>(state.range(3));
  const std::size_t groups = hl::quant::num_groups(n, group);

  std::vector<std::uint8_t> h_in(n);
  std::uint32_t x = 12345;
  for (std::uint8_t& b : h_in) {
    x = x * 1664525u + 1013904223u;
    b = static_cast<std::uint8_t>(x >> 24);
  }
  std::vector<AffineParams> h_params(groups, AffineParams{0.0123f, 0.0f});

  std::byte* d_in = nullptr;
  AffineParams* d_params = nullptr;
  std::uint16_t* d_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_in), n + in_offset) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_params), groups * sizeof(AffineParams)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_out), n * sizeof(std::uint16_t)) != cudaSuccess ||
      cudaStreamCreate(&stream) != cudaSuccess || cudaEventCreate(&start) != cudaSuccess ||
      cudaEventCreate(&stop) != cudaSuccess) {
    state.SkipWithError("CUDA allocation failed");
    return;
  }
  (void)cudaMemcpy(d_in + in_offset, h_in.data(), n, cudaMemcpyHostToDevice);
  (void)cudaMemcpy(d_params, h_params.data(), groups * sizeof(AffineParams),
                   cudaMemcpyHostToDevice);

  auto launch = [&] {
    return hl::quant::cuda::dequantize_fp8_to_fp16(format, d_in + in_offset, n, d_params, group,
                                                   d_out, stream);
  };
  if (!launch().ok() || cudaStreamSynchronize(stream) != cudaSuccess) {  // warm-up
    state.SkipWithError("kernel launch failed");
  } else {
    for (auto _ : state) {
      (void)cudaEventRecord(start, stream);
      (void)launch();
      (void)cudaEventRecord(stop, stream);
      (void)cudaEventSynchronize(stop);
      float ms = 0.0f;
      (void)cudaEventElapsedTime(&ms, start, stop);
      state.SetIterationTime(static_cast<double>(ms) / 1e3);
    }
    const auto bytes = static_cast<std::int64_t>(3 * n + groups * sizeof(AffineParams));
    state.SetBytesProcessed(state.iterations() * bytes);
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
  }
  (void)cudaEventDestroy(start);
  (void)cudaEventDestroy(stop);
  (void)cudaStreamDestroy(stream);
  (void)cudaFree(d_in);
  (void)cudaFree(d_params);
  (void)cudaFree(d_out);
}

BENCHMARK(BM_Fp8ToFp16)
    ->ArgsProduct({{1, 2}, {1 << 20, 1 << 24, 1 << 28}, {128, 0}, {0, 1}})
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

void print_device() {
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    std::printf("CUDA device: none available\n");
    return;
  }
  int mem_clock_khz = 0;
  int bus_bits = 0;
  (void)cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device);
  (void)cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device);
  // Context only: a theoretical figure derived from attributes, not a measurement.
  const double peak_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
  std::printf("CUDA device: %s (sm_%d%d, %d SMs); theoretical DRAM peak from attributes: %.0f GB/s\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount, peak_gbs);
}

}  // namespace

int main(int argc, char** argv) {
  print_device();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
