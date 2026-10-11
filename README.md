# HL-KV — Hetero-Latent ABI

Heterogeneous KV-cache representation and execution: KV blocks of one sequence may use different
storage formats (FP16, FP8, INT4, INT2) and are described to kernels through a compact, versioned
per-block ABI.

The program is split into five projects (allocator → dequantization → attention → fused
heterogeneous attention → engine integration). **Current state: Project 1 implemented and
CPU-benchmarked; Project 2 has its CPU numerical reference. No GPU code exists yet.** See
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

## Layout

```
project1_allocator/   hl_allocator — BlockDescriptor ABI + HeteroBlockTable (C++20, no CUDA)
project2_quant/       hl_quant — KV format definitions, CPU reference (de)quantizers, CUDA flags
tests/project1/       allocator unit tests + Google Benchmark harness (+ homogeneous baseline)
tests/project2/       exhaustive decode/encode tests and accuracy tests vs FP32 targets
cmake/                compiler options, pinned dependencies
docs/                 architecture specs, ADRs, dated experiment logs, project status
```

## Requirements

* CMake ≥ 3.20 and a C++20 compiler. Verified: GCC 13.3 and Clang 18.1 (Linux x86-64, locally and in
  CI), MSVC 2022 `/W4 /WX` (Windows x64, CI only — `.github/workflows/ci.yml`).
* GoogleTest and Google Benchmark: used from an installed package if found, otherwise fetched from
  GitHub at pinned commits (googletest v1.15.2, benchmark v1.9.1). Offline: pass
  `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<dir> -DFETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK=<dir>`.
* Optional: CUDA ≥ 12 for `-DHLKV_ENABLE_CUDA=ON` (no kernels yet; configures the toolchain only).

## Build

### Linux (GCC or Clang)

```bash
cmake --preset linux-gcc-release        # or linux-clang-release
cmake --build --preset linux-gcc-release
ctest --preset linux-gcc-release

# Sanitizers (Debug + AddressSanitizer + UBSan, tests only)
cmake --preset linux-gcc-asan && cmake --build --preset linux-gcc-asan && ctest --preset linux-gcc-asan
```

Without presets:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Windows (MSVC)

From a "Developer PowerShell for VS 2022":

```powershell
cmake --preset windows-msvc
cmake --build --preset windows-msvc        # Release configuration
ctest --preset windows-msvc
```

Without presets: `cmake -S . -B build -G "Visual Studio 17 2022" -A x64`, then
`cmake --build build --config Release` and `ctest --test-dir build -C Release`.

### Options

| Option | Default | Effect |
|---|---|---|
| `HLKV_BUILD_TESTS` | ON | GoogleTest suites |
| `HLKV_BUILD_BENCHMARKS` | ON | Google Benchmark harness |
| `HLKV_WARNINGS_AS_ERRORS` | OFF (ON in presets/CI) | `-Werror` / `/WX` on HL-KV targets |
| `HLKV_SANITIZE` | empty | e.g. `address,undefined` (GCC/Clang) |
| `HLKV_ENABLE_CUDA` | OFF | enable CUDA 12+, define `hl::quant_cuda_options` |
| `HLKV_CUDA_VERBOSE_PTXAS` | OFF | `-Xptxas=-v` (registers/spills per kernel) |
| `CMAKE_CUDA_ARCHITECTURES` | `80;89;90` | when CUDA is enabled |

## Benchmarks

```bash
taskset -c 2 ./build/linux-gcc-release/tests/hl_allocator_bench \
  --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

Each `BM_Hetero_*` case has a `BM_Baseline_*` twin (minimal homogeneous int32 block pool). Results
and their caveats are recorded in [`docs/benchmarks/`](docs/benchmarks/); numbers are only reported
from runs that were actually executed, with host and compiler details.

## Documentation

* [`docs/architecture/hl_allocator_abi_v0.md`](docs/architecture/hl_allocator_abi_v0.md) — byte layout, state machine, contracts, limitations
* [`docs/architecture/hl_quant_formats_v0.md`](docs/architecture/hl_quant_formats_v0.md) — format definitions, rounding, correctness method
* [`docs/decisions/`](docs/decisions/) — architecture decision records
* [`docs/benchmarks/`](docs/benchmarks/) — dated experiment logs (P1 CPU benchmarks, P2 reference error)
