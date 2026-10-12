# HL-KV — Hetero-Latent ABI

Heterogeneous KV-cache representation and execution: KV blocks of one sequence may use different
storage formats (FP16, FP8, INT4) and are described to kernels through a compact, versioned
per-block ABI.

The program is split into five projects (allocator → dequantization → attention → fused
heterogeneous attention → engine integration). **Current state:** Project 1 is implemented and
CPU-benchmarked (ABI v1 with persistent device tables and delta updates). Project 2 has its CPU
numerical reference and a first CUDA kernel (FP8 → FP16) that compiles and is validated through host
emulation, **but has not yet run on a GPU**. See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

## Layout

```
project1_allocator/   hl_allocator — block descriptor ABI, HeteroBlockTable, IdArena (C++20, no CUDA)
project2_quant/       hl_quant — KV formats, CPU reference (de)quantizers; hl_quant_cuda kernels
tests/project1/       allocator unit/differential tests + Google Benchmark harness (+ homogeneous baseline)
tests/project2/       exhaustive decode/encode tests, accuracy tests; CUDA kernel tests + benchmark
cmake/                compiler options, pinned dependencies
docs/                 architecture specs, ADRs, dated experiment logs, project status
```

## Requirements

* CMake ≥ 3.20 and a C++20 compiler. Verified: GCC 13.3 and Clang 18.1 (Linux x86-64, locally and in
  CI), MSVC 2022 `/W4 /WX` (Windows x64, CI only — `.github/workflows/ci.yml`).
* GoogleTest and Google Benchmark: used from an installed package if found, otherwise fetched from
  GitHub at pinned commits (googletest v1.15.2, benchmark v1.9.1). Offline: pass
  `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<dir> -DFETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK=<dir>`.
* Optional: CUDA ≥ 12 (nvcc + runtime headers) for the GPU kernels. Verified to compile with nvcc
  12.6 + GCC 13 for sm_80/89/90. Running the device tests and the GPU benchmark needs a GPU and driver.

## Build

### Linux (GCC or Clang)

```bash
cmake --preset linux-gcc-release        # or linux-clang-release
cmake --build --preset linux-gcc-release
ctest --preset linux-gcc-release

# Sanitizers (Debug + AddressSanitizer + UBSan, tests only)
cmake --preset linux-gcc-asan && cmake --build --preset linux-gcc-asan && ctest --preset linux-gcc-asan

# CUDA kernels required (fails if nvcc >= 12 is not found)
cmake --preset linux-cuda && cmake --build --preset linux-cuda && ctest --preset linux-cuda
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
| `HLKV_BUILD_BENCHMARKS` | ON | Google Benchmark harnesses |
| `HLKV_CUDA` | `AUTO` | `AUTO`: build the CUDA kernels when a working nvcc ≥ 12 is found; `ON`: require it; `OFF`: never |
| `HLKV_CUDA_VERBOSE_PTXAS` | OFF | `-Xptxas=-v` (registers/spills per kernel) |
| `CMAKE_CUDA_ARCHITECTURES` | `80;89;90` | when CUDA is enabled |
| `HLKV_WARNINGS_AS_ERRORS` | OFF (ON in presets/CI) | `-Werror` / `/WX` / `--Werror=all-warnings` on HL-KV targets |
| `HLKV_SANITIZE` | empty | e.g. `address,undefined` (GCC/Clang) |

With a non-standard CUDA location pass `-DCMAKE_CUDA_COMPILER=<cuda>/bin/nvcc` (and, if needed,
`-DCUDAToolkit_ROOT=<cuda>`).

## Tests

* `hl_allocator_tests` — ABI layout (golden bytes), table behaviour, id arena, and a randomized
  differential test of the delta protocol against the full rebuild.
* `hl_quant_tests` — exhaustive FP16/FP8 decode and encode, accuracy vs FP32 targets, argument
  validation.
* `hl_quant_cuda_tests` (CUDA builds) — `HostEmulation.*` run the kernel's arithmetic on the CPU and
  always run; `Device.*` run the real kernels and **skip themselves when no GPU is present**.

## Benchmarks

```bash
taskset -c 2 ./build/linux-gcc-release/tests/hl_allocator_bench \
  --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
./build/linux-cuda/tests/hl_quant_cuda_bench --benchmark_repetitions=5   # needs a GPU
```

Each `BM_Hetero_*` case has a `BM_Baseline_*` twin (minimal homogeneous block pool on the same id
arena). Results and their caveats are recorded in [`docs/benchmarks/`](docs/benchmarks/); numbers
are only reported from runs that were actually executed, with host and compiler details.

## Documentation

* [`docs/architecture/hl_allocator_abi.md`](docs/architecture/hl_allocator_abi.md) — byte layout, state machine, delta protocol, contracts, limitations
* [`docs/architecture/hl_quant_formats.md`](docs/architecture/hl_quant_formats.md) — format definitions, rounding, CUDA kernel, correctness method
* [`docs/decisions/`](docs/decisions/) — architecture decision records (ADR-0001 … ADR-0005)
* [`docs/benchmarks/`](docs/benchmarks/) — dated experiment logs
