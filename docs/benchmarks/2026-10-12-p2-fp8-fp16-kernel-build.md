# Experiment log — P2 FP8 → FP16 CUDA kernel: build and validation without a GPU

| | |
|---|---|
| Date | 2026-10-12 |
| Commit | the commit that adds this file |
| Project | P2 `hl_quant`, kernel `project2_quant/src/dequant_kernel.cu` |
| Toolchain | nvcc 12.6.85 (conda-forge CUDA 12.6 packages, SHA-256-verified, extracted into a local prefix; NVIDIA's download host was blocked by the environment's network policy), host compiler GCC 13.3.0 |
| Flags | CMake Release, `-std=c++20`, `--expt-relaxed-constexpr -lineinfo -Xptxas=-v`, `--Werror=all-warnings`, architectures `80;89;90` |
| GPU | **none** — no GPU or driver in the development container |
| Command | `cmake -S . -B build/cuda -DHLKV_CUDA=ON -DHLKV_CUDA_VERBOSE_PTXAS=ON -DCMAKE_CUDA_COMPILER=<prefix>/bin/nvcc -DCUDAToolkit_ROOT=<prefix> ...` then `cmake --build build/cuda` and `build/cuda/tests/hl_quant_cuda_tests` |

## What this log can and cannot show

* **Can:** the kernel compiles warning-free for sm_80/89/90; its static resource usage (from ptxas);
  that its per-element arithmetic, run on the CPU via host emulation, is bit-exact against the CPU
  reference.
* **Cannot:** runtime behaviour on a GPU — correctness of the device instructions, indexing and
  launch geometry, achieved bandwidth, occupancy. **No GPU performance number exists.** The device
  tests and `hl_quant_cuda_bench` were built but skipped or not run.

## ptxas resource usage (static, per kernel variant)

Registers per thread; every variant reports 0 bytes stack frame, 0 spill stores, 0 spill loads,
0 barriers and no shared memory.

| Kernel | Index type | sm_80 | sm_89 | sm_90 |
|---|---|---|---|---|
| scalar, E4M3 | u32 | 12 | 14 | 16 |
| scalar, E4M3 | u64 | 18 | 18 | 22 |
| scalar, E5M2 | u32 | 13 | 14 | 16 |
| scalar, E5M2 | u64 | 18 | 18 | 22 |
| vec8, E4M3 | u32 | 16 | 24 | 26 |
| vec8, E4M3 | u64 | 16 | 22 | 24 |
| vec8, E5M2 | u32 | 24 | 24 | 26 |
| vec8, E5M2 | u64 | 18 | 22 | 24 |

Engineering inference (not measured): at ≤ 26 registers × 256 threads per block, registers do not
limit occupancy on these architectures (65 536 registers per SM). Whether the kernel reaches DRAM
bandwidth must be measured.

## Host-emulation validation

`HostEmulation.*` in `tests/project2/test_dequant_cuda.cpp` (the kernel's `__host__ __device__`
conversion functions compiled for the host by nvcc):

| Test | Coverage | Result |
|---|---|---|
| AllCodesManyScalesBothPaths | 256 codes × 16 scales × {E4M3, E5M2} × {scalar, vec8} | bit-exact vs `dequantize_reference_to_fp16` |
| RandomShapesAndGroups | n ∈ {1, 7, 8, 9, 17, 64, 65, 1000, 4099} × group ∈ {1, 3, 8, 24, 128, n}, random codes and log-uniform signed scales, both paths | bit-exact |
| RejectsBadArguments | FP16 / INT4 / retired code 4, group 0, parameter count, short output | as specified |

Mutation check: swapping the element order in the 8-element path made 256 of 256 elements mismatch.
A 1e-4 relative perturbation of the scale in that path produced detected 1-ulp mismatches. The tests
pass again after restoring the source (byte-identical).

`Device.*` (5 tests): skipped — "CUDA driver version is insufficient for CUDA runtime version" (no
driver present).

## Next measurements required

1. Run `hl_quant_cuda_tests` on at least one sm_89+/sm_90 GPU (device conversion instructions) and
   one sm_80 GPU (software FP8 path).
2. Run `hl_quant_cuda_bench` (achieved bandwidth for 1M / 16M / 268M elements, vectorized vs scalar
   path, per-group vs per-tensor scale) and profile with Nsight Compute (achieved DRAM throughput,
   occupancy, instruction mix).
