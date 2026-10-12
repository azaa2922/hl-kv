# ADR-0004 — First CUDA kernel: FP8 → FP16 dequantization

Date: 2026-10-12 · Status: accepted (implemented; device execution not yet verified)

## Decision

* Contract: `out = fp16_rne(decode_fp8(q) * scale)`, i.e. the binary32 reference value rounded to
  nearest-even binary16 — two roundings by definition. A new CPU function,
  `dequantize_reference_to_fp16`, defines it, and GPU output must match it bit for bit (NaN payloads
  excepted).
* FP8 decode uses NVIDIA's `cuda_fp8.h` conversions (`__nv_cvt_fp8x2_to_halfraw2`,
  `__nv_cvt_fp8_to_halfraw`) rather than custom bit manipulation; FP32 multiply; `__float2half_rn` /
  `__float22half2_rn` for the final rounding.
* Two kernels: an 8-elements-per-thread path (8-B load, 16-B store) when alignment and group size
  allow, and a scalar path for everything else and the tail. Grid-stride loops; 32-bit group-index
  division whenever n < 2^32.
* The per-element code is `__host__ __device__`, and the library exports a host emulation of both
  paths so the arithmetic is tested even without a GPU.
* Parameters are passed as the existing `AffineParams` array in device memory; FP8 reads `.scale`
  only (bias is 0 by definition and cannot be validated on the device).
* CMake: `HLKV_CUDA=AUTO|ON|OFF`, where AUTO builds the kernels when a working nvcc ≥ 12 is present.

## Alternatives

| Alternative | Why not |
|---|---|
| Single rounding (exact product → FP16) | Needs wider arithmetic; no practical GPU kernel computes it, so a reference built on it could not be matched bit for bit. |
| Multiply in FP16 (`__hmul2` with an FP16 scale) | Rounds the scale to FP16 and the product in FP16: larger error and a different contract. |
| Custom bit-twiddling FP8 decode | `cuda_fp8.h` is NVIDIA-maintained, uses the hardware `cvt` on sm_89+ and has a host implementation. The CPU reference stays an independent implementation, so the comparison still means something. |
| Byte-per-thread only | Simpler, but byte loads and 2-byte stores waste memory transactions on a bandwidth-bound kernel. The 8-element path should be measurably better; that is to be confirmed on a GPU. |
| 16 elements per thread (16-B load, 32-B output as two 16-B stores) | Each warp store instruction would write 16 B per thread at a 32-B stride; with 8 elements per thread every store instruction covers a contiguous 512-B range. |
| Triton | Compute is trivial; the open questions are memory layout, exact rounding and launch structure, which CUDA C++ exposes directly and which Project 4 will need. |

## Evidence

* Host emulation: bit-exact against the CPU reference on all 256 codes × 16 scales × 2 formats × 2
  paths, and on random shapes and group sizes. Deliberately introduced bugs in the 8-element path
  (swapped element order, a 1e-4 scale perturbation) are caught.
* ptxas (CUDA 12.6.85, sm_80/89/90): 12–26 registers, 0 spills, 0 stack, no shared memory.
* **Not yet**: execution on a GPU, achieved bandwidth, occupancy. The device tests and benchmark
  harness exist but have not been run.

## Known risks

* Device `cvt` instructions or `__float22half2_rn` could differ from the host emulation in edge cases
  (subnormals, NaN). The device tests compare against the CPU reference directly, not against the
  emulation.
* FP16 output overflows to Inf for |value| > 65504 (large scales).
* `cudaGetLastError` after the launch may report an earlier pending asynchronous error.

## Revisit conditions

* Device results differ from the reference → decide which side is wrong; update this ADR.
* Measured bandwidth far below the device's achievable DRAM bandwidth → profile (coalescing,
  instruction mix, grid size) before changing the design.
* Project 4 needs dequantization fused into attention rather than materialized FP16 → this kernel
  becomes the reference/baseline for that fusion.
