# hl_quant — KV storage formats

Status: **CPU reference encoder/decoder implemented and tested; first CUDA kernel (FP8 → FP16)
implemented, compiled for sm_80/89/90 and validated on the CPU through host emulation — not yet run
on a GPU.** Source: `project2_quant/`.

| Date | Change |
|---|---|
| 2026-10-11 | FP16, FP8 E4M3, FP8 E5M2, INT4, INT2 CPU reference |
| 2026-10-12 | INT2 removed, code 4 retired (ADR-0003); FP16-output reference; FP8 → FP16 CUDA kernel (ADR-0004) |

## 1. Formats

| Code | Name | Element | Decode (binary32, RNE) | Params per group |
|---|---|---|---|---|
| 0 | `FP16` | IEEE 754 binary16, 2 B little-endian | `x = decode(q)` | none (`group_size = 0`) |
| 1 | `FP8_E4M3` | OCP OFP8 E4M3 ("E4M3FN"), 1 B | `x = decode(q) * scale` | scale (bias must be 0) |
| 2 | `FP8_E5M2` | OCP OFP8 E5M2, 1 B | `x = decode(q) * scale` | scale (bias must be 0) |
| 3 | `INT4_PACKED` | unsigned 4-bit, 2 / byte | `x = fma(float(q), scale, bias)` | scale, bias |
| 4 | — | **retired** (was `INT2_PACKED`; rejected as `kUnsupportedFormat`) | | |

The code is the value stored in `hl_allocator`'s 4-bit `format` field. Code 4 must not be reused
without an ABI version bump, so stale data can never be misread.

### 1.1 Floating-point details

| | FP16 | E4M3 | E5M2 |
|---|---|---|---|
| exponent / mantissa bits | 5 / 10 | 4 / 3 | 5 / 2 |
| bias | 15 | 7 | 15 |
| max finite | 65504 (`0x7BFF`) | 448 (`0x7E`) | 57344 (`0x7B`) |
| min subnormal | 2^-24 | 2^-9 | 2^-16 |
| Inf | `S.11111.0…` | none | `S.11111.00` |
| NaN | `S.11111.≠0` | `S.1111.111` only | `S.11111.{01,10,11}` |

E4M3 here is the OCP "FN" variant (no Inf, exponent 15 is finite up to 448). **E4M3FNUZ / E5M2FNUZ**
(bias 8/16, single NaN at `0x80`, used on some AMD hardware) are different formats and are **not**
supported. Every E4M3 and E5M2 value is exactly representable in FP16.

### 1.2 Packing

* INT4: element `i` → byte `i/2`, bits `[4*(i%2), 4*(i%2)+3]` (element `2k` in the low nibble).
* `group_size` must be a multiple of the elements per byte, so one packed byte never needs two
  parameter sets. A partial last group and a partial last byte are allowed; padding bits are written
  as zero and ignored on decode.
* `params.size()` must equal `ceil(n / group_size)` exactly.

### 1.3 FP16 output (`dequantize_reference_to_fp16`)

Kernels that produce binary16 are defined as: the binary32 value above, then round-to-nearest-even
to binary16 (overflow → ±Inf, NaN → NaN). That is two roundings **by definition** — what a GPU
computing in FP32 and converting with `__float2half_rn` produces — so GPU output can be compared bit
for bit. FP16 output cannot represent magnitudes above 65504: large scales overflow to Inf.

## 2. Reference encoder (policy, not part of the format)

| Format | Rounding | Overflow / special values | Parameters |
|---|---|---|---|
| FP16 | RNE | overflow → ±Inf; NaN → `0x7E00` (sign kept) | — |
| E4M3 | RNE | saturating: beyond 448 and ±Inf → ±448; NaN → `S.1111.111` | `scale = max(amax/448, FLT_MIN)`, 1 if amax = 0 |
| E5M2 | RNE | saturating: beyond 57344 and ±Inf → ±57344; NaN → `S.11111.11` | `scale = max(amax/57344, FLT_MIN)` |
| INT4 | RNE in code space (`nearbyint`, default rounding mode assumed) | clamp to `[0, 15]` | `scale = (max−min)/15`, `bias = min`; constant group → scale 0 (exact) |

Quantization of FP8/INT inputs requires finite values (`kInvalidArgument` otherwise, outputs
untouched). FP16 accepts Inf/NaN.

**UNVERIFIED:** the saturating FP8 encoder is intended to match PTX `cvt.rn.satfinite.{e4m3x2,e5m2x2}.f32`
(and `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, ...)`). Not yet checked against device output.
NaN payload canonicalisation differs between implementations (all NaN encodings decode to NaN).

## 3. Why these semantics (summary; see ADR-0002)

* `fma` (single rounding) for INT decode makes CPU and GPU (`__fmaf_rn`) results bit-comparable.
  With a two-step `q*scale + bias` the result would depend on the compiler's FP-contraction setting.
* `(scale, bias)` covers both symmetric signed schemes (`bias = −2^(b−1)·scale`) and min/max
  asymmetric schemes (`bias = min`) with the same branch-free instruction.
* FP8 is multiply-only (no bias) — matches how FP8 KV caches are scaled in practice and preserves −0.

## 4. CPU reference — correctness method

Tests: `tests/project2/test_dequant.cpp`, `tests/project2/test_quant_encode.cpp`.

1. **Exhaustive decode**: all 65536 FP16 codes and all 256 codes of each FP8 format are compared
   against an independent double-precision oracle (`ldexp` of the textbook formula), including the
   sign of zero and NaN/Inf classification.
2. **Exhaustive encode**: every finite code round-trips; for every pair of adjacent codes the exact
   midpoint rounds to the even code and the neighbouring binary32 values round away from the tie.
3. **Independent FP16 cross-check** against the compiler's `_Float16` conversion on 2M random bit
   patterns (GCC/Clang only; skipped on MSVC).
4. **Accuracy vs FP32 targets**: every element within the analytical bound; error statistics printed
   (`docs/benchmarks/2026-10-11-p2-cpu-reference-error.md`).
5. **Packing, parameter semantics, edge shapes and argument validation**, including rejection of the
   retired code 4.

## 5. CUDA kernel: FP8 → FP16 (`hl::quant_cuda`)

API: `hl::quant::cuda::dequantize_fp8_to_fp16()` (`include/hl_quant/cuda_dequant.hpp`), asynchronous
on a caller-provided stream. Output is bit-identical to `dequantize_reference_to_fp16` except NaN
payloads. Design decisions: ADR-0004.

| Path | When | Per thread | Memory pattern |
|---|---|---|---|
| vec8 | input 8-B aligned, output 16-B aligned, `group_size % 8 == 0` or a single group | 8 elements: one 8-B load, one 16-B store | contiguous across the warp: 256 B in / 512 B out per instruction |
| scalar | everything else, and the tail | 1 element | byte loads / 2-byte stores, contiguous across the warp |

Conversion: `cuda_fp8.h` FP8 → FP16 (exact; hardware `cvt` on sm_89+, software path elsewhere),
FP16 → FP32, FP32 multiply by the group scale, `__float2half_rn` / `__float22half2_rn`. Group
indices use 32-bit division when n < 2^32. Grid-stride loops, 256 threads per block, at most 65535
blocks.

Resource usage reported by ptxas (CUDA 12.6.85): 12–26 registers per thread, no spills, no stack, no
shared memory, for all 24 variants (`docs/benchmarks/2026-10-12-p2-fp8-fp16-kernel-build.md`).

### 5.1 Kernel correctness method

Tests: `tests/project2/test_dequant_cuda.cpp`.

* **Host emulation (runs without a GPU):** the kernel's per-element functions are `__host__
  __device__`; `emulate_fp8_to_fp16_on_host()` runs the same source, compiled for the host by nvcc,
  through both the scalar and the 8-element path. Compared bit for bit with the CPU reference on all
  256 codes × 16 scales (including FP16-subnormal and overflowing products, zero, negative and
  `FLT_MIN` scales) × both formats, and on random shapes/group sizes. Mutation check: swapping the
  element order in the 8-element path, or perturbing the scale by 1e-4, makes these tests fail.
* **Device tests (need a GPU; skipped otherwise):** the same comparisons through the real kernels,
  plus misaligned pointers (scalar path), group sizes that are not multiples of 8, a 2^25+13-element
  input (grid-stride loop), a user stream, and argument errors.

What host emulation **cannot** validate: the device conversion instructions (sm_89+ `cvt`),
indexing, launch geometry and alignment-dependent path selection. Those are covered only by the
device tests, which have **not been run** (no GPU in the development environment or in CI).

### 5.2 Build

`HLKV_CUDA=AUTO` (default) builds the kernels when a working nvcc ≥ 12 is found, `ON` requires it,
`OFF` disables it. Compile flags (`hl::quant_cuda_options`): `--expt-relaxed-constexpr -lineinfo`,
optional `-Xptxas=-v` (`HLKV_CUDA_VERBOSE_PTXAS`). `--use_fast_math` is deliberately excluded: it
flushes subnormals and approximates division, breaking bit-exact comparison with the reference.
Default architectures: 80, 89, 90.

## 6. Limitations

* BF16 and INT8 are not defined (add only with a measured reason). FP16 output overflows to Inf for
  large scales; a BF16 output variant would not, if needed.
* Parameters are binary32 `AffineParams` (8 B per group, of which FP8 uses 4); the final on-device
  parameter encoding (fp16? in-payload?) is undecided and affects effective bits/element.
* Error statistics so far use synthetic i.i.d. Gaussian data, which lacks the channel outliers of
  real K caches. Real KV tensors and attention-output error are required before any format decision.
* No GPU performance numbers exist. The benchmark harness (`hl_quant_cuda_bench`) is built but has
  not been run.
