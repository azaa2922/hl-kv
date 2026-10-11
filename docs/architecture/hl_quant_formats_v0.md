# hl_quant — KV storage formats v0 (CPU reference)

Status: **CPU reference encoder/decoder implemented and tested. No GPU kernels yet** (reference-first
rule). Source: `project2_quant/`.

## 1. Formats

| Code | Name | Element | Decode (binary32, RNE) | Params per group |
|---|---|---|---|---|
| 0 | `FP16` | IEEE 754 binary16, 2 B little-endian | `x = decode(q)` | none (`group_size = 0`) |
| 1 | `FP8_E4M3` | OCP OFP8 E4M3 ("E4M3FN"), 1 B | `x = decode(q) * scale` | scale (bias must be 0) |
| 2 | `FP8_E5M2` | OCP OFP8 E5M2, 1 B | `x = decode(q) * scale` | scale (bias must be 0) |
| 3 | `INT4_PACKED` | unsigned 4-bit, 2 / byte | `x = fma(float(q), scale, bias)` | scale, bias |
| 4 | `INT2_PACKED` | unsigned 2-bit, 4 / byte | `x = fma(float(q), scale, bias)` | scale, bias |

The code is the value stored in `hl_allocator`'s 4-bit `format` field.

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
supported.

### 1.2 Packing

* INT4: element `i` → byte `i/2`, bits `[4*(i%2), 4*(i%2)+3]` (element `2k` in the low nibble).
* INT2: element `i` → byte `i/4`, bits `[2*(i%4), 2*(i%4)+1]`.
* `group_size` must be a multiple of the elements per byte, so one packed byte never needs two
  parameter sets. A partial last group and a partial last byte are allowed; padding bits are written
  as zero and ignored on decode.
* `params.size()` must equal `ceil(n / group_size)` exactly.

## 2. Reference encoder (policy, not part of the format)

| Format | Rounding | Overflow / special values | Parameters |
|---|---|---|---|
| FP16 | RNE | overflow → ±Inf; NaN → `0x7E00` (sign kept) | — |
| E4M3 | RNE | saturating: beyond 448 and ±Inf → ±448; NaN → `S.1111.111` | `scale = max(amax/448, FLT_MIN)`, 1 if amax = 0 |
| E5M2 | RNE | saturating: beyond 57344 and ±Inf → ±57344; NaN → `S.11111.11` | `scale = max(amax/57344, FLT_MIN)` |
| INT4/INT2 | RNE in code space (`nearbyint`, default rounding mode assumed) | clamp to `[0, 2^b−1]` | `scale = (max−min)/(2^b−1)`, `bias = min`; constant group → scale 0 (exact) |

Quantization of FP8/INT inputs requires finite values (`kInvalidArgument` otherwise, outputs
untouched). FP16 accepts Inf/NaN.

**UNVERIFIED:** the saturating FP8 behaviour is intended to match PTX `cvt.rn.satfinite.{e4m3x2,e5m2x2}.f32`
(and `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, ...)`). This has not been checked against device
output; it must be the first test of the GPU phase. NaN payload canonicalisation may differ between
implementations (all NaN encodings decode to NaN).

## 3. Why these semantics (summary; see ADR-0002)

* `fma` (single rounding) for INT decode makes CPU and GPU (`__fmaf_rn`) results bit-comparable.
  With a two-step `q*scale + bias` the result would depend on the compiler's FP-contraction setting.
* `(scale, bias)` covers both symmetric signed schemes (`bias = −2^(b−1)·scale`) and min/max
  asymmetric schemes (`bias = min`) with the same branch-free instruction.
* FP8 is multiply-only (no bias) — matches how FP8 KV caches are scaled in practice and preserves −0.

## 4. Correctness method

Tests: `tests/project2/test_dequant.cpp`, `tests/project2/test_quant_encode.cpp`.

1. **Exhaustive decode**: all 65536 FP16 codes and all 256 codes of each FP8 format are compared
   against an independent double-precision oracle (`ldexp` of the textbook formula), including the
   sign of zero and NaN/Inf classification.
2. **Exhaustive encode**: every finite code round-trips; for every pair of adjacent codes the exact
   midpoint rounds to the even code and the neighbouring binary32 values round away from the tie
   (validates RNE, subnormals, binade carries). Overflow, saturation, underflow and signed zero are
   tested explicitly.
3. **Independent FP16 cross-check** against the compiler's `_Float16` conversion on 2M random bit
   patterns (GCC/Clang only; skipped on MSVC).
4. **Accuracy vs FP32 targets**: Gaussian inputs, quantize → dequantize, every element must be within
   the analytical bound (FP: half a spacing in the scaled domain; INT: `scale/2`), plus a few ulps of
   binary32 slack. Error statistics are printed (see `docs/benchmarks/2026-10-11-p2-cpu-reference-error.md`).
5. **Packing, parameter semantics, edge shapes** (odd n, partial groups/bytes, n = 0) and argument
   validation (outputs untouched on every error).

All of the above pass under GCC 13 (Release and ASan+UBSan Debug), Clang 18 (Release) and MSVC 2022
(Release, CI; the `_Float16` cross-check is skipped there).

## 5. CUDA toolchain contract

`-DHLKV_ENABLE_CUDA=ON` enables the CUDA language (C++20, CUDA ≥ 12) and defines the interface
target `hl::quant_cuda_options` with `--expt-relaxed-constexpr -lineinfo` (and `-Xptxas=-v` with
`HLKV_CUDA_VERBOSE_PTXAS=ON` for registers/spills). `--use_fast_math` is deliberately excluded (it
flushes subnormals and approximates division, breaking bit-exact comparison with this reference).
Default architectures: 80, 89, 90. **Not exercised in this repository yet** — the container used for
development has no CUDA toolkit or GPU.

## 6. Limitations

* BF16 and INT8 are not defined (not requested for this phase; add only with a measured reason).
* Parameters are binary32 in the reference API; the on-device parameter encoding (fp16? in-payload?)
  is undecided and affects effective bits/element.
* Error statistics so far use synthetic i.i.d. Gaussian data, which lacks the channel outliers of
  real K caches. Real KV tensors and attention-output error are required before any format decision.
* Reference code is scalar and unoptimised by design.
