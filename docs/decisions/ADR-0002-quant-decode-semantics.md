# ADR-0002 — Quantized KV decode semantics (hl_quant v0)

Date: 2026-10-11 · Status: accepted for the CPU reference; to be re-confirmed by the first GPU kernel

## Decision

* Five formats: FP16, FP8 E4M3 (OCP "FN"), FP8 E5M2, INT4 packed, INT2 packed.
* Per-group affine parameters `(scale, bias)`:
  FP8 `x = decode(q) * scale` (bias must be 0); INT `x = fma(float(q), scale, bias)`; FP16 unscaled.
* Unsigned integer codes; little-nibble/LSB-first packing; groups aligned to byte boundaries.
* Reference encoder: round-to-nearest-even everywhere; FP8 saturates to max finite; FP16 overflows to Inf.
* Scalar decoders build the binary32 bit pattern directly (no libm, no host/GPU shared code).

## Why it was considered

Project 2 needs exact numerical truth before any CUDA kernel exists, and that truth must be
reproducible bit-for-bit on CPU and GPU, across MSVC and GCC/Clang.

## Alternatives

| Alternative | Why not |
|---|---|
| `x = (q − zero_point) * scale` with integer zero point | Equivalent to `fma(q, scale, −zp·scale)` but forces the zero point onto the integer grid; min/max schemes (KIVI-style) want a real-valued offset. The `(scale, bias)` form covers both. |
| Two-step `q*scale + bias` | Result depends on FP contraction (`-ffp-contract`, nvcc `-fmad`); not bit-reproducible. |
| Signed INT codes | Expressible with unsigned codes + `bias = −2^(b−1)·scale`; one code path instead of two. |
| Groups that may split a packed byte | A byte would need two parameter sets; adds a branch per byte on the GPU. |
| Non-saturating FP8 encode (overflow → NaN/Inf) | A single outlier would poison attention scores; saturating is the conventional KV-cache choice. |
| Share decoder code between CPU reference and CUDA | The reference must be independent of the implementation it validates. |
| E4M3FNUZ / E5M2FNUZ | Different encodings (bias, NaN, no −0); out of scope until an AMD target is chosen. |

## Evidence

* Exhaustive decode (all FP16/FP8 codes) vs an independent double-precision oracle; exhaustive RNE
  midpoint tests; `_Float16` cross-check — `tests/project2/`.
* First error measurements on synthetic Gaussian data:
  `docs/benchmarks/2026-10-11-p2-cpu-reference-error.md`.

## Trade-offs

* `(scale, bias)` in binary32 costs 64 bits/group for INT formats — significant at small groups
  (INT2 with group 16: +4 bits/element). The device parameter encoding is not decided.
* The encoder policy (min/max, amax scaling) is simple, not MSE-optimal.

## Known risks

* Device FP8 conversion instructions may differ from the reference on edge cases (NaN payload,
  saturation of Inf). Unverified until run on sm_89+/sm_90 hardware.
* `nearbyint` depends on the current rounding mode (default round-to-nearest-even assumed).

## Revisit conditions

* GPU results differ from the reference on any code → decide which side is wrong, update this ADR.
* Real KV tensors show outlier structure → per-channel (K) / per-token (V) grouping or clipping.
* Measured attention-output error makes a format non-viable → drop it (INT2 is the first candidate;
  see the error report).
