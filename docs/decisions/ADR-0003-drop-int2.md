# ADR-0003 — Remove INT2 from the supported KV formats

Date: 2026-10-12 · Status: accepted · Amends: ADR-0002

## Decision

Remove `INT2_PACKED` from `hl::quant::QuantFormat`, the CPU reference quantizer/dequantizer and the
tests. Retire format code 4: it is rejected as `kUnsupportedFormat` and must not be reused without
an ABI version bump. Supported formats: FP16, FP8 E4M3, FP8 E5M2, INT4.

## Why it was considered

The 2026-10-11 reference-error measurement (`docs/benchmarks/2026-10-11-p2-cpu-reference-error.md`)
found INT2 with min/max groups at 33–50 % relative RMSE (≈6–10 dB SNR) on Gaussian data, and
dominated by INT4 at every parameter-inclusive bit budget measured (6 bits: INT4/g32 7.8 % vs
INT2/g16 32.7 %). The master plan's failure rule says not to keep a format for narrative reasons.

## Alternatives

| Alternative | Why not now |
|---|---|
| Keep INT2 behind a flag | Every supported format multiplies kernel variants, tests and dispatch cost in Project 4. |
| Improve INT2 first (clipping / MSE-optimal ranges, per-channel K grouping, high-precision residual window, as in published 2-bit KV schemes) | Possible, but there is no evidence yet that any 2-bit policy beats INT4 at equal bits on real KV data. |

## Evidence and its limits

The evidence is a **research result on synthetic i.i.d. Gaussian data with a simple min/max
policy**, not a measurement on real K/V tensors or of attention-output / model quality. Real KV
caches have structure (channel outliers in K, token locality) that both helps and hurts 2-bit
schemes. This decision is therefore reversible.

## Revisit conditions

Re-introduce a 2-bit format only with: real K/V tensors from at least two model families, a
non-trivial quantization policy, and attention-output error at an equal bit budget that is
competitive with INT4. It would take a new format code, not code 4.
