# Experiment log — P2 hl_quant CPU reference: quantization error on synthetic data

| | |
|---|---|
| Date | 2026-10-11 |
| Commit | the commit that adds this file |
| Project | P2 `hl_quant` (CPU reference only) |
| Host / compiler | same VM as the P1 log; GCC 13.3.0 Release. Results are deterministic given the inputs; the Gaussian generator uses libm `log`/`cos`, so other platforms may differ in the last digits. |
| Source | `tests/project2/test_dequant.cpp`, test suite `DequantAccuracy` (prints `[ error ]` lines) |
| Command | `build/gcc/tests/hl_quant_tests --gtest_filter='DequantAccuracy.*'` |
| Data | 65 536 i.i.d. N(0, 1) samples (FP16 row: N(0, 100²)), SplitMix64 + Box–Muller, fixed seeds |
| Policy | encoder policy of `cpu_quant.hpp`: FP8 amax scaling, INT min/max per group |
| GPU | none |

This is a **numerical-property measurement of the reference**, not a model-quality result. Real K/V
tensors have channel outliers and token structure that i.i.d. Gaussian data lacks.

## Results

`rel. RMSE` = RMSE / RMS(signal). `bits/elem` = element bits + binary32 parameters amortised over the
group (64 bits/group for INT, 32 for FP8) — the device parameter encoding is not decided, so this
column is an upper bound for a given group size.

| Format | Group | bits/elem | max abs err | mean abs err | rel. RMSE | worst err / analytical bound |
|---|---|---|---|---|---|---|
| FP16 | — | 16 | 0.125 (σ=100) | 1.41e-2 | 2.07e-4 | 0.994 |
| FP8 E4M3 | 16 | 10.0 | 0.131 | 1.50e-2 | 2.22e-2 | 0.941 |
| FP8 E4M3 | 128 | 8.25 | 0.132 | 1.75e-2 | 2.56e-2 | 0.941 |
| FP8 E4M3 | per-tensor | ≈8 | 0.154 | 1.80e-2 | 2.65e-2 | 0.941 |
| FP8 E5M2 | 16 | 10.0 | 0.254 | 3.01e-2 | 4.45e-2 | 0.889 |
| FP8 E5M2 | 128 | 8.25 | 0.279 | 3.49e-2 | 5.13e-2 | 0.889 |
| FP8 E5M2 | per-tensor | ≈8 | 0.308 | 3.59e-2 | 5.28e-2 | 0.889 |
| INT4 | 16 | 8.0 | 0.225 | 5.14e-2 | 6.51e-2 | 1.000 |
| INT4 | 32 | 6.0 | 0.226 | 6.47e-2 | 7.79e-2 | 1.000 |
| INT4 | 64 | 5.0 | 0.258 | 7.57e-2 | 8.93e-2 | 1.000 |
| INT4 | 128 | 4.5 | 0.253 | 8.47e-2 | 9.99e-2 | 1.000 |
| INT2 | 16 | 6.0 | 1.14 | 0.259 | 0.327 | 1.000 |
| INT2 | 32 | 4.0 | 1.11 | 0.324 | 0.390 | 1.000 |
| INT2 | 64 | 3.0 | 1.31 | 0.383 | 0.450 | 1.000 |
| INT2 | 128 | 2.5 | 1.27 | 0.425 | 0.500 | 1.000 |

Every element of every case is within its analytical bound (`over_bound = 0`; the last column is the
largest observed error divided by its bound, 1.000 meaning the bound is attained by exact ties).

## Interpretation (RESEARCH RESULT on synthetic data — not yet validated on real KV)

* At an equal parameter-inclusive bit budget, the measured ordering on Gaussian data is
  FP8 E4M3 > INT4 > INT2:
  * ~8 bits: E4M3 per-tensor 2.7 % vs INT4/g16 6.5 % rel. RMSE.
  * 6 bits: INT4/g32 7.8 % vs INT2/g16 32.7 %.
* **INT2 with min/max groups is dominated** at every budget measured (33–50 % rel. RMSE, i.e. a
  signal-to-noise ratio of roughly 6–10 dB). Per the master plan's failure rule, INT2 should not be
  carried forward on narrative grounds; it needs either a better policy (clipping/MSE-optimal
  ranges, per-channel K grouping, residual high-precision window) *and* real-KV evidence, or it
  should be dropped.
* E5M2 has about twice the error of E4M3 at the same size; its wider range is not needed after
  per-group/per-tensor scaling of this data.

## Next measurements required

1. Real K/V tensors captured from at least two model families (GQA and MLA-style), per layer.
2. Error of the **attention output** (not just of K/V) against an FP32 reference.
3. Same table with fp16 (or in-payload) parameter encoding to get the true bits/element.
