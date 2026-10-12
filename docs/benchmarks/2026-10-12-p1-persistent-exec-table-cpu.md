# Experiment log — P1 persistent exec table (delta protocol), CPU

| | |
|---|---|
| Date | 2026-10-12 |
| Commit | the commit that adds this file (benchmark, allocator and baseline sources unchanged since the run) |
| Project | P1 `hl_allocator`, ABI v1 |
| Host | cloud VM (KVM), 4 vCPU "Intel(R) Xeon(R) Processor @ 2.80GHz", L1d 32 KiB, L2 1 MiB per core, L3 33 MiB shared. **Not the same machine as the 2026-10-11 log** (2.1 GHz, 2 MiB L2, 260 MiB L3): compare only within this run. |
| OS | Linux 6.18 |
| Compiler | GCC 13.3.0, `-O3 -DNDEBUG -std=c++20` (CMake Release) |
| Harness | Google Benchmark v1.9.1, `tests/project1/bench_allocator.cpp` |
| GPU / CUDA | none (CPU-only experiment) |
| Command | `taskset -c 2 build/gcc/tests/hl_allocator_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true` |
| Runs | 5 repetitions per case; **median** reported; CV = coefficient of variation of wall time across the 5 |

**Environment caveat.** Shared virtualised host; CVs up to 26 % observed. Ratios between
implementations measured back-to-back on the same core are the meaningful output.

## Hypothesis

Keeping the device-facing tables persistent and uploading only deltas makes the per-step cost of
heterogeneous metadata proportional to what changed (≈ one record per sequence per step) instead of
to the total number of blocks.

## What is measured

`DecodeStep_*`: one decode step for S sequences. Every sequence gets one token; a full block is sealed
and a new one appended; a sequence that reaches its target length (uniform in 1..max blocks) is
released and replaced. Then:

* `FullRebuild` — hetero: `build_exec_table()` (materialized CSR of 16-B descriptors); baseline: CSR
  of 32-bit ids built by bulk copy.
* `Delta` — hetero: `batch_segments()` + `collect_delta()`; baseline: segment list + id patches +
  per-sequence lengths (homogeneous kernels need `seq_lens` every step).

Both baseline and hetero store sequences in the same `IdArena`. `upload_B_per_step` is the
host→device payload each step produces, computed from the table/patch sizes of the run.

**Harness fix vs 2026-10-11.** The old DecodeStep benchmark started all sequences at one block and
never reached steady state: sequences grow one block per 16 steps, so the measured table size
depended on how many iterations ran. This run adds an untimed warm-up of 2 × max_blocks × 16 steps
and reports `live_blocks` (table size at the end). The 2026-10-11 DecodeStep figures understate the
full-rebuild cost at long contexts and are superseded by this log.

## Results — decode step (median)

| Shape (sequences / max blocks; tokens per seq) | Mode | Hetero µs/step | Baseline µs/step | Hetero ÷ baseline | Hetero upload B/step | Baseline upload B/step | live blocks (H / B) | CV H / B |
|---|---|---|---|---|---|---|---|---|
| 64 / 128 (≤ 2K) | FullRebuild | 6.00 | 0.62 | 9.7× | 44 362 | 11 377 | 3 203 / 3 219 | 9 % / 16 % |
| 64 / 128 (≤ 2K) | Delta | 0.81 | 0.31 | 2.6× | 1 712 | 838 | 2 818 / 2 517 | 4 % / 16 % |
| 256 / 128 (≤ 2K) | FullRebuild | 24.2 | 3.60 | 6.7× | 178 033 | 45 399 | 10 799 / 10 585 | 6 % / 5 % |
| 256 / 128 (≤ 2K) | Delta | 4.42 | 1.08 | 4.1× | 6 780 | 3 354 | 11 678 / 11 561 | 12 % / 8 % |
| 256 / 512 (≤ 8K) | FullRebuild | 157 | 6.95 | 22.6× | 715 938 | 175 991 | 41 407 / 45 118 | 10 % / 3 % |
| 256 / 512 (≤ 8K) | Delta | 4.94 | 1.15 | 4.3× | 6 866 | 3 366 | 44 788 / 43 425 | 11 % / 12 % |
| 256 / 2048 (≤ 32K) | FullRebuild | 736 | 42.9 | 17.2× | 2 684 009 | 710 610 | 169 969 / 182 611 | 6 % / 9 % |
| 256 / 2048 (≤ 32K) | Delta | 5.26 | 1.17 | 4.5× | 6 921 | 3 370 | 182 841 / 176 754 | 9 % / 11 % |

`live_blocks` differ slightly between rows because each benchmark function's repetitions end at a
different point of the random sequence turnover.

**Effect of the delta protocol on the hetero step (same run):**

| Shape | FullRebuild → Delta | Upload per step |
|---|---|---|
| 64 / 128 | 6.00 → 0.81 µs (7.4×) | 44.4 KB → 1.7 KB |
| 256 / 128 | 24.2 → 4.42 µs (5.5×) | 178 KB → 6.8 KB |
| 256 / 512 | 157 → 4.94 µs (32×) | 716 KB → 6.9 KB |
| 256 / 2048 | 736 → 5.26 µs (140×) | 2.68 MB → 6.9 KB |

## Results — other P1 microbenchmarks (median ns per item; same run)

| Benchmark | 4 096 | 65 536 | 1 048 576 | 16 777 216 |
|---|---|---|---|---|
| Hetero alloc+release churn (per op) | 12.5 | 19.8 | 104 | 192 |
| Baseline alloc+release churn | 3.09 | 3.56 | 9.95 | 43.4 |
| Hetero lookup, random independent | 3.04 | 5.27 | 28.2 | 46.8 |
| Baseline lookup, random independent | 1.24 | 1.08 | 3.28 | 18.1 |
| Hetero lookup, dependent chain | 8.76 | 28.7 | 170 | 369 |
| Baseline lookup, dependent chain | 4.09 | 6.05 | 23.2 | 157 |
| Hetero batch lookup (per block) | 1.81 | 3.70 | 19.5 | 34.5 |
| Baseline batch lookup (per block) | 0.59 | 0.68 | 1.82 | 13.9 |

| Reference full build (sequences × blocks) | Hetero µs | Baseline µs | Hetero ns/block |
|---|---|---|---|
| 64 × 64 | 8.30 | 0.50 | 2.00 |
| 256 × 64 | 33.8 | 3.66 | 2.06 |
| 256 × 256 | 214 | 7.77 | 3.24 |
| 32 × 2048 | 210 | 6.22 | 3.18 |

These rows exercise code that the delta protocol did not change (except that sequences now live in
the arena). Absolute values differ from 2026-10-11 because the machine differs.

## Interpretation

* **Hypothesis confirmed on CPU:** with deltas, the hetero step cost (4.4–5.3 µs for 256 sequences)
  and its upload (≈6.9 KB) are flat from 2K to 32K tokens per sequence, while the full rebuild grows
  with total blocks (24 µs → 736 µs, 178 KB → 2.7 MB). The 32K-token case improves by 140× in CPU
  time and 390× in upload bytes.
* **Remaining cost of heterogeneity:** 4.1–4.5× an equally persistent homogeneous baseline, i.e.
  3.3–4.1 µs more per step for 256 sequences (≈13–16 ns per sequence-step). From the operation counts
  (not a profile), most of it is maintaining one 16-B descriptor per sequence per step (the
  `valid_tokens` update and its patch); the baseline sends 4-B lengths instead.
* **Not measured:** the device side — applying patches (two scatter kernels), H2D copy latency, and
  the cost of the extra id → descriptor load inside attention. These need a GPU (Project 3/4).
* Whether 4–5 µs per step is acceptable still depends on the GPU step time, which is unmeasured.
  At these figures it no longer grows with context length.
