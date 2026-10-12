# Experiment log — P1 hl_allocator CPU microbenchmarks

> **Erratum (2026-10-12).** The *Decode step* table below was measured before steady state: every
> sequence started at one block and grows one block per 16 steps, so the table size depended on the
> number of iterations run, and the full-rebuild cost at long contexts is understated. The harness
> now warms up to steady state; see `2026-10-12-p1-persistent-exec-table-cpu.md`, which supersedes
> those rows (it was measured on a different machine). The other tables are unaffected.

| | |
|---|---|
| Date | 2026-10-11 |
| Commit | the commit that adds this file (benchmark and allocator sources unchanged since the run) |
| Project | P1 `hl_allocator` |
| Host | cloud VM (KVM), 4 vCPU "Intel(R) Xeon(R) Processor @ 2.10GHz", L1d 48 KiB, L2 2 MiB per core, L3 260 MiB shared (as reported by Google Benchmark / lscpu) |
| OS | Linux 6.18 |
| Compiler | GCC 13.3.0, `-O3 -DNDEBUG -std=c++20` (CMake Release) |
| Harness | Google Benchmark v1.9.1, `tests/project1/bench_allocator.cpp` |
| GPU / CUDA | none (CPU-only experiment) |
| Command | `taskset -c 2 build/gcc/tests/hl_allocator_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true` |
| Runs | 5 repetitions per case; **median** reported; CV = coefficient of variation of wall time across the 5 |

**Environment caveat.** A shared, virtualised 4-vCPU host. CVs up to 24 % were observed. Treat the
numbers as ratios between two implementations measured back-to-back on the same core, not as
absolute properties of HL-KV. Re-measure on the target serving host before using absolute values.

## Hypothesis

Heterogeneous per-block metadata (32 B host record, 16 B device record) can be maintained at a CPU
cost that is small relative to a minimal homogeneous design.

## Baseline

`HomogeneousBlockPool` (`tests/project1/baseline_block_pool.*`): uint16 refcount per block, LIFO free
stack, per-sequence `int32` block lists, and a CSR int32 block table built by **bulk copy** — the
representation consumed by today's paged-attention kernels. It is a *lower bound* on homogeneous
bookkeeping cost (no stale-handle detection, no valid-token tracking, no format/residency state); it
is **not** vLLM's actual (Python) block manager. Both implementations are compiled in separate
translation units, so neither gets cross-TU inlining the other lacks.

## Results (median; ns per item unless noted; ratio = hetero / baseline)

### Allocation churn — steady state at 50 % occupancy, one allocate + one release of a random live block

| Table size (blocks) | Hetero ns/op | Baseline ns/op | Ratio | Hetero ops/s | CV H / B |
|---|---|---|---|---|---|
| 4 096 | 12.6 | 3.05 | 4.1× | 79.1 M | 13 % / 10 % |
| 65 536 | 13.0 | 3.03 | 4.3× | 77.0 M | 24 % / 6 % |
| 1 048 576 | 50.9 | 5.64 | 9.0× | 19.6 M | 14 % / 2 % |
| 16 777 216 | 86.8 | 25.2 | 3.4× | 11.5 M | 15 % / 9 % |

### Single validated lookup — independent random order (throughput-bound)

| Table size | Hetero table | Hetero ns | Baseline ns | Ratio |
|---|---|---|---|---|
| 4 096 | 128 KiB | 2.95 | 1.33 | 2.2× |
| 65 536 | 2 MiB | 4.47 | 1.27 | 3.5× |
| 1 048 576 | 32 MiB | 9.65 | 2.41 | 4.0× |
| 16 777 216 | 512 MiB | 29.2 | 6.58 | 4.4× |

### Single validated lookup — dependent chain (next id derived from the loaded record; latency-bound)

| Table size | Hetero ns | Baseline ns | Ratio |
|---|---|---|---|
| 4 096 | 11.4 | 4.65 | 2.4× |
| 65 536 | 24.9 | 7.17 | 3.5× |
| 1 048 576 | 138 | 20.3 | 6.8× |
| 16 777 216 | 216 | 137 | 1.6× |

### Batch lookup — validated gather of 1024 random blocks into a flat array

| Table size | Hetero ns/block | Baseline ns/block | Ratio |
|---|---|---|---|
| 4 096 | 2.33 | 0.48 | 4.9× |
| 65 536 | 2.95 | 0.56 | 5.3× |
| 1 048 576 | 8.40 | 1.22 | 6.9× |
| 16 777 216 | 21.3 | 6.81 | 3.1× |

### Exec-table / block-table build — batch of sequences, scattered block ids (16 tokens/block)

| Sequences × blocks | Tokens/seq | Hetero µs | Baseline µs | Hetero ns/block | Ratio | CV H / B |
|---|---|---|---|---|---|---|
| 64 × 64 | 1 024 | 7.92 | 0.47 | 1.90 | 17× | 5 % / 18 % |
| 256 × 64 | 1 024 | 31.1 | 2.83 | 1.87 | 11× | 7 % / 4 % |
| 256 × 256 | 4 096 | 186 | 9.64 | 2.82 | 19× | 2 % / 3 % |
| 32 × 2048 | 32 768 | 190 | 7.54 | 2.82 | 25× | 5 % / 2 % |

### Decode step — every sequence +1 token; seal/append on block boundary; sequence turnover at a random target length; full table rebuild

| Sequences / max blocks per seq | Hetero µs/step | Baseline µs/step | Ratio | CV H / B |
|---|---|---|---|---|
| 64 / 128 | 5.34 | 0.68 | 7.8× | 10 % / 11 % |
| 256 / 128 | 21.9 | 3.27 | 6.7× | 17 % / 9 % |
| 256 / 512 | 93.6 | 7.25 | 12.9× | 8 % / 6 % |

## Interpretation

* **The heterogeneous table is 2–25× slower than the minimal homogeneous baseline on every
  operation measured.** The core Project 1 question (is the CPU overhead acceptable?) is therefore
  *not yet answered positively*; it depends on the GPU step time, which has not been measured.
* The largest and most structural gap is **exec-table construction**: the baseline's per-sequence
  int32 list already is the kernel format and is copied in bulk (~0.1–0.2 ns/block), while the
  heterogeneous table gathers a 16 B record through an id indirection and checks residency
  (~1.9–2.8 ns/block). This cost dominates the decode-step benchmark and scales with *total* blocks,
  not with the number of blocks that changed.
* For table sizes plausible per GPU (ENGINEERING INFERENCE: a Llama-3-8B-class model with ~60 GiB of
  FP16 KV holds roughly 0.5 M tokens ≈ 31 K blocks of 16 tokens), hetero lookups cost ~3–5 ns and
  churn ~13 ns per operation — small in absolute terms. The 16 M-block rows are a stress point beyond
  that regime and mostly measure memory latency (512 MiB hetero vs 32 MiB baseline footprint).
* Metadata *capacity* overhead is negligible (≈1 B/token device-side, see ABI doc §7); the cost of
  heterogeneity on the host is CPU time, not bytes.

## Decision / next action

1. Replace the full per-step rebuild with a **persistent, incrementally patched exec table**
   (append on new block, patch `valid_tokens` of the last block, patch on format/residency change;
   shared blocks need a reverse index or an epoch check). Success criterion: decode-step cost
   proportional to sequences, not total blocks; re-run `BM_*_DecodeStep` and `BM_*_BuildExecTable`.
2. Measure real decode-step GPU time (Project 3/5) before optimising further: only then can the
   µs figures above be judged acceptable or not.
3. Re-run on the intended serving host (bare metal, fixed frequency) with `--benchmark_repetitions≥10`.
