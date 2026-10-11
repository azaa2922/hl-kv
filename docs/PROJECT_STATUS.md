# HL-KV PROJECT STATUS

_Last updated: 2026-10-11_

**Current Project:** P1 (finalize) + P2 (scaffolding)

**Project Objective:**
P1 — a heterogeneous KV block representation whose metadata is compact, deterministic and cheap to
maintain. P2 — exact numerical definitions and a CPU reference for the candidate KV formats before
any GPU kernel exists.

**Current Hypothesis:**
Per-block format/residency metadata (16 B device / 32 B host per block) is cheap enough for a
serving scheduler; low-bit formats can be decoded with one branch-free FMA per element.

**Current Implementation:**
- `hl_allocator`: `BlockExecDescriptor` (16 B) / `BlockDescriptor` (32 B), `HeteroBlockTable` with
  per-format slot pools, generation-checked handles, refcounted prefix sharing, two-phase format
  change, CSR exec-table gather, `validate()`, `stats()`.
- `hl_quant`: FP16 / FP8 E4M3 / FP8 E5M2 / INT4 / INT2 format definitions; exact CPU reference
  decoders and RNE encoders; optional CUDA toolchain contract (no kernels).

**Baseline:**
P1 — minimal homogeneous int32 block pool (lower bound on homogeneous bookkeeping; not vLLM itself).
P2 — FP32 targets; independent double-precision decode oracle; compiler `_Float16`.

**Best Measured Result:** (CPU VM, GCC 13, medians of 5 — `docs/benchmarks/2026-10-11-p1-allocator-cpu.md`)
- Hetero is 2–25× slower than the homogeneous baseline in every P1 microbenchmark.
- Decode-step metadata update + full exec-table rebuild, 256 sequences: 21.9 µs (≤2 K tokens/seq)
  to 93.6 µs (≤8 K tokens/seq), vs 3.3–7.3 µs baseline.
- P2 (synthetic Gaussian): rel. RMSE E4M3 2.2–2.7 %, INT4 6.5–10 %, INT2 33–50 %.

**Correctness Status:**
67 unit tests pass on GCC 13 (Release; Debug+ASan+UBSan) and Clang 18 (Release); clang-tidy
(bugprone/performance/analyzer) clean on library sources. CI (`.github/workflows/ci.yml`) builds with
warnings-as-errors and passes all tests on Ubuntu (GCC, Clang, GCC+ASan/UBSan) and Windows (MSVC 2022).

**Performance Status:**
P1 measured on CPU only. P2 has no performance measurements (no GPU code).

**Known Risks:**
- Full exec-table rebuild is O(total blocks) per step — the dominant P1 cost.
- Per-format pools strand capacity across formats.
- INT2 looks dominated on synthetic data; may not survive real-KV evaluation.
- FP8 saturation semantics not yet checked against device conversion instructions.
- Layer association (one table for all layers vs one per layer) undecided.

**Open Research Questions:**
- Does vLLM / SGLang / FlashInfer / TensorRT-LLM already express per-block heterogeneous precision?
  (Not researched in this session — must be verified against current source before claiming novelty.)
- Real decode-step GPU time vs the µs-scale metadata cost measured here.
- Real K/V error and attention-output error per format.

**Next Engineering Objective:**
1. P1: persistent, incrementally patched exec table; re-run DecodeStep/BuildExecTable.
2. P2: capture real K/V tensors; measure attention-output error; first CUDA dequant kernel validated
   bit-exactly against the CPU reference (needs a CUDA 12 / sm_89+ machine).

**Definition of Done (P1):**
Incremental exec table implemented; decode-step cost measured against a real GPU step time; ABI v0
frozen or revised; current-state research on existing block managers recorded.

**Do Not Build Yet:**
Fused attention (P4), TMA paths, framework integration (P5), BF16/INT8 formats, multi-threaded allocator.
