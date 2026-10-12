# HL-KV PROJECT STATUS

_Last updated: 2026-10-12_

**Current Project:** P1 (finalize) + P2 (first GPU kernel)

**Project Objective:**
P1 — a heterogeneous KV block representation whose metadata is compact, deterministic and cheap to
maintain per scheduler step. P2 — exact numerical definitions, a CPU reference, and GPU kernels that
match it, for the KV formats that survive measurement.

**Current Hypothesis:**
Per-block format/residency metadata can be maintained with per-step work proportional to what
changed; FP8 KV data can be dequantized on the GPU close to DRAM bandwidth with bit-exact,
reference-defined results.

**Current Implementation:**
- `hl_allocator` (ABI v1): 16 B device / 32 B host block records; `HeteroBlockTable` with per-format
  slot pools, generation-checked handles, prefix sharing, two-phase format change; persistent device
  state (descriptor table by block id + per-sequence id arena) with delta collection and O(sequences)
  batch segments; full-rebuild path kept as the oracle.
- `hl_quant`: FP16 / FP8 E4M3 / FP8 E5M2 / INT4 (INT2 removed, code 4 retired); exact CPU reference
  incl. FP16-output reference; first CUDA kernel FP8 → FP16 (vectorized + scalar paths), host
  emulation of its arithmetic; `HLKV_CUDA=AUTO|ON|OFF`.

**Baseline:**
P1 — minimal homogeneous pool on the same id arena (full rebuild by bulk copy, and an equally
persistent delta path). P2 — CPU reference (`dequantize_reference_to_fp16`).

**Best Measured Result:** (CPU VM, GCC 13, medians of 5 — `docs/benchmarks/2026-10-12-p1-persistent-exec-table-cpu.md`)
- Hetero decode step, 256 sequences: 4.4–5.3 µs and ≈6.9 KB upload per step with deltas, flat from
  2K to 32K tokens per sequence; full rebuild 24 µs → 736 µs and 178 KB → 2.7 MB over the same range.
- Hetero vs equally persistent homogeneous baseline: 4.1–4.5× (≈13–16 ns per sequence-step).
- P2: no GPU measurement. Kernel compiles for sm_80/89/90 with 12–26 registers, no spills.

**Correctness Status:**
CPU: 78 tests pass on GCC 13 (Release; Debug+ASan+UBSan) and Clang 18; includes a differential test
of the delta protocol against the full rebuild. CUDA build: 3 host-emulation tests pass (bit-exact vs
the reference); 5 device tests are skipped (no GPU). CI runs GCC, Clang, ASan, MSVC and a CUDA
compile-and-test job.

**Performance Status:**
P1 measured on CPU only. P2 kernel: static resource usage only; no runtime numbers.

**Known Risks:**
- The FP8 kernel has never executed on a GPU: device conversion instructions, indexing and launch
  geometry are untested.
- The id → descriptor indirection adds a dependent load per block in attention (unmeasured).
- Per-format pools strand capacity across formats; the id arena never compacts.
- FP16 output overflows to Inf for large scales.
- Error data is synthetic Gaussian only (INT2 was dropped on that basis — reversible, ADR-0003).

**Open Research Questions:**
- Does vLLM / SGLang / FlashInfer / TensorRT-LLM already express per-block heterogeneous precision?
  (Not yet researched — required before claiming novelty.)
- GPU step time vs the µs-scale host metadata cost.
- Real K/V error and attention-output error per format.

**Next Engineering Objective:**
1. Run the CUDA device tests and `hl_quant_cuda_bench` on sm_89/sm_90 and sm_80 GPUs; profile with
   Nsight Compute.
2. Device-side delta application (scatter kernels) and a measurement of the descriptor indirection.
3. Capture real K/V tensors; measure attention-output error for FP8 and INT4.

**Definition of Done (P2, first kernel):**
Device tests pass on GPUs with and without FP8 hardware conversion; achieved bandwidth measured and
compared with the device's achievable DRAM bandwidth; results logged.

**Do Not Build Yet:**
Fused attention (P4), TMA paths, framework integration (P5), BF16/INT8 formats, INT4 GPU kernel
(after FP8 is measured), multi-threaded allocator.
