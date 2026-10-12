# ADR-0001 — Block descriptor layout (HL-KV ABI v0)

Date: 2026-10-11 · Status: accepted for v0; record layouts kept in v1. The per-step full rebuild was
replaced by persistent tables with deltas in ADR-0005 (first revisit condition below).

## Decision

* Device-facing `BlockExecDescriptor`: 16 B, `alignas(16)` — `slot`, packed `info`
  (format 4 b, scale_mode 2 b, layout 4 b, valid_tokens 10 b), `scale_ref`, `block_id`.
* Host `BlockDescriptor`: 32 B, `alignas(32)` = exec descriptor + `generation`, packed `control`
  (state 2 b, residency 2 b, pending_format 4 b, refcount 16 b), `pending_slot`, 4 reserved bytes.
* Explicit shift/mask packing on `uint32_t` words; no C++ bit-fields.
* One fixed-size slot pool per format; block identity (id) separated from physical slot.
* Two-phase format change (begin → convert → commit/abort) with both slots live in between.

## Why it was considered

The task required compact, bit-packed, cache-line-aligned metadata able to express per-block
dtype, scale and residency. The thesis (mixed precision inside one sequence) only saves memory if a
re-encoded block actually occupies fewer bytes, and a conversion kernel needs source and destination
alive at the same time.

## Alternatives

| Alternative | Why not (for v0) |
|---|---|
| C++ bit-fields | Allocation order is implementation-defined (MSVC vs GCC/Clang/NVCC); breaks a cross-compiler ABI. |
| Single 16 B record, no host/device split | Not enough room for generation + refcount + state + pending slot alongside slot/format/scale. |
| 64 B record | 2× the footprint; nothing justified the extra bytes. |
| SoA (exec array + control array) | Better for pure gathers (4 records per line instead of 2) but a validated lookup then touches two lines. Kept as a revisit option. |
| Uniform slot size for all formats | Simplest, but re-encoding FP16 → INT4 would free no capacity — defeats the purpose. |
| Scale always inline in the descriptor | Dynamic (GPU-computed) scales would need a device→host copy. `scale_mode` lets formats keep scales in the payload. |
| Store sequence id / layer id in the descriptor | Sequence → block mapping lives in sequence lists; shared blocks have no single owner. Layer association is an open question (ABI doc §8). |

## Evidence

* Layout fixed by `static_assert`s and golden byte tests (`tests/project1/test_block_descriptor.cpp`).
* Behaviour: 35 allocator tests incl. a 20 000-operation randomized stress test with `validate()`.
* Cost: `docs/benchmarks/2026-10-11-p1-allocator-cpu.md` — the hetero table is 2–25× slower than a
  minimal homogeneous int32 baseline in every CPU microbenchmark; the largest gap is the exec-table
  rebuild (gather through ids vs bulk copy).

## Trade-offs

* 4 B/block device table becomes 16 B/block (≈1 B/token at 16 tokens/block). Negligible versus KV
  payload; measurable in CPU gather time.
* 8 of 32 host bytes are reserved/padding forced by the alignment choice.
* Per-format pools introduce cross-pool capacity stranding.

## Known risks

* Full exec-table rebuild per step is `O(total blocks)`; with long contexts this grows with context
  length while the useful per-step change is `O(sequences)`.
* Per-layer heterogeneous formats would multiply both host records and rebuild cost by the layer count.

## Revisit conditions

* If exec-table construction is a measurable fraction of the decode step on a real GPU (Project 3/5),
  implement persistent incremental tables (patch only new/changed entries) or an SoA layout.
* If GPU profiling shows the 16 B load or the `info` decode in the inner loop matters, consider
  pre-bucketing descriptors by format (Project 4 dtype-bucketing) or an 8 B exec record with
  in-payload scales.
* If cross-pool stranding appears in realistic traces, move pools onto a shared arena of large chunks.
