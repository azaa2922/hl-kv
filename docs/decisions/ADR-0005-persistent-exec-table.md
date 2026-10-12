# ADR-0005 — Persistent device tables with delta updates (HL-KV ABI v1)

Date: 2026-10-12 · Status: accepted · Amends: ADR-0001 (revisit condition met)

## Decision

Replace the per-step full gather of execution descriptors with persistent device-facing state that
is patched:

* a **descriptor table** of `BlockExecDescriptor`, indexed by block id, holding each block's device
  view (a non-executable record, format 15 and slot `0xFFFFFFFF`, for free or non-device-resident
  blocks);
* an **id arena** where each sequence owns one power-of-two segment of block ids (relocated with
  doubling, free segments reused by size class);
* `collect_delta()` returns self-addressed descriptor patches and `{offset, block_id}` id patches
  (each target at most once, final values); `batch_segments()` returns the per-step
  `{offset, length}` list in O(sequences).

The kernel reads block j of sequence i as `descriptor_table[arena[seg_i.offset + j]]`.
`build_exec_table()` stays as the O(total blocks) reference path and test oracle.

## Why

`docs/benchmarks/2026-10-11-p1-allocator-cpu.md` showed that rebuilding the exec table every step
dominated the hetero decode step and scaled with total blocks rather than with what changed. ADR-0001
named an incremental table as the fix.

## Alternatives

| Alternative | Why not |
|---|---|
| Persistent **materialized** per-sequence descriptor arrays (no indirection on the device) | One 16-B load per block, but every block-level change must patch every sequence that references the block, which needs a block → (sequence, position) reverse index maintained on every append and release. Shared prefix blocks make that O(sharers). |
| Rebuild only the sequences that changed | Every sequence changes every decode step (valid tokens of its last block), so this saves little. |
| Keep valid tokens out of the descriptor and pass `seq_lens` | Loses per-block holes (shared partial blocks, token rollback/eviction) that the ABI supports. |
| Dense dirty *ranges* for upload | Changed entries are scattered (one per sequence), so ranges degenerate to singletons; one patch list plus one scatter kernel is cheaper than many small copies. |

## Evidence

`docs/benchmarks/2026-10-12-p1-persistent-exec-table-cpu.md` (same run, same machine, 256
sequences, steady state):

| Context per sequence | Full rebuild | Delta | Upload per step: full → delta |
|---|---|---|---|
| ≤ 2K tokens | 24.2 µs | 4.4 µs | 178 KB → 6.8 KB |
| ≤ 8K tokens | 157 µs | 4.9 µs | 716 KB → 6.9 KB |
| ≤ 32K tokens | 736 µs | 5.3 µs | 2.68 MB → 6.9 KB |

The delta step cost and upload size are essentially independent of context length. Correctness: a
differential stress test (20 000 random operations incl. sharing, format changes, residency changes,
irregular upload cadence) checks that a simulated device mirror always reads exactly what the full
rebuild produces.

## Trade-offs

* One extra dependent load per block on the device (id → descriptor). The descriptor table is
  16 B × max_blocks, small enough to stay cache-resident on current GPUs, but this is unmeasured.
* Host memory: +1 B dirty flag and +4 B dirty-list capacity per block id and per arena entry.
* The arena never shrinks.
* The remaining hetero cost over an equally persistent homogeneous baseline is about 4–4.5× (≈3.3–4.1 µs
  per step for 256 sequences). From the operation counts (not a profile), most of it is the
  per-sequence descriptor maintenance: the `valid_tokens` update and its 16-B patch every step.

## Integration note (engineering inference)

The per-sequence id lists keep the shape of today's int32 block tables. An engine could keep its own
block table and add only the HL-KV descriptor table indexed by block id. That makes the Project 5
integration surface smaller than a materialized table would. To be validated against vLLM/SGLang
source in Project 5.

## Revisit conditions

* GPU profiling shows the indirection in the attention inner loop matters → reconsider materialized
  per-sequence descriptors (with the reverse index) for hot sequences only.
* Arena growth without compaction becomes significant in long-running traces → add compaction or
  epoch-based rebuild.
