# hl_allocator — HL-KV ABI v0

Status: **implemented, unit-tested, CPU-benchmarked**. Not yet consumed by any kernel or engine.
Source: `project1_allocator/include/hl_allocator/`.

## 1. Purpose

Project 1 defines how a heterogeneous KV block is described on the host and what a kernel receives
per block. It answers one question: can blocks of different storage formats coexist in one logical
sequence with metadata that is small, deterministic and cheap to maintain?

## 2. Model

| Concept | Definition |
|---|---|
| Block id | Stable logical identity, index into the host descriptor array. Guarded by a 32-bit generation counter (`BlockHandle = {id, generation}`), incremented on every free. |
| Format | Opaque 4-bit code (`FormatId`, 0..14). The allocator does not interpret it; canonical assignment is `hl::quant::QuantFormat`. |
| Pool | One fixed-size slot pool per configured format (`slot_bytes`, `num_slots`). Changing a block FP16 → INT4 moves it to a smaller slot, so capacity is actually freed. |
| Slot | Physical index inside the pool of the block's current format. |
| Sequence | Ordered list of block ids; holds one reference per block. Blocks can be shared (prefix sharing) once sealed. |
| Exec table | CSR gather for a batch of sequences: `indptr[num_seqs+1]` + `BlockExecDescriptor[]`. This is what is uploaded to the GPU. |

Device address of a block payload (computed by the kernel):

```
addr = pool_base[format] + slot * slot_bytes[format]
```

`pool_base` / `slot_bytes` are ≤15 per-table constants passed as kernel arguments.

## 3. Byte layout (little-endian; normative)

Packed words use explicit shift/mask, not C++ bit-fields (bit-field order is
implementation-defined and differs between MSVC and GCC/Clang/NVCC).

### 3.1 `BlockExecDescriptor` — 16 bytes, `alignas(16)` (device-facing)

| Bytes | Field | Meaning |
|---|---|---|
| 0–3 | `slot` | slot index in the pool of `format` |
| 4–7 | `info` | bits 0–3 `format`, 4–5 `scale_mode`, 6–9 `layout`, 10–19 `valid_tokens`, 20–31 reserved (0) |
| 8–11 | `scale_ref` | interpreted per `scale_mode` (below) |
| 12–15 | `block_id` | stable identity; lets kernels emit per-block statistics and makes records self-checking |

One block = one 128-bit load on the device.

`scale_mode`:

| Value | Name | `scale_ref` |
|---|---|---|
| 0 | `kNone` | unused: unscaled format, per-tensor kernel parameter, or scales stored inside the slot payload |
| 1 | `kInlineF32` | binary32 bits of one per-block scale |
| 2 | `kInlineF16Pair` | binary16 scale (bits 0–15) + binary16 bias (bits 16–31) |
| 3 | `kExternal` | index into an externally managed scale pool |

### 3.2 `BlockDescriptor` — 32 bytes, `alignas(32)` (host-side)

| Bytes | Field | Meaning |
|---|---|---|
| 0–15 | `exec` | the `BlockExecDescriptor`, copied verbatim into exec tables |
| 16–19 | `generation` | ABA counter, +1 on free (wraps at 2^32) |
| 20–23 | `control` | bits 0–1 `state`, 2–3 `residency`, 4–7 `pending_format`, 8–23 `refcount`, 24–31 reserved (0) |
| 24–27 | `pending_slot` | target slot while `kTranscoding`, else `0xFFFFFFFF` |
| 28–31 | `reserved` | 0; ABI v1 extension space |

`alignas(32)` ⇒ a record never straddles a 64-byte line; a validated lookup (generation check +
exec copy) touches exactly one cache line. Golden byte-level tests:
`tests/project1/test_block_descriptor.cpp`.

### 3.3 Field limits

| Field | Width | Limit |
|---|---|---|
| format | 4 | 15 codes, 15 = invalid |
| valid_tokens | 10 | `block_tokens ≤ kMaxBlockTokens = 512` |
| layout | 4 | 0..15, opaque to the allocator |
| refcount | 16 | 65535 (`kRefcountOverflow` beyond) |
| block ids / slots | 32 | 2^32 − 2 (`0xFFFFFFFF` is the sentinel) |

## 4. Block state machine

```
            allocate                seal                 begin_format_change
   kFree ─────────────► kOpen ─────────────► kSealed ─────────────────────► kTranscoding
     ▲                    │  set_valid_tokens   ▲  │  ◄──── commit / abort ─────┘
     └── release (rc→0) ──┴─────────────────────┴──┘
```

* `kOpen`: tokens may be appended (`set_valid_tokens`, also allows rollback).
* `kSealed`: immutable payload; may be shared and re-encoded.
* `kTranscoding`: both the old and the target slot are live. The block remains readable through its
  old format/slot, so attention can run while a conversion kernel writes the new slot. The last
  reference cannot be released in this state (`kInvalidState`); commit or abort first.
* Residency (`kDevice` / `kHost` / `kDropped`) is orthogonal; only `kDevice` blocks can be gathered
  into exec tables (`kNotDeviceResident`).

## 5. Contracts

**Concurrency.** Not thread-safe. One owner thread (the scheduler) mutates; `const` functions may
run concurrently with each other only while no mutation is in progress.

**GPU lifetime.** `release()` and `commit_format_change()` make slots reusable immediately. The
caller must not free a slot that in-flight GPU work still reads (defer until the step's stream
event completes). The allocator has no knowledge of streams.

**Determinism.** Free lists are LIFO stacks initialised so the first allocations return 0, 1, 2, …
The same sequence of calls always yields the same ids and slots.

**Errors.** Recoverable conditions return `Status` (never throw on the hot path). Invalid
configuration throws `std::invalid_argument`; `Result::value()` on an error throws `BadResultAccess`.
Multi-step operations (`release_sequence`, `append_block`) are all-or-nothing.

**Validation.** `validate()` checks every invariant (free lists, unique slot ownership per pool,
reserved bits, refcount ≥ sequence references, pending-slot consistency, counters). Used by the
randomized stress test.

## 6. Fragmentation (precise statement)

* Within a pool, slots are fixed-size and interchangeable: any free slot satisfies any request for
  that format, so there is **no external fragmentation inside a pool**.
* Internal fragmentation is the unused tail of partially filled blocks:
  `token_capacity_in_use − valid_tokens` (reported by `stats()`).
* **Cross-pool stranding is real and unmodelled in v0**: capacity is partitioned between formats at
  construction. A full FP16 pool cannot borrow from a half-empty INT4 pool. This is the main
  fragmentation risk of the per-format-pool design (see Limitations).

## 7. Metadata cost

Computed from the layout (arithmetic, not measurement):

| Item | Heterogeneous (v0) | Homogeneous int32 baseline |
|---|---|---|
| Host record / block | 32 B | 2 B refcount (+ free-list entry) |
| Device table entry / block | 16 B | 4 B |
| Device table / token (16-token blocks) | 1 B | 0.25 B |
| Per-sequence host list / block | 4 B | 4 B |

For scale: one layer of a model with 8 KV heads × 128 dims stores 4096 B/token of FP16 K+V, so a
1 B/token device table is ≈0.02 % of the payload. Capacity cost is negligible; **CPU time is not**
(see `docs/benchmarks/2026-10-11-p1-allocator-cpu.md`).

## 8. Limitations (v0)

1. **Exec tables are rebuilt in full** (`O(total blocks)` per call). Measured 2–25× slower than a
   bulk copy of int32 block lists. An incremental/persistent table is the next P1 objective.
2. **Static pool partitioning** — cross-format capacity stranding (Section 6).
3. **Non-device residency is a tag only.** `kHost`/`kDropped` do not release the device slot and the
   table manages no host memory, so offload saves no HBM yet.
4. **Layer association is not represented.** One table may serve all layers (uniform format across
   layers per block) or one table per layer (per-layer formats, N× the rebuild cost). Undecided.
5. **Scales produced on the GPU** (dynamic quantization) cannot be placed in `scale_ref` without a
   device→host copy; such formats should use `kNone` with in-payload scales.
6. No serialization API for a full table snapshot yet (the descriptor itself is a fixed, documented
   byte layout).
7. Single-threaded; no lock-free or sharded variant.

## 9. Versioning

`kAbiVersion = 0`. Any change to Section 3 requires bumping the version and updating the golden
tests. Reserved bits/bytes must be written as zero and are checked by `validate()`.
