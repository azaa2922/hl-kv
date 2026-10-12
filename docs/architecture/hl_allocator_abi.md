# hl_allocator — HL-KV ABI

Current version: **v1** (`kAbiVersion = 1`). Status: **implemented, unit-tested, CPU-benchmarked**.
Not yet consumed by any kernel or engine. Source: `project1_allocator/include/hl_allocator/`.

| Version | Change |
|---|---|
| v0 (2026-10-11) | Record layouts (`BlockExecDescriptor` 16 B, `BlockDescriptor` 32 B); per-step materialized CSR exec table |
| v1 (2026-10-12) | Record layouts unchanged. Adds the persistent device-table protocol (§5) and the non-executable record rule (§3.1). The materialized CSR remains as the reference path. |

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
| Sequence | Ordered list of block ids stored as one segment of the persistent id arena (§5); holds one reference per block. Blocks can be shared (prefix sharing) once sealed. |

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
| 12–15 | `block_id` | stable identity; makes device-table patches self-addressed (§5) and records self-checking |

One block = one 128-bit load on the device.

**Non-executable records (v1).** A device record with `format == 15` (and `slot == 0xFFFFFFFF`) must
never be dereferenced. The device table holds such a record for every block that is free or not
device-resident (`Residency::kHost` / `kDropped`). Kernels may check the format before computing an
address (one compare) and treat a hit as a scheduler bug.

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
| 0–15 | `exec` | the `BlockExecDescriptor` |
| 16–19 | `generation` | ABA counter, +1 on free (wraps at 2^32) |
| 20–23 | `control` | bits 0–1 `state`, 2–3 `residency`, 4–7 `pending_format`, 8–23 `refcount`, 24–31 reserved (0) |
| 24–27 | `pending_slot` | target slot while `kTranscoding`, else `0xFFFFFFFF` |
| 28–31 | `reserved` | 0; extension space |

`alignas(32)` ⇒ a record never straddles a 64-byte line; a validated lookup (generation check +
exec copy) touches exactly one cache line. Golden byte-level tests:
`tests/project1/test_block_descriptor.cpp`.

### 3.3 Field limits

| Field | Width | Limit |
|---|---|---|
| format | 4 | 15 codes, 15 = invalid / non-executable |
| valid_tokens | 10 | `block_tokens ≤ kMaxBlockTokens = 512` |
| layout | 4 | 0..15, opaque to the allocator |
| refcount | 16 | 65535 (`kRefcountOverflow` beyond) |
| block ids / slots | 32 | 2^32 − 2 (`0xFFFFFFFF` is the sentinel) |
| arena offsets | 32 | < 2^32 − 1 entries; one sequence ≤ 2^31 blocks (`kCapacityExceeded` beyond) |

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
* Residency (`kDevice` / `kHost` / `kDropped`) is orthogonal.

## 5. Device-facing state and the delta protocol (v1)

The device keeps two persistent arrays, maintained by patches instead of being rebuilt:

| Array | Element | Indexed by | Changes when |
|---|---|---|---|
| descriptor table | `BlockExecDescriptor` (16 B) | block id (`max_blocks` entries) | allocate, free, `set_valid_tokens`, `set_scale`, `set_residency`, `commit_format_change` |
| id arena | `uint32` block id | arena offset | a block is appended to a sequence, or a sequence segment is relocated |

Kernel view of a batch, produced by `batch_segments()` in O(number of sequences):

```
segment_i = {offset, length}                       // one per sequence in the batch
block j of sequence i:  d = descriptor_table[ arena[ segment_i.offset + j ] ]
```

**Delta.** `collect_delta(ExecDelta&)` returns everything changed since the previous call:

```
descriptor_patches : BlockExecDescriptor[]   // self-addressed: table[p.block_id] = p
id_patches         : {offset, block_id}[]    // arena[p.offset] = p.block_id
arena_size         : device arena must hold at least this many entries (it only grows)
```

Each record/entry appears at most once per delta, carrying its final value, so the patches can be
applied in any order or in parallel (e.g. one scatter kernel per list). The first delta after
construction carries every descriptor, so device memory needs no other initialization. A change to a
shared block is one descriptor patch regardless of how many sequences reference it — no reverse
index is needed. `collect_delta()` gives the strong guarantee (if allocating the output throws,
nothing is lost).

**Id arena.** Each sequence owns one segment; capacities are powers of two (8 … 2^31 entries). A
full segment is relocated to one of twice the size (amortized O(1) copies per append; the moved
entries become id patches); released segments are reused through per-size-class free lists. The
arena never shrinks and is never compacted in v1.

**Reference path.** `build_exec_table()` still materializes a CSR of exec descriptors in
O(total blocks); it is the oracle of the differential tests, and it refuses non-resident blocks with
`kNotDeviceResident`.

## 6. Contracts

**Concurrency.** Not thread-safe. One owner thread (the scheduler) mutates; `const` functions may
run concurrently with each other only while no mutation is in progress.

**GPU lifetime.** `release()` and `commit_format_change()` make slots reusable immediately. The
caller must not free a slot that in-flight GPU work still reads (defer until the step's stream
event completes). Deltas are applied in stream order before the next step's kernels.

**Residency.** `batch_segments()` does not scan blocks (that would be O(total blocks) again); the
device table marks non-resident blocks non-executable instead. The scheduler is responsible for
scheduling only device-resident sequences; `build_exec_table()` provides the host-side check.

**Determinism.** Free lists are LIFO stacks initialised so the first allocations return 0, 1, 2, …
The same sequence of calls always yields the same ids, slots and arena offsets.

**Errors.** Recoverable conditions return `Status` (never throw on the hot path). Invalid
configuration throws `std::invalid_argument`; `Result::value()` on an error throws `BadResultAccess`.
`Result<T>::value() &&` returns by value so that `for (x : f().value())` cannot dangle.
Multi-step operations (`release_sequence`, `append_block`) leave sequence contents unchanged on
failure.

**Validation.** `validate()` checks every invariant: free lists, unique slot ownership per pool,
reserved bits, refcount ≥ sequence references, pending-slot consistency, counters, an exact
partition of the arena into live and free segments, and the dirty-tracking bookkeeping.

## 7. Fragmentation (precise statement)

* Within a slot pool, slots are fixed-size and interchangeable: **no external fragmentation inside a
  pool**. Internal fragmentation = `token_capacity_in_use − valid_tokens` (`stats()`).
* **Cross-pool stranding is real and unmodelled**: capacity is partitioned between formats at
  construction.
* Id arena: segments are at most 2× their length (power-of-two classes), and free segments are only
  reused by the same size class. `stats()` reports `arena_entries`, `arena_live_entries`,
  `arena_reserved_entries` and `arena_free_entries`.

## 8. Metadata cost

Computed from the layout (arithmetic, not measurement):

| Item | Heterogeneous (v1) | Homogeneous baseline |
|---|---|---|
| Host record / block | 32 B (+1 B dirty flag, +4 B dirty-list capacity) | 2 B refcount |
| Device descriptor table / block | 16 B | — |
| Device id arena / block reference | 4 B (+ ≤ 2× capacity slack) | 4 B (same arena) |
| Per-step upload, steady decode | ≈ 16 B × sequences + 8 B × appended blocks + 8 B × sequences | 8 B × appended blocks + 12 B × sequences |

Measured per-step costs: `docs/benchmarks/2026-10-12-p1-persistent-exec-table-cpu.md`.

## 9. Limitations (v1)

1. **Static pool partitioning** — cross-format capacity stranding (§7).
2. **Non-device residency is a tag only.** `kHost`/`kDropped` do not release the device slot and the
   table manages no host memory, so offload saves no HBM yet.
3. **Layer association is not represented.** One table may serve all layers (uniform format across
   layers per block) or one table per layer (per-layer formats). Undecided.
4. **Scales produced on the GPU** (dynamic quantization) cannot be placed in `scale_ref` without a
   device→host copy; such formats should use `kNone` with in-payload scales.
5. The id arena never shrinks or compacts; a burst of long sequences leaves free segments behind.
6. The descriptor-table indirection costs the kernel one extra dependent load per block compared with
   a materialized table. Unmeasured on a GPU.
7. No serialization API for a full table snapshot; single-threaded.
