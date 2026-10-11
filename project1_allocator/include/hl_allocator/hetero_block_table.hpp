#pragma once

// HeteroBlockTable — host-side manager for heterogeneous KV blocks.
//
// Model
//   * A block has a stable logical identity (block id, guarded by a generation
//     counter) and a physical slot inside the pool of its current format.
//   * Each format has its own fixed-size slot pool (slot_bytes differs per
//     format), so re-encoding FP16 -> INT4 actually frees capacity. Pools are
//     sized at construction; v0 does not rebalance capacity between pools.
//   * Sequences are ordered lists of block ids. A sequence holds one reference
//     on each of its blocks; blocks can be shared (prefix sharing) by retaining.
//   * build_exec_table() gathers BlockExecDescriptors for a batch of sequences
//     into a CSR table ready for upload (indptr + flat descriptors).
//
// Concurrency
//   Not thread-safe. One owner thread (the scheduler) mutates the table;
//   const member functions may run concurrently with each other only while no
//   mutation is in progress.
//
// GPU lifetime
//   release()/commit_format_change() make slots reusable immediately. Callers
//   must not release a slot that in-flight GPU work still reads (e.g. defer the
//   call until the step's stream event has completed).

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "hl_allocator/block_descriptor.hpp"
#include "hl_allocator/status.hpp"

namespace hl::alloc {

struct BlockHandle {
  std::uint32_t id;
  std::uint32_t generation;
  friend constexpr bool operator==(BlockHandle, BlockHandle) noexcept = default;
};
inline constexpr BlockHandle kInvalidBlock{0xFFFF'FFFFu, 0};

struct SequenceHandle {
  std::uint32_t id;
  std::uint32_t generation;
  friend constexpr bool operator==(SequenceHandle, SequenceHandle) noexcept = default;
};
inline constexpr SequenceHandle kInvalidSequence{0xFFFF'FFFFu, 0};

struct PoolConfig {
  FormatId format{};
  std::uint32_t slot_bytes = 0;  ///< payload bytes per slot (> 0)
  std::uint32_t num_slots = 0;   ///< capacity of this pool (> 0)
};

struct TableConfig {
  std::uint32_t max_blocks = 0;     ///< number of block ids, 1 .. 2^32-2
  std::uint32_t block_tokens = 16;  ///< tokens per block, 1 .. kMaxBlockTokens
  std::vector<PoolConfig> pools;    ///< one entry per format, codes unique and < kMaxFormats
};

struct AllocRequest {
  FormatId format{};
  std::uint32_t valid_tokens = 0;  ///< <= block_tokens
  std::uint32_t layout = 0;        ///< opaque layout tag, <= kMaxLayoutId
  ScaleMode scale_mode = ScaleMode::kNone;
  std::uint32_t scale_ref = 0;
};

/// Result of begin_format_change(): both slots are live until commit/abort.
struct FormatChange {
  FormatId from_format;
  std::uint32_t from_slot;
  FormatId to_format;
  std::uint32_t to_slot;
};

/// CSR gather of execution descriptors for a batch of sequences:
/// blocks of sequence i are blocks[indptr[i] .. indptr[i+1]).
struct ExecTable {
  std::vector<std::uint32_t> indptr;
  std::vector<BlockExecDescriptor> blocks;
};

struct TableStats {
  std::uint32_t blocks_in_use = 0;
  std::uint32_t blocks_capacity = 0;
  std::uint32_t active_sequences = 0;
  std::array<std::uint32_t, kMaxFormats> blocks_per_format{};
  std::array<std::uint32_t, 4> blocks_per_residency{};
  std::array<std::uint32_t, 4> blocks_per_state{};
  std::uint64_t valid_tokens = 0;          ///< sum of valid_tokens over in-use blocks
  std::uint64_t token_capacity_in_use = 0; ///< blocks_in_use * block_tokens
  std::uint64_t payload_bytes_in_use = 0;  ///< sum of slot_bytes (incl. pending slots)
  std::uint64_t format_changes_committed = 0;
};

struct ValidationReport {
  Status status = Status::kOk;
  std::string message;
  [[nodiscard]] bool ok() const noexcept { return status == Status::kOk; }
};

class HeteroBlockTable {
 public:
  /// Throws std::invalid_argument if validate_config(config) != kOk.
  explicit HeteroBlockTable(const TableConfig& config);

  [[nodiscard]] static Status validate_config(const TableConfig& config) noexcept;

  // ---- block lifecycle ------------------------------------------------------

  /// Allocates a block id and a slot in the pool of request.format.
  /// The new block is kOpen, kDevice, refcount 1.
  [[nodiscard]] Result<BlockHandle> allocate(const AllocRequest& request);
  /// Adds a reference (prefix sharing).
  [[nodiscard]] Status retain(BlockHandle handle);
  /// Drops a reference; at zero the slot(s) and id are returned to the free lists.
  [[nodiscard]] Status release(BlockHandle handle);

  // ---- state transitions ----------------------------------------------------

  [[nodiscard]] Status set_valid_tokens(BlockHandle handle, std::uint32_t valid_tokens);
  [[nodiscard]] Status seal(BlockHandle handle);
  [[nodiscard]] Status set_residency(BlockHandle handle, Residency residency);
  [[nodiscard]] Status set_scale(BlockHandle handle, ScaleMode mode, std::uint32_t scale_ref);

  /// Phase 1 of a precision change: reserves a slot in the target pool. The
  /// block must be kSealed and device-resident. Until commit/abort the block
  /// stays readable through its old format and slot.
  [[nodiscard]] Result<FormatChange> begin_format_change(BlockHandle handle, FormatId target);
  /// Phase 2: switch the block to the target slot/format and free the old slot.
  [[nodiscard]] Status commit_format_change(BlockHandle handle, ScaleMode mode,
                                            std::uint32_t scale_ref);
  /// Cancel: free the reserved target slot; the block keeps its old format.
  [[nodiscard]] Status abort_format_change(BlockHandle handle);

  // ---- lookup ---------------------------------------------------------------

  /// Validated copy of the full host record.
  [[nodiscard]] Result<BlockDescriptor> lookup(BlockHandle handle) const;
  /// Validated gather of execution descriptors. `out.size()` must be >=
  /// `handles.size()`. Every block must be device-resident. On error the
  /// contents of `out` are unspecified.
  [[nodiscard]] Status lookup_batch(std::span<const BlockHandle> handles,
                                    std::span<BlockExecDescriptor> out) const;
  /// Current handle for an allocated block id (kInvalidBlock if free/out of range).
  [[nodiscard]] BlockHandle handle_of(std::uint32_t block_id) const noexcept;

  // ---- sequences ------------------------------------------------------------

  [[nodiscard]] Result<SequenceHandle> create_sequence();
  /// Allocates a new block and appends it to the sequence (the sequence holds
  /// the only reference).
  [[nodiscard]] Result<BlockHandle> append_block(SequenceHandle seq, const AllocRequest& request);
  /// Appends an existing block and takes a reference on it (prefix sharing).
  [[nodiscard]] Status append_shared_block(SequenceHandle seq, BlockHandle block);
  /// Releases every block reference held by the sequence and retires the handle.
  [[nodiscard]] Status release_sequence(SequenceHandle seq);
  /// Block ids of a sequence, in logical order. The span is invalidated by any
  /// mutating call.
  [[nodiscard]] Result<std::span<const std::uint32_t>> sequence_block_ids(
      SequenceHandle seq) const;

  /// Gathers execution descriptors for `seqs` into `out` (cleared first; its
  /// capacity is reused). Every gathered block must be device-resident; on
  /// error `out` is left empty.
  [[nodiscard]] Status build_exec_table(std::span<const SequenceHandle> seqs,
                                        ExecTable& out) const;

  // ---- introspection --------------------------------------------------------

  [[nodiscard]] std::uint32_t block_tokens() const noexcept { return block_tokens_; }
  [[nodiscard]] std::uint32_t max_blocks() const noexcept {
    return static_cast<std::uint32_t>(blocks_.size());
  }
  [[nodiscard]] std::uint32_t blocks_in_use() const noexcept { return blocks_in_use_; }
  [[nodiscard]] bool has_pool(FormatId format) const noexcept;
  [[nodiscard]] std::uint32_t free_slots(FormatId format) const noexcept;
  [[nodiscard]] std::uint32_t pool_slots(FormatId format) const noexcept;
  [[nodiscard]] std::uint32_t slot_bytes(FormatId format) const noexcept;

  /// O(max_blocks) scan; intended for observability, not the hot path.
  [[nodiscard]] TableStats stats() const;
  /// O(max_blocks + total slots + sequence lengths) full consistency check.
  [[nodiscard]] ValidationReport validate() const;

 private:
  struct Pool {
    std::uint32_t slot_bytes = 0;
    std::uint32_t num_slots = 0;
    std::vector<std::uint32_t> free_slots;  // LIFO stack
    bool configured = false;
  };

  struct Sequence {
    std::vector<std::uint32_t> blocks;
    std::uint32_t generation = 0;
    bool active = false;
  };

  [[nodiscard]] Status check_block(BlockHandle handle) const noexcept;
  [[nodiscard]] Status check_sequence(SequenceHandle seq) const noexcept;
  [[nodiscard]] Status check_request(const AllocRequest& request) const noexcept;
  [[nodiscard]] Pool* pool_for(FormatId format) noexcept;
  [[nodiscard]] const Pool* pool_for(FormatId format) const noexcept;
  void free_block(std::uint32_t id) noexcept;

  std::uint32_t block_tokens_;
  std::uint32_t blocks_in_use_ = 0;
  std::uint64_t format_changes_committed_ = 0;
  std::vector<BlockDescriptor> blocks_;
  std::vector<std::uint32_t> free_ids_;  // LIFO stack
  std::array<Pool, kMaxFormats> pools_{};
  std::vector<Sequence> sequences_;
  std::vector<std::uint32_t> free_sequences_;  // LIFO stack
};

}  // namespace hl::alloc
