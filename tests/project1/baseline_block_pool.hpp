#pragma once

// Benchmark baseline: a minimal-metadata HOMOGENEOUS block pool.
//
// It models the representation consumed by today's paged-attention kernels:
// every block has the same dtype and size, so a block id IS the physical
// location and the per-sequence block table is a flat int32 array (the
// `block_tables` / `kv_indices` tensors of vLLM / FlashInfer). Host metadata
// is a uint16 refcount per block plus a LIFO free stack.
//
// This is deliberately a lower bound on homogeneous bookkeeping cost, not a
// reproduction of vLLM's Python BlockPool (which carries much more per-block
// state). Comparing HeteroBlockTable against it isolates the cost of the extra
// heterogeneous metadata. It performs no stale-handle detection.

#include <cstdint>
#include <span>
#include <vector>

namespace hl::bench {

class HomogeneousBlockPool {
 public:
  explicit HomogeneousBlockPool(std::uint32_t num_blocks);

  /// Returns -1 when exhausted.
  [[nodiscard]] std::int32_t allocate() noexcept;
  void release(std::int32_t id) noexcept;
  /// Validated lookup: returns the physical block index, or -1 if not allocated.
  [[nodiscard]] std::int32_t lookup(std::int32_t id) const noexcept;
  /// Gathers physical indices; returns false if any id is not allocated.
  [[nodiscard]] bool lookup_batch(std::span<const std::int32_t> ids,
                                  std::span<std::int32_t> out) const noexcept;

  [[nodiscard]] std::uint32_t create_sequence();
  [[nodiscard]] bool append_block(std::uint32_t seq);
  void release_sequence(std::uint32_t seq) noexcept;
  [[nodiscard]] std::size_t sequence_length(std::uint32_t seq) const noexcept;
  /// CSR int32 block table (indptr + indices) for a batch of sequences.
  void build_block_table(std::span<const std::uint32_t> seqs, std::vector<std::uint32_t>& indptr,
                         std::vector<std::int32_t>& indices) const;

 private:
  std::vector<std::uint16_t> refcount_;
  std::vector<std::int32_t> free_;
  std::vector<std::vector<std::int32_t>> seqs_;
  std::vector<std::uint32_t> free_seqs_;
};

}  // namespace hl::bench
