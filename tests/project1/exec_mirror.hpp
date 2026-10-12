#pragma once

// Test helper: a host-side simulation of the GPU copy of HeteroBlockTable's
// persistent device-facing state. It applies ExecDelta exactly as documented
// on the struct, and reads a batch the way a kernel would:
//   descriptor_table[ arena[ segment.offset + j ] ].
// Differential tests compare what it reads against the full-rebuild reference.

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <vector>

#include "hl_allocator/hetero_block_table.hpp"

namespace hl::alloc::testing {

class ExecMirror {
 public:
  /// Applies one delta. Fresh device memory is filled with garbage so that a
  /// missing initial patch would be detected.
  void apply(const ExecDelta& delta) {
    if (table_.size() < delta.descriptor_table_size) {
      BlockExecDescriptor garbage{};
      garbage.slot = 0xABAB'ABABu;
      garbage.info = 0xABAB'ABABu;
      garbage.scale_ref = 0xABAB'ABABu;
      garbage.block_id = 0xABAB'ABABu;
      table_.resize(delta.descriptor_table_size, garbage);
    }
    if (arena_.size() < delta.arena_size) arena_.resize(delta.arena_size, 0xDEAD'BEEFu);

    // Targets must be unique within a delta (patches may be applied in parallel).
    std::vector<std::uint8_t> seen_desc(table_.size(), 0);
    for (const BlockExecDescriptor& p : delta.descriptor_patches) {
      ASSERT_LT(p.block_id, table_.size());
      EXPECT_EQ(seen_desc[p.block_id], 0) << "duplicate descriptor patch " << p.block_id;
      seen_desc[p.block_id] = 1;
      table_[p.block_id] = p;
    }
    std::vector<std::uint8_t> seen_arena(arena_.size(), 0);
    for (const IdPatch& p : delta.id_patches) {
      ASSERT_LT(p.offset, arena_.size());
      EXPECT_EQ(seen_arena[p.offset], 0) << "duplicate id patch at " << p.offset;
      seen_arena[p.offset] = 1;
      arena_[p.offset] = p.block_id;
    }
  }

  /// What a kernel reads for one sequence segment.
  [[nodiscard]] std::vector<BlockExecDescriptor> read(SequenceSegment segment) const {
    std::vector<BlockExecDescriptor> out;
    for (std::uint32_t j = 0; j < segment.length; ++j) {
      out.push_back(table_.at(arena_.at(std::size_t{segment.offset} + j)));
    }
    return out;
  }

  /// Materializes a batch in the same CSR shape as HeteroBlockTable::build_exec_table.
  [[nodiscard]] ExecTable read_batch(std::span<const SequenceSegment> segments) const {
    ExecTable t;
    t.indptr.push_back(0);
    for (const SequenceSegment s : segments) {
      for (const BlockExecDescriptor& d : read(s)) t.blocks.push_back(d);
      t.indptr.push_back(static_cast<std::uint32_t>(t.blocks.size()));
    }
    return t;
  }

  [[nodiscard]] const std::vector<BlockExecDescriptor>& table() const noexcept { return table_; }
  [[nodiscard]] const std::vector<std::uint32_t>& arena() const noexcept { return arena_; }

 private:
  std::vector<BlockExecDescriptor> table_;
  std::vector<std::uint32_t> arena_;
};

/// Every descriptor in the mirror equals the table's device view.
inline ::testing::AssertionResult mirror_matches(const ExecMirror& mirror,
                                                 const HeteroBlockTable& table) {
  if (mirror.table().size() != table.max_blocks()) {
    return ::testing::AssertionFailure() << "descriptor table size " << mirror.table().size();
  }
  for (std::uint32_t id = 0; id < table.max_blocks(); ++id) {
    if (!(mirror.table()[id] == table.device_descriptor(id))) {
      return ::testing::AssertionFailure() << "descriptor mismatch for block " << id;
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace hl::alloc::testing
