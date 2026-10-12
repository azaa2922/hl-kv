// Behavioural tests for HeteroBlockTable: configuration, deterministic
// allocation, handle safety (stale/invalid), refcounting, state machine,
// two-phase format change, sequences, CSR exec tables, and a randomized
// stress test that checks validate() invariants throughout.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "exec_mirror.hpp"
#include "hl_allocator/hetero_block_table.hpp"

namespace hl::alloc {
namespace {

constexpr FormatId kFp16{0};
constexpr FormatId kFp8{1};
constexpr FormatId kInt4{3};
constexpr FormatId kUnconfigured{9};

TableConfig small_config(std::uint32_t blocks = 8, std::uint32_t fp16_slots = 4,
                         std::uint32_t fp8_slots = 4, std::uint32_t int4_slots = 4) {
  TableConfig c;
  c.max_blocks = blocks;
  c.block_tokens = 16;
  c.pools = {{kFp16, 4096, fp16_slots}, {kFp8, 2048, fp8_slots}, {kInt4, 1088, int4_slots}};
  return c;
}

AllocRequest req(FormatId f, std::uint32_t tokens = 0) {
  AllocRequest r;
  r.format = f;
  r.valid_tokens = tokens;
  return r;
}

void expect_valid(const HeteroBlockTable& t) {
  const ValidationReport rep = t.validate();
  EXPECT_TRUE(rep.ok()) << rep.message;
}

// ---- configuration --------------------------------------------------------------------

TEST(HeteroBlockTableConfig, RejectsInvalidConfigs) {
  EXPECT_EQ(HeteroBlockTable::validate_config(small_config()), Status::kOk);

  TableConfig c = small_config();
  c.max_blocks = 0;
  EXPECT_EQ(HeteroBlockTable::validate_config(c), Status::kInvalidArgument);

  c = small_config();
  c.block_tokens = 0;
  EXPECT_EQ(HeteroBlockTable::validate_config(c), Status::kInvalidArgument);
  c.block_tokens = kMaxBlockTokens + 1;
  EXPECT_EQ(HeteroBlockTable::validate_config(c), Status::kInvalidArgument);
  c.block_tokens = kMaxBlockTokens;
  EXPECT_EQ(HeteroBlockTable::validate_config(c), Status::kOk);

  c = small_config();
  c.pools.clear();
  EXPECT_EQ(HeteroBlockTable::validate_config(c), Status::kInvalidArgument);

  c = small_config();
  c.pools.push_back({kFp16, 4096, 1});  // duplicate format
  EXPECT_EQ(HeteroBlockTable::validate_config(c), Status::kInvalidArgument);

  c = small_config();
  c.pools.push_back({kInvalidFormat, 16, 1});
  EXPECT_EQ(HeteroBlockTable::validate_config(c), Status::kUnknownFormat);

  c = small_config();
  c.pools[0].slot_bytes = 0;
  EXPECT_EQ(HeteroBlockTable::validate_config(c), Status::kInvalidArgument);

  c = small_config();
  c.pools[0].num_slots = 0;
  EXPECT_EQ(HeteroBlockTable::validate_config(c), Status::kInvalidArgument);

  c = small_config();
  c.max_blocks = 0;
  EXPECT_THROW(HeteroBlockTable{c}, std::invalid_argument);
}

TEST(HeteroBlockTableConfig, FreshTableIsConsistent) {
  const HeteroBlockTable t(small_config());
  EXPECT_EQ(t.max_blocks(), 8u);
  EXPECT_EQ(t.blocks_in_use(), 0u);
  EXPECT_EQ(t.free_slots(kFp16), 4u);
  EXPECT_EQ(t.slot_bytes(kFp8), 2048u);
  EXPECT_FALSE(t.has_pool(kUnconfigured));
  EXPECT_EQ(t.free_slots(kUnconfigured), 0u);
  EXPECT_EQ(t.free_slots(kInvalidFormat), 0u);
  expect_valid(t);
}

// ---- allocation ---------------------------------------------------------------------

TEST(HeteroBlockTable, AllocationOrderIsDeterministic) {
  HeteroBlockTable t(small_config());
  for (std::uint32_t i = 0; i < 3; ++i) {
    const Result<BlockHandle> h = t.allocate(req(kFp16));
    ASSERT_TRUE(h.ok());
    EXPECT_EQ(h.value().id, i);
    EXPECT_EQ(t.lookup(h.value()).value().exec.slot, i);
  }
  const Result<BlockHandle> h = t.allocate(req(kInt4));
  ASSERT_TRUE(h.ok());
  EXPECT_EQ(h.value().id, 3u);
  EXPECT_EQ(t.lookup(h.value()).value().exec.slot, 0u);  // first slot of the INT4 pool
}

TEST(HeteroBlockTable, AllocatedDescriptorReflectsRequest) {
  HeteroBlockTable t(small_config());
  AllocRequest r = req(kFp8, 5);
  r.layout = 2;
  r.scale_mode = ScaleMode::kInlineF32;
  r.scale_ref = 0x3F80'0000u;  // 1.0f
  const BlockHandle h = t.allocate(r).value();
  const BlockDescriptor d = t.lookup(h).value();
  EXPECT_EQ(d.exec.format(), kFp8);
  EXPECT_EQ(d.exec.valid_tokens(), 5u);
  EXPECT_EQ(d.exec.layout(), 2u);
  EXPECT_EQ(d.exec.scale_mode(), ScaleMode::kInlineF32);
  EXPECT_EQ(d.exec.inline_scale_f32(), 1.0f);
  EXPECT_EQ(d.exec.block_id, h.id);
  EXPECT_EQ(d.state(), BlockState::kOpen);
  EXPECT_EQ(d.residency(), Residency::kDevice);
  EXPECT_EQ(d.refcount(), 1u);
  EXPECT_EQ(d.pending_slot, kNoSlot);
}

TEST(HeteroBlockTable, RejectsInvalidRequests) {
  HeteroBlockTable t(small_config());
  EXPECT_EQ(t.allocate(req(kUnconfigured)).status(), Status::kUnknownFormat);
  EXPECT_EQ(t.allocate(req(kInvalidFormat)).status(), Status::kUnknownFormat);
  EXPECT_EQ(t.allocate(req(kFp16, 17)).status(), Status::kInvalidArgument);
  AllocRequest r = req(kFp16);
  r.layout = kMaxLayoutId + 1;
  EXPECT_EQ(t.allocate(r).status(), Status::kInvalidArgument);
  r = req(kFp16);
  r.scale_mode = static_cast<ScaleMode>(4);
  EXPECT_EQ(t.allocate(r).status(), Status::kInvalidArgument);
  EXPECT_EQ(t.blocks_in_use(), 0u);
  expect_valid(t);
}

TEST(HeteroBlockTable, PoolExhaustionDoesNotLeakIds) {
  HeteroBlockTable t(small_config(/*blocks=*/8, /*fp16_slots=*/2));
  ASSERT_TRUE(t.allocate(req(kFp16)).ok());
  ASSERT_TRUE(t.allocate(req(kFp16)).ok());
  EXPECT_EQ(t.allocate(req(kFp16)).status(), Status::kOutOfSlots);
  EXPECT_EQ(t.blocks_in_use(), 2u);
  // Other pools are unaffected.
  EXPECT_TRUE(t.allocate(req(kFp8)).ok());
  expect_valid(t);
}

TEST(HeteroBlockTable, IdExhaustion) {
  HeteroBlockTable t(small_config(/*blocks=*/2));
  ASSERT_TRUE(t.allocate(req(kFp16)).ok());
  ASSERT_TRUE(t.allocate(req(kFp8)).ok());
  EXPECT_EQ(t.allocate(req(kInt4)).status(), Status::kOutOfBlocks);
  EXPECT_EQ(t.free_slots(kInt4), 4u);
  expect_valid(t);
}

TEST(HeteroBlockTable, ValueOfFailedResultThrows) {
  HeteroBlockTable t(small_config());
  const Result<BlockHandle> r = t.allocate(req(kUnconfigured));
  EXPECT_THROW((void)r.value(), BadResultAccess);
}

// ---- handle safety ------------------------------------------------------------------

TEST(HeteroBlockTable, StaleHandlesAreRejectedAfterReuse) {
  HeteroBlockTable t(small_config());
  const BlockHandle a = t.allocate(req(kFp16)).value();
  ASSERT_EQ(t.release(a), Status::kOk);
  EXPECT_EQ(t.lookup(a).status(), Status::kStaleHandle);
  EXPECT_EQ(t.release(a), Status::kStaleHandle);  // double free

  const BlockHandle b = t.allocate(req(kFp16)).value();  // LIFO: reuses the same id
  EXPECT_EQ(b.id, a.id);
  EXPECT_NE(b.generation, a.generation);
  EXPECT_EQ(t.lookup(a).status(), Status::kStaleHandle);
  EXPECT_EQ(t.retain(a), Status::kStaleHandle);
  EXPECT_TRUE(t.lookup(b).ok());
  EXPECT_EQ(t.handle_of(b.id), b);
  expect_valid(t);
}

TEST(HeteroBlockTable, InvalidHandles) {
  HeteroBlockTable t(small_config());
  EXPECT_EQ(t.lookup(kInvalidBlock).status(), Status::kInvalidHandle);
  EXPECT_EQ(t.release(BlockHandle{8, 0}), Status::kInvalidHandle);
  EXPECT_EQ(t.lookup(BlockHandle{0, 0}).status(), Status::kStaleHandle);  // never allocated
  EXPECT_EQ(t.handle_of(0), kInvalidBlock);
  EXPECT_EQ(t.handle_of(100), kInvalidBlock);
}

// ---- refcounting --------------------------------------------------------------------

TEST(HeteroBlockTable, RefcountKeepsBlockAlive) {
  HeteroBlockTable t(small_config());
  const BlockHandle h = t.allocate(req(kFp16)).value();
  ASSERT_EQ(t.retain(h), Status::kOk);
  ASSERT_EQ(t.release(h), Status::kOk);
  EXPECT_TRUE(t.lookup(h).ok());
  EXPECT_EQ(t.free_slots(kFp16), 3u);
  ASSERT_EQ(t.release(h), Status::kOk);
  EXPECT_EQ(t.lookup(h).status(), Status::kStaleHandle);
  EXPECT_EQ(t.free_slots(kFp16), 4u);
  expect_valid(t);
}

TEST(HeteroBlockTable, RefcountOverflowIsReported) {
  HeteroBlockTable t(small_config());
  const BlockHandle h = t.allocate(req(kFp16)).value();
  for (std::uint32_t i = 1; i < kMaxRefcount; ++i) ASSERT_EQ(t.retain(h), Status::kOk);
  EXPECT_EQ(t.lookup(h).value().refcount(), kMaxRefcount);
  EXPECT_EQ(t.retain(h), Status::kRefcountOverflow);
  EXPECT_EQ(t.lookup(h).value().refcount(), kMaxRefcount);
  expect_valid(t);
}

// ---- state machine ------------------------------------------------------------------

TEST(HeteroBlockTable, OpenSealedTransitions) {
  HeteroBlockTable t(small_config());
  const BlockHandle h = t.allocate(req(kFp16)).value();
  EXPECT_EQ(t.set_valid_tokens(h, 16), Status::kOk);
  EXPECT_EQ(t.set_valid_tokens(h, 17), Status::kInvalidArgument);
  EXPECT_EQ(t.set_valid_tokens(h, 3), Status::kOk);  // rollback (e.g. rejected draft tokens)
  EXPECT_EQ(t.seal(h), Status::kOk);
  EXPECT_EQ(t.lookup(h).value().state(), BlockState::kSealed);
  EXPECT_EQ(t.seal(h), Status::kInvalidState);
  EXPECT_EQ(t.set_valid_tokens(h, 4), Status::kInvalidState);
  EXPECT_EQ(t.lookup(h).value().exec.valid_tokens(), 3u);
}

TEST(HeteroBlockTable, ResidencyGatesKernelVisibility) {
  HeteroBlockTable t(small_config());
  const BlockHandle h = t.allocate(req(kFp16)).value();
  std::vector<BlockExecDescriptor> out(1);
  const std::vector<BlockHandle> handles{h};
  EXPECT_EQ(t.lookup_batch(handles, out), Status::kOk);
  ASSERT_EQ(t.set_residency(h, Residency::kHost), Status::kOk);
  EXPECT_EQ(t.lookup_batch(handles, out), Status::kNotDeviceResident);
  EXPECT_EQ(t.set_residency(h, static_cast<Residency>(3)), Status::kInvalidArgument);
  ASSERT_EQ(t.set_residency(h, Residency::kDevice), Status::kOk);
  EXPECT_EQ(t.lookup_batch(handles, out), Status::kOk);
}

TEST(HeteroBlockTable, SetScale) {
  HeteroBlockTable t(small_config());
  const BlockHandle h = t.allocate(req(kFp8)).value();
  ASSERT_EQ(t.set_scale(h, ScaleMode::kExternal, 42), Status::kOk);
  const BlockDescriptor d = t.lookup(h).value();
  EXPECT_EQ(d.exec.scale_mode(), ScaleMode::kExternal);
  EXPECT_EQ(d.exec.scale_ref, 42u);
  EXPECT_EQ(t.set_scale(h, static_cast<ScaleMode>(7), 0), Status::kInvalidArgument);
}

// ---- two-phase format change ------------------------------------------------------------

TEST(HeteroBlockTable, FormatChangeCommit) {
  HeteroBlockTable t(small_config());
  const BlockHandle h = t.allocate(req(kFp16, 16)).value();
  EXPECT_EQ(t.begin_format_change(h, kInt4).status(), Status::kInvalidState);  // still open
  ASSERT_EQ(t.seal(h), Status::kOk);

  const FormatChange fc = t.begin_format_change(h, kInt4).value();
  EXPECT_EQ(fc.from_format, kFp16);
  EXPECT_EQ(fc.to_format, kInt4);
  EXPECT_EQ(fc.from_slot, 0u);
  EXPECT_EQ(fc.to_slot, 0u);
  EXPECT_EQ(t.free_slots(kFp16), 3u);  // both slots live during conversion
  EXPECT_EQ(t.free_slots(kInt4), 3u);

  // Still readable through the old representation while transcoding.
  const BlockDescriptor mid = t.lookup(h).value();
  EXPECT_EQ(mid.state(), BlockState::kTranscoding);
  EXPECT_EQ(mid.exec.format(), kFp16);
  EXPECT_EQ(t.seal(h), Status::kInvalidState);
  EXPECT_EQ(t.set_scale(h, ScaleMode::kNone, 0), Status::kInvalidState);
  EXPECT_EQ(t.set_residency(h, Residency::kHost), Status::kInvalidState);
  EXPECT_EQ(t.begin_format_change(h, kFp8).status(), Status::kInvalidState);
  expect_valid(t);

  ASSERT_EQ(t.commit_format_change(h, ScaleMode::kInlineF16Pair, 0x1234'5678u), Status::kOk);
  const BlockDescriptor d = t.lookup(h).value();
  EXPECT_EQ(d.state(), BlockState::kSealed);
  EXPECT_EQ(d.exec.format(), kInt4);
  EXPECT_EQ(d.exec.slot, fc.to_slot);
  EXPECT_EQ(d.exec.scale_mode(), ScaleMode::kInlineF16Pair);
  EXPECT_EQ(d.exec.scale_ref, 0x1234'5678u);
  EXPECT_EQ(d.exec.valid_tokens(), 16u);
  EXPECT_EQ(d.pending_slot, kNoSlot);
  EXPECT_EQ(t.free_slots(kFp16), 4u);
  EXPECT_EQ(t.free_slots(kInt4), 3u);
  EXPECT_EQ(t.stats().format_changes_committed, 1u);
  EXPECT_EQ(t.commit_format_change(h, ScaleMode::kNone, 0), Status::kInvalidState);
  expect_valid(t);
}

TEST(HeteroBlockTable, FormatChangeAbort) {
  HeteroBlockTable t(small_config());
  const BlockHandle h = t.allocate(req(kFp16)).value();
  ASSERT_EQ(t.seal(h), Status::kOk);
  ASSERT_TRUE(t.begin_format_change(h, kFp8).ok());
  ASSERT_EQ(t.abort_format_change(h), Status::kOk);
  const BlockDescriptor d = t.lookup(h).value();
  EXPECT_EQ(d.state(), BlockState::kSealed);
  EXPECT_EQ(d.exec.format(), kFp16);
  EXPECT_EQ(t.free_slots(kFp8), 4u);
  EXPECT_EQ(t.abort_format_change(h), Status::kInvalidState);
  EXPECT_EQ(t.stats().format_changes_committed, 0u);
  expect_valid(t);
}

TEST(HeteroBlockTable, FormatChangeErrors) {
  HeteroBlockTable t(small_config(/*blocks=*/8, 4, 4, /*int4_slots=*/1));
  const BlockHandle a = t.allocate(req(kFp16)).value();
  const BlockHandle b = t.allocate(req(kFp16)).value();
  ASSERT_EQ(t.seal(a), Status::kOk);
  ASSERT_EQ(t.seal(b), Status::kOk);
  EXPECT_EQ(t.begin_format_change(a, kFp16).status(), Status::kInvalidArgument);
  EXPECT_EQ(t.begin_format_change(a, kUnconfigured).status(), Status::kUnknownFormat);
  ASSERT_TRUE(t.begin_format_change(a, kInt4).ok());
  EXPECT_EQ(t.begin_format_change(b, kInt4).status(), Status::kOutOfSlots);
  ASSERT_EQ(t.set_residency(b, Residency::kHost), Status::kOk);
  EXPECT_EQ(t.begin_format_change(b, kFp8).status(), Status::kNotDeviceResident);
  expect_valid(t);
}

TEST(HeteroBlockTable, LastReferenceCannotBeDroppedWhileTranscoding) {
  HeteroBlockTable t(small_config());
  const BlockHandle h = t.allocate(req(kFp16)).value();
  ASSERT_EQ(t.seal(h), Status::kOk);
  ASSERT_TRUE(t.begin_format_change(h, kInt4).ok());
  EXPECT_EQ(t.release(h), Status::kInvalidState);
  ASSERT_EQ(t.retain(h), Status::kOk);
  EXPECT_EQ(t.release(h), Status::kOk);  // not the last reference
  EXPECT_EQ(t.release(h), Status::kInvalidState);
  ASSERT_EQ(t.commit_format_change(h, ScaleMode::kNone, 0), Status::kOk);
  EXPECT_EQ(t.release(h), Status::kOk);
  EXPECT_EQ(t.free_slots(kFp16), 4u);
  EXPECT_EQ(t.free_slots(kInt4), 4u);
  expect_valid(t);
}

// ---- sequences and exec tables ----------------------------------------------------------

TEST(HeteroBlockTable, SequenceLifecycle) {
  HeteroBlockTable t(small_config());
  const SequenceHandle s = t.create_sequence().value();
  const BlockHandle b0 = t.append_block(s, req(kFp16, 16)).value();
  const BlockHandle b1 = t.append_block(s, req(kInt4, 16)).value();
  const BlockHandle b2 = t.append_block(s, req(kFp8, 3)).value();
  const std::span<const std::uint32_t> ids = t.sequence_block_ids(s).value();
  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], b0.id);
  EXPECT_EQ(ids[1], b1.id);
  EXPECT_EQ(ids[2], b2.id);
  expect_valid(t);

  ASSERT_EQ(t.release_sequence(s), Status::kOk);
  EXPECT_EQ(t.blocks_in_use(), 0u);
  EXPECT_EQ(t.lookup(b1).status(), Status::kStaleHandle);
  EXPECT_EQ(t.release_sequence(s), Status::kStaleHandle);
  EXPECT_EQ(t.append_block(s, req(kFp16)).status(), Status::kStaleHandle);
  const SequenceHandle s2 = t.create_sequence().value();  // reuses the id
  EXPECT_EQ(s2.id, s.id);
  EXPECT_NE(s2.generation, s.generation);
  expect_valid(t);
}

TEST(HeteroBlockTable, AppendFailureLeavesSequenceUnchanged) {
  HeteroBlockTable t(small_config(/*blocks=*/8, /*fp16_slots=*/1));
  const SequenceHandle s = t.create_sequence().value();
  ASSERT_TRUE(t.append_block(s, req(kFp16)).ok());
  EXPECT_EQ(t.append_block(s, req(kFp16)).status(), Status::kOutOfSlots);
  EXPECT_EQ(t.sequence_block_ids(s).value().size(), 1u);
  expect_valid(t);
}

TEST(HeteroBlockTable, PrefixSharing) {
  HeteroBlockTable t(small_config());
  const SequenceHandle a = t.create_sequence().value();
  const SequenceHandle b = t.create_sequence().value();
  const BlockHandle prefix = t.append_block(a, req(kFp16, 16)).value();
  EXPECT_EQ(t.append_shared_block(b, prefix), Status::kInvalidState);  // open blocks not shareable
  ASSERT_EQ(t.seal(prefix), Status::kOk);
  ASSERT_EQ(t.append_shared_block(b, prefix), Status::kOk);
  EXPECT_EQ(t.lookup(prefix).value().refcount(), 2u);
  ASSERT_TRUE(t.append_block(b, req(kFp8, 1)).ok());
  expect_valid(t);

  ASSERT_EQ(t.release_sequence(a), Status::kOk);
  EXPECT_TRUE(t.lookup(prefix).ok());
  EXPECT_EQ(t.lookup(prefix).value().refcount(), 1u);
  ASSERT_EQ(t.release_sequence(b), Status::kOk);
  EXPECT_EQ(t.lookup(prefix).status(), Status::kStaleHandle);
  EXPECT_EQ(t.blocks_in_use(), 0u);
  expect_valid(t);
}

TEST(HeteroBlockTable, ReleaseSequenceIsAllOrNothing) {
  HeteroBlockTable t(small_config());
  const SequenceHandle s = t.create_sequence().value();
  const BlockHandle b0 = t.append_block(s, req(kFp16, 16)).value();
  const BlockHandle b1 = t.append_block(s, req(kFp16, 16)).value();
  ASSERT_EQ(t.seal(b1), Status::kOk);
  ASSERT_TRUE(t.begin_format_change(b1, kInt4).ok());
  EXPECT_EQ(t.release_sequence(s), Status::kInvalidState);
  EXPECT_TRUE(t.lookup(b0).ok());  // nothing was released
  EXPECT_EQ(t.sequence_block_ids(s).value().size(), 2u);
  ASSERT_EQ(t.abort_format_change(b1), Status::kOk);
  EXPECT_EQ(t.release_sequence(s), Status::kOk);
  expect_valid(t);
}

TEST(HeteroBlockTable, BuildExecTableIsCsrOfDescriptors) {
  HeteroBlockTable t(small_config(/*blocks=*/16, 8, 8, 8));
  const SequenceHandle a = t.create_sequence().value();
  const SequenceHandle b = t.create_sequence().value();
  const SequenceHandle empty = t.create_sequence().value();
  std::vector<BlockHandle> a_blocks;
  std::vector<BlockHandle> b_blocks;
  a_blocks.push_back(t.append_block(a, req(kFp16, 16)).value());
  a_blocks.push_back(t.append_block(a, req(kInt4, 16)).value());
  b_blocks.push_back(t.append_block(b, req(kFp8, 16)).value());
  a_blocks.push_back(t.append_block(a, req(kFp8, 7)).value());

  ExecTable table;
  const std::vector<SequenceHandle> batch{b, empty, a};
  ASSERT_EQ(t.build_exec_table(batch, table), Status::kOk);
  ASSERT_EQ(table.indptr, (std::vector<std::uint32_t>{0, 1, 1, 4}));
  ASSERT_EQ(table.blocks.size(), 4u);
  EXPECT_EQ(table.blocks[0], t.lookup(b_blocks[0]).value().exec);
  for (std::size_t i = 0; i < a_blocks.size(); ++i) {
    EXPECT_EQ(table.blocks[1 + i], t.lookup(a_blocks[i]).value().exec);
  }
  EXPECT_EQ(table.blocks[3].valid_tokens(), 7u);
  EXPECT_EQ(table.blocks[2].format(), kInt4);

  // Matches lookup_batch on the same handles.
  std::vector<BlockExecDescriptor> direct(a_blocks.size());
  ASSERT_EQ(t.lookup_batch(a_blocks, direct), Status::kOk);
  EXPECT_TRUE(std::equal(direct.begin(), direct.end(), table.blocks.begin() + 1));

  // Reuse of the output object, and failure leaves it empty.
  ASSERT_EQ(t.set_residency(a_blocks[1], Residency::kDropped), Status::kOk);
  EXPECT_EQ(t.build_exec_table(batch, table), Status::kNotDeviceResident);
  EXPECT_TRUE(table.indptr.empty());
  EXPECT_TRUE(table.blocks.empty());
  const std::vector<SequenceHandle> stale{kInvalidSequence};
  EXPECT_EQ(t.build_exec_table(stale, table), Status::kInvalidHandle);
}

TEST(HeteroBlockTable, LookupBatchChecksOutputSize) {
  HeteroBlockTable t(small_config());
  const std::vector<BlockHandle> handles{t.allocate(req(kFp16)).value(),
                                         t.allocate(req(kFp16)).value()};
  std::vector<BlockExecDescriptor> out(1);
  EXPECT_EQ(t.lookup_batch(handles, out), Status::kInvalidArgument);
}

TEST(HeteroBlockTable, StatsReflectMixedPrecision) {
  HeteroBlockTable t(small_config());
  const BlockHandle a = t.allocate(req(kFp16, 16)).value();
  ASSERT_TRUE(t.allocate(req(kFp8, 8)).ok());
  ASSERT_TRUE(t.allocate(req(kInt4, 4)).ok());
  ASSERT_EQ(t.set_residency(a, Residency::kHost), Status::kOk);
  const TableStats st = t.stats();
  EXPECT_EQ(st.blocks_in_use, 3u);
  EXPECT_EQ(st.blocks_capacity, 8u);
  EXPECT_EQ(st.blocks_per_format[0], 1u);
  EXPECT_EQ(st.blocks_per_format[1], 1u);
  EXPECT_EQ(st.blocks_per_format[3], 1u);
  EXPECT_EQ(st.blocks_per_residency[static_cast<std::size_t>(Residency::kHost)], 1u);
  EXPECT_EQ(st.blocks_per_state[static_cast<std::size_t>(BlockState::kFree)], 5u);
  EXPECT_EQ(st.valid_tokens, 28u);
  EXPECT_EQ(st.token_capacity_in_use, 48u);
  EXPECT_EQ(st.payload_bytes_in_use, 4096u + 2048u + 1088u);
}

// ---- persistent device state (delta protocol) ----------------------------------------

using testing::ExecMirror;
using testing::mirror_matches;

bool non_executable(const BlockExecDescriptor& d) {
  return d.format() == kInvalidFormat && d.slot == kNoSlot;
}

std::vector<BlockExecDescriptor> device_view(const HeteroBlockTable& t, SequenceHandle seq) {
  std::vector<BlockExecDescriptor> out;
  const std::span<const std::uint32_t> ids = t.sequence_block_ids(seq).value();
  for (const std::uint32_t id : ids) out.push_back(t.device_descriptor(id));
  return out;
}

TEST(HeteroBlockTableDelta, FirstDeltaInitializesTheWholeTable) {
  HeteroBlockTable t(small_config());
  ExecDelta delta;
  t.collect_delta(delta);
  EXPECT_EQ(delta.descriptor_table_size, 8u);
  EXPECT_EQ(delta.arena_size, 0u);
  EXPECT_TRUE(delta.id_patches.empty());
  ASSERT_EQ(delta.descriptor_patches.size(), 8u);
  for (std::uint32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(delta.descriptor_patches[i].block_id, i);
    EXPECT_TRUE(non_executable(delta.descriptor_patches[i]));
  }
  ExecMirror mirror;
  mirror.apply(delta);
  EXPECT_TRUE(mirror_matches(mirror, t));

  t.collect_delta(delta);  // nothing changed since
  EXPECT_TRUE(delta.descriptor_patches.empty());
  EXPECT_TRUE(delta.id_patches.empty());
  expect_valid(t);
}

TEST(HeteroBlockTableDelta, OnlyChangedRecordsArePatched) {
  HeteroBlockTable t(small_config());
  ExecDelta delta;
  t.collect_delta(delta);  // initial upload
  const SequenceHandle s = t.create_sequence().value();
  const BlockHandle b = t.append_block(s, req(kFp16, 1)).value();
  ASSERT_EQ(t.set_valid_tokens(b, 2), Status::kOk);
  ASSERT_EQ(t.set_valid_tokens(b, 3), Status::kOk);
  EXPECT_EQ(t.pending_descriptor_patches(), 1u);  // deduplicated
  EXPECT_EQ(t.pending_id_patches(), 1u);
  t.collect_delta(delta);
  ASSERT_EQ(delta.descriptor_patches.size(), 1u);
  EXPECT_EQ(delta.descriptor_patches[0], t.lookup(b).value().exec);  // final value
  EXPECT_EQ(delta.descriptor_patches[0].valid_tokens(), 3u);
  ASSERT_EQ(delta.id_patches.size(), 1u);
  EXPECT_EQ(delta.id_patches[0].block_id, b.id);

  ASSERT_EQ(t.seal(b), Status::kOk);  // host-only state: no device change
  t.collect_delta(delta);
  EXPECT_TRUE(delta.descriptor_patches.empty());
  EXPECT_TRUE(delta.id_patches.empty());
}

TEST(HeteroBlockTableDelta, NonResidentAndFreedBlocksBecomeNonExecutable) {
  HeteroBlockTable t(small_config());
  ExecMirror mirror;
  ExecDelta delta;
  const SequenceHandle s = t.create_sequence().value();
  const BlockHandle a = t.append_block(s, req(kFp16, 16)).value();
  const BlockHandle b = t.append_block(s, req(kInt4, 16)).value();
  ASSERT_EQ(t.set_residency(b, Residency::kHost), Status::kOk);
  t.collect_delta(delta);
  mirror.apply(delta);
  EXPECT_TRUE(mirror_matches(mirror, t));
  EXPECT_TRUE(non_executable(mirror.table()[b.id]));
  EXPECT_FALSE(non_executable(mirror.table()[a.id]));
  // The materialized reference path refuses; the persistent path marks the record.
  ExecTable table;
  const std::vector<SequenceHandle> batch{s};
  EXPECT_EQ(t.build_exec_table(batch, table), Status::kNotDeviceResident);
  std::vector<SequenceSegment> segments;
  ASSERT_EQ(t.batch_segments(batch, segments), Status::kOk);
  EXPECT_EQ(mirror.read(segments[0]), device_view(t, s));

  ASSERT_EQ(t.set_residency(b, Residency::kDevice), Status::kOk);
  ASSERT_EQ(t.release_sequence(s), Status::kOk);
  t.collect_delta(delta);
  mirror.apply(delta);
  EXPECT_TRUE(mirror_matches(mirror, t));
  EXPECT_TRUE(non_executable(mirror.table()[a.id]));
  EXPECT_TRUE(non_executable(mirror.table()[b.id]));
}

TEST(HeteroBlockTableDelta, SharedBlockChangeIsASinglePatch) {
  HeteroBlockTable t(small_config(/*blocks=*/16, 8, 8, 8));
  ExecMirror mirror;
  ExecDelta delta;
  std::vector<SequenceHandle> seqs;
  for (int i = 0; i < 3; ++i) seqs.push_back(t.create_sequence().value());
  const BlockHandle prefix = t.append_block(seqs[0], req(kFp16, 16)).value();
  ASSERT_EQ(t.seal(prefix), Status::kOk);
  ASSERT_EQ(t.append_shared_block(seqs[1], prefix), Status::kOk);
  ASSERT_EQ(t.append_shared_block(seqs[2], prefix), Status::kOk);
  for (const SequenceHandle s : seqs) ASSERT_TRUE(t.append_block(s, req(kFp8, 1)).ok());
  t.collect_delta(delta);
  mirror.apply(delta);

  ASSERT_TRUE(t.begin_format_change(prefix, kInt4).ok());
  ASSERT_EQ(t.commit_format_change(prefix, ScaleMode::kInlineF32, 0x3F80'0000u), Status::kOk);
  t.collect_delta(delta);
  EXPECT_EQ(delta.descriptor_patches.size(), 1u);  // one record, three readers
  EXPECT_TRUE(delta.id_patches.empty());
  mirror.apply(delta);

  std::vector<SequenceSegment> segments;
  ASSERT_EQ(t.batch_segments(seqs, segments), Status::kOk);
  ExecTable reference;
  ASSERT_EQ(t.build_exec_table(seqs, reference), Status::kOk);
  const ExecTable seen = mirror.read_batch(segments);
  EXPECT_EQ(seen.indptr, reference.indptr);
  EXPECT_EQ(seen.blocks, reference.blocks);
  for (std::size_t i = 0; i < 3; ++i) EXPECT_EQ(seen.blocks[2 * i].format(), kInt4);
}

TEST(HeteroBlockTableDelta, RelocationIsCarriedByIdPatches) {
  TableConfig cfg = small_config(/*blocks=*/64, 64, 64, 64);
  HeteroBlockTable t(cfg);
  ExecMirror mirror;
  ExecDelta delta;
  const SequenceHandle a = t.create_sequence().value();
  const SequenceHandle b = t.create_sequence().value();
  for (int i = 0; i < 20; ++i) {  // a: 8 -> 16 -> 32 entries, interleaved with b
    ASSERT_TRUE(t.append_block(a, req(kFp16, 16)).ok());
    if (i % 4 == 0) {
      ASSERT_TRUE(t.append_block(b, req(kFp8, 16)).ok());
    }
    if (i % 7 == 0) {  // apply some deltas mid-way, some batched
      t.collect_delta(delta);
      mirror.apply(delta);
    }
  }
  t.collect_delta(delta);
  mirror.apply(delta);
  EXPECT_EQ(delta.arena_size, t.stats().arena_entries);
  std::vector<SequenceSegment> segments;
  const std::vector<SequenceHandle> batch{a, b};
  ASSERT_EQ(t.batch_segments(batch, segments), Status::kOk);
  EXPECT_EQ(segments[0].length, 20u);
  EXPECT_EQ(mirror.read(segments[0]), device_view(t, a));
  EXPECT_EQ(mirror.read(segments[1]), device_view(t, b));
  expect_valid(t);
}

TEST(HeteroBlockTableDelta, BatchSegmentsValidatesHandles) {
  HeteroBlockTable t(small_config());
  const SequenceHandle s = t.create_sequence().value();
  std::vector<SequenceSegment> segments{{1, 2}};
  const std::vector<SequenceHandle> ok_batch{s};
  ASSERT_EQ(t.batch_segments(ok_batch, segments), Status::kOk);
  EXPECT_EQ(segments, (std::vector<SequenceSegment>{{0, 0}}));  // no storage yet
  const std::vector<SequenceHandle> bad_batch{s, kInvalidSequence};
  EXPECT_EQ(t.batch_segments(bad_batch, segments), Status::kInvalidHandle);
  EXPECT_TRUE(segments.empty());
}

TEST(HeteroBlockTableDelta, StatsReportArenaAndPendingPatches) {
  HeteroBlockTable t(small_config());
  ExecDelta delta;
  EXPECT_EQ(t.stats().pending_descriptor_patches, 8u);  // initial upload pending
  t.collect_delta(delta);
  const SequenceHandle s = t.create_sequence().value();
  ASSERT_TRUE(t.append_block(s, req(kFp16, 4)).ok());
  const TableStats st = t.stats();
  EXPECT_EQ(st.arena_entries, IdArena::kMinSegment);
  EXPECT_EQ(st.arena_live_entries, 1u);
  EXPECT_EQ(st.arena_reserved_entries, IdArena::kMinSegment);
  EXPECT_EQ(st.arena_free_entries, 0u);
  EXPECT_EQ(st.pending_descriptor_patches, 1u);
  EXPECT_EQ(st.pending_id_patches, 1u);
  ASSERT_EQ(t.release_sequence(s), Status::kOk);
  EXPECT_EQ(t.stats().arena_free_entries, IdArena::kMinSegment);
}

// ---- randomized invariants + differential check of the delta protocol ------------------

TEST(HeteroBlockTableStress, RandomOperationsPreserveInvariants) {
  TableConfig cfg;
  cfg.max_blocks = 96;
  cfg.block_tokens = 16;
  cfg.pools = {{kFp16, 4096, 48}, {kFp8, 2048, 48}, {kInt4, 1088, 64}};
  HeteroBlockTable t(cfg);
  ExecMirror mirror;
  ExecDelta delta;

  std::mt19937 rng(20261011);  // deterministic; only the engine (not distributions) is portable
  auto pick = [&](std::size_t n) { return static_cast<std::size_t>(rng() % n); };
  const std::array<FormatId, 3> formats{kFp16, kFp8, kInt4};
  const std::array<Residency, 3> residencies{Residency::kDevice, Residency::kHost,
                                             Residency::kDropped};
  auto random_block = [&](const std::vector<SequenceHandle>& seqs) -> BlockHandle {
    if (seqs.empty()) return kInvalidBlock;
    const std::span<const std::uint32_t> ids = t.sequence_block_ids(seqs[pick(seqs.size())]).value();
    return ids.empty() ? kInvalidBlock : t.handle_of(ids[pick(ids.size())]);
  };

  std::vector<SequenceHandle> seqs;
  std::vector<BlockHandle> transcoding;
  std::size_t comparisons = 0;
  for (int step = 0; step < 20000; ++step) {
    switch (pick(11)) {
      case 0:
        if (Result<SequenceHandle> s = t.create_sequence(); s.ok()) seqs.push_back(s.value());
        break;
      case 1:
      case 2:
        if (!seqs.empty()) {
          (void)t.append_block(seqs[pick(seqs.size())], req(formats[pick(3)], 16));
        }
        break;
      case 3:  // share a random sealed block into another sequence
        if (const BlockHandle h = random_block(seqs); h != kInvalidBlock) {
          (void)t.seal(h);
          (void)t.append_shared_block(seqs[pick(seqs.size())], h);
        }
        break;
      case 4:  // start a format change
        if (const BlockHandle h = random_block(seqs); h != kInvalidBlock) {
          (void)t.seal(h);
          if (t.begin_format_change(h, formats[pick(3)]).ok()) transcoding.push_back(h);
        }
        break;
      case 5:  // finish a format change
        if (!transcoding.empty()) {
          const std::size_t i = pick(transcoding.size());
          const Status s = pick(2) == 0 ? t.commit_format_change(transcoding[i], ScaleMode::kNone, 0)
                                        : t.abort_format_change(transcoding[i]);
          ASSERT_EQ(s, Status::kOk);
          transcoding.erase(transcoding.begin() + static_cast<std::ptrdiff_t>(i));
        }
        break;
      case 6:  // offload / drop / restore
        if (const BlockHandle h = random_block(seqs); h != kInvalidBlock) {
          (void)t.set_residency(h, residencies[pick(3)]);
        }
        break;
      case 7:  // token append or rollback on an open block
        if (const BlockHandle h = random_block(seqs); h != kInvalidBlock) {
          (void)t.set_valid_tokens(h, static_cast<std::uint32_t>(pick(17)));
        }
        break;
      default:  // release a sequence (may legitimately fail while transcoding)
        if (!seqs.empty()) {
          const std::size_t i = pick(seqs.size());
          const Status s = t.release_sequence(seqs[i]);
          ASSERT_TRUE(s == Status::kOk || s == Status::kInvalidState) << to_string(s);
          if (s == Status::kOk) seqs.erase(seqs.begin() + static_cast<std::ptrdiff_t>(i));
        }
        break;
    }
    if (step % 97 == 0) {
      const ValidationReport rep = t.validate();
      ASSERT_TRUE(rep.ok()) << "step " << step << ": " << rep.message;
    }
    if (pick(23) == 0) {  // irregular upload cadence: deltas accumulate across many ops
      t.collect_delta(delta);
      mirror.apply(delta);
      ASSERT_TRUE(mirror_matches(mirror, t)) << "step " << step;
      std::vector<SequenceSegment> segments;
      ASSERT_EQ(t.batch_segments(seqs, segments), Status::kOk);
      for (std::size_t i = 0; i < seqs.size(); ++i) {
        ASSERT_EQ(mirror.read(segments[i]), device_view(t, seqs[i])) << "step " << step;
      }
      ExecTable reference;
      if (t.build_exec_table(seqs, reference) == Status::kOk) {
        const ExecTable seen = mirror.read_batch(segments);
        ASSERT_EQ(seen.indptr, reference.indptr) << "step " << step;
        ASSERT_EQ(seen.blocks, reference.blocks) << "step " << step;
        ++comparisons;
      }
    }
  }
  EXPECT_GT(comparisons, 10u);  // the full-rebuild oracle was exercised, not just skipped

  for (const BlockHandle h : transcoding) ASSERT_EQ(t.abort_format_change(h), Status::kOk);
  for (const SequenceHandle s : seqs) ASSERT_EQ(t.release_sequence(s), Status::kOk);
  EXPECT_EQ(t.blocks_in_use(), 0u);
  EXPECT_EQ(t.free_slots(kFp16), 48u);
  EXPECT_EQ(t.free_slots(kFp8), 48u);
  EXPECT_EQ(t.free_slots(kInt4), 64u);
  t.collect_delta(delta);
  mirror.apply(delta);
  EXPECT_TRUE(mirror_matches(mirror, t));
  expect_valid(t);
}

}  // namespace
}  // namespace hl::alloc
