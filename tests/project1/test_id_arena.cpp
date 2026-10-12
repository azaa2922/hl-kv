// Unit tests for IdArena: segment growth/relocation, free-list reuse, dirty
// tracking (unique, final-value patches), partition validation and limits.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include "hl_allocator/id_arena.hpp"

namespace hl::alloc {
namespace {

std::vector<std::uint32_t> to_vector(std::span<const std::uint32_t> s) {
  return {s.begin(), s.end()};
}

void expect_valid(const IdArena& arena, std::initializer_list<IdArena::Segment> live) {
  const std::vector<IdArena::Segment> v(live);
  const std::string e = arena.validate(v);
  EXPECT_TRUE(e.empty()) << e;
}

TEST(IdArena, FirstReserveUsesTheSmallestClass) {
  IdArena arena;
  IdArena::Segment seg;
  EXPECT_TRUE(arena.view(seg).empty());
  EXPECT_EQ(IdArena::kernel_view(seg), (SequenceSegment{0, 0}));
  ASSERT_EQ(arena.reserve(seg, 1), Status::kOk);
  EXPECT_EQ(seg.capacity, IdArena::kMinSegment);
  EXPECT_EQ(seg.offset, 0u);
  EXPECT_EQ(seg.length, 0u);
  EXPECT_EQ(arena.size(), IdArena::kMinSegment);
  EXPECT_EQ(arena.reserve(seg, IdArena::kMinSegment), Status::kOk);  // fits: no change
  EXPECT_EQ(arena.size(), IdArena::kMinSegment);
  expect_valid(arena, {seg});
}

TEST(IdArena, GrowthRelocatesAndPreservesContents) {
  IdArena arena;
  IdArena::Segment a;
  IdArena::Segment b;
  std::vector<std::uint32_t> expect_a;
  for (std::uint32_t i = 0; i < 100; ++i) {
    ASSERT_EQ(arena.reserve(a, a.length + 1), Status::kOk);
    arena.push_back(a, 1000 + i);
    expect_a.push_back(1000 + i);
    if (i % 3 == 0) {  // interleave a second sequence so relocations are not trivial
      ASSERT_EQ(arena.reserve(b, b.length + 1), Status::kOk);
      arena.push_back(b, i);
    }
  }
  EXPECT_EQ(to_vector(arena.view(a)), expect_a);
  EXPECT_EQ(a.capacity, 128u);  // 8 -> 16 -> 32 -> 64 -> 128
  EXPECT_EQ(b.length, 34u);
  EXPECT_EQ(b.capacity, 64u);
  expect_valid(arena, {a, b});
  // Old segments were returned to the free lists, nothing leaked.
  EXPECT_EQ(arena.free_entries() + a.capacity + b.capacity, arena.size());
}

TEST(IdArena, PatchesAreUniqueAndCarryFinalValues) {
  IdArena arena;
  IdArena::Segment seg;
  for (std::uint32_t i = 0; i < 20; ++i) {  // crosses two relocations (8 -> 16 -> 32)
    ASSERT_EQ(arena.reserve(seg, seg.length + 1), Status::kOk);
    arena.push_back(seg, 7 * i);
  }
  std::vector<IdPatch> patches;
  arena.collect(patches);
  std::set<std::uint32_t> offsets;
  for (const IdPatch& p : patches) {
    EXPECT_TRUE(offsets.insert(p.offset).second) << "duplicate offset " << p.offset;
    EXPECT_LT(p.offset, arena.size());
  }
  // Every live entry of the final segment has a patch with its current value.
  for (std::uint32_t j = 0; j < seg.length; ++j) {
    const auto it = std::find_if(patches.begin(), patches.end(),
                                 [&](IdPatch p) { return p.offset == seg.offset + j; });
    ASSERT_NE(it, patches.end());
    EXPECT_EQ(it->block_id, 7 * j);
  }
  EXPECT_EQ(arena.pending_patches(), 0u);
  arena.collect(patches);
  EXPECT_TRUE(patches.empty());
  expect_valid(arena, {seg});
}

TEST(IdArena, ReleasedSegmentsAreReused) {
  IdArena arena;
  IdArena::Segment a;
  IdArena::Segment b;
  ASSERT_EQ(arena.reserve(a, 5), Status::kOk);
  ASSERT_EQ(arena.reserve(b, 5), Status::kOk);
  const std::uint32_t a_offset = a.offset;
  arena.release(a);
  EXPECT_EQ(a.capacity, 0u);
  expect_valid(arena, {b});
  IdArena::Segment c;
  ASSERT_EQ(arena.reserve(c, 3), Status::kOk);
  EXPECT_EQ(c.offset, a_offset);  // LIFO reuse of the same size class
  EXPECT_EQ(arena.size(), 2 * IdArena::kMinSegment);
  expect_valid(arena, {b, c});
}

TEST(IdArena, LimitsAreReported) {
  IdArena arena;
  IdArena::Segment seg;
  EXPECT_EQ(arena.reserve(seg, IdArena::kMaxSegment + 1), Status::kCapacityExceeded);
  EXPECT_EQ(seg.capacity, 0u);
  EXPECT_EQ(arena.size(), 0u);
}

TEST(IdArena, ValidateDetectsLeaksAndOverlaps) {
  IdArena arena;
  IdArena::Segment a;
  IdArena::Segment b;
  ASSERT_EQ(arena.reserve(a, 1), Status::kOk);
  ASSERT_EQ(arena.reserve(b, 1), Status::kOk);
  EXPECT_FALSE(arena.validate(std::vector<IdArena::Segment>{a}).empty());     // b leaked
  EXPECT_FALSE(arena.validate(std::vector<IdArena::Segment>{a, a, b}).empty());  // overlap
  IdArena::Segment bogus = a;
  bogus.capacity = 12;  // not a size class
  EXPECT_FALSE(arena.validate(std::vector<IdArena::Segment>{bogus, b}).empty());
  bogus = a;
  bogus.length = bogus.capacity + 1;
  EXPECT_FALSE(arena.validate(std::vector<IdArena::Segment>{bogus, b}).empty());
}

}  // namespace
}  // namespace hl::alloc
