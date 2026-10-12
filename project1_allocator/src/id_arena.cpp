#include "hl_allocator/id_arena.hpp"

#include <algorithm>
#include <bit>
#include <cassert>

namespace hl::alloc {
namespace {

constexpr std::uint32_t kMinSegmentShift = 3;
static_assert(IdArena::kMinSegment == 1u << kMinSegmentShift);
static_assert(IdArena::kMaxSegment == 1u << 31);

constexpr std::uint32_t capacity_of(std::uint32_t cls) noexcept {
  return IdArena::kMinSegment << cls;
}

// Smallest size class whose capacity is >= needed (needed >= 1).
constexpr std::uint32_t class_for(std::uint32_t needed) noexcept {
  if (needed <= IdArena::kMinSegment) return 0;
  return static_cast<std::uint32_t>(std::bit_width(needed - 1)) - kMinSegmentShift;
}

// Size class of a capacity, or kNumClasses if it is not a class size.
constexpr std::uint32_t class_of(std::uint32_t capacity) noexcept {
  if (capacity < IdArena::kMinSegment || !std::has_single_bit(capacity)) {
    return IdArena::kNumClasses;
  }
  const auto cls = static_cast<std::uint32_t>(std::countr_zero(capacity)) - kMinSegmentShift;
  return cls < IdArena::kNumClasses ? cls : IdArena::kNumClasses;
}

static_assert(class_for(1) == 0 && class_for(8) == 0 && class_for(9) == 1 && class_for(16) == 1);
static_assert(class_for(17) == 2 && class_for(IdArena::kMaxSegment) == IdArena::kNumClasses - 1);
static_assert(class_of(8) == 0 && class_of(IdArena::kMaxSegment) == IdArena::kNumClasses - 1);
static_assert(class_of(12) == IdArena::kNumClasses && class_of(4) == IdArena::kNumClasses);

template <typename T>
void reserve_geometric(std::vector<T>& v, std::size_t n) {
  if (v.capacity() < n) v.reserve(std::max(n, 2 * v.capacity()));
}

}  // namespace

Status IdArena::take_segment(std::uint32_t cls, std::uint32_t& offset) {
  std::vector<std::uint32_t>& free_list = free_[cls];
  if (!free_list.empty()) {
    offset = free_list.back();
    free_list.pop_back();
    return Status::kOk;
  }
  const std::size_t old_size = data_.size();
  const std::size_t new_size = old_size + capacity_of(cls);
  if (new_size > kMaxEntries) return Status::kCapacityExceeded;

  // Allocate everything the no-throw paths will need before changing state:
  // dirty_list_ must be able to hold every entry, and the free list of this
  // class must be able to hold every segment of the class ever carved.
  reserve_geometric(data_, new_size);
  reserve_geometric(dirty_, new_size);
  reserve_geometric(dirty_list_, new_size);
  reserve_geometric(free_list, static_cast<std::size_t>(carved_[cls]) + 1);

  data_.resize(new_size);  // within capacity: cannot throw
  dirty_.resize(new_size, 0);
  ++carved_[cls];
  offset = static_cast<std::uint32_t>(old_size);
  return Status::kOk;
}

void IdArena::give_back(std::uint32_t offset, std::uint32_t capacity) noexcept {
  free_[class_of(capacity)].push_back(offset);  // capacity >= carved_ >= size: no reallocation
}

void IdArena::mark_dirty(std::uint32_t offset) noexcept {
  if (dirty_[offset] != 0) return;
  dirty_[offset] = 1;
  dirty_list_.push_back(offset);  // capacity >= data_.size(): no reallocation
}

Status IdArena::reserve(Segment& seg, std::uint32_t needed) {
  if (needed <= seg.capacity) return Status::kOk;
  if (needed > kMaxSegment) return Status::kCapacityExceeded;
  const std::uint32_t cls = class_for(needed);
  std::uint32_t offset = 0;
  if (const Status s = take_segment(cls, offset); s != Status::kOk) return s;
  for (std::uint32_t i = 0; i < seg.length; ++i) {
    data_[offset + i] = data_[seg.offset + i];
    mark_dirty(offset + i);
  }
  if (seg.capacity != 0) give_back(seg.offset, seg.capacity);
  seg.offset = offset;
  seg.capacity = capacity_of(cls);
  return Status::kOk;
}

void IdArena::push_back(Segment& seg, std::uint32_t block_id) noexcept {
  assert(seg.length < seg.capacity && "IdArena::push_back without reserve()");
  const std::uint32_t pos = seg.offset + seg.length;
  data_[pos] = block_id;
  mark_dirty(pos);
  ++seg.length;
}

void IdArena::release(Segment& seg) noexcept {
  if (seg.capacity != 0) give_back(seg.offset, seg.capacity);
  seg = Segment{};
}

std::span<const std::uint32_t> IdArena::view(const Segment& seg) const noexcept {
  if (seg.capacity == 0) return {};
  return std::span<const std::uint32_t>(data_.data() + seg.offset, seg.length);
}

void IdArena::collect(std::vector<IdPatch>& out) {
  out.clear();
  out.reserve(dirty_list_.size());  // the only allocation; no state changed yet
  for (const std::uint32_t offset : dirty_list_) {
    out.push_back(IdPatch{offset, data_[offset]});
    dirty_[offset] = 0;
  }
  dirty_list_.clear();
}

std::uint64_t IdArena::free_entries() const noexcept {
  std::uint64_t total = 0;
  for (std::uint32_t cls = 0; cls < kNumClasses; ++cls) {
    total += static_cast<std::uint64_t>(free_[cls].size()) * capacity_of(cls);
  }
  return total;
}

std::string IdArena::validate(std::span<const Segment> live) const {
  const std::size_t n = data_.size();
  if (dirty_.size() != n) return "arena: dirty flag array size mismatch";
  if (dirty_list_.capacity() < n) return "arena: dirty list capacity below arena size";

  std::uint64_t carved_total = 0;
  for (std::uint32_t cls = 0; cls < kNumClasses; ++cls) {
    carved_total += static_cast<std::uint64_t>(carved_[cls]) * capacity_of(cls);
    if (free_[cls].size() > carved_[cls] || free_[cls].capacity() < carved_[cls]) {
      return "arena: free-list bookkeeping broken for class " + std::to_string(cls);
    }
  }
  if (carved_total != n) return "arena: carved segments do not add up to the arena size";

  // Live and free segments must partition the arena exactly.
  std::vector<std::uint8_t> covered(n, 0);
  auto cover = [&](std::uint32_t offset, std::uint32_t capacity) -> std::string {
    if (class_of(capacity) == kNumClasses) {
      return "arena: capacity " + std::to_string(capacity) + " is not a size class";
    }
    if (std::uint64_t{offset} + capacity > n) {
      return "arena: segment at " + std::to_string(offset) + " out of bounds";
    }
    for (std::uint32_t i = offset; i < offset + capacity; ++i) {
      if (covered[i] != 0) return "arena: overlapping segments at entry " + std::to_string(i);
      covered[i] = 1;
    }
    return {};
  };
  for (const Segment& s : live) {
    if (s.capacity == 0) {
      if (s.length != 0) return "arena: segment without storage has a length";
      continue;
    }
    if (s.length > s.capacity) return "arena: segment length exceeds capacity";
    if (std::string e = cover(s.offset, s.capacity); !e.empty()) return e;
  }
  for (std::uint32_t cls = 0; cls < kNumClasses; ++cls) {
    for (const std::uint32_t offset : free_[cls]) {
      if (std::string e = cover(offset, capacity_of(cls)); !e.empty()) return e;
    }
  }
  if (std::find(covered.begin(), covered.end(), std::uint8_t{0}) != covered.end()) {
    return "arena: entry not owned by any segment (leak)";
  }

  // Dirty list == set of flagged entries, without duplicates.
  std::vector<std::uint8_t> listed(n, 0);
  for (const std::uint32_t offset : dirty_list_) {
    if (offset >= n || dirty_[offset] == 0 || listed[offset] != 0) {
      return "arena: dirty list entry " + std::to_string(offset) + " inconsistent";
    }
    listed[offset] = 1;
  }
  if (static_cast<std::size_t>(std::count(dirty_.begin(), dirty_.end(), std::uint8_t{1})) !=
      dirty_list_.size()) {
    return "arena: dirty flags set without a dirty list entry";
  }
  return {};
}

}  // namespace hl::alloc
