#pragma once

// IdArena — persistent storage for per-sequence block-id lists, with dirty
// tracking so that a device-side mirror can be kept current with deltas.
//
// Every sequence owns one contiguous segment of the arena. Segment capacities
// are powers of two (kMinSegment << k); a full segment is relocated to one of
// twice the size, so appends cost amortized O(1) copies. Released segments go
// to per-size-class free lists and are reused; the arena never shrinks and is
// never compacted (v0).
//
// Every write (append or relocation copy) marks the entry dirty. collect()
// returns one patch per dirty entry, carrying the entry's current value, and
// clears the dirty set. Offsets within one collected batch are unique, so the
// patches can be applied in any order (e.g. by a parallel scatter kernel).
//
// Exception safety: reserve() offers the strong guarantee (it allocates before
// changing any observable state); push_back(), release() and the dirty
// bookkeeping never allocate and are noexcept.

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "hl_allocator/status.hpp"

namespace hl::alloc {

/// Device-side update of one arena entry: arena[offset] = block_id.
struct IdPatch {
  std::uint32_t offset;
  std::uint32_t block_id;
  friend constexpr bool operator==(IdPatch, IdPatch) noexcept = default;
};

/// Per-step kernel view of one sequence: block ids arena[offset .. offset + length).
struct SequenceSegment {
  std::uint32_t offset;
  std::uint32_t length;
  friend constexpr bool operator==(SequenceSegment, SequenceSegment) noexcept = default;
};

class IdArena {
 public:
  static constexpr std::uint32_t kMinSegment = 8;
  static constexpr std::uint32_t kNumClasses = 29;  // capacities 8 .. 2^31
  static constexpr std::uint32_t kMaxSegment = kMinSegment << (kNumClasses - 1);
  /// Offsets are 32-bit and 0xFFFF'FFFF is reserved, so the arena stays below this size.
  static constexpr std::uint64_t kMaxEntries = 0xFFFF'FFFFull;

  /// A sequence's storage. capacity == 0 means "no segment yet" (offset unused).
  struct Segment {
    std::uint32_t offset = 0;
    std::uint32_t capacity = 0;
    std::uint32_t length = 0;
  };

  /// Ensures `seg.capacity >= needed`, relocating the segment (and marking the
  /// moved entries dirty) if necessary. Returns kCapacityExceeded if `needed`
  /// exceeds kMaxSegment or the arena would exceed kMaxEntries. Strong guarantee.
  [[nodiscard]] Status reserve(Segment& seg, std::uint32_t needed);

  /// Appends `block_id`. Precondition: seg.length < seg.capacity (call reserve first).
  void push_back(Segment& seg, std::uint32_t block_id) noexcept;

  /// Returns the segment's storage to the free lists; `seg` becomes empty.
  void release(Segment& seg) noexcept;

  [[nodiscard]] std::span<const std::uint32_t> view(const Segment& seg) const noexcept;

  /// Per-step kernel view of a segment (offset 0 for a sequence without storage).
  [[nodiscard]] static SequenceSegment kernel_view(const Segment& seg) noexcept {
    return SequenceSegment{seg.capacity == 0 ? 0u : seg.offset, seg.length};
  }

  /// Replaces `out` with one patch per dirty entry and clears the dirty set.
  /// Strong guarantee (only `out` may be left reallocated-but-empty on throw).
  void collect(std::vector<IdPatch>& out);

  [[nodiscard]] std::uint32_t size() const noexcept {
    return static_cast<std::uint32_t>(data_.size());
  }
  [[nodiscard]] std::size_t pending_patches() const noexcept { return dirty_list_.size(); }
  /// Sum of the capacities of all segments currently on the free lists.
  [[nodiscard]] std::uint64_t free_entries() const noexcept;

  /// Checks that `live` segments plus the free lists partition the arena
  /// exactly, plus the dirty-tracking invariants. Returns an empty string when
  /// consistent, otherwise a description of the first violation.
  [[nodiscard]] std::string validate(std::span<const Segment> live) const;

 private:
  [[nodiscard]] Status take_segment(std::uint32_t cls, std::uint32_t& offset);
  void give_back(std::uint32_t offset, std::uint32_t capacity) noexcept;
  void mark_dirty(std::uint32_t offset) noexcept;

  std::vector<std::uint32_t> data_;
  std::vector<std::uint8_t> dirty_;           // one flag per entry
  std::vector<std::uint32_t> dirty_list_;     // unique offsets; capacity >= data_.size()
  std::array<std::vector<std::uint32_t>, kNumClasses> free_{};  // free offsets per class
  std::array<std::uint32_t, kNumClasses> carved_{};  // segments ever carved per class
};

}  // namespace hl::alloc
