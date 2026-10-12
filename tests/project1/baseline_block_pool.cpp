#include "baseline_block_pool.hpp"

namespace hl::bench {

HomogeneousBlockPool::HomogeneousBlockPool(std::uint32_t num_blocks) : refcount_(num_blocks, 0) {
  free_.reserve(num_blocks);
  for (std::uint32_t i = num_blocks; i > 0; --i) free_.push_back(static_cast<std::int32_t>(i - 1));
}

std::int32_t HomogeneousBlockPool::allocate() noexcept {
  if (free_.empty()) return -1;
  const std::int32_t id = free_.back();
  free_.pop_back();
  refcount_[static_cast<std::size_t>(id)] = 1;
  return id;
}

void HomogeneousBlockPool::release(std::int32_t id) noexcept {
  std::uint16_t& rc = refcount_[static_cast<std::size_t>(id)];
  if (--rc == 0) free_.push_back(id);
}

std::int32_t HomogeneousBlockPool::lookup(std::int32_t id) const noexcept {
  return refcount_[static_cast<std::size_t>(id)] != 0 ? id : -1;
}

bool HomogeneousBlockPool::lookup_batch(std::span<const std::int32_t> ids,
                                        std::span<std::int32_t> out) const noexcept {
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (refcount_[static_cast<std::size_t>(ids[i])] == 0) return false;
    out[i] = ids[i];
  }
  return true;
}

std::uint32_t HomogeneousBlockPool::create_sequence() {
  if (!free_seqs_.empty()) {
    const std::uint32_t id = free_seqs_.back();
    free_seqs_.pop_back();
    return id;
  }
  seqs_.emplace_back();
  return static_cast<std::uint32_t>(seqs_.size() - 1);
}

bool HomogeneousBlockPool::append_block(std::uint32_t seq) {
  alloc::IdArena::Segment& segment = seqs_[seq];
  if (arena_.reserve(segment, segment.length + 1) != alloc::Status::kOk) return false;
  const std::int32_t id = allocate();
  if (id < 0) return false;
  arena_.push_back(segment, static_cast<std::uint32_t>(id));
  return true;
}

void HomogeneousBlockPool::release_sequence(std::uint32_t seq) noexcept {
  alloc::IdArena::Segment& segment = seqs_[seq];
  for (const std::uint32_t id : arena_.view(segment)) release(static_cast<std::int32_t>(id));
  arena_.release(segment);
  free_seqs_.push_back(seq);
}

std::size_t HomogeneousBlockPool::sequence_length(std::uint32_t seq) const noexcept {
  return seqs_[seq].length;
}

void HomogeneousBlockPool::build_block_table(std::span<const std::uint32_t> seqs,
                                             std::vector<std::uint32_t>& indptr,
                                             std::vector<std::uint32_t>& indices) const {
  indptr.clear();
  indices.clear();
  std::size_t total = 0;
  for (const std::uint32_t s : seqs) total += seqs_[s].length;
  indptr.reserve(seqs.size() + 1);
  indices.reserve(total);
  indptr.push_back(0);
  // Each segment already is the kernel format: a bulk copy.
  for (const std::uint32_t s : seqs) {
    const std::span<const std::uint32_t> ids = arena_.view(seqs_[s]);
    indices.insert(indices.end(), ids.begin(), ids.end());
    indptr.push_back(static_cast<std::uint32_t>(indices.size()));
  }
}

void HomogeneousBlockPool::batch_segments(std::span<const std::uint32_t> seqs,
                                          std::vector<alloc::SequenceSegment>& out) const {
  out.clear();
  out.reserve(seqs.size());
  for (const std::uint32_t s : seqs) out.push_back(alloc::IdArena::kernel_view(seqs_[s]));
}

}  // namespace hl::bench
