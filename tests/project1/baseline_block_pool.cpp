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
  const std::int32_t id = allocate();
  if (id < 0) return false;
  seqs_[seq].push_back(id);
  return true;
}

void HomogeneousBlockPool::release_sequence(std::uint32_t seq) noexcept {
  for (const std::int32_t id : seqs_[seq]) release(id);
  seqs_[seq].clear();
  free_seqs_.push_back(seq);
}

std::size_t HomogeneousBlockPool::sequence_length(std::uint32_t seq) const noexcept {
  return seqs_[seq].size();
}

void HomogeneousBlockPool::build_block_table(std::span<const std::uint32_t> seqs,
                                             std::vector<std::uint32_t>& indptr,
                                             std::vector<std::int32_t>& indices) const {
  indptr.clear();
  indices.clear();
  std::size_t total = 0;
  for (const std::uint32_t s : seqs) total += seqs_[s].size();
  indptr.reserve(seqs.size() + 1);
  indices.reserve(total);
  indptr.push_back(0);
  // The per-sequence int32 list already is the kernel format: a bulk copy.
  for (const std::uint32_t s : seqs) {
    indices.insert(indices.end(), seqs_[s].begin(), seqs_[s].end());
    indptr.push_back(static_cast<std::uint32_t>(indices.size()));
  }
}

}  // namespace hl::bench
