#include "hl_allocator/hetero_block_table.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace hl::alloc {
namespace {

constexpr std::uint32_t kMaxIds = 0xFFFF'FFFEu;  // 0xFFFF'FFFF is the invalid-id sentinel

constexpr std::uint32_t code(FormatId f) noexcept { return static_cast<std::uint32_t>(f); }

constexpr bool valid_scale_mode(ScaleMode m) noexcept {
  return static_cast<std::uint32_t>(m) <= static_cast<std::uint32_t>(ScaleMode::kExternal);
}

constexpr bool valid_residency(Residency r) noexcept {
  return static_cast<std::uint32_t>(r) <= static_cast<std::uint32_t>(Residency::kDropped);
}

// Geometric growth that never reserves "size + 1" exactly (which would make
// repeated appends quadratic). Called before acquiring resources so that a
// bad_alloc cannot leak a block.
void ensure_append_capacity(std::vector<std::uint32_t>& v) {
  if (v.size() == v.capacity()) v.reserve(std::max<std::size_t>(8, v.capacity() * 2));
}

// Fills a LIFO stack so that pops return 0, 1, 2, ... (deterministic order).
void fill_descending(std::vector<std::uint32_t>& stack, std::uint32_t count) {
  stack.clear();
  stack.reserve(count);
  for (std::uint32_t i = count; i > 0; --i) stack.push_back(i - 1);
}

ValidationReport corrupt(std::string message) {
  return ValidationReport{Status::kCorrupt, std::move(message)};
}

}  // namespace

// ---- construction -------------------------------------------------------------

Status HeteroBlockTable::validate_config(const TableConfig& config) noexcept {
  if (config.max_blocks == 0 || config.max_blocks > kMaxIds) return Status::kInvalidArgument;
  if (config.block_tokens == 0 || config.block_tokens > kMaxBlockTokens) {
    return Status::kInvalidArgument;
  }
  if (config.pools.empty()) return Status::kInvalidArgument;
  std::array<bool, kMaxFormats> seen{};
  for (const PoolConfig& p : config.pools) {
    if (code(p.format) >= kMaxFormats) return Status::kUnknownFormat;
    if (seen[code(p.format)]) return Status::kInvalidArgument;  // duplicate format
    seen[code(p.format)] = true;
    if (p.slot_bytes == 0 || p.num_slots == 0 || p.num_slots > kMaxIds) {
      return Status::kInvalidArgument;
    }
  }
  return Status::kOk;
}

HeteroBlockTable::HeteroBlockTable(const TableConfig& config) : block_tokens_(config.block_tokens) {
  if (const Status s = validate_config(config); s != Status::kOk) {
    throw std::invalid_argument(std::string("HeteroBlockTable: invalid TableConfig: ") +
                                to_string(s));
  }
  blocks_.resize(config.max_blocks);
  for (std::uint32_t i = 0; i < config.max_blocks; ++i) {
    BlockDescriptor& d = blocks_[i];
    d = BlockDescriptor{};
    d.exec.slot = kNoSlot;
    d.exec.block_id = i;
    d.exec.set_format(kInvalidFormat);
    d.set_state(BlockState::kFree);
    d.set_pending_format(kInvalidFormat);
    d.pending_slot = kNoSlot;
  }
  fill_descending(free_ids_, config.max_blocks);
  for (const PoolConfig& pc : config.pools) {
    Pool& pool = pools_[code(pc.format)];
    pool.slot_bytes = pc.slot_bytes;
    pool.num_slots = pc.num_slots;
    pool.configured = true;
    fill_descending(pool.free_slots, pc.num_slots);
  }
}

// ---- internal helpers ---------------------------------------------------------------

HeteroBlockTable::Pool* HeteroBlockTable::pool_for(FormatId format) noexcept {
  if (code(format) >= kMaxFormats) return nullptr;
  Pool& p = pools_[code(format)];
  return p.configured ? &p : nullptr;
}

const HeteroBlockTable::Pool* HeteroBlockTable::pool_for(FormatId format) const noexcept {
  if (code(format) >= kMaxFormats) return nullptr;
  const Pool& p = pools_[code(format)];
  return p.configured ? &p : nullptr;
}

Status HeteroBlockTable::check_block(BlockHandle handle) const noexcept {
  if (handle.id >= blocks_.size()) return Status::kInvalidHandle;
  const BlockDescriptor& d = blocks_[handle.id];
  if (d.generation != handle.generation || d.state() == BlockState::kFree) {
    return Status::kStaleHandle;
  }
  return Status::kOk;
}

Status HeteroBlockTable::check_sequence(SequenceHandle seq) const noexcept {
  if (seq.id >= sequences_.size()) return Status::kInvalidHandle;
  const Sequence& s = sequences_[seq.id];
  if (!s.active || s.generation != seq.generation) return Status::kStaleHandle;
  return Status::kOk;
}

Status HeteroBlockTable::check_request(const AllocRequest& request) const noexcept {
  if (pool_for(request.format) == nullptr) return Status::kUnknownFormat;
  if (request.valid_tokens > block_tokens_ || request.layout > kMaxLayoutId ||
      !valid_scale_mode(request.scale_mode)) {
    return Status::kInvalidArgument;
  }
  return Status::kOk;
}

void HeteroBlockTable::free_block(std::uint32_t id) noexcept {
  BlockDescriptor& d = blocks_[id];
  // Pushing back cannot reallocate: a pool stack never holds more than num_slots
  // entries and was reserved to that size at construction.
  if (Pool* p = pool_for(d.exec.format()); p != nullptr && d.exec.slot != kNoSlot) {
    p->free_slots.push_back(d.exec.slot);
  }
  if (Pool* p = pool_for(d.pending_format()); p != nullptr && d.pending_slot != kNoSlot) {
    p->free_slots.push_back(d.pending_slot);
  }
  const std::uint32_t next_generation = d.generation + 1;  // wraps at 2^32 by design
  d = BlockDescriptor{};
  d.exec.slot = kNoSlot;
  d.exec.block_id = id;
  d.exec.set_format(kInvalidFormat);
  d.generation = next_generation;
  d.set_state(BlockState::kFree);
  d.set_pending_format(kInvalidFormat);
  d.pending_slot = kNoSlot;
  free_ids_.push_back(id);  // capacity reserved to max_blocks at construction
  --blocks_in_use_;
}

// ---- block lifecycle ----------------------------------------------------------------

Result<BlockHandle> HeteroBlockTable::allocate(const AllocRequest& request) {
  if (const Status s = check_request(request); s != Status::kOk) return s;
  if (free_ids_.empty()) return Status::kOutOfBlocks;
  Pool& pool = *pool_for(request.format);
  if (pool.free_slots.empty()) return Status::kOutOfSlots;

  const std::uint32_t id = free_ids_.back();
  free_ids_.pop_back();
  const std::uint32_t slot = pool.free_slots.back();
  pool.free_slots.pop_back();

  BlockDescriptor& d = blocks_[id];
  d.exec.slot = slot;
  d.exec.info = BlockExecDescriptor::pack_info(request.format, request.scale_mode, request.layout,
                                               request.valid_tokens);
  d.exec.scale_ref = request.scale_ref;
  d.control = 0;
  d.set_state(BlockState::kOpen);
  d.set_residency(Residency::kDevice);
  d.set_pending_format(kInvalidFormat);
  d.set_refcount(1);
  d.pending_slot = kNoSlot;
  d.reserved = 0;
  ++blocks_in_use_;
  return BlockHandle{id, d.generation};
}

Status HeteroBlockTable::retain(BlockHandle handle) {
  if (const Status s = check_block(handle); s != Status::kOk) return s;
  BlockDescriptor& d = blocks_[handle.id];
  if (d.refcount() >= kMaxRefcount) return Status::kRefcountOverflow;
  d.set_refcount(d.refcount() + 1);
  return Status::kOk;
}

Status HeteroBlockTable::release(BlockHandle handle) {
  if (const Status s = check_block(handle); s != Status::kOk) return s;
  BlockDescriptor& d = blocks_[handle.id];
  // A conversion kernel may be reading the old slot: the last reference cannot
  // go away until the format change is committed or aborted.
  if (d.refcount() == 1 && d.state() == BlockState::kTranscoding) return Status::kInvalidState;
  const std::uint32_t remaining = d.refcount() - 1;
  if (remaining == 0) {
    free_block(handle.id);
  } else {
    d.set_refcount(remaining);
  }
  return Status::kOk;
}

// ---- state transitions --------------------------------------------------------------

Status HeteroBlockTable::set_valid_tokens(BlockHandle handle, std::uint32_t valid_tokens) {
  if (const Status s = check_block(handle); s != Status::kOk) return s;
  BlockDescriptor& d = blocks_[handle.id];
  if (d.state() != BlockState::kOpen) return Status::kInvalidState;
  if (valid_tokens > block_tokens_) return Status::kInvalidArgument;
  d.exec.set_valid_tokens(valid_tokens);
  return Status::kOk;
}

Status HeteroBlockTable::seal(BlockHandle handle) {
  if (const Status s = check_block(handle); s != Status::kOk) return s;
  BlockDescriptor& d = blocks_[handle.id];
  if (d.state() != BlockState::kOpen) return Status::kInvalidState;
  d.set_state(BlockState::kSealed);
  return Status::kOk;
}

Status HeteroBlockTable::set_residency(BlockHandle handle, Residency residency) {
  if (const Status s = check_block(handle); s != Status::kOk) return s;
  if (!valid_residency(residency)) return Status::kInvalidArgument;
  BlockDescriptor& d = blocks_[handle.id];
  if (d.state() == BlockState::kTranscoding) return Status::kInvalidState;
  d.set_residency(residency);
  return Status::kOk;
}

Status HeteroBlockTable::set_scale(BlockHandle handle, ScaleMode mode, std::uint32_t scale_ref) {
  if (const Status s = check_block(handle); s != Status::kOk) return s;
  if (!valid_scale_mode(mode)) return Status::kInvalidArgument;
  BlockDescriptor& d = blocks_[handle.id];
  if (d.state() == BlockState::kTranscoding) return Status::kInvalidState;
  d.exec.set_scale_mode(mode);
  d.exec.scale_ref = scale_ref;
  return Status::kOk;
}

Result<FormatChange> HeteroBlockTable::begin_format_change(BlockHandle handle, FormatId target) {
  if (const Status s = check_block(handle); s != Status::kOk) return s;
  BlockDescriptor& d = blocks_[handle.id];
  if (d.state() != BlockState::kSealed) return Status::kInvalidState;
  if (d.residency() != Residency::kDevice) return Status::kNotDeviceResident;
  Pool* target_pool = pool_for(target);
  if (target_pool == nullptr) return Status::kUnknownFormat;
  if (target == d.exec.format()) return Status::kInvalidArgument;
  if (target_pool->free_slots.empty()) return Status::kOutOfSlots;

  const std::uint32_t to_slot = target_pool->free_slots.back();
  target_pool->free_slots.pop_back();
  d.pending_slot = to_slot;
  d.set_pending_format(target);
  d.set_state(BlockState::kTranscoding);
  return FormatChange{d.exec.format(), d.exec.slot, target, to_slot};
}

Status HeteroBlockTable::commit_format_change(BlockHandle handle, ScaleMode mode,
                                              std::uint32_t scale_ref) {
  if (const Status s = check_block(handle); s != Status::kOk) return s;
  if (!valid_scale_mode(mode)) return Status::kInvalidArgument;
  BlockDescriptor& d = blocks_[handle.id];
  if (d.state() != BlockState::kTranscoding) return Status::kInvalidState;

  pool_for(d.exec.format())->free_slots.push_back(d.exec.slot);
  d.exec.slot = d.pending_slot;
  d.exec.set_format(d.pending_format());
  d.exec.set_scale_mode(mode);
  d.exec.scale_ref = scale_ref;
  d.pending_slot = kNoSlot;
  d.set_pending_format(kInvalidFormat);
  d.set_state(BlockState::kSealed);
  ++format_changes_committed_;
  return Status::kOk;
}

Status HeteroBlockTable::abort_format_change(BlockHandle handle) {
  if (const Status s = check_block(handle); s != Status::kOk) return s;
  BlockDescriptor& d = blocks_[handle.id];
  if (d.state() != BlockState::kTranscoding) return Status::kInvalidState;
  pool_for(d.pending_format())->free_slots.push_back(d.pending_slot);
  d.pending_slot = kNoSlot;
  d.set_pending_format(kInvalidFormat);
  d.set_state(BlockState::kSealed);
  return Status::kOk;
}

// ---- lookup -------------------------------------------------------------------------

Result<BlockDescriptor> HeteroBlockTable::lookup(BlockHandle handle) const {
  if (const Status s = check_block(handle); s != Status::kOk) return s;
  return blocks_[handle.id];
}

Status HeteroBlockTable::lookup_batch(std::span<const BlockHandle> handles,
                                      std::span<BlockExecDescriptor> out) const {
  if (out.size() < handles.size()) return Status::kInvalidArgument;
  for (std::size_t i = 0; i < handles.size(); ++i) {
    const BlockHandle h = handles[i];
    if (const Status s = check_block(h); s != Status::kOk) return s;
    const BlockDescriptor& d = blocks_[h.id];
    if (d.residency() != Residency::kDevice) return Status::kNotDeviceResident;
    out[i] = d.exec;
  }
  return Status::kOk;
}

BlockHandle HeteroBlockTable::handle_of(std::uint32_t block_id) const noexcept {
  if (block_id >= blocks_.size()) return kInvalidBlock;
  const BlockDescriptor& d = blocks_[block_id];
  if (d.state() == BlockState::kFree) return kInvalidBlock;
  return BlockHandle{block_id, d.generation};
}

// ---- sequences ----------------------------------------------------------------------

Result<SequenceHandle> HeteroBlockTable::create_sequence() {
  std::uint32_t id = 0;
  if (!free_sequences_.empty()) {
    id = free_sequences_.back();
    free_sequences_.pop_back();
  } else {
    if (sequences_.size() >= kMaxIds) return Status::kInvalidArgument;
    // Keep the free list able to hold every sequence so release_sequence()
    // never allocates.
    if (free_sequences_.capacity() < sequences_.size() + 1) {
      free_sequences_.reserve(std::max<std::size_t>(8, 2 * (sequences_.size() + 1)));
    }
    id = static_cast<std::uint32_t>(sequences_.size());
    sequences_.emplace_back();
  }
  Sequence& s = sequences_[id];
  s.active = true;
  s.blocks.clear();  // keeps capacity from a previous owner
  return SequenceHandle{id, s.generation};
}

Result<BlockHandle> HeteroBlockTable::append_block(SequenceHandle seq,
                                                   const AllocRequest& request) {
  if (const Status s = check_sequence(seq); s != Status::kOk) return s;
  std::vector<std::uint32_t>& ids = sequences_[seq.id].blocks;
  ensure_append_capacity(ids);
  Result<BlockHandle> block = allocate(request);
  if (!block.ok()) return block.status();
  ids.push_back(block.value().id);  // cannot throw: capacity ensured above
  return block;
}

Status HeteroBlockTable::append_shared_block(SequenceHandle seq, BlockHandle block) {
  if (const Status s = check_sequence(seq); s != Status::kOk) return s;
  if (const Status s = check_block(block); s != Status::kOk) return s;
  // Only immutable blocks may be shared; an open block would receive appends
  // from several sequences.
  if (blocks_[block.id].state() == BlockState::kOpen) return Status::kInvalidState;
  std::vector<std::uint32_t>& ids = sequences_[seq.id].blocks;
  ensure_append_capacity(ids);
  if (const Status s = retain(block); s != Status::kOk) return s;
  ids.push_back(block.id);
  return Status::kOk;
}

Status HeteroBlockTable::release_sequence(SequenceHandle seq) {
  if (const Status s = check_sequence(seq); s != Status::kOk) return s;
  Sequence& sq = sequences_[seq.id];
  // Validate before mutating so the operation is all-or-nothing.
  for (const std::uint32_t id : sq.blocks) {
    if (blocks_[id].state() == BlockState::kTranscoding) return Status::kInvalidState;
  }
  for (const std::uint32_t id : sq.blocks) {
    BlockDescriptor& d = blocks_[id];
    const std::uint32_t remaining = d.refcount() - 1;
    if (remaining == 0) {
      free_block(id);
    } else {
      d.set_refcount(remaining);
    }
  }
  sq.blocks.clear();
  sq.active = false;
  ++sq.generation;
  free_sequences_.push_back(seq.id);  // capacity reserved in create_sequence()
  return Status::kOk;
}

Result<std::span<const std::uint32_t>> HeteroBlockTable::sequence_block_ids(
    SequenceHandle seq) const {
  if (const Status s = check_sequence(seq); s != Status::kOk) return s;
  const std::vector<std::uint32_t>& ids = sequences_[seq.id].blocks;
  return std::span<const std::uint32_t>(ids.data(), ids.size());
}

Status HeteroBlockTable::build_exec_table(std::span<const SequenceHandle> seqs,
                                          ExecTable& out) const {
  out.indptr.clear();
  out.blocks.clear();
  std::size_t total = 0;
  for (const SequenceHandle seq : seqs) {
    if (const Status s = check_sequence(seq); s != Status::kOk) return s;
    total += sequences_[seq.id].blocks.size();
  }
  if (total > std::numeric_limits<std::uint32_t>::max()) return Status::kInvalidArgument;

  out.indptr.reserve(seqs.size() + 1);
  out.blocks.reserve(total);
  out.indptr.push_back(0);
  for (const SequenceHandle seq : seqs) {
    for (const std::uint32_t id : sequences_[seq.id].blocks) {
      const BlockDescriptor& d = blocks_[id];
      if (d.residency() != Residency::kDevice) {
        out.indptr.clear();
        out.blocks.clear();
        return Status::kNotDeviceResident;
      }
      out.blocks.push_back(d.exec);
    }
    out.indptr.push_back(static_cast<std::uint32_t>(out.blocks.size()));
  }
  return Status::kOk;
}

// ---- introspection ------------------------------------------------------------------

bool HeteroBlockTable::has_pool(FormatId format) const noexcept {
  return pool_for(format) != nullptr;
}

std::uint32_t HeteroBlockTable::free_slots(FormatId format) const noexcept {
  const Pool* p = pool_for(format);
  return p == nullptr ? 0 : static_cast<std::uint32_t>(p->free_slots.size());
}

std::uint32_t HeteroBlockTable::pool_slots(FormatId format) const noexcept {
  const Pool* p = pool_for(format);
  return p == nullptr ? 0 : p->num_slots;
}

std::uint32_t HeteroBlockTable::slot_bytes(FormatId format) const noexcept {
  const Pool* p = pool_for(format);
  return p == nullptr ? 0 : p->slot_bytes;
}

TableStats HeteroBlockTable::stats() const {
  TableStats st;
  st.blocks_capacity = max_blocks();
  st.format_changes_committed = format_changes_committed_;
  for (const Sequence& s : sequences_) st.active_sequences += s.active ? 1u : 0u;
  for (const BlockDescriptor& d : blocks_) {
    ++st.blocks_per_state[static_cast<std::size_t>(d.state())];
    if (d.state() == BlockState::kFree) continue;
    ++st.blocks_in_use;
    ++st.blocks_per_format[code(d.exec.format())];
    ++st.blocks_per_residency[static_cast<std::size_t>(d.residency())];
    st.valid_tokens += d.exec.valid_tokens();
    st.payload_bytes_in_use += slot_bytes(d.exec.format());
    if (d.state() == BlockState::kTranscoding) {
      st.payload_bytes_in_use += slot_bytes(d.pending_format());
    }
  }
  st.token_capacity_in_use = static_cast<std::uint64_t>(st.blocks_in_use) * block_tokens_;
  return st;
}

ValidationReport HeteroBlockTable::validate() const {
  const std::size_t n = blocks_.size();

  // Free id stack: in range, unique, all marked free.
  std::vector<std::uint8_t> id_seen(n, 0);
  for (const std::uint32_t id : free_ids_) {
    if (id >= n) return corrupt("free id out of range: " + std::to_string(id));
    if (id_seen[id] != 0) return corrupt("duplicate free id: " + std::to_string(id));
    id_seen[id] = 1;
    if (blocks_[id].state() != BlockState::kFree) {
      return corrupt("id on free list is not free: " + std::to_string(id));
    }
  }

  // Slot ownership per pool: every slot is either free or owned exactly once.
  std::array<std::vector<std::uint8_t>, kMaxFormats> slot_seen;
  for (std::uint32_t f = 0; f < kMaxFormats; ++f) {
    const Pool& p = pools_[f];
    if (!p.configured) continue;
    slot_seen[f].assign(p.num_slots, 0);
    for (const std::uint32_t slot : p.free_slots) {
      if (slot >= p.num_slots || slot_seen[f][slot] != 0) {
        return corrupt("bad free slot " + std::to_string(slot) + " in pool " + std::to_string(f));
      }
      slot_seen[f][slot] = 1;
    }
  }
  auto claim = [&](FormatId format, std::uint32_t slot, std::uint32_t id) -> std::string {
    const Pool* p = pool_for(format);
    if (p == nullptr) return "block " + std::to_string(id) + " uses unconfigured format";
    if (slot >= p->num_slots) return "block " + std::to_string(id) + " slot out of range";
    std::uint8_t& seen = slot_seen[code(format)][slot];
    if (seen != 0) return "slot " + std::to_string(slot) + " owned twice (block " +
                          std::to_string(id) + ")";
    seen = 1;
    return {};
  };

  std::uint32_t in_use = 0;
  for (std::uint32_t id = 0; id < n; ++id) {
    const BlockDescriptor& d = blocks_[id];
    const std::string who = "block " + std::to_string(id) + ": ";
    if (d.exec.block_id != id) return corrupt(who + "block_id field mismatch");
    if (d.state() == BlockState::kFree) {
      if (id_seen[id] == 0) return corrupt(who + "free but not on free list");
      continue;
    }
    ++in_use;
    if (d.refcount() == 0) return corrupt(who + "allocated with refcount 0");
    if (d.exec.info_reserved() != 0 || d.control_reserved() != 0 || d.reserved != 0) {
      return corrupt(who + "reserved bits set");
    }
    if (d.exec.valid_tokens() > block_tokens_) return corrupt(who + "valid_tokens > block_tokens");
    if (!valid_residency(d.residency())) return corrupt(who + "invalid residency");
    if (std::string e = claim(d.exec.format(), d.exec.slot, id); !e.empty()) return corrupt(e);
    if (d.state() == BlockState::kTranscoding) {
      if (d.pending_format() == d.exec.format()) return corrupt(who + "pending format == format");
      if (std::string e = claim(d.pending_format(), d.pending_slot, id); !e.empty()) {
        return corrupt(e);
      }
    } else if (d.pending_slot != kNoSlot || d.pending_format() != kInvalidFormat) {
      return corrupt(who + "pending slot set outside kTranscoding");
    }
  }
  if (in_use != blocks_in_use_) return corrupt("blocks_in_use counter mismatch");
  if (free_ids_.size() + in_use != n) return corrupt("free list size mismatch");

  for (std::uint32_t f = 0; f < kMaxFormats; ++f) {
    if (!pools_[f].configured) continue;
    if (std::find(slot_seen[f].begin(), slot_seen[f].end(), 0) != slot_seen[f].end()) {
      return corrupt("pool " + std::to_string(f) + " leaked a slot");
    }
  }

  // Sequence references never exceed block refcounts.
  std::vector<std::uint32_t> seq_refs(n, 0);
  std::size_t inactive = 0;
  for (std::size_t s = 0; s < sequences_.size(); ++s) {
    const Sequence& sq = sequences_[s];
    if (!sq.active) {
      ++inactive;
      if (!sq.blocks.empty()) return corrupt("inactive sequence holds blocks");
      continue;
    }
    for (const std::uint32_t id : sq.blocks) {
      if (id >= n || blocks_[id].state() == BlockState::kFree) {
        return corrupt("sequence " + std::to_string(s) + " references a free block");
      }
      ++seq_refs[id];
    }
  }
  for (std::uint32_t id = 0; id < n; ++id) {
    if (seq_refs[id] > blocks_[id].refcount()) {
      return corrupt("block " + std::to_string(id) + " has more sequence refs than refcount");
    }
  }
  if (inactive != free_sequences_.size()) return corrupt("sequence free list size mismatch");
  std::vector<std::uint8_t> seq_seen(sequences_.size(), 0);
  for (const std::uint32_t s : free_sequences_) {
    if (s >= sequences_.size() || sequences_[s].active || seq_seen[s] != 0) {
      return corrupt("bad sequence free list entry " + std::to_string(s));
    }
    seq_seen[s] = 1;
  }
  return {};
}

}  // namespace hl::alloc
