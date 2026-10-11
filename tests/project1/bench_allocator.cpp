// Project 1 benchmarks: HeteroBlockTable vs a minimal homogeneous baseline.
//
// Every Hetero benchmark has a Baseline twin running the same access pattern
// against HomogeneousBlockPool (see baseline_block_pool.hpp). Both live in
// separately compiled translation units, so neither gets inlined into the loop.
//
//   AllocReleaseChurn  steady-state allocate + release of a random live block
//   LookupRandom       independent validated lookups in random order (throughput)
//   LookupChain        dependent lookups: next id derived from the loaded record (latency)
//   LookupBatch        validated gather of 1024 random blocks into a flat array
//   BuildExecTable     CSR gather for a batch of sequences with scattered block ids
//   DecodeStep         one decode step for S sequences (token append, seal, block
//                      append, sequence turnover) + exec-table rebuild
//
// Run:  hl_allocator_bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true

#include <benchmark/benchmark.h>

#include <array>
#include <cstdlib>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "baseline_block_pool.hpp"
#include "hl_allocator/hetero_block_table.hpp"

namespace {

using hl::alloc::AllocRequest;
using hl::alloc::BlockDescriptor;
using hl::alloc::BlockExecDescriptor;
using hl::alloc::BlockHandle;
using hl::alloc::ExecTable;
using hl::alloc::FormatId;
using hl::alloc::HeteroBlockTable;
using hl::alloc::Result;
using hl::alloc::SequenceHandle;
using hl::alloc::Status;
using hl::alloc::TableConfig;
using hl::bench::HomogeneousBlockPool;

constexpr std::uint32_t kBlockTokens = 16;
constexpr std::array<FormatId, 3> kFormats{FormatId{0}, FormatId{1}, FormatId{3}};  // FP16, FP8, INT4
constexpr std::size_t kInner = 256;  // operations per benchmark-loop iteration

// Deterministic, platform-independent PRNG (std distributions are not portable).
struct SplitMix64 {
  std::uint64_t state;
  std::uint64_t next() noexcept {
    std::uint64_t z = (state += 0x9E37'79B9'7F4A'7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58'476D'1CE4'E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D0'49BB'1331'11EBull;
    return z ^ (z >> 31);
  }
  /// Uniform in [0, n) via multiply-shift (no division in the timed loop).
  std::size_t below(std::size_t n) noexcept {
    return static_cast<std::size_t>(((next() >> 32) * static_cast<std::uint64_t>(n)) >> 32);
  }
};

template <typename T>
void shuffle(std::vector<T>& v, SplitMix64& rng) {
  for (std::size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[rng.below(i)]);
}

// Slot sizes are illustrative (one Llama-3-8B-like layer: 16 tokens x 8 KV heads
// x 128 dims x {K,V}); they do not influence any timing below. Every pool can
// hold every block so pool exhaustion never interferes.
TableConfig make_config(std::uint32_t blocks) {
  TableConfig c;
  c.max_blocks = blocks;
  c.block_tokens = kBlockTokens;
  c.pools = {{kFormats[0], 65536, blocks}, {kFormats[1], 32768, blocks}, {kFormats[2], 16384 + 1024, blocks}};
  return c;
}

AllocRequest req(std::size_t i, std::uint32_t tokens = 0) {
  AllocRequest r;
  r.format = kFormats[i % kFormats.size()];
  r.valid_tokens = tokens;
  return r;
}

template <typename T>
T must(Result<T> r) {
  if (!r.ok()) std::abort();
  return std::move(r).value();
}

void must(Status s) {
  if (s != Status::kOk) std::abort();
}

double mib(std::uint32_t n, std::size_t bytes_each) {
  return static_cast<double>(n) * static_cast<double>(bytes_each) / (1024.0 * 1024.0);
}

std::int64_t per_iteration(const benchmark::State& state, std::size_t count) {
  return state.iterations() * static_cast<std::int64_t>(count);
}

// ---- allocation churn ------------------------------------------------------------------

void BM_Hetero_AllocReleaseChurn(benchmark::State& state) {
  const auto n = static_cast<std::uint32_t>(state.range(0));
  HeteroBlockTable t(make_config(n));
  SplitMix64 rng{1};
  std::vector<BlockHandle> live;
  for (std::uint32_t i = 0; i < n; ++i) live.push_back(must(t.allocate(req(i))));
  shuffle(live, rng);
  for (std::size_t i = n / 2; i < n; ++i) must(t.release(live[i]));  // 50% occupancy, scattered
  live.resize(n / 2);

  std::size_t fmt = 0;
  for (auto _ : state) {
    for (std::size_t k = 0; k < kInner; ++k) {
      const BlockHandle h = must(t.allocate(req(fmt++)));
      BlockHandle& victim = live[rng.below(live.size())];
      must(t.release(victim));
      victim = h;
    }
  }
  state.SetItemsProcessed(per_iteration(state, kInner * 2));
  state.SetLabel("items = allocate + release ops");
}

void BM_Baseline_AllocReleaseChurn(benchmark::State& state) {
  const auto n = static_cast<std::uint32_t>(state.range(0));
  HomogeneousBlockPool p(n);
  SplitMix64 rng{1};
  std::vector<std::int32_t> live;
  for (std::uint32_t i = 0; i < n; ++i) live.push_back(p.allocate());
  shuffle(live, rng);
  for (std::size_t i = n / 2; i < n; ++i) p.release(live[i]);
  live.resize(n / 2);

  for (auto _ : state) {
    for (std::size_t k = 0; k < kInner; ++k) {
      const std::int32_t id = p.allocate();
      if (id < 0) std::abort();
      std::int32_t& victim = live[rng.below(live.size())];
      p.release(victim);
      victim = id;
    }
  }
  state.SetItemsProcessed(per_iteration(state, kInner * 2));
  state.SetLabel("items = allocate + release ops");
}

// ---- single lookups -----------------------------------------------------------------------

void BM_Hetero_LookupRandom(benchmark::State& state) {
  const auto n = static_cast<std::uint32_t>(state.range(0));  // power of two
  HeteroBlockTable t(make_config(n));
  std::vector<BlockHandle> order;
  for (std::uint32_t i = 0; i < n; ++i) order.push_back(must(t.allocate(req(i))));
  SplitMix64 rng{2};
  shuffle(order, rng);

  std::size_t pos = 0;
  for (auto _ : state) {
    for (std::size_t k = 0; k < kInner; ++k) {
      const Result<BlockDescriptor> r = t.lookup(order[pos]);
      std::uint32_t slot = r.value().exec.slot;
      benchmark::DoNotOptimize(slot);
      pos = (pos + 1) & (n - 1);
    }
  }
  state.SetItemsProcessed(per_iteration(state, kInner));
  state.counters["table_MiB"] = mib(n, sizeof(BlockDescriptor));
}

void BM_Baseline_LookupRandom(benchmark::State& state) {
  const auto n = static_cast<std::uint32_t>(state.range(0));
  HomogeneousBlockPool p(n);
  std::vector<std::int32_t> order;
  for (std::uint32_t i = 0; i < n; ++i) order.push_back(p.allocate());
  SplitMix64 rng{2};
  shuffle(order, rng);

  std::size_t pos = 0;
  for (auto _ : state) {
    for (std::size_t k = 0; k < kInner; ++k) {
      std::int32_t phys = p.lookup(order[pos]);
      benchmark::DoNotOptimize(phys);
      pos = (pos + 1) & (n - 1);
    }
  }
  state.SetItemsProcessed(per_iteration(state, kInner));
  state.counters["table_MiB"] = mib(n, sizeof(std::uint16_t));
}

// Full-period LCG over [0, n) for power-of-two n (Hull–Dobell: a % 4 == 1, c odd).
constexpr std::uint32_t kLcgA = 0x9E37'79B1u;
constexpr std::uint32_t kLcgC = 0x7F4A'7C15u;

void BM_Hetero_LookupChain(benchmark::State& state) {
  const auto n = static_cast<std::uint32_t>(state.range(0));
  HeteroBlockTable t(make_config(n));
  for (std::uint32_t i = 0; i < n; ++i) (void)must(t.allocate(req(i)));  // all generation 0

  std::uint32_t id = 0;
  for (auto _ : state) {
    for (std::size_t k = 0; k < kInner; ++k) {
      const Result<BlockDescriptor> r = t.lookup(BlockHandle{id, 0});
      id = (kLcgA * r.value().exec.block_id + kLcgC) & (n - 1);  // depends on the load
    }
  }
  benchmark::DoNotOptimize(id);
  state.SetItemsProcessed(per_iteration(state, kInner));
  state.counters["table_MiB"] = mib(n, sizeof(BlockDescriptor));
}

void BM_Baseline_LookupChain(benchmark::State& state) {
  const auto n = static_cast<std::uint32_t>(state.range(0));
  HomogeneousBlockPool p(n);
  for (std::uint32_t i = 0; i < n; ++i) (void)p.allocate();

  std::uint32_t id = 0;
  for (auto _ : state) {
    for (std::size_t k = 0; k < kInner; ++k) {
      const auto phys = static_cast<std::uint32_t>(p.lookup(static_cast<std::int32_t>(id)));
      id = (kLcgA * phys + kLcgC) & (n - 1);
    }
  }
  benchmark::DoNotOptimize(id);
  state.SetItemsProcessed(per_iteration(state, kInner));
  state.counters["table_MiB"] = mib(n, sizeof(std::uint16_t));
}

// ---- batch lookup -------------------------------------------------------------------------

constexpr std::size_t kBatch = 1024;

void BM_Hetero_LookupBatch(benchmark::State& state) {
  const auto n = static_cast<std::uint32_t>(state.range(0));  // power of two >= kBatch
  HeteroBlockTable t(make_config(n));
  std::vector<BlockHandle> order;
  for (std::uint32_t i = 0; i < n; ++i) order.push_back(must(t.allocate(req(i))));
  SplitMix64 rng{3};
  shuffle(order, rng);
  std::vector<BlockExecDescriptor> out(kBatch);

  std::size_t pos = 0;
  for (auto _ : state) {
    must(t.lookup_batch(std::span<const BlockHandle>(order).subspan(pos, kBatch), out));
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
    pos = (pos + kBatch) & (n - 1);
  }
  state.SetItemsProcessed(per_iteration(state, kBatch));
  state.SetBytesProcessed(
      per_iteration(state, kBatch * sizeof(BlockExecDescriptor)));
}

void BM_Baseline_LookupBatch(benchmark::State& state) {
  const auto n = static_cast<std::uint32_t>(state.range(0));
  HomogeneousBlockPool p(n);
  std::vector<std::int32_t> order;
  for (std::uint32_t i = 0; i < n; ++i) order.push_back(p.allocate());
  SplitMix64 rng{3};
  shuffle(order, rng);
  std::vector<std::int32_t> out(kBatch);

  std::size_t pos = 0;
  for (auto _ : state) {
    if (!p.lookup_batch(std::span<const std::int32_t>(order).subspan(pos, kBatch), out)) {
      std::abort();
    }
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
    pos = (pos + kBatch) & (n - 1);
  }
  state.SetItemsProcessed(per_iteration(state, kBatch));
  state.SetBytesProcessed(
      per_iteration(state, kBatch * sizeof(std::int32_t)));
}

// ---- exec / block table build for a batch of sequences --------------------------------------

// Scrambles the free list (allocate everything, release in random order) so
// that later appends receive scattered ids, as after a period of churn.
void scramble(HeteroBlockTable& t, std::uint32_t n, SplitMix64& rng) {
  std::vector<BlockHandle> all;
  for (std::uint32_t i = 0; i < n; ++i) all.push_back(must(t.allocate(req(0))));
  shuffle(all, rng);
  for (const BlockHandle h : all) must(t.release(h));
}

void scramble(HomogeneousBlockPool& p, std::uint32_t n, SplitMix64& rng) {
  std::vector<std::int32_t> all;
  for (std::uint32_t i = 0; i < n; ++i) all.push_back(p.allocate());
  shuffle(all, rng);
  for (const std::int32_t id : all) p.release(id);
}

void BM_Hetero_BuildExecTable(benchmark::State& state) {
  const auto seqs = static_cast<std::uint32_t>(state.range(0));
  const auto blocks_per_seq = static_cast<std::uint32_t>(state.range(1));
  const std::uint32_t n = seqs * blocks_per_seq;
  HeteroBlockTable t(make_config(n));
  SplitMix64 rng{4};
  scramble(t, n, rng);
  std::vector<SequenceHandle> batch;
  for (std::uint32_t s = 0; s < seqs; ++s) batch.push_back(must(t.create_sequence()));
  for (std::uint32_t b = 0; b < blocks_per_seq; ++b) {  // interleaved growth
    for (std::uint32_t s = 0; s < seqs; ++s) (void)must(t.append_block(batch[s], req(b, kBlockTokens)));
  }

  ExecTable table;
  for (auto _ : state) {
    must(t.build_exec_table(batch, table));
    benchmark::DoNotOptimize(table.blocks.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(per_iteration(state, n));
  state.SetBytesProcessed(per_iteration(
      state, n * sizeof(BlockExecDescriptor) + (seqs + 1) * sizeof(std::uint32_t)));
  state.counters["tokens_per_seq"] = static_cast<double>(blocks_per_seq) * kBlockTokens;
}

void BM_Baseline_BuildBlockTable(benchmark::State& state) {
  const auto seqs = static_cast<std::uint32_t>(state.range(0));
  const auto blocks_per_seq = static_cast<std::uint32_t>(state.range(1));
  const std::uint32_t n = seqs * blocks_per_seq;
  HomogeneousBlockPool p(n);
  SplitMix64 rng{4};
  scramble(p, n, rng);
  std::vector<std::uint32_t> batch;
  for (std::uint32_t s = 0; s < seqs; ++s) batch.push_back(p.create_sequence());
  for (std::uint32_t b = 0; b < blocks_per_seq; ++b) {
    for (std::uint32_t s = 0; s < seqs; ++s) {
      if (!p.append_block(batch[s])) std::abort();
    }
  }

  std::vector<std::uint32_t> indptr;
  std::vector<std::int32_t> indices;
  for (auto _ : state) {
    p.build_block_table(batch, indptr, indices);
    benchmark::DoNotOptimize(indices.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(per_iteration(state, n));
  state.SetBytesProcessed(
      per_iteration(state, n * sizeof(std::int32_t) + (seqs + 1) * sizeof(std::uint32_t)));
  state.counters["tokens_per_seq"] = static_cast<double>(blocks_per_seq) * kBlockTokens;
}

// ---- dynamic sequence tables: full decode step ----------------------------------------------

// Each sequence has a target length (uniform in [1, max_blocks] blocks). Every
// step appends one token to every sequence; a full block is sealed and a new
// one appended; a sequence that reaches its target is released and replaced
// (continuous batching). The exec table for all sequences is rebuilt each step.
struct SeqState {
  std::uint32_t target_blocks;
  std::uint32_t blocks;
  std::uint32_t tokens_in_last;
};

void BM_Hetero_DecodeStep(benchmark::State& state) {
  const auto num_seqs = static_cast<std::uint32_t>(state.range(0));
  const auto max_blocks = static_cast<std::uint32_t>(state.range(1));
  HeteroBlockTable t(make_config(num_seqs * max_blocks));
  SplitMix64 rng{5};
  std::vector<SequenceHandle> seqs(num_seqs);
  std::vector<BlockHandle> last(num_seqs);
  std::vector<SeqState> st(num_seqs);
  auto start = [&](std::uint32_t s) {
    seqs[s] = must(t.create_sequence());
    last[s] = must(t.append_block(seqs[s], req(rng.below(3))));
    st[s] = {1 + static_cast<std::uint32_t>(rng.below(max_blocks)), 1, 0};
  };
  for (std::uint32_t s = 0; s < num_seqs; ++s) start(s);

  ExecTable table;
  for (auto _ : state) {
    for (std::uint32_t s = 0; s < num_seqs; ++s) {
      SeqState& q = st[s];
      must(t.set_valid_tokens(last[s], ++q.tokens_in_last));
      if (q.tokens_in_last < kBlockTokens) continue;
      must(t.seal(last[s]));
      if (q.blocks == q.target_blocks) {
        must(t.release_sequence(seqs[s]));
        start(s);
      } else {
        last[s] = must(t.append_block(seqs[s], req(q.blocks)));
        ++q.blocks;
        q.tokens_in_last = 0;
      }
    }
    must(t.build_exec_table(seqs, table));
    benchmark::DoNotOptimize(table.blocks.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(per_iteration(state, num_seqs));
  state.SetLabel("items = sequence-steps");
}

void BM_Baseline_DecodeStep(benchmark::State& state) {
  const auto num_seqs = static_cast<std::uint32_t>(state.range(0));
  const auto max_blocks = static_cast<std::uint32_t>(state.range(1));
  HomogeneousBlockPool p(num_seqs * max_blocks);
  SplitMix64 rng{5};
  std::vector<std::uint32_t> seqs(num_seqs);
  std::vector<SeqState> st(num_seqs);
  auto start = [&](std::uint32_t s) {
    seqs[s] = p.create_sequence();
    if (!p.append_block(seqs[s])) std::abort();
    (void)rng.below(3);  // keep the RNG stream identical to the Hetero twin
    st[s] = {1 + static_cast<std::uint32_t>(rng.below(max_blocks)), 1, 0};
  };
  for (std::uint32_t s = 0; s < num_seqs; ++s) start(s);

  std::vector<std::uint32_t> indptr;
  std::vector<std::int32_t> indices;
  for (auto _ : state) {
    for (std::uint32_t s = 0; s < num_seqs; ++s) {
      SeqState& q = st[s];
      if (++q.tokens_in_last < kBlockTokens) continue;
      if (q.blocks == q.target_blocks) {
        p.release_sequence(seqs[s]);
        start(s);
      } else {
        if (!p.append_block(seqs[s])) std::abort();
        ++q.blocks;
        q.tokens_in_last = 0;
      }
    }
    p.build_block_table(seqs, indptr, indices);
    benchmark::DoNotOptimize(indices.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(per_iteration(state, num_seqs));
  state.SetLabel("items = sequence-steps");
}

// ---- registration -------------------------------------------------------------------------

void table_sizes(benchmark::internal::Benchmark* b) {
  // 4K / 64K / 1M / 16M blocks -> 128 KiB / 2 MiB / 32 MiB / 512 MiB of descriptors
  for (const std::int64_t n : {1 << 12, 1 << 16, 1 << 20, 1 << 24}) b->Arg(n);
}

void batch_shapes(benchmark::internal::Benchmark* b) {
  // {sequences, blocks per sequence}; 16 tokens per block
  b->Args({64, 64})->Args({256, 64})->Args({256, 256})->Args({32, 2048});
}

void decode_shapes(benchmark::internal::Benchmark* b) {
  b->Args({64, 128})->Args({256, 128})->Args({256, 512});
}

BENCHMARK(BM_Hetero_AllocReleaseChurn)->Apply(table_sizes);
BENCHMARK(BM_Baseline_AllocReleaseChurn)->Apply(table_sizes);
BENCHMARK(BM_Hetero_LookupRandom)->Apply(table_sizes);
BENCHMARK(BM_Baseline_LookupRandom)->Apply(table_sizes);
BENCHMARK(BM_Hetero_LookupChain)->Apply(table_sizes);
BENCHMARK(BM_Baseline_LookupChain)->Apply(table_sizes);
BENCHMARK(BM_Hetero_LookupBatch)->Apply(table_sizes);
BENCHMARK(BM_Baseline_LookupBatch)->Apply(table_sizes);
BENCHMARK(BM_Hetero_BuildExecTable)->Apply(batch_shapes);
BENCHMARK(BM_Baseline_BuildBlockTable)->Apply(batch_shapes);
BENCHMARK(BM_Hetero_DecodeStep)->Apply(decode_shapes);
BENCHMARK(BM_Baseline_DecodeStep)->Apply(decode_shapes);

}  // namespace

BENCHMARK_MAIN();
