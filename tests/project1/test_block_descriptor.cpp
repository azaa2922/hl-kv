// ABI-level tests for BlockExecDescriptor / BlockDescriptor: byte layout,
// field packing, independence of packed fields, and golden encodings that must
// not change without bumping kAbiVersion.

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hl_allocator/block_descriptor.hpp"

namespace hl::alloc {
namespace {

TEST(BlockDescriptorAbi, SizesAndAlignment) {
  EXPECT_EQ(sizeof(BlockExecDescriptor), 16u);
  EXPECT_EQ(alignof(BlockExecDescriptor), 16u);
  EXPECT_EQ(sizeof(BlockDescriptor), 32u);
  EXPECT_EQ(alignof(BlockDescriptor), 32u);
  EXPECT_EQ(kAbiVersion, 0u);
}

TEST(BlockDescriptorAbi, GoldenInfoWord) {
  // format 5 | scale_mode 1 << 4 | layout 3 << 6 | valid_tokens 16 << 10
  constexpr std::uint32_t info =
      BlockExecDescriptor::pack_info(FormatId{5}, ScaleMode::kInlineF32, 3, 16);
  static_assert(info == 0x0000'40D5u);
  EXPECT_EQ(info, 0x0000'40D5u);
}

TEST(BlockDescriptorAbi, GoldenByteLayout) {
  BlockDescriptor d{};
  d.exec.slot = 0x0403'0201u;
  d.exec.info = BlockExecDescriptor::pack_info(FormatId{2}, ScaleMode::kExternal, 0, 512);
  d.exec.scale_ref = 0x0C0B'0A09u;
  d.exec.block_id = 0x100F'0E0Du;
  d.generation = 0x1413'1211u;
  d.set_state(BlockState::kSealed);
  d.set_residency(Residency::kHost);
  d.set_pending_format(FormatId{7});
  d.set_refcount(0xABCD);
  d.pending_slot = 0x1C1B'1A19u;
  d.reserved = 0;

  std::array<std::uint8_t, 32> bytes{};
  std::memcpy(bytes.data(), &d, sizeof(d));
  const std::array<std::uint8_t, 32> expected = {
      0x01, 0x02, 0x03, 0x04,  // slot
      0x32, 0x00, 0x08, 0x00,  // info: fmt 2 | mode 3<<4 | valid 512<<10 = 0x00080032
      0x09, 0x0A, 0x0B, 0x0C,  // scale_ref
      0x0D, 0x0E, 0x0F, 0x10,  // block_id
      0x11, 0x12, 0x13, 0x14,  // generation
      0x76, 0xCD, 0xAB, 0x00,  // control: state 2 | res 1<<2 | pending 7<<4 | rc 0xABCD<<8
      0x19, 0x1A, 0x1B, 0x1C,  // pending_slot
      0x00, 0x00, 0x00, 0x00,  // reserved
  };
  EXPECT_EQ(bytes, expected);
}

TEST(BlockDescriptorAbi, ExecFieldsRoundTripAtLimits) {
  for (std::uint32_t f = 0; f < kMaxFormats; ++f) {
    for (std::uint32_t m = 0; m < 4; ++m) {
      for (const std::uint32_t layout : {0u, 1u, kMaxLayoutId}) {
        for (const std::uint32_t tokens : {0u, 1u, 16u, kMaxBlockTokens}) {
          BlockExecDescriptor e{};
          e.info = BlockExecDescriptor::pack_info(FormatId{static_cast<std::uint8_t>(f)},
                                                  static_cast<ScaleMode>(m), layout, tokens);
          EXPECT_EQ(static_cast<std::uint32_t>(e.format()), f);
          EXPECT_EQ(static_cast<std::uint32_t>(e.scale_mode()), m);
          EXPECT_EQ(e.layout(), layout);
          EXPECT_EQ(e.valid_tokens(), tokens);
          EXPECT_EQ(e.info_reserved(), 0u);
        }
      }
    }
  }
}

TEST(BlockDescriptorAbi, SettersDoNotDisturbNeighbourFields) {
  BlockExecDescriptor e{};
  e.info = BlockExecDescriptor::pack_info(FormatId{14}, ScaleMode::kInlineF16Pair, kMaxLayoutId,
                                          kMaxBlockTokens);
  e.set_valid_tokens(7);
  EXPECT_EQ(static_cast<std::uint32_t>(e.format()), 14u);
  EXPECT_EQ(e.scale_mode(), ScaleMode::kInlineF16Pair);
  EXPECT_EQ(e.layout(), kMaxLayoutId);
  EXPECT_EQ(e.valid_tokens(), 7u);
  e.set_format(FormatId{1});
  e.set_scale_mode(ScaleMode::kNone);
  EXPECT_EQ(e.valid_tokens(), 7u);
  EXPECT_EQ(e.layout(), kMaxLayoutId);

  BlockDescriptor d{};
  d.set_refcount(kMaxRefcount);
  d.set_state(BlockState::kTranscoding);
  d.set_residency(Residency::kDropped);
  d.set_pending_format(FormatId{14});
  EXPECT_EQ(d.refcount(), kMaxRefcount);
  EXPECT_EQ(d.state(), BlockState::kTranscoding);
  EXPECT_EQ(d.residency(), Residency::kDropped);
  EXPECT_EQ(static_cast<std::uint32_t>(d.pending_format()), 14u);
  EXPECT_EQ(d.control_reserved(), 0u);
  d.set_refcount(1);
  EXPECT_EQ(d.state(), BlockState::kTranscoding);
  EXPECT_EQ(d.residency(), Residency::kDropped);
}

TEST(BlockDescriptorAbi, OversizedValuesAreMaskedNotSpilled) {
  BlockExecDescriptor e{};
  e.info = BlockExecDescriptor::pack_info(FormatId{3}, ScaleMode::kNone, 0, 0);
  e.set_valid_tokens(0xFFFF'FFFFu);  // callers validate; the setter must still not corrupt
  EXPECT_EQ(static_cast<std::uint32_t>(e.format()), 3u);
  EXPECT_EQ(e.info_reserved(), 0u);
}

TEST(BlockDescriptorAbi, InlineScaleIsBinary32Bits) {
  BlockExecDescriptor e{};
  e.scale_ref = std::bit_cast<std::uint32_t>(0.0123f);
  EXPECT_EQ(e.inline_scale_f32(), 0.0123f);
}

TEST(BlockDescriptorAbi, VectorStorageNeverStraddlesCacheLines) {
  std::vector<BlockDescriptor> v(257);
  for (const BlockDescriptor& d : v) {
    const auto addr = reinterpret_cast<std::uintptr_t>(&d);
    EXPECT_EQ(addr % 32, 0u);
    EXPECT_EQ(addr / 64, (addr + sizeof(BlockDescriptor) - 1) / 64);
  }
}

}  // namespace
}  // namespace hl::alloc
