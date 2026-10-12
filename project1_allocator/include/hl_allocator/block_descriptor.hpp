#pragma once

// HL-KV ABI v0 — per-block metadata records.
//
// Two records are defined here:
//
//   BlockExecDescriptor (16 B, alignas(16))
//     The device-facing part. It is exactly what an attention / dequant kernel
//     needs to consume one KV block: where the payload lives (format + slot),
//     how many tokens are valid, and how scales are found. The device keeps a
//     persistent table of these records indexed by block id (updated with
//     self-addressed patches, see HeteroBlockTable::collect_delta), so one block
//     costs one 128-bit load on the device.
//
//   BlockDescriptor (32 B, alignas(32))
//     The host-side record held in HeteroBlockTable, indexed by block id. The
//     first 16 bytes ARE the BlockExecDescriptor; the second 16 bytes are
//     host-only control state (generation, lifecycle, residency, refcount,
//     pending format change). alignas(32) guarantees a record never straddles
//     a 64-byte cache line, so a validated lookup (generation check + exec
//     copy) touches exactly one line.
//
// Packed fields use explicit shift/mask on uint32_t words instead of C++
// bit-fields: bit-field allocation order is implementation-defined and differs
// between MSVC and GCC/Clang/NVCC, which would make the byte layout
// compiler-dependent. The layout below is part of the ABI and is documented in
// docs/architecture/hl_allocator_abi.md.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hl::alloc {

static_assert(std::endian::native == std::endian::little,
              "HL-KV ABI v0 is defined for little-endian hosts (x86-64, AArch64); "
              "NVIDIA GPUs are little-endian as well.");

/// v0: record layouts below. v1: unchanged record layouts, plus the persistent
/// device-table protocol (descriptor table indexed by block id, per-sequence id
/// arena, deltas) and the rule that a device record with format == kInvalidFormat
/// is not executable. See docs/architecture/hl_allocator_abi.md.
inline constexpr std::uint32_t kAbiVersion = 1;

/// Opaque 4-bit storage-format code. The allocator never interprets it: it
/// only selects which slot pool backs a block. The canonical code assignment
/// is hl::quant::QuantFormat (project2_quant/include/hl_quant/quant_types.hpp).
enum class FormatId : std::uint8_t {};

inline constexpr std::uint32_t kMaxFormats = 15;  // valid codes are 0..14
inline constexpr FormatId kInvalidFormat{15};

/// Largest supported tokens-per-block (the valid_tokens field is 10 bits).
inline constexpr std::uint32_t kMaxBlockTokens = 512;

/// Sentinel for "no slot" (pending_slot when no format change is in flight).
inline constexpr std::uint32_t kNoSlot = 0xFFFF'FFFFu;

/// How the scale metadata of a block is located.
enum class ScaleMode : std::uint8_t {
  /// No descriptor-carried scale: the format is unscaled (e.g. FP16), uses a
  /// per-tensor kernel parameter, or stores its scales inside the slot payload
  /// as defined by the format.
  kNone = 0,
  /// scale_ref holds the IEEE-754 binary32 bits of one per-block scale.
  kInlineF32 = 1,
  /// scale_ref holds two binary16 values: scale in bits 0..15, bias in 16..31.
  kInlineF16Pair = 2,
  /// scale_ref is an index into an externally managed scale pool.
  kExternal = 3,
};

/// Block lifecycle.
enum class BlockState : std::uint8_t {
  kFree = 0,         ///< not allocated
  kOpen = 1,         ///< allocated, tokens may still be appended
  kSealed = 2,       ///< immutable payload; eligible for format change / sharing
  kTranscoding = 3,  ///< sealed + a format change is in flight (both slots live)
};

/// Where the authoritative payload of a block lives.
enum class Residency : std::uint8_t {
  kDevice = 0,   ///< in a device pool slot; usable by kernels
  kHost = 1,     ///< offloaded to host memory (not managed by v0)
  kDropped = 2,  ///< payload discarded; must be recomputed before use
  // 3 reserved
};

namespace bits {

[[nodiscard]] constexpr std::uint32_t mask(std::uint32_t width) noexcept {
  return width >= 32 ? 0xFFFF'FFFFu : ((1u << width) - 1u);
}

[[nodiscard]] constexpr std::uint32_t get(std::uint32_t word, std::uint32_t shift,
                                          std::uint32_t width) noexcept {
  return (word >> shift) & mask(width);
}

[[nodiscard]] constexpr std::uint32_t set(std::uint32_t word, std::uint32_t shift,
                                          std::uint32_t width, std::uint32_t value) noexcept {
  const std::uint32_t m = mask(width) << shift;
  return (word & ~m) | ((value << shift) & m);
}

// BlockExecDescriptor::info
inline constexpr std::uint32_t kFormatShift = 0;
inline constexpr std::uint32_t kFormatWidth = 4;
inline constexpr std::uint32_t kScaleModeShift = 4;
inline constexpr std::uint32_t kScaleModeWidth = 2;
inline constexpr std::uint32_t kLayoutShift = 6;
inline constexpr std::uint32_t kLayoutWidth = 4;
inline constexpr std::uint32_t kValidTokensShift = 10;
inline constexpr std::uint32_t kValidTokensWidth = 10;
inline constexpr std::uint32_t kInfoReservedShift = 20;  // bits 20..31 must be zero

// BlockDescriptor::control
inline constexpr std::uint32_t kStateShift = 0;
inline constexpr std::uint32_t kStateWidth = 2;
inline constexpr std::uint32_t kResidencyShift = 2;
inline constexpr std::uint32_t kResidencyWidth = 2;
inline constexpr std::uint32_t kPendingFormatShift = 4;
inline constexpr std::uint32_t kPendingFormatWidth = 4;
inline constexpr std::uint32_t kRefcountShift = 8;
inline constexpr std::uint32_t kRefcountWidth = 16;
inline constexpr std::uint32_t kControlReservedShift = 24;  // bits 24..31 must be zero

static_assert(kValidTokensWidth >= static_cast<std::uint32_t>(std::bit_width(kMaxBlockTokens)));
static_assert(kMaxFormats < (1u << kFormatWidth));

}  // namespace bits

inline constexpr std::uint32_t kMaxLayoutId = bits::mask(bits::kLayoutWidth);
inline constexpr std::uint32_t kMaxRefcount = bits::mask(bits::kRefcountWidth);

/// Device-facing block record (16 bytes). Layout (little-endian):
///
///   bytes  0..3   slot       physical slot index inside the pool of `format`
///   bytes  4..7   info       format:4 | scale_mode:2 | layout:4 | valid_tokens:10 | 0:12
///   bytes  8..11  scale_ref  meaning selected by scale_mode
///   bytes 12..15  block_id   stable logical identity (index in the host table)
///
/// Device address of the payload = pool_base[format] + slot * slot_bytes[format];
/// pool bases and slot sizes are per-table constants passed as kernel arguments.
struct alignas(16) BlockExecDescriptor {
  std::uint32_t slot;
  std::uint32_t info;
  std::uint32_t scale_ref;
  std::uint32_t block_id;

  [[nodiscard]] static constexpr std::uint32_t pack_info(FormatId format, ScaleMode scale_mode,
                                                         std::uint32_t layout,
                                                         std::uint32_t valid_tokens) noexcept {
    std::uint32_t w = 0;
    w = bits::set(w, bits::kFormatShift, bits::kFormatWidth, static_cast<std::uint32_t>(format));
    w = bits::set(w, bits::kScaleModeShift, bits::kScaleModeWidth,
                  static_cast<std::uint32_t>(scale_mode));
    w = bits::set(w, bits::kLayoutShift, bits::kLayoutWidth, layout);
    w = bits::set(w, bits::kValidTokensShift, bits::kValidTokensWidth, valid_tokens);
    return w;
  }

  [[nodiscard]] constexpr FormatId format() const noexcept {
    return static_cast<FormatId>(bits::get(info, bits::kFormatShift, bits::kFormatWidth));
  }
  [[nodiscard]] constexpr ScaleMode scale_mode() const noexcept {
    return static_cast<ScaleMode>(bits::get(info, bits::kScaleModeShift, bits::kScaleModeWidth));
  }
  [[nodiscard]] constexpr std::uint32_t layout() const noexcept {
    return bits::get(info, bits::kLayoutShift, bits::kLayoutWidth);
  }
  [[nodiscard]] constexpr std::uint32_t valid_tokens() const noexcept {
    return bits::get(info, bits::kValidTokensShift, bits::kValidTokensWidth);
  }
  [[nodiscard]] constexpr std::uint32_t info_reserved() const noexcept {
    return info >> bits::kInfoReservedShift;
  }
  /// Only meaningful when scale_mode() == ScaleMode::kInlineF32.
  [[nodiscard]] constexpr float inline_scale_f32() const noexcept {
    return std::bit_cast<float>(scale_ref);
  }

  constexpr void set_format(FormatId f) noexcept {
    info = bits::set(info, bits::kFormatShift, bits::kFormatWidth, static_cast<std::uint32_t>(f));
  }
  constexpr void set_scale_mode(ScaleMode m) noexcept {
    info = bits::set(info, bits::kScaleModeShift, bits::kScaleModeWidth,
                     static_cast<std::uint32_t>(m));
  }
  constexpr void set_valid_tokens(std::uint32_t n) noexcept {
    info = bits::set(info, bits::kValidTokensShift, bits::kValidTokensWidth, n);
  }

  friend constexpr bool operator==(const BlockExecDescriptor&,
                                   const BlockExecDescriptor&) noexcept = default;
};

/// Host-side block record (32 bytes). Layout (little-endian):
///
///   bytes  0..15  exec          BlockExecDescriptor (copied verbatim to device tables)
///   bytes 16..19  generation    incremented on every free; detects stale handles (ABA)
///   bytes 20..23  control       state:2 | residency:2 | pending_format:4 | refcount:16 | 0:8
///   bytes 24..27  pending_slot  target slot while state == kTranscoding, else kNoSlot
///   bytes 28..31  reserved      must be zero (ABI v1 extension space)
struct alignas(32) BlockDescriptor {
  BlockExecDescriptor exec;
  std::uint32_t generation;
  std::uint32_t control;
  std::uint32_t pending_slot;
  std::uint32_t reserved;

  [[nodiscard]] constexpr BlockState state() const noexcept {
    return static_cast<BlockState>(bits::get(control, bits::kStateShift, bits::kStateWidth));
  }
  [[nodiscard]] constexpr Residency residency() const noexcept {
    return static_cast<Residency>(
        bits::get(control, bits::kResidencyShift, bits::kResidencyWidth));
  }
  [[nodiscard]] constexpr FormatId pending_format() const noexcept {
    return static_cast<FormatId>(
        bits::get(control, bits::kPendingFormatShift, bits::kPendingFormatWidth));
  }
  [[nodiscard]] constexpr std::uint32_t refcount() const noexcept {
    return bits::get(control, bits::kRefcountShift, bits::kRefcountWidth);
  }
  [[nodiscard]] constexpr std::uint32_t control_reserved() const noexcept {
    return control >> bits::kControlReservedShift;
  }

  constexpr void set_state(BlockState s) noexcept {
    control = bits::set(control, bits::kStateShift, bits::kStateWidth,
                        static_cast<std::uint32_t>(s));
  }
  constexpr void set_residency(Residency r) noexcept {
    control = bits::set(control, bits::kResidencyShift, bits::kResidencyWidth,
                        static_cast<std::uint32_t>(r));
  }
  constexpr void set_pending_format(FormatId f) noexcept {
    control = bits::set(control, bits::kPendingFormatShift, bits::kPendingFormatWidth,
                        static_cast<std::uint32_t>(f));
  }
  constexpr void set_refcount(std::uint32_t n) noexcept {
    control = bits::set(control, bits::kRefcountShift, bits::kRefcountWidth, n);
  }

  friend constexpr bool operator==(const BlockDescriptor&,
                                   const BlockDescriptor&) noexcept = default;
};

// ---- ABI layout guarantees -------------------------------------------------

static_assert(sizeof(BlockExecDescriptor) == 16);
static_assert(alignof(BlockExecDescriptor) == 16);
static_assert(std::is_trivially_copyable_v<BlockExecDescriptor>);
static_assert(std::is_standard_layout_v<BlockExecDescriptor>);
static_assert(offsetof(BlockExecDescriptor, slot) == 0);
static_assert(offsetof(BlockExecDescriptor, info) == 4);
static_assert(offsetof(BlockExecDescriptor, scale_ref) == 8);
static_assert(offsetof(BlockExecDescriptor, block_id) == 12);

static_assert(sizeof(BlockDescriptor) == 32);
static_assert(alignof(BlockDescriptor) == 32);
static_assert(64 % alignof(BlockDescriptor) == 0, "a descriptor must never straddle a cache line");
static_assert(std::is_trivially_copyable_v<BlockDescriptor>);
static_assert(std::is_standard_layout_v<BlockDescriptor>);
static_assert(offsetof(BlockDescriptor, exec) == 0);
static_assert(offsetof(BlockDescriptor, generation) == 16);
static_assert(offsetof(BlockDescriptor, control) == 20);
static_assert(offsetof(BlockDescriptor, pending_slot) == 24);
static_assert(offsetof(BlockDescriptor, reserved) == 28);

}  // namespace hl::alloc
