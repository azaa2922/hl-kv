#pragma once

// Error reporting for hl_allocator.
//
// Expected, recoverable conditions (pool exhausted, stale handle, wrong block
// state) are returned as Status codes — they occur on the scheduler hot path
// and must not throw. Programming errors (invalid table configuration,
// accessing the value of a failed Result) throw.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace hl::alloc {

enum class Status : std::uint8_t {
  kOk = 0,
  kOutOfBlocks,        ///< no free block id
  kOutOfSlots,         ///< the pool for the requested format has no free slot
  kInvalidHandle,      ///< id out of range / never valid
  kStaleHandle,        ///< handle generation does not match (block was freed/reused)
  kInvalidState,       ///< operation not allowed in the block's current state
  kInvalidArgument,    ///< argument outside documented bounds
  kUnknownFormat,      ///< format has no configured pool
  kNotDeviceResident,  ///< block cannot be handed to a kernel (residency != kDevice)
  kRefcountOverflow,   ///< refcount would exceed kMaxRefcount
  kCorrupt,            ///< validate() found an inconsistency
};

[[nodiscard]] const char* to_string(Status status) noexcept;

/// Thrown when value() is called on a Result that holds an error.
class BadResultAccess : public std::logic_error {
 public:
  explicit BadResultAccess(Status status);
  [[nodiscard]] Status status() const noexcept { return status_; }

 private:
  Status status_;
};

/// Minimal value-or-status type (std::expected is C++23).
template <typename T>
class [[nodiscard]] Result {
 public:
  Result(const T& value) : value_(value), status_(Status::kOk) {}  // NOLINT(google-explicit-constructor)
  Result(T&& value) : value_(std::move(value)), status_(Status::kOk) {}  // NOLINT(google-explicit-constructor)
  /// `status` must not be Status::kOk (an OK result must carry a value).
  Result(Status status) : status_(status == Status::kOk ? Status::kInvalidArgument : status) {}  // NOLINT(google-explicit-constructor)

  [[nodiscard]] bool ok() const noexcept { return status_ == Status::kOk; }
  [[nodiscard]] Status status() const noexcept { return status_; }

  [[nodiscard]] const T& value() const& {
    if (!ok()) throw BadResultAccess(status_);
    return *value_;
  }
  [[nodiscard]] T& value() & {
    if (!ok()) throw BadResultAccess(status_);
    return *value_;
  }
  [[nodiscard]] T&& value() && {
    if (!ok()) throw BadResultAccess(status_);
    return std::move(*value_);
  }

 private:
  std::optional<T> value_;
  Status status_;
};

}  // namespace hl::alloc
