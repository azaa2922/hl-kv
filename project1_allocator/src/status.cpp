#include "hl_allocator/status.hpp"

#include <string>

namespace hl::alloc {

const char* to_string(Status status) noexcept {
  switch (status) {
    case Status::kOk: return "ok";
    case Status::kOutOfBlocks: return "out of blocks";
    case Status::kOutOfSlots: return "out of slots";
    case Status::kInvalidHandle: return "invalid handle";
    case Status::kStaleHandle: return "stale handle";
    case Status::kInvalidState: return "invalid state";
    case Status::kInvalidArgument: return "invalid argument";
    case Status::kUnknownFormat: return "unknown format";
    case Status::kNotDeviceResident: return "not device resident";
    case Status::kRefcountOverflow: return "refcount overflow";
    case Status::kCorrupt: return "corrupt";
    case Status::kCapacityExceeded: return "capacity exceeded";
  }
  return "unknown status";
}

BadResultAccess::BadResultAccess(Status status)
    : std::logic_error(std::string("hl_allocator: value() on failed Result: ") + to_string(status)),
      status_(status) {}

}  // namespace hl::alloc
