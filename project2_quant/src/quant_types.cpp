#include "hl_quant/quant_types.hpp"

namespace hl::quant {

const char* to_string(QuantStatus status) noexcept {
  switch (status) {
    case QuantStatus::kOk: return "ok";
    case QuantStatus::kInvalidArgument: return "invalid argument";
    case QuantStatus::kBufferTooSmall: return "buffer too small";
    case QuantStatus::kUnsupportedFormat: return "unsupported format";
    case QuantStatus::kCudaError: return "CUDA error";
  }
  return "unknown status";
}

}  // namespace hl::quant
