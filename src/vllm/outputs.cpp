// Ported from: vllm/outputs.py @ e24d1b24
// (FinishReason string mapping from vllm/v1/engine/__init__.py @ e24d1b24:
// FINISH_REASON_STRINGS). See include/vllm/outputs.h for the deferred-field /
// deviation notes.
#include "vllm/outputs.h"

#include <string>

#include "vllm/v1/request.h"

namespace vllm {

// str(FinishReason) == FINISH_REASON_STRINGS[value], with
// FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition").
// These strings are part of the external API (RequestOutput.finish_reason).
std::string FinishReasonToString(vllm::v1::FinishReason reason) {
  switch (reason) {
    case vllm::v1::FinishReason::kStop:
      return "stop";
    case vllm::v1::FinishReason::kLength:
      return "length";
    case vllm::v1::FinishReason::kAbort:
      return "abort";
    case vllm::v1::FinishReason::kError:
      return "error";
    case vllm::v1::FinishReason::kRepetition:
      return "repetition";
  }
  // Unreachable for the closed enum; keeps the compiler happy under -Werror.
  return "stop";
}

}  // namespace vllm
