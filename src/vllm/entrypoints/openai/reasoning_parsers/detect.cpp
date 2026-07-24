// See detect.h. ORIGINAL packaging-layer component (no upstream mirror).
#include "vllm/entrypoints/openai/reasoning_parsers/detect.h"

namespace vllm::entrypoints::openai {

namespace {

// ORDER MATTERS: first match wins, more specific first. "[THINK]" is the
// Mistral special-token literal and cannot appear in a <think>-family
// template; "<think>" is the broad Qwen/DeepSeek-family convention and sits
// last. Families whose markers are not template-stable (minimax_m2's
// end-token-only </think>, olmo3's plain-vocab <think>) stay EXPLICIT-ONLY:
// a template-level probe cannot distinguish them from deepseek_r1, and
// deepseek_r1's split behavior is the correct default for a plain
// <think>...</think> stream.
constexpr ReasoningParserMarker kReasoningParserMarkers[] = {
    {"mistral", "[THINK]"},
    {"deepseek_r1", "<think>"},
};

constexpr std::size_t kReasoningParserMarkerCount =
    sizeof(kReasoningParserMarkers) / sizeof(kReasoningParserMarkers[0]);

}  // namespace

std::string DetectReasoningParser(const std::string& chat_template,
                                  const ReasoningParserMarker* table,
                                  std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (table[i].template_marker != nullptr &&
        chat_template.find(table[i].template_marker) != std::string::npos) {
      return table[i].parser;
    }
  }
  return std::string();
}

std::string DetectReasoningParser(const std::string& chat_template) {
  return DetectReasoningParser(chat_template, kReasoningParserMarkers,
                               kReasoningParserMarkerCount);
}

const ReasoningParserMarker* ReasoningParserMarkerTable(std::size_t* out_count) {
  if (out_count != nullptr) *out_count = kReasoningParserMarkerCount;
  return kReasoningParserMarkers;
}

}  // namespace vllm::entrypoints::openai
