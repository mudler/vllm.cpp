// See think_auto.h. ORIGINAL packaging-layer component (no upstream mirror).
#include "vllm/entrypoints/openai/reasoning_parsers/think_auto.h"

namespace vllm::entrypoints::openai {

namespace {
const std::string kStart = "<think>";
const std::string kEnd = "</think>";

// Leading-whitespace-tolerant "does the output open a think block" probe.
std::size_t SkipWs(const std::string& s) {
  std::size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' ||
                          s[i] == '\r')) {
    ++i;
  }
  return i;
}
}  // namespace

const std::string& ThinkAutoReasoningParser::start_token() const {
  return kStart;
}
const std::string& ThinkAutoReasoningParser::end_token() const { return kEnd; }

ExtractedReasoning ThinkAutoReasoningParser::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& request) {
  // Symmetric rule: NO marker at all => the whole output is CONTENT (the
  // hybrid-thinking "model chose not to think" case R1 semantics get wrong).
  if (model_output.find(kStart) == std::string::npos &&
      model_output.find(kEnd) == std::string::npos) {
    ExtractedReasoning out;
    out.content = model_output;
    return out;
  }
  // Any marker present: the base (R1-identical) split is correct, including
  // the pre-filled-<think> end-only case.
  return BaseThinkingReasoningParser::extract_reasoning(model_output, request);
}

std::optional<DeltaMessage> ThinkAutoReasoningParser::extract_reasoning_streaming(
    const std::string& previous_text, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  if (mode_ == Mode::kThink) {
    return BaseThinkingReasoningParser::extract_reasoning_streaming(
        previous_text, current_text, delta_text, request);
  }
  if (mode_ == Mode::kContent) {
    DeltaMessage out;
    out.content = delta_text;
    return out;
  }
  // Undecided: a think block, when present, opens the turn (Qwen/DeepSeek
  // convention). Withhold until the head is decidable, then commit the mode
  // and FLUSH everything withheld so far in one piece.
  const std::size_t head = SkipWs(current_text);
  const std::string head_text = current_text.substr(head);
  if (head_text.size() < kStart.size() &&
      kStart.compare(0, head_text.size(), head_text) == 0) {
    return std::nullopt;  // still ambiguous: could become "<think>".
  }
  if (head_text.rfind(kStart, 0) == 0) {
    mode_ = Mode::kThink;
    // Prime the base parser on ITS contract (upstream basic_parsers.py
    // cadence: a marker arrives as a lone delta): feed the start marker as a
    // lone skipped delta, then the withheld remainder as the second delta.
    const std::string after_marker = head_text.substr(kStart.size());
    (void)BaseThinkingReasoningParser::extract_reasoning_streaming(
        std::string(), kStart, kStart, request);
    if (after_marker.empty()) return std::nullopt;
    return BaseThinkingReasoningParser::extract_reasoning_streaming(
        kStart, current_text, after_marker, request);
  }
  mode_ = Mode::kContent;
  DeltaMessage out;
  out.content = current_text;  // flush the withheld head with this delta.
  return out;
}

}  // namespace vllm::entrypoints::openai
