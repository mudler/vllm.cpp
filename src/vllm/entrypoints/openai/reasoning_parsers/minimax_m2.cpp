// Ported from: vllm/reasoning/minimax_m2_reasoning_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/reasoning_parsers/minimax_m2.h"

#include <cstddef>
#include <string>

namespace vllm::entrypoints::openai {

namespace {
bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// "last special marker is the END marker" (minimax_m2:39-45): scanning from the
// end, the first of {start, end} encountered decides. In text form that is the
// marker with the larger rfind position.
bool EndIsLastMarker(const std::string& text, const std::string& start,
                     const std::string& end) {
  const std::size_t pe = text.rfind(end);
  if (pe == std::string::npos) return false;
  const std::size_t ps = text.rfind(start);
  if (ps == std::string::npos) return true;
  return pe > ps;
}
}  // namespace

// ── minimax_m2 (end-marker-only) ──────────────────────────────────────────────
ExtractedReasoning MiniMaxM2ReasoningParser::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  ExtractedReasoning out;
  const std::size_t ep = model_output.find(end_);
  if (ep == std::string::npos) {
    out.reasoning = model_output;
    out.content = std::nullopt;
    return out;
  }
  out.reasoning = model_output.substr(0, ep);
  std::string content = model_output.substr(ep + end_.size());
  out.content = content.empty() ? std::nullopt : std::optional<std::string>(content);
  return out;
}

std::optional<DeltaMessage> MiniMaxM2ReasoningParser::extract_reasoning_streaming(
    const std::string& previous_text, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  if (delta_text == end_) {
    return std::nullopt;
  }
  if (Contains(delta_text, end_)) {
    const std::size_t end_index = delta_text.find(end_);
    DeltaMessage msg;
    msg.reasoning = delta_text.substr(0, end_index);
    std::string content = delta_text.substr(end_index + end_.size());
    if (!content.empty()) msg.content = content;
    return msg;
  }
  if (Contains(previous_text, end_)) {
    DeltaMessage msg;
    msg.content = delta_text;
    return msg;
  }
  DeltaMessage msg;
  msg.reasoning = delta_text;
  return msg;
}

bool MiniMaxM2ReasoningParser::is_reasoning_end(const std::string& text) const {
  return EndIsLastMarker(text, start_, end_);
}

// ── minimax_m2_append_think (pass-through, prepend <think>) ───────────────────
ExtractedReasoning MiniMaxM2AppendThinkReasoningParser::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  ExtractedReasoning out;
  out.reasoning = std::nullopt;
  out.content = start_ + model_output;
  return out;
}

std::optional<DeltaMessage>
MiniMaxM2AppendThinkReasoningParser::extract_reasoning_streaming(
    const std::string& previous_text, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  DeltaMessage msg;
  // First delta of the stream: restore the opening marker (minimax_m2:59-60).
  msg.content = previous_text.empty() ? (start_ + delta_text) : delta_text;
  return msg;
}

bool MiniMaxM2AppendThinkReasoningParser::is_reasoning_end(
    const std::string& text) const {
  return EndIsLastMarker(text, start_, end_);
}

}  // namespace vllm::entrypoints::openai
