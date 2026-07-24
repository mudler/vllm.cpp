// Ported from: vllm/reasoning/basic_parsers.py @ e24d1b24
#include "vllm/entrypoints/openai/reasoning_parsers/basic.h"

#include <cstddef>
#include <string>

namespace vllm::entrypoints::openai {

namespace {
bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// Build a DeltaMessage carrying a reasoning and/or content span. An empty
// content span collapses to nullopt (upstream `content if content else None`).
DeltaMessage MakeReasoningDelta(std::optional<std::string> reasoning,
                                std::optional<std::string> content) {
  DeltaMessage msg;
  msg.reasoning = std::move(reasoning);
  if (content.has_value() && !content->empty()) {
    msg.content = std::move(content);
  }
  return msg;
}
}  // namespace

ExtractedReasoning BaseThinkingReasoningParser::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  const std::string& st = start_token();
  const std::string& et = end_token();

  // partition(start_token): drop everything up to and including the first start
  // marker when present (basic_parsers.py:168-171).
  std::string body = model_output;
  const std::size_t sp = body.find(st);
  if (sp != std::string::npos) {
    body = body.substr(sp + st.size());
  }

  ExtractedReasoning out;
  const std::size_t ep = body.find(et);
  if (ep == std::string::npos) {
    // No end marker: the whole (post-start) body is reasoning (basic:175-176).
    out.reasoning = body;
    out.content = std::nullopt;
    return out;
  }
  out.reasoning = body.substr(0, ep);
  std::string content = body.substr(ep + et.size());
  out.content = content.empty() ? std::nullopt : std::optional<std::string>(content);
  return out;
}

std::optional<DeltaMessage>
BaseThinkingReasoningParser::extract_reasoning_streaming(
    const std::string& previous_text, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  const std::string& st = start_token();
  const std::string& et = end_token();

  // Skip a lone special-token delta (basic_parsers.py:113-116).
  if (delta_text == st || delta_text == et) {
    return std::nullopt;
  }

  if (Contains(previous_text, st)) {
    if (Contains(delta_text, et)) {
      const std::size_t end_index = delta_text.find(et);
      return MakeReasoningDelta(delta_text.substr(0, end_index),
                                delta_text.substr(end_index + et.size()));
    }
    if (Contains(previous_text, et)) {
      DeltaMessage msg;
      msg.content = delta_text;
      return msg;
    }
    return MakeReasoningDelta(delta_text, std::nullopt);
  }

  if (Contains(delta_text, st)) {
    if (Contains(delta_text, et)) {
      const std::size_t start_index = delta_text.find(st);
      const std::size_t end_index = delta_text.find(et);
      return MakeReasoningDelta(
          delta_text.substr(start_index + st.size(),
                            end_index - (start_index + st.size())),
          delta_text.substr(end_index + et.size()));
    }
    return MakeReasoningDelta(delta_text, std::nullopt);
  }

  // No start marker seen: plain content (basic_parsers.py:153-155).
  DeltaMessage msg;
  msg.content = delta_text;
  return msg;
}

bool BaseThinkingReasoningParser::is_reasoning_end(const std::string& text) const {
  const std::size_t pe = text.rfind(end_token());
  if (pe == std::string::npos) return false;
  const std::size_t ps = text.rfind(start_token());
  if (ps == std::string::npos) return true;
  return pe > ps;
}

}  // namespace vllm::entrypoints::openai
