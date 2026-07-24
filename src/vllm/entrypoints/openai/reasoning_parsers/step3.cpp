// Ported from: vllm/reasoning/step3_reasoning_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/reasoning_parsers/step3.h"

#include <cstddef>
#include <string>

namespace vllm::entrypoints::openai {

namespace {
bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}
}  // namespace

ExtractedReasoning Step3ReasoningParser::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  ExtractedReasoning out;
  const std::size_t ep = model_output.find(end_);
  if (ep == std::string::npos) {
    // No </think>: everything is reasoning (step3:97-99).
    out.reasoning = model_output;
    out.content = std::nullopt;
    return out;
  }
  out.reasoning = model_output.substr(0, ep);
  std::string content = model_output.substr(ep + end_.size());
  out.content = content.empty() ? std::nullopt : std::optional<std::string>(content);
  return out;
}

std::optional<DeltaMessage> Step3ReasoningParser::extract_reasoning_streaming(
    const std::string& previous_text, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  // Skip a lone </think> delta (step3:73-75).
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

bool Step3ReasoningParser::is_reasoning_end(const std::string& text) const {
  return Contains(text, end_);
}

}  // namespace vllm::entrypoints::openai
