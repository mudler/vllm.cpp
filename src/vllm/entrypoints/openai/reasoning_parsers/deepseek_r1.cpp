// Ported from: vllm/reasoning/deepseek_r1_reasoning_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/reasoning_parsers/deepseek_r1.h"

#include <cstddef>
#include <string>

namespace vllm::entrypoints::openai {

namespace {
bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}
}  // namespace

std::optional<DeltaMessage> DeepSeekR1ReasoningParser::extract_reasoning_streaming(
    const std::string& previous_text, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  std::optional<DeltaMessage> ret = BaseThinkingReasoningParser::
      extract_reasoning_streaming(previous_text, current_text, delta_text,
                                  request);

  // deepseek_r1_reasoning_parser.py:45 - when the start marker never appears
  // (neither in the accumulated prefix nor this delta), the base would have
  // classified the delta as content; DeepSeek instead treats a start-less
  // prefix as reasoning until the first end marker.
  if (ret.has_value() && !Contains(previous_text, start_token()) &&
      !Contains(delta_text, start_token())) {
    if (Contains(delta_text, end_token())) {
      const std::size_t end_index = delta_text.find(end_token());
      DeltaMessage msg;
      msg.reasoning = delta_text.substr(0, end_index);
      std::string content = delta_text.substr(end_index + end_token().size());
      if (!content.empty()) msg.content = content;
      return msg;
    }
    if (Contains(previous_text, end_token())) {
      DeltaMessage msg;
      msg.content = delta_text;
      return msg;
    }
    DeltaMessage msg;
    msg.reasoning = delta_text;
    return msg;
  }

  return ret;
}

}  // namespace vllm::entrypoints::openai
