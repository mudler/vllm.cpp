// Ported from: vllm/reasoning/mistral_reasoning_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/reasoning_parsers/mistral.h"

#include <cstddef>
#include <string>

namespace vllm::entrypoints::openai {

namespace {
std::optional<std::string> OrNullopt(std::string s) {
  if (s.empty()) return std::nullopt;
  return s;
}
}  // namespace

ExtractedReasoning MistralReasoningParser::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  ExtractedReasoning out;
  if (model_output.empty()) {
    // mistral_reasoning_parser.py:126-127.
    out.reasoning = std::nullopt;
    out.content = "";
    return out;
  }

  const std::string& bot = start_;
  const std::string& eot = end_;

  // partition(start_token): prev_bot | [THINK] | post_bot.
  const std::size_t bot_pos = model_output.find(bot);
  const bool has_bot = bot_pos != std::string::npos;
  const std::string prev_bot =
      has_bot ? model_output.substr(0, bot_pos) : model_output;
  const std::string post_bot =
      has_bot ? model_output.substr(bot_pos + bot.size()) : std::string();

  const bool has_valid_eot =
      has_bot && post_bot.find(eot) != std::string::npos;

  if (has_bot && has_valid_eot) {
    // 1. [THINK] ... [/THINK] (mistral:140-144).
    const std::size_t eot_pos = post_bot.find(eot);
    const std::string prev_eot = post_bot.substr(0, eot_pos);
    const std::string post_eot = post_bot.substr(eot_pos + eot.size());
    const std::string content = prev_bot + post_eot;
    out.reasoning = prev_eot;
    out.content = OrNullopt(content);
    return out;
  }
  if (has_bot) {
    // 2. Only [THINK] (mistral:145-148).
    out.reasoning = post_bot;
    out.content = OrNullopt(prev_bot);
    return out;
  }
  // 3. No [THINK] (mistral:149-162).
  const std::size_t stray_eot = prev_bot.find(eot);
  if (stray_eot != std::string::npos) {
    // 3.a stray [/THINK] with no [THINK]: strip it, all content.
    const std::string prev_eot = prev_bot.substr(0, stray_eot);
    const std::string post_eot = prev_bot.substr(stray_eot + eot.size());
    out.reasoning = std::nullopt;
    out.content = prev_eot + post_eot;
    return out;
  }
  // 3.b Neither marker: all content.
  out.reasoning = std::nullopt;
  out.content = prev_bot;
  return out;
}

bool MistralReasoningParser::is_reasoning_end(const std::string& text) const {
  const std::size_t ps = text.rfind(start_);
  if (ps == std::string::npos) return false;
  return text.find(end_, ps + start_.size()) != std::string::npos;
}

}  // namespace vllm::entrypoints::openai
