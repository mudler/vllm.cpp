// Ported from: vllm/reasoning/abs_reasoning_parsers.py @ e24d1b24
// (ReasoningParserManager.get_reasoning_parser). Hand-wired factory over the T0
// reasoning-parser formats, mirroring tool_parsers/abstract.cpp get_tool_parser.
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

#include <memory>
#include <string>

#include "vllm/entrypoints/openai/reasoning_parsers/deepseek_r1.h"
#include "vllm/entrypoints/openai/reasoning_parsers/think_auto.h"
#include "vllm/entrypoints/openai/reasoning_parsers/minimax_m2.h"
#include "vllm/entrypoints/openai/reasoning_parsers/mistral.h"
#include "vllm/entrypoints/openai/reasoning_parsers/olmo3.h"
#include "vllm/entrypoints/openai/reasoning_parsers/step3.h"

namespace vllm::entrypoints::openai {

std::unique_ptr<ReasoningParser> get_reasoning_parser(const std::string& name) {
  if (name == "think_auto") {
    return std::make_unique<ThinkAutoReasoningParser>();
  }
  if (name == "deepseek_r1") {
    return std::make_unique<DeepSeekR1ReasoningParser>();
  }
  if (name == "mistral") {
    return std::make_unique<MistralReasoningParser>();
  }
  if (name == "minimax_m2") {
    return std::make_unique<MiniMaxM2ReasoningParser>();
  }
  if (name == "minimax_m2_append_think") {
    return std::make_unique<MiniMaxM2AppendThinkReasoningParser>();
  }
  if (name == "step3") {
    return std::make_unique<Step3ReasoningParser>();
  }
  if (name == "olmo3") {
    return std::make_unique<Olmo3ReasoningParser>();
  }
  return nullptr;
}

}  // namespace vllm::entrypoints::openai
