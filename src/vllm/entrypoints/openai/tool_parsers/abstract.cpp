// Ported from: vllm/tool_parsers/abstract_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

#include "vllm/entrypoints/openai/tool_parsers/deepseek_v3.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v31.h"
#include "vllm/entrypoints/openai/tool_parsers/granite.h"
#include "vllm/entrypoints/openai/tool_parsers/granite4.h"
#include "vllm/entrypoints/openai/tool_parsers/granite_20b_fc.h"
#include "vllm/entrypoints/openai/tool_parsers/hermes.h"
#include "vllm/entrypoints/openai/tool_parsers/longcat.h"
#include "vllm/entrypoints/openai/tool_parsers/mistral.h"
#include "vllm/entrypoints/openai/tool_parsers/llama.h"
#include "vllm/entrypoints/openai/tool_parsers/llama4_pythonic.h"
#include "vllm/entrypoints/openai/tool_parsers/pythonic.h"
#include "vllm/entrypoints/openai/tool_parsers/qwen3.h"

namespace vllm::entrypoints::openai {

// abstract_tool_parser.py:218 — the base raises NotImplementedError; derived
// streaming-capable parsers (Hermes) override this.
std::optional<DeltaMessage> ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& /*current_text*/,
    const std::string& /*delta_text*/, const ChatCompletionRequest& /*request*/) {
  throw std::runtime_error(
      "ToolParser::extract_tool_calls_streaming has not been implemented!");
}

std::string make_tool_call_id() {
  // Upstream: f"chatcmpl-tool-{random_uuid()}" with random_uuid() == the hex of
  // a uuid4 (32 hex chars). We emit 32 random hex chars (uniqueness only).
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> dist;
  const uint64_t hi = dist(rng);
  const uint64_t lo = dist(rng);
  std::ostringstream oss;
  oss << "chatcmpl-tool-" << std::hex << std::setfill('0') << std::setw(16)
      << hi << std::setw(16) << lo;
  return oss.str();
}

std::unique_ptr<ToolParser> get_tool_parser(const std::string& name) {
  // Ported from ToolParserManager.get_tool_parser (abstract_tool_parser.py:235).
  // Registry: "hermes" -> Hermes2ProToolParser; "qwen3" -> the gate model's
  // Hermes-format parser; "deepseek_v3"/"deepseek_v31" -> the DeepSeek marker
  // formats; "longcat" -> the Hermes subclass with the <longcat_tool_call> tag.
  // Unknown name -> nullptr (upstream raises KeyError).
  //
  // DETECTION MARKERS (for any auto-detect layer built on top of this factory;
  // upstream vLLM selects by explicit --tool-call-parser NAME, so ordering does
  // not affect this factory itself):
  //   - DeepSeek: the OUTER begin marker <｜tool▁calls▁begin｜> (fullwidth ｜ +
  //     U+2581) is unique and unambiguous — probe it directly.
  //   - longcat: probe <longcat_tool_call>. VERIFIED against the upstream chat
  //     template (huggingface.co/meituan-longcat/LongCat-Flash-Chat
  //     tokenizer_config.json): LongCat wraps tool calls EXCLUSIVELY in
  //     <longcat_tool_call>…</longcat_tool_call> and never emits the bare Hermes
  //     <tool_call>. Since "<tool_call>" is NOT a substring of
  //     "<longcat_tool_call>", a hermes-first check would NOT false-positive on
  //     standard longcat output; still, a loose/substring auto-detector should
  //     prefer the more specific <longcat_tool_call> marker before <tool_call>.
  // Hermes-format parser; "llama3_json"/"llama4_json" -> LlamaToolParser;
  // "pythonic" -> PythonicToolParser; "llama4_pythonic" ->
  // Llama4PythonicToolParser. Unknown name -> nullptr (upstream raises KeyError).
  if (name == "hermes") {
    return std::make_unique<HermesToolParser>();
  }
  if (name == "qwen3") {
    return std::make_unique<Qwen3ToolParser>();
  }
  if (name == "deepseek_v3") {
    return std::make_unique<DeepSeekV3ToolParser>();
  }
  if (name == "deepseek_v31") {
    return std::make_unique<DeepSeekV31ToolParser>();
  }
  if (name == "longcat") {
    return std::make_unique<LongcatToolParser>();
  }
  // "mistral" -> the modern v11 name-first form (mistral_tool_parser.py:96).
  // The pre-v11 bracketed-array form is selected via MistralToolParser(true).
  if (name == "mistral") {
    return std::make_unique<MistralToolParser>();
  }
  // Granite family (granite_tool_parser.py / granite4_tool_parser.py /
  // granite_20b_fc_tool_parser.py @ e24d1b24).
  if (name == "granite") {
    return std::make_unique<GraniteToolParser>();
  }
  if (name == "granite4") {
    return std::make_unique<Granite4ToolParser>();
  }
  if (name == "granite-20b-fc") {
    return std::make_unique<Granite20bFCToolParser>();
  }
  // llama_tool_parser.py registers Llama3JsonToolParser under BOTH names
  // (llama3_json + llama4_json); they share one implementation.
  if (name == "llama3_json" || name == "llama4_json") {
    return std::make_unique<LlamaToolParser>();
  }
  if (name == "pythonic") {
    return std::make_unique<PythonicToolParser>();
  }
  if (name == "llama4_pythonic") {
    return std::make_unique<Llama4PythonicToolParser>();
  }
  return nullptr;
}

}  // namespace vllm::entrypoints::openai
