// Ported from: vllm/tool_parsers/abstract_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

#include "vllm/entrypoints/openai/tool_parsers/deepseek_v3.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v31.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v32.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v4.h"
#include "vllm/entrypoints/openai/tool_parsers/functiongemma.h"
#include "vllm/entrypoints/openai/tool_parsers/granite.h"
#include "vllm/entrypoints/openai/tool_parsers/granite4.h"
#include "vllm/entrypoints/openai/tool_parsers/granite_20b_fc.h"
#include "vllm/entrypoints/openai/tool_parsers/apertus.h"
#include "vllm/entrypoints/openai/tool_parsers/ernie45.h"
#include "vllm/entrypoints/openai/tool_parsers/gigachat3.h"
#include "vllm/entrypoints/openai/tool_parsers/hermes.h"
#include "vllm/entrypoints/openai/tool_parsers/hunyuan_a13b.h"
#include "vllm/entrypoints/openai/tool_parsers/internlm.h"
#include "vllm/entrypoints/openai/tool_parsers/jamba.h"
#include "vllm/entrypoints/openai/tool_parsers/hy_v3.h"
#include "vllm/entrypoints/openai/tool_parsers/lfm2.h"
#include "vllm/entrypoints/openai/tool_parsers/longcat.h"
#include "vllm/entrypoints/openai/tool_parsers/minicpm5.h"
#include "vllm/entrypoints/openai/tool_parsers/mistral.h"
#include "vllm/entrypoints/openai/tool_parsers/phi4_mini.h"
#include "vllm/entrypoints/openai/tool_parsers/poolside_v1.h"
#include "vllm/entrypoints/openai/tool_parsers/xlam.h"
#include "vllm/entrypoints/openai/tool_parsers/llama.h"
#include "vllm/entrypoints/openai/tool_parsers/llama4_pythonic.h"
#include "vllm/entrypoints/openai/tool_parsers/olmo3.h"
#include "vllm/entrypoints/openai/tool_parsers/pythonic.h"
#include "vllm/entrypoints/openai/tool_parsers/qwen3.h"
#include "vllm/entrypoints/openai/tool_parsers/step3.h"
#include "vllm/entrypoints/openai/tool_parsers/step3p5.h"

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
  // deepseek_v32 (DSML <｜DSML｜function_calls>…) + deepseek_v4 (the same DSML
  // grammar wrapped in <｜DSML｜tool_calls>). deepseek_v4 subclasses v32.
  if (name == "deepseek_v32") {
    return std::make_unique<DeepSeekV32ToolParser>();
  }
  if (name == "deepseek_v4") {
    return std::make_unique<DeepSeekV4ToolParser>();
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
  // xlam_tool_parser.py (name "xlam") - fenced / [TOOL_CALLS] / <tool_call> /
  // bare-list envelopes around a JSON tool-call array.
  if (name == "xlam") {
    return std::make_unique<xLAMToolParser>();
  }
  // phi4mini_tool_parser.py (name "phi4_mini_json") - `functools[...]`.
  if (name == "phi4_mini_json") {
    return std::make_unique<Phi4MiniJsonToolParser>();
  }
  // internlm2_tool_parser.py (name "internlm") - <|action_start|><|plugin|>{}<|action_end|>.
  if (name == "internlm") {
    return std::make_unique<Internlm2ToolParser>();
  }
  // jamba_tool_parser.py (name "jamba") - <tool_calls>[...]</tool_calls>.
  if (name == "jamba") {
    return std::make_unique<JambaToolParser>();
  }
  // StepFun step3 (opaque fullwidth-｜ markers) + step3.5 (XML-parametrized
  // <tool_call><function=…><parameter=…> streaming state machine).
  if (name == "step3") {
    return std::make_unique<Step3ToolParser>();
  }
  if (name == "step3p5") {
    return std::make_unique<Step3p5ToolParser>();
  }
  // olmo3_tool_parser.py: the pythonic-inside-<function_calls> wrapper format.
  if (name == "olmo3") {
    return std::make_unique<Olmo3ToolParser>();
  }
  // minicpm5xml_tool_parser.py: the <function name="x"><param ...> XML dialect.
  if (name == "minicpm5") {
    return std::make_unique<MiniCPM5ToolParser>();
  }
  // hy_v3_tool_parser.py: the suffix-parametrized <tool_calls..><tool_call..>
  // <tool_sep..><arg_key..><arg_value..> XML dialect (default empty suffix).
  if (name == "hy_v3") {
    return std::make_unique<HYV3ToolParser>();
  }
  // hunyuan_a13b_tool_parser.py: <tool_calls>[{"name":..,"arguments":..}]
  // </tool_calls> (a JSON array; <think>-guarded; bot_string + consume_space).
  if (name == "hunyuan_a13b") {
    return std::make_unique<HunyuanA13BToolParser>();
  }
  // apertus_tool_parser.py: <|tools_prefix|>[{name:{args}}]<|tools_suffix|>.
  if (name == "apertus") {
    return std::make_unique<ApertusToolParser>();
  }
  // ernie45_tool_parser.py: <tool_call>{json}</tool_call> (thinking/response
  // interplay; upstream vocab-aware, reworked to text - see ernie45.h).
  if (name == "ernie45") {
    return std::make_unique<Ernie45ToolParser>();
  }
  // gigachat3_tool_parser.py: "function call<|role_sep|>\n{json}" or
  // "<|function_call|>{json}"; generation may end with "</s>".
  if (name == "gigachat3") {
    return std::make_unique<GigaChat3ToolParser>();
  // lfm2_tool_parser.py (name "lfm2") - pythonic `[func(arg=val), ...]` list
  // wrapped in <|tool_call_start|>...<|tool_call_end|> sentinels.
  if (name == "lfm2") {
    return std::make_unique<Lfm2ToolParser>();
  }
  // poolside_v1_tool_parser.py (name "poolside_v1") - GLM-4-derived bare
  // <tool_call> with a NAME + <arg_key>/<arg_value> XML body.
  if (name == "poolside_v1") {
    return std::make_unique<PoolsideV1ToolParser>();
  }
  // functiongemma_tool_parser.py (name "functiongemma") - Google FunctionGemma
  // <start_function_call>call:NAME{KEY:<escape>VALUE<escape>}<end_function_call>.
  if (name == "functiongemma") {
    return std::make_unique<FunctionGemmaToolParser>();
  }
  return nullptr;
}

}  // namespace vllm::entrypoints::openai
