// Ported from: vllm/tool_parsers/qwen3_engine_tool_parser.py @ e24d1b24
//
// FORMAT DECISION (cited): the gate model Qwen3.6-35B (arch qwen35moe,
// Qwen3_5MoeForConditionalGeneration) emits the SAME Hermes
// `<tool_call>{"name":...,"arguments":{...}}</tool_call>` JSON format as the
// Hermes parser — its tokenizer carries the <tool_call> / </tool_call> tokens
// and NO <function=>/<parameter=> XML tokens (tests/parity/goldens/
// tokenizer_qwen36/tokenizer.json), and docs/features/tool_calling.md:317 maps
// Qwen (non-Coder) models to the `hermes` parser. So Qwen3ToolParser reuses the
// Hermes extraction verbatim (identical start/end tokens).
//
// NOTE: upstream's registered "qwen3_coder"/"qwen3_xml"/"mimo" parser
// (Qwen3EngineToolParser -> Qwen3ParserToolAdapter) targets the Qwen3-**Coder**
// models' XML tool syntax (<function=..><parameter=..>..); that engine-based
// ParserEngine state machine is a distinct, non-gate-model format and is
// DEFERRED (out of T0 scope). Our "qwen3" == the gate model's Hermes format.
#pragma once

#include "vllm/entrypoints/openai/tool_parsers/hermes.h"

namespace vllm::entrypoints::openai {

// The gate model's Hermes-format parser. Behaviorally identical to
// HermesToolParser (same `<tool_call>{json}</tool_call>` tokens).
class Qwen3ToolParser : public HermesToolParser {
 public:
  Qwen3ToolParser() = default;
  // Out-of-line key function (defined in qwen3.cpp) — anchors the vtable.
  ~Qwen3ToolParser() override;
};

}  // namespace vllm::entrypoints::openai
