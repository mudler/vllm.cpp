// Ported from: vllm/parser/qwen3.py:201 (Qwen3Parser) @ 555967922
// (vLLM 0.26.0.dev0).
//
// Qwen3Parser: the assembly-layer ParserEngine subclass for the Qwen3 family —
// <think>/</think> reasoning plus the <tool_call> XML tool format, one engine.
// The grammar, transition table and _qwen3_arg_converter live in
// engine::qwen3_config(); this class carries the two behaviours the base engine
// cannot express:
//
//   1. thinking-off passthrough (qwen3.py:247 extract_reasoning) — with
//      enable_thinking=False the whole model output is CONTENT, unsplit. The
//      config alone does NOT give this: initial_state=CONTENT still lets a
//      <tool_call> steal the tool body out of the content span.
//   2. an UNPAIRED tool-call marker also ends reasoning (qwen3.py:256
//      is_reasoning_end), on top of the base "last </think> after the last
//      <think>" rule.
//
// seed_oss is the SAME class upstream (seed_oss.py: `class
// SeedOssParser(Qwen3Parser)`, only the four wrapper token strings differ), so
// it is constructed from this class with the seed_oss config rather than from a
// second parallel path. Both behaviours above read their literals from the
// config's terminals, so the seed_oss wrappers are handled by construction.
#pragma once

#include <optional>
#include <string>
#include <utility>

#include "vllm/parser/engine/parser_engine.h"
#include "vllm/parser/engine/parser_engine_config.h"
#include "vllm/parser/engine/token_id_scanner.h"

namespace vllm::parser {

class Qwen3Parser : public engine::ParserEngine {
 public:
  // `thinking` is upstream's chat_template_kwargs["enable_thinking"]
  // (qwen3.py:225): it selects the config's initial state AND gates
  // extract_reasoning, so it is threaded through both here.
  Qwen3Parser(engine::ParserEngineConfig config, bool thinking,
              const engine::EngineTokenizer* tokenizer = nullptr)
      : engine::ParserEngine(std::move(config), tokenizer),
        thinking_enabled_(thinking) {}

  // qwen3.py:247 — thinking off: (None, model_output), no state machine.
  std::pair<std::optional<std::string>, std::optional<std::string>>
  extract_reasoning(const std::string& model_output,
                    const engine::ParserRequest& request) override;

  // qwen3.py:256 — base rule, then an unpaired tool-call marker (TEXT form of
  // the token-ID scan; see ParserEngine::is_reasoning_end).
  bool is_reasoning_end(const std::string& text) const override;

  bool thinking_enabled() const { return thinking_enabled_; }

 private:
  bool thinking_enabled_ = true;
};

}  // namespace vllm::parser
