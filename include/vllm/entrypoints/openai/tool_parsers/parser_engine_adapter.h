// Ported from: vllm/parser/engine/adapters.py:128 (ParserEngineToolAdapter) +
// vllm/parser/engine/registered_adapters.py:68-70 (make_adapters(InklingParser)
// -> InklingParserToolAdapter) + vllm/tool_parsers/inkling_tool_parser.py:7
// (InklingEngineToolParser, the class the registry entry
// vllm/tool_parsers/__init__.py:177 names) @ 555967922 (vLLM 0.26.0.dev0).
//
// The TOOL half of `make_adapters`. Its reasoning twin already lives in
// reasoning_parsers/parser_engine_adapter.h; this is the same re-shaping in the
// other direction — a ParserEngine presented through the legacy ToolParser ABC,
// so an engine-backed dialect is reachable from `get_tool_parser(name)` (and
// therefore from `--tool-call-parser NAME`, whose validation runs through
// ResolveToolParserName -> get_tool_parser).
//
// WHY THIS EXISTS AT ALL, given src/vllm/parser/inkling.cpp already ports the
// whole Inkling engine: the serving path picks the engine over the legacy seam
// (serving_chat.cpp MakeParserEngine -> parser::get_parser_engine, which has
// answered "inkling" since the assembly work landed), but the NAME was never in
// the tool-parser registry, so `--tool-call-parser inkling` threw at startup and
// the ported dialect was unreachable. Registering it is the whole gap.
//
// DEVIATIONS from adapters.py:128, all inherited from the legacy ToolParser seam
// documented in tool_parsers/abstract.h:
//   - The ABC has no tokenizer and no `tools` list, so the ctor takes the built
//     engine instead of a (tokenizer, tools) pair, exactly as the reasoning
//     adapter does.
//   - The streaming signature drops the three token-ID spans; the engine is fed
//     text-only deltas and its incremental lexer holds a marker split across
//     deltas. Inkling opts INTO text-lexer terminal recognition upstream
//     (inkling.py:286 `token_id_terminals={}`), so this seam loses nothing for
//     this family.
//   - adjust_request (adapters.py:151) sets skip_special_tokens=False for the
//     detokenizer; our ToolParser ABC carries no request-adjust hook (the
//     engine-backed serving path handles it), so it is not modelled.
//   - finish_streaming (adapters.py:189) is exposed on this class but is NOT on
//     the ToolParser ABC, so the legacy seam never calls it. Production serving
//     of an engine-backed name does not go through this adapter — it drives
//     ParserEngine::parse_delta(..., finished=true), which IS the flush — so the
//     method is here for parity of shape and for direct callers/tests.
//
// STRUCTURAL TAGS. inkling_tool_parser.py:10-11 sets `structural_tag_model =
// None` and `supports_required_and_named = False`, i.e. named/required tool
// choice falls back to unconstrained auto parsing. Our
// ToolChoiceStructuralTagSpecFor returns nullopt for every mode of an unmapped
// family (structural_tags.h COVERAGE), which is that behaviour already, so no
// registry row is added there.
#ifndef VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_PARSER_ENGINE_ADAPTER_H_
#define VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_PARSER_ENGINE_ADAPTER_H_

#include <memory>
#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/parser/engine/parser_engine.h"

namespace vllm::entrypoints::openai {

// adapters.py:128 (ParserEngineToolAdapter). Holds ONE engine for the whole
// request, like every streaming tool parser on this seam.
class ParserEngineToolAdapter : public ToolParser {
 public:
  explicit ParserEngineToolAdapter(
      std::unique_ptr<vllm::parser::engine::ParserEngine> engine);
  ~ParserEngineToolAdapter() override;

  // adapters.py:158 — extract_tool_calls starts the engine in CONTENT state so
  // it parses reasoning-stripped content (the output of extract_reasoning).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // adapters.py:167 — initialize_streaming(CONTENT) then delegate. The
  // initialize is idempotent after the first delta (parser_engine.py:_reset is
  // guarded by _streaming_initialized), so the CONTENT seed applies once.
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // adapters.py:189. Not an ABC method here — see the DEVIATIONS note above.
  std::optional<DeltaMessage> finish_streaming();

 protected:
  std::unique_ptr<vllm::parser::engine::ParserEngine> engine_;
};

// inkling_tool_parser.py:7 (InklingEngineToolParser) over
// registered_adapters.py:68-70's make_adapters(InklingParser) tool half.
// Registered as "inkling" (tool_parsers/__init__.py:177).
class InklingEngineToolParser final : public ParserEngineToolAdapter {
 public:
  InklingEngineToolParser();
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_PARSER_ENGINE_ADAPTER_H_
