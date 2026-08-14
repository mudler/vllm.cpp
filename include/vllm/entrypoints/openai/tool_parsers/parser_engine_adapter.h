// Ported from: vllm/parser/engine/adapters.py:128 (ParserEngineToolAdapter) +
// vllm/parser/engine/registered_adapters.py:67-70 (make_adapters(InklingParser)
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
//     detokenizer. IT IS DROPPED, AND THAT IS AN OPEN GAP — nothing on this side
//     handles it. The shared ToolParser seam has no `adjust_request` DISPATCH
//     SITE at all (`KimiK2ToolParser::adjust_request`, kimi_k2.cpp:87, has no
//     callers either; see reasoning_parsers/muse_glimmer.h, which records the
//     same seam gap), and the engine-backed serving path does not compensate:
//     `serving_chat.cpp` applies no skip_special_tokens override for the engine
//     branch. The consequence is MATERIAL for Inkling specifically, because its
//     whole grammar is special tokens — `skip_special_tokens` is declared `=
//     true` at protocol.h:240/:461, forwarded verbatim by to_sampling_params
//     (protocol.cpp:583) and honoured at v1/engine/detokenizer.cpp:68, so at
//     server defaults `<|content_invoke_tool_json|>` and its siblings are
//     stripped BEFORE the parser runs. The cases in tests/.../test_inkling.cpp
//     pass because they feed the adapter marker text directly; they do not
//     traverse the detokenizer. Tracked as issue #695, NOT fixed here.
//   - finish_streaming (adapters.py:189) is exposed on this class but is NOT on
//     the ToolParser ABC, so the legacy seam never calls it. Production serving
//     of an engine-backed name does not go through this adapter — it drives
//     ParserEngine::parse_delta(..., finished=true), which IS the flush — so the
//     method is here for parity of shape and for direct callers/tests. Stated
//     plainly: it has NO production caller anywhere in src/, include/ or
//     examples/, and it never will while `MakeToolParser` routes engine-backed
//     names away from this adapter (serving_chat.cpp:546-548, the
//     `get_parser_engine(tool_parser_name_) != nullptr` early return). Its only
//     caller is the ported upstream `_stream_text_only` harness in
//     tests/.../test_inkling.cpp, which calls it exactly as upstream does — and
//     that is a caller, NOT a guarantee: replacing this body with `return
//     std::nullopt` leaves that whole suite green, because no ported input
//     leaves anything deferred past the last delta. So the method ships
//     functionally UNGATED. Said plainly here because the alternative is a
//     header that implies coverage the mutation says does not exist.
//
// STRUCTURAL TAGS. inkling_tool_parser.py:10-11 sets `structural_tag_model =
// None` and `supports_required_and_named = False`, i.e. named/required tool
// choice falls back to unconstrained auto parsing. Our
// ToolChoiceStructuralTagSpecFor returns nullopt for every mode of an unmapped
// family (structural_tags.h COVERAGE), which is that behaviour already, so no
// registry row is added there. That equivalence is GATED, not merely asserted:
// tests/.../test_inkling.cpp `test_adapters_resolve` (the port of
// test_inkling.py:488, which asserts `supports_required_and_named is False`)
// checks nullopt for auto, required and named.
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

  // adapters.py:158 — straight delegation to extract_tool_calls_from_content.
  // NOTE: the class docstring (adapters.py:131) says extract_tool_calls "starts
  // the parser engine in CONTENT state"; the CODE seeds CONTENT only in the
  // STREAMING entrypoint (adapters.py:178). This mirrors the code.
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // adapters.py:167/:178 — initialize_streaming(CONTENT) then delegate. The
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
// registered_adapters.py:67-70's make_adapters(InklingParser) tool half.
// Registered as "inkling" (tool_parsers/__init__.py:177).
class InklingEngineToolParser final : public ParserEngineToolAdapter {
 public:
  InklingEngineToolParser();
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_PARSER_ENGINE_ADAPTER_H_
