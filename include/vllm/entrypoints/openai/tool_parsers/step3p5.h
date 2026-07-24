// Ported from: vllm/tool_parsers/step3p5_tool_parser.py @ e24d1b24
//
// Step3p5ToolParser - the StepFun Step-3.5 tool-call format. Tool calls are
// XML-PARAMETRIZED and wrapped in a BARE <tool_call> block (like Hermes), but
// the body is XML, not JSON:
//   <tool_call>
//   <function=NAME>
//   <parameter=KEY>
//   VALUE
//   </parameter>
//   ...
//   </function>
//   </tool_call>
// Parameter VALUES are typed from the tool JSON-schema (string / integer /
// number|float / boolean / object / array): numbers, bools and JSON/py-literal
// containers are coerced, everything else stays a string. The coercion is
// BEHAVIOUR (json.loads round-trips depend on it), not cosmetics, and is
// mirrored exactly (repair_param_type + _convert_param_value +
// _convert_for_json_streaming + the ast.literal_eval deferred path).
//
// The whole parser is ONE stateful streaming XML state machine
// (StreamingXMLToolCallParser). The NON-STREAMING extract_tool_calls feeds the
// entire model output as a single chunk and merges the emitted deltas; the
// STREAMING extract_tool_calls_streaming feeds each delta and returns that
// chunk's merged delta. Both share the same machine, so the port reproduces the
// full machine.
//
// DEVIATIONS from upstream (per the TEXT-ONLY, tokenizer-free abstract.h seam):
//   1. EXPAT -> a MINIMAL in-file XML event emitter. Upstream drives Python's
//      xml.parsers.expat over the preprocessed fragments (buffer_text=True). We
//      reproduce the exact event contract the handlers rely on - StartElement /
//      coalesced CharacterData (entity-unescaped) / EndElement, in document
//      order, with character data buffered ACROSS Parse() calls and flushed
//      right before the next tag - with a small hand-written emitter. It does
//      NOT enforce XML well-formedness / nesting and NEVER throws: expat would
//      abort on the malformed (missing </parameter>) case, after which upstream
//      RELIES on its own _end_element fallback to finish the call; our emitter
//      simply fires the tags in order and the handlers' auto-close logic
//      converges to the identical arguments. The fallback in
//      parse_single_streaming_chunks is still ported and stays guarded (its
//      has_*_close checks make it a no-op once a close was already emitted), so
//      neither path double-emits.
//   2. TOKEN IDs DROPPED. Upstream's streaming signature carries the
//      previous/current/delta token-id spans; the only use is the empty-
//      delta_text + delta_token_ids branch (an EOS-only step). Text-only, we map
//      empty delta_text -> nullopt (no test drives an empty delta).
//   3. TOOLS via REQUEST, not constructor. Upstream reads self.tools (set at
//      construction / via set_tools); the C++ seam carries the tools on the per-
//      call ChatCompletionRequest, so each call re-points the machine's tools at
//      request.tools before parsing.
//   4. ast.literal_eval -> a small Python-literal evaluator (numbers, bools,
//      None, single/double-quoted strings, lists, tuples, dicts) into
//      nlohmann::ordered_json. Used only on the deferred (complex/object-with-
//      single-quote) parameter path; on any parse failure it falls back to
//      emitting the raw text as a JSON string, exactly like upstream's except:.
//   5. ORDERED json throughout (nlohmann::ordered_json): json.dumps preserves
//      dict INSERTION order, and both the argument re-serialization and the
//      streaming character diff only reconstruct correctly with stable key order.
#pragma once

#include <memory>
#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

namespace step3p5_detail {
class StreamingXmlToolCallParser;  // defined in step3p5.cpp
}  // namespace step3p5_detail

class Step3p5ToolParser : public ToolParser {
 public:
  Step3p5ToolParser();
  ~Step3p5ToolParser() override;

  // step3p5_tool_parser.py:1379 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // step3p5_tool_parser.py:1444 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  // The heavy streaming XML state machine (step3p5_tool_parser.py:30-1363).
  // Held by pointer so the header stays free of the implementation detail.
  std::unique_ptr<step3p5_detail::StreamingXmlToolCallParser> parser_;
};

}  // namespace vllm::entrypoints::openai
