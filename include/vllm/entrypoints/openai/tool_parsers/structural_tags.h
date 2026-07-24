// Ported from: vllm/tool_parsers/structural_tag_registry.py @ e24d1b24
// (the per-family STRUCTURAL-TAG registry: get_model_structural_tag +
// _VLLM_STRUCTURAL_TAG_REGISTRY + the xgrammar builtin specs it dispatches to).
//
// PURPOSE. tool_choice (auto / required / named) is enforced during decode by a
// STRUCTURAL-TAG constraint. Upstream picks the tag surface PER MODEL FAMILY
// (the active tool parser's `structural_tag_model`) so `required`/`named` force
// the model's OWN native tool syntax. The pre-existing C++ port built the HERMES
// spec for EVERY request regardless of the active parser, which forced Hermes
// `<tool_call>{...}</tool_call>` syntax onto the other 20+ dialects that were
// never trained on it (`auto` degraded gracefully because the Hermes trigger
// never fired, but `required`/`named` wrong-forced). This registry replaces that
// one-size-fits-all builder with a per-family lookup keyed by the active tool
// parser name.
//
// The native structural-tag JSON shape (consumed by backend_native.cpp
// kStructuralTag; the SEAM is 1:1 with vLLM's xgrammar StructuralTag, the
// content backend-private):
//   {"lazy": bool, "triggers": [str], "stop_after_first": bool,
//    "tags": [{"begin": str, "content_schema": <schema|true>, "end": str}]}
// Each tag lowers to `begin-literal + content + end-literal`; the tool name is
// baked into `begin`, `content_schema` is the tool's `parameters` (or `true` =
// any JSON when it declares none), and `end` closes the family's wrapper. The
// begin/end byte strings are the ACTIVE family's OWN markers (read from each
// parser's constants), so `required`/`named` force the native syntax the model's
// parser can extract.
//
// tool_choice -> spec (same per-family semantics as get_hermes_structural_tag,
// structural_tag_registry.py:248-267):
//   auto     -> LAZY: {lazy:true, triggers:[<family begin marker>],
//               tags:[all tools]} - plain text is FREE until the marker, then the
//               call is constrained. NOT forced (the model may just reply).
//   required -> FORCED >=1: {lazy:false, stop_after_first:false, tags:[all]}.
//   named    -> FORCED exactly one: {lazy:false, stop_after_first:true,
//               tags:[that one tool]}.
//
// COVERAGE (family -> auto/required/named). A family has a spec only when its
// wire format is a `begin-literal + JSON-args + end-literal` surface a JSON
// content_schema can express:
//   hermes, qwen3        -> full  (the two Hermes `<tool_call>` surface variants)
//   longcat              -> full  (Hermes surface, `<longcat_tool_call>` wrapper)
//   llama3_json/llama4_json -> full  (`{"name":..,"parameters":{..}}`)
//   deepseek_v3          -> full  (deepseek_r1 markers + ```json fence)
//   deepseek_v31         -> full  (deepseek markers, args after the separator)
//   mistral              -> full  (`[TOOL_CALLS]<name>{args}`, v11 name-first)
//   kimi_k2              -> full  (section+call markers, args a JSON object;
//       `functions.NAME:0` id + section wrapper baked - single-call faithful,
//       multi-call `required` re-wraps per call, same as DeepSeek's deviation)
// Families that return NULLOPT for every mode (documented at their site):
//   deepseek_v32, deepseek_v4 -> the DSML per-parameter XML surface
//       (<｜DSML｜parameter name=..>..) is NOT a JSON args object, so a JSON
//       content_schema cannot express it. Upstream uses a full xgrammar builtin
//       grammar we do not port; a flat native structural tag cannot represent it.
//   qwen3_coder, qwen3_xml, mimo -> the Qwen3-Coder XML surface
//       (<function=NAME><parameter=KEY>VALUE</parameter>..) has the same defeat
//       as DSML: each parameter is its OWN <parameter=..> element, so the
//       per-call content is NOT a single JSON args object a content_schema can
//       express. Upstream drives it through the xgrammar `qwen_3_coder` BUILTIN
//       (structural_tag_registry.py XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS), NOT
//       a _VLLM_STRUCTURAL_TAG_REGISTRY flat-tag builder, so there is no flat
//       begin/content/end shape to mirror - nullopt for every mode, exactly the
//       deepseek_v32 precedent.
//   glm45, glm47 -> the GLM per-argument XML surface (<arg_key>K</arg_key>
//       <arg_value>V</arg_value>) is the same per-arg-XML shape as DSML, NOT a JSON
//       args object. Upstream's structural_tag_model is the `glm_4_7` XGRAMMAR
//       BUILTIN grammar (not a vLLM-owned tag builder we could mirror); a flat
//       native structural tag cannot represent per-arg XML. Same precedent as
//       deepseek_v32.
//   minimax_m2 -> upstream DOES register a `minimax` tag builder, but its
//       per-parameter content uses xgrammar's `minimax_xml` style (it renders the
//       JSON schema AS <parameter name=..>..</parameter> XML, not a JSON args
//       object). The flat native tag's JSON content_schema cannot express that
//       per-argument XML, so nullopt - same per-arg-XML precedent as GLM/DSML.
//   pythonic, llama4_pythonic -> Python bracket call grammar (`[fn(a=1)]`), not
//       a taggable begin/end surface.
//   xlam                 -> multi-envelope (fenced / [TOOL_CALLS] / bare-list),
//       no single begin/end.
//   granite, granite4, granite-20b-fc, phi4_mini_json, internlm, jamba, step3,
//   step3p5, minicpm5, hy_v3, olmo3 and any other unmapped parser -> no
//       upstream vLLM spec and no simple JSON-args begin/end surface.
// For a NULLOPT family every mode applies NO decode constraint: `auto` is the
// same graceful behavior as before, and `required`/`named` let the model emit
// its OWN trained syntax UNCONSTRAINED (its parser still extracts it) instead of
// being wrong-forced into Hermes tags.
#ifndef VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_STRUCTURAL_TAGS_H_
#define VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_STRUCTURAL_TAGS_H_

#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/sampling_params.h"

namespace vllm::entrypoints::openai {

// Build the native structural-tag JSON for the ACTIVE tool parser family
// (`tool_parser_name`, the same name get_tool_parser dispatches on). Returns
// nullopt when no constraint applies: no tools / tool_choice="none", a named
// choice matching no tool, or a family with no expressible tag surface (see the
// COVERAGE list above). An empty/unknown parser name is treated as an unmapped
// family (nullopt for every mode).
std::optional<nlohmann::json> ToolChoiceStructuralTagSpecFor(
    const std::string& tool_parser_name, const ChatCompletionRequest& request);

// Apply ToolChoiceStructuralTagSpecFor(tool_parser_name, request) onto
// `sampling_params`: sets structured_outputs.structural_tag = dump(spec) (the ONE
// structured-output constraint; json/grammar stay unset) and re-runs Verify().
// No-op when the spec is nullopt. Called in create_chat_completion before
// add_request with the serving handler's active tool_parser_name.
void ApplyToolChoiceStructuredOutput(const std::string& tool_parser_name,
                                     const ChatCompletionRequest& request,
                                     SamplingParams& sampling_params);

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_STRUCTURAL_TAGS_H_
