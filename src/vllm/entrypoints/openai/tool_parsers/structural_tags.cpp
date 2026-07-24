// Ported from: vllm/tool_parsers/structural_tag_registry.py @ e24d1b24
// See structural_tags.h for the registry purpose, the native spec shape and the
// per-family coverage table.
#include "vllm/entrypoints/openai/tool_parsers/structural_tags.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vllm/entrypoints/openai/serving_chat.h"  // ToolsEnabled
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v3.h"
#include "vllm/entrypoints/openai/tool_parsers/hermes.h"
#include "vllm/entrypoints/openai/tool_parsers/kimi_k2.h"
#include "vllm/entrypoints/openai/tool_parsers/longcat.h"
#include "vllm/entrypoints/openai/tool_parsers/mistral.h"

namespace vllm::entrypoints::openai {

namespace {

// _get_function_parameters (structural_tag_registry.py:207-210): the tool's
// `parameters` JSON schema is the tag content, or `true` (any JSON) when the
// tool declares none. `true` lowers to an any-JSON `value` in the native
// backend (backend_native.cpp / json_schema_to_gbnf.cpp).
nlohmann::json ToolArgumentsSchema(const FunctionDefinition& fn) {
  if (fn.parameters.has_value()) return *fn.parameters;
  return true;
}

// A family tag builder: the request's tools -> the structural-tag `tags` array
// (each tag = {begin, content_schema, end}). One builder per registered family.
using TagBuilder =
    std::function<nlohmann::json(const std::vector<ChatCompletionToolsParam>&)>;

// ── Family tag builders ─────────────────────────────────────────────────────

// Hermes surface (structural_tag_registry.py:213-234, _hermes_tool_tags): each
// tool yields TWO tags (the two wrapper variants - pretty-printed with a leading
// newline, and compact). The tool name is baked into `begin`; the args schema is
// the content; `end` closes the wrapper. `open`/`close` are the family wrapper
// tokens (`<tool_call>`/`</tool_call>` for Hermes/Qwen3, `<longcat_tool_call>`/
// `</longcat_tool_call>` for LongCat - the ONLY difference, mirroring how
// longcat.cpp subclasses hermes.cpp).
nlohmann::json HermesSurfaceTags(
    const std::vector<ChatCompletionToolsParam>& tools, const std::string& open,
    const std::string& close) {
  // arguments_field_prefix (structural_tag_registry.py:214): closes the name
  // string and opens the arguments value.
  static constexpr const char* kArgumentsFieldPrefix = "\", \"arguments\": ";
  const std::pair<std::string, std::string> formats[] = {
      {open + "\n{\"name\": \"", "}\n" + close},
      {open + "{\"name\": \"", "}" + close},
  };
  nlohmann::json tags = nlohmann::json::array();
  // Tools OUTER, formats INNER (structural_tag_registry.py:232-233).
  for (const ChatCompletionToolsParam& tool : tools) {
    for (const auto& [begin, end] : formats) {
      nlohmann::json tag = nlohmann::json::object();
      tag["begin"] = begin + tool.function.name + kArgumentsFieldPrefix;
      tag["content_schema"] = ToolArgumentsSchema(tool.function);
      tag["end"] = end;
      tags.push_back(std::move(tag));
    }
  }
  return tags;
}

nlohmann::json HermesTags(const std::vector<ChatCompletionToolsParam>& tools) {
  return HermesSurfaceTags(tools, HermesToolParser::kToolCallStartToken,
                           HermesToolParser::kToolCallEndToken);
}

nlohmann::json LongcatTags(const std::vector<ChatCompletionToolsParam>& tools) {
  return HermesSurfaceTags(tools, LongcatToolParser::kToolCallStartToken,
                           LongcatToolParser::kToolCallEndToken);
}

// Llama 3.x/4 JSON surface (llama.h / llama_tool_parser.py): a bare object
// `{"name": "<fn>", "parameters": {<args>}}`. The name is baked into `begin`,
// the args schema is the content, `end` closes the object. Upstream accepts both
// the `parameters` and `arguments` keys and an optional `<|python_tag|>` prefix;
// a forced tag must pick ONE surface, so we emit the canonical `parameters`
// form the parser tests use (the parser reads either key back).
nlohmann::json LlamaTags(const std::vector<ChatCompletionToolsParam>& tools) {
  nlohmann::json tags = nlohmann::json::array();
  for (const ChatCompletionToolsParam& tool : tools) {
    nlohmann::json tag = nlohmann::json::object();
    tag["begin"] = "{\"name\": \"" + tool.function.name + "\", \"parameters\": ";
    tag["content_schema"] = ToolArgumentsSchema(tool.function);
    tag["end"] = "}";
    tags.push_back(std::move(tag));
  }
  return tags;
}

// DeepSeek-V3 / R1 surface (deepseek_v3.h / deepseekv3_tool_parser.py). A call
// is the OUTER wrapper + one INNER call `<｜tool▁call▁begin｜>function
// <｜tool▁sep｜>NAME\n```json\nARGS\n```<｜tool▁call▁end｜>`. The type is always
// "function"; the args are a JSON object inside a ```json fence.
//
// DEVIATION (multi-call `required`): the flat native tag shape cannot factor the
// shared OUTER wrapper out of the inner repetition, so each tag bakes the OUTER
// begin/end in. A single call (auto / named / the common `required` case) is
// byte-faithful; a `required` run of >1 call re-emits the OUTER wrapper per call
// rather than one wrapper around many. Documented; the primary goal - forcing
// the DeepSeek markers instead of Hermes - holds for every case.
nlohmann::json DeepSeekV3Tags(
    const std::vector<ChatCompletionToolsParam>& tools) {
  const std::string outer_begin = DeepSeekV3ToolParser::kToolCallsBeginToken;
  const std::string outer_end = DeepSeekV3ToolParser::kToolCallsEndToken;
  const std::string call_begin = DeepSeekV3ToolParser::kToolCallBeginToken;
  const std::string call_end = DeepSeekV3ToolParser::kToolCallEndToken;
  const std::string sep = DeepSeekV3ToolParser::kToolSepToken;
  nlohmann::json tags = nlohmann::json::array();
  for (const ChatCompletionToolsParam& tool : tools) {
    nlohmann::json tag = nlohmann::json::object();
    tag["begin"] = outer_begin + call_begin + "function" + sep +
                   tool.function.name + "\n```json\n";
    tag["content_schema"] = ToolArgumentsSchema(tool.function);
    tag["end"] = "\n```" + call_end + outer_end;
    tags.push_back(std::move(tag));
  }
  return tags;
}

// DeepSeek-V3.1 surface (deepseek_v31.h): same markers as V3 but NO ```json
// fence - the args follow the separator directly, and the type is always
// "function": `<｜tool▁call▁begin｜>NAME<｜tool▁sep｜>ARGS<｜tool▁call▁end｜>`.
// Same OUTER-wrapper multi-call deviation as DeepSeekV3Tags.
nlohmann::json DeepSeekV31Tags(
    const std::vector<ChatCompletionToolsParam>& tools) {
  const std::string outer_begin = DeepSeekV3ToolParser::kToolCallsBeginToken;
  const std::string outer_end = DeepSeekV3ToolParser::kToolCallsEndToken;
  const std::string call_begin = DeepSeekV3ToolParser::kToolCallBeginToken;
  const std::string call_end = DeepSeekV3ToolParser::kToolCallEndToken;
  const std::string sep = DeepSeekV3ToolParser::kToolSepToken;
  nlohmann::json tags = nlohmann::json::array();
  for (const ChatCompletionToolsParam& tool : tools) {
    nlohmann::json tag = nlohmann::json::object();
    tag["begin"] = outer_begin + call_begin + tool.function.name + sep;
    tag["content_schema"] = ToolArgumentsSchema(tool.function);
    tag["end"] = call_end + outer_end;
    tags.push_back(std::move(tag));
  }
  return tags;
}

// Mistral v11 name-first surface (mistral.h / mistral_tool_parser.py): each call
// is `[TOOL_CALLS]<name>{args}` (the args JSON object follows the bare name with
// no separator, and there is no closing wrapper - the next `[TOOL_CALLS]` or EOS
// ends it). Each tag begins with `[TOOL_CALLS]` so the one-or-more (`required`)
// repetition is byte-faithful (unlike DeepSeek there is no shared outer wrapper).
nlohmann::json MistralTags(const std::vector<ChatCompletionToolsParam>& tools) {
  const std::string bot = MistralToolParser::kBotToken;
  nlohmann::json tags = nlohmann::json::array();
  for (const ChatCompletionToolsParam& tool : tools) {
    nlohmann::json tag = nlohmann::json::object();
    tag["begin"] = bot + tool.function.name;
    tag["content_schema"] = ToolArgumentsSchema(tool.function);
    tag["end"] = "";
    tags.push_back(std::move(tag));
  }
  return tags;
}

// Kimi K2 surface (kimi_k2.h / vllm/parser/kimi_k2.py). A call is the SECTION
// wrapper + one INNER call: `<|tool_calls_section_begin|><|tool_call_begin|>
// functions.NAME:0 <|tool_call_argument_begin|>{args}<|tool_call_end|>
// <|tool_calls_section_end|>`. The args ARE a JSON object, so a JSON
// content_schema expresses the surface (unlike the DSML families). The native id
// header is baked as `functions.NAME:0` (the index the parser reads back), and a
// single space separates it from the argument-begin marker (the parser strips
// the header, so the exact spacing is not load-bearing for extraction).
//
// DEVIATION (multi-call `required`, index): like DeepSeek, the flat native tag
// cannot factor the shared SECTION wrapper out of the inner repetition, so each
// tag bakes the SECTION begin/end in - a `required` run of >1 call re-emits the
// section wrapper per call rather than one section around many; and the index is
// baked `:0` for every tool (a parallel call's real `:1`/`:2` would not match the
// forced literal). A single call (auto / named / the common `required` case) is
// byte-faithful; the primary goal - forcing the Kimi markers instead of Hermes -
// holds for every case. Documented (mirrors DeepSeekV3Tags).
nlohmann::json KimiK2Tags(const std::vector<ChatCompletionToolsParam>& tools) {
  const std::string section_begin =
      KimiK2ToolParser::kToolCallsSectionBeginToken;
  const std::string section_end = KimiK2ToolParser::kToolCallsSectionEndToken;
  const std::string call_begin = KimiK2ToolParser::kToolCallBeginToken;
  const std::string call_end = KimiK2ToolParser::kToolCallEndToken;
  const std::string arg_begin = KimiK2ToolParser::kToolCallArgumentBeginToken;
  nlohmann::json tags = nlohmann::json::array();
  for (const ChatCompletionToolsParam& tool : tools) {
    nlohmann::json tag = nlohmann::json::object();
    tag["begin"] = section_begin + call_begin + "functions." +
                   tool.function.name + ":0 " + arg_begin;
    tag["content_schema"] = ToolArgumentsSchema(tool.function);
    tag["end"] = call_end + section_end;
    tags.push_back(std::move(tag));
  }
  return tags;
}

// ── The registry: parser name -> (auto trigger marker, tag builder) ──────────

struct FamilySpec {
  std::string trigger;   // the LAZY (auto) trigger - the family's begin marker.
  TagBuilder build_tags;
};

const std::unordered_map<std::string, FamilySpec>& Registry() {
  static const std::unordered_map<std::string, FamilySpec> registry = [] {
    std::unordered_map<std::string, FamilySpec> r;
    // Hermes-format families (Qwen3 shares the Hermes surface, qwen3.h).
    r.emplace("hermes",
              FamilySpec{HermesToolParser::kToolCallStartToken, HermesTags});
    r.emplace("qwen3",
              FamilySpec{HermesToolParser::kToolCallStartToken, HermesTags});
    r.emplace("longcat", FamilySpec{LongcatToolParser::kToolCallStartToken,
                                    LongcatTags});
    // Llama 3.x/4 JSON - trigger on the `{"name": "` prefix the call opens with.
    r.emplace("llama3_json", FamilySpec{"{\"name\": \"", LlamaTags});
    r.emplace("llama4_json", FamilySpec{"{\"name\": \"", LlamaTags});
    // DeepSeek marker families - trigger on the OUTER begin marker.
    r.emplace("deepseek_v3",
              FamilySpec{DeepSeekV3ToolParser::kToolCallsBeginToken,
                         DeepSeekV3Tags});
    r.emplace("deepseek_v31",
              FamilySpec{DeepSeekV3ToolParser::kToolCallsBeginToken,
                         DeepSeekV31Tags});
    // Mistral v11 name-first - trigger on `[TOOL_CALLS]`.
    r.emplace("mistral",
              FamilySpec{MistralToolParser::kBotToken, MistralTags});
    // DELIBERATELY UNMAPPED (nullopt for every mode), documented in
    // structural_tags.h COVERAGE: qwen3_coder / qwen3_xml / mimo speak the
    // Qwen3-Coder per-parameter XML surface (<function=..><parameter=..>..),
    // which - like the DSML deepseek_v32/v4 families - is not a flat JSON-args
    // begin/content/end tag. Upstream constrains it with the xgrammar
    // `qwen_3_coder` builtin, not a flat structural-tag builder, so there is no
    // family builder to add here.
    // Kimi K2 - trigger on the section-begin marker.
    r.emplace("kimi_k2",
              FamilySpec{KimiK2ToolParser::kToolCallsSectionBeginToken,
                         KimiK2Tags});
    return r;
  }();
  return registry;
}

}  // namespace

std::optional<nlohmann::json> ToolChoiceStructuralTagSpecFor(
    const std::string& tool_parser_name, const ChatCompletionRequest& request) {
  // No tools, or tool_choice="none": no structural tag (model unconstrained).
  if (!ToolsEnabled(request)) return std::nullopt;
  if (!request.tools.has_value() || request.tools->empty()) return std::nullopt;

  // Unmapped family (or empty/unknown parser name): NO constraint for any mode.
  // `auto` stays graceful (as before); `required`/`named` let the model emit its
  // OWN trained syntax unconstrained rather than wrong-forcing Hermes tags.
  const auto& registry = Registry();
  const auto it = registry.find(tool_parser_name);
  if (it == registry.end()) return std::nullopt;
  const FamilySpec& family = it->second;

  const std::vector<ChatCompletionToolsParam>& tools = *request.tools;
  // tool_choice unset defaults to "auto" when tools are present.
  const std::string mode =
      request.tool_choice.has_value() ? request.tool_choice->mode : "auto";

  nlohmann::json spec = nlohmann::json::object();

  // named ("function"): exactly one tag, forced from token 0
  // (structural_tag_registry.py:255-261, stop_after_first).
  if (mode == "function") {
    const std::string name = request.tool_choice->function_name.value_or("");
    std::vector<ChatCompletionToolsParam> chosen;
    for (const ChatCompletionToolsParam& tool : tools) {
      if (tool.function.name == name) chosen.push_back(tool);
    }
    // A named choice matching no tool: no constraint (upstream 400s earlier at
    // validation).
    if (chosen.empty()) return std::nullopt;
    spec["lazy"] = false;
    spec["stop_after_first"] = true;
    spec["tags"] = family.build_tags(chosen);
    return spec;
  }

  // required: any listed tool, at_least_one, forced from token 0
  // (structural_tag_registry.py:262-267 - one-or-more, no stop_after_first).
  if (mode == "required") {
    spec["lazy"] = false;
    spec["stop_after_first"] = false;
    spec["tags"] = family.build_tags(tools);
    return spec;
  }

  // auto (and the tools-present unset default): LAZY
  // (structural_tag_registry.py:248-254 - TriggeredTagsFormat, inert until the
  // family's begin marker). Plain text is free; a call is constrained only after
  // the model chooses to start one. NOT forced.
  spec["lazy"] = true;
  spec["triggers"] = nlohmann::json::array({family.trigger});
  spec["stop_after_first"] = false;
  spec["tags"] = family.build_tags(tools);
  return spec;
}

void ApplyToolChoiceStructuredOutput(const std::string& tool_parser_name,
                                     const ChatCompletionRequest& request,
                                     SamplingParams& sampling_params) {
  const std::optional<nlohmann::json> spec =
      ToolChoiceStructuralTagSpecFor(tool_parser_name, request);
  if (!spec.has_value()) return;
  // The structural tag IS the structured-output constraint (a tool-constrained
  // request carries no response_format constraint in practice). Route it through
  // `structural_tag` (the kStructuralTag native compile path): the native
  // backend builds the family wrapper + a LAZY (auto) or FORCED (required/named)
  // grammar the family's tool parser extracts. structured_outputs then holds
  // exactly `structural_tag`; json/grammar stay unset.
  StructuredOutputsParams so;
  so.structural_tag = spec->dump();
  sampling_params.structured_outputs = std::move(so);
  sampling_params.structured_outputs->Verify();  // exactly-one-constraint check.
}

}  // namespace vllm::entrypoints::openai
