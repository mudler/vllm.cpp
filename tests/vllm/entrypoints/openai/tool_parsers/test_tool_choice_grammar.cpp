// LOOP-CLOSING test for the M3.3 tool_choice forced-grammar fix.
//
// THE BUG (adversarial-review-confirmed): for tool_choice="required" / a named
// function, ApplyToolChoiceStructuredOutput set structured_outputs.json to the
// BARE tool-call object schema, so the constrained decode emitted BARE JSON
// (`{"name":...,"arguments":...}`) with NO `<tool_call>` wrapper. The Hermes/Qwen
// parser's extract_tool_calls FIRST guard is `if find("<tool_call>") == npos ->
// tools_called=false`, so the forced call was DROPPED (returned as plain content,
// finish_reason="stop").
//
// THE FIX: force the wrapper IN THE GRAMMAR (WrapSchemaAsToolCallGbnf): the
// constrained decode emits `<tool_call>\n{...}\n</tool_call>` and BOTH the
// non-stream and stream parsers extract it.
//
// The original Task-4 test only checked schema SHAPE, never that the constrained
// output round-trips through the parser. This test closes that loop end-to-end:
//   forced schema -> WrapSchemaAsToolCallGbnf -> compile in the NATIVE grammar ->
//   (a) the WRAPPED output is accepted char-by-char + terminates,
//   (b) BARE JSON is REJECTED at the first `{`,
//   (c) extract_tool_calls on the wrapped output yields the ToolCall, and
//   (d) ShapeChatMessage (serving level) sets tool_calls + finish_reason=
//       "tool_calls" for a forced (named) request.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"
#include "vllm/entrypoints/openai/tool_parsers/hermes.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/structured_output/backend_native.h"
#include "vllm/v1/structured_output/backend_types.h"
#include "vllm/v1/structured_output/json_schema_to_gbnf.h"

using nlohmann::json;
using vllm::SamplingParams;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::BitmaskWordsForVocab;
using vllm::v1::NativeGrammar;
using vllm::v1::NativeStructuredOutputBackend;
using vllm::v1::StructuredOutputGrammar;
using vllm::v1::StructuredOutputOptions;
using vllm::v1::TokenBitmask;
using vllm::v1::WrapSchemaAsToolCallGbnf;
using namespace vllm::entrypoints::openai;

namespace {

// A byte-level fixture whose vocab covers every character the wrapped tool call
// needs: the `<tool_call>` / `</tool_call>` letters + `<`, `>`, `/`, newline, and
// the JSON structure / value characters used by the schema and arguments.
const std::string kChars =
    "<>/_\n{}[]\":,. -0123456789abcdefghijklmnopqrstuvwxyzP";

constexpr int32_t kEos = 250;

class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_toolchoice_gbnf_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

Tokenizer BuildFixture() {
  json vocab = json::object();
  for (std::size_t i = 0; i < kChars.size(); ++i) {
    vocab[MapBytesToUnicode(std::string(1, kChars[i]))] = static_cast<int>(i);
  }
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", kEos}, {"content", "<eos>"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", json::array()}};
  TempJson f(doc.dump());
  return Tokenizer::FromHfJson(f.path());
}

const Tokenizer& Fixture() {
  static const Tokenizer tok = BuildFixture();
  return tok;
}

int VocabSize() { return static_cast<int>(Fixture().VocabSize()); }

std::vector<int32_t> Encode(const std::string& text) {
  std::vector<int32_t> ids;
  for (const char ch : text) {
    const std::size_t idx = kChars.find(ch);
    REQUIRE_MESSAGE(idx != std::string::npos,
                    "test char not in fixture vocab: '" << ch << "'");
    ids.push_back(static_cast<int32_t>(idx));
  }
  return ids;
}

std::unique_ptr<NativeStructuredOutputBackend> MakeBackend() {
  return std::make_unique<NativeStructuredOutputBackend>(
      Fixture(), VocabSize(), std::vector<int32_t>{kEos});
}

std::unique_ptr<NativeGrammar> CompileGrammar(
    NativeStructuredOutputBackend& backend, const std::string& gbnf) {
  std::unique_ptr<StructuredOutputGrammar> g =
      backend.compile_grammar(StructuredOutputOptions::kGrammar, gbnf);
  return std::unique_ptr<NativeGrammar>(
      static_cast<NativeGrammar*>(g.release()));
}

bool Allowed(NativeGrammar& g, int32_t t) {
  TokenBitmask bm;
  bm.num_seqs = 1;
  bm.num_words = BitmaskWordsForVocab(VocabSize());
  bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
  g.fill_bitmask(bm, 0);
  const int32_t word = bm.data[static_cast<std::size_t>(t >> 5)];
  return (word & (1 << (t & 31))) != 0;
}

// The per-tool forced object schema (mirrors ToolCallObjectSchema in
// serving_chat.cpp): {"type":"object","properties":{"name":{const},
// "arguments":<params>},"required":[...],"additionalProperties":false}.
json ToolCallObjectSchema(const std::string& name, const json& params) {
  json props = json::object();
  props["name"] = json{{"const", name}};
  props["arguments"] = params;
  return json{{"type", "object"},
              {"properties", props},
              {"required", json::array({"name", "arguments"})},
              {"additionalProperties", false}};
}

}  // namespace

// ── (a)+(b): the wrapped grammar ACCEPTS the wrapped form, REJECTS bare JSON ──
TEST_CASE("ToolChoice grammar: wraps <tool_call>, rejects bare JSON") {
  // arguments schema: {"type":"object","properties":{"city":{"type":"string"}},
  //                    "required":["city"]}
  const json params = {{"type", "object"},
                       {"properties", {{"city", {{"type", "string"}}}}},
                       {"required", json::array({"city"})}};
  const json schema = ToolCallObjectSchema("get_weather", params);
  const std::string gbnf = WrapSchemaAsToolCallGbnf(schema);

  auto backend = MakeBackend();

  // The grammar's FIRST byte must be '<' (the wrapper), never '{' — this is what
  // makes bare JSON impossible.
  {
    auto g = CompileGrammar(*backend, gbnf);
    CHECK(Allowed(*g, static_cast<int32_t>(kChars.find('<'))));
    CHECK_FALSE(Allowed(*g, static_cast<int32_t>(kChars.find('{'))));
    CHECK_FALSE(Allowed(*g, kEos));  // empty stream is not a complete call.
  }

  // The exact wrapped output the constrained decode would emit is accepted
  // char-by-char AND terminates (EOS allowed) at the end.
  {
    auto g = CompileGrammar(*backend, gbnf);
    // nlohmann sorts object keys, so the grammar's canonical order is
    // "arguments" before "name". The regex parser extracts either order.
    const std::string wrapped =
        "<tool_call>\n{\"arguments\":{\"city\":\"Paris\"},"
        "\"name\":\"get_weather\"}\n</tool_call>";
    for (const int32_t id : Encode(wrapped)) {
      REQUIRE(g->accept_tokens("r", {id}));
    }
    CHECK(g->is_terminated());
    CHECK(Allowed(*g, kEos));
  }

  // BARE JSON (no wrapper) is REJECTED at the very first byte: the grammar
  // requires '<', so '{' cannot be accepted.
  {
    auto g = CompileGrammar(*backend, gbnf);
    const int32_t brace = static_cast<int32_t>(kChars.find('{'));
    CHECK_FALSE(g->accept_tokens("r", {brace}));
  }
}

// ── (c): the wrapped output round-trips through the non-stream parser ─────────
TEST_CASE("ToolChoice grammar: extract_tool_calls extracts the forced call") {
  const json params = {{"type", "object"},
                       {"properties", {{"city", {{"type", "string"}}}}},
                       {"required", json::array({"city"})}};
  const json schema = ToolCallObjectSchema("get_weather", params);
  const std::string gbnf = WrapSchemaAsToolCallGbnf(schema);

  // Confirm the grammar actually produces this exact wrapped byte string by
  // driving the native FSM over it (done in test (a)); here assert the parser
  // extracts the tool call from that same wrapped output.
  const std::string wrapped =
      "<tool_call>\n{\"arguments\":{\"city\":\"Paris\"},"
      "\"name\":\"get_weather\"}\n</tool_call>";

  HermesToolParser parser;
  ChatCompletionRequest req;
  auto info = parser.extract_tool_calls(wrapped, req);

  CHECK(info.tools_called == true);
  REQUIRE(info.tool_calls.size() == 1);
  CHECK(info.tool_calls[0].type == "function");
  CHECK(info.tool_calls[0].function.name == "get_weather");
  CHECK(info.tool_calls[0].function.arguments == "{\"city\":\"Paris\"}");
  CHECK_FALSE(info.content.has_value());  // no leading text before the wrapper.

  // Contrast: the BARE JSON the OLD (buggy) json-constraint would have emitted
  // is DROPPED by the parser (the bug this fix closes).
  const std::string bare = "{\"name\":\"get_weather\",\"arguments\":"
                           "{\"city\":\"Paris\"}}";
  auto bare_info = parser.extract_tool_calls(bare, req);
  CHECK(bare_info.tools_called == false);
  REQUIRE(bare_info.content.has_value());
  CHECK(*bare_info.content == bare);  // returned as plain content — the bug.
}

// ── (d): serving level — a forced (named) request yields tool_calls +
//         finish_reason="tool_calls" when shaping the WRAPPED constrained output.
TEST_CASE("ToolChoice grammar: serving shapes wrapped output as a tool call") {
  ChatCompletionRequest req;
  ChatMessage user;
  user.role = "user";
  user.content = "weather in Paris?";
  req.messages.push_back(user);

  ChatCompletionToolsParam tool;
  tool.type = "function";
  tool.function.name = "get_weather";
  tool.function.parameters =
      json{{"type", "object"},
           {"properties", {{"city", {{"type", "string"}}}}},
           {"required", json::array({"city"})}};
  req.tools = std::vector<ChatCompletionToolsParam>{tool};
  ToolChoice tc;
  tc.mode = "function";
  tc.function_name = "get_weather";
  req.tool_choice = tc;

  // The grammar the serving path would wire (asserts the forced-schema -> wrapped
  // grammar step compiles + constrains the exact wrapped output).
  const auto forced = ToolChoiceForcedSchema(req);
  REQUIRE(forced.has_value());
  const std::string gbnf = WrapSchemaAsToolCallGbnf(*forced);
  auto backend = MakeBackend();
  const std::string wrapped =
      "<tool_call>\n{\"arguments\":{\"city\":\"Paris\"},"
      "\"name\":\"get_weather\"}\n</tool_call>";
  {
    auto g = CompileGrammar(*backend, gbnf);
    for (const int32_t id : Encode(wrapped)) {
      REQUIRE(g->accept_tokens("r", {id}));
    }
    CHECK(g->is_terminated());
  }

  // And ApplyToolChoiceStructuredOutput routes it through structured_outputs
  // .grammar (NOT .json), passing the one-constraint Verify().
  SamplingParams sp;
  ApplyToolChoiceStructuredOutput(req, sp);
  REQUIRE(sp.structured_outputs.has_value());
  CHECK(sp.structured_outputs->grammar.has_value());
  CHECK_FALSE(sp.structured_outputs->json.has_value());

  // Shaping the wrapped constrained output → a real tool call + "tool_calls".
  HermesToolParser parser;
  ShapedChatMessage shaped =
      ShapeChatMessage("assistant", wrapped, std::string("stop"), req, &parser);
  REQUIRE(shaped.message.tool_calls.has_value());
  REQUIRE(shaped.message.tool_calls->size() == 1);
  CHECK((*shaped.message.tool_calls)[0].function.name == "get_weather");
  CHECK((*shaped.message.tool_calls)[0].function.arguments ==
        "{\"city\":\"Paris\"}");
  REQUIRE(shaped.finish_reason.has_value());
  CHECK(*shaped.finish_reason == "tool_calls");
}
