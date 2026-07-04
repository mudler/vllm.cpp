// M3.3b Task 3: tool_choice -> STRUCTURAL-TAG, exercised END-TO-END over the
// NATIVE backend (the real lazy/forced grammar seam).
//
// ApplyToolChoiceStructuredOutput now sets structured_outputs.structural_tag (the
// native kStructuralTag compile), NOT the old forced-json grammar. tool_choice
// maps (mirror get_hermes_structural_tag, structural_tag_registry.py:237-269):
//   auto     -> a LAZY tag (inert until `<tool_call>`; a plain reply is free) —
//               the user-facing payoff: auto no longer FORCES a tool call.
//   required -> forced from token 0 (>=1 tag).
//   named    -> forced exactly one tag.
//
// The load-bearing proof (the exact user concern): for tool_choice=auto, compile
// the structural_tag over the real native backend and show
//   (a) BEFORE `<tool_call>` every token is allowed (a free reply is possible,
//       including EOS — the model may just answer), and
//   (b) AFTER the `<tool_call>` trigger the tool-call JSON is CONSTRAINED (a
//       random letter is rejected, EOS is rejected — a call must complete), and
//   (c) a never-triggering reply stays free and can end (NOT forced), while
//   (d) a full `<tool_call>{...}</tool_call>` call is accepted + terminates and
//       round-trips through the Hermes parser.
// Plus: required/named compile FORCED (first byte must be `<`, bare `{`/letters
// rejected), and ApplyToolChoiceStructuredOutput routes to .structural_tag only.
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

// Compile a structural-tag spec (the JSON dumped by ToolChoiceStructuralTagSpec)
// through the NATIVE kStructuralTag path — the real serving seam.
std::unique_ptr<NativeGrammar> CompileStructuralTag(
    NativeStructuredOutputBackend& backend, const std::string& spec) {
  std::unique_ptr<StructuredOutputGrammar> g =
      backend.compile_grammar(StructuredOutputOptions::kStructuralTag, spec);
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

// A one-tool chat request (get_weather{city:string}) at the given tool_choice.
ChatCompletionRequest WeatherRequest(std::optional<ToolChoice> choice) {
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
  req.tool_choice = std::move(choice);
  return req;
}

}  // namespace

// ── (g) auto -> LAZY: inert before `<tool_call>`, constrained after it ────────
TEST_CASE("ToolChoice structural tag: auto is LAZY (free until <tool_call>)") {
  ChatCompletionRequest req = WeatherRequest(ToolChoice{"auto", std::nullopt});
  const auto spec = ToolChoiceStructuralTagSpec(req);
  REQUIRE(spec.has_value());
  REQUIRE(spec->at("lazy") == true);
  const std::string spec_str = spec->dump();

  auto backend = MakeBackend();

  // (a) BEFORE the trigger the grammar is inert — EVERY token is allowed,
  // including a bare '{', a plain letter, AND EOS (the model may just reply).
  {
    auto g = CompileStructuralTag(*backend, spec_str);
    CHECK(Allowed(*g, static_cast<int32_t>(kChars.find('z'))));  // free text
    CHECK(Allowed(*g, static_cast<int32_t>(kChars.find('{'))));  // free '{'
    CHECK(Allowed(*g, static_cast<int32_t>(kChars.find('<'))));  // free '<'
    CHECK(Allowed(*g, kEos));  // a plain reply may END — NOT forced to call.
  }

  // (b) AFTER feeding the `<tool_call>` trigger the tag body is CONSTRAINED: the
  // next byte must open one of the two surface variants (`\n` or `{`); a random
  // letter is rejected, and EOS is rejected (a started call must complete).
  {
    auto g = CompileStructuralTag(*backend, spec_str);
    for (const int32_t id : Encode("<tool_call>")) {
      REQUIRE(g->accept_tokens("r", {id}));  // free text while awaiting.
    }
    CHECK(Allowed(*g, static_cast<int32_t>(kChars.find('\n'))));  // variant 0
    CHECK(Allowed(*g, static_cast<int32_t>(kChars.find('{'))));   // variant 1
    CHECK_FALSE(Allowed(*g, static_cast<int32_t>(kChars.find('z'))));  // no free
    CHECK_FALSE(Allowed(*g, kEos));  // must complete the call now.
  }

  // (c) a reply that NEVER triggers stays free and can END (not forced).
  {
    auto g = CompileStructuralTag(*backend, spec_str);
    for (const int32_t id : Encode("it is sunny today")) {
      REQUIRE(g->accept_tokens("r", {id}));
    }
    CHECK(Allowed(*g, static_cast<int32_t>(kChars.find('a'))));  // still free
    CHECK(Allowed(*g, kEos));                                    // may end
  }

  // (d) the model CHOSE to call: a full `<tool_call>\n{...}\n</tool_call>` is
  // accepted char-by-char; the call may then END (EOS allowed — auto's tag list
  // is one-or-more, so the grammar is accepting but may also start another tag).
  // The Hermes parser extracts it.
  {
    auto g = CompileStructuralTag(*backend, spec_str);
    const std::string call =
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
        "{\"city\":\"Paris\"}}\n</tool_call>";
    for (const int32_t id : Encode(call)) {
      REQUIRE(g->accept_tokens("r", {id}));
    }
    CHECK(Allowed(*g, kEos));  // the call is complete — the model may stop.

    HermesToolParser parser;
    ChatCompletionRequest parse_req;
    auto info = parser.extract_tool_calls(call, parse_req);
    CHECK(info.tools_called == true);
    REQUIRE(info.tool_calls.size() == 1);
    CHECK(info.tool_calls[0].function.name == "get_weather");
    CHECK(info.tool_calls[0].function.arguments == "{\"city\":\"Paris\"}");
  }
}

// ── auto routes onto SamplingParams as a LAZY structural_tag (not grammar/json)
TEST_CASE("ToolChoice structural tag: auto flows to structured_outputs.structural_tag") {
  ChatCompletionRequest req = WeatherRequest(ToolChoice{"auto", std::nullopt});
  SamplingParams sp;
  ApplyToolChoiceStructuredOutput(req, sp);
  REQUIRE(sp.structured_outputs.has_value());
  REQUIRE(sp.structured_outputs->structural_tag.has_value());
  CHECK_FALSE(sp.structured_outputs->grammar.has_value());
  CHECK_FALSE(sp.structured_outputs->json.has_value());
  CHECK(json::parse(*sp.structured_outputs->structural_tag).at("lazy") == true);
}

// ── required -> FORCED from token 0: first byte must be `<`, bare JSON rejected
TEST_CASE("ToolChoice structural tag: required is forced (first byte '<')") {
  ChatCompletionRequest req =
      WeatherRequest(ToolChoice{"required", std::nullopt});
  const auto spec = ToolChoiceStructuralTagSpec(req);
  REQUIRE(spec.has_value());
  REQUIRE(spec->at("lazy") == false);

  auto backend = MakeBackend();
  auto g = CompileStructuralTag(*backend, spec->dump());
  // Forced: the first byte must be the `<tool_call>` wrapper — a bare '{' or a
  // letter is REJECTED, and EOS is rejected (>=1 call required).
  CHECK(Allowed(*g, static_cast<int32_t>(kChars.find('<'))));
  CHECK_FALSE(Allowed(*g, static_cast<int32_t>(kChars.find('{'))));
  CHECK_FALSE(Allowed(*g, static_cast<int32_t>(kChars.find('z'))));
  CHECK_FALSE(Allowed(*g, kEos));
  // A bare '{' (the old buggy unwrapped JSON) cannot even be accepted.
  CHECK_FALSE(g->accept_tokens("r", {static_cast<int32_t>(kChars.find('{'))}));
}

// ── named -> FORCED exactly one; the wrapped call is accepted + parser-extracted
TEST_CASE("ToolChoice structural tag: named forces one wrapped call") {
  ChatCompletionRequest req =
      WeatherRequest(ToolChoice{"function", std::string("get_weather")});
  const auto spec = ToolChoiceStructuralTagSpec(req);
  REQUIRE(spec.has_value());
  CHECK(spec->at("lazy") == false);
  CHECK(spec->at("stop_after_first") == true);

  auto backend = MakeBackend();
  auto g = CompileStructuralTag(*backend, spec->dump());
  // Forced (first byte '<'), and the compact-variant call is accepted + ends.
  CHECK(Allowed(*g, static_cast<int32_t>(kChars.find('<'))));
  CHECK_FALSE(Allowed(*g, static_cast<int32_t>(kChars.find('{'))));
  const std::string call =
      "<tool_call>{\"name\": \"get_weather\", \"arguments\": "
      "{\"city\":\"Paris\"}}</tool_call>";
  for (const int32_t id : Encode(call)) {
    REQUIRE(g->accept_tokens("r", {id}));
  }
  CHECK(g->is_terminated());

  // ApplyToolChoiceStructuredOutput + the serving shaping close the loop.
  SamplingParams sp;
  ApplyToolChoiceStructuredOutput(req, sp);
  REQUIRE(sp.structured_outputs.has_value());
  REQUIRE(sp.structured_outputs->structural_tag.has_value());
  CHECK_FALSE(sp.structured_outputs->grammar.has_value());

  HermesToolParser parser;
  ShapedChatMessage shaped =
      ShapeChatMessage("assistant", call, std::string("stop"), req, &parser);
  REQUIRE(shaped.message.tool_calls.has_value());
  REQUIRE(shaped.message.tool_calls->size() == 1);
  CHECK((*shaped.message.tool_calls)[0].function.name == "get_weather");
  REQUIRE(shaped.finish_reason.has_value());
  CHECK(*shaped.finish_reason == "tool_calls");
}
