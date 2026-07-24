// Per-family STRUCTURAL-TAG registry (structural_tags.h). Proves tool_choice
// forces each model family's NATIVE tool syntax instead of always Hermes.
//
// For EACH family with a spec (hermes, qwen3, longcat, llama3_json/llama4_json,
// deepseek_v3, deepseek_v31, mistral) this suite asserts:
//   (a) the auto spec is LAZY with the family's OWN begin-marker trigger,
//   (b) the named spec bakes the tool name into a begin literal,
//   (c) the compiled (named/forced) grammar - driven through the SAME native
//       kStructuralTag compile path test_tool_choice_grammar uses - ACCEPTS a
//       canonical native-syntax call for that family and TERMINATES, and
//   (d) for the non-Hermes families, the compiled grammar REJECTS a Hermes
//       `<tool_call>{...}</tool_call>` call - proving the old wrong-forcing
//       (Hermes syntax onto every dialect) is gone.
// Plus: the NULLOPT families (deepseek_v32/v4 DSML, pythonic/xlam/granite/... )
// are asserted to yield NO constraint for auto/required/named (documented in
// structural_tags.h), and the Apply* path routes only structural_tag.
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
#include "vllm/entrypoints/openai/tool_parsers/structural_tags.h"
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

// EOS lives just past the 256 single-byte tokens (0..255).
constexpr int32_t kEos = 256;

class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_structtags_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// A byte-level fixture whose vocab is ALL 256 single-byte tokens (id == byte
// value), so any UTF-8 marker (the fullwidth ｜ / ▁ bytes of the DeepSeek
// markers, the `[TOOL_CALLS]` / `<tool_call>` ASCII, etc.) is a sequence of
// byte tokens. Encode() maps each string byte to its id directly.
Tokenizer BuildFixture() {
  json vocab = json::object();
  for (int b = 0; b < 256; ++b) {
    vocab[MapBytesToUnicode(std::string(1, static_cast<char>(b)))] = b;
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
  ids.reserve(text.size());
  for (const char ch : text) {
    ids.push_back(static_cast<int32_t>(static_cast<unsigned char>(ch)));
  }
  return ids;
}

std::unique_ptr<NativeStructuredOutputBackend> MakeBackend() {
  return std::make_unique<NativeStructuredOutputBackend>(
      Fixture(), VocabSize(), std::vector<int32_t>{kEos});
}

std::unique_ptr<NativeGrammar> CompileStructuralTag(
    NativeStructuredOutputBackend& backend, const std::string& spec) {
  std::unique_ptr<StructuredOutputGrammar> g =
      backend.compile_grammar(StructuredOutputOptions::kStructuralTag, spec);
  return std::unique_ptr<NativeGrammar>(
      static_cast<NativeGrammar*>(g.release()));
}

// Feed every byte of `text` as a token; returns false the moment one is
// rejected (used to prove a family's grammar rejects a foreign-syntax call).
bool AcceptAll(NativeGrammar& g, const std::string& text) {
  for (const int32_t id : Encode(text)) {
    if (!g.accept_tokens("r", {id})) return false;
  }
  return true;
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

// A canonical Hermes call - the foreign syntax that non-Hermes families reject.
const char* kHermesCall =
    "<tool_call>{\"name\": \"get_weather\", \"arguments\": "
    "{\"city\":\"Paris\"}}</tool_call>";

// Drive one family end-to-end: (a) auto lazy+trigger, (b) named bakes the name,
// (c) the named grammar accepts the canonical native call + terminates.
struct FamilyCase {
  std::string parser;         // the tool_parser_name
  std::string trigger;        // expected auto trigger marker
  std::string canonical_call; // a native-syntax get_weather{city:Paris} call
  bool non_hermes;            // if true, assert it REJECTS the Hermes call
};

void RunFamily(const FamilyCase& fc) {
  CAPTURE(fc.parser);

  // (a) auto -> LAZY with the family's own begin-marker trigger.
  {
    ChatCompletionRequest req =
        WeatherRequest(ToolChoice{"auto", std::nullopt});
    const auto spec = ToolChoiceStructuralTagSpecFor(fc.parser, req);
    REQUIRE(spec.has_value());
    CHECK(spec->at("lazy") == true);
    REQUIRE(spec->at("triggers").is_array());
    REQUIRE(spec->at("triggers").size() == 1);
    CHECK(spec->at("triggers")[0].get<std::string>() == fc.trigger);
  }

  // (b) named -> forced exactly one, tool name baked into a begin literal.
  const auto named =
      ToolChoiceStructuralTagSpecFor(
          fc.parser, WeatherRequest(ToolChoice{"function",
                                               std::string("get_weather")}));
  REQUIRE(named.has_value());
  CHECK(named->at("lazy") == false);
  CHECK(named->at("stop_after_first") == true);
  REQUIRE(named->at("tags").is_array());
  REQUIRE(!named->at("tags").empty());
  const std::string first_begin =
      named->at("tags")[0].at("begin").get<std::string>();
  CHECK(first_begin.find("get_weather") != std::string::npos);

  // (c) the named grammar ACCEPTS the canonical native call and TERMINATES.
  {
    auto backend = MakeBackend();
    auto g = CompileStructuralTag(*backend, named->dump());
    CHECK(AcceptAll(*g, fc.canonical_call));
    CHECK(g->is_terminated());
  }

  // (d) non-Hermes families REJECT a Hermes-syntax call (old wrong-forcing gone).
  if (fc.non_hermes) {
    auto backend = MakeBackend();
    auto g = CompileStructuralTag(*backend, named->dump());
    CHECK_FALSE(AcceptAll(*g, kHermesCall));
  }
}

}  // namespace

TEST_CASE("structural tags: hermes family (native <tool_call> surface)") {
  RunFamily({"hermes", "<tool_call>",
             "<tool_call>{\"name\": \"get_weather\", \"arguments\": "
             "{\"city\":\"Paris\"}}</tool_call>",
             /*non_hermes=*/false});
}

TEST_CASE("structural tags: qwen3 family (Hermes-format, reuses <tool_call>)") {
  RunFamily({"qwen3", "<tool_call>",
             "<tool_call>{\"name\": \"get_weather\", \"arguments\": "
             "{\"city\":\"Paris\"}}</tool_call>",
             /*non_hermes=*/false});
}

TEST_CASE("structural tags: longcat family (<longcat_tool_call> wrapper)") {
  RunFamily({"longcat", "<longcat_tool_call>",
             "<longcat_tool_call>{\"name\": \"get_weather\", \"arguments\": "
             "{\"city\":\"Paris\"}}</longcat_tool_call>",
             /*non_hermes=*/true});
}

TEST_CASE("structural tags: llama3_json family (bare {name,parameters})") {
  RunFamily({"llama3_json", "{\"name\": \"",
             "{\"name\": \"get_weather\", \"parameters\": {\"city\":\"Paris\"}}",
             /*non_hermes=*/true});
}

TEST_CASE("structural tags: llama4_json family (same JSON surface as llama3)") {
  RunFamily({"llama4_json", "{\"name\": \"",
             "{\"name\": \"get_weather\", \"parameters\": {\"city\":\"Paris\"}}",
             /*non_hermes=*/true});
}

TEST_CASE("structural tags: deepseek_v3 family (markers + ```json fence)") {
  RunFamily(
      {"deepseek_v3", "<｜tool▁calls▁begin｜>",
       "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
       "get_weather\n```json\n{\"city\":\"Paris\"}\n```<｜tool▁call▁end｜>"
       "<｜tool▁calls▁end｜>",
       /*non_hermes=*/true});
}

TEST_CASE("structural tags: deepseek_v31 family (markers, args after sep)") {
  RunFamily(
      {"deepseek_v31", "<｜tool▁calls▁begin｜>",
       "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
       "{\"city\":\"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
       /*non_hermes=*/true});
}

TEST_CASE("structural tags: mistral family ([TOOL_CALLS]name{args}, v11)") {
  RunFamily({"mistral", "[TOOL_CALLS]",
             "[TOOL_CALLS]get_weather{\"city\":\"Paris\"}",
             /*non_hermes=*/true});
}

// ── NULLOPT families: no expressible tag surface -> NO constraint (documented) ─
TEST_CASE("structural tags: DSML + non-taggable families are nullopt") {
  // Every mode (auto/required/named) must yield nullopt for a family with no
  // JSON-args begin/end surface (structural_tags.h COVERAGE list).
  const std::vector<std::string> nullopt_families = {
      "deepseek_v32", "deepseek_v4",   "pythonic",     "llama4_pythonic",
      "xlam",         "granite",       "granite4",     "granite-20b-fc",
      "phi4_mini_json", "internlm",    "jamba",        "step3",
      "step3p5",      "minicpm5",      "hy_v3",        "olmo3",
      "",             "not_a_parser",
  };
  for (const std::string& fam : nullopt_families) {
    CAPTURE(fam);
    CHECK_FALSE(
        ToolChoiceStructuralTagSpecFor(
            fam, WeatherRequest(ToolChoice{"auto", std::nullopt}))
            .has_value());
    CHECK_FALSE(
        ToolChoiceStructuralTagSpecFor(
            fam, WeatherRequest(ToolChoice{"required", std::nullopt}))
            .has_value());
    CHECK_FALSE(ToolChoiceStructuralTagSpecFor(
                    fam, WeatherRequest(ToolChoice{
                             "function", std::string("get_weather")}))
                    .has_value());
  }
}

TEST_CASE("structural tags: nullopt family applies NO decode constraint") {
  // A DSML family (deepseek_v4) even with required leaves structured_outputs
  // unset - the model emits its OWN native syntax unconstrained (its parser
  // still extracts it), NOT wrong-forced Hermes.
  ChatCompletionRequest req =
      WeatherRequest(ToolChoice{"required", std::nullopt});
  SamplingParams sp;
  ApplyToolChoiceStructuredOutput("deepseek_v4", req, sp);
  CHECK_FALSE(sp.structured_outputs.has_value());
}

// ── none / no-tools short-circuits, for a mapped family ──────────────────────
TEST_CASE("structural tags: none and no-tools yield nullopt") {
  CHECK_FALSE(ToolChoiceStructuralTagSpecFor(
                  "hermes", WeatherRequest(ToolChoice{"none", std::nullopt}))
                  .has_value());
  ChatCompletionRequest no_tools;
  ChatMessage user;
  user.role = "user";
  user.content = "hi";
  no_tools.messages.push_back(user);
  CHECK_FALSE(
      ToolChoiceStructuralTagSpecFor("mistral", no_tools).has_value());
}

// ── the Apply* path routes exactly structural_tag, per active family ─────────
TEST_CASE("structural tags: Apply routes only structural_tag with family markers") {
  // required for deepseek_v3 -> structural_tag carrying the DeepSeek OUTER
  // marker and NOT the Hermes `<tool_call>` (proof the constraint is native).
  ChatCompletionRequest req =
      WeatherRequest(ToolChoice{"required", std::nullopt});
  SamplingParams sp;
  ApplyToolChoiceStructuredOutput("deepseek_v3", req, sp);
  REQUIRE(sp.structured_outputs.has_value());
  REQUIRE(sp.structured_outputs->structural_tag.has_value());
  CHECK_FALSE(sp.structured_outputs->grammar.has_value());
  CHECK_FALSE(sp.structured_outputs->json.has_value());
  const std::string tag = *sp.structured_outputs->structural_tag;
  CHECK(tag.find("<｜tool▁calls▁begin｜>") != std::string::npos);
  CHECK(tag.find("<tool_call>") == std::string::npos);
}
