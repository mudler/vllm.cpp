// Tests for the M3.4 Task 5 JSON-schema -> GBNF lowering (json_schema_to_gbnf.
// {h,cpp}), an ORIGINAL vllm.cpp component (§9), plus the native backend's
// kJson / kJsonObject compile paths that consume it.
//
// The tokenizer is a REAL byte-level BPE fixture whose vocab is single-character
// ASCII tokens covering every byte needed to spell JSON ({ } [ ] " : , the
// letters/digits used, space) so a JSON string can be fed to the native grammar
// one character-token at a time, and the accept/reject decision is the real
// byte-level FSM decision. Cases: (a) object with required props matches valid /
// rejects malformed-missing-required-wrong-type; (b) json_object accepts any
// valid JSON, rejects invalid; (c) enum -> only the enum values; (d) nested
// object/array.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/structured_output/backend_native.h"
#include "vllm/v1/structured_output/backend_types.h"
#include "vllm/v1/structured_output/json_schema_to_gbnf.h"

using nlohmann::json;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::BitmaskWordsForVocab;
using vllm::v1::JsonObjectGbnf;
using vllm::v1::JsonSchemaToGbnf;
using vllm::v1::NativeGrammar;
using vllm::v1::NativeStructuredOutputBackend;
using vllm::v1::StructuredOutputGrammar;
using vllm::v1::StructuredOutputOptions;
using vllm::v1::TokenBitmask;

namespace {

// The characters the fixture assigns single-char tokens to (one token per byte).
// Covers JSON structure + every letter/digit used by the test schemas/values.
const std::string kChars =
    "{}[]\":,. -0123456789abcdefghijklmnopqrstuvwxyz";

// The EOS/stop token id (an added, special token past every char token).
constexpr int32_t kEos = 200;

class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_json_gbnf_test_" + std::to_string(counter++) + ".json"))
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
    // The vocab content is the byte-level (GPT-2 bytes_to_unicode) encoding of
    // the raw character — identity for printable ASCII 0x21..0x7E, but space
    // (0x20) maps to "Ġ". Storing the raw byte for space would make the
    // tokenizer's inverse byte-map exclude it from the grammar trie.
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

// Maps a JSON text to the per-character token id stream (each char is its own
// token). Every char must be in kChars.
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

std::unique_ptr<NativeGrammar> Compile(NativeStructuredOutputBackend& backend,
                                       StructuredOutputOptions type,
                                       const std::string& spec) {
  std::unique_ptr<StructuredOutputGrammar> g =
      backend.compile_grammar(type, spec);
  return std::unique_ptr<NativeGrammar>(
      static_cast<NativeGrammar*>(g.release()));
}

// Whether token id `t` is allowed by a freshly filled single-row bitmask.
bool Allowed(NativeGrammar& g, int32_t t) {
  TokenBitmask bm;
  bm.num_seqs = 1;
  bm.num_words = BitmaskWordsForVocab(VocabSize());
  bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
  g.fill_bitmask(bm, 0);
  const int32_t word = bm.data[static_cast<std::size_t>(t >> 5)];
  return (word & (1 << (t & 31))) != 0;
}

// Feed `text` char-by-char through a FRESH grammar for `spec`. Returns true iff
// every character token was accepted AND the grammar is at an accepting state
// afterwards (EOS allowed) — i.e. `text` is a complete schema-valid document.
bool Matches(NativeStructuredOutputBackend& backend, StructuredOutputOptions type,
             const std::string& spec, const std::string& text) {
  auto g = Compile(backend, type, spec);
  for (const int32_t id : Encode(text)) {
    if (!g->accept_tokens("r", {id})) return false;
  }
  return Allowed(*g, kEos);  // accepting state == a complete value.
}

}  // namespace

// ─── (a) object with required properties ─────────────────────────────────────
// schema: {type:object, properties:{name:string, age:integer}, required:[name,age]}
// nlohmann sorts keys -> canonical order is age, then name.
TEST_CASE("JsonSchemaToGbnf: object required props match valid, reject malformed") {
  auto backend = MakeBackend();
  const std::string schema =
      R"({"type":"object","properties":{"name":{"type":"string"},)"
      R"("age":{"type":"integer"}},"required":["name","age"]})";

  // A valid object (canonical age-before-name order).
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema,
                R"({"age":30,"name":"bob"})"));
  // Whitespace flexibility between tokens.
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema,
                R"({ "age" : 7 , "name" : "z" })"));

  // Missing a required field -> rejected.
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({"age":30})"));
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({"name":"bob"})"));
  // Wrong type: age as a string -> rejected.
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({"age":"30","name":"bob"})"));
  // Wrong type: name as a number -> rejected.
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({"age":30,"name":5})"));
  // Empty object -> missing required -> rejected.
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({})"));
  // Unknown extra key -> rejected (closed object).
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({"age":30,"name":"bob","x":1})"));

  // The grammar constrains the FIRST byte to '{' (nothing else can start).
  auto g = Compile(*backend, StructuredOutputOptions::kJson, schema);
  CHECK(Allowed(*g, static_cast<int32_t>(kChars.find('{'))));
  CHECK_FALSE(Allowed(*g, static_cast<int32_t>(kChars.find('['))));
  CHECK_FALSE(Allowed(*g, kEos));  // an empty stream is not a complete object.
}

// An optional property may be present or absent.
TEST_CASE("JsonSchemaToGbnf: optional property present-or-absent") {
  auto backend = MakeBackend();
  const std::string schema =
      R"({"type":"object","properties":{"a":{"type":"integer"},)"
      R"("b":{"type":"boolean"}},"required":["a"]})";
  // b present.
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema,
                R"({"a":1,"b":true})"));
  // b absent (optional).
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema, R"({"a":1})"));
  // a absent (required) -> rejected.
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({"b":false})"));
}

// A property-less object schema -> any object.
TEST_CASE("JsonSchemaToGbnf: property-less object accepts any object") {
  auto backend = MakeBackend();
  const std::string schema = R"({"type":"object"})";
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema, R"({})"));
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema,
                R"({"anything":123})"));
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema, "5"));
}

// ─── (b) json_object: any valid JSON accepted, invalid rejected ──────────────
TEST_CASE("JsonObjectGbnf: any valid JSON accepted, invalid rejected") {
  auto backend = MakeBackend();
  const std::string kEmpty;  // json_object spec string is empty.
  const StructuredOutputOptions kObj = StructuredOutputOptions::kJsonObject;

  CHECK(Matches(*backend, kObj, kEmpty, R"({"a":1})"));
  CHECK(Matches(*backend, kObj, kEmpty, R"([1,2,3])"));
  CHECK(Matches(*backend, kObj, kEmpty, R"("hello")"));
  CHECK(Matches(*backend, kObj, kEmpty, R"(true)"));
  CHECK(Matches(*backend, kObj, kEmpty, R"(null)"));
  CHECK(Matches(*backend, kObj, kEmpty, R"(-12.5)"));
  CHECK(Matches(*backend, kObj, kEmpty, R"({"a":[1,{"b":null}],"c":true})"));

  // Malformed JSON -> rejected.
  CHECK_FALSE(Matches(*backend, kObj, kEmpty, R"({"a":})"));   // no value
  CHECK_FALSE(Matches(*backend, kObj, kEmpty, R"([1,])"));     // trailing comma
  CHECK_FALSE(Matches(*backend, kObj, kEmpty, R"({"a" 1})"));  // missing colon
  CHECK_FALSE(Matches(*backend, kObj, kEmpty, R"(tru)"));      // partial literal
}

// ─── (c) enum -> only the enum values ────────────────────────────────────────
TEST_CASE("JsonSchemaToGbnf: enum accepts only the listed values") {
  auto backend = MakeBackend();
  const std::string schema = R"({"enum":["red","green","blue"]})";
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema, R"("red")"));
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema, R"("green")"));
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema, R"("blue")"));
  // A value not in the enum -> rejected.
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"("yellow")"));
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"("re")"));

  // A mixed-type enum (string / integer / boolean).
  const std::string mixed = R"({"enum":["a",7,true]})";
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, mixed, R"("a")"));
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, mixed, R"(7)"));
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, mixed, R"(true)"));
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, mixed, R"(8)"));
}

// const -> the single value.
TEST_CASE("JsonSchemaToGbnf: const matches only that value") {
  auto backend = MakeBackend();
  const std::string schema = R"({"const":"fixed"})";
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema, R"("fixed")"));
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"("other")"));
}

// ─── (d) nested object / array ───────────────────────────────────────────────
TEST_CASE("JsonSchemaToGbnf: nested object and typed array") {
  auto backend = MakeBackend();
  // {type:object, properties:{items:{type:array, items:{type:integer}},
  //                           meta:{type:object, properties:{ok:{type:boolean}},
  //                                 required:[ok]}},
  //  required:[items,meta]}
  const std::string schema =
      R"({"type":"object","properties":{)"
      R"("items":{"type":"array","items":{"type":"integer"}},)"
      R"("meta":{"type":"object","properties":{"ok":{"type":"boolean"}},)"
      R"("required":["ok"]}},"required":["items","meta"]})";
  // Canonical key order: items, meta.
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema,
                R"({"items":[1,2,3],"meta":{"ok":true}})"));
  // Empty array is valid.
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema,
                R"({"items":[],"meta":{"ok":false}})"));
  // A non-integer array element -> rejected.
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({"items":[1,"x"],"meta":{"ok":true}})"));
  // Nested required field missing -> rejected.
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({"items":[1],"meta":{}})"));
}

// A typed top-level array.
TEST_CASE("JsonSchemaToGbnf: top-level typed array") {
  auto backend = MakeBackend();
  const std::string schema =
      R"({"type":"array","items":{"type":"string"}})";
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema,
                R"(["a","bc"])"));
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema, R"([])"));
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"([1])"));  // integer, not string
}

// ─── (f) anyOf / oneOf -> alternation (M3.3 Task 4: tool_choice=required) ─────
// anyOf of two DISJOINT object schemas (distinct `name` const) — the shape
// tool_choice=required forces over multiple tools.
TEST_CASE("JsonSchemaToGbnf: anyOf accepts either alternative, rejects neither") {
  auto backend = MakeBackend();
  const std::string schema =
      R"({"anyOf":[)"
      R"({"type":"object","properties":{"name":{"const":"a"},)"
      R"("v":{"type":"integer"}},"required":["name","v"]},)"
      R"({"type":"object","properties":{"name":{"const":"b"},)"
      R"("v":{"type":"string"}},"required":["name","v"]}]})";
  // nlohmann sorts keys -> canonical order name, then v.
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema,
                R"({"name":"a","v":7})"));    // alt A
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema,
                R"({"name":"b","v":"x"})"));  // alt B
  // Neither alternative matches: wrong const, or A's name with B's value type.
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({"name":"c","v":7})"));
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({"name":"a","v":"x"})"));
}

// oneOf lowers to the same alternation (primitive alternatives here).
TEST_CASE("JsonSchemaToGbnf: oneOf lowers to an alternation") {
  auto backend = MakeBackend();
  const std::string schema =
      R"({"oneOf":[{"type":"integer"},{"type":"boolean"}]})";
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema, R"(42)"));
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema, R"(true)"));
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema, R"("s")"));
}

// A nested anyOf (a property value that is a union of two types).
TEST_CASE("JsonSchemaToGbnf: nested anyOf inside a property") {
  auto backend = MakeBackend();
  const std::string schema =
      R"({"type":"object","properties":{)"
      R"("x":{"anyOf":[{"type":"integer"},{"type":"string"}]}},)"
      R"("required":["x"]})";
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema, R"({"x":5})"));
  CHECK(Matches(*backend, StructuredOutputOptions::kJson, schema, R"({"x":"hi"})"));
  CHECK_FALSE(Matches(*backend, StructuredOutputOptions::kJson, schema,
                      R"({"x":true})"));
}

// An unsupported schema construct throws (loud, not silent mis-constraint).
TEST_CASE("JsonSchemaToGbnf: unsupported constructs throw") {
  CHECK_THROWS(JsonSchemaToGbnf(json::parse(R"({"allOf":[{"type":"string"}]})")));
  CHECK_THROWS(JsonSchemaToGbnf(json::parse(R"({"not":{"type":"string"}})")));
  CHECK_THROWS(JsonSchemaToGbnf(json::parse(R"({"$ref":"#/defs/x"})")));
  CHECK_THROWS(JsonSchemaToGbnf(json::parse(R"({"type":"widget"})")));
  // anyOf/oneOf must be a non-empty array.
  CHECK_THROWS(JsonSchemaToGbnf(json::parse(R"({"anyOf":[]})")));
  // The native backend surfaces a schema that is not valid JSON as a throw.
  auto backend = MakeBackend();
  CHECK_THROWS(backend->compile_grammar(StructuredOutputOptions::kJson,
                                        "{not json"));
  // A `required` key not declared in `properties` would be silently unenforced
  // (over-permit) — must throw rather than mis-constrain.
  CHECK_THROWS(JsonSchemaToGbnf(json::parse(
      R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["z"]})")));
}
