// Tests for the M3.4 Task 4 NATIVE grammar engine (backend_native.{h,cpp}),
// an ORIGINAL vllm.cpp component (§9) behind the 1:1-ported StructuredOutput
// seam. Exercises: GBNF (yes/no), CHOICE (cat/dog/bird), char-class ([0-9]+),
// byte-alignment across a token boundary + a multi-byte UTF-8 codepoint, the
// EOS/stop token only at an accepting state, the sub-O(vocab) fill (a trie-node
// visit counter), rollback/reset, validate_tokens, and regex lowering.
//
// The tokenizer is a REAL byte-level BPE fixture (built via Tokenizer::FromHfJson
// like tests/vllm/test_detokenizer.cpp) so token->raw-bytes decoding is real.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/structured_output/backend_native.h"
#include "vllm/v1/structured_output/backend_types.h"

using nlohmann::json;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::BitmaskWordsForVocab;
using vllm::v1::NativeGrammar;
using vllm::v1::NativeStructuredOutputBackend;
using vllm::v1::StructuredOutputGrammar;
using vllm::v1::StructuredOutputOptions;
using vllm::v1::TokenBitmask;

namespace {

using Ids = std::vector<int32_t>;

class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_native_grammar_test_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// The EOS/stop token id used across tests (an added, special token).
constexpr int32_t kEos = 40;
// Number of junk tokens inflating the vocab (for the perf property).
constexpr int kJunk = 300;

// A real byte-level BPE tokenizer whose vocab is designed for the grammar
// tests. All hand-picked tokens are ASCII (so the byte-map is identity) except
// the é fragments; junk tokens all start with 'q' so a "yes"/"[0-9]"-anchored
// grammar never descends into them (the perf property).
Tokenizer BuildFixture() {
  json vocab = {
      {"y", 0},    {"e", 1},   {"s", 2},    {"n", 3},   {"o", 4},
      {"ye", 5},   {"es", 6},  {"yes", 7},  {"no", 8},  {"c", 9},
      {"a", 10},   {"t", 11},  {"d", 12},   {"g", 13},  {"b", 14},
      {"i", 15},   {"r", 16},  {"cat", 17}, {"dog", 18}, {"bird", 19},
      {"x", 20},   {"z", 21},  {"0", 22},   {"1", 23},   {"5", 24},
      {"9", 25},   {"12", 26}, {"ab", 27},  {"!", 31}};
  // é = C3 A9; its two byte-fragments; used for multi-byte grammar matching.
  vocab[MapBytesToUnicode("\xC3\xA9")] = 28;  // full é
  vocab[MapBytesToUnicode("\xC3")] = 29;      // lead byte
  vocab[MapBytesToUnicode("\xA9")] = 30;      // trailing byte
  // Junk tokens q0..q(kJunk-1): never a prefix of the tested grammars.
  for (int i = 0; i < kJunk; ++i) {
    vocab["q" + std::to_string(i)] = 100 + i;
  }

  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", kEos}, {"content", "<eos>"}, {"special", true}},
       {{"id", 41}, {"content", "<tool>"}, {"special", false}}});
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

std::unique_ptr<NativeStructuredOutputBackend> MakeBackend() {
  return std::make_unique<NativeStructuredOutputBackend>(
      Fixture(), VocabSize(), std::vector<int32_t>{kEos});
}

// Compiles a grammar and returns it as the concrete type (so tests can read the
// perf counter).
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

}  // namespace

// (a) GBNF: root ::= "yes" | "no".
TEST_CASE("native GBNF: yes/no allows only valid prefixes, terminates on match") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes" | "no")");

  // At the start only tokens whose bytes are a valid prefix of "yes"/"no".
  CHECK(Allowed(*g, 0));   // "y"
  CHECK(Allowed(*g, 5));   // "ye"
  CHECK(Allowed(*g, 7));   // "yes"
  CHECK(Allowed(*g, 3));   // "n"
  CHECK(Allowed(*g, 8));   // "no"
  CHECK_FALSE(Allowed(*g, 2));   // "s" — not a valid first byte
  CHECK_FALSE(Allowed(*g, 20));  // "x" — forbidden
  CHECK_FALSE(Allowed(*g, 6));   // "es"
  CHECK_FALSE(Allowed(*g, kEos));  // EOS not allowed before a complete match

  CHECK_FALSE(g->is_terminated());
  REQUIRE(g->accept_tokens("r", {7}));  // "yes"
  CHECK(g->is_terminated());
  // After a full match EOS is the only continuation (fill even when terminated).
  CHECK(Allowed(*g, kEos));
}

// (d) byte-alignment across a token boundary: "ye" then "s".
TEST_CASE("native GBNF: FSM advances correctly across a token boundary") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes" | "no")");
  REQUIRE(g->accept_tokens("r", {5}));  // "ye"
  CHECK_FALSE(g->is_terminated());
  // Now only "s" continues; "yes" (starts with 'y') no longer valid.
  CHECK(Allowed(*g, 2));         // "s"
  CHECK_FALSE(Allowed(*g, 7));   // "yes"
  CHECK_FALSE(Allowed(*g, 0));   // "y"
  REQUIRE(g->accept_tokens("r", {2}));  // "s"
  CHECK(g->is_terminated());
}

// A token forbidden by the grammar is rejected by accept_tokens.
TEST_CASE("native GBNF: accept_tokens rejects an invalid token") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes")");
  CHECK_FALSE(g->accept_tokens("r", {8}));  // "no"
}

// (b) CHOICE: ["cat","dog","bird"].
TEST_CASE("native CHOICE: only the listed completions") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kChoice,
                   R"(["cat","dog","bird"])");
  CHECK(Allowed(*g, 9));    // "c"
  CHECK(Allowed(*g, 17));   // "cat"
  CHECK(Allowed(*g, 12));   // "d"
  CHECK(Allowed(*g, 18));   // "dog"
  CHECK(Allowed(*g, 14));   // "b"
  CHECK(Allowed(*g, 19));   // "bird"
  CHECK_FALSE(Allowed(*g, 0));   // "y"
  CHECK_FALSE(Allowed(*g, 11));  // "t" — not a valid first byte

  REQUIRE(g->accept_tokens("r", {18}));  // "dog"
  CHECK(g->is_terminated());
}

// (c) char class: root ::= [0-9]+.
TEST_CASE("native char class: [0-9]+ allows digits, forbids letters") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= [0-9]+)");
  // Digit tokens (incl. the multi-digit "12") allowed; letters forbidden.
  CHECK(Allowed(*g, 22));   // "0"
  CHECK(Allowed(*g, 24));   // "5"
  CHECK(Allowed(*g, 26));   // "12"
  CHECK_FALSE(Allowed(*g, 20));  // "x"
  CHECK_FALSE(Allowed(*g, 27));  // "ab"
  CHECK_FALSE(Allowed(*g, kEos));  // + requires >= 1 digit before EOS

  REQUIRE(g->accept_tokens("r", {24}));  // "5"
  CHECK_FALSE(g->is_terminated());       // "+" can still repeat
  CHECK(Allowed(*g, 22));                // another digit
  CHECK(Allowed(*g, kEos));              // now EOS is allowed (accepting)
}

// Multi-byte UTF-8 codepoint literal in the grammar, matched via byte-level
// tokens (both the whole-codepoint token and its byte fragments).
TEST_CASE("native GBNF: multi-byte UTF-8 codepoint literal") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "é" "!")");  // é then !
  CHECK(Allowed(*g, 28));        // full "é" (C3 A9)
  CHECK(Allowed(*g, 29));        // just the C3 lead byte (valid prefix)
  CHECK_FALSE(Allowed(*g, 30));  // A9 alone is not valid first
  CHECK_FALSE(Allowed(*g, 31));  // "!" not valid before é

  SUBCASE("whole codepoint in one token") {
    REQUIRE(g->accept_tokens("r", {28}));  // é
    CHECK(Allowed(*g, 31));                // now "!"
    REQUIRE(g->accept_tokens("r", {31}));
    CHECK(g->is_terminated());
  }
  SUBCASE("codepoint split across two byte tokens") {
    REQUIRE(g->accept_tokens("r", {29}));  // C3
    CHECK(Allowed(*g, 30));                // A9 completes é
    CHECK_FALSE(Allowed(*g, 31));          // "!" not yet
    REQUIRE(g->accept_tokens("r", {30}));  // A9
    CHECK(Allowed(*g, 31));
    REQUIRE(g->accept_tokens("r", {31}));
    CHECK(g->is_terminated());
  }
}

// (e) EOS/stop token allowed only at an accepting state; consuming it
// transitions to terminated and nothing may follow.
TEST_CASE("native: EOS allowed only at an accepting state") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "ab")");  // tokens: "a"=10, "b"=14
  CHECK_FALSE(Allowed(*g, kEos));              // start: not accepting
  CHECK_FALSE(g->accept_tokens("r", {kEos}));  // EOS rejected mid-grammar
  REQUIRE(g->accept_tokens("r", {10}));        // "a"
  CHECK_FALSE(Allowed(*g, kEos));              // still not complete
  REQUIRE(g->accept_tokens("r", {14}));        // "b" -> complete
  CHECK(g->is_terminated());
  CHECK(Allowed(*g, kEos));
  REQUIRE(g->accept_tokens("r", {kEos}));      // consume EOS
  CHECK(g->is_terminated());
  CHECK_FALSE(g->accept_tokens("r", {14}));    // nothing follows EOS
}

// The non-special ADDED token (<tool>) and any excluded id are never matchable.
TEST_CASE("native: added/special tokens are never grammar-matchable") {
  auto backend = MakeBackend();
  // A permissive grammar that would accept the literal bytes "<tool>".
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= [<a-z>]+)");
  CHECK_FALSE(Allowed(*g, 41));  // "<tool>" added token, not in the trie
}

// (f) perf: fill_bitmask visits far fewer trie nodes than the vocab size.
TEST_CASE("native: fill_bitmask is sub-O(vocab) (trie-node visit counter)") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes")");
  TokenBitmask bm;
  bm.num_seqs = 1;
  bm.num_words = BitmaskWordsForVocab(VocabSize());
  bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
  g->fill_bitmask(bm, 0);
  // Only the trie nodes along "y","ye","yes" (+ root) are visited; nothing near
  // the ~330-token vocab (hundreds of junk 'q...' tokens are pruned).
  CHECK(g->last_fill_visited_nodes() < 10);
  CHECK(g->last_fill_visited_nodes() < VocabSize() / 4);
}

// (g) rollback / reset.
TEST_CASE("native: rollback and reset restore prior FSM state") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes")");
  REQUIRE(g->accept_tokens("r", {5}));  // "ye"
  CHECK(Allowed(*g, 2));                // "s" continues
  CHECK_FALSE(Allowed(*g, 0));          // "y" no longer

  g->rollback(1);  // back to start
  CHECK(Allowed(*g, 0));   // "y" allowed again
  CHECK(Allowed(*g, 7));   // "yes" allowed again
  CHECK_FALSE(Allowed(*g, 2));  // "s" not valid at start

  REQUIRE(g->accept_tokens("r", {7}));  // "yes"
  CHECK(g->is_terminated());
  g->reset();
  CHECK_FALSE(g->is_terminated());
  CHECK(Allowed(*g, 0));  // start state again
}

// (h) validate_tokens returns the accepted prefix without advancing.
TEST_CASE("native: validate_tokens returns the accepted prefix") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes")");
  // "ye","s" accepted; then "x" rejected -> prefix stops there.
  const Ids result = g->validate_tokens({5, 2, 20, 0});
  CHECK(result == Ids{5, 2});
  // validate_tokens must NOT have advanced the real FSM.
  CHECK(Allowed(*g, 0));  // still at the start
}

// REGEX lowering: a common pattern.
TEST_CASE("native REGEX: [0-9]+ via regex lowering") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kRegex, R"([0-9]+)");
  CHECK(Allowed(*g, 22));   // "0"
  CHECK(Allowed(*g, 26));   // "12"
  CHECK_FALSE(Allowed(*g, 20));  // "x"
  REQUIRE(g->accept_tokens("r", {26}));  // "12"
  CHECK(Allowed(*g, kEos));
}

TEST_CASE("native REGEX: \\d escape + quantifier") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kRegex, R"(\d{1,3})");
  CHECK(Allowed(*g, 24));   // "5"
  CHECK_FALSE(Allowed(*g, 20));
}

// The factory helper the StructuredOutputManager consumes builds a working
// native backend from (tokenizer, vocab_size).
TEST_CASE("native: MakeNativeBackendFactory builds a working backend") {
  auto factory = vllm::v1::MakeNativeBackendFactory(
      Fixture(), VocabSize(), std::vector<int32_t>{kEos});
  std::unique_ptr<vllm::v1::StructuredOutputBackend> backend = factory();
  REQUIRE(backend != nullptr);

  TokenBitmask mask = backend->allocate_token_bitmask(4);
  CHECK(mask.num_seqs == 4);
  CHECK(mask.num_words == BitmaskWordsForVocab(VocabSize()));

  auto g = backend->compile_grammar(StructuredOutputOptions::kChoice,
                                    R"(["yes","no"])");
  REQUIRE(g != nullptr);
  CHECK(g->accept_tokens("r", {7}));  // "yes"
  CHECK(g->is_terminated());
  backend->destroy();
}

// JSON is deferred to Task 5: compile_grammar throws.
TEST_CASE("native: JSON/json_object throw (deferred to M3.4 Task 5)") {
  auto backend = MakeBackend();
  CHECK_THROWS(backend->compile_grammar(StructuredOutputOptions::kJson,
                                        R"({"type":"object"})"));
  CHECK_THROWS(backend->compile_grammar(StructuredOutputOptions::kJsonObject,
                                        ""));
}
