// LOAD-SENTENCEPIECE — the SentencePiece (Metaspace + byte-fallback) BPE
// tokenizer parity gate. Mistral-7B-v0.3's tokenizer.json is the first of this
// family in the tree: a `Metaspace` pre_tokenizer (spaces -> U+2581,
// prepend_scheme="first", split=false), a byte-fallback vocab (`<0xNN>` tokens),
// and a `TemplateProcessing` post-processor prepending BOS=1. Goldens
// (tests/parity/goldens/tokenizer_mistral/) are produced by the REAL HF
// `tokenizers` library via tools/parity/dump_tokenizer_mistral.py (the exact
// tokenizer.json is committed alongside, so oracle and this test parse identical
// bytes; vLLM 0.25.0 tokenizes through the same HF backend). Per corpus entry:
//   - Encode(text)                 == golden ids           (no special tokens)
//   - EncodeWithSpecialTokens(text) == golden ids_special  (BOS=1 prepended)
//   - Decode(ids)                  == golden decode        (SentencePiece chain)
//   - IncrementalDetokenizer fed id-by-id reproduces the decode, both as the
//     concatenated deltas and as OutputText().
// The corpus stresses Metaspace edges (leading/trailing/repeated spaces, a
// literal ▁, the empty string), byte-fallback triggers (emoji, rare Unicode,
// control bytes, CJK/Hangul), and the special-token interaction (BOS/EOS,
// [INST], a mid-string special where the "first" prepend must NOT re-fire).
#include <doctest/doctest.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/detokenizer.h"

using nlohmann::json;
using vllm::tok::Tokenizer;
using vllm::v1::DetokenizerRequest;
using vllm::v1::IncrementalDetokenizer;

namespace {

const std::string kGoldenDir =
    std::string(PARITY_GOLDENS_DIR) + "/tokenizer_mistral";

// One tokenizer load for the whole binary (the golden tokenizer.json is ~1.9MB
// / 32768 vocab + 58980 merges; parsing it per-case would dominate runtime).
const Tokenizer& GoldenTokenizer() {
  static const Tokenizer tok =
      Tokenizer::FromHfJson(kGoldenDir + "/tokenizer.json");
  return tok;
}

struct Entry {
  std::string text;
  std::vector<int32_t> ids;
  std::vector<int32_t> ids_special;
  std::string decode;
};

const std::vector<Entry>& GoldenEntries() {
  static const std::vector<Entry> entries = [] {
    std::ifstream in(kGoldenDir + "/encodings.json", std::ios::binary);
    REQUIRE_MESSAGE(in.good(),
                    "missing golden encodings.json — regenerate with "
                    "tools/parity/dump_tokenizer_mistral.py");
    const json doc = json::parse(in);
    std::vector<Entry> out;
    for (const json& e : doc.at("entries")) {
      out.push_back({e.at("text").get<std::string>(),
                     e.at("ids").get<std::vector<int32_t>>(),
                     e.at("ids_special").get<std::vector<int32_t>>(),
                     e.at("decode").get<std::string>()});
    }
    return out;
  }();
  return entries;
}

}  // namespace

TEST_CASE("Mistral tokenizer loads as the SentencePiece family with BOS=1") {
  const Tokenizer& tok = GoldenTokenizer();
  CHECK(tok.IsSentencePiece());
  CHECK(tok.BosId() == 1);        // TemplateProcessing "single" prepends <s>
  CHECK(tok.EosId() == -1);       // template appends no special token
}

TEST_CASE("corpus stresses Metaspace, byte-fallback and special tokens") {
  const auto& entries = GoldenEntries();
  CHECK(entries.size() >= 40);
  bool has_empty = false;
  bool has_byte_fallback = false;  // an entry whose ids include a <0xNN> token
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : entries) {
    has_empty |= e.text.empty();
    for (const int32_t id : e.ids) {
      const std::string& t = tok.TokenText(id);
      if (t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x') {
        has_byte_fallback = true;
      }
    }
  }
  CHECK(has_empty);
  CHECK(has_byte_fallback);
}

TEST_CASE("Encode matches HF tokenizers on every corpus entry") {
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    CHECK(tok.Encode(e.text) == e.ids);
  }
}

TEST_CASE("EncodeWithSpecialTokens prepends BOS on every corpus entry") {
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    CHECK(tok.EncodeWithSpecialTokens(e.text) == e.ids_special);
  }
}

TEST_CASE("Decode reproduces HF decode on every corpus entry") {
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    CHECK(tok.Decode(e.ids) == e.decode);
  }
}

TEST_CASE("IncrementalDetokenizer fed id-by-id reproduces HF decode") {
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    DetokenizerRequest request;
    request.skip_special_tokens = false;
    std::unique_ptr<IncrementalDetokenizer> det =
        IncrementalDetokenizer::FromNewRequest(&tok, std::move(request));
    std::string streamed;
    for (size_t i = 0; i < e.ids.size(); ++i) {
      CHECK_FALSE(det->Update({e.ids[i]}, false).has_value());  // no stop set
      streamed += det->GetNextOutputText(/*finished=*/i + 1 == e.ids.size(),
                                         /*delta=*/true);
    }
    CHECK(streamed == e.decode);
    CHECK(det->GetNextOutputText(/*finished=*/true, /*delta=*/false) ==
          e.decode);
  }
}
