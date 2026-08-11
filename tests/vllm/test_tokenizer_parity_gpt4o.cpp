// Token-for-token parity with HF tokenizers for the GPT-4o / o200k
// pre-tokenizer — the family a GGUF tags `tokenizer.ggml.pre = "llama4"`
// (issue #347, Muse Glimmer 30B). Structural pretokenizer tests cannot stand in
// for this: a wrong-but-plausible split (kLlama3, kQwen2) still produces a
// valid-looking id stream, and only comparing ids against the reference catches
// it.
//
// Goldens in tests/parity/goldens/tokenizer_muse_glimmer/ come from the REAL HF
// tokenizers library run over the checkpoint's own tokenizer.json by
// tools/parity/dump_tokenizer_gpt4o.py. The committed tokenizer.json is that
// file with its merge list cut to the merges this corpus can reach (vocabulary
// untouched, ids unchanged); the generator proves the cut is faithful by
// re-encoding the whole corpus with it before writing anything, and records the
// full file's sha256 in encodings.json. The 17 GB GGUF is checked separately
// and by hand — see the command in the generator's docstring.
#include <doctest/doctest.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/tokenizer/pretokenizer.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/detokenizer.h"

using nlohmann::json;
using vllm::tok::SplitPattern;
using vllm::tok::Tokenizer;
using vllm::v1::DetokenizerRequest;
using vllm::v1::IncrementalDetokenizer;

namespace {

const std::string kGoldenDir =
    std::string(PARITY_GOLDENS_DIR) + "/tokenizer_muse_glimmer";

const Tokenizer& GoldenTokenizer() {
  static const Tokenizer tok =
      Tokenizer::FromHfJson(kGoldenDir + "/tokenizer.json");
  return tok;
}

struct Entry {
  std::string text;
  std::vector<int32_t> ids;
};

const std::vector<Entry>& GoldenEntries() {
  static const std::vector<Entry> entries = [] {
    std::ifstream in(kGoldenDir + "/encodings.json", std::ios::binary);
    REQUIRE_MESSAGE(in.good(),
                    "missing golden encodings.json — regenerate with "
                    "tools/parity/dump_tokenizer_gpt4o.py");
    const json doc = json::parse(in);
    std::vector<Entry> out;
    for (const json& e : doc.at("entries")) {
      out.push_back({e.at("text").get<std::string>(),
                     e.at("ids").get<std::vector<int32_t>>()});
    }
    return out;
  }();
  return entries;
}

}  // namespace

TEST_CASE("the checkpoint's own regex selects kGpt4o, not the kLlama3 fallback") {
  // DetectPattern() reaches a `\p{N}{1,3}` heuristic that would otherwise
  // claim this regex; if that ordering ever regresses, every id below stays
  // plausible and this is the only assertion that says why.
  CHECK(GoldenTokenizer().Pattern() == SplitPattern::kGpt4o);
}

TEST_CASE("corpus exercises the axes that separate kGpt4o from its neighbours") {
  const auto& entries = GoldenEntries();
  CHECK(entries.size() >= 40);
  bool has_empty = false;
  bool has_camel = false;
  bool has_contraction = false;
  bool has_slash = false;
  bool has_special = false;
  bool has_digits = false;
  bool has_non_ascii = false;
  for (const Entry& e : entries) {
    has_empty |= e.text.empty();
    has_camel |= e.text.find("XMLHttpRequest") != std::string::npos;
    has_contraction |= e.text.find("don't") != std::string::npos;
    has_slash |= e.text.find("path/to/file.txt") != std::string::npos;
    has_special |= e.text.find("<|begin_of_text|>") != std::string::npos;
    has_digits |= e.text.find("1234567890") != std::string::npos;
    for (const char c : e.text) {
      has_non_ascii |= static_cast<unsigned char>(c) >= 0x80;
    }
  }
  CHECK(has_empty);
  CHECK(has_camel);
  CHECK(has_contraction);
  CHECK(has_slash);
  CHECK(has_special);
  CHECK(has_digits);
  CHECK(has_non_ascii);
}

TEST_CASE("Encode matches HF tokenizers on every corpus entry") {
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    CHECK(tok.Encode(e.text) == e.ids);
  }
}

TEST_CASE("Decode round-trips every corpus entry byte-exactly") {
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    CHECK(tok.Decode(e.ids) == e.text);
  }
}

TEST_CASE("IncrementalDetokenizer fed id-by-id reproduces every entry") {
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
    CHECK(streamed == e.text);
    CHECK(det->GetNextOutputText(/*finished=*/true, /*delta=*/false) == e.text);
  }
}
