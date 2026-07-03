// M0.5 DoD gate: token-for-token parity with HF tokenizers on the committed
// corpus. Goldens (tests/parity/goldens/tokenizer_qwen36/) are produced by
// the REAL HF tokenizers library on dgx.casa via
// tools/parity/dump_tokenizer.py from the unsloth Qwen3.6-27B snapshot's
// tokenizer.json (the exact file is committed alongside, so oracle and this
// test parse identical bytes). Per corpus entry:
//   - Encode(text) == golden ids
//   - Decode(ids) == text (byte-exact round-trip)
//   - IncrementalDetokenizer fed id-by-id (skip_special_tokens=false)
//     reproduces text, both as the concatenated deltas and as OutputText().
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
    std::string(PARITY_GOLDENS_DIR) + "/tokenizer_qwen36";

// One tokenizer load for the whole binary (the golden tokenizer.json is
// ~20MB / 248k vocab; parsing it per-case would dominate the test runtime).
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
                    "tools/parity/dump_tokenizer.py on dgx.casa");
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

TEST_CASE("corpus covers the plan's category count and the empty string") {
  const auto& entries = GoldenEntries();
  CHECK(entries.size() >= 60);
  bool has_empty = false;
  for (const Entry& e : entries) has_empty |= e.text.empty();
  CHECK(has_empty);
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
    // Full accumulated text agrees (empty entries: nothing was ever fed).
    CHECK(det->GetNextOutputText(/*finished=*/true, /*delta=*/false) ==
          e.text);
  }
}
