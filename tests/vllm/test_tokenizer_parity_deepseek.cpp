// MLA campaign W8 — token-for-token parity with HF `tokenizers` on the DeepSeek
// corpus. Sibling of test_tokenizer_parity.cpp (Qwen3.6); same harness, same
// oracle tool, different family.
//
// WHY THIS FILE EXISTS. Bringing DeepSeek-V2-Lite up to the SACRED paged-engine
// gate was BLOCKED by the tokenizer, not by the model: our loader hard-rejected
// the checkpoint with `unsupported normalizer "Sequence"`, and behind that sat a
// pre-tokenizer we could not express at all. DeepSeek's is not another
// alternation regex like Qwen/Llama-3/GPT-2 — it is a HF `Sequence` PIPELINE of
// seven stages (five `Split(Isolated)` over ENUMERATED codepoint ranges, then
// `Digits(individual_digits=true)`, then a `ByteLevel(use_regex=false)`), and
// its stage ORDER is load-bearing (stage 2's punctuation class spans 0x3A-0x7E
// and therefore CONTAINS A-Z/a-z; it is only correct because stage 1 already
// isolated the letter runs). See src/vllm/tokenizer/pretokenizer.cpp.
//
// A hand-written pipeline over transcribed codepoint ranges is exactly the kind
// of code that can look right and be subtly wrong on one range boundary, and a
// tokenizer bug is SILENT and correctness-fatal (OPT's missing BOS scored 0/6
// while emitting fluent English). So it is not argued, it is MEASURED against
// HF's own tokenizer over a corpus that stresses every stage: individual-digit
// splitting, the letter/punctuation class overlap, fullwidth and CJK
// punctuation, CJK/Hangul, the cased-letter class boundaries (Greek, Cyrillic,
// Armenian, Georgian, Cherokee, Coptic, Deseret are IN it; Arabic, Hebrew,
// Devanagari, Han are NOT), newline isolation, trailing whitespace, and how a
// `\s?` prefix attaches across every class boundary.
//
// Goldens (tests/parity/goldens/tokenizer_deepseek_v2/) are produced by the REAL
// HF tokenizers library on dgx.casa from the deepseek-ai/DeepSeek-V2-Lite
// snapshot's tokenizer.json (the exact file is committed alongside, so oracle
// and test parse identical bytes):
//   ~/venvs/vllm-oracle/bin/python tools/parity/dump_tokenizer.py
//     --tokenizer-json <snapshot>/tokenizer.json
//     --golden-dir tests/parity/goldens/tokenizer_deepseek_v2
//     --label "deepseek-ai/DeepSeek-V2-Lite @ 604d5664"
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
using vllm::tok::SplitPattern;
using vllm::tok::Tokenizer;
using vllm::v1::DetokenizerRequest;
using vllm::v1::IncrementalDetokenizer;

namespace {

const std::string kGoldenDir =
    std::string(PARITY_GOLDENS_DIR) + "/tokenizer_deepseek_v2";

// One tokenizer load for the whole binary (the golden tokenizer.json is ~4.6 MB
// / 100k vocab; parsing it per-case would dominate the runtime).
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

TEST_CASE("DeepSeek tokenizer.json resolves to the kDeepSeek pipeline") {
  // The loader recognizes DeepSeek's seven-stage Sequence EXACTLY (all five
  // Split regexes compared verbatim). If a future checkpoint ships different
  // codepoint ranges this must NOT silently fall back to another family — it
  // would mis-tokenize. Pinning the resolved pattern is what catches that.
  CHECK(GoldenTokenizer().Pattern() == SplitPattern::kDeepSeek);
}

TEST_CASE("DeepSeek corpus covers the pipeline-stress categories") {
  const auto& entries = GoldenEntries();
  CHECK(entries.size() >= 90);
  bool has_empty = false;
  for (const Entry& e : entries) has_empty |= e.text.empty();
  CHECK(has_empty);
}

TEST_CASE("DeepSeek Encode matches HF tokenizers on every corpus entry") {
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    CHECK(tok.Encode(e.text) == e.ids);
  }
}

TEST_CASE("DeepSeek Decode round-trips every corpus entry byte-exactly") {
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    CHECK(tok.Decode(e.ids) == e.text);
  }
}

TEST_CASE("DeepSeek IncrementalDetokenizer fed id-by-id reproduces every entry") {
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

TEST_CASE("DeepSeek EncodeWithSpecialTokens adds NOTHING (no BOS)") {
  // MEASURED against the oracle's own tokenizer on dgx (2026-07-22):
  //   AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite",
  //                                 trust_remote_code=True)
  //   -> class TokenizersBackend, add_bos_token resolves FALSE, and
  //      encode("The capital of France is", add_special_tokens=True)
  //      == [549, 6077, 280, 7239, 317]  (NO BOS)
  // even though tokenizer_config.json declares `add_bos_token: true` with
  // bos_token `<|begin of sentence|>` (id 100000). The class HF actually
  // instantiates is driven by tokenizer.json, whose post_processor is a plain
  // ByteLevel declaring no special tokens. See the guard test in test_bpe.cpp.
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    CHECK(tok.EncodeWithSpecialTokens(e.text) == e.ids);
  }
  CHECK(tok.Encode("The capital of France is") ==
        std::vector<int32_t>{549, 6077, 280, 7239, 317});
}
