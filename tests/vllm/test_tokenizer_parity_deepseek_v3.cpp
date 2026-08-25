// TOK-DEEPSEEK-V3-PRE (issue #1924) — token-for-token parity with HF
// `tokenizers` on the DeepSeek-V3 / V4-Flash pre-tokenizer. Sibling of
// test_tokenizer_parity_deepseek.cpp (DeepSeek-V2/V2-Lite); same harness, same
// oracle tool, a DIFFERENT pipeline.
//
// WHY THIS FILE EXISTS. `vllm-server` died on every DeepSeek-V4-Flash
// safetensors checkpoint before reading a weight byte:
//
//     server: fatal: tokenizer: expected exactly one Split pre-tokenizer, found 3
//
// The checkpoint's `pre_tokenizer` is a FOUR-stage HF `Sequence` — three
// `Split(Isolated)` stages then `ByteLevel(use_regex=false)` — and this tree
// could express neither shape: the generic walk accepts exactly one `Split`,
// and the one DeepSeek escape hatch (`IsDeepSeekPreTokenizer`) demands the
// SEVEN-element DeepSeek-V2 shape. Those are two DIFFERENT families, and the
// naming in this tree used to say otherwise: what `SplitPattern::kDeepSeek`
// implements is llama.cpp's `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM` (GGUF pre
// `deepseek-llm`), while this file's family is
// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM` (GGUF pre `deepseek-v3`) —
// llama.cpp/src/llama-vocab.cpp:308-325 @ b10451 (10bf611e5, the pinned
// llama.cpp oracle), two separate cases of one switch. DeepSeek-V3, DeepSeek-R1
// and DeepSeek-V4-Flash are all the second.
//
// WHAT MAKES IT MEASURABLE RATHER THAN ARGUABLE. Stage order is load-bearing
// in a way a single-pass alternation scanner cannot reproduce: stage 2's
// six-alternative regex matches NO digit at all (`\p{N}` is outside every one
// of its alternatives), so a digit is only ever its own piece because stage 0
// already isolated it — and stage 2's rule-2 prefix class
// `[^\r\n\p{L}\p{P}\p{S}]?` would otherwise absorb a trailing digit into the
// following word, turning "abc123def" into "abc" + "123def". A pre-tokenizer
// defect is SILENT: it emits plausible text with the wrong ids. So it is
// compared against HF `tokenizers` — the SAME library vLLM's
// `AutoTokenizer` runs — over a corpus that holds every stage boundary.
//
// Goldens (tests/parity/goldens/tokenizer_deepseek_v3/) come from the REAL
// checkpoint: `0xSero/deepseek-v4-flash-0731-spark`, the EXL3 snapshot
// `MODEL-DSV4-EXL3` (#1875) is built against. `tokenizer.json` here is a
// byte-for-byte copy of that checkpoint's own, sha256 8f9f37ca…33cf, recorded
// in encodings.json — the sha is the pin, because a NAS path is not one. It was
// read from `/mnt/nas_share/rc/ckpt/dsv4-flash-0731-spark-exl3/tokenizer.json`
// on 2026-08-25; that path is NOT under `$CHECKPOINT_ROOT`, which names
// `/usr/local/nas_share/checkpoints` and does not exist on the authoring host
// (#1073 records why the two locations differ). Oracle and test therefore parse
// identical bytes, and no fixture can reproduce a MISREADING of the shape:
//   python3 tools/parity/dump_tokenizer.py
//     --tokenizer-json <checkpoint>/tokenizer.json
//     --golden-dir tests/parity/goldens/tokenizer_deepseek_v3
//     --label "0xSero/deepseek-v4-flash-0731-spark (EXL3 snapshot on the NAS)"
#include <doctest/doctest.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/tokenizer/pretokenizer.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/detokenizer.h"

using nlohmann::json;
using vllm::tok::Pretokenize;
using vllm::tok::SplitPattern;
using vllm::tok::Tokenizer;
using vllm::v1::DetokenizerRequest;
using vllm::v1::IncrementalDetokenizer;

namespace {

const std::string kGoldenDir =
    std::string(PARITY_GOLDENS_DIR) + "/tokenizer_deepseek_v3";

// One tokenizer load for the whole binary (the golden tokenizer.json is 6.4 MB
// / 128k vocab / 127741 merges; parsing it per-case would dominate runtime).
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
                    "tools/parity/dump_tokenizer.py against the checkpoint");
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

std::vector<std::string> Pieces(std::string_view text) {
  std::vector<std::string> out;
  for (const auto& [b, e] : Pretokenize(text, SplitPattern::kDeepSeekV3)) {
    out.emplace_back(text.substr(b, e - b));
  }
  return out;
}

}  // namespace

TEST_CASE("DeepSeek-V3 tokenizer.json resolves to the kDeepSeekV3 pipeline") {
  // The loader recognizes the three-Split Sequence EXACTLY (every regex
  // compared verbatim). A variant that ships different patterns must NOT
  // silently fall back to another family — it would mis-tokenize in silence,
  // which is what the `\p{N}{1,3}` heuristic already did once (#347). Pinning
  // the resolved pattern is what catches that. It also pins that this is NOT
  // SplitPattern::kDeepSeek, the SEVEN-stage DeepSeek-V2 family.
  CHECK(GoldenTokenizer().Pattern() == SplitPattern::kDeepSeekV3);
  CHECK(GoldenTokenizer().Pattern() != SplitPattern::kDeepSeek);
}

TEST_CASE("DeepSeek-V3 corpus covers the pipeline-stress categories") {
  const auto& entries = GoldenEntries();
  CHECK(entries.size() >= 100);
  bool has_empty = false;
  for (const Entry& e : entries) has_empty |= e.text.empty();
  CHECK(has_empty);
}

TEST_CASE("DeepSeek-V3 Encode matches HF tokenizers on every corpus entry") {
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    CHECK(tok.Encode(e.text) == e.ids);
  }
}

TEST_CASE("DeepSeek-V3 Decode round-trips every corpus entry byte-exactly") {
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    CHECK(tok.Decode(e.ids) == e.text);
  }
}

TEST_CASE("DeepSeek-V3 IncrementalDetokenizer fed id-by-id reproduces entries") {
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

// The stage-ORDER cases. Each expectation below was produced by running the
// checkpoint's own three Split stages through HF `tokenizers` 0.22.2 (the
// oracle's `Sequence([Split(Regex(r), "isolated") for r in stages])`), then
// transcribed. They are here, separate from the id gate above, because an id
// mismatch says "wrong" and these say WHICH STAGE is wrong.
TEST_CASE("DeepSeek-V3 stage 0 isolates digits in greedy groups of three") {
  CHECK(Pieces("1234567890") ==
        std::vector<std::string>{"123", "456", "789", "0"});
  CHECK(Pieces("12345") == std::vector<std::string>{"123", "45"});
  // The order proof. Stage 2 matches no digit, and its rule-2 prefix class
  // WOULD take the '3' as the optional prefix of "def" if the digits had not
  // already been carved out by stage 0.
  CHECK(Pieces("abc123def") ==
        std::vector<std::string>{"abc", "123", "def"});
  // Non-ASCII \p{N} groups the same way (Arabic-Indic U+0661..).
  CHECK(Pieces("\xD9\xA1\xD9\xA2\xD9\xA3\xD9\xA4") ==
        std::vector<std::string>{"\xD9\xA1\xD9\xA2\xD9\xA3", "\xD9\xA4"});
}

TEST_CASE("DeepSeek-V3 stage 1 CJK class is U+3040-U+30FF and U+4E00-U+9FA5") {
  // Inside: one maximal run is one piece, with NO whitespace prefix rule
  // (unlike DeepSeek-V2's stage 4 siblings, which carry `\s?`).
  CHECK(Pieces("\xE4\xBD\xA0\xE5\xA5\xBD") ==  // 你好
        std::vector<std::string>{"\xE4\xBD\xA0\xE5\xA5\xBD"});
  // U+4E00 and U+9FA5 are the inclusive ends; U+9FA6 is OUTSIDE, so it falls
  // to stage 2 (where it is a letter) and forms its own piece.
  CHECK(Pieces("\xE4\xB8\x80\xE9\xBE\xA5\xE9\xBE\xA6") ==  // 一龥龦
        std::vector<std::string>{"\xE4\xB8\x80\xE9\xBE\xA5", "\xE9\xBE\xA6"});
  // HANGUL IS NOT IN THIS CLASS. It IS in DeepSeek-V2's (U+AC00-U+D7FF), which
  // is exactly why the two families cannot share one implementation. The
  // discriminator is the BOUNDARY: if Hangul were in the class, "가힣中" would
  // be one stage-1 run. It is not, so stage 1 carves out only 中 and stage 2
  // then makes the Hangul its own letter run.
  CHECK(Pieces("\xEA\xB0\x80\xED\x9E\xA3\xE4\xB8\xAD") ==  // 가힣中
        std::vector<std::string>{"\xEA\xB0\x80\xED\x9E\xA3", "\xE4\xB8\xAD"});
  CHECK(Pieces("\xE4\xB8\xAD\xE6\x96\x87" "abc") ==  // 中文abc
        std::vector<std::string>{"\xE4\xB8\xAD\xE6\x96\x87", "abc"});
  // STAGE 1 MUST RUN BEFORE STAGE 2, and this is the only shape that shows it.
  // Everywhere else the two orders agree, because stage 2 can only ADD
  // boundaries to what stage 1 produced. The exception is alternative 5,
  // `\s+(?!\S)`: on the whole string it gives its last space back to the CJK
  // run's prefix, so a two-space run before 你 comes back as TWO one-space
  // pieces. Running stage 1 first isolates 你 into its own piece, the
  // whitespace piece then ends where the run ends, and `(?!\S)` is satisfied
  // there. MEASURED: reordering these two stages left every other assertion in
  // this file green.
  CHECK(Pieces("  \xE4\xBD\xA0") ==  // two spaces then 你
        std::vector<std::string>{"  ", "\xE4\xBD\xA0"});
  CHECK(Pieces("a  \xE4\xB8\xAD\xE6\x96\x87") ==  // a, two spaces, 中文
        std::vector<std::string>{"a", "  ", "\xE4\xB8\xAD\xE6\x96\x87"});
  // STAGE 1 DOES NOT HAVE THE LAST WORD: stage 2 re-splits its output. The
  // class spans U+3040-U+30FF whole, so ぀ゟ゠ヿ is ONE stage-1 piece — and
  // stage 2 then cuts it in three, because U+3040 is unassigned (its rule-2
  // prefix), U+30A0 is \p{Pd} (its rule 3) and U+309F/U+30FF are \p{Lo}.
  CHECK(Pieces("\xE3\x81\x80\xE3\x82\x9F\xE3\x82\xA0\xE3\x83\xBF") ==
        std::vector<std::string>{"\xE3\x81\x80\xE3\x82\x9F",
                                 "\xE3\x82\xA0", "\xE3\x83\xBF"});
}

TEST_CASE("DeepSeek-V3 stage 2 alternative 1 is [ASCII punct][A-Za-z]+") {
  // `(x` is ONE piece: the first alternative binds an ASCII punctuation
  // character to the ASCII letters after it. No other family here does this,
  // and alternative 3 (` ?[\p{P}\p{S}]+`) would give "(" + "x" instead.
  CHECK(Pieces("def foo(x): return x") ==
        std::vector<std::string>{"def", " foo", "(x", "):", " return", " x"});
  CHECK(Pieces("$var") == std::vector<std::string>{"$var"});
  CHECK(Pieces("_name") == std::vector<std::string>{"_name"});
  // Only ASCII letters follow it: a non-ASCII letter falls to alternative 2.
  CHECK(Pieces("$\xC3\xA9") == std::vector<std::string>{"$", "\xC3\xA9"});
}

TEST_CASE("DeepSeek-V3 stage 2 whitespace alternatives") {
  CHECK(Pieces("a b") == std::vector<std::string>{"a", " b"});
  CHECK(Pieces("a   b") == std::vector<std::string>{"a", "  ", " b"});
  CHECK(Pieces("a\n\nb") == std::vector<std::string>{"a", "\n\n", "b"});
  CHECK(Pieces("trailing  ") == std::vector<std::string>{"trailing", "  "});
  CHECK(Pieces("") == std::vector<std::string>{});
}

TEST_CASE("DeepSeek-V3 EncodeWithSpecialTokens adds NOTHING (no BOS)") {
  // The checkpoint's post_processor is a plain `ByteLevel` declaring no
  // special tokens, so HF's TokenizersBackend adds none — same finding as the
  // DeepSeek-V2 sibling. Measured: every golden id list was produced with
  // add_special_tokens=False, and this asserts the True path agrees.
  const Tokenizer& tok = GoldenTokenizer();
  for (const Entry& e : GoldenEntries()) {
    CAPTURE(e.text);
    CHECK(tok.EncodeWithSpecialTokens(e.text) == e.ids);
  }
}
