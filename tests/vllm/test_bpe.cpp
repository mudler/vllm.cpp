// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE. The handcrafted fixture below (kTinyJson) was verified against the
// real HF tokenizers library (NOT hand-guessed):
//   scp tiny_tokenizer.json oracle_tiny.py dgx.casa:/tmp/ &&
//   ssh dgx.casa '~/venvs/vllm-oracle/bin/python /tmp/oracle_tiny.py'
// (tokenizers 0.22.2). Every expected-id vector in the encode/decode tests
// below is copied verbatim from that oracle run, including the
// ignore_merges=true variant.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"

using vllm::tok::BpeSplit;
using vllm::tok::ByteToUnicode;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::MergeKey;
using vllm::tok::MergeRanks;
using vllm::tok::SplitPattern;
using vllm::tok::Tokenizer;
using vllm::tok::UnicodeToByte;
using vllm::tok::UnmapUnicodeToBytes;

namespace {

using Ids = std::vector<int32_t>;

// Writes `body` to a unique file under the system temp dir; removed in the
// destructor so test runs don't accumulate files.
class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_bpe_test_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// Replaces exactly one occurrence of `from` (which must be present).
std::string ReplaceOnce(std::string s, const std::string& from,
                        const std::string& to) {
  const size_t pos = s.find(from);
  REQUIRE(pos != std::string::npos);
  return s.replace(pos, from.size(), to);
}

// Checks that `fn` throws std::runtime_error whose message contains `needle`.
template <typename Fn>
void CheckThrowsContains(Fn&& fn, const std::string& needle) {
  try {
    fn();
    FAIL_CHECK("expected std::runtime_error containing \"" << needle << "\"");
  } catch (const std::runtime_error& e) {
    CHECK_MESSAGE(std::string(e.what()).find(needle) != std::string::npos,
                  "message was: " << e.what());
  }
}

// Handcrafted 19-token vocab + 3 added tokens; oracle-verified (see header).
// Merge order deliberately interleaves ("hello" needs ll before "he llo";
// "Ġworld" needs or+ld -> orld before "Ġw orld").
constexpr const char* kTinyJson = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 19, "content": "<|end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 20, "content": "<tool>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 21, "content": "<|end|>of", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
      {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
    ]
  },
  "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
  "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "w": 4, "r": 5, "d": 6, "Ġ": 7, "1": 8, "2": 9, "ll": 10, "he": 11, "llo": 12, "hello": 13, "Ġw": 14, "or": 15, "orld": 16, "Ġworld": 17, "ld": 18},
    "merges": [["l", "l"], ["h", "e"], ["ll", "o"], ["he", "llo"], ["Ġ", "w"], ["o", "r"], ["l", "d"], ["or", "ld"], ["Ġw", "orld"]]
  }
})json";

// Verbatim regex variants (as they appear escaped inside tokenizer.json).
constexpr const char* kQwen36RegexJson =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";
constexpr const char* kClassicQwen2RegexJson =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";
constexpr const char* kLlama3RegexJson =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";

Tokenizer LoadTiny(const std::string& body = kTinyJson) {
  TempJson f(body);
  return Tokenizer::FromHfJson(f.path());
}

std::string WithRegex(const char* regex_json) {
  return ReplaceOnce(kTinyJson, kQwen36RegexJson, regex_json);
}

}  // namespace

TEST_CASE("bytes_to_unicode bijection: 256 distinct cps, exact GPT-2 table") {
  std::set<uint32_t> seen;
  for (int b = 0; b < 256; ++b) {
    const uint32_t cp = ByteToUnicode(static_cast<uint8_t>(b));
    CHECK(seen.insert(cp).second);  // injective
    CHECK(UnicodeToByte(cp) == b);  // round-trip
    const bool printable = (b >= 0x21 && b <= 0x7E) ||
                           (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    if (printable) CHECK(cp == static_cast<uint32_t>(b));
  }
  // Spot values of the 0x100+n branch (n counts non-printables in byte order).
  CHECK(ByteToUnicode(0x00) == 0x100);
  CHECK(ByteToUnicode(0x0A) == 0x10A);  // '\n' -> 'Ċ'
  CHECK(ByteToUnicode(0x20) == 0x120);  // ' '  -> 'Ġ'
  CHECK(ByteToUnicode(0x7F) == 0x121);
  CHECK(ByteToUnicode(0xA0) == 0x142);
  CHECK(ByteToUnicode(0xAD) == 0x143);
  // Codepoints outside the image.
  CHECK(UnicodeToByte(0x20) == -1);
  CHECK(UnicodeToByte(0x144) == -1);
  CHECK(UnicodeToByte(0x20AC) == -1);
}

TEST_CASE("MapBytesToUnicode/UnmapUnicodeToBytes round-trip all bytes") {
  std::string all;
  for (int b = 0; b < 256; ++b) all.push_back(static_cast<char>(b));
  const std::string mapped = MapBytesToUnicode(all);
  CHECK(UnmapUnicodeToBytes(mapped) == all);
  CHECK(MapBytesToUnicode(" hi\n") == "Ġhi\xC4\x8A");  // Ġ + hi + Ċ
  CHECK(MapBytesToUnicode("") == "");
  // Codepoints outside the bijection image throw.
  CHECK_THROWS_AS((void)UnmapUnicodeToBytes(" "), std::runtime_error);
  CHECK_THROWS_AS((void)UnmapUnicodeToBytes("€"), std::runtime_error);
  // Invalid UTF-8 decodes to U+FFFD, which is outside the image.
  CHECK_THROWS_AS((void)UnmapUnicodeToBytes("\xFF"), std::runtime_error);
}

TEST_CASE("BpeSplit merges lowest rank first, leftmost on ties") {
  MergeRanks ranks;
  ranks[MergeKey("l", "l")] = 0;
  ranks[MergeKey("h", "e")] = 1;
  ranks[MergeKey("ll", "o")] = 2;
  ranks[MergeKey("he", "llo")] = 3;
  using V = std::vector<std::string>;
  // h e l l o -> h e ll o -> he ll o -> he llo -> hello
  CHECK(BpeSplit("hello", ranks) == V{"hello"});
  // Rank-0 "l l" fires at the LEFT of the lll run (oracle: "helllo" ->
  // [11, 10, 2, 3] = he ll l o), then nothing else applies.
  CHECK(BpeSplit("helllo", ranks) == V{"he", "ll", "l", "o"});
  CHECK(BpeSplit("lll", ranks) == V{"ll", "l"});  // oracle: [10, 2]
  CHECK(BpeSplit("eh", ranks) == V{"e", "h"});  // no pair applies
  CHECK(BpeSplit("", ranks).empty());
  CHECK(BpeSplit("h", ranks) == V{"h"});
  // Multi-byte mapped symbols merge as whole codepoints.
  MergeRanks r2;
  r2[MergeKey("Ġ", "Ġ")] = 0;
  CHECK(BpeSplit("ĠĠĠ", r2) == V{"ĠĠ", "Ġ"});  // leftmost tie
}

TEST_CASE("FromHfJson: oracle-verified encode/decode on tiny fixture") {
  const Tokenizer tok = LoadTiny();
  CHECK(tok.VocabSize() == 22);
  CHECK(tok.Pattern() == SplitPattern::kQwen2);
  CHECK(tok.EosId() == -1);  // ByteLevel post_processor carries no eos/bos
  CHECK(tok.BosId() == -1);
  CHECK(tok.AddedTokens().size() == 3);
  CHECK(tok.AddedTokens()[0].special);
  CHECK_FALSE(tok.AddedTokens()[1].special);

  // Expected ids verbatim from the HF tokenizers oracle run (see header).
  CHECK(tok.Encode("hello") == Ids{13});
  CHECK(tok.Encode("hello world") == Ids{13, 17});
  CHECK(tok.Encode(" world") == Ids{17});
  CHECK(tok.Encode("world") == Ids{4, 16});  // "w orld" is not a merge
  CHECK(tok.Encode("hello12") == Ids{13, 8, 9});  // qwen splits digits
  CHECK(tok.Encode("") == Ids{});
  // Merge ties resolve leftmost (oracle: "lll" -> [10, 2], "llll" -> two ll).
  CHECK(tok.Encode("lll") == Ids{10, 2});
  CHECK(tok.Encode("llll") == Ids{10, 10});

  // Decode inverts, including the byte-level space (Ġ) unmapping.
  CHECK(tok.Decode({13, 17}) == "hello world");
  CHECK(tok.Decode({4, 16}) == "world");
  CHECK(tok.Decode({}) == "");

  // TokenText returns the raw byte-level-alphabet string.
  CHECK(tok.TokenText(17) == "Ġworld");
  CHECK(tok.TokenText(7) == "Ġ");
}

TEST_CASE("added tokens: leftmost-longest match before pretokenization") {
  const Tokenizer tok = LoadTiny();
  // Oracle: special and non-special added tokens split identically.
  CHECK(tok.Encode("hello<|end|> world") == Ids{13, 19, 17});
  CHECK(tok.Encode("hello<tool>world") == Ids{13, 20, 4, 16});
  // Longest wins at the same start: "<|end|>of" (21) beats "<|end|>" (19).
  CHECK(tok.Encode("hello<|end|>of world") == Ids{13, 21, 17});
  // Added tokens decode to their literal content.
  CHECK(tok.Decode({13, 19, 17}) == "hello<|end|> world");
  CHECK(tok.Decode({13, 21, 17}) == "hello<|end|>of world");
  CHECK(tok.TokenText(19) == "<|end|>");
}

TEST_CASE("string-form merges parse identically to array form") {
  const std::string body = ReplaceOnce(
      kTinyJson,
      R"("merges": [["l", "l"], ["h", "e"], ["ll", "o"], ["he", "llo"], ["Ġ", "w"], ["o", "r"], ["l", "d"], ["or", "ld"], ["Ġw", "orld"]])",
      R"("merges": ["l l", "h e", "ll o", "he llo", "Ġ w", "o r", "l d", "or ld", "Ġw orld"])");
  const Tokenizer tok = LoadTiny(body);
  CHECK(tok.Encode("hello world") == Ids{13, 17});
}

TEST_CASE("ignore_merges: whole-pretoken vocab hit wins, else normal BPE") {
  // Oracle-verified variant: ignore_merges=true, vocab += {"wo": 22}.
  std::string body =
      ReplaceOnce(kTinyJson, R"("ignore_merges": false)",
                  R"("ignore_merges": true)");
  body = ReplaceOnce(body, R"("h": 0,)", R"("h": 0, "wo": 22,)");
  const Tokenizer tok = LoadTiny(body);
  CHECK(tok.Encode("hello") == Ids{13});
  CHECK(tok.Encode("wo") == Ids{22});          // direct vocab hit, no merges
  CHECK(tok.Encode("world") == Ids{4, 16});    // not in vocab -> BPE ("wo" is
  CHECK(tok.Encode("hello world") == Ids{13, 17});  // NOT used mid-word)
}

TEST_CASE("NFC normalizer entry is accepted (Qwen3.6 has it)") {
  const std::string body = ReplaceOnce(kTinyJson, R"("normalizer": null)",
                                       R"("normalizer": {"type": "NFC"})");
  CHECK(LoadTiny(body).Encode("hello") == Ids{13});
}

TEST_CASE("post_processor TemplateProcessing: trivially extractable bos/eos") {
  const std::string body = ReplaceOnce(
      kTinyJson,
      R"("post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})",
      R"("post_processor": {"type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<|end|>", "type_id": 0}},
                   {"Sequence": {"id": "A", "type_id": 0}},
                   {"SpecialToken": {"id": "<|end|>of", "type_id": 0}}],
        "pair": [],
        "special_tokens": {
          "<|end|>": {"id": "<|end|>", "ids": [19], "tokens": ["<|end|>"]},
          "<|end|>of": {"id": "<|end|>of", "ids": [21], "tokens": ["<|end|>of"]}}})");
  const Tokenizer tok = LoadTiny(body);
  CHECK(tok.BosId() == 19);
  CHECK(tok.EosId() == 21);
}

TEST_CASE("llama3-style \\p{N}{1,3} split regex maps to kLlama3") {
  const Tokenizer tok = LoadTiny(WithRegex(kLlama3RegexJson));
  CHECK(tok.Pattern() == SplitPattern::kLlama3);
  // Digit grouping differs from qwen but the merged ids here coincide
  // (no digit merges in the tiny vocab).
  CHECK(tok.Encode("hello12") == Ids{13, 8, 9});
}

TEST_CASE("loader failure modes throw with actionable messages") {
  SUBCASE("unknown model type") {
    CheckThrowsContains(
        [&] { LoadTiny(ReplaceOnce(kTinyJson, R"("type": "BPE")",
                                   R"("type": "Unigram")")); },
        "Unigram");
  }
  SUBCASE("classic qwen2 regex is rejected, not silently accepted") {
    CheckThrowsContains([&] { LoadTiny(WithRegex(kClassicQwen2RegexJson)); },
                        "add kQwen2Classic before accepting");
  }
  SUBCASE("unrecognized split regex names the regex") {
    CheckThrowsContains(
        [&] { LoadTiny(WithRegex(R"([a-z]+|\\s+)")); },
        R"([a-z]+|\s+)");  // message carries the decoded regex
  }
  SUBCASE("unknown normalizer") {
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(kTinyJson, R"("normalizer": null)",
                               R"("normalizer": {"type": "Lowercase"})"));
        },
        "Lowercase");
  }
  SUBCASE("ByteLevel add_prefix_space=true unsupported") {
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(
              kTinyJson,
              R"("type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
    ])",
              R"("type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": false}
    ])"));
        },
        "add_prefix_space");
  }
  SUBCASE("added token id colliding with a vocab id") {
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(kTinyJson, R"({"id": 19, "content": "<|end|>")",
                               R"({"id": 13, "content": "<|end|>")"));
        },
        "duplicate token id");
  }
  SUBCASE("added token id far beyond the vocab") {
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(kTinyJson, R"({"id": 19, "content": "<|end|>")",
                               R"({"id": 90000000, "content": "<|end|>")"));
        },
        "out of range");
  }
  SUBCASE("malformed merge entry") {
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(kTinyJson, R"(["l", "l"])", R"(["l"])"));
        },
        "merge");
  }
  SUBCASE("duplicate merge pair names the pair") {
    // HF keeps the LAST rank for duplicates; we keep neither and fail loud.
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(
              kTinyJson,
              R"("merges": [["l", "l"], ["h", "e"], ["ll", "o"], ["he", "llo"], ["Ġ", "w"], ["o", "r"], ["l", "d"], ["or", "ld"], ["Ġw", "orld"]])",
              R"("merges": ["a b", "b c", "a b"])"));
        },
        "a b");
  }
  SUBCASE("added token lstrip/rstrip/single_word semantics unsupported") {
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(
              kTinyJson, R"("single_word": false, "lstrip": false)",
              R"("single_word": false, "lstrip": true)"));
        },
        "lstrip");
  }
  SUBCASE("missing file") {
    CHECK_THROWS_AS((void)Tokenizer::FromHfJson("/nonexistent/tok.json"),
                    std::runtime_error);
  }
}

TEST_CASE("encode/decode runtime failure modes") {
  const Tokenizer tok = LoadTiny();
  // 'z' maps to symbol "z" which is not in the tiny vocab.
  CHECK_THROWS_AS((void)tok.Encode("z"), std::runtime_error);
  // Unknown / out-of-range ids.
  CHECK_THROWS_AS((void)tok.Decode({22}), std::runtime_error);
  CHECK_THROWS_AS((void)tok.Decode({-1}), std::runtime_error);
  CHECK_THROWS_AS((void)tok.Decode({1000000}), std::runtime_error);
  CHECK_THROWS_AS((void)tok.TokenText(1000000), std::runtime_error);
}
