// The tiktoken reader and BPE encoder (#634).
//
// The reference tokenizations were taken from PYTHON'S tiktoken driven over the
// shipped vocabulary with upstream's own pattern
// (`indextts/utils/tokenizer.py:215`), so this is checked against the library
// the model ships against, not against my reading of the algorithm.
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/tiktoken_bpe.h"

namespace tk = vllm::models::tiktoken;

namespace {

std::string WriteVocab(const std::string& body) {
  const auto p = std::filesystem::temp_directory_path() /
                 ("tiktoken_" + std::to_string(::rand()) + ".txt");
  std::ofstream f(p);
  f << body;
  return p.string();
}

}  // namespace

TEST_CASE("base64 entries are decoded, and a malformed line is REFUSED") {
  // "IQ==" is '!', "Ig==" is '"'.
  const std::string ok = WriteVocab("IQ== 0\nIg== 1\n");
  const auto ranks = tk::LoadRanks(ok);
  CHECK(ranks.size() == 2);
  CHECK(ranks.at("!") == 0);
  CHECK(ranks.at("\"") == 1);
  std::filesystem::remove(ok);

  // A dropped token shifts nothing visible and changes every encoding that
  // would have used it, so a malformed line must throw rather than be skipped.
  const std::string bad = WriteVocab("IQ== 0\nnot-base64-at-all\n");
  CHECK_THROWS_AS(tk::LoadRanks(bad), std::runtime_error);
  std::filesystem::remove(bad);

  const std::string neg = WriteVocab("IQ== -1\n");
  CHECK_THROWS_AS(tk::LoadRanks(neg), std::runtime_error);
  std::filesystem::remove(neg);
}

TEST_CASE("byte-pair encoding merges by LOWEST rank") {
  // Ranks chosen so first-match and lowest-rank disagree. "ab" is rank 9 and
  // "bc" is rank 2, so encoding "abc" must merge "bc" FIRST, leaving [a][bc].
  tk::Ranks r;
  r["a"] = 0;
  r["b"] = 1;
  r["c"] = 3;
  r["ab"] = 9;
  r["bc"] = 2;
  const auto got = tk::BytePairEncode("abc", r);
  REQUIRE(got.size() == 2);
  CHECK(got[0] == 0);  // "a"
  CHECK(got[1] == 2);  // "bc"
}

TEST_CASE("the pretokenizer splits letters from digits") {
  bool exact = false;
  const auto pieces = tk::Pretokenize("abc123", &exact);
  REQUIRE(pieces.size() == 2);
  CHECK(pieces[0] == "abc");
  CHECK(pieces[1] == "123");
  CHECK(exact);
}

// KNOWN GAP, recorded rather than claimed. A mutation that disables EVERY
// `Decided` check still passes this case, so the `exact` flag is NOT yet proven
// load-bearing by anything here. The flag is kept because a caller that needs a
// guarantee should have something to ask, but until a case fails without it,
// treat it as undemonstrated. Owed under #634.
TEST_CASE("a codepoint the range table does not decide sets exact FALSE") {
  // The table is deliberately narrow. What matters is that it SAYS SO rather
  // than guessing: a silent mis-split is the failure this flag exists to stop.
  bool exact = true;
  // U+0905 DEVANAGARI LETTER A: a letter, and outside every range here.
  tk::Pretokenize("\xe0\xa4\x85", &exact);
  CHECK_FALSE(exact);

  bool ascii_exact = false;
  tk::Pretokenize("hello world 123!", &ascii_exact);
  CHECK(ascii_exact);
}

TEST_CASE("the SHIPPED vocabulary encodes exactly as python tiktoken does") {
  const char* env = std::getenv("VLLM_CPP_INDEXTTS2_TIKTOKEN");
  if (env == nullptr) {
    MESSAGE("SKIPPED: set VLLM_CPP_INDEXTTS2_TIKTOKEN to the shipped "
            "multilingual_zh_ja_yue_char_del.tiktoken");
    return;
  }
  const auto ranks = tk::LoadRanks(std::string(env));
  CHECK(ranks.size() == 58836);

  struct Case {
    const char* text;
    std::vector<int64_t> want;
  };
  const std::vector<Case> cases = {
      {"hello", {675, 1909}},
      {"\xe4\xb8\x96\xe7\x95\x8c", {48721, 53743}},  // 世界
      {"abc123", {455, 66, 4714, 18}},
  };
  for (const auto& c : cases) {
    bool exact = false;
    const auto got = tk::Encode(c.text, ranks, &exact);
    CHECK(exact);
    CHECK(got == c.want);
  }
}
