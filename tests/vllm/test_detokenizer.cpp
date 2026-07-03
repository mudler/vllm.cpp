// Tests for the incremental detokenizer ported from
// vllm/v1/engine/detokenizer.py @ e24d1b24. Ported upstream cases:
// - tests/tokenizers_/test_detokenize.py::test_decode_streaming
//   ("streaming decode matches full decode", incl. with/without prompt,
//   skip_special_tokens on/off, spaces_between_special_tokens on/off)
// - tests/tokenizers_/test_detokenize.py::test_oov_decode
// - tests/tokenizers_/test_detokenize.py::test_mistral_edge_case is the
//   incomplete-UTF-8 semantics; covered here by the byte-level
//   "UTF-8 across token boundaries" cases on the real byte-mapped alphabet.
// - stop strings / min_tokens / stop_terminated: detokenizer.py owns
//   check_stop_strings and the stop bookkeeping (upstream exercises them at
//   the OutputProcessor level in tests/v1/engine/test_output_processor.py::
//   {test_stop_string,test_incremental_detokenization}); the semantics are
//   tested here directly against the ported functions.
// The fixture extends tests/vllm/test_bpe.cpp's oracle-verified tiny vocab
// with byte-level fragments of multi-byte UTF-8 characters. Every hardcoded
// expected value below was verified against the REAL upstream python code
// (executed verbatim from the pinned checkout with imports stubbed):
//   python3 tools/parity/oracle_detokenizer.py /home/mudler/_git/vllm
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/detokenizer.h"

using nlohmann::json;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::CheckStopStrings;
using vllm::v1::ConvertPromptIdsToTokens;
using vllm::v1::DetokenizeIncrementally;
using vllm::v1::DetokenizerRequest;
using vllm::v1::IncrementalDetokenizer;
using vllm::v1::PromptTokens;
using vllm::v1::SlowIncrementalDetokenizer;

namespace {

using Ids = std::vector<int32_t>;

// Writes `body` to a unique file under the system temp dir; removed in the
// destructor so test runs don't accumulate files.
class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_detok_test_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// The oracle-verified tiny fixture from test_bpe.cpp (ids 0..21), extended
// with byte-level fragments of multi-byte UTF-8 characters so incremental
// decoding can be driven across codepoint boundaries:
//   22 = bytes F0 9F        (first half of 🌍 = F0 9F 8C 8D)
//   23 = bytes 8C 8D        (second half of 🌍)
//   24 = byte  C3           (first byte of é = C3 A9)
//   25 = byte  A9           (second byte of é)
//   26..29 = single bytes F0, 9F, 8C, 8D
//   30 = bytes 8D 21        (continuation byte + '!')
//   31 = bytes E5 81 9C     (停)
//   32 = bytes E6 AD A2     (止)
//   33 = literal "£中"       (£ U+00A3 is IN the byte-map image, 中 U+4E2D is
//                            not — per-token all-or-nothing fallback case)
// Fragment tokens are fed to the detokenizer by id (never via Encode), so no
// merges are needed for them.
Tokenizer BuildFixture() {
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", 19}, {"content", "<|end|>"}, {"special", true}},
       {{"id", 20}, {"content", "<tool>"}, {"special", false}},
       {{"id", 21}, {"content", "<|end|>of"}, {"special", true}}});
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
  json vocab = {{"h", 0},  {"e", 1},     {"l", 2},  {"o", 3},    {"w", 4},
                {"r", 5},  {"d", 6},     {"Ġ", 7},  {"1", 8},    {"2", 9},
                {"ll", 10}, {"he", 11},  {"llo", 12}, {"hello", 13},
                {"Ġw", 14}, {"or", 15},  {"orld", 16}, {"Ġworld", 17},
                {"ld", 18}};
  vocab[MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[MapBytesToUnicode("\x8C\x8D")] = 23;
  vocab[MapBytesToUnicode("\xC3")] = 24;
  vocab[MapBytesToUnicode("\xA9")] = 25;
  vocab[MapBytesToUnicode("\xF0")] = 26;
  vocab[MapBytesToUnicode("\x9F")] = 27;
  vocab[MapBytesToUnicode("\x8C")] = 28;
  vocab[MapBytesToUnicode("\x8D")] = 29;
  vocab[MapBytesToUnicode("\x8D!")] = 30;
  vocab[MapBytesToUnicode("\xE5\x81\x9C")] = 31;
  vocab[MapBytesToUnicode("\xE6\xAD\xA2")] = 32;
  vocab["\xC2\xA3\xE4\xB8\xAD"] = 33;  // literal "£中", NOT byte-mapped
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       json::array({json::array({"l", "l"}), json::array({"h", "e"}),
                    json::array({"ll", "o"}), json::array({"he", "llo"}),
                    json::array({"Ġ", "w"}), json::array({"o", "r"}),
                    json::array({"l", "d"}), json::array({"or", "ld"}),
                    json::array({"Ġw", "orld"})})}};
  TempJson f(doc.dump());
  return Tokenizer::FromHfJson(f.path());
}

const Tokenizer& Fixture() {
  static const Tokenizer tok = BuildFixture();
  return tok;
}

// tokenizer.decode(ids, skip_special_tokens=...): specials are standalone
// tokens, so filtering them before Decode matches HF's skip semantics.
std::string DecodeSkipAware(const Tokenizer& tok, const Ids& ids,
                            bool skip_special_tokens) {
  Ids kept;
  for (const int32_t id : ids) {
    if (skip_special_tokens && tok.IsSpecial(id)) continue;
    kept.push_back(id);
  }
  return tok.Decode(kept);
}

// Port of tests/tokenizers_/test_detokenize.py::_run_incremental_decode.
std::pair<std::string, Ids> RunIncrementalDecode(
    const Tokenizer& tok, const Ids& all_input_ids, bool skip_special_tokens,
    size_t starting_index, bool spaces_between_special_tokens = true) {
  DetokenizerRequest request;
  request.prompt_token_ids =
      Ids(all_input_ids.begin(),
          all_input_ids.begin() + static_cast<std::ptrdiff_t>(starting_index));
  request.skip_special_tokens = skip_special_tokens;
  request.spaces_between_special_tokens = spaces_between_special_tokens;
  const auto detokenizer =
      IncrementalDetokenizer::FromNewRequest(&tok, std::move(request));

  std::string output_text;
  for (size_t i = starting_index; i < all_input_ids.size(); ++i) {
    detokenizer->Update({all_input_ids[i]}, false);
    const bool finished = i + 1 == all_input_ids.size();
    output_text += detokenizer->GetNextOutputText(finished, /*delta=*/true);
  }
  return {output_text, detokenizer->OutputTokenIds()};
}

// Structural UTF-8 validity: no truncated sequence, no dangling lead byte,
// no stray continuation byte (what a streamed delta must never contain).
bool IsValidUtf8(std::string_view s) {
  size_t i = 0;
  while (i < s.size()) {
    const uint8_t b0 = static_cast<uint8_t>(s[i]);
    size_t len;
    if (b0 < 0x80) {
      len = 1;
    } else if ((b0 & 0xE0) == 0xC0 && b0 >= 0xC2) {
      len = 2;
    } else if ((b0 & 0xF0) == 0xE0) {
      len = 3;
    } else if ((b0 & 0xF8) == 0xF0 && b0 <= 0xF4) {
      len = 4;
    } else {
      return false;  // stray continuation or invalid lead
    }
    if (i + len > s.size()) return false;  // truncated at end
    for (size_t j = 1; j < len; ++j) {
      if ((static_cast<uint8_t>(s[i + j]) & 0xC0) != 0x80) return false;
    }
    i += len;
  }
  return true;
}

// Builds a Slow detokenizer directly (for stop-string / delta tests).
SlowIncrementalDetokenizer MakeSlow(const Tokenizer& tok,
                                    DetokenizerRequest request) {
  return SlowIncrementalDetokenizer(tok, std::move(request));
}

}  // namespace

TEST_CASE(
    "streaming decode matches full decode "
    "(ported: test_detokenize.py::test_decode_streaming)") {
  const Tokenizer& tok = Fixture();
  const std::vector<std::string> truths = {
      "hello world",
      "hello12 world",
      " world",
      "llll",
      // SPECIAL_TOKS_TRUTH analog: adjacent special and non-special added
      // tokens next to plain text.
      "hello<|end|><|end|><tool> world",
      "hello<|end|>of world",
      "hello<tool>world",
  };
  for (const std::string& truth : truths) {
    CAPTURE(truth);
    for (const bool with_prompt : {false, true}) {
      for (const bool skip_special_tokens : {true, false}) {
        CAPTURE(with_prompt);
        CAPTURE(skip_special_tokens);
        Ids truth_tokens = tok.Encode(truth);
        truth_tokens.push_back(19);  // eos-append analog (<|end|>)

        const std::string new_truth =
            DecodeSkipAware(tok, truth_tokens, skip_special_tokens);

        size_t starting_index = 0;
        std::string generated = new_truth;
        if (with_prompt) {
          starting_index = truth_tokens.size() / 2;
          const Ids prompt_ids(
              truth_tokens.begin(),
              truth_tokens.begin() +
                  static_cast<std::ptrdiff_t>(starting_index));
          const std::string prompt =
              DecodeSkipAware(tok, prompt_ids, skip_special_tokens);
          generated = new_truth.substr(prompt.size());
        }

        const auto [decoded_text, out_ids] = RunIncrementalDecode(
            tok, truth_tokens, skip_special_tokens, starting_index);
        CHECK(decoded_text == generated);
        CHECK(out_ids == Ids(truth_tokens.begin() +
                                 static_cast<std::ptrdiff_t>(starting_index),
                             truth_tokens.end()));

        // spaces_between_special_tokens is inert for the byte-level BPE
        // family (upstream is_fast=True branch): identical output.
        const auto [no_spaces_text, no_spaces_ids] = RunIncrementalDecode(
            tok, truth_tokens, skip_special_tokens, starting_index,
            /*spaces_between_special_tokens=*/false);
        CHECK(no_spaces_text == decoded_text);
      }
    }
  }
}

TEST_CASE("UTF-8 across token boundaries is held back until complete") {
  const Tokenizer& tok = Fixture();

  SUBCASE("4-byte emoji split over 2 tokens") {
    DetokenizerRequest request;
    auto det = MakeSlow(tok, std::move(request));
    det.Update({22}, false);  // F0 9F: incomplete
    CHECK(det.GetNextOutputText(false, true) == "");
    CHECK(det.OutputText() == "");
    det.Update({23}, false);  // 8C 8D: completes U+1F30D
    CHECK(det.GetNextOutputText(false, true) == "\xF0\x9F\x8C\x8D");  // 🌍
    CHECK(det.OutputText() == "\xF0\x9F\x8C\x8D");
  }

  SUBCASE("2-byte char split over 2 tokens") {
    const auto [text, ids] = RunIncrementalDecode(tok, {24, 25}, true, 0);
    CHECK(text == "\xC3\xA9");  // é
  }

  SUBCASE("4-byte emoji split over 4 single-byte tokens") {
    DetokenizerRequest request;
    auto det = MakeSlow(tok, std::move(request));
    for (const int32_t id : {26, 27, 28}) {
      det.Update({id}, false);
      CHECK(det.GetNextOutputText(false, true) == "");
    }
    det.Update({29}, false);
    CHECK(det.GetNextOutputText(false, true) == "\xF0\x9F\x8C\x8D");
  }

  SUBCASE("unfinished sequence at end of generation stays withheld") {
    const auto [text, ids] = RunIncrementalDecode(tok, {13, 22}, true, 0);
    CHECK(text == "hello");  // trailing F0 9F never completes
  }

  SUBCASE("text continuing after an invalid sequence is flushed raw") {
    // Upstream's lossy str would emit "�!"; we emit the raw bytes (recorded
    // deviation: no U+FFFD substitution).
    const auto [text, ids] = RunIncrementalDecode(tok, {22, 30}, true, 0);
    CHECK(text == "\xF0\x9F\x8D!");
  }

  SUBCASE("multi-byte char straddling the prompt/generation boundary") {
    // Prompt ends with the first half of the emoji; the generation completes
    // it. Upstream drops the straddling character (its lossy char count never
    // exceeds the prefix window's) — we replicate that exactly.
    const auto [text, ids] = RunIncrementalDecode(tok, {13, 22, 23, 17}, true,
                                                  /*starting_index=*/2);
    CHECK(text == " world");
  }
}

TEST_CASE("mixed in/out-of-image token passes through verbatim (per-token)") {
  // HF's ByteLevel decoder reverses the byte map per-TOKEN, all-or-nothing:
  // 中 (U+4E2D) is outside the bytes_to_unicode image, so the whole token
  // "£中" passes through as its literal UTF-8 text — even though £ (U+00A3)
  // alone is in the image (it would byte-map to the single byte A3). A
  // per-character mapping would emit the corrupt bytes A3 E4 B8 AD.
  const Tokenizer& tok = Fixture();
  const auto [text, ids] = RunIncrementalDecode(tok, {13, 33}, true, 0);
  CHECK(text == "hello\xC2\xA3\xE4\xB8\xAD");  // "hello£中", £ NOT remapped
}

TEST_CASE(
    "out-of-vocab ids decode to empty text "
    "(ported: test_detokenize.py::test_oov_decode)") {
  const Tokenizer& tok = Fixture();
  const auto [text, ids] = RunIncrementalDecode(tok, {tok.VocabSize()}, true,
                                                /*starting_index=*/0);
  CHECK(text == "");
  CHECK(ids == Ids{tok.VocabSize()});
  const auto [text2, ids2] = RunIncrementalDecode(tok, {-5, 13}, true, 0);
  CHECK(text2 == "hello");
}

TEST_CASE("special tokens skipped or included per skip_special_tokens") {
  const Tokenizer& tok = Fixture();
  SUBCASE("skipped") {
    const auto [text, ids] = RunIncrementalDecode(tok, {13, 19, 19, 20, 17},
                                                  /*skip=*/true, 0);
    CHECK(text == "hello<tool> world");  // 20 is added but NOT special
  }
  SUBCASE("included") {
    const auto [text, ids] = RunIncrementalDecode(tok, {13, 19, 19, 20, 17},
                                                  /*skip=*/false, 0);
    CHECK(text == "hello<|end|><|end|><tool> world");
  }
  SUBCASE("only a special token") {
    CHECK(RunIncrementalDecode(tok, {19}, true, 0).first == "");
    CHECK(RunIncrementalDecode(tok, {19}, false, 0).first == "<|end|>");
  }
}

TEST_CASE("CheckStopStrings (ported: detokenizer.py::check_stop_strings)") {
  const std::vector<std::string> stop{"wor", "ld"};

  SUBCASE("no new chars -> no match") {
    CHECK_FALSE(
        CheckStopStrings("hello world", 0, stop, false).has_value());
  }
  SUBCASE("no stop strings -> no match") {
    CHECK_FALSE(CheckStopStrings("hello world", 5, {}, false).has_value());
  }
  SUBCASE("match truncates to start of stop string") {
    const auto r = CheckStopStrings("hello world", 6, stop, false);
    REQUIRE(r.has_value());
    CHECK(r->first == "wor");
    CHECK(r->second == 6);
  }
  SUBCASE("include_in_output truncates to end of stop string") {
    const auto r = CheckStopStrings("hello world", 6, stop, true);
    REQUIRE(r.has_value());
    CHECK(r->first == "wor");
    CHECK(r->second == 9);
  }
  SUBCASE("include_in_output with stop at very end -> no truncation (-1)") {
    const auto r = CheckStopStrings("hello world", 2, {"ld"}, true);
    REQUIRE(r.has_value());
    CHECK(r->first == "ld");
    CHECK(r->second == -1);
  }
  SUBCASE("first stop string in list order wins") {
    const auto r = CheckStopStrings("hello world", 11, {"ld", "hell"}, false);
    REQUIRE(r.has_value());
    CHECK(r->first == "ld");
  }
  SUBCASE("already-searched text is not re-matched") {
    // "ab" occurs only outside the window reachable by 1 new char.
    CHECK_FALSE(CheckStopStrings("abcabc", 1, {"ab"}, false).has_value());
    // ...but a match overlapping the new chars is found.
    const auto r = CheckStopStrings("abcabc", 1, {"abc"}, false);
    REQUIRE(r.has_value());
    CHECK(r->second == 3);
  }
}

TEST_CASE("stop strings via Update (detokenizer.py stop bookkeeping)") {
  const Tokenizer& tok = Fixture();

  SUBCASE("stop string excluded from output") {
    DetokenizerRequest request;
    request.stop = {"wor"};
    auto det = MakeSlow(tok, std::move(request));
    CHECK_FALSE(det.Update({13}, false).has_value());  // "hello"
    const auto stop = det.Update({17}, false);         // " world"
    REQUIRE(stop.has_value());
    CHECK(*stop == "wor");
    CHECK(det.OutputText() == "hello ");
    CHECK(det.GetNextOutputText(true, false) == "hello ");
  }

  SUBCASE("stop string included in output") {
    DetokenizerRequest request;
    request.stop = {"wor"};
    request.include_stop_str_in_output = true;
    auto det = MakeSlow(tok, std::move(request));
    det.Update({13}, false);
    const auto stop = det.Update({17}, false);
    REQUIRE(stop.has_value());
    CHECK(*stop == "wor");
    CHECK(det.OutputText() == "hello wor");
  }

  SUBCASE("streamed deltas hold back stop_buffer_length chars") {
    DetokenizerRequest request;
    request.stop = {"xyz"};  // len 3 -> buffer 2
    auto det = MakeSlow(tok, std::move(request));
    det.Update({13}, false);
    CHECK(det.GetNextOutputText(false, true) == "hel");
    det.Update({17}, false);
    CHECK(det.GetNextOutputText(false, true) == "lo wor");
    CHECK(det.GetNextOutputText(true, true) == "ld");
    // Non-delta full text also respects the buffer until finished.
    CHECK(det.GetNextOutputText(false, false) == "hello wor");
    CHECK(det.GetNextOutputText(true, false) == "hello world");
  }

  SUBCASE("multi-byte stop string: hold-back window is chars, not bytes") {
    DetokenizerRequest request;
    // 停止: 6 bytes but 2 chars -> upstream buffer = len("停止") - 1 = 1 char.
    request.stop = {"\xE5\x81\x9C\xE6\xAD\xA2"};
    auto det = MakeSlow(tok, std::move(request));
    CHECK_FALSE(det.Update({13}, false).has_value());  // "hello"
    // Only 1 char is held back (a byte-based window would hold back 5).
    CHECK(det.GetNextOutputText(false, true) == "hell");
    CHECK_FALSE(det.Update({31}, false).has_value());  // 停
    CHECK(det.GetNextOutputText(false, true) == "o");  // 停 held back whole
    const auto stop = det.Update({32}, false);         // 止 completes 停止
    REQUIRE(stop.has_value());
    CHECK(*stop == "\xE5\x81\x9C\xE6\xAD\xA2");
    CHECK(det.OutputText() == "hello");  // stop string excluded
    CHECK(det.GetNextOutputText(true, true) == "");
    CHECK(det.GetNextOutputText(true, false) == "hello");
  }

  SUBCASE("deltas never end mid-UTF-8-character while holding back") {
    DetokenizerRequest request;
    request.stop = {"xyz"};  // len 3 -> buffer 2 chars
    auto det = MakeSlow(tok, std::move(request));
    std::string streamed;
    // "hello" + 🌍 (two byte-fragment tokens) + é (two single-byte tokens)
    // + " world": the 2-char hold-back lands inside the multi-byte chars if
    // counted in bytes.
    for (const int32_t id : {13, 22, 23, 24, 25, 17}) {
      det.Update({id}, false);
      const std::string delta = det.GetNextOutputText(false, true);
      CAPTURE(delta);
      CHECK(IsValidUtf8(delta));  // no dangling lead/continuation bytes
      CHECK(IsValidUtf8(det.GetNextOutputText(false, false)));
      streamed += delta;
    }
    streamed += det.GetNextOutputText(true, true);
    CHECK(streamed == "hello\xF0\x9F\x8C\x8D\xC3\xA9 world");
  }

  SUBCASE("stop_terminated excludes the final token from detokenization") {
    DetokenizerRequest request;
    auto det = MakeSlow(tok, std::move(request));
    det.Update({13}, false);
    CHECK_FALSE(det.Update({17}, true).has_value());
    CHECK(det.OutputText() == "hello");
    CHECK(det.OutputTokenIds() == Ids{13, 17});  // id still recorded
  }

  SUBCASE("min_tokens suppresses stop matches inside the minimum") {
    DetokenizerRequest request;
    request.stop = {"hello"};
    request.min_tokens = 1;
    auto det = MakeSlow(tok, std::move(request));
    CHECK_FALSE(det.Update({13}, false).has_value());  // within min_tokens
    CHECK_FALSE(det.Update({17}, false).has_value());  // window excludes it
    CHECK(det.OutputText() == "hello world");
    // Without min_tokens the same stream stops immediately.
    DetokenizerRequest request2;
    request2.stop = {"hello"};
    auto det2 = MakeSlow(tok, std::move(request2));
    const auto stop = det2.Update({13}, false);
    REQUIRE(stop.has_value());
    CHECK(*stop == "hello");
    CHECK(det2.OutputText() == "");
  }
}

TEST_CASE("ConvertPromptIdsToTokens seeds the incremental offsets") {
  const Tokenizer& tok = Fixture();
  // 10 prompt ids -> only the last 7 (5 + 2) are converted.
  const Ids prompt(10, 13);
  const PromptTokens seeded = ConvertPromptIdsToTokens(tok, prompt, false);
  CHECK(seeded.tokens.size() == 7);
  CHECK(seeded.read_offset == 7);
  CHECK(seeded.prefix_offset == 2);
  // Specials are dropped from the seed when skipping them.
  const PromptTokens skipped =
      ConvertPromptIdsToTokens(tok, {13, 19, 17}, true);
  CHECK(skipped.tokens == std::vector<std::string>{"hello", "Ġworld"});
  CHECK(skipped.read_offset == 2);
  CHECK(skipped.prefix_offset == 0);
  const PromptTokens empty = ConvertPromptIdsToTokens(tok, {}, true);
  CHECK(empty.tokens.empty());
  CHECK(empty.read_offset == 0);
}

TEST_CASE("DetokenizeIncrementally first-iteration branch (prev == null)") {
  const Tokenizer& tok = Fixture();
  const Ids all = {13, 17};
  const auto r = DetokenizeIncrementally(tok, all, nullptr, 0, 0,
                                         /*skip_special_tokens=*/false,
                                         /*spaces_between=*/true);
  CHECK(r.new_tokens == std::vector<std::string>{"hello", "Ġworld"});
  CHECK(r.new_text == " world");
  CHECK(r.prefix_offset == 1);
  CHECK(r.read_offset == 2);
}

TEST_CASE("FromNewRequest without a tokenizer skips detokenization") {
  const auto det =
      IncrementalDetokenizer::FromNewRequest(nullptr, DetokenizerRequest{});
  CHECK_FALSE(det->Update({1, 2, 3}, false).has_value());
  CHECK(det->GetNextOutputText(true, false) == "");
  CHECK(det->OutputTokenIds() == Ids{1, 2, 3});
  CHECK(det->NumOutputTokens() == 3);
}
