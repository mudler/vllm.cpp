// Laya sequence construction parity gate (MODEL-LAYA, Phase 3).
//
// Mirrors build_sequence from laya/common.py:48-85
// (convaiinnovations/laya). Both this test and the golden generator
// (scripts/gen-laya-sequence-goldens.py) build the SAME minimal byte-level
// BPE tokenizer (256 byte chars + 4 special tokens, no merges) and run
// sequence construction with known inputs. The goldens verify parity with
// the Python reference.
//
// The tokenizer has no merges, so every character becomes one token. This
// isolates sequence-construction logic (budget management, marker tracking,
// truncation) from BPE merge correctness, which test_bpe.cpp covers.
//
// Perturbation tests verify the features this function gets wrong quietly:
// (1) wrong question-type string still runs; (2) missing MASK at option start
// makes markers point to wrong positions; (3) wrong truncation direction
// produces plausible but different state; (4) not replacing the mask token
// string leaks it as a real token.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "laya_sequence_goldens.inc"
#include "vllm/model_executor/models/laya_sequence.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"

using vllm::tok::ByteToUnicode;
using vllm::tok::Tokenizer;

namespace {

// JSON-escapes a string for embedding in a JSON string value.
std::string JsonEscape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    if (c == '\\') {
      out += "\\\\";
    } else if (c == '"') {
      out += "\\\"";
    } else if (c < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out;
}

// Encodes a Unicode codepoint as UTF-8.
std::string CodepointToUtf8(uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return out;
}

// Builds the same minimal byte-level BPE tokenizer.json that the golden
// generator builds: 256 byte-level chars (id=byte value), 4 special tokens
// ([CLS]=256, [SEP]=257, [PAD]=258, <mask>=259), no merges. Every character
// becomes one token (no BPE merges), isolating sequence-construction logic.
std::string BuildTokenizerJson() {
  std::string vocab;
  for (int b = 0; b < 256; ++b) {
    const uint32_t cp = ByteToUnicode(static_cast<uint8_t>(b));
    const std::string key = CodepointToUtf8(cp);
    if (b > 0) vocab += ',';
    vocab += "\"" + JsonEscape(key) + "\":" + std::to_string(b);
  }
  vocab += ",\"[CLS]\":256,\"[SEP]\":257,\"[PAD]\":258,\"<mask>\":259";

  // The Qwen3.6 split regex (kQwen36Regex in tokenizer.cpp). The C++
  // tokenizer recognizes it and maps to SplitPattern::kQwen2. With no merges
  // the split does not affect the final token IDs — every byte is one token
  // regardless — but the tokenizer must still parse it.
  const std::string regex =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|)"
      R"([^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|)"
      R"( ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|)"
      R"(\s*[\r\n]+|\s+(?!\S)|\s+)";

  std::string json = R"({"version":"1.0","truncation":null,"padding":null,)";
  json += R"("added_tokens":[)";
  json += R"({"id":256,"content":"[CLS]","single_word":false,"lstrip":false,)"
          R"("rstrip":false,"normalized":false,"special":true},)";
  json += R"({"id":257,"content":"[SEP]","single_word":false,"lstrip":false,)"
          R"("rstrip":false,"normalized":false,"special":true},)";
  json += R"({"id":258,"content":"[PAD]","single_word":false,"lstrip":false,)"
          R"("rstrip":false,"normalized":false,"special":true},)";
  json += R"({"id":259,"content":"<mask>","single_word":false,"lstrip":false,)"
          R"("rstrip":false,"normalized":false,"special":true}],)";
  json += R"("normalizer":null,)";
  json += R"("pre_tokenizer":{"type":"Sequence","pretokenizers":[)";
  json += "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"" +
          JsonEscape(regex) +
          "\"},\"behavior\":\"Isolated\",\"invert\":false},";
  json += R"({"type":"ByteLevel","add_prefix_space":false,)"
          R"("trim_offsets":false,"use_regex":false}]},)";
  json += R"("post_processor":{"type":"ByteLevel","add_prefix_space":false,)"
          R"("trim_offsets":false,"use_regex":false},)";
  json += R"("decoder":{"type":"ByteLevel","add_prefix_space":false,)"
          R"("trim_offsets":false,"use_regex":false},)";
  json += R"("model":{"type":"BPE","dropout":null,"unk_token":null,)"
          R"("continuing_subword_prefix":null,"end_of_word_suffix":null,)"
          R"("fuse_unk":false,"byte_fallback":false,"ignore_merges":false,)";
  json += "\"vocab\":{" + vocab + "},\"merges\":[]}}";
  return json;
}

// Loads the fixture tokenizer. Cached after first call.
const Tokenizer& FixtureTokenizer() {
  static const Tokenizer tok =
      Tokenizer::FromHfJsonBytes(BuildTokenizerJson(), "laya_sequence_test");
  return tok;
}

vllm::laya::SpecialTokens FixtureSpecial() {
  vllm::laya::SpecialTokens sp;
  sp.cls_id = laya_sequence_goldens::kClsId;
  sp.sep_id = laya_sequence_goldens::kSepId;
  sp.pad_id = laya_sequence_goldens::kPadId;
  sp.mask_id = laya_sequence_goldens::kMaskId;
  sp.mask_str = "<mask>";
  return sp;
}

void CheckSequence(const std::string& name, const vllm::laya::SequenceOutput& out,
                   const int* want_ids, int n_ids,
                   const int* want_markers, int n_markers) {
  INFO("test case: ", name);
  CHECK(out.ids.size() == static_cast<size_t>(n_ids));
  for (size_t i = 0; i < out.ids.size() && i < static_cast<size_t>(n_ids); ++i) {
    CHECK(out.ids[i] == want_ids[i]);
  }
  CHECK(out.markers.size() == static_cast<size_t>(n_markers));
  for (size_t i = 0; i < out.markers.size() && i < static_cast<size_t>(n_markers);
       ++i) {
    CHECK(out.markers[i] == want_markers[i]);
  }
}

}  // namespace

// --- Golden parity tests (5 cases) ---

TEST_CASE("laya sequence: choice_basic matches golden") {
  const auto& tok = FixtureTokenizer();
  vllm::laya::Question q;
  q.type = vllm::laya::QType::kChoice;
  q.instructions = "which color";
  q.options = {"red", "blue", "green"};
  q.state = "the car is fast";
  const auto out = vllm::laya::BuildSequence(
      tok, FixtureSpecial(), q,
      laya_sequence_goldens::kchoice_basic_max_len,
      laya_sequence_goldens::kchoice_basic_head_max_len);
  CheckSequence("choice_basic", out,
                laya_sequence_goldens::kchoice_basic_ids,
                laya_sequence_goldens::kchoice_basic_n_ids,
                laya_sequence_goldens::kchoice_basic_markers,
                laya_sequence_goldens::kchoice_basic_n_markers);
}

TEST_CASE("laya sequence: score_basic matches golden") {
  const auto& tok = FixtureTokenizer();
  vllm::laya::Question q;
  q.type = vllm::laya::QType::kScore;
  q.instructions = "rate quality";
  q.options = {"level 0: bad", "level 1: ok", "level 2: good"};
  q.state = "product review";
  const auto out = vllm::laya::BuildSequence(
      tok, FixtureSpecial(), q,
      laya_sequence_goldens::kscore_basic_max_len,
      laya_sequence_goldens::kscore_basic_head_max_len);
  CheckSequence("score_basic", out,
                laya_sequence_goldens::kscore_basic_ids,
                laya_sequence_goldens::kscore_basic_n_ids,
                laya_sequence_goldens::kscore_basic_markers,
                laya_sequence_goldens::kscore_basic_n_markers);
}

TEST_CASE("laya sequence: noul_basic matches golden") {
  const auto& tok = FixtureTokenizer();
  vllm::laya::Question q;
  q.type = vllm::laya::QType::kNoul;
  q.instructions = "is it true";
  q.options = {"false: no", "true: yes"};
  q.state = "the statement";
  const auto out = vllm::laya::BuildSequence(
      tok, FixtureSpecial(), q,
      laya_sequence_goldens::knoul_basic_max_len,
      laya_sequence_goldens::knoul_basic_head_max_len);
  CheckSequence("noul_basic", out,
                laya_sequence_goldens::knoul_basic_ids,
                laya_sequence_goldens::knoul_basic_n_ids,
                laya_sequence_goldens::knoul_basic_markers,
                laya_sequence_goldens::knoul_basic_n_markers);
}

TEST_CASE("laya sequence: truncation matches golden") {
  const auto& tok = FixtureTokenizer();
  vllm::laya::Question q;
  q.type = vllm::laya::QType::kChoice;
  q.instructions = "which color is best";
  q.options = {"red", "blue"};
  q.state = "the car is fast and red";
  const auto out = vllm::laya::BuildSequence(
      tok, FixtureSpecial(), q,
      laya_sequence_goldens::ktruncation_max_len,
      laya_sequence_goldens::ktruncation_head_max_len);
  CheckSequence("truncation", out,
                laya_sequence_goldens::ktruncation_ids,
                laya_sequence_goldens::ktruncation_n_ids,
                laya_sequence_goldens::ktruncation_markers,
                laya_sequence_goldens::ktruncation_n_markers);
}

TEST_CASE("laya sequence: truncate_left matches golden") {
  const auto& tok = FixtureTokenizer();
  vllm::laya::Question q;
  q.type = vllm::laya::QType::kNoul;
  q.instructions = "is it true";
  q.options = {"false: no", "true: yes"};
  q.state = "the car is fast and red";
  const auto out = vllm::laya::BuildSequence(
      tok, FixtureSpecial(), q,
      laya_sequence_goldens::ktruncate_left_max_len,
      laya_sequence_goldens::ktruncate_left_head_max_len, {}, true);
  CheckSequence("truncate_left", out,
                laya_sequence_goldens::ktruncate_left_ids,
                laya_sequence_goldens::ktruncate_left_n_ids,
                laya_sequence_goldens::ktruncate_left_markers,
                laya_sequence_goldens::ktruncate_left_n_markers);
}

// --- Perturbation tests ---

TEST_CASE("laya sequence: question type string changes the head tokens") {
  // Without the correct type string in the head ("choice question: ..." vs
  // "score question: ..."), the model still runs and emits plausible logits.
  // The distinction is asserted to matter: different types must produce
  // different sequences.
  const auto& tok = FixtureTokenizer();
  const auto sp = FixtureSpecial();
  vllm::laya::Question q;
  q.instructions = "rate quality";
  q.options = {"a", "b"};
  q.state = "ctx";

  q.type = vllm::laya::QType::kChoice;
  const auto out_choice = vllm::laya::BuildSequence(tok, sp, q);
  q.type = vllm::laya::QType::kScore;
  const auto out_score = vllm::laya::BuildSequence(tok, sp, q);

  // The type string ("choice" vs "score") changes the head tokens, so the
  // sequences must differ. They may differ in length too (6 vs 5 chars).
  const auto& shorter = out_choice.ids.size() <= out_score.ids.size()
                            ? out_choice.ids
                            : out_score.ids;
  bool differ = false;
  for (size_t i = 0; i < shorter.size(); ++i) {
    if (out_choice.ids[i] != out_score.ids[i]) {
      differ = true;
      break;
    }
  }
  if (!differ && out_choice.ids.size() != out_score.ids.size()) differ = true;
  CHECK(differ);
}

TEST_CASE("laya sequence: markers point to MASK tokens") {
  // If the MASK token is missing at option start, markers point to the wrong
  // positions. Every marker must be at a position whose token id is mask_id.
  const auto& tok = FixtureTokenizer();
  const auto sp = FixtureSpecial();
  vllm::laya::Question q;
  q.type = vllm::laya::QType::kChoice;
  q.instructions = "which color";
  q.options = {"red", "blue", "green"};
  q.state = "the car is fast";
  const auto out = vllm::laya::BuildSequence(tok, sp, q);

  REQUIRE(out.markers.size() == 3);
  for (int m : out.markers) {
    CHECK(m >= 0);
    CHECK(static_cast<size_t>(m) < out.ids.size());
    CHECK(out.ids[static_cast<size_t>(m)] == sp.mask_id);
  }
}

TEST_CASE("laya sequence: truncation direction changes the state tokens") {
  // With a long state and small max_len, truncate_left keeps the END of the
  // state while the default keeps the START. The sequences must differ.
  const auto& tok = FixtureTokenizer();
  const auto sp = FixtureSpecial();
  vllm::laya::Question q;
  q.type = vllm::laya::QType::kNoul;
  q.instructions = "is it true";
  q.options = {"false: no", "true: yes"};
  q.state = "the car is fast and red";

  const auto out_right = vllm::laya::BuildSequence(
      tok, sp, q, 25, 18, {}, false);
  const auto out_left = vllm::laya::BuildSequence(
      tok, sp, q, 25, 18, {}, true);

  REQUIRE(out_right.ids.size() == out_left.ids.size());
  bool differ = false;
  for (size_t i = 0; i < out_right.ids.size(); ++i) {
    if (out_right.ids[i] != out_left.ids[i]) {
      differ = true;
      break;
    }
  }
  CHECK(differ);
}

TEST_CASE("laya sequence: mask string in input text is replaced") {
  // If the mask token string ("<mask>") in instructions/options/state is NOT
  // replaced with a space, it leaks as real tokens and the sequence differs.
  // The replacement is verified by checking that no consecutive token sequence
  // spells out "<mask>" (the '<', 'm', 'a', 's', 'k', '>' characters).
  const auto& tok = FixtureTokenizer();
  const auto sp = FixtureSpecial();
  vllm::laya::Question q;
  q.type = vllm::laya::QType::kChoice;
  q.instructions = "which <mask> color";
  q.options = {"red", "blue"};
  q.state = "the <mask> car";
  const auto out = vllm::laya::BuildSequence(tok, sp, q);

  // With replacement, "<mask>" (chars: < m a s k >) becomes spaces. The token
  // for '<' is 60, 'm' is 109, 'a' is 97, 's' is 115, 'k' is 107, '>' is 62.
  // If replacement worked, these 6 bytes should NOT appear consecutively in
  // the ids. Instead, spaces (32) should appear where "<mask>" was.
  // Verify by checking the encoded instructions portion contains spaces (32)
  // where "<mask>" was removed.
  //
  // "which <mask> color" with replacement becomes "which  color" (two spaces).
  // Without replacement, it stays "which <mask> color".
  // The head text is "choice question: which  color" (replaced) vs
  // "choice question: which <mask> color" (not replaced).
  // Check that '<' (60) does not appear in the head portion of the sequence.
  bool has_lt = false;
  for (size_t i = 1; i < out.ids.size(); ++i) {
    if (out.ids[i] == 60) {  // '<'
      has_lt = true;
      break;
    }
  }
  CHECK(!has_lt);
}

// --- Long option truncation tests (Finding 2) ---
// The 48-token option truncation (laya_sequence.cpp:92) was untested. These
// tests use options longer than 48 tokens (60 chars = 61 tokens with the
// leading space) and verify the truncation, max_len compliance, and marker
// positions.

TEST_CASE("laya sequence: long option truncated to 48 tokens") {
  // 60 'a' chars → " " + 60 'a' = 61 tokens. The 48-token limit cuts it to
  // 48. With MASK prepended the option segment is 49 tokens.
  const auto& tok = FixtureTokenizer();
  const auto sp = FixtureSpecial();
  vllm::laya::Question q;
  q.type = vllm::laya::QType::kChoice;
  q.instructions = "which color";
  q.options = {std::string(60, 'a')};
  q.state = "ctx";
  const auto out = vllm::laya::BuildSequence(tok, sp, q, 512, 192);

  // Total sequence must not exceed max_len.
  CHECK(static_cast<int>(out.ids.size()) <= 512);

  // One option → one marker.
  REQUIRE(out.markers.size() == 1);
  const int m = out.markers[0];

  // Marker points to a MASK token.
  CHECK(out.ids[static_cast<size_t>(m)] == sp.mask_id);

  // The option segment is MASK + 48 truncated tokens = 49 tokens.
  // The token right after is the SEP that closes the options section.
  REQUIRE(static_cast<int>(out.ids.size()) > m + 49);
  CHECK(out.ids[static_cast<size_t>(m + 49)] == sp.sep_id);

  // The 48 truncated tokens are the first 48 of " " + 60 'a's:
  // space (32) + 47 'a' (97).
  CHECK(out.ids[static_cast<size_t>(m + 1)] == 32);  // space
  for (int i = 2; i <= 48; ++i) {
    CHECK(out.ids[static_cast<size_t>(m + i)] == 97);  // 'a'
  }
}

TEST_CASE("laya sequence: mixed long and short options have correct markers") {
  // Option 0: 60 chars → truncated to 48 (segment = MASK + 48 = 49).
  // Option 1: "red" → " red" = 4 tokens, not truncated (segment = MASK + 4 = 5).
  const auto& tok = FixtureTokenizer();
  const auto sp = FixtureSpecial();
  vllm::laya::Question q;
  q.type = vllm::laya::QType::kChoice;
  q.instructions = "which color";
  q.options = {std::string(60, 'a'), "red"};
  q.state = "ctx";
  const auto out = vllm::laya::BuildSequence(tok, sp, q, 512, 192);

  CHECK(static_cast<int>(out.ids.size()) <= 512);

  REQUIRE(out.markers.size() == 2);
  const int m0 = out.markers[0];
  const int m1 = out.markers[1];

  // Both markers point to MASK tokens.
  CHECK(out.ids[static_cast<size_t>(m0)] == sp.mask_id);
  CHECK(out.ids[static_cast<size_t>(m1)] == sp.mask_id);

  // Option 0 segment: MASK + 48 truncated = 49 tokens.
  CHECK(m1 - m0 == 49);

  // Option 1 segment: MASK + " red" (4 tokens) = 5 tokens.
  // The token after is the SEP closing the options section.
  REQUIRE(static_cast<int>(out.ids.size()) > m1 + 5);
  CHECK(out.ids[static_cast<size_t>(m1 + 5)] == sp.sep_id);
}
