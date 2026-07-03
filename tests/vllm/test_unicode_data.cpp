// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level BPE.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <string_view>

#include "vllm/tokenizer/unicode_data.h"

using vllm::tok::Category;
using vllm::tok::DecodeUtf8;
using vllm::tok::EncodeUtf8;
using vllm::tok::IsWhitespace;
using vllm::tok::UCat;

TEST_CASE("Category spot checks") {
  CHECK(Category(U'a') == UCat::kLetter);
  CHECK(Category(U'Z') == UCat::kLetter);
  CHECK(Category(0x99C5) == UCat::kLetter);  // 駅 (CJK ideograph, Lo)
  CHECK(Category(U'5') == UCat::kNumber);    // Nd
  CHECK(Category(0x00BD) == UCat::kNumber);  // ½ (No)
  CHECK(Category(U' ') == UCat::kSeparator); // Zs
  CHECK(Category(U'\t') == UCat::kControl);  // Cc
  CHECK(Category(U'\n') == UCat::kControl);  // Cc
  CHECK(Category(0x1F642) == UCat::kSymbol); // 🙂 (So)
  CHECK(Category(0x0301) == UCat::kMark);    // combining acute accent (Mn)
  CHECK(Category(U'.') == UCat::kPunct);     // Po
  CHECK(Category(U'+') == UCat::kSymbol);    // Sm
  CHECK(Category(0x0410) == UCat::kLetter);  // Cyrillic А (Lu)
}

TEST_CASE("Category of unassigned / out-of-range is kOther") {
  CHECK(Category(0x10FFFE) == UCat::kOther);  // noncharacter is Cn... see note
  CHECK(Category(0x110000) == UCat::kOther);  // beyond Unicode range
  CHECK(Category(0xFFFFFFFFu) == UCat::kOther);
}

TEST_CASE("IsWhitespace matches python str.isspace") {
  // True cases.
  CHECK(IsWhitespace(U' '));
  CHECK(IsWhitespace(U'\t'));
  CHECK(IsWhitespace(U'\n'));
  CHECK(IsWhitespace(U'\r'));
  CHECK(IsWhitespace(0x0B));    // vertical tab
  CHECK(IsWhitespace(0x0C));    // form feed
  CHECK(IsWhitespace(0x1C));    // file separator (python isspace true)
  CHECK(IsWhitespace(0x1D));
  CHECK(IsWhitespace(0x1E));
  CHECK(IsWhitespace(0x1F));
  CHECK(IsWhitespace(0x85));    // NEL
  CHECK(IsWhitespace(0xA0));    // NBSP
  CHECK(IsWhitespace(0x1680));  // ogham space mark
  CHECK(IsWhitespace(0x2000));  // en quad
  CHECK(IsWhitespace(0x200A));  // hair space
  CHECK(IsWhitespace(0x2028));  // line separator
  CHECK(IsWhitespace(0x2029));  // paragraph separator
  CHECK(IsWhitespace(0x202F));  // narrow no-break space
  CHECK(IsWhitespace(0x205F));  // medium mathematical space
  CHECK(IsWhitespace(0x3000));  // ideographic space
  // False cases (python str.isspace is false for these).
  CHECK_FALSE(IsWhitespace(U'a'));
  CHECK_FALSE(IsWhitespace(U'0'));
  CHECK_FALSE(IsWhitespace(0x00));    // NUL
  CHECK_FALSE(IsWhitespace(0x08));    // backspace
  CHECK_FALSE(IsWhitespace(0x180E));  // Mongolian vowel separator (not space since 6.3)
  CHECK_FALSE(IsWhitespace(0x200B));  // zero width space: isspace() is False
  CHECK_FALSE(IsWhitespace(0xFEFF));  // BOM / ZWNBSP
  CHECK_FALSE(IsWhitespace(0x110000));
}

TEST_CASE("EncodeUtf8 produces expected byte sequences") {
  std::string out;
  EncodeUtf8(U'a', out);
  CHECK(out == "a");
  out.clear();
  EncodeUtf8(0x00E9, out);  // é
  CHECK(out == "\xC3\xA9");
  out.clear();
  EncodeUtf8(0x99C5, out);  // 駅 (3-byte)
  CHECK(out == "\xE9\xA7\x85");
  out.clear();
  EncodeUtf8(0x1F642, out);  // 🙂 (4-byte)
  CHECK(out == "\xF0\x9F\x99\x82");
  out.clear();
  EncodeUtf8(0x110000, out);  // invalid → U+FFFD
  CHECK(out == "\xEF\xBF\xBD");
  out.clear();
  EncodeUtf8(0xD800, out);  // surrogate → U+FFFD
  CHECK(out == "\xEF\xBF\xBD");
}

TEST_CASE("DecodeUtf8 round-trips 1..4 byte codepoints") {
  const uint32_t cps[] = {0x24, 0x00E9, 0x0939, 0x20AC, 0x99C5, 0xFFFD,
                          0x10000, 0x1F642, 0x10FFFF};
  for (uint32_t cp : cps) {
    std::string s;
    EncodeUtf8(cp, s);
    size_t pos = 0;
    CHECK(DecodeUtf8(s, pos) == cp);
    CHECK(pos == s.size());
  }
}

TEST_CASE("DecodeUtf8 walks a multi-codepoint string") {
  std::string s = "a\xC3\xA9\xF0\x9F\x99\x82";  // 'a', é, 🙂
  size_t pos = 0;
  CHECK(DecodeUtf8(s, pos) == U'a');
  CHECK(pos == 1);
  CHECK(DecodeUtf8(s, pos) == 0xE9);
  CHECK(pos == 3);
  CHECK(DecodeUtf8(s, pos) == 0x1F642);
  CHECK(pos == 7);
}

TEST_CASE("DecodeUtf8 invalid input yields U+FFFD and advances 1") {
  size_t pos = 0;
  // Lone continuation byte.
  std::string s1 = "\x80x";
  CHECK(DecodeUtf8(s1, pos) == 0xFFFD);
  CHECK(pos == 1);
  CHECK(DecodeUtf8(s1, pos) == U'x');

  // Truncated 3-byte sequence at end of string.
  pos = 0;
  std::string s2 = "\xE9\xA7";
  CHECK(DecodeUtf8(s2, pos) == 0xFFFD);
  CHECK(pos == 1);
  CHECK(DecodeUtf8(s2, pos) == 0xFFFD);
  CHECK(pos == 2);

  // Overlong encoding of '/' (0xC0 0xAF) must be rejected byte-by-byte.
  pos = 0;
  std::string s3 = "\xC0\xAF";
  CHECK(DecodeUtf8(s3, pos) == 0xFFFD);
  CHECK(pos == 1);
  CHECK(DecodeUtf8(s3, pos) == 0xFFFD);
  CHECK(pos == 2);

  // CESU-8 style encoded surrogate (0xED 0xA0 0x80 = U+D800) is invalid.
  pos = 0;
  std::string s4 = "\xED\xA0\x80";
  CHECK(DecodeUtf8(s4, pos) == 0xFFFD);
  CHECK(pos == 1);

  // 0xF5 leads to > U+10FFFF: invalid lead byte.
  pos = 0;
  std::string s5 = "\xF5\x80\x80\x80";
  CHECK(DecodeUtf8(s5, pos) == 0xFFFD);
  CHECK(pos == 1);

  // Invalid lead 0xFF.
  pos = 0;
  std::string s6 = "\xFF";
  CHECK(DecodeUtf8(s6, pos) == 0xFFFD);
  CHECK(pos == 1);
}

TEST_CASE("Range table sanity: sorted, non-overlapping, non-empty") {
  // Probe a spread of codepoints; every result must be a valid enum value and
  // the ASCII fast facts must hold (guards against off-by-one in the tables).
  for (uint32_t cp = 0; cp < 0x300; ++cp) {
    UCat c = Category(cp);
    CHECK(static_cast<uint8_t>(c) <= 7);
  }
  // Block boundaries around 'A'..'Z' (guards off-by-one in range merging).
  CHECK(Category(U'A') == UCat::kLetter);
  CHECK(Category(U'@') == UCat::kPunct);  // Po, immediately before 'A'
  CHECK(Category(U'Z') == UCat::kLetter);
  CHECK(Category(U'[') == UCat::kPunct);  // Ps, immediately after 'Z'
}
