// Unit tests for SanitizeUtf8 (M3.1 Task 4) — the serving-boundary UTF-8
// sanitizer that keeps a raw-byte detokenizer output from making
// nlohmann::json::dump() throw (→ HTTP 500). Asserts: valid text is unchanged;
// each maximal invalid subpart → one U+FFFD; and a synthetically-invalid
// completion/chat response serializes without throwing.
#include "vllm/entrypoints/openai/serving_utils.h"

#include <doctest/doctest.h>

#include <string>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"

using vllm::entrypoints::openai::SanitizeUtf8;

namespace {
const std::string kReplacement = "\xEF\xBF\xBD";  // U+FFFD
}

TEST_CASE("SanitizeUtf8: valid ASCII + multibyte text is unchanged") {
  CHECK(SanitizeUtf8("") == "");
  CHECK(SanitizeUtf8("hello world") == "hello world");
  CHECK(SanitizeUtf8("caf\xC3\xA9") == "caf\xC3\xA9");        // café (2-byte)
  CHECK(SanitizeUtf8("\xE2\x82\xAC") == "\xE2\x82\xAC");      // € (3-byte)
  CHECK(SanitizeUtf8("\xF0\x9F\x98\x80") == "\xF0\x9F\x98\x80");  // 😀 (4-byte)
  // A literal U+FFFD is already valid UTF-8 and must pass through untouched.
  CHECK(SanitizeUtf8(kReplacement) == kReplacement);
}

TEST_CASE("SanitizeUtf8: a lone continuation / invalid lead byte → one U+FFFD") {
  CHECK(SanitizeUtf8("\x80") == kReplacement);       // stray continuation
  CHECK(SanitizeUtf8("\xFF") == kReplacement);       // invalid lead
  CHECK(SanitizeUtf8("a\xFF"
                     "b") == "a" + kReplacement + "b");
}

TEST_CASE("SanitizeUtf8: a truncated multibyte sequence → one U+FFFD") {
  // Lead byte of a 4-byte sequence with no continuations (the classic split
  // emoji across DELTA chunks our raw-byte detokenizer can emit).
  CHECK(SanitizeUtf8("\xF0") == kReplacement);
  CHECK(SanitizeUtf8("\xF0\x9F") == kReplacement);       // 2 of 4 bytes
  CHECK(SanitizeUtf8("\xF0\x9F\x98") == kReplacement);   // 3 of 4 bytes
  // A 2-byte lead followed by a non-continuation: maximal subpart is the lead.
  CHECK(SanitizeUtf8("\xC3z") == kReplacement + "z");
  // Valid text, then a truncated tail.
  CHECK(SanitizeUtf8("hi\xF0\x9F") == "hi" + kReplacement);
}

TEST_CASE("SanitizeUtf8: overlong / surrogate encodings → U+FFFD per subpart") {
  // 0xE0 0x80 ... is an overlong form: the first continuation 0x80 is below the
  // valid 0xA0 floor, so the lead is its own maximal invalid subpart.
  CHECK(SanitizeUtf8("\xE0\x80\x80") ==
        kReplacement + kReplacement + kReplacement);
  // UTF-8-encoded surrogate D800 (0xED 0xA0 0x80): 0xA0 exceeds the 0x9F ceiling
  // 0xED enforces → three invalid subparts.
  CHECK(SanitizeUtf8("\xED\xA0\x80") ==
        kReplacement + kReplacement + kReplacement);
}

TEST_CASE("SanitizeUtf8: sanitized text always dumps as valid JSON") {
  // The 500-repro: a raw-byte response text with an invalid multibyte run would
  // make nlohmann::json::dump() throw. After SanitizeUtf8 it must serialize.
  const std::string bad = "hello\xF0\x9F world\xFF!";  // split emoji + lone FF

  // Completion response.
  {
    vllm::entrypoints::openai::CompletionResponse resp;
    resp.id = "cmpl-0";
    resp.model = "m";
    vllm::entrypoints::openai::CompletionResponseChoice choice;
    choice.text = SanitizeUtf8(bad);
    choice.finish_reason = "stop";
    resp.choices.push_back(choice);
    CHECK_NOTHROW(nlohmann::json(resp).dump());
  }

  // Chat response.
  {
    vllm::entrypoints::openai::ChatCompletionResponse resp;
    resp.id = "chatcmpl-0";
    resp.model = "m";
    vllm::entrypoints::openai::ChatCompletionResponseChoice choice;
    choice.message.role = "assistant";
    choice.message.content = SanitizeUtf8(bad);
    choice.finish_reason = "stop";
    resp.choices.push_back(choice);
    CHECK_NOTHROW(nlohmann::json(resp).dump());
  }

  // Sanity: dumping the RAW invalid text WOULD throw (proves the guard matters).
  {
    vllm::entrypoints::openai::CompletionResponse resp;
    resp.id = "cmpl-0";
    resp.model = "m";
    vllm::entrypoints::openai::CompletionResponseChoice choice;
    choice.text = bad;  // NOT sanitized
    resp.choices.push_back(choice);
    CHECK_THROWS(nlohmann::json(resp).dump());
  }
}
