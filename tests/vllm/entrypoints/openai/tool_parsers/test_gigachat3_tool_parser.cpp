// Tests for the "gigachat3" tool-call parser (GigaChat3ToolParser).
// Ported from: tests/tool_parsers/test_gigachat3_tool_parser.py @ e24d1b24 - the
// SAME fixtures for BOTH header forms (GigaChat 3 "function call<|role_sep|>\n"
// and GigaChat 3.1 "<|function_call|>"): no-tool-call, simple / parameterless /
// complex / mixed-content / mixed+</s> function calls (non-streaming AND
// streaming), the two large-step streaming cases, plus 3 ADDED edges.
//
// The streaming token-id spans are dropped (text-only seam, see gigachat3.h) -
// upstream gigachat3 streaming ALREADY ignores every token id, so this is a pure
// signature trim. Upstream feeds these outputs through a gpt2 tokenizer that has
// the sep/header/</s> literals ADDED as single tokens, so the markers arrive as
// ATOMIC deltas; we reproduce that cadence. Arguments compared as PARSED JSON.
#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"
#include "vllm/entrypoints/openai/tool_parsers/gigachat3.h"

using namespace vllm::entrypoints::openai;

namespace {

const std::string kMsgSep = "<|message_sep|>\n\n";
const std::string kRoleSep = "<|role_sep|>\n";
const std::string kEos = "</s>";
const std::string kHeader3 = "function call" + kRoleSep;
const std::string kHeader31 = "<|function_call|>";

const std::string kSimpleJson =
    R"({"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences"}})";
const std::string kSimpleArgs = R"({"action": "create", "id": "preferences"})";

const std::string kParamlessJson =
    R"({"name": "manage_user_memory", "arguments": {}})";

const std::string kComplexJson =
    R"({"name": "manage_user_memory", "arguments": {"action": "create", "id": "preferences", "content": {"short_answers": true, "hate_emojis": true, "english_ui": false, "russian_math_explanations": true}}})";
const std::string kComplexArgs =
    R"({"action": "create", "id": "preferences", "content": {"short_answers": true, "hate_emojis": true, "english_ui": false, "russian_math_explanations": true}})";

const std::string kContentText = "I'll check that for you.";

ChatCompletionRequest EmptyRequest() { return ChatCompletionRequest{}; }

struct CollectedCall {
  std::string name;
  std::string arguments;
  std::string content;
};

CollectedCall Drive(GigaChat3ToolParser& parser,
                    const std::vector<std::string>& deltas) {
  CollectedCall out;
  std::string prev;
  ChatCompletionRequest req;
  for (const std::string& delta : deltas) {
    const std::string cur = prev + delta;
    auto dm = parser.extract_tool_calls_streaming(prev, cur, delta, req);
    prev = cur;
    if (!dm.has_value()) continue;
    if (dm->content.has_value()) out.content += *dm->content;
    if (!dm->tool_calls.has_value()) continue;
    for (const DeltaToolCall& tc : *dm->tool_calls) {
      if (tc.function.name.has_value()) out.name += *tc.function.name;
      if (tc.function.arguments.has_value()) out.arguments += *tc.function.arguments;
    }
  }
  return out;
}

// Split a string into fixed-size chunks (so the name lands before the args are
// complete, exercising the name-first-then-arguments-diff path).
std::vector<std::string> Chunks(const std::string& s, std::size_t n) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i < s.size(); i += n) out.push_back(s.substr(i, n));
  return out;
}

void CheckNonStreaming(const std::string& output, const std::string& expected_args,
                       const std::optional<std::string>& expected_content) {
  GigaChat3ToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls(output, req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].type == "function");
  CHECK(r.tool_calls[0].function.name == "manage_user_memory");
  CHECK(nlohmann::json::parse(r.tool_calls[0].function.arguments) ==
        nlohmann::json::parse(expected_args));
  if (expected_content.has_value()) {
    REQUIRE(r.content.has_value());
    CHECK(*r.content == *expected_content);
  } else {
    CHECK_FALSE(r.content.has_value());
  }
}

}  // namespace

// ─── Non-streaming ──────────────────────────────────────────────────────────

TEST_CASE("gigachat3: no tool call") {
  GigaChat3ToolParser parser;
  auto req = EmptyRequest();
  auto r = parser.extract_tool_calls("How can I help you today?", req);
  CHECK_FALSE(r.tools_called);
  CHECK(r.tool_calls.empty());
  REQUIRE(r.content.has_value());
  CHECK(*r.content == "How can I help you today?");
}

TEST_CASE("gigachat3: function calls (both header forms)") {
  // GigaChat 3 form.
  CheckNonStreaming(kMsgSep + kHeader3 + kSimpleJson, kSimpleArgs, std::nullopt);
  CheckNonStreaming(kMsgSep + kHeader3 + kParamlessJson, "{}", std::nullopt);
  CheckNonStreaming(kMsgSep + kHeader3 + kComplexJson, kComplexArgs, std::nullopt);
  CheckNonStreaming(kContentText + kMsgSep + kHeader3 + kSimpleJson, kSimpleArgs,
                    kContentText);
  CheckNonStreaming(kContentText + kMsgSep + kHeader3 + kSimpleJson + kEos,
                    kSimpleArgs, kContentText);
  // GigaChat 3.1 form.
  CheckNonStreaming(kHeader31 + kSimpleJson, kSimpleArgs, std::nullopt);
  CheckNonStreaming(kHeader31 + kParamlessJson, "{}", std::nullopt);
  CheckNonStreaming(kHeader31 + kComplexJson, kComplexArgs, std::nullopt);
  CheckNonStreaming(kContentText + kHeader31 + kSimpleJson, kSimpleArgs,
                    kContentText);
  CheckNonStreaming(kContentText + kHeader31 + kSimpleJson + kEos, kSimpleArgs,
                    kContentText);
}

// ─── Streaming ──────────────────────────────────────────────────────────────

TEST_CASE("gigachat3 streaming: large steps (gigachat3 header)") {
  GigaChat3ToolParser parser;
  auto r = Drive(parser, {kContentText.substr(0, 3), kContentText.substr(3, 2),
                          kContentText.substr(5), kMsgSep, kHeader3,
                          kComplexJson.substr(0, 40),
                          kComplexJson.substr(40, kComplexJson.size() - 41),
                          kComplexJson.substr(kComplexJson.size() - 1)});
  CHECK(r.name == "manage_user_memory");
  CHECK(nlohmann::json::parse(r.arguments) == nlohmann::json::parse(kComplexArgs));
}

TEST_CASE("gigachat3 streaming: large steps (gigachat31 header)") {
  GigaChat3ToolParser parser;
  auto r = Drive(parser, {kContentText.substr(0, 3), kContentText.substr(3, 2),
                          kContentText.substr(5), kHeader31,
                          kComplexJson.substr(0, 40),
                          kComplexJson.substr(40, kComplexJson.size() - 41),
                          kComplexJson.substr(kComplexJson.size() - 1)});
  CHECK(r.name == "manage_user_memory");
  CHECK(nlohmann::json::parse(r.arguments) == nlohmann::json::parse(kComplexArgs));
}

TEST_CASE("gigachat3 streaming: simple with atomic markers") {
  GigaChat3ToolParser parser;
  std::vector<std::string> deltas = {kMsgSep, kHeader3};
  for (auto& c : Chunks(kSimpleJson, 12)) deltas.push_back(c);
  auto r = Drive(parser, deltas);
  CHECK(r.name == "manage_user_memory");
  CHECK(nlohmann::json::parse(r.arguments) == nlohmann::json::parse(kSimpleArgs));
}

TEST_CASE("gigachat3 streaming edge: parameterless call") {
  GigaChat3ToolParser parser;
  std::vector<std::string> deltas = {kHeader31};
  for (auto& c : Chunks(kParamlessJson, 10)) deltas.push_back(c);
  auto r = Drive(parser, deltas);
  CHECK(r.name == "manage_user_memory");
  CHECK(nlohmann::json::parse(r.arguments) == nlohmann::json::object());
}

TEST_CASE("gigachat3 streaming edge: mixed content then call") {
  GigaChat3ToolParser parser;
  std::vector<std::string> deltas = {kContentText, kMsgSep, kHeader3};
  for (auto& c : Chunks(kSimpleJson, 12)) deltas.push_back(c);
  auto r = Drive(parser, deltas);
  CHECK(r.content == kContentText);
  CHECK(r.name == "manage_user_memory");
  CHECK(nlohmann::json::parse(r.arguments) == nlohmann::json::parse(kSimpleArgs));
}

TEST_CASE("gigachat3 streaming edge: trailing </s> after args") {
  GigaChat3ToolParser parser;
  std::vector<std::string> deltas = {kHeader31};
  for (auto& c : Chunks(kSimpleJson, 12)) deltas.push_back(c);
  deltas.push_back(kEos);
  auto r = Drive(parser, deltas);
  CHECK(r.name == "manage_user_memory");
  CHECK(nlohmann::json::parse(r.arguments) == nlohmann::json::parse(kSimpleArgs));
}

// ─── Factory ────────────────────────────────────────────────────────────────

TEST_CASE("Factory: get_tool_parser(\"gigachat3\") works") {
  auto parser = get_tool_parser("gigachat3");
  REQUIRE(parser != nullptr);
  auto req = EmptyRequest();
  auto r = parser->extract_tool_calls(kHeader31 + kSimpleJson, req);
  CHECK(r.tools_called);
  REQUIRE(r.tool_calls.size() == 1);
  CHECK(r.tool_calls[0].function.name == "manage_user_memory");
}
