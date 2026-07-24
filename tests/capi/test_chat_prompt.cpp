// Unit tests for the ABI v3 chat-prompt resolution (src/capi/chat_prompt.*):
// the hermes-aware fallback prompt and the probe-render degradation policy
// for templates the minja subset cannot serve. Pure string-level tests — no
// engine, no disk model.
#include "capi/chat_prompt.h"

#include <doctest/doctest.h>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using vllm::capi::HermesToolsFallbackPrompt;
using vllm::capi::ResolveTemplatePromptFn;
using vllm::entrypoints::openai::ChatCompletionToolsParam;
using vllm::entrypoints::openai::ChatMessage;

namespace {

std::vector<ChatMessage> Msgs() {
  return {ChatMessage{"system", std::string("be brief")},
          ChatMessage{"user", std::string("weather in Rome?")}};
}

std::vector<ChatCompletionToolsParam> WeatherTool() {
  ChatCompletionToolsParam t;
  t.function.name = "get_weather";
  t.function.description = "Get the weather.";
  t.function.parameters = nlohmann::json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}}})");
  return {t};
}

}  // namespace

TEST_CASE("chat_prompt: hermes fallback renders the tools block + role join") {
  const std::string out = HermesToolsFallbackPrompt(Msgs(), true, WeatherTool());
  CHECK(out.find("<tools>") != std::string::npos);
  CHECK(out.find("\"name\":\"get_weather\"") != std::string::npos);
  CHECK(out.find("\"parameters\"") != std::string::npos);
  CHECK(out.find("<tool_call>") != std::string::npos);
  // The conversation follows the tools preamble.
  CHECK(out.find("system: be brief") != std::string::npos);
  CHECK(out.find("user: weather in Rome?") != std::string::npos);
  // Tools block precedes the conversation.
  CHECK(out.find("<tools>") < out.find("system: be brief"));
}

TEST_CASE("chat_prompt: hermes fallback without tools is the plain role join") {
  const std::string out = HermesToolsFallbackPrompt(Msgs(), true, {});
  CHECK(out.find("<tools>") == std::string::npos);
  CHECK(out.find("system: be brief") != std::string::npos);
}

TEST_CASE("chat_prompt: a renderable template is used as-is") {
  const auto fn = ResolveTemplatePromptFn(
      "{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}", "",
      "", "test-origin");
  const std::string out = fn(Msgs(), true, {});
  CHECK(out == "[system]be brief[user]weather in Rome?");
}

TEST_CASE("chat_prompt: an unrenderable template degrades to the hermes fallback") {
  // namespace() is deliberately unsupported by the minja subset (the full
  // Qwen3.5 template's shape), so the probe fails and every request goes to
  // the fallback instead of throwing per request.
  const auto fn = ResolveTemplatePromptFn(
      "{%- set c = namespace(value=0) %}{{ c.value }}", "", "", "test-origin");
  const std::string out = fn(Msgs(), true, WeatherTool());
  CHECK(out.find("<tools>") != std::string::npos);
  CHECK(out.find("user: weather in Rome?") != std::string::npos);
}

TEST_CASE("chat_prompt: a template whose tools branch fails goes hybrid") {
  // Renders fine without tools; the tools branch only fails at EVALUATION
  // (subscripting a member the tool object does not have), because the parser
  // sees the whole template up front - an unsupported-syntax failure would
  // fail both probes and degrade fully instead.
  const auto fn = ResolveTemplatePromptFn(
      "{% if tools %}{{ tools[0]['no_such_member']['deep'] }}{% endif %}"
      "{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}",
      "", "", "test-origin");
  CHECK(fn(Msgs(), true, {}) == "[system]be brief[user]weather in Rome?");
  const std::string with_tools = fn(Msgs(), true, WeatherTool());
  CHECK(with_tools.find("<tools>") != std::string::npos);
  CHECK(with_tools.find("user: weather in Rome?") != std::string::npos);
}
