// Ported from deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
// encoding/test_encoding_dsv4.py at revision
// 86f746b36186f0e567729a5c06a8c918caba82a9. Every upstream case and failure
// is preserved below.
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/multimodal/deepseek_v4_processor.h"

namespace {

using Json = nlohmann::ordered_json;
using vllm::multimodal::DeepSeekV4EncodedPrompt;
using vllm::multimodal::EncodeDeepSeekV4Messages;
using vllm::multimodal::ParseDeepSeekV4TaggedText;

constexpr const char* kImagePlaceholder = "<｜deepseek_image｜>";

Json Message(const std::string& role, Json content) {
  return Json{{"role", role}, {"content", std::move(content)}};
}

DeepSeekV4EncodedPrompt Encode(const Json& messages,
                               const Json& context = Json::array()) {
  return EncodeDeepSeekV4Messages(messages, "chat", context);
}

}  // namespace

TEST_CASE("deepseek-v4 encoding keeps a plain text prompt unchanged") {
  const auto encoded = Encode(Json::array({Message("user", "hello")}));
  CHECK(encoded.prompt ==
        "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>");
  CHECK(encoded.images.empty());
}

TEST_CASE("deepseek-v4 encoding emits every pinned task transition") {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"action",
       "<｜begin▁of▁sentence｜><｜User｜>classify"
       "<｜Assistant｜></think><｜action｜>"},
      {"query", "<｜begin▁of▁sentence｜><｜User｜>classify<｜query｜>"},
      {"authority",
       "<｜begin▁of▁sentence｜><｜User｜>classify<｜authority｜>"},
      {"domain", "<｜begin▁of▁sentence｜><｜User｜>classify<｜domain｜>"},
      {"title", "<｜begin▁of▁sentence｜><｜User｜>classify<｜title｜>"},
      {"read_url",
       "<｜begin▁of▁sentence｜><｜User｜>classify<｜read_url｜>"},
  };
  for (const auto& [task, expected] : cases) {
    CAPTURE(task);
    Json message = Message("user", "classify");
    message["task"] = task;
    CHECK(Encode(Json::array({message})).prompt == expected);
  }
}

TEST_CASE("deepseek-v4 encoding rejects non-string tasks") {
  const std::vector<Json> invalid = {
      42,
      true,
      Json::array({"action"}),
      Json{{"name", "action"}},
  };
  for (const Json& task : invalid) {
    CAPTURE(task);
    Json message = Message("user", "classify");
    message["task"] = task;
    CHECK_THROWS_AS(Encode(Json::array({message})), std::invalid_argument);
  }
}

TEST_CASE("deepseek-v4 encoding rejects invalid tasks before transitions") {
  Json message = Message("user", "classify");
  message["task"] = "invalid";
  CHECK_THROWS_WITH_AS(
      Encode(Json::array({message, Message("user", "next")})),
      "Invalid task: 'invalid'", std::invalid_argument);
}

TEST_CASE("deepseek-v4 encoding treats a null task as absent") {
  Json user = Message("user", "q");
  user["task"] = nullptr;
  Json assistant = Message("assistant", "a");
  assistant["reasoning_content"] = "r";
  CHECK(EncodeDeepSeekV4Messages(
            Json::array({user, assistant}), "thinking", Json::array(), false)
            .prompt ==
        "<｜begin▁of▁sentence｜><｜User｜>q<｜Assistant｜><think>"
        "r</think>a<｜end▁of▁sentence｜>");
}

TEST_CASE("deepseek-v4 encoding keeps a multiturn text prompt unchanged") {
  const Json messages = Json::array({
      Message("system", "sys"), Message("user", "q1"),
      Message("assistant", "a1"), Message("user", "q2")});
  CHECK(Encode(messages).prompt ==
        "<｜begin▁of▁sentence｜>sys<｜User｜>q1<｜Assistant｜></think>"
        "a1<｜end▁of▁sentence｜><｜User｜>q2<｜Assistant｜></think>");
}

TEST_CASE("deepseek-v4 thinking drops historical reasoning by default") {
  Json assistant = Message("assistant", "a1");
  assistant["reasoning_content"] = "r1";
  const Json messages = Json::array(
      {Message("user", "q1"), assistant, Message("user", "q2")});
  CHECK(EncodeDeepSeekV4Messages(messages, "thinking").prompt ==
        "<｜begin▁of▁sentence｜><｜User｜>q1<｜Assistant｜></think>"
        "a1<｜end▁of▁sentence｜><｜User｜>q2<｜Assistant｜><think>");
}

TEST_CASE("deepseek-v4 thinking retains historical reasoning when requested") {
  Json assistant = Message("assistant", "a1");
  assistant["reasoning_content"] = "r1";
  const Json messages = Json::array(
      {Message("developer", "q1"), assistant, Message("user", "q2")});
  CHECK(EncodeDeepSeekV4Messages(messages, "thinking", Json::array(), false)
            .prompt ==
        "<｜begin▁of▁sentence｜><｜User｜>q1<｜Assistant｜><think>"
        "r1</think>a1<｜end▁of▁sentence｜><｜User｜>q2"
        "<｜Assistant｜><think>");
}

TEST_CASE("deepseek-v4 encoding returns one placeholder and image record") {
  const Json content = Json::array({
      Json{{"type", "image_url"},
           {"image_url", Json{{"url", "images/image_1.jpeg"}}}},
      Json{{"type", "text"}, {"text", "describe"}}});
  const auto encoded = Encode(Json::array({Message("user", content)}));
  CHECK(encoded.prompt ==
        "<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>\n\n"
        "describe<｜Assistant｜></think>");
  CHECK(encoded.images.size() == 1);
  CHECK(encoded.images[0].at("url") == "images/image_1.jpeg");
}

TEST_CASE("deepseek-v4 tagged text equals standard image content blocks") {
  const Json tagged = ParseDeepSeekV4TaggedText(
      "before<image>images/image_1.jpeg</image>after");
  const Json standard = Json::array({
      Json{{"type", "text"}, {"text", "before"}},
      Json{{"type", "image_url"},
           {"image_url", Json{{"url", "images/image_1.jpeg"}}}},
      Json{{"type", "text"}, {"text", "after"}}});
  const auto tagged_encoded =
      Encode(Json::array({Message("user", tagged)}));
  const auto standard_encoded =
      Encode(Json::array({Message("user", standard)}));
  CHECK(tagged_encoded.prompt == standard_encoded.prompt);
  CHECK(tagged_encoded.images == standard_encoded.images);
}

TEST_CASE("deepseek-v4 tagged text preserves multiple image order") {
  const Json content = ParseDeepSeekV4TaggedText(
      "<image>first.png</image>middle<image>second.png</image>");
  const auto encoded = Encode(Json::array({Message("user", content)}));
  size_t placeholders = 0;
  for (size_t pos = encoded.prompt.find(kImagePlaceholder);
       pos != std::string::npos;
       pos = encoded.prompt.find(kImagePlaceholder, pos + 1)) {
    ++placeholders;
  }
  CHECK(placeholders == 2);
  REQUIRE(encoded.images.size() == 2);
  CHECK(encoded.images[0].at("url") == "first.png");
  CHECK(encoded.images[1].at("url") == "second.png");
}

TEST_CASE("deepseek-v4 pinned TXT and JSON examples encode identically") {
  const std::string text =
      "请按“第一张、第二张”的顺序回答：第一张图"
      "<image>examples/images/carrots.jpeg</image>和第二张图"
      "<image>examples/images/corn.jpeg</image>"
      "中分别是什么食材？它们通常食用的部位分别是什么？";
  const Json standard = Json::array({
      Json{{"type", "text"},
           {"text", "请按“第一张、第二张”的顺序回答：第一张图"}},
      Json{{"type", "image_url"},
           {"image_url", Json{{"url", "examples/images/carrots.jpeg"}}}},
      Json{{"type", "text"}, {"text", "和第二张图"}},
      Json{{"type", "image_url"},
           {"image_url", Json{{"url", "examples/images/corn.jpeg"}}}},
      Json{{"type", "text"},
           {"text", "中分别是什么食材？它们通常食用的部位分别是什么？"}}});
  const auto txt = Encode(Json::array(
      {Message("user", ParseDeepSeekV4TaggedText(text))}));
  const auto json = Encode(Json::array({Message("user", standard)}));
  CHECK(txt.prompt == json.prompt);
  CHECK(txt.images == json.images);
  REQUIRE(txt.images.size() == 2);
  CHECK(txt.images[0].at("url") == "examples/images/carrots.jpeg");
  CHECK(txt.images[1].at("url") == "examples/images/corn.jpeg");
}

TEST_CASE("deepseek-v4 malformed tagged text is rejected") {
  CHECK_THROWS_WITH_AS(ParseDeepSeekV4TaggedText("<image>missing end tag"),
                       "Malformed <image>path</image> tag",
                       std::invalid_argument);
}

TEST_CASE("deepseek-v4 nested tool result preserves image placeholder") {
  const Json nested = Json::array({
      Json{{"type", "image_url"},
           {"image_url", Json{{"url", "images/image_1.jpeg"}}}},
      Json{{"type", "text"}, {"text", "nested"}}});
  const Json content = Json::array({
      Json{{"type", "tool_result"}, {"tool_use_id", "call-1"},
           {"content", nested}}});
  const auto encoded = Encode(Json::array({Message("user", content)}));
  CHECK(encoded.prompt.find(
            "<tool_result><｜deepseek_image｜>\n\nnested</tool_result>") !=
        std::string::npos);
  CHECK(encoded.images.size() == 1);
}

TEST_CASE("deepseek-v4 tool role with image blocks preserves placeholder") {
  const Json calls = Json::array({
      Json{{"id", "call-1"}, {"type", "function"},
           {"function", Json{{"name", "inspect"}, {"arguments", "{}"}}}}});
  Json assistant = Message("assistant", "");
  assistant["tool_calls"] = calls;
  Json tool = Message(
      "tool", Json::array({
                  Json{{"type", "image_url"},
                       {"image_url",
                        Json{{"url", "images/image_1.jpeg"}}}},
                  Json{{"type", "text"}, {"text", "tool image"}}}));
  tool["tool_call_id"] = "call-1";
  const auto encoded = Encode(Json::array({assistant, tool}));
  CHECK(encoded.prompt.find(
            "<tool_result><｜deepseek_image｜>\n\ntool image</tool_result>") !=
        std::string::npos);
  CHECK(encoded.images.size() == 1);
}

TEST_CASE("deepseek-v4 context images are excluded from current media") {
  const Json context = Json::array({Message(
      "user", Json::array({
                  Json{{"type", "image_url"},
                       {"image_url",
                        Json{{"url", "images/image_1.jpeg"}}}},
                  Json{{"type", "text"}, {"text", "previous"}}}))});
  const auto encoded =
      Encode(Json::array({Message("user", "now")}), context);
  CHECK(encoded.prompt.find(kImagePlaceholder) == std::string::npos);
  CHECK(encoded.images.empty());
}

TEST_CASE("deepseek-v4 user supplied placeholder is rejected") {
  CHECK_THROWS_WITH_AS(
      Encode(Json::array({Message("user", kImagePlaceholder)})),
      "Message content contains image special token '<｜deepseek_image｜>'. "
      "Images should be provided as image content blocks.",
      std::invalid_argument);
}

TEST_CASE("deepseek-v4 image block without source is rejected") {
  const Json content = Json::array(
      {Json{{"type", "image_url"}, {"image_url", Json::object()}}});
  CHECK_THROWS_WITH_AS(
      Encode(Json::array({Message("user", content)})),
      "Image block does not contain a valid source", std::invalid_argument);
}

TEST_CASE("deepseek-v4 falsey image sources use the pinned missing-source error") {
  const std::vector<Json> blocks = {
      Json{{"type", "image"}, {"source", Json::object()}},
      Json{{"type", "image"}, {"source", Json::array()}},
      Json{{"type", "image"}, {"source", ""}},
      Json{{"type", "image"}, {"url", Json::object()}},
      Json{{"type", "image"}, {"url", Json::array()}},
      Json{{"type", "image"}, {"url", ""}},
      Json{{"type", "image"}, {"data", Json::object()}},
      Json{{"type", "image"}, {"data", Json::array()}},
      Json{{"type", "image"}, {"data", ""}},
      Json{{"type", "image_url"}, {"image_url", Json::array()}},
  };
  for (const Json& block : blocks) {
    const Json content = Json::array({block});
    CHECK_THROWS_WITH_AS(
        Encode(Json::array({Message("user", content)})),
        "Image block does not contain a valid source", std::invalid_argument);
  }
}

TEST_CASE("deepseek-v4 text content block placeholder is rejected") {
  const Json content = Json::array({
      Json{{"type", "text"},
           {"text", std::string("bad ") + kImagePlaceholder}}});
  CHECK_THROWS_WITH_AS(
      Encode(Json::array({Message("user", content)})),
      "Text block contains image placeholder '<｜deepseek_image｜>': 'bad "
      "<｜deepseek_image｜>'. Images should be separate content blocks.",
      std::invalid_argument);
}

TEST_CASE("deepseek-v4 system tools and response format follow the pinned template") {
  Json system = Message("system", "sys");
  system["tools"] = Json::array({
      Json{{"type", "function"},
           {"function", Json{{"name", "inspect"},
                              {"description", "inspect an image"},
                              {"parameters", Json{{"type", "object"}}}}}}});
  system["response_format"] = Json{{"type", "json_object"}};
  const auto encoded = Encode(Json::array({system, Message("user", "go")}));
  CHECK(encoded.prompt.find("## Tools\n\nYou have access to a set of tools") !=
        std::string::npos);
  CHECK(encoded.prompt.find("{\"name\": \"inspect\"") !=
        std::string::npos);
  CHECK(encoded.prompt.find(
            "## Response Format:\n\nYou MUST strictly adhere to the following "
            "schema to reply:\n{\"type\": \"json_object\"}") !=
        std::string::npos);
}
