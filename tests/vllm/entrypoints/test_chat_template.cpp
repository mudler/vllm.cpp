// Unit tests for the minja-subset chat-template renderer (M3.1 Task 3 / M3.2).
//
// The KEY oracle (cases below marked "GOLDEN") is a transformers
// `apply_chat_template` dump of the pinned Qwen3 chat_template
// (models--Qwen--Qwen3-30B-A3B tokenizer_config.json — the Qwen3-family
// template the gate model Qwen3.6 shares) for a no-tools/no-thinking
// conversation. For that path the full upstream template is byte-identical to
// the subset template exercised here; the expected strings were captured with:
//   AutoTokenizer.from_pretrained(qwen3).apply_chat_template(
//       msgs, tokenize=False, add_generation_prompt=...)
// and re-verified to render identically from the subset template via jinja2
// configured exactly as transformers does (trim_blocks=True, lstrip_blocks=True).
#include "vllm/entrypoints/chat_template.h"

#include <doctest/doctest.h>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"

using vllm::entrypoints::apply_chat_template;
using vllm::entrypoints::ChatTemplateError;
using vllm::entrypoints::MakeChatTemplatePromptFn;
using vllm::entrypoints::openai::ChatCompletionToolsParam;
using vllm::entrypoints::openai::ChatMessage;

namespace {

// The minja-subset Qwen chat template. Uses only supported constructs (for / if
// / interpolation / `+` concat / member access / `-` whitespace trim) and
// reproduces the full upstream Qwen3 template's output on the no-tools path.
const char* kQwenTemplate =
    "{%- for message in messages %}\n"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' "
    "+ '\\n' }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "{{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif %}";

std::vector<ChatMessage> SystemUser() {
  return {ChatMessage{"system", std::string("You are a helpful assistant.")},
          ChatMessage{"user", std::string("Hello, who are you?")}};
}

std::vector<ChatMessage> MultiTurn() {
  return {ChatMessage{"system", std::string("You are a helpful assistant.")},
          ChatMessage{"user", std::string("Hello, who are you?")},
          ChatMessage{"assistant", std::string("I am Qwen.")},
          ChatMessage{"user", std::string("What can you do?")}};
}

}  // namespace

// ─── (b) GOLDEN: [system,user], add_generation_prompt=true ───────────────────
TEST_CASE("chat_template: Qwen3 [system,user] with generation prompt (GOLDEN)") {
  const std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello, who are you?<|im_end|>\n"
      "<|im_start|>assistant\n";
  CHECK(apply_chat_template(kQwenTemplate, SystemUser(),
                            /*add_generation_prompt=*/true) == expected);
}

// ─── (d) add_generation_prompt=false drops the assistant header ──────────────
TEST_CASE("chat_template: Qwen3 [system,user] without generation prompt (GOLDEN)") {
  const std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello, who are you?<|im_end|>\n";
  CHECK(apply_chat_template(kQwenTemplate, SystemUser(),
                            /*add_generation_prompt=*/false) == expected);
}

// ─── (c) GOLDEN: multi-turn [system,user,assistant,user] ─────────────────────
TEST_CASE("chat_template: Qwen3 multi-turn conversation (GOLDEN)") {
  const std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello, who are you?<|im_end|>\n"
      "<|im_start|>assistant\nI am Qwen.<|im_end|>\n"
      "<|im_start|>user\nWhat can you do?<|im_end|>\n"
      "<|im_start|>assistant\n";
  CHECK(apply_chat_template(kQwenTemplate, MultiTurn(),
                            /*add_generation_prompt=*/true) == expected);
}

// ─── ChatPromptFn adapter drives the same render through Task 2's seam ───────
TEST_CASE("chat_template: MakeChatTemplatePromptFn adapts to the ChatPromptFn seam") {
  auto fn = MakeChatTemplatePromptFn(kQwenTemplate);
  const std::string expected =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello, who are you?<|im_end|>\n"
      "<|im_start|>assistant\n";
  CHECK(fn(SystemUser(), /*add_generation_prompt=*/true, /*tools=*/{}) ==
        expected);
}

// ─── (a) interpolation + `+` concat ──────────────────────────────────────────
TEST_CASE("chat_template: interpolation and string concat") {
  CHECK(apply_chat_template("{{ 'a' + 'b' }}", {}, false) == "ab");
  CHECK(apply_chat_template("{{ 'x' ~ 'y' ~ 'z' }}", {}, false) == "xyz");
  CHECK(apply_chat_template("[{{ bos_token }}]", {}, false, "<s>", "</s>") ==
        "[<s>]");
}

// ─── (a) for loop + member access + loop vars ────────────────────────────────
TEST_CASE("chat_template: for loop with loop.first/last/index0") {
  const std::string t =
      "{% for m in messages %}{{ loop.index0 }}:{{ m.role }}"
      "{% if not loop.last %},{% endif %}{% endfor %}";
  CHECK(apply_chat_template(t, SystemUser(), false) == "0:system,1:user");
}

// ─── (a) if / elif / else ────────────────────────────────────────────────────
TEST_CASE("chat_template: if / elif / else branches") {
  const std::string t =
      "{% for m in messages %}"
      "{% if m.role == 'system' %}S{% elif m.role == 'user' %}U"
      "{% else %}?{% endif %}"
      "{% endfor %}";
  std::vector<ChatMessage> msgs = {
      ChatMessage{"system", std::string("")}, ChatMessage{"user", std::string("")},
      ChatMessage{"assistant", std::string("")}};
  CHECK(apply_chat_template(t, msgs, false) == "SU?");
}

// ─── (a) membership `in` / `not in` ──────────────────────────────────────────
TEST_CASE("chat_template: membership tests") {
  CHECK(apply_chat_template("{% if 'im' in 'system' %}Y{% else %}N{% endif %}",
                            {}, false) == "N");
  CHECK(apply_chat_template("{% if 'sys' in 'system' %}Y{% else %}N{% endif %}",
                            {}, false) == "Y");
  CHECK(apply_chat_template(
            "{% if 'x' not in 'system' %}Y{% else %}N{% endif %}", {}, false) ==
        "Y");
}

// ─── (a) set + strip ─────────────────────────────────────────────────────────
TEST_CASE("chat_template: set binding and .strip()") {
  CHECK(apply_chat_template("{% set g = 'hi' %}{{ g }}!", {}, false) == "hi!");
  CHECK(apply_chat_template("{{ '  pad  '.strip() }}", {}, false) == "pad");
  CHECK(apply_chat_template("{{ '\\n\\nx\\n'.strip('\\n') }}", {}, false) == "x");
}

// ─── (a) whitespace control: `-` trim markers ────────────────────────────────
TEST_CASE("chat_template: whitespace trim control") {
  // No trim: the surrounding newlines/spaces are preserved.
  CHECK(apply_chat_template("A\n  {{ 'x' }}  \nB", {}, false) == "A\n  x  \nB");
  // Left+right trim collapses the surrounding whitespace.
  CHECK(apply_chat_template("A\n  {{- 'x' -}}  \nB", {}, false) == "AxB");
  // Block-tag trim_blocks/lstrip_blocks: the leading line-ws + trailing newline
  // around a block tag are removed even without explicit `-`.
  CHECK(apply_chat_template("A\n  {% set z = '1' %}\nB", {}, false) == "A\nB");
}

// ─── (e) an unsupported construct throws loudly ──────────────────────────────
TEST_CASE("chat_template: unsupported constructs throw ChatTemplateError") {
  // Filters are unsupported.
  CHECK_THROWS_AS(apply_chat_template("{{ x | upper }}", {}, false),
                  ChatTemplateError);
  // Macros are unsupported.
  CHECK_THROWS_AS(
      apply_chat_template("{% macro f() %}{% endmacro %}", {}, false),
      ChatTemplateError);
  // namespace()/is-tests (the Qwen tool/thinking path) are unsupported.
  CHECK_THROWS_AS(
      apply_chat_template("{% set n = namespace(x=1) %}", {}, false),
      ChatTemplateError);
  // Unterminated tag.
  CHECK_THROWS_AS(apply_chat_template("{{ 'x'", {}, false), ChatTemplateError);
  // Unbalanced endfor.
  CHECK_THROWS_AS(apply_chat_template("{% endfor %}", {}, false),
                  ChatTemplateError);
}

// ─── M3.3 Task 3: tools rendered into the prompt via the tool branch ─────────
namespace {
// A minja-subset tool template mirroring the Qwen3.6/Hermes tool-system-prompt
// shape (the full qwen35.jinja tool branch uses namespace/macros/is-tests/`[::-1]`
// slicing beyond the subset; this reproduces its tool-schema injection using the
// constructs the engine supports + the M3.3 `tojson` filter).
const char* kToolTemplate =
    "{%- if tools %}"
    "<|im_start|>system\n# Tools\n<tools>"
    "{%- for tool in tools %}\n{{ tool | tojson }}{%- endfor %}"
    "\n</tools><|im_end|>\n"
    "{%- endif %}"
    "{%- for message in messages %}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{%- endfor %}";

std::vector<ChatCompletionToolsParam> WeatherTool() {
  ChatCompletionToolsParam t;
  t.type = "function";
  t.function.name = "get_weather";
  t.function.description = "Get the weather for a city.";
  t.function.parameters = nlohmann::json::parse(
      R"({"type":"object","properties":{"city":{"type":"string"}}})");
  return {t};
}
}  // namespace

TEST_CASE("chat_template: tools render the function schemas into the prompt") {
  const std::vector<ChatMessage> msgs = {
      ChatMessage{"user", std::string("weather?")}};
  const std::string out = apply_chat_template(
      kToolTemplate, msgs, /*add_generation_prompt=*/false, /*bos=*/"",
      /*eos=*/"", WeatherTool());

  // The tool system prompt + the schema JSON (name/description/parameters) are
  // present, and the user turn follows.
  CHECK(out.find("# Tools") != std::string::npos);
  CHECK(out.find("<tools>") != std::string::npos);
  CHECK(out.find("\"name\":\"get_weather\"") != std::string::npos);
  CHECK(out.find("\"description\":\"Get the weather for a city.\"") !=
        std::string::npos);
  CHECK(out.find("\"parameters\"") != std::string::npos);
  CHECK(out.find("\"type\":\"function\"") != std::string::npos);
  CHECK(out.find("<|im_start|>user\nweather?<|im_end|>") != std::string::npos);
}

TEST_CASE("chat_template: no tools leaves the tool branch out (falsy tools)") {
  const std::vector<ChatMessage> msgs = {
      ChatMessage{"user", std::string("hi")}};
  const std::string out =
      apply_chat_template(kToolTemplate, msgs, /*add_generation_prompt=*/false);
  CHECK(out.find("# Tools") == std::string::npos);
  // The `{%- endfor %}` left-trim drops the loop body's trailing newline.
  CHECK(out == "<|im_start|>user\nhi<|im_end|>");
}

// `tojson` serializes a nested tool schema faithfully.
TEST_CASE("chat_template: tojson filter serializes the tool object") {
  const std::string out =
      apply_chat_template("{{ tools[0] | tojson }}", {}, false, "", "",
                          WeatherTool());
  const nlohmann::json j = nlohmann::json::parse(out);
  CHECK(j.at("type") == "function");
  CHECK(j.at("function").at("name") == "get_weather");
  CHECK(j.at("function").at("parameters").at("properties").at("city").at("type") ==
        "string");
}
