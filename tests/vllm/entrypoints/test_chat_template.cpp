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

#include "../gguf_builder.h"

#include <doctest/doctest.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
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

// ─── (e) a malformed template throws ChatTemplateError ───────────────────────
// The engine is now the full google/minja renderer, so constructs the old
// hand-written subset rejected (filters, macros, namespace(), is-tests, slicing)
// render fine (see the "minja engine renders" case below). Only a genuinely
// malformed template (a parse/eval error) still throws, and it is surfaced as a
// ChatTemplateError so the capi probe-and-fallback safety net keeps working.
TEST_CASE("chat_template: malformed templates throw ChatTemplateError") {
  // Unterminated tag.
  CHECK_THROWS_AS(apply_chat_template("{{ 'x'", {}, false), ChatTemplateError);
  // Unbalanced endfor.
  CHECK_THROWS_AS(apply_chat_template("{% endfor %}", {}, false),
                  ChatTemplateError);
  // Unterminated block body.
  CHECK_THROWS_AS(apply_chat_template("{% for m in messages %}", {}, false),
                  ChatTemplateError);
}

// ─── the vendored minja engine renders constructs the old subset rejected ────
// namespace()/macros/filters are exactly what the real Qwen3.5 template needs;
// the previous subset renderer threw on them. Assert they now render.
TEST_CASE("chat_template: minja engine renders namespace() and macros") {
  // namespace() + a mutating loop (the Qwen tool/thinking idiom).
  const char* ns =
      "{%- set found = namespace(value=false) %}"
      "{%- for m in messages %}{%- if m.role == 'user' %}"
      "{%- set found.value = true %}{%- endif %}{%- endfor %}"
      "{{- found.value }}";
  CHECK(apply_chat_template(ns, SystemUser(), false) == "True");

  // A macro definition + call.
  const char* macro =
      "{%- macro tag(role) -%}<|{{ role }}|>{%- endmacro -%}"
      "{{- tag('assistant') }}";
  CHECK(apply_chat_template(macro, {}, false) == "<|assistant|>");

  // A filter the subset never supported.
  CHECK(apply_chat_template("{{ 'hi' | upper }}", {}, false) == "HI");
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
  // present, and the user turn follows. minja's `tojson` uses Python/Jinja2
  // json.dumps separators (`": "`, `", "`) rather than the deleted subset's
  // compact nlohmann dump, so the key/value pairs carry a space after the colon.
  CHECK(out.find("# Tools") != std::string::npos);
  CHECK(out.find("<tools>") != std::string::npos);
  CHECK(out.find("\"name\": \"get_weather\"") != std::string::npos);
  CHECK(out.find("\"description\": \"Get the weather for a city.\"") !=
        std::string::npos);
  CHECK(out.find("\"parameters\"") != std::string::npos);
  CHECK(out.find("\"type\": \"function\"") != std::string::npos);
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

// ─── The REAL Qwen3.5 chat template (namespace()/macros/is-tests/<tool_call>) ─
// Extracted from the Qwen3.5-2B GGUF (tokenizer.chat_template) into
// tests/fixtures/qwen35_chat_template.jinja. The old hand-written subset threw
// on its namespace()/macro/is-test constructs; the vendored minja engine renders
// it. This is the real-world driver for the whole vendoring effort.
namespace {
std::string ReadFixture(const std::string& name) {
  std::ifstream f(std::string(VLLM_TEST_FIXTURES_DIR) + "/" + name,
                  std::ios::binary);
  if (!f) throw std::runtime_error("cannot open fixture: " + name);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
}  // namespace

TEST_CASE("chat_template: real Qwen3.5 template renders a plain conversation") {
  const std::string tmpl = ReadFixture("qwen35_chat_template.jinja");
  std::string out;
  REQUIRE_NOTHROW(out = apply_chat_template(tmpl, SystemUser(),
                                            /*add_generation_prompt=*/true,
                                            /*bos=*/"", /*eos=*/"<|im_end|>"));
  // The chatml turns are present and the assistant generation header is added.
  CHECK(out.find("<|im_start|>system\nYou are a helpful assistant.<|im_end|>") !=
        std::string::npos);
  CHECK(out.find("<|im_start|>user\nHello, who are you?<|im_end|>") !=
        std::string::npos);
  CHECK(out.find("<|im_start|>assistant") != std::string::npos);
  // With no tools, the tools system prompt must NOT appear.
  CHECK(out.find("<tools>") == std::string::npos);
}

TEST_CASE("chat_template: real Qwen3.5 template renders the tools branch") {
  const std::string tmpl = ReadFixture("qwen35_chat_template.jinja");
  const std::vector<ChatMessage> msgs = {
      ChatMessage{"user", std::string("what is the weather?")}};
  std::string out;
  REQUIRE_NOTHROW(out = apply_chat_template(
                      tmpl, msgs, /*add_generation_prompt=*/true, /*bos=*/"",
                      /*eos=*/"<|im_end|>", WeatherTool()));
  // The tool system prompt, the <tool_call> instruction surface, and the tool
  // name are all injected by the template's tool branch.
  CHECK(out.find("# Tools") != std::string::npos);
  CHECK(out.find("<tools>") != std::string::npos);
  CHECK(out.find("<tool_call>") != std::string::npos);
  CHECK(out.find("get_weather") != std::string::npos);
  CHECK(out.find("<|im_start|>user\nwhat is the weather?<|im_end|>") !=
        std::string::npos);
}

// ─── LoadChatTemplateFromGguf (ABI v3) ───────────────────────────────────────
// GGUF models carry their chat template as `tokenizer.chat_template` metadata
// (no tokenizer_config.json exists). Build minimal synthetic GGUFs and assert
// the loader extracts the template / errors loudly.
TEST_CASE("chat_template: LoadChatTemplateFromGguf extracts tokenizer.chat_template") {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "qwen35"));
  b.AddKv(gguf_test::StrKv("tokenizer.chat_template",
                           "{{ messages[0].content }}"));
  gguf_test::TempFile f(b.Build());
  CHECK(vllm::entrypoints::LoadChatTemplateFromGguf(f.path()) ==
        "{{ messages[0].content }}");
}

TEST_CASE("chat_template: LoadChatTemplateFromGguf errors without the key") {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "qwen35"));
  gguf_test::TempFile f(b.Build());
  CHECK_THROWS_AS(vllm::entrypoints::LoadChatTemplateFromGguf(f.path()),
                  vllm::entrypoints::ChatTemplateError);
}

TEST_CASE("chat_template: LoadChatTemplateFromGguf errors on an unreadable file") {
  CHECK_THROWS_AS(
      vllm::entrypoints::LoadChatTemplateFromGguf("/nonexistent/x.gguf"),
      vllm::entrypoints::ChatTemplateError);
}
