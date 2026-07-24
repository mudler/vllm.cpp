// Ported from: vllm/entrypoints/chat_utils.py @ e24d1b24
//   (the apply_hf_chat_template / resolve+render path — vLLM delegates chat
//   templating to transformers' `apply_chat_template`, i.e. a Jinja2 render of
//   the tokenizer_config.json `chat_template` string; see also
//   vllm/renderers/hf.py::safe_apply_chat_template @ e24d1b24).
//
// DEVIATION (recorded, like the vt:: runtime): transformers renders the chat
// template with the full CPython Jinja2 engine. We have no Python at runtime, so
// this file is an ORIGINAL component — a small MINJA-SUBSET Jinja renderer
// (there is no upstream C++ mirror to port). It supports ONLY the constructs the
// gate-model (Qwen3.6 / Qwen-family, and best-effort Llama-family) chat
// templates use for the T0 no-tools / no-thinking path:
//   - `{{ expr }}` interpolation; variables `messages`, `add_generation_prompt`,
//     `bos_token`, `eos_token`, the `for` loop var and `loop`, plus `{% set %}`
//     bindings; member access `x.y`, subscript `x[i]` / `x['k']`.
//   - `{% for v in expr %}` … `{% endfor %}` (with `loop.first/last/index0/
//     index/length/revindex/revindex0`).
//   - `{% if %}` / `{% elif %}` / `{% else %}` / `{% endif %}` with `==`, `!=`,
//     `in`, `not in`, `and`, `or`, `not`, truthiness.
//   - `{% set x = expr %}` bindings.
//   - `+` / `~` string concatenation (and `+` integer add), `.strip()` /
//     `.lstrip()` / `.rstrip()` (optional char arg).
//   - Whitespace control: the `-` trim markers (`{%- -%}`, `{{- -}}`) PLUS
//     transformers' `trim_blocks=True, lstrip_blocks=True` block-whitespace
//     policy — reproduced exactly (chat_template_utils.py:474 @ transformers).
// ANY other construct (filters `|`, macros, `include`, tuple-unpack `for`,
// slicing `[::-1]`, `namespace()`, `is`-tests, unknown methods/functions) is a
// LOUD ERROR (throws vllm::entrypoints::ChatTemplateError with the offending
// construct + source position) — never a silently-wrong prompt.
//
// The full upstream Qwen3.6 template additionally drives tool-calling and
// <think> reasoning via those unsupported constructs; those branches are DEFER-
// RED to M3.3 (tool calling) / M3.4. For a no-tools/no-thinking conversation the
// upstream template's output is byte-identical to the subset template rendered
// here (verified against a transformers `apply_chat_template` dump — see
// tests/vllm/entrypoints/test_chat_template.cpp).
#ifndef VLLM_ENTRYPOINTS_CHAT_TEMPLATE_H_
#define VLLM_ENTRYPOINTS_CHAT_TEMPLATE_H_

#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/serving_chat.h"

namespace vllm::entrypoints {

// Thrown on any template-render failure: an unsupported Jinja construct, a
// syntax error, or a type error during evaluation. `what()` carries the reason
// and the source position (line:col).
class ChatTemplateError : public std::runtime_error {
 public:
  explicit ChatTemplateError(const std::string& msg)
      : std::runtime_error(msg) {}
};

// Render a minja-subset Jinja chat template to the prompt string.
//   template_str          the tokenizer_config.json `chat_template` Jinja source
//   messages              the {role, content} conversation
//   add_generation_prompt appends the assistant generation header when the
//                         template gates on it (Qwen: `<|im_start|>assistant\n`)
//   bos_token / eos_token exposed to the template as `bos_token` / `eos_token`
//   tools                 the available function schemas, exposed to the template
//                         as `tools` (a list of the OpenAI tool JSON objects) for
//                         the `{% if tools %}...{{ tool | tojson }}...{% endif %}`
//                         branch. Empty => the `tools` variable is an empty list
//                         (falsy). Requires the `tojson` filter (M3.3 minja ext).
// Throws ChatTemplateError on any unsupported construct or evaluation error.
std::string apply_chat_template(
    const std::string& template_str,
    const std::vector<openai::ChatMessage>& messages, bool add_generation_prompt,
    const std::string& bos_token = "", const std::string& eos_token = "",
    const std::vector<openai::ChatCompletionToolsParam>& tools = {});

// Adapt a chat template to Task 2's ChatPromptFn seam (serving_chat.h). The
// returned callable renders `template_str` for the messages + generation flag it
// is handed, so an OpenAIServingChat constructed with it applies the real chat
// template instead of DefaultChatPromptFallback.
openai::ChatPromptFn MakeChatTemplatePromptFn(std::string template_str,
                                              std::string bos_token = "",
                                              std::string eos_token = "");

// Load the `chat_template` string out of a tokenizer_config.json file. Handles
// both the plain-string form and the list-of-{name,template} form (picks the
// entry named "default", else the first). Throws ChatTemplateError if the file
// cannot be read or carries no chat_template.
std::string LoadChatTemplateFromConfig(const std::string& tokenizer_config_path);

// Load the chat template out of a .gguf file's `tokenizer.chat_template`
// metadata (the llama.cpp-ecosystem convention for shipping the template
// inside the model file, since a GGUF has no tokenizer_config.json). Throws
// ChatTemplateError if the file cannot be read or carries no such key.
std::string LoadChatTemplateFromGguf(const std::string& gguf_path);

}  // namespace vllm::entrypoints

#endif  // VLLM_ENTRYPOINTS_CHAT_TEMPLATE_H_
