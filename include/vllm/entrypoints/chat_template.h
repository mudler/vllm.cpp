// Ported from: vllm/entrypoints/chat_utils.py @ e24d1b24
//   (the apply_hf_chat_template / resolve+render path — vLLM delegates chat
//   templating to transformers' `apply_chat_template`, i.e. a Jinja2 render of
//   the tokenizer_config.json `chat_template` string; see also
//   vllm/renderers/hf.py::safe_apply_chat_template @ e24d1b24).
//
// DEVIATION (recorded, like the vt:: runtime): transformers renders the chat
// template with the full CPython Jinja2 engine. We have no Python at runtime, so
// the render engine is the vendored google/minja header-only Jinja renderer
// (third_party/minja/, the same engine llama.cpp historically vendored). This
// file is the ADAPTER over it: it parses the tokenizer's raw Jinja
// `chat_template` and renders it with transformers' whitespace policy
// (trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False, per
// chat_template_utils.py @ transformers), exposing `messages`,
// `add_generation_prompt`, `bos_token`, `eos_token`, `tools` (and strftime_now).
// minja supports the full construct surface real templates use (filters,
// macros, `namespace()`, is-tests, slicing, tuple-unpack loops), so the whole
// upstream Qwen/Llama templates (tool-calling and <think> branches included)
// render, not just a hand-picked subset.
//
// It renders the template LITERALLY (like transformers), NOT via minja's
// high-level `chat_template` wrapper: that wrapper adds a heuristic capability
// probe + a "polyfill" pass (rewriting the message list) that is not
// transformers behavior and diverges on byte-exact output. One local minja
// modification restores exact transformers parity for lstrip_blocks before
// expression tags (see third_party/minja/minja.hpp and third_party/README.md).
//
// A render/parse failure (syntax error, evaluation error) throws
// vllm::entrypoints::ChatTemplateError, never a silently-wrong prompt; the capi
// probe-and-fallback layer (src/capi/chat_prompt.*) relies on that. Byte-exact
// outputs are verified against transformers/jinja2 in
// tests/vllm/entrypoints/test_chat_template.cpp.
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

// Render a Jinja chat template (via the vendored minja engine) to the prompt.
//   template_str          the tokenizer_config.json `chat_template` Jinja source
//   messages              the {role, content} conversation
//   add_generation_prompt appends the assistant generation header when the
//                         template gates on it (Qwen: `<|im_start|>assistant\n`)
//   bos_token / eos_token exposed to the template as `bos_token` / `eos_token`
//   tools                 the available function schemas, exposed to the template
//                         as `tools` (a list of the OpenAI tool JSON objects) for
//                         the `{% if tools %}...{{ tool | tojson }}...{% endif %}`
//                         branch. Empty => the `tools` variable is an empty list
//                         (falsy). The `tojson` filter is a minja builtin.
// Throws ChatTemplateError on any parse or evaluation error.
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
