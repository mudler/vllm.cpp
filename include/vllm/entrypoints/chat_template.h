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

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

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
//
// chat_template_kwargs: the extra Jinja variables the caller supplied, from the
// request's `chat_template_kwargs` merged over the server default (upstream
// `--default-chat-template-kwargs`, itself defaulting to none). Each key becomes
// a template variable; **a key that is absent stays Jinja-UNDEFINED**, which is
// what upstream does (vllm/renderers/hf.py:633-661 passes only the resolved
// keys) and what a template asking `{% if enable_thinking is undefined %}`
// needs. Defining every known name unconditionally would make `is undefined`
// permanently false and silently invert the model's own default (#1681).
//
// A key that names something the renderer ITSELF supplies is REFUSED with
// `vllm::v1::InputValidationError`, which api_server maps to HTTP 400 and the C
// ABI to VLLM_ERR_INVALID_ARGUMENT, mirroring the pinned vLLM (`555967922`):
//   `chat_template`, `tokenize` -- apply_chat_template's own parameters;
//     resolve_chat_template_kwargs RAISES on them (hf.py:639-648), and its only
//     call site takes the default raise_on_unexpected=True (hf.py:731-735).
//   `messages`, `tools` -- resolve_chat_template_kwargs KEEPS these, and
//     transformers then dies on the duplicate keyword before rendering
//     (`TypeError: got multiple values for keyword argument 'messages'`).
//     Binding them let a request REPLACE the conversation the caller passed,
//     which no upstream path can do.
// `add_generation_prompt` and `continue_final_message` are accepted and
// SKIPPED, because upstream's build_chat_params has already overwritten each
// with the request field of the same name before the filter runs
// (chat_completion/protocol.py:530-544, merge_kwargs params.py:28-40) -- and
// the first of those fields is this function's parameter of the same name.
// A key that names one of the ENGINE's built-ins is also SKIPPED. minja resolves
// a global, a filter and an is-test through one Context chain, so binding such a
// key shadows the built-in the template needs -- `{"namespace": 1}` broke line 1
// of the shipped Qwen3.8 template as a 500. CPython Jinja2 keeps all three kinds
// out of the variable namespace, so find_undeclared_variables never reports one
// and upstream's accept_vars always drops it. `raise_exception` is the ONE
// exception and it is upstream's: transformers adds it to the environment after
// that parse, so jinja2 does report it and the request value wins there too.
// Every other key binds, `bos_token`/`eos_token` included: a template that
// names either has it in find_undeclared_variables, so upstream keeps the
// request value and lets it win over the tokenizer's special_tokens_map; a
// template that names neither cannot observe it on either side.
//
// A render failure throws ChatTemplateError, which api_server maps to HTTP 400
// as well: upstream wraps every apply_chat_template exception into a ValueError
// (hf.py:785-789) and create_error_response answers BadRequestError for that and
// for jinja2.TemplateError alike (error_response.py:48-52,61-65).
std::string apply_chat_template(
    const std::string& template_str,
    const std::vector<openai::ChatMessage>& messages, bool add_generation_prompt,
    const std::string& bos_token = "", const std::string& eos_token = "",
    const std::vector<openai::ChatCompletionToolsParam>& tools = {},
    const nlohmann::ordered_json& chat_template_kwargs =
        nlohmann::ordered_json::object());

// Adapt a chat template to Task 2's ChatPromptFn seam (serving_chat.h).
// `default_chat_template_kwargs` is the SERVER-level default (our
// `--enable-thinking` / `--no-enable-thinking` write `enable_thinking` into it;
// neither flag leaves it empty). The per-request kwargs the seam hands the
// returned function are merged OVER it, mirroring merge_kwargs
// (vllm/renderers/params.py:28-40 @ 555967922) as reached through
// ChatParams.with_defaults (params.py:93-122) from
// vllm/entrypoints/openai/chat_completion/serving.py:208. A request value of
// null or "auto" is `unset_values` and does NOT override: the server default
// stands.
openai::ChatPromptFn MakeChatTemplatePromptFn(
    std::string template_str, std::string bos_token = "",
    std::string eos_token = "",
    nlohmann::ordered_json default_chat_template_kwargs =
        nlohmann::ordered_json::object());

// The RULE behind `--enable-thinking` / `--no-enable-thinking`, lifted out of
// server_main so it can be driven from a gate: neither flag (nullopt) yields an
// EMPTY object, which is upstream's `--default-chat-template-kwargs` default
// (`None`, `openai/cli_args.py:93`) and leaves `enable_thinking` Jinja-undefined. Either
// flag yields `{"enable_thinking": <bool>}`. The difference between "unset" and
// "explicitly false" is the whole point and is invisible to a bool (#1681).
nlohmann::ordered_json DefaultChatTemplateKwargs(
    std::optional<bool> enable_thinking);

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
