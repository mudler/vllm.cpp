// Ported from: vllm/entrypoints/chat_utils.py @ e24d1b24 (see chat_template.h
// for the deviation note). vLLM delegates chat templating to transformers'
// full CPython Jinja2 (`apply_chat_template`), which we cannot depend on at
// runtime. This file is the ADAPTER over the vendored google/minja Jinja
// engine (third_party/minja/, the same engine llama.cpp historically vendored):
// minja renders the tokenizer's Jinja `chat_template` with transformers'
// whitespace policy (trim_blocks=True, lstrip_blocks=True, keep_trailing_newline
// =False) and supports the full construct surface real templates use
// (namespace(), macros, filters, is-tests, slicing, ...), not the reduced
// subset the previous hand-written renderer accepted.
#include "vllm/entrypoints/chat_template.h"

#include "vllm/model_executor/model_loader/gguf_reader.h"

#include <chrono>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <minja/minja.hpp>

namespace vllm::entrypoints {
namespace {

// Build the `messages` variable minja exposes to the template, mirroring what
// transformers passes to `apply_chat_template`: a list of {role, content} maps,
// carrying `tool_calls` (OpenAI shape) when a ChatMessage has them. ordered_json
// keeps key order stable so `{{ message | tojson }}`-style branches are
// deterministic.
nlohmann::ordered_json BuildMessages(
    const std::vector<openai::ChatMessage>& messages) {
  nlohmann::ordered_json arr = nlohmann::ordered_json::array();
  for (const openai::ChatMessage& m : messages) {
    nlohmann::ordered_json o = nlohmann::ordered_json::object();
    o["role"] = m.role;
    if (m.content.has_value()) {
      o["content"] = *m.content;
    } else {
      o["content"] = nullptr;
    }
    // Multi-turn tool identity + prior reasoning: exposed only when present so
    // `{% if message.tool_call_id %}`-style template gates stay falsy on plain
    // turns (an always-present null would still be falsy in Jinja, but absent
    // matches transformers' dict shape exactly).
    if (m.tool_call_id.has_value()) o["tool_call_id"] = *m.tool_call_id;
    if (m.name.has_value()) o["name"] = *m.name;
    if (m.reasoning.has_value()) o["reasoning"] = *m.reasoning;
    if (m.tool_calls.has_value()) {
      nlohmann::ordered_json calls = nlohmann::ordered_json::array();
      for (const openai::ToolCall& tc : *m.tool_calls) {
        nlohmann::ordered_json fn = nlohmann::ordered_json::object();
        fn["name"] = tc.function.name;
        // OpenAI carries arguments as a JSON-encoded STRING. minja's
        // chat-template wrapper polyfills string->object when the template
        // requires object arguments (requires_object_arguments caps).
        fn["arguments"] = tc.function.arguments;
        nlohmann::ordered_json call = nlohmann::ordered_json::object();
        call["id"] = tc.id;
        call["type"] = tc.type;
        call["function"] = std::move(fn);
        calls.push_back(std::move(call));
      }
      o["tool_calls"] = std::move(calls);
    }
    arr.push_back(std::move(o));
  }
  return arr;
}

// Rebuild the OpenAI tool JSON objects the request carried, exposed to the
// template as `tools` exactly as transformers' apply_chat_template(tools=...)
// sees them:
//   {"type": <t>, "function": {"name": .., "description"?: .., "parameters"?: ..}}
nlohmann::ordered_json BuildTools(
    const std::vector<openai::ChatCompletionToolsParam>& tools) {
  nlohmann::ordered_json arr = nlohmann::ordered_json::array();
  for (const openai::ChatCompletionToolsParam& t : tools) {
    nlohmann::ordered_json fn = nlohmann::ordered_json::object();
    fn["name"] = t.function.name;
    if (t.function.description.has_value()) {
      fn["description"] = *t.function.description;
    }
    if (t.function.parameters.has_value()) {
      // parameters is a plain nlohmann::json (unordered). Round-trip through a
      // string to land it in ordered_json without an implicit cross-container
      // conversion.
      fn["parameters"] =
          nlohmann::ordered_json::parse(t.function.parameters->dump());
    }
    nlohmann::ordered_json tool = nlohmann::ordered_json::object();
    tool["type"] = t.type;
    tool["function"] = std::move(fn);
    arr.push_back(std::move(tool));
  }
  return arr;
}

}  // namespace

// ─── Public API ──────────────────────────────────────────────────────────────
std::string apply_chat_template(
    const std::string& template_str,
    const std::vector<openai::ChatMessage>& messages, bool add_generation_prompt,
    const std::string& bos_token, const std::string& eos_token,
    const std::vector<openai::ChatCompletionToolsParam>& tools) {
  try {
    // Render the template LITERALLY, exactly as transformers'
    // `apply_chat_template` does: parse the raw Jinja source and render it with
    // transformers' whitespace policy (trim_blocks / lstrip_blocks, no trailing
    // newline). We deliberately use minja's low-level engine rather than its
    // high-level `chat_template` wrapper: that wrapper runs a heuristic
    // capability probe (6+ speculative renders per construction, plus stderr
    // diagnostics) and a "polyfill" pass that rewrites the message list (merging
    // the system role into a user turn, injecting a synthetic tools system
    // prompt, ...). None of that is transformers behavior, and the probe
    // misfires on templates that do not echo message content verbatim. The
    // low-level path is the faithful, quiet, per-request-cheap equivalent.
    //
    // Parser::parse throws on a syntax error; render() throws on an evaluation
    // error. Both surface below as ChatTemplateError.
    std::shared_ptr<minja::TemplateNode> root = minja::Parser::parse(
        template_str, minja::Options{/*trim_blocks=*/true,
                                     /*lstrip_blocks=*/true,
                                     /*keep_trailing_newline=*/false});

    nlohmann::ordered_json top = nlohmann::ordered_json::object();
    top["messages"] = BuildMessages(messages);
    top["add_generation_prompt"] = add_generation_prompt;
    // Context::make's default parent is minja::Context::builtins(), which
    // provides the standard Jinja filters/functions/tests (tojson, upper, map,
    // selectattr, is-tests, ...) the real templates rely on.
    std::shared_ptr<minja::Context> context =
        minja::Context::make(minja::Value(top));
    context->set("bos_token", minja::Value(bos_token));
    context->set("eos_token", minja::Value(eos_token));
    // An empty tools array is falsy in Jinja (`{% if tools %}` skips), matching
    // transformers passing tools=None.
    context->set("tools", minja::Value(BuildTools(tools)));
    // Some templates (e.g. Llama 3.x) call strftime_now(fmt) for the date line.
    const auto now = std::chrono::system_clock::now();
    context->set(
        "strftime_now",
        minja::Value::callable([now](const std::shared_ptr<minja::Context>&,
                                     minja::ArgumentsValue& args) {
          args.expectArgs("strftime_now", {1, 1}, {0, 0});
          const auto format = args.args[0].get<std::string>();
          const std::time_t t = std::chrono::system_clock::to_time_t(now);
          std::tm local_time{};
#if defined(_WIN32)
          localtime_s(&local_time, &t);
#else
          localtime_r(&t, &local_time);
#endif
          std::ostringstream ss;
          ss << std::put_time(&local_time, format.c_str());
          return minja::Value(ss.str());
        }));

    return root->render(context);
  } catch (const ChatTemplateError&) {
    throw;
  } catch (const std::exception& e) {
    throw ChatTemplateError(std::string("chat template render failed: ") +
                            e.what());
  }
}

openai::ChatPromptFn MakeChatTemplatePromptFn(std::string template_str,
                                              std::string bos_token,
                                              std::string eos_token) {
  return [tmpl = std::move(template_str), bos = std::move(bos_token),
          eos = std::move(eos_token)](
             const std::vector<openai::ChatMessage>& messages,
             bool add_generation_prompt,
             const std::vector<openai::ChatCompletionToolsParam>& tools) {
    return apply_chat_template(tmpl, messages, add_generation_prompt, bos, eos,
                               tools);
  };
}

std::string LoadChatTemplateFromConfig(
    const std::string& tokenizer_config_path) {
  std::ifstream f(tokenizer_config_path, std::ios::binary);
  if (!f) {
    throw ChatTemplateError("cannot open tokenizer_config.json: " +
                            tokenizer_config_path);
  }
  nlohmann::json doc;
  try {
    f >> doc;
  } catch (const std::exception& e) {
    throw ChatTemplateError(std::string("failed to parse tokenizer_config.json: ") +
                            e.what());
  }
  auto it = doc.find("chat_template");
  if (it == doc.end() || it->is_null()) {
    throw ChatTemplateError("tokenizer_config.json has no 'chat_template': " +
                            tokenizer_config_path);
  }
  if (it->is_string()) return it->get<std::string>();
  // List-of-{name,template} form: pick "default", else the first.
  if (it->is_array()) {
    const nlohmann::json* chosen = nullptr;
    for (const auto& entry : *it) {
      if (entry.is_object() && entry.value("name", std::string()) == "default") {
        chosen = &entry;
        break;
      }
    }
    if (!chosen && !it->empty()) chosen = &it->front();
    if (chosen && chosen->contains("template") &&
        (*chosen)["template"].is_string()) {
      return (*chosen)["template"].get<std::string>();
    }
  }
  throw ChatTemplateError("unrecognized 'chat_template' shape in " +
                          tokenizer_config_path);
}

std::string LoadChatTemplateFromGguf(const std::string& gguf_path) {
  try {
    const vllm::GgufFile gguf = vllm::GgufFile::Open(gguf_path);
    const vllm::GgufValue* kv = gguf.FindKv("tokenizer.chat_template");
    if (kv == nullptr) {
      throw ChatTemplateError("gguf has no 'tokenizer.chat_template': " +
                              gguf_path);
    }
    const std::string* tmpl = std::get_if<std::string>(&kv->v);
    if (tmpl == nullptr || tmpl->empty()) {
      throw ChatTemplateError(
          "gguf 'tokenizer.chat_template' is not a non-empty string: " +
          gguf_path);
    }
    return *tmpl;
  } catch (const ChatTemplateError&) {
    throw;
  } catch (const std::exception& e) {
    // GgufFile::Open throws std::runtime_error on any malformation.
    throw ChatTemplateError(std::string("cannot read gguf chat template: ") +
                            e.what());
  }
}

}  // namespace vllm::entrypoints
