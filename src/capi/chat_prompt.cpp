// See chat_prompt.h. ORIGINAL packaging-layer component (no upstream mirror):
// upstream vLLM delegates templating to transformers' full Jinja2 and never
// needs a degradation path.
#include "capi/chat_prompt.h"

#include <iostream>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/chat_template.h"

namespace vllm::capi {

namespace oai = vllm::entrypoints::openai;

std::string HermesToolsFallbackPrompt(
    const std::vector<oai::ChatMessage>& messages, bool add_generation_prompt,
    const std::vector<oai::ChatCompletionToolsParam>& tools) {
  std::string prompt;
  if (!tools.empty()) {
    // The Hermes/Qwen tools preamble: the SAME <tools>/<tool_call> surface the
    // structural-tag constraint triggers on (get_hermes_structural_tag).
    prompt +=
        "# Tools\n\nYou may call one or more functions to assist with the "
        "user query.\n\nYou are provided with function signatures within "
        "<tools></tools> XML tags:\n<tools>\n";
    for (const oai::ChatCompletionToolsParam& t : tools) {
      nlohmann::json fn{{"name", t.function.name}};
      if (t.function.description.has_value()) {
        fn["description"] = *t.function.description;
      }
      if (t.function.parameters.has_value()) {
        fn["parameters"] = *t.function.parameters;
      }
      prompt += nlohmann::json{{"type", t.type}, {"function", std::move(fn)}}
                    .dump() +
                "\n";
    }
    prompt +=
        "</tools>\n\nFor each function call, return a json object with "
        "function name and arguments within <tool_call></tool_call> XML "
        "tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
        "<args-json-object>}\n</tool_call>\n\n";
  }
  prompt += oai::DefaultChatPromptFallback(messages, add_generation_prompt);
  return prompt;
}

oai::ChatPromptFn ResolveTemplatePromptFn(const std::string& template_str,
                                          const std::string& bos,
                                          const std::string& eos,
                                          const std::string& origin) {
  oai::ChatPromptFn template_fn =
      vllm::entrypoints::MakeChatTemplatePromptFn(template_str, bos, eos);

  // Probe once: minja throws LOUDLY on a genuinely broken template (syntax or
  // evaluation error) at render time, and a chat surface that throws on every
  // request is useless. A short two-turn conversation exercises the message
  // loop; the dummy tool exercises the `{% if tools %}` branch.
  const std::vector<oai::ChatMessage> probe_messages = {
      oai::ChatMessage{"system", std::string("probe")},
      oai::ChatMessage{"user", std::string("probe")}};
  oai::ChatCompletionToolsParam probe_tool;
  probe_tool.function.name = "probe";

  std::string plain_error;
  try {
    (void)template_fn(probe_messages, /*add_generation_prompt=*/true, {});
  } catch (const std::exception& e) {
    plain_error = e.what();
  }
  if (!plain_error.empty()) {
    std::cerr << "vllm-capi: chat template from " << origin
              << " is not renderable (" << plain_error
              << "); falling back to the hermes-aware plain prompt\n";
    return HermesToolsFallbackPrompt;
  }

  std::string tools_error;
  try {
    (void)template_fn(probe_messages, /*add_generation_prompt=*/true,
                      {probe_tool});
  } catch (const std::exception& e) {
    tools_error = e.what();
  }
  if (tools_error.empty()) return template_fn;

  std::cerr << "vllm-capi: chat template from " << origin
            << " cannot render the tools branch (" << tools_error
            << "); tool requests fall back to the hermes-aware plain prompt\n";
  return [template_fn = std::move(template_fn)](
             const std::vector<oai::ChatMessage>& messages,
             bool add_generation_prompt,
             const std::vector<oai::ChatCompletionToolsParam>& tools) {
    if (tools.empty()) {
      return template_fn(messages, add_generation_prompt, tools);
    }
    return HermesToolsFallbackPrompt(messages, add_generation_prompt, tools);
  };
}

}  // namespace vllm::capi
