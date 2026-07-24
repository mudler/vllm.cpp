// INTERNAL (not installed / not part of the public ABI). Chat-prompt
// resolution for the ABI v3 chat entry points: pick the model's real chat
// template when the minja-subset renderer can serve it, and otherwise degrade
// — with a stderr witness, never silently — to a Hermes-aware fallback prompt
// that still primes the model for the engine's structural-tag tool flow.
//
// Rationale: MakeChatTemplatePromptFn renders LAZILY, so an exotic template
// (namespace()/macro — e.g. the full Qwen3.5 template) would otherwise fail
// EVERY chat request with a ChatTemplateError. The bundled server already
// falls back to a plain prompt when no template loads; this extends the same
// policy to "loads but cannot render", probed ONCE at resolution.
#ifndef VLLM_CAPI_CHAT_PROMPT_H_
#define VLLM_CAPI_CHAT_PROMPT_H_

#include <string>

#include "vllm/entrypoints/openai/serving_chat.h"

namespace vllm::capi {

// Role-join fallback that ALSO renders the Hermes/Qwen tools convention when
// tools are present: a "# Tools" system block with the function schemas inside
// <tools></tools> and the <tool_call> emission instruction. This matches the
// tag surface ToolChoiceStructuralTagSpec constrains on, so lazy (auto)
// tool engagement works even without the model's own template.
std::string HermesToolsFallbackPrompt(
    const std::vector<vllm::entrypoints::openai::ChatMessage>& messages,
    bool add_generation_prompt,
    const std::vector<vllm::entrypoints::openai::ChatCompletionToolsParam>&
        tools);

// Resolve the ChatPromptFn for a template string: probe-render the template
// (with and without tools) and return
//   - the real template fn when both probes render,
//   - a hybrid (template for tool-less requests, hermes fallback for tool
//     requests) when only the tools branch fails,
//   - the hermes fallback when the template cannot render at all.
// Fallbacks print one stderr witness naming the template failure. `origin` is
// the template's source (path) for that witness.
vllm::entrypoints::openai::ChatPromptFn ResolveTemplatePromptFn(
    const std::string& template_str, const std::string& bos,
    const std::string& eos, const std::string& origin);

}  // namespace vllm::capi

#endif  // VLLM_CAPI_CHAT_PROMPT_H_
