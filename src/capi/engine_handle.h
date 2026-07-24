// INTERNAL (not installed / not part of the public ABI). Exposes the C++ seam
// the C ABI impl uses to build a vllm_engine handle from an already-constructed
// LoadedEngine, so tests can inject synthetic in-memory weights (no disk load)
// and still exercise the real vllm_complete path. Public consumers use
// include/vllm.h only.
#ifndef VLLM_CAPI_ENGINE_HANDLE_H_
#define VLLM_CAPI_ENGINE_HANDLE_H_

#include <memory>
#include <string>

#include "vllm.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/entrypoints/openai/serving_chat.h"  // ChatPromptFn

namespace vllm::capi {

// Wrap an owned LoadedEngine into an opaque vllm_engine handle (transfers
// ownership). The returned handle is freed with vllm_engine_free, exactly like
// one from vllm_engine_load. Never throws; returns nullptr on allocation
// failure. This is the test hook: build a LoadedEngine from synthetic pieces,
// wrap it here, then drive vllm_complete().
vllm_engine* MakeEngineHandle(
    std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded) noexcept;

// Test-hook overload for the ABI v3 chat entry points: the handle's chat
// serving is built with `prompt_fn` instead of the disk-resolved chat template,
// so a synthetic-vocab engine can keep its prompt in-vocab (mirrors the
// api-server test harness's InVocabChatPrompt seam).
vllm_engine* MakeEngineHandle(
    std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded,
    vllm::entrypoints::openai::ChatPromptFn prompt_fn) noexcept;

// Test hook for the ABI v4 tool-parser selection: set the handle's tool-parser
// name as vllm_engine_load would from vllm_model_params.tool_parser. Must be
// called BEFORE the first chat call (the chat serving is built lazily and caches
// the parser). NULL handle is a no-op. Empty name restores AUTO detection.
void SetEngineToolParser(vllm_engine* handle, const std::string& name) noexcept;

// Test hook: select the ABI v5 reasoning parser on a handle (mirrors
// SetEngineToolParser).
void SetEngineReasoningParser(vllm_engine* engine, std::string name) noexcept;

}  // namespace vllm::capi

#endif  // VLLM_CAPI_ENGINE_HANDLE_H_
