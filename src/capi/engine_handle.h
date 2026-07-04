// INTERNAL (not installed / not part of the public ABI). Exposes the C++ seam
// the C ABI impl uses to build a vllm_engine handle from an already-constructed
// LoadedEngine, so tests can inject synthetic in-memory weights (no disk load)
// and still exercise the real vllm_complete path. Public consumers use
// include/vllm.h only.
#ifndef VLLM_CAPI_ENGINE_HANDLE_H_
#define VLLM_CAPI_ENGINE_HANDLE_H_

#include <memory>

#include "vllm.h"
#include "vllm/entrypoints/model_loader.h"

namespace vllm::capi {

// Wrap an owned LoadedEngine into an opaque vllm_engine handle (transfers
// ownership). The returned handle is freed with vllm_engine_free, exactly like
// one from vllm_engine_load. Never throws; returns nullptr on allocation
// failure. This is the test hook: build a LoadedEngine from synthetic pieces,
// wrap it here, then drive vllm_complete().
vllm_engine* MakeEngineHandle(
    std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded) noexcept;

}  // namespace vllm::capi

#endif  // VLLM_CAPI_ENGINE_HANDLE_H_
