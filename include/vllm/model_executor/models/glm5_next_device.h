// GLM-5.3-Flash W9c-0 — the availability probe for this model's device-only op
// family. Issue #2415, `.agents/specs/glm5-next-flash.md` section W9c-0.
//
// The k-pool DSA indexer's two ops (`vt::OpId::kGlm5NextKpoolCompress`,
// `kGlm5NextKpoolSelect`) are registered on `kCUDA` only, by
// `src/vt/cuda/cuda_glm5_next.cu`. On a CPU-only build nothing is registered for
// them, so `vt::GetOp` throws and `vt::Glm5NextKpoolCompress` surfaces a clean
// device-only error rather than linking a stub that returns a plausible wrong
// selection. That is the arrangement `deepseek_v4_device.h` sets out for the
// four V4 families, and this header is its one-family mirror.
//
// The probe exists so a forward can decide BEFORE it builds operands rather
// than after it throws — the shape `deepseek_v4_device.cpp:30-35` uses. **No
// production path consults it yet**: W9c-3 owns the compose that constructs a
// CUDA queue for this model and deletes the refusal at
// `glm5_next_forward.cpp:231-238`, and the row's spec records the debt as O36.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_DEVICE_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_DEVICE_H_

#include <cstdint>
#include <vector>

#include "vt/ops.h"  // vt::Queue

namespace vllm {
struct Glm5NextWeights;  // defined in glm5_next_loader.h (vllm, not vllm::glm5_next)
}  // namespace vllm

namespace vllm::glm5_next {

struct LayerCache;

// True iff BOTH k-pool ops have a CUDA provider. Both, because the family is
// only useful as a pair: the compress op publishes the compacted pool count the
// select op reads, so half a family is not a usable half of the capability.
bool KpoolDeviceOpsAvailable();

// W9c-3 — the device-resident compose forward. Mirrors `Glm5NextHostForward`'s
// signature but routes the nine device-capable arms through `vt::*` device ops
// on the queue, with MLA attention and mHC sites as host-fallback islands (the
// kimi_linear_device.cpp single-queue pattern). Reached when
// `VT_GLM5_NEXT_DEVICE=1`.
//
// On a CPU queue the `vt::*` kernels use float32 accumulation where the host
// reference uses double, so the output agrees within a float-vs-double envelope
// rather than byte-exact. On a GPU the device kernels match upstream PyTorch's
// float32 numerics.
//
// `caches` is null for a one-shot forward, or exactly `num_hidden_layers`
// layer states carried across steps — the same contract as
// `Glm5NextHostForward`.
std::vector<float> Glm5NextDeviceForward(
    const Glm5NextWeights& weights, const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& logits_indices, vt::Queue& queue,
    std::vector<LayerCache>* caches,
    int64_t lm_head_chunk_bytes = int64_t{64} << 20);

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_DEVICE_H_
