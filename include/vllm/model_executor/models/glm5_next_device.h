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

namespace vllm::glm5_next {

// True iff BOTH k-pool ops have a CUDA provider. Both, because the family is
// only useful as a pair: the compress op publishes the compacted pool count the
// select op reads, so half a family is not a usable half of the capability.
bool KpoolDeviceOpsAvailable();

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_DEVICE_H_
