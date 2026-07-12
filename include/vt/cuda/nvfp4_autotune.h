// vllm.cpp — process-local scope for FlashInfer-style pre-serve NVFP4 tuning.
//
// Mirrors vLLM model_executor/warmup/kernel_warmup.py::flashinfer_autotune and
// FlashInfer autotuner.py's context-managed optimization-profile generation.
// The shared model loader opens this scope around one maximum-token dummy
// request. Each W4A4 GEMM then materializes every hybrid M profile in memory
// before serving starts; later cache misses remain legal but are diagnostic.
#pragma once

#include <cstdint>

namespace vt::cuda {

struct Nvfp4AutotuneWarmupStats {
  uint64_t scopes_started = 0;
  uint64_t scopes_completed = 0;
  uint64_t profiles_requested = 0;
  uint64_t profiles_tuned = 0;
  uint64_t lazy_misses = 0;
};

class Nvfp4AutotuneWarmupScope {
 public:
  explicit Nvfp4AutotuneWarmupScope(uint32_t max_num_tokens);
  ~Nvfp4AutotuneWarmupScope();

  Nvfp4AutotuneWarmupScope(const Nvfp4AutotuneWarmupScope&) = delete;
  Nvfp4AutotuneWarmupScope& operator=(const Nvfp4AutotuneWarmupScope&) = delete;

  // Mark the surrounding maximum-token dummy run successful. Destruction
  // without Complete() cancels the scope and does not enable lazy-miss reports.
  void Complete();

 private:
  uint32_t max_num_tokens_ = 0;
  uint64_t requested_before_ = 0;
  uint64_t tuned_before_ = 0;
  bool active_ = false;
  bool completed_ = false;
};

Nvfp4AutotuneWarmupStats GetNvfp4AutotuneWarmupStats();

}  // namespace vt::cuda
