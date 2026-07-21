// ORIGINAL packaging helper (no upstream mirror). Factors the model-load +
// engine-stack wiring shared by the OpenAI HTTP server (examples/server/main.cpp)
// and the C ABI (src/capi/vllm_c.cpp) into one place, so both drive the same
// LLMEngine construction path. The wiring itself mirrors the M1.8 LLMEngine
// __init__ (vllm/v1/engine/llm_engine.py @ e24d1b24) and the test harness
// (tests/vllm/v1/test_llm_engine.cpp); this file only owns the pieces + their
// lifetimes so the LLMEngine's by-reference seams stay valid.
#ifndef VLLM_ENTRYPOINTS_MODEL_LOADER_H_
#define VLLM_ENTRYPOINTS_MODEL_LOADER_H_

#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "vllm/config/scheduler.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/async_scheduler.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/async_llm.h"
#include "vllm/v1/engine/input_processor.h"
#include "vllm/v1/engine/llm_engine.h"
#include "vllm/v1/engine/output_processor.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"

namespace vllm::entrypoints {

// Knobs that size the engine stack. Zero/negative fields fall back to the
// documented defaults (see below), so a default-constructed EngineParams is
// valid.
struct EngineParams {
  int block_size = 32;     // KV block size (tokens/block).
  int num_blocks = 256;    // KV blocks to allocate.
  int max_model_len = 0;   // 0 => config.max_position_embeddings.
  int max_num_seqs = 8;    // max concurrent sequences.
  // Per-step token budget (the chunked-prefill knob). 0 => the bounded PER-ARCH
  // default (see LoadedEngine::ResolveMaxNumBatchedTokens): dense arch 2048 flat
  // (vLLM's DEFAULT_MAX_NUM_BATCHED_TOKENS, vllm/config/scheduler.py:42 @
  // e24d1b24); MoE arch 8192 at max_num_seqs >= 32 else 4096 (GB10-tuned). The
  // budget does NOT scale with max_num_seqs, so a long/many-request prefill is
  // split across steps (enable_chunked_prefill is always true) and the per-step
  // GDN chunked-scan activation stays bounded regardless of concurrency. This is
  // the fix for the 27B 8x1024 conc-8 OOM: the old max_model_len*max_num_seqs
  // product let an 8x1024 (8192-token) prefill run in ONE step, blowing the GDN
  // prefill activation.
  int max_num_batched_tokens = 0;
  // Mirrors vLLM's tri-state --[no-]enable-prefix-caching resolution. Hybrid
  // and attention-free generation models default OFF at the parity pin;
  // decoder-only models default ON. An explicit value overrides the model
  // capability default.
  std::optional<bool> enable_prefix_caching = std::nullopt;
  // Scheduling policy (mirrors SchedulerConfig.policy / the vLLM
  // --scheduling-policy flag). Default fcfs. Set kPriority to schedule by
  // (priority, arrival_time); requests then carry a `priority` field.
  vllm::SchedulerPolicy policy = vllm::SchedulerPolicy::kFCFS;
};

// Owns the full V1 engine stack (config + weights + tokenizer + Scheduler +
// runner -> Executor -> EngineCore; Input/OutputProcessor -> LLMEngine) for a
// registered model. The concrete weights/forward are held behind LoadedModel;
// the Scheduler / Executor / EngineCore / processors are arch-agnostic (they
// touch the runner only through ModelRunnerBase).
// Members are declared in dependency order so the LLMEngine's by-reference
// collaborators stay valid for this object's lifetime. NON-COPYABLE /
// NON-MOVABLE (the internal references would dangle) — always heap-hold behind a
// unique_ptr and hand out engine() by reference.
class LoadedEngine {
 public:
  // Build the stack from already-loaded model pieces. This is the shared seam:
  // FromModelDir() loads config/tokenizer/weights from disk then calls this, and
  // tests construct it directly with synthetic in-memory weights (no disk). The
  // pieces are moved into members that outlive every collaborator that
  // references them.
  LoadedEngine(HfConfig config, Qwen3_5MoeWeights weights,
               tok::Tokenizer tokenizer, const EngineParams& params);

  // DENSE-arch overload (27B). Identical stack; the runner runs the dense
  // Qwen3_5DenseModel::Forward over the dense weights instead of the MoE forward.
  LoadedEngine(HfConfig config, Qwen3_5DenseWeights weights,
               tok::Tokenizer tokenizer, const EngineParams& params);

  LoadedEngine(const LoadedEngine&) = delete;
  LoadedEngine& operator=(const LoadedEngine&) = delete;
  LoadedEngine(LoadedEngine&&) = delete;
  LoadedEngine& operator=(LoadedEngine&&) = delete;

  // Load config.json + tokenizer.json + *.safetensors from `model_dir` and build
  // the stack. Throws std::runtime_error on any load failure (bad path, missing
  // shards, unparseable config).
  static std::unique_ptr<LoadedEngine> FromModelDir(const std::string& model_dir,
                                                    const EngineParams& params);

  // Resolve the per-step token budget (max_num_batched_tokens) for chunked
  // prefill. An explicit EngineParams override wins; otherwise a PER-ARCH
  // default (see the definition in model_loader.cpp for the measurements):
  //  * DENSE arch: 2048 flat — vLLM's own scheduler default
  //    (DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048, vllm/config/scheduler.py:42 @
  //    e24d1b24). The dense prefill is expensive per token; a bigger budget
  //    lets one giant mixed step run several full prompts' prefill and starves
  //    every decode stream behind it.
  //  * MoE arch: the GB10-tuned concurrency-aware budget (8192 at
  //    max_num_seqs >= 32, else 4096) — the cheap A3B expert prefill wants the
  //    bigger chunk.
  // Invariants (SchedulerConfig.verify_max_model_len,
  // vllm/config/scheduler.py:87): result >= max_num_seqs; the tiny-model
  // (max_model_len * max_num_seqs) ceiling is preserved. Exposed for testing
  // the default policy without a disk load.
  static int ResolveMaxNumBatchedTokens(const EngineParams& params,
                                        int max_model_len, bool is_dense_arch);
  static bool ResolveEnablePrefixCaching(const EngineParams& params,
                                         const ModelInfo& model_info);

  vllm::v1::LLMEngine& engine() { return engine_; }
  // Lazily start W2's EngineCoreProc + output-handler threads. Once created,
  // online/server callers use this frontend rather than the synchronous
  // LLMEngine over the same scheduler/executor.
  vllm::v1::AsyncLLM& async_engine();
  const tok::Tokenizer& tokenizer() const { return tokenizer_; }
  const HfConfig& config() const { return config_; }
  int max_model_len() const { return max_model_len_; }
  bool prefix_caching_enabled() const { return prefix_caching_enabled_; }
  const vllm::v1::GPUModelRunner& runner() const { return runner_; }

  // The async-scheduling enable-flip, resolved ONCE at construction (W3
  // ENG-ASYNC-SCHED). `async_scheduling_enabled()` is
  // AsyncSchedulingEnabled(SchedulerConfig::ResolveAsyncScheduling(
  // runner_.runner_supports_async())) — i.e. vLLM's default-ON-when-compatible
  // resolution (vllm/config/vllm.py:990-1038) gated on the MRV2 runner
  // advertising the async device path (VT_ASYNC_RUNNER), with the house
  // VT_ASYNC_SCHED=0 rollback applied. When ON, scheduler() is an AsyncScheduler
  // and max_concurrent_batches() is 2 (depth-2 batch queue, step_with_batch_queue);
  // OFF keeps the byte-identical synchronous Scheduler + depth-1. Since the
  // 2026-07-17 flip the production default (no env) resolves ON (VT_ASYNC_RUNNER
  // default ON → AsyncScheduler + mcb=2), mirroring vLLM; VT_ASYNC_RUNNER=0 or
  // VT_ASYNC_SCHED=0 roll it back in the same binary.
  bool async_scheduling_enabled() const { return async_scheduling_enabled_; }
  int max_concurrent_batches() const { return max_concurrent_batches_; }
  // The engine's scheduler (Scheduler or, under async scheduling, AsyncScheduler).
  // Exposed for the construction-resolution gate; production drives it via
  // engine()/async_engine().
  const vllm::v1::Scheduler& scheduler() const { return *scheduler_; }

  // ResolveAsyncEnabled: the construction-time resolution, factored out so the
  // CPU construction-matrix test can assert it directly over the
  // runner_supports_async x VT_ASYNC_SCHED matrix without a disk load. Applies
  // SchedulerConfig::ResolveAsyncScheduling then the VT_ASYNC_SCHED rollback env.
  static bool ResolveAsyncEnabled(const vllm::SchedulerConfig& scheduler_config,
                                  bool runner_supports_async);

 private:
  // Type-erased constructor used by FromModelDir and the concrete-weight
  // compatibility overloads above.
  LoadedEngine(HfConfig config, std::unique_ptr<LoadedModel> model,
               tok::Tokenizer tokenizer, const EngineParams& params,
               vt::Queue* preselected_queue = nullptr);

  static vllm::SchedulerConfig MakeSchedulerConfig(
      int max_model_len, int max_num_seqs, int max_num_batched_tokens,
      vllm::SchedulerPolicy policy = vllm::SchedulerPolicy::kFCFS);
  // Construct the concrete scheduler for the resolved mode: an AsyncScheduler
  // (async scheduling ON) or the synchronous Scheduler. Mirrors upstream
  // get_scheduler_cls (scheduler.py:180-189) selecting AsyncScheduler when
  // async_scheduling. Both take the same ctor arguments.
  static std::unique_ptr<vllm::v1::Scheduler> MakeScheduler(
      bool async_enabled, vllm::SchedulerConfig scheduler_config,
      vllm::v1::KVCacheConfig kv_cache_config, int block_size,
      bool enable_caching);
  // Ensure NONE_HASH is initialized before the scheduler/hasher are built
  // (upstream global init). Idempotent; runs as the first member initializer.
  static bool EnsureNoneHash();
  // Mirrors kernel_warmup.py::flashinfer_autotune: before any async/server
  // frontend starts, one maximum-token synthetic request runs under the NVFP4
  // all-bucket autotune scope. CUDA dense W4A4 only; CPU/other models are no-op.
  void WarmupKernels();

  bool hash_ready_;  // declared first: forces EnsureNoneHash() ahead of the rest.
  HfConfig config_;
  // Concrete weights and model-specific runtime state behind the central
  // registry contract. Declared before runner_ so its borrow remains live.
  std::unique_ptr<LoadedModel> model_;
  tok::Tokenizer tokenizer_;
  int max_model_len_;
  int max_num_batched_tokens_;
  bool prefix_caching_enabled_;
  vllm::v1::KVCacheConfig kv_cfg_;
  // runner_ is declared BEFORE the scheduler (W3): the async-scheduling flip is
  // resolved from runner_.runner_supports_async(), so the runner must be fully
  // constructed before async_scheduling_enabled_ / scheduler_ are initialized.
  vllm::v1::GPUModelRunner runner_;
  // Resolved once, in dependency order after runner_ (see the accessors above).
  bool async_scheduling_enabled_;
  int max_concurrent_batches_;
  // Polymorphic scheduler: a plain Scheduler by default, an AsyncScheduler when
  // async_scheduling_enabled_. Heap-held so the concrete class is chosen at
  // construction; engine_core_ / async_engine_ share this one instance by
  // reference (*scheduler_).
  std::unique_ptr<vllm::v1::Scheduler> scheduler_;
  vllm::v1::Executor executor_;
  vllm::v1::EngineCore engine_core_;
  vllm::v1::InputProcessor input_processor_;
  vllm::v1::OutputProcessor output_processor_;
  vllm::v1::BlockHasher block_hasher_;
  vllm::v1::LLMEngine engine_;
  std::mutex async_engine_mutex_;
  std::unique_ptr<vllm::v1::AsyncLLM> async_engine_;
};

}  // namespace vllm::entrypoints

#endif  // VLLM_ENTRYPOINTS_MODEL_LOADER_H_
