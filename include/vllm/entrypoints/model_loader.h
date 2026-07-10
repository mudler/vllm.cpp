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
#include <optional>
#include <string>

#include "vllm/config/scheduler.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
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
};

// Owns the full V1 engine stack (config + weights + tokenizer + Scheduler +
// runner -> Executor -> EngineCore; Input/OutputProcessor -> LLMEngine) for a
// Qwen3.5-family model — the MoE 35B (Qwen3_5MoeForConditionalGeneration) OR the
// dense 27B (Qwen3_5ForConditionalGeneration, W4A4 text path). Only the WEIGHTS
// type and the runner's forward differ by arch; the Scheduler / Executor /
// EngineCore / processors are arch-agnostic (they touch the runner only through
// ModelRunnerBase). The engine holds exactly one of {moe,dense}_weights_ — the
// same {moe,dense} pointer-pair the GPUModelRunner carries, lifted UP the stack.
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

  // Arch-select for FromModelDir: true iff `config` is the DENSE 27B
  // (Qwen3_5ForConditionalGeneration / text_config qwen3_5_text), routed to
  // LoadQwen3_5Dense; false selects the MoE 35B (LoadQwen3_5Moe). num_experts is
  // the structural discriminator (dense == 0, MoE > 0). Exposed for testing the
  // dispatch without a disk load.
  static bool IsDenseArch(const HfConfig& config);

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

  vllm::v1::LLMEngine& engine() { return engine_; }
  const tok::Tokenizer& tokenizer() const { return tokenizer_; }
  const HfConfig& config() const { return config_; }
  int max_model_len() const { return max_model_len_; }

 private:
  // Build the hybrid KV-cache config (one full-attention group + one GDN/mamba
  // group) sized from the config. The runner keys off the spec KIND, not the
  // group-id strings, so this generalizes to any Qwen3.5 layer count.
  static vllm::v1::KVCacheConfig MakeKvConfig(const HfConfig& c, int block_size,
                                              int num_blocks);
  static vllm::SchedulerConfig MakeSchedulerConfig(int max_model_len,
                                                   int max_num_seqs,
                                                   int max_num_batched_tokens);
  // Ensure NONE_HASH is initialized before the scheduler/hasher are built
  // (upstream global init). Idempotent; runs as the first member initializer.
  static bool EnsureNoneHash();

  bool hash_ready_;  // declared first: forces EnsureNoneHash() ahead of the rest.
  HfConfig config_;
  // Exactly one is engaged, selected by the constructor overload; the other stays
  // nullopt. The runner borrows whichever is set (mirrors GPUModelRunner's own
  // {moe,dense}_weights_ pointer-pair). Declared before runner_ so the borrow is
  // live when runner_ is constructed.
  std::optional<Qwen3_5MoeWeights> moe_weights_;
  std::optional<Qwen3_5DenseWeights> dense_weights_;
  tok::Tokenizer tokenizer_;
  int max_model_len_;
  vllm::v1::KVCacheConfig kv_cfg_;
  vllm::v1::Scheduler scheduler_;
  vllm::v1::GPUModelRunner runner_;
  vllm::v1::Executor executor_;
  vllm::v1::EngineCore engine_core_;
  vllm::v1::InputProcessor input_processor_;
  vllm::v1::OutputProcessor output_processor_;
  vllm::v1::LLMEngine engine_;
};

}  // namespace vllm::entrypoints

#endif  // VLLM_ENTRYPOINTS_MODEL_LOADER_H_
