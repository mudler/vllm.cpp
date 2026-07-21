// Qwen3-Coder-30B-A3B (`Qwen3MoeForCausalLM`) registry TU — the first
// full-attention MoE bring-up (W0). Self-registers "Qwen3MoeForCausalLM" via
// REGISTER_VLLM_MODEL and owns the arch-specific entry points: the config hook,
// the full-attention-ONLY KV-cache spec (one FA group, NO MambaSpec/GDN group —
// cloned from qwen3_dense.cpp's MakeQwen3ForCausalLMKVCache), the LoadedModel
// subclass, and the factory. Mirrors the qwen3_dense.cpp seam (new TU + one in-TU
// REGISTER line -> ZERO shared-array edit).
//
// W2/W3 landed: load_weights dispatches to the BF16 per-expert safetensors loader
// (qwen3_moe_weights.cpp), and forward dispatches to Qwen3MoeModel (qwen3_moe.cpp)
// which composes the reusable dense `AttnBlock` (dense_attn_block.h) + the exposed
// `RunMoeBlock` (qwen3_5_moe_block.h) per layer, with NO GDN, NO shared expert, and
// an untied lm_head. See .agents/specs/sweep-qwen3-coder-30b.md §3, §7. The
// correctness gate (near-tie token-exact vs vLLM 0.25.0) is W4.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"           // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"    // HostLogits (W3)
#include "vllm/model_executor/models/qwen3_moe.h"
#include "vllm/platforms/interface.h"  // GetPlatform(device.type).is_cuda()
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Qwen3-Coder: text generation, NOT hybrid (pure
// full-attention MoE — no GDN), NOT multimodal (text-only, no vision tower).
inline constexpr ModelInfo kQwen3MoeInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// Opaque owned model: the loaded Qwen3-Coder MoE weights plus (W7) the model's
// decode CUDA-graph driver state, held here exactly as the 35B/27B registrations
// hold theirs (the graph outlives a single forward call).
class Qwen3MoeLoadedModel final : public LoadedModel {
 public:
  Qwen3MoeLoadedModel(const ModelRegistration& registration,
                      Qwen3MoeWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const Qwen3MoeWeights& weights() const { return weights_; }
  std::unique_ptr<Qwen3MoeDecodeGraph>& decode_graph() { return decode_graph_; }

 private:
  Qwen3MoeWeights weights_;
  std::unique_ptr<Qwen3MoeDecodeGraph> decode_graph_;
};

std::unique_ptr<LoadedModel> LoadQwen3MoeForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  // W2: BF16 safetensors name map + NEW bf16 per-expert loader + untied lm_head.
  // Qwen3-Coder is text-only BF16 safetensors (no GGUF path yet).
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Qwen3MoeForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Qwen3MoeLoadedModel>(
      registration, LoadQwen3MoeForCausalLMWeights(*source.safetensors, config));
}

void PrepareQwen3MoeForCausalLM(LoadedModel& model, const HfConfig& config,
                                vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardQwen3MoeForCausalLM(LoadedModel& model,
                                         const ModelForwardInput& input) {
  auto& qwen = static_cast<Qwen3MoeLoadedModel&>(model);
  const Qwen3MoeWeights& weights = qwen.weights();

  // DECODE CUDA-GRAPH path (W7): route a PURE-DECODE CUDA step through the
  // model's graph driver, which pads the batch up to the nearest captured size
  // and replays that size's graph (mirrors vLLM's FULL_AND_PIECEWISE decode
  // graphs — gpu_model_runner.py capture/replay + compilation/cuda_graph.py
  // pad-to-nearest dispatch @ e24d1b24). Real-row output is bit-identical to the
  // eager forward: the same kernels in the same order over the same buffers, with
  // inert padding rows. This closes the measured ~5 ms/step host/launch tax at
  // concurrency 1 (spec §9 W6 residual 1). Batches above the capture set
  // (max_num_seqs), prefill and mixed steps, and CPU stay eager — the driver
  // itself falls back internally, so this is the single dispatch point.
  //
  // gdn_state_slots carries max_num_reqs for EVERY arch (runner.cpp:374 sets it
  // from max_num_reqs_ regardless of whether the model has GDN layers), so a
  // pure full-attention model reads its capture-size cap from it unchanged.
  if (input.pure_decode &&
      platforms::GetPlatform(input.queue.device.type).is_cuda()) {
    if (!qwen.decode_graph()) {
      qwen.decode_graph() = std::make_unique<Qwen3MoeDecodeGraph>(
          weights, input.config, input.queue, input.gdn_state_slots);
    }
    return qwen.decode_graph()->Step(input.token_ids, input.positions,
                                     input.attn_meta, input.attn_kv);
  }

  // DEVICE-resident logits (sampler-on-device) on the gather path; HOST logits on
  // the opt-out. Qwen3-Coder is pure full-attention MoE (input.gdn_* unused).
  if (input.gather_logits) {
    return Qwen3MoeModel::ForwardDevice(input.token_ids, input.positions,
                                        input.attn_meta, input.attn_kv, weights,
                                        input.config, input.queue,
                                        input.logits_indices);
  }
  return HostLogits(
      Qwen3MoeModel::Forward(input.token_ids, input.positions, input.attn_meta,
                             input.attn_kv, weights, input.config, input.queue,
                             input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kQwen3MoeFactory{
    .parse_config = &ParseQwen3MoeConfig,
    .load_weights = &LoadQwen3MoeForCausalLM,
    .prepare = &PrepareQwen3MoeForCausalLM,
    .forward = &ForwardQwen3MoeForCausalLM,
    .make_kv_cache = &MakeQwen3MoeKVCache,
    .is_dense_model = false,
};

}  // namespace

void ParseQwen3MoeConfig(const HfConfig& config) {
  // Verify the MoE fields the W2 loader / W3 forward consume are materialized by
  // LoadHfConfig. shared_expert_intermediate_size may legitimately be 0
  // (Qwen3-Coder has no shared expert — the no-shared guard, SEAM GAP #3).
  if (config.num_experts <= 0) {
    throw std::runtime_error(
        "Qwen3MoeForCausalLM config: num_experts must be > 0");
  }
  if (config.num_experts_per_tok <= 0 ||
      config.num_experts_per_tok > config.num_experts) {
    throw std::runtime_error(
        "Qwen3MoeForCausalLM config: num_experts_per_tok must be in "
        "[1, num_experts]");
  }
  if (config.moe_intermediate_size <= 0) {
    throw std::runtime_error(
        "Qwen3MoeForCausalLM config: moe_intermediate_size must be > 0");
  }
  (void)config.shared_expert_intermediate_size;  // 0 is valid (no shared expert)
}

v1::KVCacheConfig MakeQwen3MoeKVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  // Full-attention MoE: exactly ONE full-attention KV group, NO MambaSpec/GDN
  // group. Byte-for-byte the pure-dense topology (clone of
  // MakeQwen3ForCausalLMKVCache) — the runner's full-attention-only path
  // (gdn_group_id_ < 0) already handles it with zero runner change.
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      // Spec-driven allocation (MLA campaign W1): the spec carries the paged-KV
      // storage dtype the runner allocates and views with.
      std::make_shared<v1::FullAttentionSpec>(
          block_size, num_kv_heads, head_dim, v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(qwen3_moe, "Qwen3MoeForCausalLM", kQwen3MoeFactory,
                    kQwen3MoeInfo)

}  // namespace vllm
