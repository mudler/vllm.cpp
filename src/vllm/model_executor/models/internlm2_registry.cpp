// InternLM2 (`InternLM2ForCausalLM`) registry TU — the near-additive dense
// bring-up whose ONLY delta is a loader-side `wqkv` de-interleave (ZERO new
// compute kernel). Self-registers "InternLM2ForCausalLM" via REGISTER_VLLM_MODEL
// and owns the arch entry points: the config hook, the full-attention-ONLY
// KV-cache spec, the LoadedModel subclass, and the factory. Mirrors the
// llama_registry.cpp seam (new TU + one in-TU REGISTER line -> ZERO shared-array
// edit).
//
// The forward + KV topology are REUSED VERBATIM from the Qwen3-dense path
// (InternLM2Model == Qwen3DenseModel): InternLM2 is that forward with qk-norm
// skipped, plain NeoX rope (theta 1e6; rope_scaling "dynamic" is identity for
// in-window contexts), and the fused `wqkv` split done at LOAD. So this TU only
// wires the loader + forward hooks. See
// .agents/specs/sweep-recent-dense-batch.md §0.2 row 5.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/internlm2.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::DeviceTokenIdsScope
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for InternLM2: text generation, NOT hybrid (pure
// full-attention), NOT multimodal (text-only checkpoint).
inline constexpr ModelInfo kInternLM2Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// Opaque owned model: holds the loaded InternLM2 dense weights (shared dense
// container); the forward reuses the Qwen3-dense forward machinery.
class InternLM2LoadedModel final : public LoadedModel {
 public:
  InternLM2LoadedModel(const ModelRegistration& registration,
                       InternLM2Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const InternLM2Weights& weights() const { return weights_; }
  // W7: the shared pure-dense decode CUDA-graph driver state (InternLM2Model ==
  // Qwen3DenseModel, so this is the SAME driver the Qwen3-dense path uses).
  std::unique_ptr<Qwen3DenseDecodeGraph>& decode_graph() { return decode_graph_; }

 private:
  InternLM2Weights weights_;
  std::unique_ptr<Qwen3DenseDecodeGraph> decode_graph_;
};

std::unique_ptr<LoadedModel> LoadInternLM2ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture InternLM2ForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<InternLM2LoadedModel>(
      registration, LoadInternLM2ForCausalLMWeights(*source.safetensors, config));
}

void PrepareInternLM2ForCausalLM(LoadedModel& model, const HfConfig& config,
                                 vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardInternLM2ForCausalLM(LoadedModel& model,
                                          const ModelForwardInput& input) {
  // ROW-SERVE-ASYNC-DENSE-MIRROR (sibling scope): same guard qwen3_dense.cpp
  // establishes. This family routes through the SAME shared EmbedInto (decode
  // graph + eager Forward/ForwardDevice), so without it the async runner's
  // device-resident ids are ignored and the stale host `token_ids` races the
  // combine's device write — the #31 batch-1 token-0 degeneration. Null on every
  // non-async-CUDA path, RAII-scoped, byte-identical when the mirror is off.
  const detail::DeviceTokenIdsScope device_ids_scope(
      input.device_token_ids, static_cast<int64_t>(input.token_ids.size()));
  auto& im2 = ModelAs<InternLM2LoadedModel>(model, "InternLM2ForCausalLM");
  const InternLM2Weights& weights = im2.weights();
  // Shared pure-dense decode CUDA-graph (opt-in via VLLM_CPP_QWEN3_DENSE_DECODE_
  // GRAPH); std::nullopt falls through to the byte-identical eager path below.
  if (auto fl = DenseDecodeGraphForward(im2.decode_graph(), weights, input))
    return std::move(*fl);
  // DEVICE-resident logits (sampler-on-device) on the gather path; HOST logits on
  // the opt-out. InternLM2 is pure full-attention (input.gdn_* unused).
  if (input.gather_logits) {
    return InternLM2Model::ForwardDevice(input.token_ids, input.positions,
                                         input.attn_meta, input.attn_kv, weights,
                                         input.config, input.queue,
                                         input.logits_indices);
  }
  return HostLogits(
      InternLM2Model::Forward(input.token_ids, input.positions, input.attn_meta,
                              input.attn_kv, weights, input.config, input.queue,
                              input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kInternLM2Factory{
    .parse_config = &ParseInternLM2ForCausalLMConfig,
    .load_weights = &LoadInternLM2ForCausalLM,
    .prepare = &PrepareInternLM2ForCausalLM,
    .forward = &ForwardInternLM2ForCausalLM,
    .make_kv_cache = &MakeInternLM2ForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

void ParseInternLM2ForCausalLMConfig(const HfConfig& config) {
  // LoadHfConfig already materializes + validates every consumed InternLM2 field
  // (including the rope_scaling "dynamic" dictionary). Validate full (non-partial)
  // NeoX rope — InternLM2 rotates the whole head_dim.
  VT_CHECK(config.rotary_dim == config.head_dim,
           "internlm2: expected full NeoX rope (rotary_dim == head_dim)");
}

v1::KVCacheConfig MakeInternLM2ForCausalLMKVCache(const HfConfig& config,
                                                  int block_size, int num_blocks) {
  // Pure dense: exactly ONE full-attention KV group, NO MambaSpec/GDN group.
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(
          block_size, num_kv_heads, head_dim, v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(internlm2, "InternLM2ForCausalLM", kInternLM2Factory,
                    kInternLM2Info)

}  // namespace vllm
