// DeepSeek-V4-Flash (`DeepseekV4ForCausalLM`) registry TU — the ADDITIVE self-
// registration seam for the DeepSeek-V4 bring-up (`CLAIM-DEEPSEEK-V4-IMPL`,
// W1/W2). Follows the deepseek_v2_registry.cpp / gemma4_registry.cpp seam exactly:
// a NEW translation unit with ONE REGISTER_VLLM_MODEL line and ZERO edit to any
// shared array. It owns the arch entry points: the config hook (config-descent
// validation), the KV-cache spec (STUB), the LoadedModel subclass + factory.
//
// Registry routing upstream (`registry.py:94`) sends `DeepseekV4ForCausalLM` into
// `vllm.models.deepseek_v4`. We register exactly that ONE string. The `DSparkDraftModel`
// (registry.py:596) speculator is a separate row (INVENTORIED). The native MTP head
// `DeepSeekV4MTPModel` (registry.py:617 -> vllm.models.deepseek_v4.DeepSeekV4MTP) is
// WIRED at W1: its tiny-config draft forward + the lossless self-spec verify land in
// deepseek_v4.cpp (`DeepseekV4MtpDraftLogitsHost`) + deepseek_v4.h + the loader's
// absence guard (`DeepseekV4GgufHasMtp`). It is NOT registered as an engine
// speculator here yet — the DS4-native propose/verify DECODE-LOOP integration + the
// engine spec-config seam are named residuals (R2/R3), and the real-model gate is
// weight-BLOCKED (both shipped GGUFs dropped the nextn tail). See
// `.agents/specs/deepseek-v4-mtp.md`.
//
// SCOPE HONESTY: registering this arch makes it RESOLVE + parse config + account
// for the checkpoint tensors; it does NOT make it forward. `DeepseekV4Model` is a
// W3-W8 stub that VT_CHECK(false, ...) — so a load succeeds through the loader's
// accounting pass but a FORWARD loudly reports the pending brick. The model-matrix
// row stays SPIKE until the strict gate (W8) passes. HW note: the NVFP4 checkpoint
// is 156.7 GiB and does NOT fit ONE GB10 (119 GiB) — the runnable oracle gate (W1)
// is MEMORY-INFEASIBLE on a single Spark. See .agents/specs/deepseek-v4-flash.md.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for DeepSeek-V4: text generation, NOT hybrid (MLA is
// full attention over a paged cache), NOT multimodal.
inline constexpr ModelInfo kDeepseekV4Info{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class DeepseekV4LoadedModel final : public LoadedModel {
 public:
  DeepseekV4LoadedModel(const ModelRegistration& registration,
                        DeepseekV4Weights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const DeepseekV4Weights& weights() const { return weights_; }

 private:
  DeepseekV4Weights weights_;
};

std::unique_ptr<LoadedModel> LoadDeepseekV4ForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // W2b (LANDED): the single-Spark `deepseek4` GGUF vehicle
    // (`unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_XXS`, ~91 GiB — the only build that
    // fits ONE GB10's 119 GiB unified pool). The keep-quant compute for the
    // ~2-3-bit routed experts (IQ2_XXS/IQ3_XXS/Q2_K, vt::MatmulBTQuant) + the
    // blk.N.* -> V4 name map (EXACT 1328/1328 coverage) landed in W8; W2b wires
    // them into the DeepseekV4 weight towers (keep-quant blocks stay COMPRESSED;
    // the small MHC/DSA/norm/embed tensors dequant). See
    // .agents/specs/deepseek-v4-flash.md §W2b. HONEST 3-state: the materialization
    // is gated at tiny synthetic shape (test_deepseek_v4_gguf_load.cpp); the real
    // 91 GiB load + generate is the W8-final operational residual (download + DGX).
    if (source.gguf == nullptr) {
      throw std::runtime_error("deepseek-v4 GGUF model source is empty");
    }
    return std::make_unique<DeepseekV4LoadedModel>(
        registration, LoadDeepseekV4FromGguf(*source.gguf, config));
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<DeepseekV4LoadedModel>(
      registration, LoadDeepseekV4ForCausalLMWeights(*source.safetensors, config));
}

void PrepareDeepseekV4ForCausalLM(LoadedModel& model, const HfConfig& config,
                                  vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardDeepseekV4ForCausalLM(LoadedModel& model,
                                           const ModelForwardInput& input) {
  auto& ds = ModelAs<DeepseekV4LoadedModel>(model, "DeepseekV4ForCausalLM");
  const DeepseekV4Weights& weights = ds.weights();
  if (input.gather_logits) {
    return DeepseekV4Model::ForwardDevice(input.token_ids, input.positions,
                                          input.attn_meta, input.attn_kv, weights,
                                          input.queue, input.logits_indices);
  }
  return HostLogits(
      DeepseekV4Model::Forward(input.token_ids, input.positions, input.attn_meta,
                               input.attn_kv, weights, input.queue,
                               input.logits_indices),
      weights.params.vocab_size);
}

const ModelFactory kDeepseekV4Factory{
    .parse_config = &ParseDeepseekV4Config,
    .load_weights = &LoadDeepseekV4ForCausalLM,
    .prepare = &PrepareDeepseekV4ForCausalLM,
    .forward = &ForwardDeepseekV4ForCausalLM,
    .make_kv_cache = &MakeDeepseekV4KVCache,
    .is_dense_model = false,
};

}  // namespace

v1::KVCacheConfig MakeDeepseekV4KVCache(const HfConfig& config, int block_size,
                                        int num_blocks) {
  // STUB (W3): V4's TRUE KV topology is the fp8_ds_mla UE8M0 576B-paged latent
  // (attention.py:89) PLUS the DSA indexer/compressor caches — a multi-cache
  // geometry not yet representable. We emit ONE placeholder MLA group sized to
  // the compressed latent + rope so the arch RESOLVES and the spec builder is
  // wired; the real topology (and the indexer/compressor caches) is a named W3
  // residual. Never exercised this pass — the forward VT_CHECKs pending.
  const DeepseekV4Params p = ParseDeepseekV4Params(config);
  // Placeholder latent width: the wkv-compressed KV (num_key_value_heads * 512
  // rows in the checkpoint) is head-independent for MLA; use head_dim + rope as
  // an honest per-page upper bound until the fp8_ds_mla geometry lands.
  const int head_size =
      static_cast<int>(p.head_dim + p.qk_rope_head_dim);  // 512 + 64 = 576 (W3 TODO)

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"mla"},
      std::make_shared<v1::MLAAttentionSpec>(block_size, head_size,
                                             v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(deepseek_v4, "DeepseekV4ForCausalLM", kDeepseekV4Factory,
                    kDeepseekV4Info)

}  // namespace vllm
