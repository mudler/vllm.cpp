// Muse Glimmer registry TU — the ADDITIVE self-registration seam (W0). Follows
// the kimi_k3_registry.cpp / deepseek_v4_registry.cpp seam exactly: a NEW
// translation unit with REGISTER_VLLM_MODEL lines and ZERO edit to any shared
// array.
//
// Upstream registers BOTH architecture strings onto the SAME class
// (registry.py @ vllm#51655): `MuseGlimmerForCausalLM` -> muse_glimmer
// and `MuseGlimmerForConditionalGeneration` -> `MuseGlimmerForCausalLM`. We mirror
// that: one factory, two registered names, so a text-only and a multimodal
// checkpoint both RESOLVE.
//
// W1: the arch RESOLVES, parses config, accounts the structural name map, loads the
// TEXT tower and FORWARDS it. The perception encoder is still W3, so an image or
// video prompt is a pending brick. Muse Glimmer is beyond the pinned oracle
// (555967922) and is anchored to the OPEN vllm#51655 — see porting-inventory §9
// deviation 16 and specs/muse-glimmer.md §0. No speed axis is claimable for this
// model at all while the pin lacks `muse_glimmer`.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/interfaces.h"  // #607 L3 kVisionTowerStageName
#include "vllm/model_executor/models/muse_glimmer.h"
#include "vllm/model_executor/models/muse_glimmer_gguf_weights.h"  // the k-quant arm
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Muse Glimmer: text generation, NOT hybrid (the whole
// tower is dense attention — the iRoPE split is sliding vs full, both attention),
// multimodal (the perception encoder covers image AND video).
inline constexpr ModelInfo kMuseGlimmerInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

class MuseGlimmerLoadedModel final : public LoadedModel {
 public:
  MuseGlimmerLoadedModel(const ModelRegistration& registration,
                         MuseGlimmerWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const MuseGlimmerWeights& weights() const { return weights_; }

  // #607 L3: the mirror of `_tower_model_names` + `StageMissingLayer`'s
  // stage_name (interfaces.py:141,279-282,298). Non-empty ONLY when the
  // checkpoint carries a perception encoder that this load deliberately did not
  // read; a text-only checkpoint reports nothing, because nothing was skipped.
  std::vector<std::string> skipped_towers() const override {
    if (!weights_.vision_skipped) return {};
    return {std::string(kVisionTowerStageName)};
  }

 private:
  MuseGlimmerWeights weights_;
};

std::unique_ptr<LoadedModel> LoadMuseGlimmer(const ModelRegistration& registration,
                                             const HfConfig& config,
                                             const ModelSource& source) {
  // The GGUF k-quant arm (.agents/porting-a-model.md §2). This used to throw
  // "does not support GGUF weights", which was never a decision — the quantized
  // arm simply was not on any list, while a ~17 GB k-quant is what most users of
  // a 30B model can actually run. The loader lives in its OWN translation unit
  // (muse_glimmer_gguf_weights.cpp) and targets the same MuseGlimmerWeights, so
  // everything below this branch is unchanged.
  if (source.kind == ModelSource::Kind::kGguf) {
    if (source.gguf == nullptr)
      throw std::runtime_error("muse_glimmer GGUF model source is empty");
    return std::make_unique<MuseGlimmerLoadedModel>(
        registration, LoadMuseGlimmerFromGguf(*source.gguf, config));
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  // #607 L3: `source.multimodal` is the engine's limits, borrowed. Null on every
  // non-engine caller, which loads the perception encoder exactly as before.
  return std::make_unique<MuseGlimmerLoadedModel>(
      registration, LoadMuseGlimmerForConditionalGenerationWeights(
                        *source.safetensors, config, source.multimodal));
}

void PrepareMuseGlimmer(LoadedModel& model, const HfConfig& config,
                        vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardMuseGlimmer(LoadedModel& model,
                                 const ModelForwardInput& input) {
  auto& mg = ModelAs<MuseGlimmerLoadedModel>(model, "MuseGlimmerForCausalLM");
  const MuseGlimmerWeights& weights = mg.weights();
  // W4 WIRING: the MULTIMODAL branch. When ModelForwardInput.mm is set (the
  // MuseGlimmerGenerateGreedyViaRegistry driver / the runner mm-path) the hidden
  // stream starts from the ALREADY-MERGED inputs_embeds — text rows carrying
  // `embed_norm`, placeholder rows carrying the projected vision soft tokens —
  // mirroring `MuseGlimmerModel.forward`'s `inputs_embeds` branch
  // (muse_glimmer.py:1311-1315). Positions are the 1-D ModelForwardInput::positions:
  // Muse Glimmer has NO MRoPE and NO DeepStack, so no other mm field applies.
  // nullopt on every text step ⇒ the text path below is byte-identical.
  if (input.mm.has_value()) {
    const MultiModalForwardInput& mm = *input.mm;
    VT_CHECK(mm.inputs_embeds_bf16 != nullptr,
             "MuseGlimmer mm forward: null merged-embeds handle on "
             "ModelForwardInput.mm");
    return HostLogits(
        MuseGlimmerModel::ForwardMm(*mm.inputs_embeds_bf16, input.positions,
                                    input.attn_meta, input.attn_kv, weights,
                                    input.queue, input.logits_indices),
        weights.params.text.vocab_size);
  }
  if (input.gather_logits) {
    return MuseGlimmerModel::ForwardDevice(input.token_ids, input.positions,
                                           input.attn_meta, input.attn_kv, weights,
                                           input.queue, input.logits_indices);
  }
  return HostLogits(
      MuseGlimmerModel::Forward(input.token_ids, input.positions, input.attn_meta,
                                input.attn_kv, weights, input.queue,
                                input.logits_indices),
      weights.params.text.vocab_size);
}

const ModelFactory kMuseGlimmerFactory{
    .parse_config = &ParseMuseGlimmerConfig,
    .load_weights = &LoadMuseGlimmer,
    .prepare = &PrepareMuseGlimmer,
    .forward = &ForwardMuseGlimmer,
    .make_kv_cache = &MakeMuseGlimmerKVCache,
    .is_dense_model = true,
};

}  // namespace

v1::KVCacheConfig MakeMuseGlimmerKVCache(const HfConfig& config, int block_size,
                                         int num_blocks) {
  // W1 RESOLVED the W0 placeholder note. The topology IS heterogeneous in masking
  // — `no_rope_layers[i] == 1` layers are SLIDING-window and `== 0` layers are FULL
  // attention (muse_glimmer.py:1167-1168) — but the KV GEOMETRY is uniform across
  // both classes (same num_key_value_heads, same head_dim), so ONLY the window
  // differs. The window is applied at the attention-kernel level
  // (`vt::PagedAttentionArgs::window_size`, muse_glimmer.cpp), exactly as Gemma-2
  // and Laguna do for their interleaved sliding layers, so ONE full-attention group
  // is the correct spec and not a stand-in. The Gemma-4 per-layer spec seam exists
  // for models whose KV geometry differs per layer; Muse Glimmer's does not.
  const MuseGlimmerParams p = ParseMuseGlimmerParams(config);
  const int num_kv_heads = static_cast<int>(p.text.num_key_value_heads);
  const int head_dim = static_cast<int>(p.text.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(block_size, num_kv_heads, head_dim,
                                              v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(muse_glimmer, "MuseGlimmerForCausalLM", kMuseGlimmerFactory,
                    kMuseGlimmerInfo)
REGISTER_VLLM_MODEL(muse_glimmer_mm, "MuseGlimmerForConditionalGeneration",
                    kMuseGlimmerFactory, kMuseGlimmerInfo)

}  // namespace vllm
