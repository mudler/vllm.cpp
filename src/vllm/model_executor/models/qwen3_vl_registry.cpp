// Qwen3-VL (`Qwen3VLForConditionalGeneration`) registry TU — MM-ENGINE-FORWARD.
//
// Self-registers "Qwen3VLForConditionalGeneration" via REGISTER_VLLM_MODEL so the
// ENGINE (ModelRegistry::Resolve/Load/Prepare/Forward) drives the vision-language
// model, instead of the standalone Qwen3VLGenerateGreedy driver running OUTSIDE
// the registry. This is the additive-TU seam (new TU + one REGISTER line → ZERO
// shared-array edit), exactly like olmo2_registry.cpp / qwen3_dense.cpp.
//
// The registered forward (ForwardQwen3VL) folds the M2c forked decode into the
// per-step ModelRegistry::Forward contract: it consumes the ALREADY-MERGED
// embeddings + 3-D MRoPE positions + DeepStack carried on ModelForwardInput.mm
// (the runner mm-path / Qwen3VLGenerateGreedyViaRegistry driver fills them via the
// vision tower + `_merge_multimodal_embeddings`), and calls the SHARED
// Qwen3VLForwardStepLastLogits — the same step the standalone driver runs, so the
// two paths are numerically identical by construction. mm is nullopt for text
// requests, so this registration cannot perturb any text model.
//
// Ported from vllm/model_executor/models/registry.py (the Qwen3-VL entry) +
// qwen3_vl.py forward — see qwen3_vl.h / qwen3_vl.cpp for the port map.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/model_executor/models/qwen3_vl.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Qwen3-VL: text generation + MULTIMODAL (vision
// tower). The 4B text backbone is a PLAIN dense full-attention model → NOT hybrid
// (no GDN linear-attention state). This is the FIRST non-hybrid multimodal
// registration (the two Qwen3.5 ConditionalGeneration wrappers are hybrid+mm).
inline constexpr ModelInfo kQwen3VLInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

class Qwen3VLLoadedModel final : public LoadedModel {
 public:
  Qwen3VLLoadedModel(const ModelRegistration& registration, Qwen3VLWeights weights)
      : LoadedModel(registration),
        owned_weights_(std::move(weights)),
        weights_(&*owned_weights_) {}
  Qwen3VLLoadedModel(const ModelRegistration& registration,
                     const Qwen3VLWeights& weights, BorrowedWeightsTag)
      : LoadedModel(registration), weights_(&weights) {}

  const Qwen3VLWeights& weights() const { return *weights_; }

  // Build-on-first-use persistent MRoPE cos|sin cache. Deterministic + built with
  // the SAME RopeArgs/Pmax as VLGenerateCore, so it is bit-identical to the
  // standalone driver's cache — the registered and standalone paths stay numeric-
  // identical. (The gate driver is single-threaded; the mutex keeps a stray
  // concurrent Prepare/Forward safe.)
  const Qwen3VLCosSinCache& CosSinCache(vt::Queue& queue, const HfConfig& config) {
    std::lock_guard<std::mutex> lock(cos_sin_mu_);
    if (!cos_sin_.storage) {
      cos_sin_ = Qwen3VLMakeCosSinCache(queue, config);
    }
    return cos_sin_;
  }

 private:
  std::optional<Qwen3VLWeights> owned_weights_;
  const Qwen3VLWeights* weights_ = nullptr;
  std::mutex cos_sin_mu_;
  Qwen3VLCosSinCache cos_sin_;
};

std::unique_ptr<LoadedModel> LoadQwen3VLForConditionalGeneration(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Qwen3VLForConditionalGeneration does not support "
        "GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Qwen3VLLoadedModel>(
      registration, LoadQwen3VLWeights(*source.safetensors, config));
}

void PrepareQwen3VLForConditionalGeneration(LoadedModel& model,
                                            const HfConfig& config,
                                            vt::Queue& queue) {
  // Warm the persistent cos|sin cache so the first forward step does not build it.
  // The call stays INLINE on the checked reference rather than gaining a local:
  // `ModelAs` establishes the dynamic type before the member call either way, so
  // a binding would change this site's shape without changing what it does.
  ModelAs<Qwen3VLLoadedModel>(model, "Qwen3VLForConditionalGeneration")
      .CosSinCache(queue, config);
}

ForwardLogits ForwardQwen3VLForConditionalGeneration(
    LoadedModel& model, const ModelForwardInput& input) {
  auto& vl = ModelAs<Qwen3VLLoadedModel>(model, "Qwen3VLForConditionalGeneration");
  VT_CHECK(input.mm.has_value(),
           "Qwen3VLForConditionalGeneration registered forward requires "
           "multimodal inputs (ModelForwardInput.mm). Text-only Qwen3-VL through "
           "this arch is a named MM-ENGINE-FORWARD residual.");
  const MultiModalForwardInput& mm = *input.mm;
  VT_CHECK(mm.inputs_embeds_bf16 != nullptr && mm.positions3 != nullptr &&
               mm.deepstack_bf16 != nullptr,
           "Qwen3-VL mm forward: null merged-embeds / positions3 / deepstack "
           "handle on ModelForwardInput.mm");
  const int64_t num_tokens = static_cast<int64_t>(mm.positions3->size()) / 3;
  const Qwen3VLCosSinCache& cos_sin = vl.CosSinCache(input.queue, input.config);
  // DEVICE-resident logits (sampler-on-device) on the gather path — the mm forward
  // produces exactly the single last-token [1, vocab] row, kept ON DEVICE so the
  // greedy driver / runner samples it straight off device (vt::GreedyArgmax) with
  // no full-vocab D2H. Mirrors the text device path (qwen3_dense.cpp:86). The host
  // path (gather_logits=false) reproduces the old download-then-sample A/B.
  if (input.gather_logits) {
    return Qwen3VLForwardStepLastLogitsDevice(
        input.queue, vl.weights().text, input.config, *mm.inputs_embeds_bf16,
        *mm.positions3, num_tokens, *mm.deepstack_bf16, mm.deepstack_levels,
        cos_sin.tensor, input.attn_meta, input.attn_kv);
  }
  std::vector<float> logits = Qwen3VLForwardStepLastLogits(
      input.queue, vl.weights().text, input.config, *mm.inputs_embeds_bf16,
      *mm.positions3, num_tokens, *mm.deepstack_bf16, mm.deepstack_levels,
      cos_sin.tensor, input.attn_meta, input.attn_kv);
  return HostLogits(std::move(logits), input.config.vocab_size);
}

v1::KVCacheConfig MakeQwen3VLForConditionalGenerationKVCache(const HfConfig& config,
                                                            int block_size,
                                                            int num_blocks) {
  // The 4B VL text backbone is pure dense: exactly ONE full-attention KV group,
  // NO MambaSpec/GDN group (identical topology to Qwen3ForCausalLM).
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(block_size, num_kv_heads, head_dim,
                                              v1::ResolveKvCacheDType()));
  return kv;
}

void ParseQwen3VLForConditionalGenerationConfig(const HfConfig& config) {
  // LoadHfConfig already resolves the Qwen3-VL text_config onto the top-level
  // HfConfig (hidden 2560, 36 layers, head_dim 128, kv 8, rope_theta 5e6, tied).
  // No extra normalization needed — the seam for future validation.
  (void)config;
}

const ModelFactory kQwen3VLFactory{
    .parse_config = &ParseQwen3VLForConditionalGenerationConfig,
    .load_weights = &LoadQwen3VLForConditionalGeneration,
    .prepare = &PrepareQwen3VLForConditionalGeneration,
    .forward = &ForwardQwen3VLForConditionalGeneration,
    .make_kv_cache = &MakeQwen3VLForConditionalGenerationKVCache,
    .is_dense_model = true,
};

}  // namespace

std::unique_ptr<LoadedModel> MakeQwen3VLLoadedModel(Qwen3VLWeights weights) {
  return std::make_unique<Qwen3VLLoadedModel>(
      RegistrationFor("Qwen3VLForConditionalGeneration"), std::move(weights));
}

std::unique_ptr<LoadedModel> BorrowQwen3VLLoadedModel(
    const Qwen3VLWeights& weights) {
  return std::make_unique<Qwen3VLLoadedModel>(
      RegistrationFor("Qwen3VLForConditionalGeneration"), weights,
      BorrowedWeightsTag{});
}

REGISTER_VLLM_MODEL(qwen3_vl, "Qwen3VLForConditionalGeneration", kQwen3VLFactory,
                    kQwen3VLInfo)

}  // namespace vllm
