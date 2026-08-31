// GLM-5.3 (`GlmMoeDsaForCausalLM`) registry TU — W2 of
// `.agents/specs/glm-dsa-latest-deepseek.md` §3.7, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
//
// Follows the `kimi_k3_registry.cpp` / `deepseek_v4_registry.cpp` seam exactly:
// a NEW translation unit with ONE `REGISTER_VLLM_MODEL` line and ZERO edit to
// any shared array. It owns the arch entry points: the config hook, the MLA
// KV-cache spec, the `LoadedModel` subclass and the factory.
//
// WHY THIS IS NOT A `Glm4MoeLiteForCausalLM`-STYLE COMPOSITION OVER DEEPSEEK-V2.
// GLM-4.7-Flash could compose deepseek_v2's forward and loader because every
// difference was a config value that code already read. GLM-5.3 is not that:
// it is a V3.2 checkpoint, and `ParseDeepseekV2Params` refuses `index_topk`
// (`deepseek_v2_weights.cpp:358-364`) precisely so that a sparse-indexer
// checkpoint cannot land on the dense-attention forward. Composing here would
// mean relaxing that refusal for DeepSeek-V2 as well.
//
// SCOPE, AS IT NOW STANDS. W2 made the architecture RESOLVE and its config
// PARSE, from a `config.json` and from a `glm-dsa` GGUF header alike. W7 made
// the GGUF arm LOAD. W9 (#2214) makes it FORWARD: `GlmMoeDsaModel::Forward` /
// `::ForwardDevice` in `glm_moe_dsa_forward.cpp` produce logits, and the
// refusal that remains is the one STEP shape this build cannot serve — a
// resumed request whose selection prunes, which needs the indexer KV side cache
// `KV-DSV4-MULTICACHE` owns (spec O4, #1925, #2323). The safetensors arm
// refuses permanently (spec D1).
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/glm_moe_dsa.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits carrier
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {
namespace {

// `registry.py:117` at the pin routes `GlmMoeDsaForCausalLM` into `deepseek_v2`,
// whose `_ModelInfo` is text generation, NOT hybrid (MLA is still full attention
// over a paged cache) and NOT multimodal. GLM-5.3 changes none of that:
// `deepseek_v2.py:1930` is a zero-override subclass.
inline constexpr ModelInfo kGlmMoeDsaInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class GlmMoeDsaLoadedModel final : public LoadedModel {
 public:
  GlmMoeDsaLoadedModel(const ModelRegistration& registration,
                       GlmMoeDsaWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const GlmMoeDsaWeights& weights() const { return weights_; }

 private:
  GlmMoeDsaWeights weights_;
};

std::unique_ptr<LoadedModel> LoadGlmMoeDsaForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    // The GGUF arm, W7. `ModelSource::FromGguf` is the only producer of a
    // non-safetensors source here, so `source.gguf` is the file the entrypoint
    // opened -- split-aware, so one `GgufFile` sees all six shards.
    VT_CHECK(source.gguf != nullptr,
             "GlmMoeDsaForCausalLM: a non-safetensors ModelSource carried no "
             "GgufFile");
    return std::make_unique<GlmMoeDsaLoadedModel>(
        registration, LoadGlmMoeDsaFromGguf(*source.gguf, config,
                                            /*policy=*/nullptr));
  }
  (void)registration;
  throw std::runtime_error(GlmMoeDsaSafetensorsRefusal());
}

void PrepareGlmMoeDsaForCausalLM(LoadedModel& model, const HfConfig& config,
                                 vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardGlmMoeDsaForCausalLM(LoadedModel& model,
                                          const ModelForwardInput& input) {
  auto& m = ModelAs<GlmMoeDsaLoadedModel>(model, "GlmMoeDsaForCausalLM");
  const GlmMoeDsaWeights& weights = m.weights();
  if (input.gather_logits) {
    return GlmMoeDsaModel::ForwardDevice(input.token_ids, input.positions,
                                         input.attn_meta, input.attn_kv, weights,
                                         input.queue, input.logits_indices);
  }
  return HostLogits(
      GlmMoeDsaModel::Forward(input.token_ids, input.positions, input.attn_meta,
                              input.attn_kv, weights, input.queue,
                              input.logits_indices),
      weights.params.vocab_size);
}

const ModelFactory kGlmMoeDsaFactory{
    .parse_config = &ParseGlmMoeDsaConfig,
    .load_weights = &LoadGlmMoeDsaForCausalLM,
    .prepare = &PrepareGlmMoeDsaForCausalLM,
    .forward = &ForwardGlmMoeDsaForCausalLM,
    .make_kv_cache = &MakeGlmMoeDsaKVCache,
    .is_dense_model = false,
    // W9 (#2214), spec O22. THE FLAG AND THE FORWARD LAND TOGETHER, because the
    // flag is a claim ABOUT the forward: it says this model's routed-expert
    // compute reads through the slot seam, and `model_registry.h` argues at
    // length that it therefore lives beside the forward that implements it.
    // `ForwardGlmMoeDsaForCausalLM` reaches `MoeBlock` -> `ExpertMlp` ->
    // `GlmExpertSlice` -> `expert_stream::ExpertSlice`
    // (`glm_moe_dsa_forward.cpp`), so the claim is true; W7 could not make it,
    // because W7 had no forward.
    //
    // THE COST OF NOT SETTING IT IS EXACT AND IS WHY THIS IS NOT COSMETIC.
    // `model_loader.cpp`'s streamed-lane block is guarded on
    // `factory->streams_routed_experts`, so without this the lane is never
    // built, `CheckDeviceWeightFit` charges the device the full 187.312 GiB of
    // towers against `dgx:gpu0`'s 119.631 GiB budget, and the load REFUSES.
    .streams_routed_experts = true,
};

}  // namespace

REGISTER_VLLM_MODEL(glm_moe_dsa, "GlmMoeDsaForCausalLM", kGlmMoeDsaFactory,
                    kGlmMoeDsaInfo)

}  // namespace vllm
