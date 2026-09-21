// cua-s1-forms registry TU (MODEL-CUA-S1-FORMS Phase 3).
//
// Registers "CuaS1Forms" as a POOLING model (is_pooling_model=true,
// is_text_generation_model=false), the same task-routing shape as the
// GLiNER2.5 and Laya precedents. The C ABI refuses vllm_complete/vllm_chat
// on this arch; the server registers /v1/score through the score callback.
//
// The forward function throws: cua-s1 has no standard encoder-only forward
// path. The model's ForwardHost takes context_ids + option_ids, not the
// standard ModelForwardInput with just token_ids. The actual work is done
// through CuaS1ScoreInference, called from the server's /v1/score handler.
//
// UPSTREAM MIRROR: cua-s1-forms has no vLLM registration (vLLM has no
// TinyTransformerScorer support). This is a from-scratch port against the
// trycua/cua source and the cua-ai/cua-s1-forms checkpoint.
#include "vllm/model_executor/models/model_registry.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/pooler/dispatch_pooler.h"
#include "vllm/model_executor/layers/pooler/methods.h"  // SequencePoolingType
#include "vllm/model_executor/layers/pooler/pooler_config.h"
#include "vllm/model_executor/models/cua_s1.h"
#include "vllm/model_executor/models/cua_s1_inference.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits carrier
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo: a POOLING model with no text-generation path.
inline constexpr ModelInfo kCuaS1Info{
    .is_text_generation_model = false,
    .is_pooling_model = true,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .supports_transcription = false,
    .supports_transcription_only = false,
    .score_type = "bi-encoder",
};

// Opaque owned model: cua-s1 params + weights + the model-owned DispatchPooler
// (first-token / CLS pooling — required by the pooling registration but never
// exercised; the /v1/score handler calls CuaS1ScoreInference directly).
class CuaS1LoadedModel final : public LoadedModel {
 public:
  CuaS1LoadedModel(const ModelRegistration& registration,
                   cua_s1::Params params, cua_s1::Weights weights)
      : LoadedModel(registration),
        params_(std::move(params)),
        weights_(std::move(weights)),
        pooler_(DispatchPooler::ForEmbedding(PoolerConfig{},
                                             SequencePoolingType::kCLS)) {}

  const cua_s1::Params& params() const { return params_; }
  const cua_s1::Weights& weights() const { return weights_; }
  const Pooler* pooler() const override { return pooler_.get(); }

 private:
  cua_s1::Params params_;
  cua_s1::Weights weights_;
  std::unique_ptr<DispatchPooler> pooler_;
};

std::unique_ptr<LoadedModel> LoadCuaS1(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture CuaS1Forms does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  CuaS1ModelWeights mw = LoadCuaS1Weights(*source.safetensors, config);
  return std::make_unique<CuaS1LoadedModel>(
      registration, std::move(mw.params), std::move(mw.weights));
}

void PrepareCuaS1(LoadedModel& model, const HfConfig& config,
                  vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardCuaS1(LoadedModel& model,
                             const ModelForwardInput& input) {
  (void)model;
  (void)input;
  // cua-s1 has no standard encoder-only forward. The model needs both context
  // and option inputs (ForwardHost takes context_ids + option_ids). The actual
  // inference is done through CuaS1ScoreInference, called from the /v1/score
  // server handler. This forward is never called in production.
  throw std::runtime_error(
      "CuaS1Forms has no standard forward path; use /v1/score");
}

void ParseCuaS1Config(const HfConfig& config) {
  (void)config;
}

v1::KVCacheConfig MakeCuaS1KVCache(const HfConfig& config, int block_size,
                                     int num_blocks) {
  // PLACEHOLDER, never exercised: cua-s1 is a scoring model with no KV cache,
  // and the /v1/score path never builds one. One minimal full-attention group
  // so resolution-time plumbing cannot crash on a null factory field
  // (gliner2_registry precedent).
  (void)config;
  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"encoder"},
      std::make_shared<v1::FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                              /*head_size=*/64,
                                              v1::ResolveKvCacheDType()));
  return kv;
}

const ModelFactory kCuaS1Factory{
    .parse_config = &ParseCuaS1Config,
    .load_weights = &LoadCuaS1,
    .prepare = &PrepareCuaS1,
    .forward = &ForwardCuaS1,
    .make_kv_cache = &MakeCuaS1KVCache,  // placeholder, never exercised
    .is_dense_model = true,
};

}  // namespace

// ── Score inference ──────────────────────────────────────────────────────
// Full scoring pipeline: context + options in, probabilities + winner out.
// Delegates to cua_s1::CuaS1Inference (ByteCollate + ForwardHost + softmax).
cua_s1::CuaS1ScoreResult CuaS1ScoreInference(
    const LoadedModel& model,
    const std::string& context,
    const std::vector<std::string>& options) {
  const auto& m = ModelAs<CuaS1LoadedModel>(model, "CuaS1Forms");
  return cua_s1::CuaS1Inference(m.params(), m.weights(), context, options);
}

REGISTER_VLLM_MODEL(cua_s1_forms, "CuaS1Forms", kCuaS1Factory, kCuaS1Info)

}  // namespace vllm
