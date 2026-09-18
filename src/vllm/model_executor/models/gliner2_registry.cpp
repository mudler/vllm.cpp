// GLiNER2.5-multi-v1 registry TU (MODEL-GLINER25 Phase 3b).
//
// Registers "BoundaryExtractor" as a POOLING model (is_pooling_model=true,
// is_text_generation_model=false), the same task-routing shape as the Llama
// embedding precedent (llama_embedding_registry.cpp). The C ABI refuses
// vllm_complete/vllm_chat on this arch and points at vllm_embed; the server
// registers /v1/embeddings instead of generate routes.
//
// The forward runs the DeBERTa v2 encoder host reference (deberta_v2::
// ForwardHost) and returns the hidden states [num_tokens, hidden_size] as
// ForwardLogits.host — the pooling runner applies first-token (CLS) pooling
// per the checkpoint's token_pooling: "first".
//
// The boundary head weights are loaded and stored on the LoadedModel, ready
// for the Phase 4 NER endpoint (jev-compatible /v1/systemon). They are not
// yet called in the forward because the standard ModelForwardInput carries no
// entity-type queries — the boundary head needs a custom entry point.
//
// UPSTREAM MIRROR: GLiNER2 has no vLLM registration (vLLM has no DeBERTa
// support, PRs #42094 and #20215 are unmerged). This is a from-scratch
// port against the HuggingFace transformers + GLiNER2 library references.
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
#include "vllm/model_executor/models/gliner2.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits carrier
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo: a POOLING model with no text-generation path.
inline constexpr ModelInfo kGliner2Info{
    .is_text_generation_model = false,
    .is_pooling_model = true,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .supports_transcription = false,
    .supports_transcription_only = false,
    .score_type = "bi-encoder",
};

// Opaque owned model: encoder + boundary head weights + the model-owned
// DispatchPooler (first-token / CLS pooling per config token_pooling: "first").
class Gliner2LoadedModel final : public LoadedModel {
 public:
  Gliner2LoadedModel(const ModelRegistration& registration,
                     Gliner2ModelWeights weights)
      : LoadedModel(registration),
        weights_(std::move(weights)),
        pooler_(DispatchPooler::ForEmbedding(PoolerConfig{},
                                             SequencePoolingType::kCLS)) {}

  const Gliner2ModelWeights& weights() const { return weights_; }
  const Pooler* pooler() const override { return pooler_.get(); }

 private:
  Gliner2ModelWeights weights_;
  std::unique_ptr<DispatchPooler> pooler_;
};

std::unique_ptr<LoadedModel> LoadGliner2(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture BoundaryExtractor does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Gliner2LoadedModel>(
      registration, LoadGliner2Weights(*source.safetensors, config));
}

void PrepareGliner2(LoadedModel& model, const HfConfig& config,
                    vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardGliner2(LoadedModel& model,
                              const ModelForwardInput& input) {
  auto& m = ModelAs<Gliner2LoadedModel>(model, "BoundaryExtractor");
  const auto& enc = m.weights();

  // Convert int32 token IDs to int64 for the host reference forward.
  std::vector<int64_t> input_ids(input.token_ids.begin(),
                                 input.token_ids.end());

  // DeBERTa encoder forward: embeddings → N layers → hidden [seq, hidden].
  // Bidirectional (no causal mask); positional info from disentangled attention.
  std::vector<float> hidden =
      deberta_v2::ForwardHost(enc.encoder_params, enc.encoder_weights, input_ids);

  // Return hidden states as the pooling carrier: [num_tokens, hidden_size]
  // f32 host rows. The pooling runner applies CLS pooling (first token).
  const int64_t num_tokens = static_cast<int64_t>(input_ids.size());
  const int64_t hidden_size = enc.encoder_params.hidden_size;

  ForwardLogits result;
  result.host = std::move(hidden);
  result.rows = num_tokens;
  result.vocab = hidden_size;
  return result;
}

void ParseGliner2Config(const HfConfig& config) {
  // LoadHfConfig materializes the config. No additional normalization needed
  // — the encoder and boundary head params are parsed from weight shapes and
  // config.raw at load time.
  (void)config;
}

v1::KVCacheConfig MakeGliner2KVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  // PLACEHOLDER, never exercised: the DeBERTa v2 encoder is bidirectional with
  // no KV cache, and the pooling path (vllm_embed / /v1/embeddings) never builds
  // one. One minimal full-attention group so resolution-time plumbing that sizes
  // specs cannot crash on a null factory field (parakeet_registry precedent).
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

const ModelFactory kGliner2Factory{
    .parse_config = &ParseGliner2Config,
    .load_weights = &LoadGliner2,
    .prepare = &PrepareGliner2,
    .forward = &ForwardGliner2,
    .make_kv_cache = &MakeGliner2KVCache,  // placeholder, never exercised
    .is_dense_model = true,
};

}  // namespace

REGISTER_VLLM_MODEL(boundary_extractor, "BoundaryExtractor", kGliner2Factory,
                     kGliner2Info)

}  // namespace vllm
