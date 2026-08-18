// `LlamaModel` EMBEDDING registry TU — ARCH-ONE-SURFACE fold ROW 6, the first
// live pooling architecture. ADDITIVE self-registration (new TU + one
// REGISTER_VLLM_MODEL line, zero shared-array edits), the parakeet_registry /
// llama_registry precedent.
//
// UPSTREAM MIRROR, exactly: the pinned vLLM's `_EMBEDDING_MODELS` maps
//   "LlamaModel": ("llama", "LlamaForCausalLM")
// (vllm/model_executor/models/registry.py:230) and, because LlamaForCausalLM is
// not itself a pooling model, `--runner pooling` resolves `--convert embed`
// (vllm/config/model.py:1058-1060) and wraps the class with `as_embedding_model`
// (adapters.py:230-261):
//   * the output layer (lm_head / logits processor) is replaced by a
//     missing-layer stage (adapters.py:135-151) — the model FORWARD returns
//     hidden states, not logits;
//   * `self.pooler = DispatchPooler.for_embedding(pooler_config)`
//     (adapters.py:248-257), LAST sequence pooling by default for a
//     decoder-only conversion (interfaces_base.py:160
//     `default_seq_pooling_type: ClassVar = "LAST"`);
//   * checkpoint weights load from BOTH the `*ForCausalLM` and bare `*Model`
//     name layouts (adapters.py:178-181 candidate_prefixes ["", "model."]).
// The registered forward here is therefore the SHARED dense backbone
// (LlamaModel == Qwen3DenseModel, llama.h:39-40) run to the post-final-norm
// hidden (Qwen3DenseModel::ForwardHidden) — no new model was built.
//
// TASK ROUTING (the #121 refuse-by-task precedent, both directions):
// info.is_pooling_model=true + is_text_generation_model=false is the registry
// truth the entrypoints dispatch on — the C ABI refuses vllm_complete/vllm_chat
// on this arch (pointing at vllm_embed / /v1/embeddings) and refuses vllm_embed
// on a text arch; the server registers /v1/embeddings INSTEAD OF the generate
// routes. The engine step routes this model's batches through the landed
// PoolingRunner instead of the sampler (runner.cpp pooling branch, the mirror
// of gpu/model_runner.py:368-369 + 1586-1607).
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "vllm/model_executor/layers/pooler/dispatch_pooler.h"
#include "vllm/model_executor/layers/pooler/pooler_config.h"
#include "vllm/model_executor/models/llama.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits carrier

namespace vllm {
namespace {

// registry.py _ModelInfo for the embedding conversion: a POOLING model with NO
// text-generation path (the wrapped class serves task "embed" only —
// pooling_runner.py:22-27 admits exactly ["embed"]).
inline constexpr ModelInfo kLlamaEmbeddingInfo{
    .is_text_generation_model = false,
    .is_pooling_model = true,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .supports_transcription = false,
    .supports_transcription_only = false,
    .score_type = "bi-encoder",
};

// Opaque owned model: the shared dense weight container + the model-owned
// DispatchPooler (the VllmModelForPooling.pooler mirror the PoolingRunner is
// built over, adapters.py:248-257).
class LlamaEmbeddingLoadedModel final : public LoadedModel {
 public:
  LlamaEmbeddingLoadedModel(const ModelRegistration& registration,
                            LlamaWeights weights)
      : LoadedModel(registration),
        weights_(std::move(weights)),
        pooler_(DispatchPooler::ForEmbedding(PoolerConfig{},
                                             SequencePoolingType::kLast)) {}

  const LlamaWeights& weights() const { return weights_; }
  const Pooler* pooler() const override { return pooler_.get(); }

 private:
  LlamaWeights weights_;
  std::unique_ptr<DispatchPooler> pooler_;
};

std::unique_ptr<LoadedModel> LoadLlamaModelEmbedding(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture LlamaModel does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<LlamaEmbeddingLoadedModel>(
      registration, LoadLlamaModelEmbeddingWeights(*source.safetensors, config));
}

void PrepareLlamaModelEmbedding(LoadedModel& model, const HfConfig& config,
                                vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardLlamaModelEmbedding(LoadedModel& model,
                                         const ModelForwardInput& input) {
  // The POOLING forward: shared dense backbone to the post-final-norm hidden,
  // NO lm_head (the as_embedding_model missing-layer stage). The returned
  // carrier holds [n_out, hidden_size] f32 host rows; the runner's pooling
  // branch (never the sampler) consumes them. logits_indices gathers the
  // per-request last-token rows exactly as the text path would — which for
  // LAST pooling IS upstream's `hidden_states[input_batch.logits_indices]`
  // (pooling_runner.py:36).
  // (The runner passes empty logits_indices when the gather toggle is off; the
  // pooling branch then host-gathers, mirroring the text host path.)
  auto& emb = ModelAs<LlamaEmbeddingLoadedModel>(model, "LlamaModel");
  return LlamaModel::ForwardHidden(input.token_ids, input.positions,
                                   input.attn_meta, input.attn_kv,
                                   emb.weights(), input.config, input.queue,
                                   input.logits_indices);
}

const ModelFactory kLlamaEmbeddingFactory{
    .parse_config = &ParseLlamaForCausalLMConfig,
    .load_weights = &LoadLlamaModelEmbedding,
    .prepare = &PrepareLlamaModelEmbedding,
    .forward = &ForwardLlamaModelEmbedding,
    .make_kv_cache = &MakeLlamaForCausalLMKVCache,
    .is_dense_model = true,
};

}  // namespace

REGISTER_VLLM_MODEL(llama_model_embedding, "LlamaModel", kLlamaEmbeddingFactory,
                    kLlamaEmbeddingInfo)

}  // namespace vllm
