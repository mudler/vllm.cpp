// Laya registry TU (MODEL-LAYA Phase 4).
//
// Registers "LayaModel" as a POOLING model (is_pooling_model=true,
// is_text_generation_model=false), the same task-routing shape as the
// GLiNER2.5 precedent (gliner2_registry.cpp). The C ABI refuses
// vllm_complete/vllm_chat on this arch; the server registers
// /v1/systemone through the decision callback instead of the NER callback.
//
// The forward runs the ModernBERT encoder host reference (modernbert::
// ForwardHost) and returns the hidden states [num_tokens, hidden_size] as
// ForwardLogits.host — the pooling runner applies first-token (CLS) pooling.
//
// The decision head weights are loaded and stored on the LoadedModel, ready
// for the /v1/systemone endpoint. They are not called in the pooling forward
// because the standard ModelForwardInput carries no question/option data —
// the decision head needs the custom LayaInference entry point.
//
// UPSTREAM MIRROR: Laya has no vLLM registration (vLLM has no ModernBERT
// support and no Laya model). This is a from-scratch port against the
// NandhaKishorM/laya source and the HuggingFace convaiinnovations/laya
// checkpoint.
#include "vllm/model_executor/models/model_registry.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/pooler/dispatch_pooler.h"
#include "vllm/model_executor/layers/pooler/methods.h"  // SequencePoolingType
#include "vllm/model_executor/layers/pooler/pooler_config.h"
#include "vllm/model_executor/models/laya.h"
#include "vllm/model_executor/models/laya_inference.h"
#include "vllm/model_executor/models/laya_sequence.h"
#include "vllm/model_executor/models/modernbert.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits carrier
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo: a POOLING model with no text-generation path.
inline constexpr ModelInfo kLayaInfo{
    .is_text_generation_model = false,
    .is_pooling_model = true,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .supports_transcription = false,
    .supports_transcription_only = false,
    .score_type = "bi-encoder",
};

// Opaque owned model: ModernBERT encoder + Laya decision head weights + the
// model-owned DispatchPooler (first-token / CLS pooling).
class LayaLoadedModel final : public LoadedModel {
 public:
  LayaLoadedModel(const ModelRegistration& registration,
                  LayaModelWeights weights)
      : LoadedModel(registration),
        weights_(std::move(weights)),
        pooler_(DispatchPooler::ForEmbedding(PoolerConfig{},
                                             SequencePoolingType::kCLS)) {}

  const LayaModelWeights& weights() const { return weights_; }
  const Pooler* pooler() const override { return pooler_.get(); }

 private:
  LayaModelWeights weights_;
  std::unique_ptr<DispatchPooler> pooler_;
};

std::unique_ptr<LoadedModel> LoadLaya(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture LayaModel does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<LayaLoadedModel>(
      registration, LoadLayaWeights(*source.safetensors, config));
}

void PrepareLaya(LoadedModel& model, const HfConfig& config,
                  vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardLaya(LoadedModel& model,
                            const ModelForwardInput& input) {
  auto& m = ModelAs<LayaLoadedModel>(model, "LayaModel");
  const auto& w = m.weights();

  // Convert int32 token IDs to int64 for the host reference forward.
  std::vector<int64_t> input_ids(input.token_ids.begin(),
                                 input.token_ids.end());

  // ModernBERT encoder forward: embeddings → N layers → hidden [seq, hidden].
  std::vector<float> hidden =
      modernbert::ForwardHost(w.encoder_params, w.encoder_weights, input_ids);

  // Return hidden states as the pooling carrier: [num_tokens, hidden_size]
  // f32 host rows. The pooling runner applies CLS pooling (first token).
  const int64_t num_tokens = static_cast<int64_t>(input_ids.size());
  const int64_t hidden_size = w.encoder_params.hidden_size;

  ForwardLogits result;
  result.host = std::move(hidden);
  result.rows = num_tokens;
  result.vocab = hidden_size;
  return result;
}

void ParseLayaConfig(const HfConfig& config) {
  // LoadHfConfig materializes the config. No additional normalization needed
  // — the encoder and head params are parsed from weight shapes and
  // config.raw at load time.
  (void)config;
}

v1::KVCacheConfig MakeLayaKVCache(const HfConfig& config, int block_size,
                                    int num_blocks) {
  // PLACEHOLDER, never exercised: the ModernBERT encoder is bidirectional with
  // no KV cache, and the pooling path never builds one. One minimal
  // full-attention group so resolution-time plumbing cannot crash on a null
  // factory field (gliner2_registry precedent).
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

const ModelFactory kLayaFactory{
    .parse_config = &ParseLayaConfig,
    .load_weights = &LoadLaya,
    .prepare = &PrepareLaya,
    .forward = &ForwardLaya,
    .make_kv_cache = &MakeLayaKVCache,  // placeholder, never exercised
    .is_dense_model = true,
};

// ── Special token resolution ────────────────────────────────────────────
// Find CLS/SEP/PAD/MASK from the tokenizer's added tokens. ModernBERT-large
// uses [CLS]=50281, [SEP]=50282, [PAD]=50283, [MASK]=50284.
laya::SpecialTokens ResolveSpecialTokens(const tok::Tokenizer& tokenizer) {
  laya::SpecialTokens st;
  for (const auto& at : tokenizer.AddedTokens()) {
    if (at.text == "[CLS]" || at.text == "<cls>") st.cls_id = at.id;
    else if (at.text == "[SEP]" || at.text == "<sep>") st.sep_id = at.id;
    else if (at.text == "[PAD]" || at.text == "<pad>") st.pad_id = at.id;
    else if (at.text == "[MASK]" || at.text == "<mask>") {
      st.mask_id = at.id;
      st.mask_str = at.text;
    }
  }
  // Fallback: if SEP is not in added tokens but EOS is set, use EOS as SEP.
  if (st.sep_id == 50282 && tokenizer.EosId() >= 0) {
    st.sep_id = tokenizer.EosId();
  }
  return st;
}

// Convert qtype string to enum.
laya::QType QTypeFromString(const std::string& qtype) {
  if (qtype == "choice") return laya::QType::kChoice;
  if (qtype == "score") return laya::QType::kScore;
  return laya::QType::kNoul;
}

}  // namespace

// ── Phase 4: Decision inference ─────────────────────────────────────────
// Full decision pipeline: build the input sequence, run the ModernBERT
// encoder, add the type embedding, run the Laya decision head (head layers +
// scorer + act_head), and return per-option scores + escalate logits.
LayaDecisionResult LayaInference(
    const LoadedModel& model,
    const tok::Tokenizer& tokenizer,
    const std::string& state,
    const std::string& qtype,
    const std::string& instructions,
    const std::vector<std::string>& options,
    int max_len,
    int head_max_len) {
  LayaDecisionResult result;
  if (options.empty()) return result;

  const auto& m = ModelAs<LayaLoadedModel>(model, "LayaModel");
  const auto& w = m.weights();

  // 1. Resolve special tokens from the tokenizer.
  laya::SpecialTokens special = ResolveSpecialTokens(tokenizer);

  // 2. Build the input sequence.
  laya::Question question;
  question.type = QTypeFromString(qtype);
  question.instructions = instructions;
  question.options = options;
  question.state = state;

  laya::SequenceOutput seq = laya::BuildSequence(
      tokenizer, special, question, max_len, head_max_len);

  if (seq.ids.empty() || seq.markers.empty()) return result;

  // 3. Run the ModernBERT encoder forward.
  std::vector<int64_t> input_ids(seq.ids.begin(), seq.ids.end());
  std::vector<float> hidden =
      modernbert::ForwardHost(w.encoder_params, w.encoder_weights, input_ids);

  const int64_t seq_len = static_cast<int64_t>(input_ids.size());
  const int64_t qt = static_cast<int64_t>(question.type);

  // 4. Build attention_mask (all 1s — no padding in a single sequence).
  //    type_emb is added inside ForwardHost (laya.cpp), mirroring upstream
  //    DecisionModel.forward. Do NOT add it here — that double-adds it.
  std::vector<int64_t> attention_mask(seq_len, 1);

  // 5. Build marker_pos and marker_mask from the SequenceOutput.
  std::vector<int64_t> marker_pos;
  std::vector<int64_t> marker_mask;
  marker_pos.reserve(seq.markers.size());
  marker_mask.reserve(seq.markers.size());
  for (int32_t pos : seq.markers) {
    marker_pos.push_back(static_cast<int64_t>(pos));
    marker_mask.push_back(1);
  }

  // 6. Run the Laya decision head forward.
  laya::ForwardOutput head_out = laya::ForwardHost(
      w.head_params, w.head_weights, hidden, attention_mask,
      marker_pos, marker_mask, qt);

  result.scores = std::move(head_out.logits);
  result.act_logits = std::move(head_out.act_logits);
  result.prompt_tokens = seq_len;

  // Apply temperature scaling before softmax (matches reference:
  //   z = logits[:k] / temperature_by_options.get(temp_bucket(qt, k),
  //                                                 temperature[qt])
  // Without this the probability distributions are far too peaked.
  {
    const int k = static_cast<int>(result.scores.size());
    const int qt_i = static_cast<int>(qt);
    const float temp = laya::TemperatureFor(
        qt_i, k, w.head_weights.temperature, w.temperature_by_options);
    for (float& s : result.scores) s /= temp;
  }

  return result;
}

REGISTER_VLLM_MODEL(laya_model, "LayaModel", kLayaFactory, kLayaInfo)

}  // namespace vllm
