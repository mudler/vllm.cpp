// GLiNER2.5-Decide registry TU (MODEL-GLINER25-DECIDE).
//
// Registers "SpanExtractor" as a pooling model with a decision callback.
// The forward runs the DeBERTa v2 encoder host reference and returns hidden
// states. The decision callback (Gliner25DecideInference) builds the
// [CLS] text [SEP] [P] task [L] label1 [L] label2 ... [SEP] sequence,
// forwards it, extracts label embeddings at [L] positions, and runs the
// classification head (Linear(H,2H) → ReLU → Linear(2H,1)).
//
// The classification head weights are loaded from the safetensors checkpoint
// alongside the DeBERTa encoder weights.
//
// Arch name: "SpanExtractor" (same as MODEL-GLINER25, but this model
// registers a separate factory with the decision head).
#include "vllm/model_executor/models/gliner25_decide.h"
#include "vllm/model_executor/models/gliner25_decide_inference.h"

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deberta_v2.h"
#include "vllm/model_executor/models/gliner2.h"
#include "vllm/model_executor/models/gliner2_ner.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits carrier
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
namespace {

// ── ModelInfo ────────────────────────────────────────────────────────────
inline constexpr ModelInfo kGliner25DecideInfo{
    .is_text_generation_model = false,
    .is_pooling_model = true,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .supports_transcription = false,
    .supports_transcription_only = false,
    .score_type = "bi-encoder",
};

// ── Loaded model ──────────────────────────────────────────────────────────
class Gliner25DecideLoadedModel final : public LoadedModel {
 public:
  Gliner25DecideLoadedModel(
      const ModelRegistration& registration,
      Gliner2ModelWeights model_weights,
      gliner25_decide::DecideHeadWeights head_weights,
      gliner25_decide::DecideHeadParams head_params)
      : LoadedModel(registration),
        model_weights_(std::move(model_weights)),
        head_weights_(std::move(head_weights)),
        head_params_(head_params) {}

  const Gliner2ModelWeights& model_weights() const { return model_weights_; }
  const gliner25_decide::DecideHeadWeights& head_weights() const {
    return head_weights_;
  }
  const gliner25_decide::DecideHeadParams& head_params() const {
    return head_params_;
  }

 private:
  Gliner2ModelWeights model_weights_;
  gliner25_decide::DecideHeadWeights head_weights_;
  gliner25_decide::DecideHeadParams head_params_;
};

// ── Head weight loading ───────────────────────────────────────────────────
gliner25_decide::DecideHeadWeights LoadDecideHead(
    const std::vector<SafetensorsFile>& shards) {
  gliner25_decide::DecideHeadWeights hw;

  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      const StTensor& t = shard.Get(name);
      if (t.dtype != "F32") continue;
      size_t count = t.nbytes / sizeof(float);
      std::vector<float> dst(count);
      std::memcpy(dst.data(), t.data, t.nbytes);

      if (name == "classifier.0.weight") hw.w0 = std::move(dst);
      else if (name == "classifier.0.bias") hw.b0 = std::move(dst);
      else if (name == "classifier.2.weight") hw.w2 = std::move(dst);
      else if (name == "classifier.2.bias") hw.b2 = std::move(dst);
    }
  }

  if (hw.w0.empty() || hw.b0.empty() || hw.w2.empty() || hw.b2.empty()) {
    throw std::runtime_error(
        "gliner25_decide: checkpoint missing classifier.0.weight/bias or "
        "classifier.2.weight/bias tensors");
  }

  return hw;
}

// ── Load ──────────────────────────────────────────────────────────────────
std::unique_ptr<LoadedModel> LoadGliner25Decide(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture SpanExtractor does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }

  // Load DeBERTa encoder weights (same as MODEL-GLINER25).
  Gliner2ModelWeights model_weights =
      LoadGliner2Weights(*source.safetensors, config);

  // Load classification head weights.
  gliner25_decide::DecideHeadWeights head_weights =
      LoadDecideHead(*source.safetensors);

  // Infer head params from encoder.
  gliner25_decide::DecideHeadParams head_params;
  head_params.hidden_size = model_weights.encoder_params.hidden_size;
  head_params.temperature = 1.0F;

  return std::make_unique<Gliner25DecideLoadedModel>(
      registration, std::move(model_weights), std::move(head_weights),
      head_params);
}

// ── Prepare ──────────────────────────────────────────────────────────────
void PrepareGliner25Decide(LoadedModel& model, const HfConfig& config,
                            vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

// ── Forward (factory) ─────────────────────────────────────────────────────
ForwardLogits ForwardGliner25Decide(LoadedModel& model,
                                     const ModelForwardInput& input) {
  auto& m = ModelAs<Gliner25DecideLoadedModel>(model, "SpanExtractor");
  const auto& w = m.model_weights();

  // Convert int32 token IDs to int64 for the host reference forward.
  std::vector<int64_t> input_ids(input.token_ids.begin(),
                                  input.token_ids.end());

  std::vector<float> hidden =
      deberta_v2::ForwardHost(w.encoder_params, w.encoder_weights, input_ids);

  const int64_t num_tokens = static_cast<int64_t>(input_ids.size());
  const int64_t hidden_size = w.encoder_params.hidden_size;

  ForwardLogits result;
  result.host = std::move(hidden);
  result.rows = num_tokens;
  result.vocab = hidden_size;
  return result;
}

void ParseGliner25DecideConfig(const HfConfig& config) {
  (void)config;
}

v1::KVCacheConfig MakeGliner25DecideKVCache(const HfConfig& config,
                                             int block_size, int num_blocks) {
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

const ModelFactory kGliner25DecideFactory{
    .parse_config = &ParseGliner25DecideConfig,
    .load_weights = &LoadGliner25Decide,
    .prepare = &PrepareGliner25Decide,
    .forward = &ForwardGliner25Decide,
    .make_kv_cache = &MakeGliner25DecideKVCache,
    .is_dense_model = true,
};

}  // namespace

// ── Decision inference ────────────────────────────────────────────────────
Gliner25DecideResult Gliner25DecideInference(
    const LoadedModel& model,
    const tok::Tokenizer& tokenizer,
    const std::string& state,
    const std::string& qtype,
    const std::string& instructions,
    const std::vector<std::string>& options) {
  (void)qtype;
  (void)instructions;

  Gliner25DecideResult result;
  if (options.empty()) return result;

  const auto& m = ModelAs<Gliner25DecideLoadedModel>(model, "SpanExtractor");
  const auto& w = m.model_weights();
  const auto& hw = m.head_weights();
  const auto& hp = m.head_params();
  const int64_t H = w.encoder_params.hidden_size;

  // Build the sequence: [CLS] text [SEP] [P] task [L] label1 [L] label2 ... [SEP]
  // For /v1/systemone: the "task" is the question type, and the labels are the options.
  // DeBERTa uses BOS as [CLS] and EOS as [SEP] (DeBERTa-v3 does not have a
  // separate CLS token; the BOS token serves as CLS).
  const int32_t cls_id = tokenizer.BosId();
  const int32_t sep_id = tokenizer.EosId();
  // Use the SEP token as [P]/[L] marker fallback (the encoder treats all
  // tokens the same — the classification head only cares about the hidden
  // state at each label position).
  const int32_t marker_id = sep_id;  // use SEP as [P]/[L] marker fallback

  std::vector<int64_t> full_ids;

  // [CLS] text [SEP]
  if (cls_id >= 0) full_ids.push_back(cls_id);
  auto text_ids = tokenizer.Encode(state);
  for (int32_t id : text_ids) full_ids.push_back(id);
  if (sep_id >= 0) full_ids.push_back(sep_id);

  // [P] task [L] label1 [L] label2 ...
  if (marker_id >= 0) full_ids.push_back(marker_id);  // [P]
  auto task_ids = tokenizer.Encode(qtype.empty() ? std::string("choice") : qtype);
  for (int32_t id : task_ids) full_ids.push_back(id);

  // Track positions of each label's first token (the [L] marker position).
  std::vector<int64_t> label_positions;
  for (const auto& opt : options) {
    if (marker_id >= 0) full_ids.push_back(marker_id);  // [L]
    label_positions.push_back(static_cast<int64_t>(full_ids.size()));
    auto opt_ids = tokenizer.Encode(opt);
    for (int32_t id : opt_ids) full_ids.push_back(id);
  }

  // Trailing [SEP]
  if (sep_id >= 0) full_ids.push_back(sep_id);

  // Forward the DeBERTa v2 encoder.
  std::vector<float> hidden =
      deberta_v2::ForwardHost(w.encoder_params, w.encoder_weights, full_ids);

  result.prompt_tokens = static_cast<int64_t>(full_ids.size());

  // Extract label embeddings at [L] marker positions.
  const int64_t num_labels = static_cast<int64_t>(options.size());
  std::vector<float> label_embs(static_cast<size_t>(num_labels * H));
  for (int64_t n = 0; n < num_labels; ++n) {
    int64_t pos = label_positions[static_cast<size_t>(n)];
    if (pos < static_cast<int64_t>(full_ids.size())) {
      const float* src = &hidden[static_cast<size_t>(pos * H)];
      std::copy_n(src, static_cast<size_t>(H),
                  &label_embs[static_cast<size_t>(n * H)]);
    }
  }

  // Run the classification head.
  result.scores = gliner25_decide::ClassifierForward(hw, hp, label_embs,
                                                       num_labels);

  return result;
}

REGISTER_VLLM_MODEL(gliner25_decide, "SpanExtractor", kGliner25DecideFactory,
                    kGliner25DecideInfo)

}  // namespace vllm
