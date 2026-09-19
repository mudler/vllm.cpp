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

#include <algorithm>
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
#include "vllm/model_executor/models/gliner2_ner.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits carrier
#include "vllm/tokenizer/tokenizer.h"
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

// ── Phase 4: NER inference ──────────────────────────────────────────────
// Full NER pipeline reachable from the C ABI (vllm_gliner_ner) and the server
// endpoint. Tokenizes text + entity labels, runs the DeBERTa v2 encoder, runs
// the boundary head (encoder + query head), and decodes marginals into entities.
//
// The query state for each label is the mean of the label's token hidden
// states — a Phase 4 approximation. Upstream GLiNER2 uses the hidden state at
// the label's representation position (the [SEP] after the label tokens);
// matching that exactly is an owed correctness refinement.
Gliner2NerResult Gliner2NerInference(
    const LoadedModel& model,
    const tok::Tokenizer& tokenizer,
    std::string_view text_sv,
    const std::vector<std::string>& labels,
    const gliner2::NerParams& params) {
  Gliner2NerResult result;
  if (text_sv.empty() || labels.empty()) return result;

  const auto& m = ModelAs<Gliner2LoadedModel>(model, "BoundaryExtractor");
  const auto& w = m.weights();
  const std::string text(text_sv);

  // 1. Tokenize text (no special tokens — we build the full sequence ourselves).
  std::vector<int32_t> text_ids = tokenizer.Encode(text);
  const int64_t L = static_cast<int64_t>(text_ids.size());
  if (L == 0) return result;

  // 2. Compute token-to-character offset mapping by greedy matching decoded
  //    tokens against the original text. The tokenizer has no offset API, so we
  //    decode each token and find it in the text. This is approximate for
  //    SentencePiece's ▁→space mapping but sufficient for entity extraction.
  std::vector<int64_t> start_mappings(L), end_mappings(L);
  size_t search_pos = 0;
  for (int64_t i = 0; i < L; ++i) {
    std::string decoded = tokenizer.Decode({text_ids[i]});
    if (decoded.empty()) {
      start_mappings[i] = static_cast<int64_t>(search_pos);
      end_mappings[i] = static_cast<int64_t>(search_pos);
      continue;
    }
    size_t found = text.find(decoded, search_pos);
    if (found == std::string::npos) {
      // Try trimming leading whitespace (SentencePiece Strip(1 leading space)).
      std::string trimmed = decoded;
      while (!trimmed.empty() &&
             (trimmed.front() == ' ' || trimmed.front() == '\t')) {
        trimmed.erase(0, 1);
      }
      if (!trimmed.empty()) {
        found = text.find(trimmed, search_pos);
        if (found != std::string::npos) {
          decoded = trimmed;
        }
      }
    }
    if (found == std::string::npos) {
      start_mappings[i] = static_cast<int64_t>(search_pos);
      end_mappings[i] = static_cast<int64_t>(search_pos);
    } else {
      start_mappings[i] = static_cast<int64_t>(found);
      end_mappings[i] = static_cast<int64_t>(found + decoded.size());
      search_pos = found + decoded.size();
    }
  }

  // 3. Tokenize each entity label (no special tokens).
  std::vector<std::vector<int32_t>> label_ids;
  label_ids.reserve(labels.size());
  for (const auto& label : labels) {
    label_ids.push_back(tokenizer.Encode(label));
  }

  // 4. Build the full sequence: [BOS] text_tokens [EOS] label1 [EOS] label2 [EOS] ...
  const int32_t bos = tokenizer.BosId();
  const int32_t eos = tokenizer.EosId();
  std::vector<int64_t> full_ids;
  if (bos >= 0) full_ids.push_back(bos);
  const size_t text_offset = full_ids.size();
  for (int32_t id : text_ids) full_ids.push_back(id);
  if (eos >= 0) full_ids.push_back(eos);

  // Track each label's token positions in the full sequence.
  struct LabelRange { size_t start; size_t end; };
  std::vector<LabelRange> label_ranges;
  label_ranges.reserve(labels.size());
  for (const auto& ids : label_ids) {
    size_t start = full_ids.size();
    for (int32_t id : ids) full_ids.push_back(id);
    label_ranges.push_back({start, full_ids.size()});
    if (eos >= 0) full_ids.push_back(eos);
  }

  // 5. Run the DeBERTa v2 encoder forward on the full sequence.
  std::vector<float> hidden =
      deberta_v2::ForwardHost(w.encoder_params, w.encoder_weights, full_ids);
  const int64_t H = w.encoder_params.hidden_size;

  // 6. Extract text_states [L, H] from the text token positions.
  std::vector<float> text_states(static_cast<size_t>(L * H));
  for (int64_t i = 0; i < L; ++i) {
    const float* src = &hidden[static_cast<size_t>((text_offset + i) * H)];
    std::copy(src, src + H, &text_states[static_cast<size_t>(i * H)]);
  }

  // 7. Extract query_states [Q, H] — mean of each label's token hidden states.
  const int64_t Q = static_cast<int64_t>(labels.size());
  std::vector<float> query_states(static_cast<size_t>(Q * H), 0.0f);
  for (int64_t q = 0; q < Q; ++q) {
    const size_t start = label_ranges[q].start;
    const size_t end = label_ranges[q].end;
    const int64_t n = static_cast<int64_t>(end - start);
    if (n > 0) {
      const float inv = 1.0f / static_cast<float>(n);
      for (int64_t h = 0; h < H; ++h) {
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
          sum += hidden[static_cast<size_t>((start + i) * H + h)];
        }
        query_states[static_cast<size_t>(q * H + h)] = sum * inv;
      }
    }
  }

  // 8. Run the boundary encoder: text_states [L, H] → boundary_states [L+1, d].
  std::vector<float> boundary_states = gliner2::BoundaryEncoderForward(
      w.boundary_params, w.boundary_head.encoder, text_states, L);

  // 9. Run the boundary query head: boundary_states + text_states + query_states → marginals.
  gliner2::BoundaryMarginals marginals = gliner2::BoundaryQueryHeadForward(
      w.boundary_params, w.boundary_head.query_head,
      boundary_states, L, text_states, query_states, Q);

  // 10. Decode marginals into entities.
  result.entities = gliner2::DecodeNer(
      marginals, labels, L, start_mappings, end_mappings, text, params);
  return result;
}

REGISTER_VLLM_MODEL(boundary_extractor, "BoundaryExtractor", kGliner2Factory,
                     kGliner2Info)

}  // namespace vllm
