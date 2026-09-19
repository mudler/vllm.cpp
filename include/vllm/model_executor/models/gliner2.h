// GLiNER2 boundary head — the NER head of GLiNER2.5-multi-v1 (MODEL-GLINER25).
//
// Ported from the GLiNER2 library (github.com/fastino-ai/GLiNER2):
//   boundary_head/boundary_encoder.py   BoundaryEncoder
//   boundary_head/heads.py              BoundaryQueryHead
//   boundary_head/constants.py           MASK_LOGIT
//
// This is the HOST REFERENCE forward: a portable f32 implementation gated
// against a Python restatement of the upstream math, the same pattern the
// DeBERTa v2 encoder (deberta_v2.{h,cpp}) uses. It is not wired to the runner,
// the ABI or the server; that is Phase 4 of .agents/specs/gliner2.5.md.
//
// The boundary architecture replaces the older span architecture. Instead of
// enumerating all (start, end) pairs, it predicts per-token boundary marginals
// (start, end, inside) and then proposes a sparse set of candidates from those
// marginals. The forward is:
//
//   text_states = DeBERTa_v2::ForwardHost(...)          [L, H]
//   boundary_states = BoundaryEncoder(text_states)        [L+1, d]
//   marginals = BoundaryQueryHead(boundary_states,
//                                 query_states)           start[L+1], end[L+1], inside[L]
//
// For the host reference test only the encoder and query head are exercised.
// The proposer, pair scorer, and classifier follow in the next slice.
#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vllm/model_executor/models/deberta_v2.h"

namespace vllm {
namespace gliner2 {

// Boundary architecture config (subset of boundary_head from config.json).
struct BoundaryParams {
  int64_t hidden_size = 768;        // encoder hidden (DeBERTa H)
  int64_t boundary_dim = 128;        // d
  int64_t boundary_attention_heads = 4;
  int64_t boundary_attention_layers = 2;
  int64_t boundary_attention_window = 128;
  int64_t boundary_refinement_layers = 1;
  double boundary_ffn_multiplier = 2.0;
  double layer_norm_eps = 1e-5;      // PyTorch default for the boundary head
  double query_scale = 0.0;          // 0 => 1/sqrt(boundary_dim)

  int64_t head_dim() const { return boundary_dim / boundary_attention_heads; }
  int64_t ffn_dim() const {
    return static_cast<int64_t>(boundary_dim * boundary_ffn_multiplier);
  }
  double scale() const {
    return query_scale > 0 ? query_scale
                           : 1.0 / std::sqrt(static_cast<double>(boundary_dim));
  }
};

// ── BoundaryEncoder weights ──────────────────────────────────────────────
struct BoundaryAttentionBlockWeights {
  std::vector<float> norm_weight, norm_bias;           // [d]
  std::vector<float> qkv_weight, qkv_bias;              // [3d, d], [3d]
  std::vector<float> output_weight, output_bias;        // [d, d], [d]
};

struct RefinementBlockWeights {
  std::vector<float> norm_weight, norm_bias;           // [d]
  std::vector<float> input_weight, input_bias;          // [ffn, d], [ffn]
  std::vector<float> output_weight, output_bias;        // [d, ffn/2], [d]
};

struct BoundaryEncoderWeights {
  std::vector<float> left_proj_weight, left_proj_bias;     // [d, H], [d]
  std::vector<float> right_proj_weight, right_proj_bias;    // [d, H], [d]
  std::vector<float> output_proj_weight, output_proj_bias;  // [d, 2d], [d]
  std::vector<float> layer_norm_weight, layer_norm_bias;    // [d], [d]
  std::vector<float> bos_state, eos_state;                  // [H], [H]
  std::vector<BoundaryAttentionBlockWeights> attention_blocks;
  std::vector<RefinementBlockWeights> refinement_blocks;
};

// ── BoundaryQueryHead weights ────────────────────────────────────────────
struct BoundaryQueryHeadWeights {
  std::vector<float> start_boundary_weight, start_boundary_bias;   // [d, d], [d]
  std::vector<float> end_boundary_weight, end_boundary_bias;        // [d, d], [d]
  std::vector<float> start_query_weight, start_query_bias;          // [d, H], [d]
  std::vector<float> end_query_weight, end_query_bias;               // [d, H], [d]
  std::vector<float> inside_text_weight, inside_text_bias;           // [d, H], [d]
  std::vector<float> inside_query_weight, inside_query_bias;         // [d, H], [d]
};

// All boundary head weights.
struct BoundaryHeadWeights {
  BoundaryEncoderWeights encoder;
  BoundaryQueryHeadWeights query_head;
};

// Output marginals from the query head.
struct BoundaryMarginals {
  std::vector<float> start_logits;   // [Q, L+1]
  std::vector<float> end_logits;      // [Q, L+1]
  std::vector<float> inside_logits;   // [Q, L]
  std::vector<float> inside_prefix;   // [Q, L+1]  (cumsum of inside, fp32)
};

// Load boundary head weights from a checkpoint by name.
BoundaryHeadWeights LoadBoundaryHead(const BoundaryParams& params,
                                     const deberta_v2::CheckpointTensors& tensors);

// BoundaryEncoder forward: text_states [L, H] → boundary_states [L+1, d].
std::vector<float> BoundaryEncoderForward(const BoundaryParams& params,
                                           const BoundaryEncoderWeights& w,
                                           const std::vector<float>& text_states,
                                           int64_t seq_len);

// BoundaryQueryHead forward: boundary_states [L+1, d] + query_states [Q, H]
// → marginals.
BoundaryMarginals BoundaryQueryHeadForward(
    const BoundaryParams& params, const BoundaryQueryHeadWeights& w,
    const std::vector<float>& boundary_states, int64_t seq_len,
    const std::vector<float>& text_states,
    const std::vector<float>& query_states, int64_t num_queries);

}  // namespace gliner2

// ── Production model (Phase 3b: registration) ───────────────────────────
// Forward declarations to keep this header free of the safetensors and
// config includes.
class SafetensorsFile;
struct HfConfig;

// Combined model weights: the DeBERTa v2 encoder + the boundary head, with
// their parsed configs. Materialized from a safetensors checkpoint by
// LoadGliner2Weights and owned by Gliner2LoadedModel.
struct Gliner2ModelWeights {
  deberta_v2::Params encoder_params;
  deberta_v2::Weights encoder_weights;
  gliner2::BoundaryParams boundary_params;
  gliner2::BoundaryHeadWeights boundary_head;
};

// Production weight loader: reads all F32 tensors from the safetensors
// shards, infers the DeBERTa encoder config from weight shapes (falling
// back to mdeberta-v3-base defaults for fields the checkpoint does not
// carry), reads the boundary head config from config.raw["boundary_head"],
// and calls deberta_v2::Load + gliner2::LoadBoundaryHead.
Gliner2ModelWeights LoadGliner2Weights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

}  // namespace vllm
