// CLM (Contrastive-LM) System 1 decision model — head weights, params, and
// host-side forward declarations.
//
// CLM is a bi-encoder: the state text and each candidate option are forwarded
// SEPARATELY through the Qwen3-8B backbone (last-token pooling), then each
// hidden is projected by an MLP head (state head or action head), L2-normalized,
// and scored via scaled cosine similarity.
//
// Ported from Contrastive-LM/CLM-v0.1-8B (spec .agents/specs/clm.md).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {

namespace clm {

// Head parameters (stored in the checkpoint's cfg dict, not config.json).
struct HeadParams {
  int64_t hidden_size = 4096;  // Qwen3-8B hidden size
  int64_t proj_dim = 1536;    // intermediate projection dimension
  int64_t embed_dim = 512;    // output embedding dimension
  float logit_scale = 4.6f;   // learned temperature (clamped to max=100)
};

// Weights for one MLP projection head:
//   Linear(H, P) → GELU → LayerNorm(P) → Linear(P, P) → GELU → Linear(P, E)
// PyTorch nn.Linear weight layout: [out_features, in_features].
struct MlpHeadWeights {
  std::vector<float> w0;  // [P, H]  Linear(4096, 1536)
  std::vector<float> b0;  // [P]
  std::vector<float> w2;  // [P]     LayerNorm(1536) gamma
  std::vector<float> b2;  // [P]     LayerNorm(1536) beta
  std::vector<float> w4;  // [P, P]  Linear(1536, 1536)
  std::vector<float> b4;  // [P]
  std::vector<float> w6;  // [E, P]  Linear(1536, 512)
  std::vector<float> b6;  // [E]
};

// Both heads: state head (query encoder) and action head (candidate encoder).
struct HeadWeights {
  MlpHeadWeights state_head;
  MlpHeadWeights action_head;
};

// Encoded question for bi-encoder forward.
struct ClmEncoded {
  std::vector<int32_t> state_token_ids;
  std::vector<int32_t> state_positions;
  std::vector<std::vector<int32_t>> option_token_ids;
  std::vector<std::vector<int32_t>> option_positions;
};

// Build positions [0, 1, ..., T-1] for a token sequence.
inline std::vector<int32_t> MakePositions(int64_t T) {
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  return pos;
}

// MLP head forward: H → P → P → E, then L2-normalize.
// hidden: [H] f32 host buffer. Returns [E] f32 L2-normalized.
std::vector<float> ClmMlpHeadForward(
    const MlpHeadWeights& hw, const HeadParams& params,
    const std::vector<float>& hidden);

// Scaled cosine scoring.
// h_state:   [E] L2-normalized
// h_options: K × [E] L2-normalized
// Returns K pre-softmax logits: score_k = exp(logit_scale) * dot(h_state, h_option_k)
std::vector<float> ClmScoreOptions(
    const std::vector<float>& h_state,
    const std::vector<std::vector<float>>& h_options,
    float logit_scale);

}  // namespace clm

}  // namespace vllm
