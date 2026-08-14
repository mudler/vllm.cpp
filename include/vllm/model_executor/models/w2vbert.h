// w2v-bert-2.0 Conformer pieces — IndexTTS-2.5's semantic front end (W3, #634).
//
// `infer_v2_5.py:174` runs HuggingFace `Wav2Vec2BertModel` over the 16 kHz
// reference clip; its features are what EnhancedCodec quantizes. The encoder is
// a CONFORMER, not a plain transformer: macaron feed-forwards wrap the attention,
// and a depthwise convolution module sits between them.
//
// Gated against `transformers` executed directly — the class IndexTTS itself
// instantiates. Under AGENTS.md that is an admissible secondary oracle: a model's
// own reference implementation, which vLLM mirrors rather than replaces.
//
// Layout is [T, hidden] (row-major over time), matching the torch tensor.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace w2vbert {

// LayerNorm over the last dimension.
std::vector<float> LayerNorm(const std::vector<float>& x, int64_t frames, int64_t dim,
                             const std::vector<float>& gamma, const std::vector<float>& beta,
                             double eps);

// SiLU / swish, the config's `hidden_act`.
double Swish(double x);

// Wav2Vec2BertFeedForward: intermediate_dense -> swish -> output_dense.
//
// THE MACARON HALF-STEP lives in the LAYER, not here: the caller adds
// `ffn(x) * 0.5 + residual`. Dropping the 0.5 still runs and still trains; it is
// simply a different architecture, so the factor is gated by the layer case.
std::vector<float> FeedForward(const std::vector<float>& x, int64_t frames, int64_t hidden,
                               int64_t intermediate, const std::vector<float>& in_w,
                               const std::vector<float>& in_b, const std::vector<float>& out_w,
                               const std::vector<float>& out_b);

struct ConvModuleWeights {
  std::vector<float> ln_gamma, ln_beta;              // pre layer_norm
  std::vector<float> pointwise1;                     // [2*hidden, hidden, 1]
  std::vector<float> depthwise;                      // [hidden, 1, k]
  std::vector<float> dw_ln_gamma, dw_ln_beta;        // after the depthwise conv
  std::vector<float> pointwise2;                     // [hidden, hidden, 1]
};

// Wav2Vec2BertConvolutionModule: layer_norm -> pointwise1 -> GLU -> CAUSAL pad ->
// depthwise -> layer_norm -> swish -> pointwise2.
//
// THE PAD IS LEFT-ONLY, `(kernel - 1, 0)`. A symmetric pad still produces a
// correctly shaped output while letting each frame see the future, which is
// exactly the defect no shape check and no length check can find.
std::vector<float> ConvModule(const std::vector<float>& x, int64_t frames, int64_t hidden,
                              int64_t kernel, const ConvModuleWeights& weights, double eps);


struct SelfAttentionWeights {
  std::vector<float> q_w, q_b, k_w, k_b, v_w, v_b;   // [H, H], [H]
  std::vector<float> out_w, out_b;
  std::vector<float> distance_embedding;             // [left+right+1, head_size]
};

// Wav2Vec2BertSelfAttention with position_embeddings_type == "relative_key".
//
//   scores = q k^T / sqrt(d)
//   distance = clamp(pos_key - pos_query, -left_max, +right_max)
//   scores += einsum("hld,lrd->hlr", q, embedding[distance + left_max]) / sqrt(d)
//
// THE CLAMP IS ASYMMETRIC (64 left, 8 right by default): a symmetric clamp still
// produces well-formed attention, and only diverges for key positions further
// ahead than right_max, which short fixtures never reach. The goldens use T=12 so
// the RIGHT clamp genuinely bites.
//
// Note the relative term is divided by sqrt(d) SEPARATELY, after the scores
// already were -- not folded into one division.
std::vector<float> SelfAttentionRelativeKey(const std::vector<float>& x, int64_t frames,
                                            int64_t hidden, int64_t heads, int64_t left_max,
                                            int64_t right_max, const SelfAttentionWeights& weights);


struct EncoderLayerWeights {
  std::vector<float> ffn1_ln_gamma, ffn1_ln_beta;
  std::vector<float> ffn1_in_w, ffn1_in_b, ffn1_out_w, ffn1_out_b;
  std::vector<float> attn_ln_gamma, attn_ln_beta;
  SelfAttentionWeights attn;
  ConvModuleWeights conv;
  std::vector<float> ffn2_ln_gamma, ffn2_ln_beta;
  std::vector<float> ffn2_in_w, ffn2_in_b, ffn2_out_w, ffn2_out_b;
  std::vector<float> final_ln_gamma, final_ln_beta;
};

// Wav2Vec2BertEncoderLayer: the CONFORMER block.
//
//   x = ffn1(ln(x)) * 0.5 + x        <- macaron HALF step
//   x = attn(ln(x))       + x
//   x = conv(x)           + x        <- conv module normalizes internally
//   x = ffn2(ln(x)) * 0.5 + x        <- macaron HALF step
//   x = final_ln(x)
//
// THE 0.5 FACTORS are what make it macaron rather than two ordinary
// feed-forwards. Dropping them still runs, still trains, and is a different
// architecture -- so they are gated by comparing the whole layer, where the
// piecewise cases cannot see them.
std::vector<float> EncoderLayer(const std::vector<float>& x, int64_t frames, int64_t hidden,
                                int64_t heads, int64_t intermediate, int64_t conv_kernel,
                                int64_t left_max, int64_t right_max,
                                const EncoderLayerWeights& weights, double eps);

}  // namespace w2vbert
}  // namespace models
}  // namespace vllm
