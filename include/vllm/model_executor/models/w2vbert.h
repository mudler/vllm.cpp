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

}  // namespace w2vbert
}  // namespace models
}  // namespace vllm
