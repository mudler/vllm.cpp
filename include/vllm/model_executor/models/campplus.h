// CAMPPlus primitives — the speaker-style encoder (W3 of #634).
//
// Ported from `indextts/s2mel/modules/campplus/layers.py` (index-tts), which the
// goldens execute DIRECTLY: it carries no vllm dependency, so the oracle is the
// real class rather than a restatement.
//
// WHY THIS IS ON THE CRITICAL PATH. The talker is constructed with
// `spk_cond_mode="campplus"` (infer_v2_5.py:138) and consumes the 192-d style
// vector this encoder produces, so CAMPPlus sits UPSTREAM of stage 0 rather than
// beside it. Signals are CHANNEL-MAJOR [C, T], matching vocoder1d.h.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace campplus {

// statistics_pooling (layers.py:26-32): mean concatenated with the UNBIASED
// (N-1) standard deviation. The biased form differs by ~0.2% at T=250 -- small
// enough to read as noise, large enough to move a style vector.
std::vector<float> StatsPool(const std::vector<float>& x, int64_t channels, int64_t frames);

// torch.nn.BatchNorm1d in EVAL mode: RUNNING statistics, never batch statistics.
// Using batch statistics still normalizes and is a different model.
std::vector<float> BatchNorm1dEval(const std::vector<float>& x, int64_t channels, int64_t frames,
                                   const std::vector<float>& gamma, const std::vector<float>& beta,
                                   const std::vector<float>& running_mean,
                                   const std::vector<float>& running_var, double eps);

// CAMLayer::seg_pooling (layers.py:100-111): avg_pool1d with kernel == stride ==
// seg_len and ceil_mode=true, each segment then EXPANDED back over seg_len
// frames and the result TRUNCATED to the input length. The final partial segment
// is where an off-by-one lives.
std::vector<float> SegPooling(const std::vector<float>& x, int64_t channels, int64_t frames,
                              int64_t seg_len);

struct CamLayerWeights {
  std::vector<float> linear_local;    // [out, bn, k]
  std::vector<float> linear1_weight;  // [bn/2, bn, 1]
  std::vector<float> linear1_bias;    // [bn/2]
  std::vector<float> linear2_weight;  // [out, bn/2, 1]
  std::vector<float> linear2_bias;    // [out]
};

// CAMLayer::forward (layers.py:93-98):
//   y = linear_local(x)
//   context = mean(x, -1, keepdim) + seg_pooling(x)
//   m = sigmoid(linear2(relu(linear1(context))))
//   return y * m
// The context is a per-channel scalar broadcast over time, so `m` gates each
// output channel uniformly -- getting the broadcast axis wrong still produces a
// plausible signal.
std::vector<float> CamLayer(const std::vector<float>& x, int64_t bn_channels, int64_t frames,
                            int64_t out_channels, int64_t kernel, int64_t dilation,
                            int64_t seg_len, const CamLayerWeights& weights);

}  // namespace campplus
}  // namespace models
}  // namespace vllm
