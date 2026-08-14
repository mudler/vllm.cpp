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
#include <map>
#include <string>
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


// BatchNorm + ReLU, the `config_str='batchnorm-relu'` nonlinearity every layer
// wraps itself in (layers.py:10-24). `batchnorm_` sets affine=false, which is
// what the FINAL DenseLayer uses -- passing a gamma/beta there is a different
// model.
std::vector<float> BatchNormRelu(const std::vector<float>& x, int64_t channels, int64_t frames,
                                 const std::vector<float>& gamma, const std::vector<float>& beta,
                                 const std::vector<float>& running_mean,
                                 const std::vector<float>& running_var, double eps);

// TransitLayer (layers.py:183-197): nonlinear THEN 1x1 conv, in that order. The
// reverse still runs and is a different model.
std::vector<float> TransitLayer(const std::vector<float>& x, int64_t in_channels, int64_t frames,
                                int64_t out_channels, const std::vector<float>& bn_gamma,
                                const std::vector<float>& bn_beta,
                                const std::vector<float>& bn_mean,
                                const std::vector<float>& bn_var,
                                const std::vector<float>& weight, const std::vector<float>& bias,
                                double eps);

// DenseLayer (layers.py:199-215): 1x1 conv THEN nonlinear -- the opposite order
// to TransitLayer. A 2-D input (the pooled stats vector) is treated as T=1.
//
// `apply_relu` mirrors `config_str`, and is NOT cosmetic: the final dense uses
// `batchnorm_`, which get_nonlinear (layers.py:10-24) expands to a SINGLE
// batchnorm with affine=false and NO relu. Applying one anyway clamps every
// negative component of the style vector to zero -- a plausible-looking
// embedding that is the wrong model.
std::vector<float> DenseLayer(const std::vector<float>& x, int64_t in_channels, int64_t frames,
                              int64_t out_channels, const std::vector<float>& weight,
                              const std::vector<float>& bias, const std::vector<float>& bn_gamma,
                              const std::vector<float>& bn_beta, const std::vector<float>& bn_mean,
                              const std::vector<float>& bn_var, double eps, bool apply_relu);

struct DenseTdnnLayerWeights {
  std::vector<float> bn1_gamma, bn1_beta, bn1_mean, bn1_var;  // nonlinear1
  std::vector<float> linear1;                                  // 1x1, no bias
  std::vector<float> bn2_gamma, bn2_beta, bn2_mean, bn2_var;  // nonlinear2
  CamLayerWeights cam;
};

// CAMDenseTDNNLayer (layers.py:143-150): nonlinear1 -> linear1 -> nonlinear2 ->
// cam_layer.
std::vector<float> DenseTdnnLayer(const std::vector<float>& x, int64_t in_channels, int64_t frames,
                                  int64_t bn_channels, int64_t out_channels, int64_t kernel,
                                  int64_t dilation, int64_t seg_len,
                                  const DenseTdnnLayerWeights& weights, double eps);

// CAMDenseTDNNBlock (layers.py:177-181): x = cat([x, layer(x)], dim=1) — the
// channel count GROWS by out_channels per layer, and each layer sees every
// earlier output. Appending in the wrong order still produces a tensor of the
// right shape.
std::vector<float> DenseTdnnBlock(const std::vector<float>& x, int64_t in_channels, int64_t frames,
                                  int64_t bn_channels, int64_t growth, int64_t kernel,
                                  int64_t dilation, int64_t seg_len,
                                  const std::vector<DenseTdnnLayerWeights>& layers, double eps);


// ── FCM 2-D front end (DTDNN.py:13-47) ──────────────────────────────────────
// The head runs over the [1, feat_dim, T] spectrogram as a 2-D image, so these
// are genuine Conv2d/BatchNorm2d, not the 1-D forms above.

struct ResBlock2dWeights {
  std::vector<float> conv1, bn1_gamma, bn1_beta, bn1_mean, bn1_var;
  std::vector<float> conv2, bn2_gamma, bn2_beta, bn2_mean, bn2_var;
  // Present only when the block downsamples or changes width.
  std::vector<float> short_conv, short_gamma, short_beta, short_mean, short_var;
  bool has_shortcut = false;
};

// BasicResBlock (layers.py:218-252). THE STRIDE IS (stride, 1): it subsamples
// the FREQUENCY axis and leaves TIME untouched. Striding both still yields a
// well-formed tensor at half the frame rate, which every later layer accepts.
// Returns [planes, ceil(h/stride), w]; `out_h` reports the height.
std::vector<float> ResBlock2d(const std::vector<float>& x, int64_t in_planes, int64_t h, int64_t w,
                              int64_t planes, int64_t stride, const ResBlock2dWeights& weights,
                              double eps, int64_t* out_h);


// ── the whole encoder ───────────────────────────────────────────────────────

// Weights keyed by UPSTREAM state_dict name, which is what a real checkpoint
// carries. Looking a tensor up by name (rather than by position) is what makes a
// missing one throw BY NAME instead of reading as zeros.
struct CampplusWeights {
  std::map<std::string, std::vector<float>> t;
  const std::vector<float>& Get(const std::string& name) const;
  bool Has(const std::string& name) const { return t.count(name) != 0; }
};

struct CampplusParams {
  int64_t feat_dim = 80;
  int64_t embedding_size = 512;
  int64_t growth_rate = 32;
  int64_t bn_size = 4;
  int64_t init_channels = 128;
  int64_t m_channels = 32;   // FCM's fixed width (DTDNN.py:17)
  int64_t seg_len = 100;
  double eps = 1e-5;
};

// Optional capture of an intermediate activation. StatsPool averages over time,
// so the final embedding CANNOT see a frame-count change: a wrong stride,
// dilation or padding in the TDNN head still yields a plausible embedding. The
// post-TDNN tensor is therefore gated directly rather than through the output.
struct ForwardTrace {
  std::vector<float> tdnn;
  int64_t tdnn_channels = 0;
  int64_t tdnn_frames = 0;
};

// CAMPPlus::forward (DTDNN.py:111-115). Input is [T, feat_dim] in (T, F) order,
// as `infer_v2_5.py` supplies it; the encoder permutes to (F, T) internally.
// Returns the embedding, [embedding_size].
std::vector<float> Forward(const CampplusParams& params, const CampplusWeights& weights,
                           const std::vector<float>& feats, int64_t frames,
                           ForwardTrace* trace = nullptr);

}  // namespace campplus
}  // namespace models
}  // namespace vllm
