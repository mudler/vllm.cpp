// The S2Mel DiT tail. See dit_tail.h for the upstream anchors.
#include "vllm/model_executor/models/dit_tail.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "vllm/model_executor/models/cfm.h"
#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace dit_tail {
namespace {

// y[row, out] = sum_in x[row, in] * W[out, in] + b[out], frame-major both sides.
std::vector<float> Dense(const std::vector<float>& x, int64_t rows, int64_t in_dim,
                         const Linear& l, int64_t out_dim) {
  VT_CHECK(l.weight.size() == static_cast<size_t>(out_dim * in_dim),
           "dit_tail: dense weight must be [out, in]");
  VT_CHECK(l.bias.empty() || l.bias.size() == static_cast<size_t>(out_dim),
           "dit_tail: dense bias must be [out]");
  std::vector<float> y(static_cast<size_t>(rows * out_dim));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t o = 0; o < out_dim; ++o) {
      double acc = l.bias.empty() ? 0.0 : static_cast<double>(l.bias[static_cast<size_t>(o)]);
      for (int64_t i = 0; i < in_dim; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(r * in_dim + i)]) *
               static_cast<double>(l.weight[static_cast<size_t>(o * in_dim + i)]);
      }
      y[static_cast<size_t>(r * out_dim + o)] = static_cast<float>(acc);
    }
  }
  return y;
}

// TimestepEmbedder: sinusoidal features -> Linear -> SiLU -> Linear.
std::vector<float> TimestepEmbedder(float t, int64_t freq_size, const Linear& mlp0,
                                    const Linear& mlp2, int64_t out_dim) {
  const std::vector<float> feats =
      cfm::TimestepFeatures({t}, freq_size, 10000.0, 1000.0);
  std::vector<float> h = Dense(feats, 1, freq_size, mlp0, out_dim);
  for (float& v : h) {
    v = static_cast<float>(static_cast<double>(v) /
                           (1.0 + std::exp(-static_cast<double>(v))));  // SiLU
  }
  return Dense(h, 1, out_dim, mlp2, out_dim);
}

}  // namespace

std::vector<float> Forward(const Config& cfg, const Weights& w,
                           const std::vector<float>& x_res, const std::vector<float>& x,
                           float t, const std::vector<float>& t1,
                           const std::vector<float>& mask) {
  VT_CHECK(cfg.hidden > 0 && cfg.in_channels > 0 && cfg.frames > 0,
           "dit_tail: hidden, in_channels and frames must be positive");
  // The coupling upstream hides by setting both to 512. `final_layer` is sized
  // at the wavenet width and conditioned on `t1`, which arrives at the DiT
  // width, so unequal widths do not compose at all -- upstream itself raises a
  // shape error. Refuse with a message that says which two numbers disagree.
  VT_CHECK(cfg.wn_hidden == cfg.hidden,
           "dit_tail: the wavenet width must equal the DiT hidden width, because "
           "final_layer is built at the former and conditioned at the latter");
  VT_CHECK(x_res.size() == static_cast<size_t>(cfg.frames * cfg.hidden),
           "dit_tail: x_res must be [frames, hidden]");
  VT_CHECK(x.size() == static_cast<size_t>(cfg.frames * cfg.in_channels),
           "dit_tail: x must be [frames, in_channels]");
  VT_CHECK(t1.size() == static_cast<size_t>(cfg.hidden), "dit_tail: t1 must be [hidden]");

  const int64_t frames = cfg.frames;
  const int64_t hidden = cfg.hidden;
  const int64_t wn_h = cfg.wn_hidden;

  // The LONG skip: concatenate the transformer output with the step's input
  // along the feature axis, then project back down.
  std::vector<float> cat(static_cast<size_t>(frames * (hidden + cfg.in_channels)));
  for (int64_t f = 0; f < frames; ++f) {
    const size_t dst = static_cast<size_t>(f * (hidden + cfg.in_channels));
    for (int64_t i = 0; i < hidden; ++i) {
      cat[dst + static_cast<size_t>(i)] = x_res[static_cast<size_t>(f * hidden + i)];
    }
    for (int64_t i = 0; i < cfg.in_channels; ++i) {
      cat[dst + static_cast<size_t>(hidden + i)] =
          x[static_cast<size_t>(f * cfg.in_channels + i)];
    }
  }
  const std::vector<float> xr =
      Dense(cat, frames, hidden + cfg.in_channels, w.skip_linear, hidden);

  // conv1 is a Linear over the FEATURE axis, then the tensor is transposed into
  // the channel-major layout the wavenet wants.
  const std::vector<float> h_fm = Dense(xr, frames, hidden, w.conv1, wn_h);
  std::vector<float> h_cm(static_cast<size_t>(wn_h * frames));
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t c = 0; c < wn_h; ++c) {
      h_cm[static_cast<size_t>(c * frames + f)] = h_fm[static_cast<size_t>(f * wn_h + c)];
    }
  }

  // The SECOND timestep embedding, at the wavenet width, conditioning the
  // wavenet. It is not the same vector as `t1`, which conditions final_layer.
  const std::vector<float> t2 =
      TimestepEmbedder(t, cfg.freq_size, w.t_embedder2_mlp0, w.t_embedder2_mlp2, wn_h);

  const std::vector<float> wn_out =
      wavenet::Forward(cfg.wn, w.wn, h_cm, frames, t2, mask);

  // Long residual: the wavenet output (back to frame-major) plus a projection of
  // the post-skip transformer output.
  const std::vector<float> res = Dense(xr, frames, hidden, w.res_projection, wn_h);
  std::vector<float> merged(static_cast<size_t>(frames * wn_h));
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t c = 0; c < wn_h; ++c) {
      merged[static_cast<size_t>(f * wn_h + c)] =
          wn_out[static_cast<size_t>(c * frames + f)] +
          res[static_cast<size_t>(f * wn_h + c)];
    }
  }

  const std::vector<float> fin =
      adaln::FinalLayer(merged, frames, wn_h, wn_h, t1, w.final_layer, 1e-6);

  // conv2 is a kernel-1 Conv1d, so it is a dense map over channels, and its
  // output stays CHANNEL-major: [in_channels, frames].
  std::vector<float> out(static_cast<size_t>(cfg.in_channels * frames));
  for (int64_t o = 0; o < cfg.in_channels; ++o) {
    const double b =
        w.conv2.bias.empty() ? 0.0 : static_cast<double>(w.conv2.bias[static_cast<size_t>(o)]);
    for (int64_t f = 0; f < frames; ++f) {
      double acc = b;
      for (int64_t c = 0; c < wn_h; ++c) {
        acc += static_cast<double>(fin[static_cast<size_t>(f * wn_h + c)]) *
               static_cast<double>(w.conv2.weight[static_cast<size_t>(o * wn_h + c)]);
      }
      out[static_cast<size_t>(o * frames + f)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace dit_tail
}  // namespace models
}  // namespace vllm
