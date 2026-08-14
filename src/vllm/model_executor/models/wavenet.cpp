// WaveNet stack. See wavenet.h for the upstream anchors.
#include "vllm/model_executor/models/wavenet.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "vllm/model_executor/models/vocoder1d.h"
#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace wavenet {
namespace {

// SConv1d's REFLECT padding (encodec.py:96 `pad1d`). With stride 1 the extra
// right-hand padding is always zero, so the pad is symmetric, but reflecting is
// not the same as zero-filling and the difference reaches the output.
//
// Reflection excludes the edge sample itself, mirroring torch's 'reflect'.
// Upstream inserts extra zeros first when the input is shorter than the pad;
// that case is refused here rather than approximated, because a silent
// approximation is exactly the kind of drift these gates exist to stop.
int64_t ReflectIndex(int64_t t, int64_t frames) {
  if (frames == 1) {
    return 0;
  }
  const int64_t period = 2 * (frames - 1);
  int64_t m = t % period;
  if (m < 0) {
    m += period;
  }
  return (m < frames) ? m : period - m;
}

// Conv1d over [in_ch, frames] -> [out_ch, frames], dilated, reflect-padded so
// the output keeps its length.
std::vector<float> Conv1dSame(const std::vector<float>& x, int64_t in_ch, int64_t frames,
                              const std::vector<float>& w, const std::vector<float>& bias,
                              int64_t out_ch, int64_t kernel, int64_t dilation) {
  const int64_t effective = (kernel - 1) * dilation + 1;
  const int64_t pad_total = effective - 1;  // stride 1
  const int64_t pad_right = pad_total / 2;
  const int64_t pad_left = pad_total - pad_right;
  VT_CHECK(frames > pad_left && frames > pad_right,
           "wavenet: reflect padding needs more frames than the pad width");

  std::vector<float> out(static_cast<size_t>(out_ch * frames));
  for (int64_t o = 0; o < out_ch; ++o) {
    for (int64_t t = 0; t < frames; ++t) {
      double acc = bias.empty() ? 0.0 : static_cast<double>(bias[static_cast<size_t>(o)]);
      for (int64_t c = 0; c < in_ch; ++c) {
        for (int64_t k = 0; k < kernel; ++k) {
          // Position in the padded signal, mapped back through the reflection.
          const int64_t src = t + k * dilation - pad_left;
          const int64_t idx = ReflectIndex(src, frames);
          const double wv =
              w[static_cast<size_t>((o * in_ch + c) * kernel + k)];
          acc += wv * static_cast<double>(x[static_cast<size_t>(c * frames + idx)]);
        }
      }
      out[static_cast<size_t>(o * frames + t)] = static_cast<float>(acc);
    }
  }
  return out;
}

std::vector<float> Materialize(const ConvWeights& c, int64_t out_ch) {
  return vocoder1d::MaterializeWeightNorm(c.g, c.v, out_ch);
}

}  // namespace

std::vector<float> Forward(const Config& cfg, const Weights& w,
                           const std::vector<float>& x, int64_t frames,
                           const std::vector<float>& g,
                           const std::vector<float>& mask) {
  VT_CHECK(cfg.hidden > 0 && cfg.layers > 0 && cfg.kernel > 0,
           "wavenet: hidden, layers and kernel must be positive");
  VT_CHECK(cfg.kernel % 2 == 1, "wavenet: kernel_size must be odd");
  VT_CHECK(x.size() == static_cast<size_t>(cfg.hidden * frames),
           "wavenet: x must be [hidden, frames]");
  VT_CHECK(static_cast<int64_t>(w.in_layers.size()) == cfg.layers &&
               static_cast<int64_t>(w.res_skip_layers.size()) == cfg.layers,
           "wavenet: one in_layer and one res_skip_layer per layer");
  VT_CHECK(mask.empty() || static_cast<int64_t>(mask.size()) == frames,
           "wavenet: mask must be empty or [frames]");

  const int64_t hidden = cfg.hidden;
  const auto masked = [&](int64_t t) -> float {
    return mask.empty() ? 1.0F : mask[static_cast<size_t>(t)];
  };

  // ONE conditioning projection, sliced per layer. Upstream applies the whole
  // cond_layer once, outside the loop, and each layer reads its own window.
  std::vector<float> cond;
  if (cfg.gin != 0) {
    VT_CHECK(static_cast<int64_t>(g.size()) == cfg.gin,
             "wavenet: g must be [gin] when gin is non-zero");
    const int64_t cond_ch = 2 * hidden * cfg.layers;
    const std::vector<float> cw = Materialize(w.cond, cond_ch);
    // g is one frame, and the conditioning conv has kernel 1, so this is a
    // plain matrix-vector product; no padding is involved.
    cond.assign(static_cast<size_t>(cond_ch), 0.0F);
    for (int64_t o = 0; o < cond_ch; ++o) {
      double acc = w.cond.bias.empty() ? 0.0
                                       : static_cast<double>(w.cond.bias[static_cast<size_t>(o)]);
      for (int64_t c = 0; c < cfg.gin; ++c) {
        acc += static_cast<double>(cw[static_cast<size_t>(o * cfg.gin + c)]) *
               static_cast<double>(g[static_cast<size_t>(c)]);
      }
      cond[static_cast<size_t>(o)] = static_cast<float>(acc);
    }
  }

  std::vector<float> cur = x;
  std::vector<float> output(static_cast<size_t>(hidden * frames), 0.0F);

  for (int64_t i = 0; i < cfg.layers; ++i) {
    int64_t dilation = 1;
    for (int64_t d = 0; d < i; ++d) {
      dilation *= cfg.dilation_rate;
    }
    const ConvWeights& in_c = w.in_layers[static_cast<size_t>(i)];
    const std::vector<float> in_w = Materialize(in_c, 2 * hidden);
    const std::vector<float> x_in =
        Conv1dSame(cur, hidden, frames, in_w, in_c.bias, 2 * hidden, cfg.kernel, dilation);

    // fused_add_tanh_sigmoid_multiply(x_in, g_l, hidden): tanh of the first
    // half times sigmoid of the second, both AFTER adding this layer's slice of
    // the conditioning. g_l is broadcast over time (it has one frame).
    std::vector<float> acts(static_cast<size_t>(hidden * frames));
    const int64_t cond_offset = i * 2 * hidden;
    for (int64_t c = 0; c < hidden; ++c) {
      const double gt =
          cond.empty() ? 0.0 : static_cast<double>(cond[static_cast<size_t>(cond_offset + c)]);
      const double gs =
          cond.empty() ? 0.0
                       : static_cast<double>(cond[static_cast<size_t>(cond_offset + hidden + c)]);
      for (int64_t t = 0; t < frames; ++t) {
        const double a = static_cast<double>(x_in[static_cast<size_t>(c * frames + t)]) + gt;
        const double b =
            static_cast<double>(x_in[static_cast<size_t>((hidden + c) * frames + t)]) + gs;
        acts[static_cast<size_t>(c * frames + t)] =
            static_cast<float>(std::tanh(a) * (1.0 / (1.0 + std::exp(-b))));
      }
    }

    const ConvWeights& rs_c = w.res_skip_layers[static_cast<size_t>(i)];
    const int64_t rs_ch = (i < cfg.layers - 1) ? 2 * hidden : hidden;
    VT_CHECK(static_cast<int64_t>(rs_c.g.size()) == rs_ch,
             "wavenet: the last res_skip layer emits hidden, the rest 2 * hidden");
    const std::vector<float> rs_w = Materialize(rs_c, rs_ch);
    const std::vector<float> rs =
        Conv1dSame(acts, hidden, frames, rs_w, rs_c.bias, rs_ch, 1, 1);

    if (i < cfg.layers - 1) {
      // First half updates the residual (masked), second half accumulates.
      for (int64_t c = 0; c < hidden; ++c) {
        for (int64_t t = 0; t < frames; ++t) {
          const size_t at = static_cast<size_t>(c * frames + t);
          cur[at] = (cur[at] + rs[at]) * masked(t);
          output[at] += rs[static_cast<size_t>((hidden + c) * frames + t)];
        }
      }
    } else {
      for (size_t at = 0; at < output.size(); ++at) {
        output[at] += rs[at];
      }
    }
  }

  for (int64_t c = 0; c < hidden; ++c) {
    for (int64_t t = 0; t < frames; ++t) {
      output[static_cast<size_t>(c * frames + t)] *= masked(t);
    }
  }
  return output;
}

}  // namespace wavenet
}  // namespace models
}  // namespace vllm
