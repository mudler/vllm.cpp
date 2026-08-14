// EnhancedCodec.quantize. See codec_encoder.h for the upstream anchors.
#include "vllm/model_executor/models/codec_encoder.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace codec_encoder {
namespace {

// exact-erf GELU, as torch's default F.gelu uses.
float Gelu(float v) {
  const double x = static_cast<double>(v);
  return static_cast<float>(0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))));
}

}  // namespace

Result Encode(const Config& cfg, const Weights& w, const std::vector<float>& x,
              int64_t frames) {
  VT_CHECK(cfg.hidden > 0 && cfg.vocos_dim > 0 && frames > 0,
           "codec_encoder: hidden, vocos_dim and frames must be positive");
  VT_CHECK(x.size() == static_cast<size_t>(frames * cfg.hidden),
           "codec_encoder: x must be [frames, hidden]");

  const int64_t H = cfg.hidden;

  // Channel-major view, which is what both the downsample and the backbone want.
  std::vector<float> cm(static_cast<size_t>(H * frames));
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t c = 0; c < H; ++c) {
      cm[static_cast<size_t>(c * frames + f)] = x[static_cast<size_t>(f * H + c)];
    }
  }

  int64_t cur_frames = frames;
  std::vector<float> after_down_copy;
  if (cfg.downsample) {
    // Conv1d(H, H, kernel 3, stride 2, padding 1), then GELU.
    const int64_t out_frames = (cur_frames + 2 * 1 - 3) / 2 + 1;
    VT_CHECK(out_frames > 0, "codec_encoder: too few frames to downsample");
    VT_CHECK(w.down_w.size() == static_cast<size_t>(H * H * 3),
             "codec_encoder: down weight must be [hidden, hidden, 3]");
    std::vector<float> down(static_cast<size_t>(H * out_frames));
    for (int64_t o = 0; o < H; ++o) {
      const double b =
          w.down_b.empty() ? 0.0 : static_cast<double>(w.down_b[static_cast<size_t>(o)]);
      for (int64_t t = 0; t < out_frames; ++t) {
        double acc = b;
        for (int64_t c = 0; c < H; ++c) {
          for (int64_t k = 0; k < 3; ++k) {
            const int64_t src = t * 2 + k - 1;  // stride 2, padding 1
            if (src < 0 || src >= cur_frames) {
              continue;  // the zero pad
            }
            acc += static_cast<double>(cm[static_cast<size_t>(c * cur_frames + src)]) *
                   static_cast<double>(w.down_w[static_cast<size_t>((o * H + c) * 3 + k)]);
          }
        }
        down[static_cast<size_t>(o * out_frames + t)] = Gelu(static_cast<float>(acc));
      }
    }
    cm = std::move(down);
    cur_frames = out_frames;
    after_down_copy = cm;
  }

  // VocosBackbone takes [hidden, frames] and returns [frames, vocos_dim].
  const std::vector<float> feats =
      vocos::Backbone(cm, H, cur_frames, cfg.vocos_dim, cfg.vocos_intermediate,
                      w.backbone, cfg.eps);
  VT_CHECK(feats.size() == static_cast<size_t>(cur_frames * cfg.vocos_dim),
           "codec_encoder: the backbone returned an unexpected shape");

  // Linear(vocos_dim -> hidden), then back to channel-major for the quantizer.
  std::vector<float> z(static_cast<size_t>(H * cur_frames));
  for (int64_t t = 0; t < cur_frames; ++t) {
    for (int64_t o = 0; o < H; ++o) {
      double acc =
          w.proj_b.empty() ? 0.0 : static_cast<double>(w.proj_b[static_cast<size_t>(o)]);
      for (int64_t i = 0; i < cfg.vocos_dim; ++i) {
        acc += static_cast<double>(feats[static_cast<size_t>(t * cfg.vocos_dim + i)]) *
               static_cast<double>(w.proj_w[static_cast<size_t>(o * cfg.vocos_dim + i)]);
      }
      z[static_cast<size_t>(o * cur_frames + t)] = static_cast<float>(acc);
    }
  }

  const fvq::QuantizeResult q =
      fvq::Quantize(z, cur_frames, H, cfg.codebook_dim, cfg.codebook_size, w.quantizer);

  Result out;
  out.after_down = after_down_copy;
  out.latent = z;
  out.indices = q.indices;
  out.quantized = q.z_q;
  out.out_frames = cur_frames;
  return out;
}

}  // namespace codec_encoder
}  // namespace models
}  // namespace vllm
