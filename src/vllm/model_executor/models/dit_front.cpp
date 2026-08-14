// S2Mel DiT front end. See dit_front.h for the upstream anchors.
#include "vllm/model_executor/models/dit_front.h"

#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace dit_front {

std::vector<float> BuildXIn(const Config& cfg, const Weights& w,
                            const std::vector<float>& x,
                            const std::vector<float>& prompt_x,
                            const std::vector<float>& cond,
                            const std::vector<float>& style, bool unconditional) {
  VT_CHECK(cfg.hidden > 0 && cfg.in_channels > 0 && cfg.frames > 0,
           "dit_front: hidden, in_channels and frames must be positive");
  const int64_t T = cfg.frames;
  const int64_t H = cfg.hidden;
  const int64_t C = cfg.in_channels;
  const int64_t S = cfg.style;
  const int64_t wide = C * 2 + H + S;

  VT_CHECK(x.size() == static_cast<size_t>(C * T), "dit_front: x must be [in_channels, frames]");
  VT_CHECK(prompt_x.size() == static_cast<size_t>(C * T),
           "dit_front: prompt_x must be [in_channels, frames]");
  VT_CHECK(cond.size() == static_cast<size_t>(T * H), "dit_front: cond must be [frames, hidden]");
  VT_CHECK(style.size() == static_cast<size_t>(S), "dit_front: style must be [style]");
  VT_CHECK(w.merge_w.size() == static_cast<size_t>(H * wide),
           "dit_front: cond_x_merge_linear must be [hidden, 2*in_channels + hidden + style]");

  // cond_projection FIRST: the concatenation takes the PROJECTED cond, not the
  // raw one. cond_embedder is not consulted -- see the header.
  std::vector<float> cond_p(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t o = 0; o < H; ++o) {
      double acc = w.cond_proj_b.empty()
                       ? 0.0
                       : static_cast<double>(w.cond_proj_b[static_cast<size_t>(o)]);
      for (int64_t i = 0; i < H; ++i) {
        acc += static_cast<double>(cond[static_cast<size_t>(t * H + i)]) *
               static_cast<double>(w.cond_proj_w[static_cast<size_t>(o * H + i)]);
      }
      cond_p[static_cast<size_t>(t * H + o)] = static_cast<float>(acc);
    }
  }

  // cat([x^T, prompt_x^T, cond, style-over-T], -1), then optionally zero
  // everything past in_channels for the unconditional branch.
  std::vector<float> cat(static_cast<size_t>(T * wide));
  for (int64_t t = 0; t < T; ++t) {
    const size_t row = static_cast<size_t>(t * wide);
    for (int64_t c = 0; c < C; ++c) {
      cat[row + static_cast<size_t>(c)] = x[static_cast<size_t>(c * T + t)];
      cat[row + static_cast<size_t>(C + c)] =
          unconditional ? 0.0F : prompt_x[static_cast<size_t>(c * T + t)];
    }
    for (int64_t i = 0; i < H; ++i) {
      cat[row + static_cast<size_t>(2 * C + i)] =
          unconditional ? 0.0F : cond_p[static_cast<size_t>(t * H + i)];
    }
    for (int64_t i = 0; i < S; ++i) {
      cat[row + static_cast<size_t>(2 * C + H + i)] =
          unconditional ? 0.0F : style[static_cast<size_t>(i)];
    }
  }

  std::vector<float> out(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t o = 0; o < H; ++o) {
      double acc =
          w.merge_b.empty() ? 0.0 : static_cast<double>(w.merge_b[static_cast<size_t>(o)]);
      for (int64_t i = 0; i < wide; ++i) {
        acc += static_cast<double>(cat[static_cast<size_t>(t * wide + i)]) *
               static_cast<double>(w.merge_w[static_cast<size_t>(o * wide + i)]);
      }
      out[static_cast<size_t>(t * H + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace dit_front
}  // namespace models
}  // namespace vllm
