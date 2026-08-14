// The S2Mel DiT FRONT END: how the conditioned input is built (#634).
//
// Upstream `indextts/s2mel/modules/diffusion_transformer.py:206-226`, index-tts
// @4f8792ff120cd3ea470dd511e997a17c86cddd10, under the shipped config
// (`style_condition: true`, `style_as_token: false`):
//
//   cond = cond_projection(cond)
//   x_in = cat([x^T, prompt_x^T, cond], -1)        // 80 + 80 + 512 = 672
//   x_in = cat([x_in, style repeated over T], -1)  // + 192          = 864
//   if class_dropout: x_in[..., in_channels:] *= 0
//   x_in = cond_x_merge_linear(x_in)               // 864 -> hidden
//
// `cond_x_merge_linear.weight` is [512, 864] in the shipped checkpoint, and 864
// is exactly 512 + 80 * 2 + 192, so the concatenation order and widths are
// pinned by the weight itself.
//
// TWO THINGS CONTRADICT WHAT A READER EXPECTS.
//
// `cond_in_module` is FORCED to `cond_projection` upstream: the `content_type`
// switch that would have chosen `cond_embedder` is commented out. So
// `cond_embedder` is present in `s2mel.pth` and DEAD in 2.5. A port that
// "restored" the switch would read a tensor this model never uses.
//
// `class_dropout` zeroes everything AFTER the first `in_channels` columns,
// keeping x and dropping prompt, cond and style. That is not a training-only
// path: `mask_content` sets it at inference, so it IS the classifier-free
// guidance unconditional branch that `cfm::EulerStepCfg` consumes.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace dit_front {

struct Weights {
  std::vector<float> cond_proj_w;   // [hidden, hidden]
  std::vector<float> cond_proj_b;   // [hidden]
  std::vector<float> merge_w;       // [hidden, in_channels * 2 + hidden + style]
  std::vector<float> merge_b;       // [hidden]
};

struct Config {
  int64_t hidden = 0;
  int64_t in_channels = 0;
  int64_t style = 0;
  int64_t frames = 0;
};

// x and prompt_x are [in_channels, frames] CHANNEL-major, as upstream holds
// them before its transpose. cond is [frames, hidden]. style is [style].
// `unconditional` selects the CFG branch that zeroes everything past
// in_channels. Returns [frames, hidden].
std::vector<float> BuildXIn(const Config& cfg, const Weights& w,
                            const std::vector<float>& x,
                            const std::vector<float>& prompt_x,
                            const std::vector<float>& cond,
                            const std::vector<float>& style, bool unconditional);

}  // namespace dit_front
}  // namespace models
}  // namespace vllm
