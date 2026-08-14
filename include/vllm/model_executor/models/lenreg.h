// S2Mel length regulator primitives (#634).
//
// `infer_v2_5.py:650` stretches the semantic sequence onto the target mel length
// through `InterpolateRegulator`, whose defining op is
// `F.interpolate(mode='nearest')`. Its stack is Conv1d(k=3)/GroupNorm/Mish
// repeated, then a 1x1 Conv1d.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace vllm {
namespace models {
namespace lenreg {

// Nearest-neighbour resample along time, [channels, in_frames] ->
// [channels, out_frames].
//
// THE INDEX RULE IS `src = floor(i * in_len / out_len)`, computed in floating
// point exactly as torch does. Rounding instead of truncating, or using
// `(i + 0.5) * ratio`, shifts frames by one at non-integer ratios and still
// produces audio -- so the goldens include an upsample to a NON-INTEGER
// multiple (7 -> 17), an exact multiple (7 -> 14), and a downsample (7 -> 3).
std::vector<float> InterpolateNearest(const std::vector<float>& x, int64_t channels,
                                      int64_t in_frames, int64_t out_frames);

// torch.nn.GroupNorm over [channels, frames]: statistics are shared across each
// GROUP of channels and all frames, not per channel.
std::vector<float> GroupNorm(const std::vector<float>& x, int64_t channels, int64_t frames,
                             int64_t groups, const std::vector<float>& gamma,
                             const std::vector<float>& beta, double eps);

// Mish: x * tanh(softplus(x)). Easy to confuse with SiLU, which it resembles.
double Mish(double x);


// The length regulator as the SHIPPED checkpoint has it (#634).
//
// Upstream `indextts/s2mel/modules/length_regulator.py:90-140` with
// `is_discrete: false`, which is what config.yaml sets:
//
//   x   = content_in_proj(x)               // [T, in_channels] -> [T, channels]
//   x   = interpolate(x^T, out_frames)     // NEAREST, to the mel frame count
//   out = model(x)^T                       // 4 x (conv3, GroupNorm, Mish), conv1
//
// `is_discrete` is false here, so `embedding` and `mask_token` ship and are NOT
// read -- the same shape of dead tensor as the DiT's `cond_embedder`. A port
// that took the discrete branch would embed the codes instead of projecting
// them, which runs and produces a differently-conditioned model.
struct RegulatorWeights {
  std::vector<float> in_proj_w, in_proj_b;      // [channels, in_channels]
  // Four (conv, norm) pairs then a final 1x1 convolution.
  std::vector<std::vector<float>> conv_w, conv_b;
  std::vector<std::vector<float>> norm_w, norm_b;
  std::vector<float> out_conv_w, out_conv_b;
};

struct RegulatorConfig {
  int64_t channels = 0;
  int64_t in_channels = 0;
  int64_t groups = 1;      // torch GroupNorm; upstream builds it with 1 group
  double eps = 1e-5;
};

// `x` is [frames, in_channels]; returns [out_frames, channels].
std::vector<float> RegulateHost(const RegulatorConfig& cfg, const RegulatorWeights& w,
                                const std::vector<float>& x, int64_t frames,
                                int64_t out_frames);

// Read them from the converted `s2mel.safetensors`.
RegulatorWeights LoadRegulator(const SafetensorsFile& file, RegulatorConfig* cfg);

}  // namespace lenreg
}  // namespace models
}  // namespace vllm
