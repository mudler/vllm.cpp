// S2Mel length regulator primitives (#634).
//
// `infer_v2_5.py:650` stretches the semantic sequence onto the target mel length
// through `InterpolateRegulator`, whose defining op is
// `F.interpolate(mode='nearest')`. Its stack is Conv1d(k=3)/GroupNorm/Mish
// repeated, then a 1x1 Conv1d.
#pragma once

#include <cstdint>
#include <vector>

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

}  // namespace lenreg
}  // namespace models
}  // namespace vllm
