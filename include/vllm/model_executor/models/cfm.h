// S2Mel flow-matching scaffolding (#634).
//
// The S2Mel decoder is a conditional flow-matching model whose estimator is a
// DiT. This header covers the scaffolding AROUND the DiT blocks: the sinusoidal
// timestep embedding, and the Euler step with classifier-free guidance.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace cfm {

// Sinusoidal timestep features, [num_t, freq_dim].
//
// TWO DEVIATIONS FROM THE COMMON CONVENTION, both deliberate upstream:
//   * `t` is multiplied by `scale` (1000) BEFORE the frequencies apply.
//   * the halves are concatenated as [cos, sin] -- COSINE FIRST. Most
//     implementations emit sine first, and swapping them yields an embedding
//     that is still smooth, still periodic and completely wrong.
std::vector<float> TimestepFeatures(const std::vector<float>& t, int64_t freq_dim,
                                    double max_period, double scale);

// The Euler update with classifier-free guidance:
//
//   dphi = (1 + rate) * conditional - rate * unconditional
//   x    = x + dt * dphi
//   x[:, :prompt_len] = 0
//
// THE PROMPT REGION IS ZEROED AFTER EVERY STEP (flow_matching.py:113). Skipping
// it lets the solver integrate over the prompt frames, which still yields a mel
// of the right shape and corrupts the region the prompt was meant to pin.
std::vector<float> EulerStepCfg(const std::vector<float>& x, const std::vector<float>& cond,
                                const std::vector<float>& uncond, int64_t channels,
                                int64_t frames, double dt, double cfg_rate, int64_t prompt_len);

}  // namespace cfm
}  // namespace models
}  // namespace vllm
