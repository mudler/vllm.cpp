// S2Mel flow-matching scaffolding. See cfm.h.
#include "vllm/model_executor/models/cfm.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace cfm {

std::vector<float> TimestepFeatures(const std::vector<float>& t, int64_t freq_dim,
                                    double max_period, double scale) {
  VT_CHECK(freq_dim > 0, "cfm: frequency dim must be positive");
  const int64_t half = freq_dim / 2;
  const int64_t num_t = static_cast<int64_t>(t.size());
  std::vector<float> out(static_cast<size_t>(num_t * freq_dim));
  for (int64_t n = 0; n < num_t; ++n) {
    for (int64_t i = 0; i < half; ++i) {
      // FLOAT32, deliberately: upstream computes this table once as a float32
      // buffer (). Computing it in double is more
      // accurate and WRONG -- with scale = 1000 the arguments reach ~130, where
      // float32's relative error becomes ~1e-5 absolute and moves cos/sin by
      // ~6.7e-6. Measured; the gate caught it.
      const float freq = static_cast<float>(
          std::exp(-std::log(max_period) * static_cast<double>(i) / static_cast<double>(half)));
      // The ARGUMENT is float32 too: upstream computes `scale * t * freqs` on
      // float32 tensors. Mirrored here so the rounding happens at the same
      // place, though it cannot make a double-precision port bit-exact -- see
      // the tolerance note in the gate.
      const float arg_f32 = static_cast<float>(scale * static_cast<double>(
                                t[static_cast<size_t>(n)])) * freq;
      const double arg = static_cast<double>(arg_f32);
      // COSINE FIRST, then sine.
      out[static_cast<size_t>(n * freq_dim + i)] = static_cast<float>(std::cos(arg));
      out[static_cast<size_t>(n * freq_dim + half + i)] = static_cast<float>(std::sin(arg));
    }
  }
  return out;
}

std::vector<float> EulerStepCfg(const std::vector<float>& x, const std::vector<float>& cond,
                                const std::vector<float>& uncond, int64_t channels,
                                int64_t frames, double dt, double cfg_rate, int64_t prompt_len) {
  const size_t n = static_cast<size_t>(channels * frames);
  VT_CHECK(x.size() == n && cond.size() == n && uncond.size() == n, "cfm: euler shapes");
  VT_CHECK(prompt_len >= 0 && prompt_len <= frames, "cfm: prompt_len out of range");

  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) {
    const double dphi = (1.0 + cfg_rate) * static_cast<double>(cond[i]) -
                        cfg_rate * static_cast<double>(uncond[i]);
    out[i] = static_cast<float>(static_cast<double>(x[i]) + dt * dphi);
  }
  // The prompt frames are re-zeroed AFTER the update, every step.
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t t = 0; t < prompt_len; ++t) {
      out[static_cast<size_t>(c * frames + t)] = 0.0F;
    }
  }
  return out;
}

}  // namespace cfm
}  // namespace models
}  // namespace vllm
