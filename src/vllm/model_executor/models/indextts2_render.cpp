// The render entry point. See indextts2_render.h for the anchors.
#include "vllm/model_executor/models/indextts2_render.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "vllm/model_executor/models/cfm.h"
#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace indextts2 {

std::vector<float> RotaryTable(int64_t frames, int64_t head_dim, double base) {
  VT_CHECK(frames > 0 && head_dim > 0 && head_dim % 2 == 0,
           "indextts2: rotary needs an even head_dim and at least one frame");
  const int64_t half = head_dim / 2;
  std::vector<float> out(static_cast<size_t>(frames * half * 2));
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t k = 0; k < half; ++k) {
      const double inv = 1.0 / std::pow(base, static_cast<double>(2 * k) /
                                                  static_cast<double>(head_dim));
      const double angle = static_cast<double>(f) * inv;
      out[static_cast<size_t>((f * half + k) * 2)] = static_cast<float>(std::cos(angle));
      out[static_cast<size_t>((f * half + k) * 2 + 1)] = static_cast<float>(std::sin(angle));
    }
  }
  return out;
}

std::vector<float> Render(const RenderConfig& cfg, RenderStages& s,
                          const std::vector<float>& content, int64_t content_frames,
                          const std::vector<float>& style,
                          const std::vector<float>& initial_noise) {
  VT_CHECK(cfg.mel_frames > 0 && cfg.steps > 0,
           "indextts2: mel_frames and steps must be positive");
  VT_CHECK(content_frames > 0, "indextts2: the content sequence is empty");

  const int64_t T = cfg.mel_frames;
  const int64_t D = s.stack_config.dim;
  const int64_t C = s.front_config.in_channels;
  VT_CHECK(initial_noise.size() == static_cast<size_t>(C * T),
           "indextts2: the initial noise must be [in_channels, mel_frames]");

  // 1. The content is resampled from the CODE rate to the MEL rate.
  const std::vector<float> cond = lenreg::RegulateHost(
      s.regulator_config, s.regulator, content, content_frames, T);

  s.front_config.frames = T;
  s.stack_config.frames = T;
  s.tail_config.frames = T;
  const std::vector<float> freqs =
      RotaryTable(T, s.stack_config.head_dim, cfg.rope_base);
  const std::vector<float> prompt(static_cast<size_t>(C * T), 0.0F);

  // 2. The CFM integration. Each step runs the estimator TWICE -- once
  // conditioned, once not -- because that is what guidance costs.
  std::vector<float> x = initial_noise;
  const double dt = 1.0 / static_cast<double>(cfg.steps);
  for (int64_t step = 0; step < cfg.steps; ++step) {
    const float t = static_cast<float>(step) / static_cast<float>(cfg.steps);
    const std::vector<float> t1(static_cast<size_t>(D), 0.01F * t);

    const std::vector<float> in_c = dit_front::BuildXIn(s.front_config, s.front, x,
                                                        prompt, cond, style, false);
    const std::vector<float> r_c =
        dit_stack::Forward(s.stack_config, s.stack, in_c, t1, freqs);
    const std::vector<float> v_c =
        dit_tail::Forward(s.tail_config, s.tail, r_c, x, t, t1, {});

    const std::vector<float> in_u = dit_front::BuildXIn(s.front_config, s.front, x,
                                                        prompt, cond, style, true);
    const std::vector<float> r_u =
        dit_stack::Forward(s.stack_config, s.stack, in_u, t1, freqs);
    const std::vector<float> v_u =
        dit_tail::Forward(s.tail_config, s.tail, r_u, x, t, t1, {});

    x = cfm::EulerStepCfg(x, v_c, v_u, C, T, dt, cfg.cfg_rate, 0);
  }

  // 3. The vocoder.
  return bigvgan::Forward(s.vocoder_config, s.vocoder, x, T);
}

}  // namespace indextts2
}  // namespace models
}  // namespace vllm
