// The single entry point: conditioning in, WAVEFORM out (#634).
//
// Joins the two halves this lane built separately. Everything it calls is
// already ported and gated on its own:
//
//   lenreg::RegulateHost      content at the code rate -> the mel frame rate
//   dit_front / dit_stack / dit_tail   the S2Mel estimator, twice per step
//   cfm::EulerStepCfg         the classifier-free-guided integration
//   bigvgan::Forward          mel -> samples
//
// The ROTARY TABLE is built here rather than zeroed, from
// `gpt_fast/model.py precompute_freqs_cis`: `theta = pos / base^(2i/head_dim)`
// laid out as (cos, sin) pairs. A zeroed table makes every position identical,
// which still integrates to a mel and still renders -- it is exactly the kind of
// stand-in that reads as a working pipeline.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/bigvgan.h"
#include "vllm/model_executor/models/dit_front.h"
#include "vllm/model_executor/models/dit_stack.h"
#include "vllm/model_executor/models/dit_tail.h"
#include "vllm/model_executor/models/lenreg.h"

namespace vllm {
namespace models {
namespace indextts2 {

// precompute_freqs_cis for `frames` positions, [frames, head_dim/2, 2].
std::vector<float> RotaryTable(int64_t frames, int64_t head_dim, double base);

struct RenderConfig {
  int64_t mel_frames = 0;   // how long the output should be, in mel frames
  int64_t steps = 10;       // CFM Euler steps
  double cfg_rate = 0.7;    // classifier-free guidance
  double rope_base = 10000.0;
};

struct RenderStages {
  lenreg::RegulatorConfig regulator_config;
  lenreg::RegulatorWeights regulator;
  dit_front::Config front_config;
  dit_front::Weights front;
  dit_stack::Config stack_config;
  dit_stack::Weights stack;
  dit_tail::Config tail_config;
  dit_tail::Weights tail;
  bigvgan::Config vocoder_config;
  bigvgan::Weights vocoder;
};

// `content` is [content_frames, regulator in_channels] -- the semantic content
// the talker's codes resolve to. `style` is the CAMPPlus vector plus the emotion
// contribution. Returns the waveform.
std::vector<float> Render(const RenderConfig& cfg, RenderStages& stages,
                          const std::vector<float>& content, int64_t content_frames,
                          const std::vector<float>& style,
                          const std::vector<float>& initial_noise);

}  // namespace indextts2
}  // namespace models
}  // namespace vllm
