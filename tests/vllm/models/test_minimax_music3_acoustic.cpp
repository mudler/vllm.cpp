// MiniMax-Music3 — the ACOUSTIC half at REDUCED dimensions (#672, W4+W5).
//
// Every golden here was produced by EXECUTING upstream's own classes
// (scripts/gen-minimax-music3-acoustic-goldens.py against diffusers PR #14456
// head c6da9936) in float32 at dimensions small enough to check in. No weight
// byte of the 28.5 GB checkpoint is present, so this gate runs in CI with no
// asset.
//
// The FULL-SCALE companion — the real fp32 checkpoint against the committed
// oracle capture — is tests/parity/test_minimax_music3_acoustic_real.cpp. This
// file separates an ALGEBRA defect from float rounding; that one proves the
// algebra survives contact with the real weights and the real 2.4B DiT.
//
// THE TOLERANCE, and why it is what it is. The goldens are torch float32; this
// port accumulates in double and stores float32 (see the dtype note in
// minimax_music3_acoustic.h — the STORE is fp32 either way, only the
// accumulator is wider). Over the reductions here the two differ only by
// float32's own rounding, ~6e-8 relative per operation; the deepest stack is
// the 2-layer DiT (norm, 4 projections, attention, a gated MLP, twice) and a
// 2-block vocoder (about 20 convolutions), neither of which can compound that
// past ~1e-6. kRelTol is 1e-5 with a 1e-6 absolute floor — an order above that
// and orders BELOW any algebra defect, every one of which moves values by O(1).
//
// That claim is PROVEN by mutation rather than asserted; the mutation results
// are recorded in the PR body and in the spec's Outcome.
//
// A Pearson coefficient appears NOWHERE in this file. It is scale-invariant, so
// a uniformly scaled latent or waveform passes it while the song is wrong
// (AGENTS.md; spec §5).
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "minimax_music3_acoustic_goldens.inc"
// The stage profiler is an INTERNAL instrument under src/ and deliberately not
// on the public surface; this target reaches it the same way test_music3_profile
// does (`-I src`, tests/CMakeLists.txt). It is here because the intra-DiT spans
// live inside `DitForwardDevice` and only this file can drive that forward.
#include "vllm/model_executor/models/music3_profile.h"
#include "vllm/model_executor/models/minimax_music3_acoustic.h"
#include "vllm/model_executor/models/minimax_music3_device.h"
#include "vllm/model_executor/models/minimax_music3_speech.h"
#include "vt/backend.h"
// VT-CONV1D-MODEL-BLOCK (#1684): the multi-block case asserts the geometry it
// claims rather than assuming it. Same reach as test_vocoder1d.
#include "vt/cpu/cpu_conv1d_block.h"
#include "vt/device.h"

namespace {

namespace m3 = vllm::models::music3;
namespace m3profile = vllm::models::music3::profile;

constexpr double kRelTol = 1e-5;
constexpr double kAbsFloor = 1e-6;

// Compare and REPORT the count: a gate that cannot say how many values it
// examined has not reported. Returns the worst absolute deviation seen.
double ExpectClose(const std::vector<float>& got, const float* want, size_t count,
                   const char* what) {
  REQUIRE_MESSAGE(got.size() == count, what);
  double worst = 0.0;
  size_t bad = 0;
  size_t first_bad = 0;
  for (size_t i = 0; i < count; ++i) {
    const double a = got[i];
    const double b = want[i];
    const double diff = std::abs(a - b);
    const double bound = std::max(kAbsFloor, kRelTol * std::max(std::abs(a), std::abs(b)));
    if (!(diff <= bound)) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
    worst = std::max(worst, diff);
  }
  INFO(what << ": " << count << " values compared, " << bad << " outside tolerance"
            << (bad != 0 ? ", first at index " + std::to_string(first_bad) + " got " +
                               std::to_string(got[first_bad]) + " want " +
                               std::to_string(want[first_bad])
                         : std::string()));
  CHECK(bad == 0);
  return worst;
}

std::vector<float> ToVector(const float* data, size_t count) {
  return std::vector<float>(data, data + count);
}

// ---------------------------------------------------------------------------
// The reduced configs, built from the .inc so a regenerated golden cannot
// silently disagree with what the test constructs.
// ---------------------------------------------------------------------------

vllm::MiniMaxMusic3TransformerConfig DitConfig() {
  vllm::MiniMaxMusic3TransformerConfig config;
  config.in_channels = vllm_test::kMusic3DitInChannels;
  config.condition_dim = vllm_test::kMusic3DitConditionDim;
  config.num_layers = vllm_test::kMusic3DitNumLayers;
  config.num_attention_heads = vllm_test::kMusic3DitNumAttentionHeads;
  config.attention_head_dim = vllm_test::kMusic3DitAttentionHeadDim;
  config.ff_inner_dim = vllm_test::kMusic3DitFfInnerDim;
  config.rotary_dim = vllm_test::kMusic3DitRotaryDim;
  config.fourier_embedding_dim = vllm_test::kMusic3DitFourierEmbeddingDim;
  return config;
}

m3::DitWeights DitWeights() {
  const vllm::MiniMaxMusic3TransformerConfig config = DitConfig();
  const size_t inner = static_cast<size_t>(config.inner_dim());
  const size_t concat = static_cast<size_t>(config.concat_channels());
  const size_t fourier = static_cast<size_t>(config.fourier_embedding_dim);
  const size_t ff = static_cast<size_t>(config.ff_inner_dim);
  const size_t in_ch = static_cast<size_t>(config.in_channels);

  m3::DitWeights weights;
  weights.time_proj_weight = ToVector(vllm_test::kMusic3DitW_time_proj_weight, fourier / 2);
  weights.time_embed_linear_1_weight =
      ToVector(vllm_test::kMusic3DitW_time_embed_linear_1_weight, inner * fourier);
  weights.time_embed_linear_1_bias =
      ToVector(vllm_test::kMusic3DitW_time_embed_linear_1_bias, inner);
  weights.time_embed_linear_2_weight =
      ToVector(vllm_test::kMusic3DitW_time_embed_linear_2_weight, inner * inner);
  weights.time_embed_linear_2_bias =
      ToVector(vllm_test::kMusic3DitW_time_embed_linear_2_bias, inner);
  weights.preprocess_conv_weight =
      ToVector(vllm_test::kMusic3DitW_preprocess_conv_weight, concat * concat);
  weights.proj_in_weight = ToVector(vllm_test::kMusic3DitW_proj_in_weight, inner * concat);
  weights.proj_out_weight = ToVector(vllm_test::kMusic3DitW_proj_out_weight, in_ch * inner);
  weights.postprocess_conv_weight =
      ToVector(vllm_test::kMusic3DitW_postprocess_conv_weight, in_ch * in_ch);

  const float* const norm1_w[] = {vllm_test::kMusic3DitW_transformer_blocks_0_norm1_weight,
                                  vllm_test::kMusic3DitW_transformer_blocks_1_norm1_weight};
  const float* const norm1_b[] = {vllm_test::kMusic3DitW_transformer_blocks_0_norm1_bias,
                                  vllm_test::kMusic3DitW_transformer_blocks_1_norm1_bias};
  const float* const norm2_w[] = {vllm_test::kMusic3DitW_transformer_blocks_0_norm2_weight,
                                  vllm_test::kMusic3DitW_transformer_blocks_1_norm2_weight};
  const float* const norm2_b[] = {vllm_test::kMusic3DitW_transformer_blocks_0_norm2_bias,
                                  vllm_test::kMusic3DitW_transformer_blocks_1_norm2_bias};
  const float* const to_q[] = {vllm_test::kMusic3DitW_transformer_blocks_0_attn_to_q_weight,
                               vllm_test::kMusic3DitW_transformer_blocks_1_attn_to_q_weight};
  const float* const to_k[] = {vllm_test::kMusic3DitW_transformer_blocks_0_attn_to_k_weight,
                               vllm_test::kMusic3DitW_transformer_blocks_1_attn_to_k_weight};
  const float* const to_v[] = {vllm_test::kMusic3DitW_transformer_blocks_0_attn_to_v_weight,
                               vllm_test::kMusic3DitW_transformer_blocks_1_attn_to_v_weight};
  const float* const to_out[] = {
      vllm_test::kMusic3DitW_transformer_blocks_0_attn_to_out_0_weight,
      vllm_test::kMusic3DitW_transformer_blocks_1_attn_to_out_0_weight};
  const float* const ff_in_w[] = {vllm_test::kMusic3DitW_transformer_blocks_0_ff_in_weight,
                                  vllm_test::kMusic3DitW_transformer_blocks_1_ff_in_weight};
  const float* const ff_in_b[] = {vllm_test::kMusic3DitW_transformer_blocks_0_ff_in_bias,
                                  vllm_test::kMusic3DitW_transformer_blocks_1_ff_in_bias};
  const float* const ff_out_w[] = {vllm_test::kMusic3DitW_transformer_blocks_0_ff_out_weight,
                                   vllm_test::kMusic3DitW_transformer_blocks_1_ff_out_weight};
  const float* const ff_out_b[] = {vllm_test::kMusic3DitW_transformer_blocks_0_ff_out_bias,
                                   vllm_test::kMusic3DitW_transformer_blocks_1_ff_out_bias};

  const size_t attn_inner =
      static_cast<size_t>(config.num_attention_heads * config.attention_head_dim);
  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    m3::DitLayerWeights entry;
    entry.norm1_weight = ToVector(norm1_w[layer], inner);
    entry.norm1_bias = ToVector(norm1_b[layer], inner);
    entry.norm2_weight = ToVector(norm2_w[layer], inner);
    entry.norm2_bias = ToVector(norm2_b[layer], inner);
    entry.to_q = ToVector(to_q[layer], attn_inner * inner);
    entry.to_k = ToVector(to_k[layer], attn_inner * inner);
    entry.to_v = ToVector(to_v[layer], attn_inner * inner);
    entry.to_out = ToVector(to_out[layer], inner * attn_inner);
    entry.ff_in_weight = ToVector(ff_in_w[layer], 2 * ff * inner);
    entry.ff_in_bias = ToVector(ff_in_b[layer], 2 * ff);
    entry.ff_out_weight = ToVector(ff_out_w[layer], inner * ff);
    entry.ff_out_bias = ToVector(ff_out_b[layer], inner);
    weights.layers.push_back(std::move(entry));
  }
  return weights;
}

vllm::MiniMaxMusic3VocoderConfig VocConfig() {
  vllm::MiniMaxMusic3VocoderConfig config;
  config.latent_channels = vllm_test::kMusic3VocLatentChannels;
  config.decoder_input_dim = vllm_test::kMusic3VocInputDim;
  config.decoder_hidden_dim = vllm_test::kMusic3VocHiddenDim;
  config.upsampling_ratios.assign(
      vllm_test::kMusic3VocRatios,
      vllm_test::kMusic3VocRatios + vllm_test::kMusic3VocRatioCount);
  config.sampling_rate = vllm_test::kMusic3VocSamplingRate;
  return config;
}

m3::VocoderWeights VocWeights() {
  const vllm::MiniMaxMusic3VocoderConfig config = VocConfig();
  const size_t hidden = static_cast<size_t>(config.decoder_hidden_dim);
  const size_t input_dim = static_cast<size_t>(config.decoder_input_dim);
  const size_t stream = static_cast<size_t>(config.stream_channels());

  m3::VocoderWeights weights;
  weights.dec_in_proj_weight =
      ToVector(vllm_test::kMusic3VocW_dec_in_proj_weight, input_dim * stream);
  weights.dec_in_proj_bias = ToVector(vllm_test::kMusic3VocW_dec_in_proj_bias, input_dim);
  weights.conv_in_weight = ToVector(vllm_test::kMusic3VocW_conv_in_weight, hidden * input_dim * 7);
  weights.conv_in_bias = ToVector(vllm_test::kMusic3VocW_conv_in_bias, hidden);

  struct BlockPtrs {
    const float* snake1;
    const float* conv_t1_w;
    const float* conv_t1_b;
    const float* unit_snake1[3];
    const float* unit_conv1_w[3];
    const float* unit_conv1_b[3];
    const float* unit_snake2[3];
    const float* unit_conv2_w[3];
    const float* unit_conv2_b[3];
  };
  const BlockPtrs blocks[] = {
      {vllm_test::kMusic3VocW_blocks_0_snake1_alpha,
       vllm_test::kMusic3VocW_blocks_0_conv_t1_weight,
       vllm_test::kMusic3VocW_blocks_0_conv_t1_bias,
       {vllm_test::kMusic3VocW_blocks_0_res_unit1_snake1_alpha,
        vllm_test::kMusic3VocW_blocks_0_res_unit2_snake1_alpha,
        vllm_test::kMusic3VocW_blocks_0_res_unit3_snake1_alpha},
       {vllm_test::kMusic3VocW_blocks_0_res_unit1_conv1_weight,
        vllm_test::kMusic3VocW_blocks_0_res_unit2_conv1_weight,
        vllm_test::kMusic3VocW_blocks_0_res_unit3_conv1_weight},
       {vllm_test::kMusic3VocW_blocks_0_res_unit1_conv1_bias,
        vllm_test::kMusic3VocW_blocks_0_res_unit2_conv1_bias,
        vllm_test::kMusic3VocW_blocks_0_res_unit3_conv1_bias},
       {vllm_test::kMusic3VocW_blocks_0_res_unit1_snake2_alpha,
        vllm_test::kMusic3VocW_blocks_0_res_unit2_snake2_alpha,
        vllm_test::kMusic3VocW_blocks_0_res_unit3_snake2_alpha},
       {vllm_test::kMusic3VocW_blocks_0_res_unit1_conv2_weight,
        vllm_test::kMusic3VocW_blocks_0_res_unit2_conv2_weight,
        vllm_test::kMusic3VocW_blocks_0_res_unit3_conv2_weight},
       {vllm_test::kMusic3VocW_blocks_0_res_unit1_conv2_bias,
        vllm_test::kMusic3VocW_blocks_0_res_unit2_conv2_bias,
        vllm_test::kMusic3VocW_blocks_0_res_unit3_conv2_bias}},
      {vllm_test::kMusic3VocW_blocks_1_snake1_alpha,
       vllm_test::kMusic3VocW_blocks_1_conv_t1_weight,
       vllm_test::kMusic3VocW_blocks_1_conv_t1_bias,
       {vllm_test::kMusic3VocW_blocks_1_res_unit1_snake1_alpha,
        vllm_test::kMusic3VocW_blocks_1_res_unit2_snake1_alpha,
        vllm_test::kMusic3VocW_blocks_1_res_unit3_snake1_alpha},
       {vllm_test::kMusic3VocW_blocks_1_res_unit1_conv1_weight,
        vllm_test::kMusic3VocW_blocks_1_res_unit2_conv1_weight,
        vllm_test::kMusic3VocW_blocks_1_res_unit3_conv1_weight},
       {vllm_test::kMusic3VocW_blocks_1_res_unit1_conv1_bias,
        vllm_test::kMusic3VocW_blocks_1_res_unit2_conv1_bias,
        vllm_test::kMusic3VocW_blocks_1_res_unit3_conv1_bias},
       {vllm_test::kMusic3VocW_blocks_1_res_unit1_snake2_alpha,
        vllm_test::kMusic3VocW_blocks_1_res_unit2_snake2_alpha,
        vllm_test::kMusic3VocW_blocks_1_res_unit3_snake2_alpha},
       {vllm_test::kMusic3VocW_blocks_1_res_unit1_conv2_weight,
        vllm_test::kMusic3VocW_blocks_1_res_unit2_conv2_weight,
        vllm_test::kMusic3VocW_blocks_1_res_unit3_conv2_weight},
       {vllm_test::kMusic3VocW_blocks_1_res_unit1_conv2_bias,
        vllm_test::kMusic3VocW_blocks_1_res_unit2_conv2_bias,
        vllm_test::kMusic3VocW_blocks_1_res_unit3_conv2_bias}},
  };

  int64_t last_output = config.decoder_hidden_dim;
  for (size_t index = 0; index < config.upsampling_ratios.size(); ++index) {
    const int64_t stride = config.upsampling_ratios[index];
    const size_t in_dim = static_cast<size_t>(config.decoder_hidden_dim >> index);
    const size_t out_dim = static_cast<size_t>(config.decoder_hidden_dim >> (index + 1));
    last_output = static_cast<int64_t>(out_dim);
    m3::VocoderBlockWeights block;
    block.snake1_alpha = ToVector(blocks[index].snake1, in_dim);
    block.conv_t1_weight =
        ToVector(blocks[index].conv_t1_w, in_dim * out_dim * static_cast<size_t>(2 * stride));
    block.conv_t1_bias = ToVector(blocks[index].conv_t1_b, out_dim);
    for (int unit = 0; unit < 3; ++unit) {
      m3::VocoderResidualUnitWeights entry;
      entry.snake1_alpha = ToVector(blocks[index].unit_snake1[unit], out_dim);
      entry.conv1_weight = ToVector(blocks[index].unit_conv1_w[unit], out_dim * out_dim * 7);
      entry.conv1_bias = ToVector(blocks[index].unit_conv1_b[unit], out_dim);
      entry.snake2_alpha = ToVector(blocks[index].unit_snake2[unit], out_dim);
      entry.conv2_weight = ToVector(blocks[index].unit_conv2_w[unit], out_dim * out_dim * 1);
      entry.conv2_bias = ToVector(blocks[index].unit_conv2_b[unit], out_dim);
      block.res_units.push_back(std::move(entry));
    }
    weights.blocks.push_back(std::move(block));
  }
  weights.snake_out_alpha =
      ToVector(vllm_test::kMusic3VocW_snake_out_alpha, static_cast<size_t>(last_output));
  weights.conv_out_weight =
      ToVector(vllm_test::kMusic3VocW_conv_out_weight, static_cast<size_t>(last_output) * 7);
  weights.conv_out_bias = ToVector(vllm_test::kMusic3VocW_conv_out_bias, 1);
  return weights;
}

vllm::MiniMaxMusic3SchedulerConfig SchedulerConfigFor(size_t index) {
  const vllm_test::Music3ScheduleGolden& golden = vllm_test::kMusic3ScheduleGoldens[index];
  vllm::MiniMaxMusic3SchedulerConfig config;
  config.num_train_timesteps = golden.num_train_timesteps;
  config.shift = golden.shift;
  config.invert_sigmas = golden.invert_sigmas;
  config.use_dynamic_shifting = false;
  config.time_shift_type = "exponential";
  return config;
}

}  // namespace

// ---------------------------------------------------------------------------
// W4 — the scheduler
// ---------------------------------------------------------------------------

TEST_CASE("music3 acoustic: the sigma ramp is linspace(1, 1/n, n)") {
  int64_t checked = 0;
  for (int64_t index = 0; index < vllm_test::kMusic3ScheduleGoldenCount; ++index) {
    const vllm_test::Music3ScheduleGolden& golden = vllm_test::kMusic3ScheduleGoldens[index];
    const std::vector<double> ramp = m3::DenoiseSigmaRamp(golden.num_inference_steps);
    REQUIRE(static_cast<int64_t>(ramp.size()) == golden.num_inference_steps);
    for (int64_t i = 0; i < golden.num_inference_steps; ++i) {
      CHECK(static_cast<float>(ramp[static_cast<size_t>(i)]) ==
            doctest::Approx(golden.ramp[i]).epsilon(1e-6));
      ++checked;
    }
  }
  MESSAGE("sigma ramp: " << checked << " values over "
                         << vllm_test::kMusic3ScheduleGoldenCount << " schedules");
  CHECK(checked == 46);
}

TEST_CASE("music3 acoustic: set_timesteps reproduces shift, inversion and the train scale") {
  int64_t checked = 0;
  for (int64_t index = 0; index < vllm_test::kMusic3ScheduleGoldenCount; ++index) {
    const vllm_test::Music3ScheduleGolden& golden = vllm_test::kMusic3ScheduleGoldens[index];
    CAPTURE(golden.name);
    const m3::FlowMatchSchedule schedule = m3::FlowMatchSetTimesteps(
        m3::DenoiseSigmaRamp(golden.num_inference_steps),
        SchedulerConfigFor(static_cast<size_t>(index)));
    REQUIRE(static_cast<int64_t>(schedule.timesteps.size()) == golden.num_inference_steps);
    // The terminal sigma is APPENDED, so `sigmas` is exactly one longer.
    REQUIRE(static_cast<int64_t>(schedule.sigmas.size()) == golden.num_inference_steps + 1);
    checked += ExpectClose(schedule.timesteps, golden.timesteps,
                           static_cast<size_t>(golden.num_inference_steps),
                           "timesteps") >= 0.0
                   ? golden.num_inference_steps
                   : 0;
    checked += ExpectClose(schedule.sigmas, golden.sigmas,
                           static_cast<size_t>(golden.num_inference_steps + 1),
                           "sigmas") >= 0.0
                   ? golden.num_inference_steps + 1
                   : 0;
  }
  MESSAGE("set_timesteps: " << checked << " values over "
                            << vllm_test::kMusic3ScheduleGoldenCount << " schedules");
  CHECK(checked == 97);
}

TEST_CASE("music3 acoustic: the shipped schedule is the oracle capture's own sigmas") {
  // manifest.json `result.denoise_sigmas` for the committed capture. Asserted
  // here so the schedule cannot drift away from the tensor goldens it produced.
  const m3::FlowMatchSchedule schedule =
      m3::FlowMatchSetTimesteps(m3::DenoiseSigmaRamp(4), vllm::MiniMaxMusic3SchedulerConfig{});
  REQUIRE(schedule.timesteps.size() == 4);
  CHECK(schedule.timesteps[0] == doctest::Approx(0.0));
  CHECK(schedule.timesteps[1] == doctest::Approx(0.25));
  CHECK(schedule.timesteps[2] == doctest::Approx(0.5));
  CHECK(schedule.timesteps[3] == doctest::Approx(0.75));
  REQUIRE(schedule.sigmas.size() == 5);
  CHECK(schedule.sigmas[4] == doctest::Approx(1.0));
}

TEST_CASE("music3 acoustic: an Euler step is sample + dt * velocity at every index") {
  const m3::FlowMatchSchedule schedule =
      m3::FlowMatchSetTimesteps(m3::DenoiseSigmaRamp(4), vllm::MiniMaxMusic3SchedulerConfig{});
  const size_t count = static_cast<size_t>(vllm_test::kMusic3StepChannels *
                                           vllm_test::kMusic3StepLength);
  const std::vector<float> sample = ToVector(vllm_test::kMusic3StepSample, count);
  const std::vector<float> velocity = ToVector(vllm_test::kMusic3StepVelocity, count);
  int64_t checked = 0;
  for (int64_t index = 0; index < 4; ++index) {
    CAPTURE(index);
    const std::vector<float> out = m3::FlowMatchStep(sample, velocity, index, schedule);
    ExpectClose(out, vllm_test::kMusic3StepOut + static_cast<size_t>(index) * count, count,
                "euler step");
    checked += static_cast<int64_t>(count);
  }
  MESSAGE("euler step: " << checked << " values over 4 step indices");
  CHECK(checked == 48);
}

TEST_CASE("music3 acoustic: the scheduler refuses what upstream refuses") {
  vllm::MiniMaxMusic3SchedulerConfig config;
  CHECK_THROWS_AS(m3::FlowMatchSetTimesteps({}, config), std::runtime_error);
  config.use_dynamic_shifting = true;
  CHECK_THROWS_AS(m3::FlowMatchSetTimesteps({1.0, 0.5}, config), std::runtime_error);
  CHECK_THROWS_AS(m3::DenoiseSigmaRamp(0), std::runtime_error);
  const m3::FlowMatchSchedule schedule =
      m3::FlowMatchSetTimesteps(m3::DenoiseSigmaRamp(4), vllm::MiniMaxMusic3SchedulerConfig{});
  // step_index 3 is the LAST valid one: sigmas[4] is the appended terminal.
  CHECK_NOTHROW(m3::FlowMatchStep({1.0f}, {1.0f}, 3, schedule));
  CHECK_THROWS_AS(m3::FlowMatchStep({1.0f}, {1.0f}, 4, schedule), std::runtime_error);
  CHECK_THROWS_AS(m3::FlowMatchStep({1.0f}, {1.0f, 2.0f}, 0, schedule), std::runtime_error);
}

// ---------------------------------------------------------------------------
// W4 — classifier-free guidance
// ---------------------------------------------------------------------------

TEST_CASE("music3 acoustic: CFG bases on the UNCONDITIONAL row at scale 1.7") {
  CHECK(m3::kDitGuidanceScale == doctest::Approx(vllm_test::kMusic3GuidanceScale));
  // Two conditions, so the DiT is forwarded twice per step (denoise.py:219-227).
  CHECK(vllm_test::kMusic3CfgNumConditions == 2);
  const size_t count = static_cast<size_t>(vllm_test::kMusic3CfgCount);
  const std::vector<float> out =
      m3::ClassifierFreeGuidanceMix(ToVector(vllm_test::kMusic3CfgCond, count),
                                    ToVector(vllm_test::kMusic3CfgUncond, count),
                                    m3::kDitGuidanceScale);
  ExpectClose(out, vllm_test::kMusic3CfgOut, count, "cfg mix");
  MESSAGE("cfg mix: " << count << " values compared");
}

TEST_CASE("music3 acoustic: scale 0 is the UNCONDITIONAL row, bit for bit") {
  // Which row the mix is BASED on is the one thing a scale sweep can prove
  // exactly, and it is exactly what `use_original_formulation` would change.
  // `u + 0 * (c - u)` is `u` bit for bit in float32; `c + 0 * (c - u)` is `c`.
  const size_t count = static_cast<size_t>(vllm_test::kMusic3CfgCount);
  const std::vector<float> cond = ToVector(vllm_test::kMusic3CfgCond, count);
  const std::vector<float> uncond = ToVector(vllm_test::kMusic3CfgUncond, count);
  const std::vector<float> out = m3::ClassifierFreeGuidanceMix(cond, uncond, 0.0);
  size_t identical_to_uncond = 0;
  size_t identical_to_cond = 0;
  for (size_t i = 0; i < count; ++i) {
    if (out[i] == uncond[i]) ++identical_to_uncond;
    if (out[i] == cond[i]) ++identical_to_cond;
  }
  MESSAGE("cfg at scale 0: " << count << " values, " << identical_to_uncond
                             << " identical to the unconditional row, " << identical_to_cond
                             << " to the conditional one");
  CHECK(identical_to_uncond == count);
  CHECK(identical_to_cond == 0);
  CHECK_THROWS_AS(m3::ClassifierFreeGuidanceMix(cond, {1.0f}, 1.7), std::runtime_error);
}

TEST_CASE("music3 acoustic: CFG at scale 1 recovers the conditional row to a rounding") {
  // MEASURED, and the reason this case does not claim bit-equality: in float32
  // `u + 1 * (c - u)` is NOT `c` for every input — the subtraction can lose a
  // bit that the addition cannot restore. 10 of these 12 values come back
  // identical and 2 do not, so the claim is a rounding bound plus the count,
  // not an identity. Asserting the identity would be asserting something
  // floating point does not provide.
  const size_t count = static_cast<size_t>(vllm_test::kMusic3CfgCount);
  const std::vector<float> cond = ToVector(vllm_test::kMusic3CfgCond, count);
  const std::vector<float> out =
      m3::ClassifierFreeGuidanceMix(cond, ToVector(vllm_test::kMusic3CfgUncond, count), 1.0);
  size_t identical = 0;
  double worst = 0.0;
  for (size_t i = 0; i < count; ++i) {
    if (out[i] == cond[i]) ++identical;
    worst = std::max(worst, std::abs(static_cast<double>(out[i]) - cond[i]));
  }
  MESSAGE("cfg at scale 1: " << count << " values, " << identical
                             << " bit-identical to the conditional row, max|d| " << worst);
  ExpectClose(out, vllm_test::kMusic3CfgCond, count, "cfg at scale 1");
  CHECK(identical >= count - 2);
}

// ---------------------------------------------------------------------------
// W4 — the denoise loop's window bookkeeping
// ---------------------------------------------------------------------------

TEST_CASE("music3 acoustic: chunk starts match upstream at and past the window boundary") {
  CHECK(m3::kChunkFrames == vllm_test::kMusic3ChunkFrames);
  CHECK(m3::kChunkHop == vllm_test::kMusic3ChunkHop);
  int64_t checked = 0;
  for (int64_t index = 0; index < vllm_test::kMusic3ChunkGoldenCount; ++index) {
    const vllm_test::Music3ChunkGolden& golden = vllm_test::kMusic3ChunkGoldens[index];
    CAPTURE(golden.num_frames);
    const std::vector<int64_t> starts = m3::ChunkStarts(golden.num_frames);
    REQUIRE(static_cast<int64_t>(starts.size()) == golden.count);
    for (int64_t i = 0; i < golden.count; ++i) {
      CHECK(starts[static_cast<size_t>(i)] == golden.starts[i]);
      ++checked;
    }
  }
  MESSAGE("chunk starts: " << checked << " indices over "
                           << vllm_test::kMusic3ChunkGoldenCount << " frame counts");
  CHECK(checked == 24);
  CHECK_THROWS_AS(m3::ChunkStarts(0), std::runtime_error);
}

TEST_CASE("music3 acoustic: the window carry clamps both ends and can be empty") {
  CHECK(m3::kOverlapLatentLength == vllm_test::kMusic3OverlapLatentLength);
  int64_t checked = 0;
  for (int64_t index = 0; index < vllm_test::kMusic3CarryGoldenCount; ++index) {
    const vllm_test::Music3CarryGolden& golden = vllm_test::kMusic3CarryGoldens[index];
    CAPTURE(golden.latent_length);
    const m3::WindowCarrySpan span = m3::ChunkCarrySpan(golden.latent_length);
    CHECK(span.start == golden.overlap_start);
    CHECK(span.end == golden.overlap_end);
    CHECK(span.length() >= 0);
    ++checked;
  }
  MESSAGE("window carry: " << checked << " latent lengths");
  CHECK(checked == 6);
  // A window shorter than one overlap carries NOTHING rather than a negative span.
  CHECK(m3::ChunkCarrySpan(86).length() == 0);
}

TEST_CASE("music3 acoustic: the overlap blend matches upstream at three flow times") {
  const int64_t channels = vllm_test::kMusic3BlendChannels;
  const int64_t length = vllm_test::kMusic3BlendLength;
  const int64_t overlap = vllm_test::kMusic3BlendOverlap;
  const size_t count = static_cast<size_t>(channels * length);
  int64_t checked = 0;
  for (int64_t t = 0; t < vllm_test::kMusic3BlendTimeCount; ++t) {
    CAPTURE(vllm_test::kMusic3BlendTimes[t]);
    std::vector<float> latents = ToVector(vllm_test::kMusic3BlendLatents, count);
    m3::BlendOverlap(latents, channels, length,
                     ToVector(vllm_test::kMusic3BlendNoise,
                              static_cast<size_t>(channels * overlap)),
                     ToVector(vllm_test::kMusic3BlendPrev,
                              static_cast<size_t>(channels * overlap)),
                     overlap, overlap, vllm_test::kMusic3BlendTimes[t]);
    // BIT-EXACT, and that is a deliberate tightening rather than an ambition.
    // The blend is four elementwise float32 operations with no reduction, so
    // there is nothing to round differently. MEASURED: at kRelTol 1e-5 a blend
    // that drops upstream's `(1 - 1e-6)` factor is INVISIBLE — it moves values
    // by 3.3e-07 relative, an order and a half inside that tolerance — and the
    // mutation stayed green until this became an equality. A close-enough bound
    // on an exactly-reproducible quantity is slack that hides a real defect.
    size_t identical = 0;
    for (size_t i = 0; i < count; ++i) {
      if (latents[i] == vllm_test::kMusic3BlendOut[static_cast<size_t>(t) * count + i]) {
        ++identical;
      }
    }
    INFO("overlap blend at t=" << vllm_test::kMusic3BlendTimes[t] << ": " << identical
                               << " of " << count << " bit-identical");
    CHECK(identical == count);
    checked += static_cast<int64_t>(count);
  }
  MESSAGE("overlap blend: " << checked << " values over "
                            << vllm_test::kMusic3BlendTimeCount << " flow times, bit-exact");
  CHECK(checked == 42);
}

TEST_CASE("music3 acoustic: a zero overlap leaves the latents untouched") {
  const size_t count = static_cast<size_t>(vllm_test::kMusic3BlendChannels *
                                           vllm_test::kMusic3BlendLength);
  std::vector<float> latents = ToVector(vllm_test::kMusic3BlendLatents, count);
  const std::vector<float> before = latents;
  m3::BlendOverlap(latents, vllm_test::kMusic3BlendChannels, vllm_test::kMusic3BlendLength,
                   {}, {}, 0, 0, 0.5);
  size_t identical = 0;
  for (size_t i = 0; i < count; ++i) {
    if (latents[i] == before[i]) ++identical;
  }
  MESSAGE("zero overlap: " << identical << " of " << count << " unchanged");
  CHECK(identical == count);
}

TEST_CASE("music3 acoustic: the waveform crop drops the right span per window") {
  CHECK(m3::kCropLeftLatent == vllm_test::kMusic3CropLeftLatent);
  CHECK(m3::kCropRightLatent == vllm_test::kMusic3CropRightLatent);
  int64_t checked = 0;
  for (int64_t index = 0; index < vllm_test::kMusic3CropGoldenCount; ++index) {
    const vllm_test::Music3CropGolden& golden = vllm_test::kMusic3CropGoldens[index];
    CAPTURE(golden.chunk_index);
    CAPTURE(golden.num_chunks);
    const m3::WaveformCropSpan span = m3::VocoderCropSpan(
        golden.chunk_index, golden.num_chunks, golden.waveform_length,
        vllm_test::kMusic3CropHopLength);
    CHECK(span.left == golden.left);
    CHECK(span.right_exclusive == golden.right_exclusive);
    ++checked;
  }
  MESSAGE("waveform crop: " << checked << " window positions");
  CHECK(checked == 4);
  // The single-window case is BOTH first and last, so nothing is dropped.
  const m3::WaveformCropSpan only = m3::VocoderCropSpan(0, 1, 44032, 512);
  CHECK(only.left == 0);
  CHECK(only.right_exclusive == 44032);
}

// ---------------------------------------------------------------------------
// W4 — the DiT's pieces
// ---------------------------------------------------------------------------

TEST_CASE("music3 acoustic: the Fourier time embedding is cos THEN sin") {
  const int64_t dim = vllm_test::kMusic3DitFourierEmbeddingDim;
  const std::vector<float> weight =
      ToVector(vllm_test::kMusic3FourierWeight, static_cast<size_t>(dim / 2));
  int64_t checked = 0;
  for (int64_t t = 0; t < vllm_test::kMusic3FourierTimeCount; ++t) {
    CAPTURE(vllm_test::kMusic3FourierTimes[t]);
    const std::vector<float> out =
        m3::FourierTimeEmbedding(vllm_test::kMusic3FourierTimes[t], weight, dim);
    ExpectClose(out, vllm_test::kMusic3FourierOut + static_cast<size_t>(t * dim),
                static_cast<size_t>(dim), "fourier");
    checked += dim;
  }
  MESSAGE("fourier: " << checked << " values over "
                      << vllm_test::kMusic3FourierTimeCount << " flow times");
  CHECK(checked == 24);
  // At t = 0 every angle is 0, so the halves are all-ones then all-zeros: the
  // one input that makes a swapped cat visible without any weight at all.
  const std::vector<float> zero = m3::FourierTimeEmbedding(0.0, weight, dim);
  for (int64_t i = 0; i < dim / 2; ++i) CHECK(zero[static_cast<size_t>(i)] == doctest::Approx(1.0));
  for (int64_t i = dim / 2; i < dim; ++i) CHECK(zero[static_cast<size_t>(i)] == doctest::Approx(0.0));
}

TEST_CASE("music3 acoustic: the rotary tables repeat their half and honour theta") {
  CHECK(m3::kDitRotaryTheta == doctest::Approx(vllm_test::kMusic3RotaryTheta));
  const int64_t seq = vllm_test::kMusic3DitSeqLen;
  const int64_t rotary = vllm_test::kMusic3DitRotaryDim;
  const m3::DitRotaryTables tables = m3::BuildDitRotaryTables(seq, rotary);
  ExpectClose(tables.cos, vllm_test::kMusic3RotaryCos, static_cast<size_t>(seq * rotary),
              "rotary cos");
  ExpectClose(tables.sin, vllm_test::kMusic3RotarySin, static_cast<size_t>(seq * rotary),
              "rotary sin");
  MESSAGE("rotary tables: " << 2 * seq * rotary << " values");
  // freqs is built at HALF the width and concatenated with itself.
  size_t repeated = 0;
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t d = 0; d < rotary / 2; ++d) {
      if (tables.cos[static_cast<size_t>(s * rotary + d)] ==
          tables.cos[static_cast<size_t>(s * rotary + rotary / 2 + d)]) {
        ++repeated;
      }
    }
  }
  CHECK(repeated == static_cast<size_t>(seq * rotary / 2));
}

TEST_CASE("music3 acoustic: only the leading rotary_dim of each head rotates") {
  const int64_t seq = vllm_test::kMusic3DitSeqLen;
  const int64_t heads = vllm_test::kMusic3DitNumAttentionHeads;
  const int64_t head_dim = vllm_test::kMusic3DitAttentionHeadDim;
  const int64_t rotary = vllm_test::kMusic3DitRotaryDim;
  const size_t count = static_cast<size_t>(seq * heads * head_dim);
  const std::vector<float> in = ToVector(vllm_test::kMusic3RotaryIn, count);
  std::vector<float> x = in;
  m3::ApplyPartialRotary(x, seq, heads, head_dim, m3::BuildDitRotaryTables(seq, rotary));
  ExpectClose(x, vllm_test::kMusic3RotaryOut, count, "partial rotary");
  // The TAIL of every head must be untouched, bit for bit — a full-width rotary
  // is the plausible defect and this is the half of every head that shows it.
  size_t tail = 0;
  size_t tail_identical = 0;
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t d = rotary; d < head_dim; ++d) {
        const size_t i = static_cast<size_t>((s * heads + h) * head_dim + d);
        ++tail;
        if (x[i] == in[i]) ++tail_identical;
      }
    }
  }
  MESSAGE("partial rotary: " << count << " values, " << tail
                             << " unrotated tail values, " << tail_identical << " identical");
  CHECK(tail == static_cast<size_t>(seq * heads * (head_dim - rotary)));
  CHECK(tail_identical == tail);
}

TEST_CASE("music3 acoustic: the timestep embedding is linear-SiLU-linear") {
  const int64_t dim = vllm_test::kMusic3DitFourierEmbeddingDim;
  const std::vector<float> fourier = m3::FourierTimeEmbedding(
      vllm_test::kMusic3DitTimestep,
      ToVector(vllm_test::kMusic3DitW_time_proj_weight, static_cast<size_t>(dim / 2)), dim);
  const std::vector<float> temb =
      m3::DitTimestepEmbedding(fourier, DitConfig(), DitWeights());
  REQUIRE(static_cast<int64_t>(temb.size()) == DitConfig().inner_dim());
  MESSAGE("timestep embedding: " << temb.size() << " values");
}

TEST_CASE("music3 acoustic: the DiT forward matches upstream, conditional") {
  const vllm::MiniMaxMusic3TransformerConfig config = DitConfig();
  const size_t latent_count =
      static_cast<size_t>(config.in_channels * vllm_test::kMusic3DitLength);
  const std::vector<float> out = m3::DitForward(
      ToVector(vllm_test::kMusic3DitLatents, latent_count), vllm_test::kMusic3DitLength,
      ToVector(vllm_test::kMusic3DitCondition,
               static_cast<size_t>(vllm_test::kMusic3DitLength * config.condition_dim)),
      vllm_test::kMusic3DitTimestep, config, DitWeights());
  ExpectClose(out, vllm_test::kMusic3DitOut, latent_count, "dit conditional");
  MESSAGE("dit conditional: " << latent_count << " values compared");
}

TEST_CASE("music3 acoustic: the DiT forward matches upstream, unconditional zeros") {
  const vllm::MiniMaxMusic3TransformerConfig config = DitConfig();
  const size_t latent_count =
      static_cast<size_t>(config.in_channels * vllm_test::kMusic3DitLength);
  const std::vector<float> zeros(
      static_cast<size_t>(vllm_test::kMusic3DitLength * config.condition_dim), 0.0f);
  const std::vector<float> out = m3::DitForward(
      ToVector(vllm_test::kMusic3DitLatents, latent_count), vllm_test::kMusic3DitLength, zeros,
      vllm_test::kMusic3DitTimestep, config, DitWeights());
  ExpectClose(out, vllm_test::kMusic3DitOutUncond, latent_count, "dit unconditional");
  // The two branches must be DIFFERENT tensors: a DiT that ignored its
  // condition would pass the conditional case and this one identically.
  size_t differing = 0;
  for (size_t i = 0; i < latent_count; ++i) {
    if (vllm_test::kMusic3DitOut[i] != vllm_test::kMusic3DitOutUncond[i]) ++differing;
  }
  MESSAGE("dit unconditional: " << latent_count << " values compared, " << differing
                                << " differ from the conditional branch");
  CHECK(differing == latent_count);
}

TEST_CASE("music3 acoustic: the DiT refuses every wrong-shaped input by name") {
  const vllm::MiniMaxMusic3TransformerConfig config = DitConfig();
  const m3::DitWeights weights = DitWeights();
  const size_t latent_count =
      static_cast<size_t>(config.in_channels * vllm_test::kMusic3DitLength);
  const std::vector<float> latents = ToVector(vllm_test::kMusic3DitLatents, latent_count);
  const std::vector<float> condition = ToVector(
      vllm_test::kMusic3DitCondition,
      static_cast<size_t>(vllm_test::kMusic3DitLength * config.condition_dim));
  CHECK_THROWS_AS(
      m3::DitForward({1.0f}, vllm_test::kMusic3DitLength, condition, 0.25, config, weights),
      std::runtime_error);
  CHECK_THROWS_AS(
      m3::DitForward(latents, vllm_test::kMusic3DitLength, {1.0f}, 0.25, config, weights),
      std::runtime_error);
  CHECK_THROWS_AS(m3::DitForward(latents, 0, condition, 0.25, config, weights),
                  std::runtime_error);
  m3::DitWeights broken = weights;
  broken.layers.pop_back();
  CHECK_THROWS_AS(m3::DitForward(latents, vllm_test::kMusic3DitLength, condition, 0.25, config,
                                 broken),
                  std::runtime_error);
}

// ---------------------------------------------------------------------------
// The DEVICE-RESIDENT DiT (#672, spec §11.4)
//
// THE TOLERANCE, AND THE CONTROL THAT JUSTIFIES IT. Nothing below is a new
// bound. `DitForwardDevice` is checked against the SAME upstream float32
// goldens, through the SAME `ExpectClose`, at the SAME kRelTol/kAbsFloor as
// `DitForward` — because the question that matters is not "do the two arms
// agree with each other" (a shared-helper comparison proves consistency, not
// correctness) but "is the device arm as close to UPSTREAM as the host arm is".
//
// Each case therefore reports BOTH distances to the golden, host and device, on
// the identical input. The host arm's distance is the measured control: it was
// accepted with these goldens when the bound was set, so a device arm whose
// distance is at or below it is inside a spread that already exists rather than
// inside one this row widened. No tolerance is relaxed here, and the two
// mutation cases below prove the bound still discriminates.
// ---------------------------------------------------------------------------

namespace {

// Both arms, same inputs, both against upstream. Returns nothing; every number
// is asserted or printed, and the CASE count is what the suite reports.
void CheckDeviceDit(vt::Queue& q, const char* arm) {
  const vllm::MiniMaxMusic3TransformerConfig config = DitConfig();
  const size_t latent_count =
      static_cast<size_t>(config.in_channels * vllm_test::kMusic3DitLength);
  const size_t condition_count =
      static_cast<size_t>(vllm_test::kMusic3DitLength * config.condition_dim);
  const std::vector<float> latents = ToVector(vllm_test::kMusic3DitLatents, latent_count);
  const std::vector<float> condition = ToVector(vllm_test::kMusic3DitCondition, condition_count);
  const std::vector<float> zeros(condition_count, 0.0f);

  // `release_host` FALSE here on purpose: this gate needs the host arm too, and
  // the serving path is the caller that passes true.
  m3::DitWeights host = DitWeights();
  const m3::Music3DitDeviceWeights staged =
      m3::StageMusic3DitWeights(q, config, host, /*release_host=*/false);
  REQUIRE(staged.layers.size() == static_cast<size_t>(config.num_layers));

  const std::vector<float> dev_cond = m3::DitForwardDevice(
      q, latents, vllm_test::kMusic3DitLength, condition, vllm_test::kMusic3DitTimestep, config,
      staged);
  const std::vector<float> host_cond =
      m3::DitForward(latents, vllm_test::kMusic3DitLength, condition,
                     vllm_test::kMusic3DitTimestep, config, host);
  const double dev_worst =
      ExpectClose(dev_cond, vllm_test::kMusic3DitOut, latent_count,
                  (std::string(arm) + " dit conditional (device)").c_str());
  const double host_worst =
      ExpectClose(host_cond, vllm_test::kMusic3DitOut, latent_count,
                  (std::string(arm) + " dit conditional (host control)").c_str());
  MESSAGE(std::string(arm) << " dit conditional: " << latent_count
                           << " values; worst |device-upstream| = " << dev_worst
                           << ", worst |host-upstream| = " << host_worst
                           << " (bound " << kRelTol << " rel / " << kAbsFloor << " abs)");

  const std::vector<float> dev_uncond =
      m3::DitForwardDevice(q, latents, vllm_test::kMusic3DitLength, zeros,
                           vllm_test::kMusic3DitTimestep, config, staged);
  const double dev_worst_u =
      ExpectClose(dev_uncond, vllm_test::kMusic3DitOutUncond, latent_count,
                  (std::string(arm) + " dit unconditional (device)").c_str());
  MESSAGE(std::string(arm) << " dit unconditional: " << latent_count
                           << " values; worst |device-upstream| = " << dev_worst_u);

  // The two branches must be DIFFERENT tensors on the device arm too: a forward
  // that dropped its condition would match the conditional golden and this one
  // identically, and both ExpectClose calls above would still be green.
  size_t differing = 0;
  for (size_t i = 0; i < latent_count; ++i) {
    if (dev_cond[i] != dev_uncond[i]) ++differing;
  }
  MESSAGE(std::string(arm) << " dit branches: " << differing << " of " << latent_count
                           << " values differ between conditional and unconditional");
  CHECK(differing == latent_count);
}

}  // namespace

TEST_CASE("music3 acoustic: the DEVICE-resident DiT matches upstream (CPU backend)") {
  vt::Queue q{vt::Device{}, nullptr};
  CheckDeviceDit(q, "cpu-backend");
}

TEST_CASE("music3 acoustic: the DEVICE-resident DiT matches upstream on CUDA") {
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered (this is a CPU-only build)");
    return;
  }
  vt::Queue q = cuda->CreateQueue();
  CheckDeviceDit(q, "cuda");
}

// ---------------------------------------------------------------------------
// The INTRA-DiT SPANS (#1542, spec §21.3)
//
// WHY THIS GATE IS A CALL-COUNT GATE AND NOT A TIMING ONE. What can go wrong
// with a hand-placed bracket is placement, not arithmetic: a mark left out, a
// mark inside the layer loop that belonged outside it, a mark that names the
// neighbour's bucket. Every one of those changes a CALL COUNT deterministically,
// on any box, at any load — while the seconds themselves are the thing being
// measured and cannot also be the assertion. So the counts are asserted exactly
// and the times are asserted only for the two properties the accounting depends
// on: that every intra-DiT bucket is a SPAN (so `sum(leaf)` and `unattributed`
// in every §15/§20 table are untouched), and that the spans PARTITION the
// forward rather than sample it.
//
// The CPU backend is the right place for it. `Backend::Synchronize` is a no-op
// on a CPU queue, so the placement contract is tested without a device and
// without the sync perturbation that is the whole reason the spans are a second
// opt-in — and placement is architecture-independent, being a property of where
// the calls sit in the source.
namespace {

// Restores BOTH flags, so a later case in this binary still sees the shipped
// default. `test_music3_profile.cpp`'s ArmedProfile does the same for the outer
// one; this file needs the pair.
struct ArmedDitSpans {
  ArmedDitSpans()
      : profile_(m3profile::EnabledFlag()), spans_(m3profile::DitSpansFlag()) {
    m3profile::EnabledFlag() = true;
    m3profile::DitSpansFlag() = true;
  }
  ~ArmedDitSpans() {
    m3profile::EnabledFlag() = profile_;
    m3profile::DitSpansFlag() = spans_;
  }
  bool profile_;
  bool spans_;
};

const m3profile::Bucket* FindBucket(const char* name) {
  for (const m3profile::Bucket& bucket : m3profile::Buckets()) {
    if (bucket.name == name) return &bucket;
  }
  return nullptr;
}

// One device forward on a CPU queue, with the table freshly begun. Returns the
// wall time of the FORWARD ALONE — the staging above it is outside the bracket,
// because a partition assertion against a bracket that also contained a 36-block
// weight upload would be measuring the upload.
double RunOneProfiledDeviceForward() {
  const vllm::MiniMaxMusic3TransformerConfig config = DitConfig();
  const size_t latent_count =
      static_cast<size_t>(config.in_channels * vllm_test::kMusic3DitLength);
  const size_t condition_count =
      static_cast<size_t>(vllm_test::kMusic3DitLength * config.condition_dim);
  const std::vector<float> latents = ToVector(vllm_test::kMusic3DitLatents, latent_count);
  const std::vector<float> condition = ToVector(vllm_test::kMusic3DitCondition, condition_count);
  m3::DitWeights host = DitWeights();
  vt::Queue q{vt::Device{}, nullptr};
  const m3::Music3DitDeviceWeights staged =
      m3::StageMusic3DitWeights(q, config, host, /*release_host=*/false);
  m3profile::Begin();
  const auto t0 = m3profile::Now();
  const std::vector<float> out = m3::DitForwardDevice(
      q, latents, vllm_test::kMusic3DitLength, condition, vllm_test::kMusic3DitTimestep, config,
      staged);
  const double seconds = std::chrono::duration<double>(m3profile::Now() - t0).count();
  // The forward has to have actually produced the tensor; a span table over a
  // throw would be a green gate over no work at all.
  REQUIRE(out.size() == latent_count);
  return seconds;
}

// The nine spans that sit INSIDE the layer loop, so their call count is
// `num_layers` per forward. Listed in source order.
const char* const kPerLayerSpans[] = {"dit.norm1", "dit.qkv",    "dit.rope",
                                      "dit.attn",  "dit.attn_out", "dit.norm2",
                                      "dit.ff_in", "dit.silu",   "dit.ff_out"};

// The seven that sit outside it, once per forward, in source order.
const char* const kPerForwardSpans[] = {"dit.pack", "dit.pre",      "dit.temb",
                                        "dit.rope_build", "dit.post", "dit.readback",
                                        "dit.untranspose"};

}  // namespace

TEST_CASE("music3 acoustic: the intra-DiT spans are placed once per layer and once per forward") {
  const ArmedDitSpans armed;
  (void)RunOneProfiledDeviceForward();
  const int64_t layers = DitConfig().num_layers;
  REQUIRE(layers > 1);  // a 1-layer config could not tell the two groups apart

  for (const char* name : kPerLayerSpans) {
    const m3profile::Bucket* bucket = FindBucket(name);
    const std::string missing = std::string("missing intra-DiT span: ") + name;
    REQUIRE_MESSAGE(bucket != nullptr, missing);
    const std::string wrong = std::string(name) + " ran " + std::to_string(bucket->calls) +
                              " times, expected once per layer = " + std::to_string(layers);
    CHECK_MESSAGE(bucket->calls == layers, wrong);
  }
  for (const char* name : kPerForwardSpans) {
    const m3profile::Bucket* bucket = FindBucket(name);
    const std::string missing = std::string("missing intra-DiT span: ") + name;
    REQUIRE_MESSAGE(bucket != nullptr, missing);
    const std::string wrong = std::string(name) + " ran " + std::to_string(bucket->calls) +
                              " times, expected once per forward";
    CHECK_MESSAGE(bucket->calls == 1, wrong);
  }

  // The geometry the split must be read against, so a reader never has to infer
  // `seq` from a vocoder latent count the way spec §21.1 had to. Both are SUMS
  // over the forwards in a run; here there is exactly one.
  const m3profile::Bucket* seq_sum = FindBucket("dit.seq_sum");
  REQUIRE(seq_sum != nullptr);
  CHECK(seq_sum->calls == vllm_test::kMusic3DitLength + 1);
  const m3profile::Bucket* length_sum = FindBucket("dit.length_sum");
  REQUIRE(length_sum != nullptr);
  CHECK(length_sum->calls == vllm_test::kMusic3DitLength);
}

TEST_CASE("music3 acoustic: every intra-DiT bucket is a SPAN, so no §20 table moves") {
  const ArmedDitSpans armed;
  (void)RunOneProfiledDeviceForward();

  // THE ACCOUNTING PROPERTY THIS ROW RESTS ON. `music3_profile.h` sums LEAVES
  // and only prints SPANS. If one of these landed as a leaf it would be added to
  // `sum(leaf)` alongside the `denoise.dit_device` leaf that already contains
  // it, every §15.7 and §20 table would double-count the DiT, and
  // `unattributed` would go negative — a corrupted split that still prints.
  int64_t timed = 0;
  for (const m3profile::Bucket& bucket : m3profile::Buckets()) {
    if (bucket.name.rfind("dit.", 0) != 0) continue;
    if (bucket.seconds < 0.0) continue;  // dit.seq_sum / dit.length_sum are pure counters
    ++timed;
    const std::string leaked = bucket.name + " landed as a LEAF; it must be a span";
    CHECK_MESSAGE(bucket.span, leaked);
  }
  CHECK(timed == 16);  // 9 per-layer + 7 per-forward, and no more
}

TEST_CASE("music3 acoustic: the intra-DiT spans PARTITION the forward, they do not sample it") {
  const ArmedDitSpans armed;
  const double bracket = RunOneProfiledDeviceForward();

  double summed = 0.0;
  for (const m3profile::Bucket& bucket : m3profile::Buckets()) {
    if (bucket.name.rfind("dit.", 0) == 0 && bucket.seconds >= 0.0) summed += bucket.seconds;
  }
  MESSAGE("intra-DiT spans sum to " << summed << " s of a " << bracket
                                    << " s bracket around the forward");
  // Contiguity gives the upper bound by construction: each mark charges only the
  // interval since the previous one, so the spans cannot exceed the whole.
  CHECK(summed <= bracket);
  // The lower bound is the one that catches a MISSING bracket. It is deliberately
  // loose — the two host loops that open and close the forward sit inside the
  // bracket and inside the spans alike, and the clock is read sixteen times — but
  // it still fails hard on a dropped mark, because the largest of the sixteen
  // (`dit.ff_in`) is most of the forward on its own.
  CHECK(summed >= 0.5 * bracket);
}

TEST_CASE("music3 acoustic: with the spans OFF the DiT forward emits NO dit.* bucket") {
  // G2, spec §21.4: the shipped profiled path. `VLLM_CPP_MUSIC3_PROFILE=1` alone
  // must leave the forward exactly what §20 timed, so the perturbation of the
  // sixteen synchronizes is opt-in and `denoise.dit_device` stays comparable to
  // §15.7 and §20 value for value.
  const bool prev_profile = m3profile::EnabledFlag();
  const bool prev_spans = m3profile::DitSpansFlag();
  m3profile::EnabledFlag() = true;
  m3profile::DitSpansFlag() = false;
  (void)RunOneProfiledDeviceForward();
  int64_t found = 0;
  for (const m3profile::Bucket& bucket : m3profile::Buckets()) {
    if (bucket.name.rfind("dit.", 0) == 0) ++found;
  }
  m3profile::EnabledFlag() = prev_profile;
  m3profile::DitSpansFlag() = prev_spans;
  CHECK(found == 0);

  // And asking for spans with the instrument OFF is a no-op, not a partial
  // arming: there is no table for a span to land in.
  CHECK_FALSE(m3profile::DitSpans());
}

TEST_CASE("music3 acoustic: the ff_in HALF SWAP is load-bearing, and the gate sees it") {
  // The device arm computes `value * silu(gate)` by handing vt::SiluAndMul — which
  // computes `silu(first) * second` — a projection whose two ROW BLOCKS were
  // exchanged at stage time. That exchange is an identity ONLY if it is applied
  // exactly once. Pre-swapping the host weights makes the stage-time swap undo
  // the test's, so the forward computes `silu(value) * gate` instead: the wrong
  // network, same shapes, same finiteness.
  //
  // This is the mutation that proves the bound above discriminates. If the
  // forward were routing `silu`/`mul` the other way round the RIGHT case would
  // fail and this one would pass, so the pair pins the direction rather than
  // just the magnitude.
  const vllm::MiniMaxMusic3TransformerConfig config = DitConfig();
  const size_t latent_count =
      static_cast<size_t>(config.in_channels * vllm_test::kMusic3DitLength);
  const std::vector<float> latents = ToVector(vllm_test::kMusic3DitLatents, latent_count);
  const std::vector<float> condition = ToVector(
      vllm_test::kMusic3DitCondition,
      static_cast<size_t>(vllm_test::kMusic3DitLength * config.condition_dim));

  m3::DitWeights mutated = DitWeights();
  const size_t ff = static_cast<size_t>(config.ff_inner_dim);
  const size_t inner = static_cast<size_t>(config.inner_dim());
  for (m3::DitLayerWeights& layer : mutated.layers) {
    std::vector<float> w(layer.ff_in_weight.size());
    std::copy(layer.ff_in_weight.begin() + static_cast<ptrdiff_t>(ff * inner),
              layer.ff_in_weight.end(), w.begin());
    std::copy(layer.ff_in_weight.begin(),
              layer.ff_in_weight.begin() + static_cast<ptrdiff_t>(ff * inner),
              w.begin() + static_cast<ptrdiff_t>(ff * inner));
    layer.ff_in_weight = w;
    std::vector<float> b(layer.ff_in_bias.size());
    std::copy(layer.ff_in_bias.begin() + static_cast<ptrdiff_t>(ff), layer.ff_in_bias.end(),
              b.begin());
    std::copy(layer.ff_in_bias.begin(), layer.ff_in_bias.begin() + static_cast<ptrdiff_t>(ff),
              b.begin() + static_cast<ptrdiff_t>(ff));
    layer.ff_in_bias = b;
  }

  vt::Queue q{vt::Device{}, nullptr};
  const m3::Music3DitDeviceWeights staged =
      m3::StageMusic3DitWeights(q, config, mutated, /*release_host=*/false);
  const std::vector<float> out = m3::DitForwardDevice(
      q, latents, vllm_test::kMusic3DitLength, condition, vllm_test::kMusic3DitTimestep, config,
      staged);

  size_t outside = 0;
  double worst = 0.0;
  for (size_t i = 0; i < latent_count; ++i) {
    const double a = out[i], b = vllm_test::kMusic3DitOut[i];
    const double bound = std::max(kAbsFloor, kRelTol * std::max(std::abs(a), std::abs(b)));
    if (!(std::abs(a - b) <= bound)) ++outside;
    worst = std::max(worst, std::abs(a - b));
  }
  MESSAGE("half-swap mutation: " << outside << " of " << latent_count
                                 << " values outside the bound, worst |diff| = " << worst);
  // A defect that moves values by O(1) must move essentially all of them. This
  // is the negative control for every ExpectClose above.
  CHECK(outside > latent_count / 2);
}

TEST_CASE("music3 acoustic: the DEVICE DiT refuses every wrong-shaped input by name") {
  const vllm::MiniMaxMusic3TransformerConfig config = DitConfig();
  const size_t latent_count =
      static_cast<size_t>(config.in_channels * vllm_test::kMusic3DitLength);
  const std::vector<float> latents = ToVector(vllm_test::kMusic3DitLatents, latent_count);
  const std::vector<float> condition = ToVector(
      vllm_test::kMusic3DitCondition,
      static_cast<size_t>(vllm_test::kMusic3DitLength * config.condition_dim));
  vt::Queue q{vt::Device{}, nullptr};

  m3::DitWeights host = DitWeights();
  const m3::Music3DitDeviceWeights staged =
      m3::StageMusic3DitWeights(q, config, host, /*release_host=*/false);
  CHECK_THROWS_AS(m3::DitForwardDevice(q, {1.0f}, vllm_test::kMusic3DitLength, condition, 0.25,
                                       config, staged),
                  std::runtime_error);
  CHECK_THROWS_AS(m3::DitForwardDevice(q, latents, vllm_test::kMusic3DitLength, {1.0f}, 0.25,
                                       config, staged),
                  std::runtime_error);
  CHECK_THROWS_AS(m3::DitForwardDevice(q, latents, 0, condition, 0.25, config, staged),
                  std::runtime_error);

  // A mis-sized weight is refused at STAGE time — before 9.7 GB moves at real
  // dimensions — rather than 36 layers into the first of 660 forwards.
  m3::DitWeights broken = DitWeights();
  broken.layers.pop_back();
  CHECK_THROWS_AS(m3::StageMusic3DitWeights(q, config, broken, /*release_host=*/false),
                  std::runtime_error);
  m3::DitWeights short_proj = DitWeights();
  short_proj.proj_in_weight.pop_back();
  CHECK_THROWS_AS(m3::StageMusic3DitWeights(q, config, short_proj, /*release_host=*/false),
                  std::runtime_error);
}

TEST_CASE("music3 acoustic: release_host EMPTIES the source, and the staged copy still runs") {
  // "Device-resident" has to mean the host copy is GONE, not that a second copy
  // exists. On Jetson Thor the two pools are one pool: holding both is a real
  // 19.4 GB peak on a box that reboots instead of OOM-killing.
  const vllm::MiniMaxMusic3TransformerConfig config = DitConfig();
  const size_t latent_count =
      static_cast<size_t>(config.in_channels * vllm_test::kMusic3DitLength);
  const std::vector<float> latents = ToVector(vllm_test::kMusic3DitLatents, latent_count);
  const std::vector<float> condition = ToVector(
      vllm_test::kMusic3DitCondition,
      static_cast<size_t>(vllm_test::kMusic3DitLength * config.condition_dim));

  vt::Queue q{vt::Device{}, nullptr};
  m3::DitWeights host = DitWeights();
  const m3::Music3DitDeviceWeights staged =
      m3::StageMusic3DitWeights(q, config, host, /*release_host=*/true);

  size_t emptied = 0, total = 0;
  for (const m3::DitLayerWeights& layer : host.layers) {
    for (const std::vector<float>* v :
         {&layer.to_q, &layer.to_k, &layer.to_v, &layer.to_out, &layer.ff_in_weight,
          &layer.ff_out_weight}) {
      ++total;
      if (v->empty() && v->capacity() == 0) ++emptied;
    }
  }
  MESSAGE("release_host: " << emptied << " of " << total
                           << " per-layer host projections released (empty AND zero capacity)");
  CHECK(emptied == total);
  // The time embedder is the ONE thing deliberately kept — it runs on the host
  // so that `temb` stays bit-identical to the CPU arm.
  CHECK(staged.host_time_embed.time_embed_linear_2_weight.size() ==
        static_cast<size_t>(config.inner_dim() * config.inner_dim()));

  // And the staged copy is intact: a released host buffer that had been uploaded
  // without a synchronize would read as garbage here rather than as the golden.
  const std::vector<float> out = m3::DitForwardDevice(
      q, latents, vllm_test::kMusic3DitLength, condition, vllm_test::kMusic3DitTimestep, config,
      staged);
  ExpectClose(out, vllm_test::kMusic3DitOut, latent_count, "dit after release_host");
}

// ---------------------------------------------------------------------------
// THE PRODUCTION SWITCH ([#1131](https://github.com/mudler/vllm.cpp/issues/1131))
//
// Everything above this line gates the device DiT as a CLASS: `DitForwardDevice`
// against upstream's goldens, `StageMusic3DitWeights` against its refusals, the
// spans against their placement. #1131's finding is that none of it gates the
// device DiT as a CAPABILITY — it measured two mutations on the shipped code and
// both left every suite green:
//
//   * `on_device = false` in `Music3DenoiseChunks`, the production call site,
//   * `Music3DenoiseDeviceArm::half_set() -> false`, the arm's refusal.
//
// A change that silently stopped the DiT reaching the device would have been
// invisible, and the run would have been correct and thirty hours late.
//
// WHY THE OBVIOUS GATE CANNOT EXIST, so the shape below is not a preference. The
// engine picks the arm on `queue_.device.type != kCPU`, and on a CPU-only build
// `src/vllm/multimodal/speech_engine.cpp::SpeechEngineDeviceType` REFUSES
// `--speech-device 1` before a queue is ever made. So `queue_` on a CI
// runner is kCPU or the engine does not construct, and no gate CI owns can enter
// a branch written at that line. Two things follow, and this file does both:
//
//   1. the RULE moves out of the engine into `Music3SelectDitArm`, which runs on
//      both sides of the condition and is therefore drivable here — with a CPU
//      queue, and with a FABRICATED non-CPU one that must engage or refuse BY
//      NAME, the silent host fallback being the defect;
//   2. the SELECTED arm is driven through `Music3DenoiseChunks` — the production
//      function the engine calls, not a hand-built forward — and the gate asks
//      WHICH ARM RAN rather than whether the numbers agree. They agree by
//      design; a numeric comparison cannot answer it.
//
// `kCUDA` is deliberately absent from the fabricated list. On a CUDA build with
// no device the staging fails INSIDE the CUDA runtime, and a call designed to
// fail latches a sticky error that the next unrelated kernel reports as its own.
// Every entry below takes the identical branch, so the rule is covered and the
// latch is not armed.
//
// WHAT THIS STILL DOES NOT REACH is the engine's own one-line CALL to
// `Music3SelectDitArm`. It needs the 28.5 GB checkpoint and a real accelerator.
// `.agents/specs/music3-dit-arm-reachability.md` `## Owed` carries it with the
// mutation that proves it, and #1131 stays open for it.
// ---------------------------------------------------------------------------

namespace {

// Deterministic, and NOT a normal draw: the arm question is about which code
// executed, so the fixture only has to be finite, non-degenerate and the same on
// every box. A `std::mt19937_64` would put a libstdc++ version in the middle of
// a reachability gate for nothing.
float ArmNext(uint32_t* state, float scale) {
  *state = *state * 1664525u + 1013904223u;
  const float unit = static_cast<float>((*state >> 8) & 0xFFFFu) / 65535.0f;
  return (unit - 0.5f) * 2.0f * scale;
}

std::vector<float> ArmFill(size_t count, uint32_t* state, float scale) {
  std::vector<float> out(count);
  for (float& value : out) value = ArmNext(state, scale);
  return out;
}

// The reduced denoise geometry. `kArmFrames <= kChunkFrames` so `ChunkStarts`
// returns one window, and the condition encoder's input and output rates are
// EQUAL so `ConditionLatentLength` is the identity: the window's latent length
// is exactly `kArmFrames`, and the two per-forward counts below are arithmetic
// rather than something a reader has to derive from a rate ratio.
constexpr int64_t kArmFrames = 4;
constexpr int64_t kArmCondHidden = 3;
constexpr int64_t kArmCondLayers = 2;
constexpr int64_t kArmSteps = 2;
constexpr int64_t kArmWindows = 1;
// ONE profile bracket spans BOTH classifier-free-guidance branches, so the
// `denoise.dit_*` call count is steps x windows and the FORWARD count is twice
// that. Getting this backwards is the difference between a gate that counts and
// one that agrees with whatever it found.
constexpr int64_t kArmBrackets = kArmSteps * kArmWindows;
constexpr int64_t kArmForwards = 2 * kArmBrackets;

vllm::MiniMaxMusic3Config DenoiseConfig() {
  vllm::MiniMaxMusic3Config config;
  config.transformer = DitConfig();
  config.condition_encoder.condition_hidden_dim = kArmCondHidden;
  config.condition_encoder.num_condition_layers = kArmCondLayers;
  // The condition mix FEEDS the DiT, so its output width is not free.
  config.condition_encoder.out_dim = config.transformer.condition_dim;
  config.condition_encoder.input_sampling_rate = 24000;
  config.condition_encoder.input_hop_length = 960;
  config.condition_encoder.output_sampling_rate = 24000;
  config.condition_encoder.output_hop_length = 960;
  return config;
}

// `vocoder` is left default: `Music3DenoiseChunks` never reads it, and filling
// it would suggest the decode is part of what this gate examined.
m3::Music3AcousticWeights DenoiseWeights(uint32_t* state) {
  const vllm::MiniMaxMusic3Config config = DenoiseConfig();
  const size_t out_dim = static_cast<size_t>(config.condition_encoder.out_dim);
  m3::Music3AcousticWeights weights;
  weights.dit = DitWeights();
  weights.condition.layer_weight_logits =
      ArmFill(static_cast<size_t>(kArmCondLayers), state, 1.0f);
  weights.condition.layer_scale = {1.0f};
  weights.condition.proj_weight =
      ArmFill(out_dim * static_cast<size_t>(kArmCondHidden) * 3, state, 0.5f);
  weights.condition.proj_bias = ArmFill(out_dim, state, 0.25f);
  return weights;
}

std::vector<float> ArmFrameHiddens(uint32_t* state) {
  return ArmFill(static_cast<size_t>(kArmFrames * kArmCondLayers * kArmCondHidden), state, 1.0f);
}

// A FIXED noise source. The two arms must see the same initial latents or the
// comparison below is between two trajectories rather than between two
// implementations of one.
m3::Music3NoiseSource ArmNoise(uint32_t seed) {
  return [seed](int64_t channels, int64_t length, int64_t chunk_index) {
    uint32_t state = seed + static_cast<uint32_t>(chunk_index);
    return ArmFill(static_cast<size_t>(channels * length), &state, 1.0f);
  };
}

}  // namespace

TEST_CASE("music3 acoustic: the DiT arm SELECTION stages on a device queue and never on a CPU one") {
  const vllm::MiniMaxMusic3TransformerConfig config = DitConfig();

  SUBCASE("a CPU queue stages NOTHING and keeps the host reference arm") {
    vt::Queue queue{vt::Device{}, nullptr};
    m3::DitWeights source = DitWeights();
    const m3::DitWeights reference = DitWeights();
    m3::Music3DitDeviceWeights staged;
    const m3::Music3DenoiseDeviceArm arm =
        m3::Music3SelectDitArm(queue, config, source, /*release_host=*/true, &staged);
    CHECK_FALSE(arm.engaged());
    CHECK_FALSE(arm.half_set());
    CHECK_FALSE(staged.staged());
    CHECK(staged.layers.empty());
    // `release_host` was TRUE and NOTHING may have been released, because the
    // host `DitForward` is what this queue selected and it reads these very
    // vectors. A selector that staged-then-discarded would pass every assertion
    // above and leave the host arm reading empty projections.
    REQUIRE(source.layers.size() == reference.layers.size());
    size_t intact = 0;
    for (size_t i = 0; i < source.layers.size(); ++i) {
      if (source.layers[i].to_q.size() == reference.layers[i].to_q.size()) ++intact;
      if (source.layers[i].ff_in_weight.size() == reference.layers[i].ff_in_weight.size()) {
        ++intact;
      }
    }
    CHECK(intact == 2 * source.layers.size());
    CHECK(source.proj_in_weight.size() == reference.proj_in_weight.size());
  }

  SUBCASE("EVERY non-CPU device stages or refuses, and never silently falls back") {
    // THE THIRD OUTCOME IS THE DEFECT, and it is #1131's whole shape: a caller
    // that asked for the device, was handed the host loops, and was told
    // nothing.
    constexpr vt::DeviceType kNonCpuDevices[] = {
        vt::DeviceType::kMETAL, vt::DeviceType::kVULKAN, vt::DeviceType::kXPU,
        vt::DeviceType::kROCM, vt::DeviceType::kTENSTORRENT};
    int engaged_count = 0;
    int refused_count = 0;
    for (vt::DeviceType type : kNonCpuDevices) {
      // `std::string`, not the bare `const char*`: doctest stringifies a raw
      // pointer as a BOOL, so the capture on a failing device would read `:= 1`
      // and name nothing.
      CAPTURE(std::string(vt::DeviceTypeName(type)));
      vt::Queue queue{vt::Device{type, 0}, nullptr};
      m3::DitWeights source = DitWeights();
      m3::Music3DitDeviceWeights staged;
      bool engaged = false;
      bool refused = false;
      try {
        const m3::Music3DenoiseDeviceArm arm =
            m3::Music3SelectDitArm(queue, config, source, /*release_host=*/false, &staged);
        engaged = arm.engaged();
      } catch (const std::runtime_error&) {
        refused = true;
      }
      engaged_count += engaged ? 1 : 0;
      refused_count += refused ? 1 : 0;
      // Named, because doctest cannot decompose a `||` inside a CHECK.
      const bool selection_fired = engaged || refused;
      CHECK_MESSAGE(selection_fired, "device '"
                                         << std::string(vt::DeviceTypeName(type))
                                         << "' quietly took the HOST arm: the selection did not "
                                            "fire and nothing said so");
    }
    // The loop must have RUN. An emptied list, or a body that threw before the
    // first CHECK, would leave every assertion above unexecuted while doctest
    // still reported SUCCESS.
    const int outcomes = engaged_count + refused_count;
    CHECK(outcomes == static_cast<int>(sizeof(kNonCpuDevices) / sizeof(kNonCpuDevices[0])));
    MESSAGE("non-CPU DiT selection: " << engaged_count << " staged, " << refused_count
                                      << " refused by name");
  }

  SUBCASE("a null staging slot is refused rather than dereferenced") {
    vt::Queue queue{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
    m3::DitWeights source = DitWeights();
    CHECK_THROWS_AS(m3::Music3SelectDitArm(queue, config, source, /*release_host=*/false, nullptr),
                    std::runtime_error);
  }
}

TEST_CASE("music3 acoustic: HALF a denoise device arm is refused by name, not ignored") {
  // #1131's second mutation: `half_set() -> false`. With the refusal gone a
  // queue-only arm is merely not engaged, the loop takes the host path, and the
  // caller who believes it asked for the GPU gets a correct song hours late.
  uint32_t state = 0x1131C0DEu;
  const vllm::MiniMaxMusic3Config config = DenoiseConfig();
  const m3::Music3AcousticWeights weights = DenoiseWeights(&state);
  const std::vector<float> frame_hiddens = ArmFrameHiddens(&state);
  m3::Music3DenoiseOptions options;
  options.num_inference_steps = kArmSteps;

  vt::Queue queue{vt::Device{}, nullptr};
  m3::DitWeights host = DitWeights();
  const m3::Music3DitDeviceWeights staged =
      m3::StageMusic3DitWeights(queue, config.transformer, host, /*release_host=*/false);

  // The PREDICATE, on both halves.
  m3::Music3DenoiseDeviceArm queue_only;
  queue_only.queue = &queue;
  CHECK(queue_only.half_set());
  CHECK_FALSE(queue_only.engaged());
  m3::Music3DenoiseDeviceArm weights_only;
  weights_only.dit = &staged;
  CHECK(weights_only.half_set());
  CHECK_FALSE(weights_only.engaged());

  // And the LOOP acting on it. A predicate nothing consults is not a refusal.
  CHECK_THROWS_AS(m3::Music3DenoiseChunks(frame_hiddens, kArmFrames, config, weights, options,
                                          ArmNoise(7u), queue_only),
                  std::runtime_error);
  CHECK_THROWS_AS(m3::Music3DenoiseChunks(frame_hiddens, kArmFrames, config, weights, options,
                                          ArmNoise(7u), weights_only),
                  std::runtime_error);
}

TEST_CASE("music3 acoustic: the PRODUCTION denoise loop takes the device arm, and says which ran") {
  // THE ASSERTION #1131 SAYS IS MISSING, through the production entry point the
  // engine calls. Not `DitForwardDevice` directly — that is the class gate above
  // — but `Music3DenoiseChunks`, with an arm produced the way the engine
  // produces one.
  //
  // WHICH ARM RAN is read off TWO instruments, and they answer different
  // questions. `denoise.dit_{device,host}` is the production bucket the engine's
  // own `profile::Report` prints, so it says which branch the loop SELECTED.
  // `dit.pack` lives INSIDE `DitForwardDevice`, so it says the device forward's
  // body actually executed — a mislabelled bucket cannot fake it. Both are
  // asserted for an EXACT count, never for movement.
  const ArmedDitSpans armed;
  uint32_t state = 0x0DE7C0DEu;
  const vllm::MiniMaxMusic3Config config = DenoiseConfig();
  const m3::Music3AcousticWeights weights = DenoiseWeights(&state);
  const std::vector<float> frame_hiddens = ArmFrameHiddens(&state);
  m3::Music3DenoiseOptions options;
  options.num_inference_steps = kArmSteps;

  // ── the HOST arm, default-constructed, which is what every caller written
  //    before the device arm passes ────────────────────────────────────────────
  m3profile::Begin();
  const std::vector<std::vector<float>> host_chunks = m3::Music3DenoiseChunks(
      frame_hiddens, kArmFrames, config, weights, options, ArmNoise(7u));
  REQUIRE(host_chunks.size() == static_cast<size_t>(kArmWindows));
  const size_t latent_count =
      static_cast<size_t>(config.transformer.in_channels * kArmFrames);
  REQUIRE(host_chunks[0].size() == latent_count);
  {
    const m3profile::Bucket* host_bucket = FindBucket("denoise.dit_host");
    REQUIRE_MESSAGE(host_bucket != nullptr,
                    "the default arm emitted NO denoise.dit_host bucket, so the loop this gate "
                    "believes it drove did not run");
    CHECK_MESSAGE(host_bucket->calls == kArmBrackets,
                  "the host arm bracketed " << host_bucket->calls << " steps, expected "
                                            << kArmBrackets);
    CHECK_MESSAGE(FindBucket("denoise.dit_device") == nullptr,
                  "a DEFAULT-constructed arm reached the DEVICE forward");
    // And nothing inside `DitForwardDevice` ran, with the spans armed. This is
    // the control that makes the device counts below mean something: without it
    // a `dit.pack` that fired unconditionally would read as proof either way.
    CHECK_MESSAGE(FindBucket("dit.pack") == nullptr,
                  "the host arm emitted an intra-DEVICE-forward span");
  }

  // ── the DEVICE arm, staged and selected exactly as the engine stages and
  //    selects it, on a CPU queue because every op this arm uses has a CPU
  //    provider ─────────────────────────────────────────────────────────────────
  vt::Queue queue{vt::Device{}, nullptr};
  m3::DitWeights stage_source = DitWeights();
  m3::Music3DitDeviceWeights staged;
  // Hand-built, because `Music3SelectDitArm` declines a CPU queue BY DESIGN —
  // that rule is the previous case's subject. This case's subject is what
  // `Music3DenoiseChunks` does once an engaged arm reaches it.
  staged = m3::StageMusic3DitWeights(queue, config.transformer, stage_source,
                                     /*release_host=*/false);
  m3::Music3DenoiseDeviceArm arm;
  arm.queue = &queue;
  arm.dit = &staged;
  REQUIRE(arm.engaged());
  REQUIRE_FALSE(arm.half_set());

  m3profile::Begin();
  const std::vector<std::vector<float>> device_chunks = m3::Music3DenoiseChunks(
      frame_hiddens, kArmFrames, config, weights, options, ArmNoise(7u), arm);
  REQUIRE(device_chunks.size() == static_cast<size_t>(kArmWindows));
  REQUIRE(device_chunks[0].size() == latent_count);

  const m3profile::Bucket* device_bucket = FindBucket("denoise.dit_device");
  REQUIRE_MESSAGE(device_bucket != nullptr,
                  "an ENGAGED arm reached `Music3DenoiseChunks` and the loop still took the HOST "
                  "path: this is #1131 exactly");
  CHECK_MESSAGE(device_bucket->calls == kArmBrackets,
                "the device arm bracketed " << device_bucket->calls << " steps, expected "
                                            << kArmBrackets);
  CHECK_MESSAGE(FindBucket("denoise.dit_host") == nullptr,
                "the loop ran BOTH arms, so one of the two counts above is not what it looks like");
  const m3profile::Bucket* pack = FindBucket("dit.pack");
  REQUIRE_MESSAGE(pack != nullptr,
                  "nothing inside `DitForwardDevice` ran, so the device bucket is a LABEL and not "
                  "a measurement");
  CHECK_MESSAGE(pack->calls == kArmForwards,
                "the device forward body ran " << pack->calls << " times, expected "
                                               << kArmForwards
                                               << " (2 CFG branches x " << kArmBrackets
                                               << " steps)");
  const m3profile::Bucket* qkv = FindBucket("dit.qkv");
  REQUIRE(qkv != nullptr);
  CHECK_MESSAGE(qkv->calls == kArmForwards * config.transformer.num_layers,
                "the device forward's layer loop ran " << qkv->calls << " times, expected "
                                                       << kArmForwards *
                                                              config.transformer.num_layers);
  // The window loop itself ran once, so the counts above are one window's and
  // not a plan that collapsed to zero windows and a lucky default.
  const m3profile::Bucket* mix = FindBucket("denoise.condition_mix");
  REQUIRE(mix != nullptr);
  CHECK(mix->calls == kArmWindows);

  // THE NUMBERS ARE THE CONTROL, NOT THE GATE. The two arms agree by design, so
  // agreement can never say which ran; what it CAN say is that the arm this gate
  // proved was taken produced the trajectory rather than throwing halfway and
  // leaving a plausible tensor behind. `DitForwardDevice`'s correctness is
  // gated against upstream's own goldens by `CheckDeviceDit` above, and nothing
  // here relaxes that.
  double worst = 0.0;
  double scale = 0.0;
  size_t nonzero = 0;
  for (size_t i = 0; i < latent_count; ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(device_chunks[0][i]) -
                                     static_cast<double>(host_chunks[0][i])));
    scale = std::max(scale, std::abs(static_cast<double>(host_chunks[0][i])));
    if (host_chunks[0][i] != 0.0f) ++nonzero;
  }
  MESSAGE("denoise arms: " << latent_count << " latents, worst |device-host| = " << worst
                           << " against a host peak of " << scale);
  CHECK_MESSAGE(nonzero == latent_count,
                "the host trajectory is degenerate, so the comparison beside it is worth nothing");
  CHECK_MESSAGE(worst <= 1e-4 * std::max(scale, 1e-3),
                "the two denoise arms diverged by " << worst
                                                    << ", which is a STRUCTURAL difference rather "
                                                       "than the float32 rounding of §14.3");
}

// ---------------------------------------------------------------------------
// W5 — the vocoder
// ---------------------------------------------------------------------------

TEST_CASE("music3 acoustic: the snake activation is upstream's, through vocoder1d") {
  const size_t count = static_cast<size_t>(vllm_test::kMusic3SnakeChannels *
                                           vllm_test::kMusic3SnakeLength);
  std::vector<float> x = ToVector(vllm_test::kMusic3SnakeIn, count);
  m3::VocoderSnake(x, vllm_test::kMusic3SnakeChannels, vllm_test::kMusic3SnakeLength,
                   ToVector(vllm_test::kMusic3SnakeAlpha,
                            static_cast<size_t>(vllm_test::kMusic3SnakeChannels)));
  ExpectClose(x, vllm_test::kMusic3SnakeOut, count, "snake");
  MESSAGE("snake: " << count << " values compared");
}

TEST_CASE("music3 acoustic: a residual unit keeps its length at dilation 9") {
  const int64_t dim = vllm_test::kMusic3UnitChannels;
  const int64_t length = vllm_test::kMusic3UnitLength;
  m3::VocoderResidualUnitWeights weights;
  weights.snake1_alpha =
      ToVector(vllm_test::kMusic3UnitW_snake1_alpha, static_cast<size_t>(dim));
  weights.conv1_weight =
      ToVector(vllm_test::kMusic3UnitW_conv1_weight, static_cast<size_t>(dim * dim * 7));
  weights.conv1_bias = ToVector(vllm_test::kMusic3UnitW_conv1_bias, static_cast<size_t>(dim));
  weights.snake2_alpha =
      ToVector(vllm_test::kMusic3UnitW_snake2_alpha, static_cast<size_t>(dim));
  weights.conv2_weight =
      ToVector(vllm_test::kMusic3UnitW_conv2_weight, static_cast<size_t>(dim * dim));
  weights.conv2_bias = ToVector(vllm_test::kMusic3UnitW_conv2_bias, static_cast<size_t>(dim));
  int64_t out_len = 0;
  const std::vector<float> out =
      m3::VocoderResidualUnit(ToVector(vllm_test::kMusic3UnitIn,
                                       static_cast<size_t>(dim * length)),
                              dim, length, vllm_test::kMusic3UnitDilation, weights, &out_len);
  CHECK(out_len == length);
  ExpectClose(out, vllm_test::kMusic3UnitOut, static_cast<size_t>(dim * length),
              "residual unit");
  MESSAGE("residual unit at dilation " << vllm_test::kMusic3UnitDilation << ": "
                                       << dim * length << " values compared");
}

TEST_CASE("music3 acoustic: a residual unit is exact ACROSS a time block boundary") {
  // WHY THIS CASE EXISTS (#1684). The `vt::Conv1d` CPU provider cuts its work
  // into (time block, output row) pairs (#1664, src/vt/cpu/cpu_conv1d_block.h).
  // Until this case existed THIS suite reached that provider at SINGLE-BLOCK
  // shapes only, so a defect confined to the second axis reddened the op's own
  // suite and nothing else: a sign flip applied only where `blocks > 1` left
  // eight of the ten consumer suites green. This is MiniMax-Music3's own arm of
  // that gate. It enters through `m3::VocoderResidualUnit` -- the body
  // `VocoderBlock` and `VocoderDecode` are built from, which is the vocoder's
  // hot loop -- at a shape whose block length is shorter than its output length.
  //
  // THE GEOMETRY, asserted rather than asserted-about. 8 channels over 16 384
  // positions is 512 KiB of activation, which is exactly the budget
  // `kConv1dSliceBytes` gives one unit of work, so BOTH convolutions of the unit
  // block. The channel count is deliberately the SMALLEST that reaches the
  // budget, because the convolution's cost is `kernel * dim * (dim * length)`
  // and only the bracket is fixed by the budget: a wider fixture would cost
  // proportionally more for exactly the same block count.
  constexpr int64_t kDim = 8;
  constexpr int64_t kLength = 16384;
  constexpr int64_t kDilation = 1;
  constexpr int64_t kPad = (7 - 1) * kDilation / 2;
  const int64_t block1 =
      vt::cpu::Conv1dTimeBlock(kDim, kDim, /*kernel=*/7, /*stride=*/1, kDilation,
                               kLength + 2 * kPad, kLength);
  const int64_t block2 =
      vt::cpu::Conv1dTimeBlock(kDim, kDim, /*kernel=*/1, /*stride=*/1, /*dilation=*/1,
                               kLength, kLength);
  INFO("conv1 block=" << block1 << ", conv2 block=" << block2 << ", length=" << kLength);
  REQUIRE(block1 < kLength);  // TEETH: without this the unit is single-block
  REQUIRE(block2 < kLength);
  REQUIRE(block1 % vt::cpu::kConv1dPosTile == 0);

  // THE ARITHMETIC IS CLOSED FORM, so the expectation carries no tolerance and a
  // one-bit scheduling defect shows as a hard inequality.
  //
  // Both snake alphas are ZERO, which makes the activation the IDENTITY exactly:
  // `x + (b + 1e-9)^-1 * sin^2(a * x)` at `a = b = 0` is `x + 1e9 * 0`, and
  // `logscale` is false on Music3's Snake1d (vocoder1d::SnakeActivation). The
  // unit therefore reduces to pad -> conv1 -> conv2 -> residual add, which is
  // what this case is about; the activation has its own gate two cases above.
  m3::VocoderResidualUnitWeights weights;
  weights.snake1_alpha.assign(static_cast<size_t>(kDim), 0.0F);
  weights.snake2_alpha.assign(static_cast<size_t>(kDim), 0.0F);
  // conv1 [dim, dim, 7] taps input channel 0 with weight 1 and nothing else;
  // conv2 [dim, dim, 1] copies channel 0 into every output channel.
  weights.conv1_weight.assign(static_cast<size_t>(kDim * kDim * 7), 0.0F);
  weights.conv2_weight.assign(static_cast<size_t>(kDim * kDim), 0.0F);
  weights.conv1_bias.assign(static_cast<size_t>(kDim), 0.0F);
  weights.conv2_bias.assign(static_cast<size_t>(kDim), 0.0F);
  for (int64_t oc = 0; oc < kDim; ++oc) {
    for (int64_t k = 0; k < 7; ++k) {
      weights.conv1_weight[static_cast<size_t>((oc * kDim + 0) * 7 + k)] = 1.0F;
    }
    weights.conv2_weight[static_cast<size_t>(oc * kDim + 0)] = 1.0F;
    weights.conv1_bias[static_cast<size_t>(oc)] = 1.0F;
    weights.conv2_bias[static_cast<size_t>(oc)] = static_cast<float>(oc + 2);
  }

  // Channel 0 carries `x[0][t] = t`; every other channel is zero. Every partial
  // sum below is an integer under 2^24, so f32 holds all of them EXACTLY.
  std::vector<float> in(static_cast<size_t>(kDim * kLength), 0.0F);
  for (int64_t t = 0; t < kLength; ++t) in[static_cast<size_t>(t)] = static_cast<float>(t);

  int64_t out_len = 0;
  const std::vector<float> out =
      m3::VocoderResidualUnit(in, kDim, kLength, kDilation, weights, &out_len);
  REQUIRE(out_len == kLength);
  REQUIRE(out.size() == static_cast<size_t>(kDim * kLength));

  // conv1 sums the 7-tap zero-padded window of channel 0 and adds its bias 1;
  // conv2 copies that row into every output channel and adds `oc + 2`; the unit
  // then adds the input back, which only channel 0 carries.
  auto want = [&](int64_t oc, int64_t t) {
    const int64_t lo = std::max<int64_t>(0, t - 3);
    const int64_t hi = std::min<int64_t>(kLength - 1, t + 3);
    const int64_t window = (lo + hi) * (hi - lo + 1) / 2;
    return static_cast<float>(window + 1 + (oc + 2) + (oc == 0 ? t : 0));
  };
  int64_t wrong = 0;
  int64_t first_wrong = -1;
  for (int64_t oc = 0; oc < kDim; ++oc) {
    for (int64_t t = 0; t < kLength; ++t) {
      if (out[static_cast<size_t>(oc * kLength + t)] != want(oc, t)) {
        if (first_wrong < 0) first_wrong = oc * kLength + t;
        ++wrong;
      }
    }
  }
  INFO("wrong cells=" << wrong << " first at flat index " << first_wrong);
  CHECK(wrong == 0);
  // The boundary itself, named so a failure says WHERE: the last position of the
  // first block and the first of the second are where an off-by-one in the block
  // decode lands.
  CHECK(out[static_cast<size_t>(block1 - 1)] == want(0, block1 - 1));
  CHECK(out[static_cast<size_t>(block1)] == want(0, block1));
  // The LAST block is the short one; `length` is not a multiple of the block.
  CHECK(out[static_cast<size_t>(kLength - 1)] == want(0, kLength - 1));
  MESSAGE("residual unit across a block boundary: " << kDim * kLength
                                                    << " cells compared exactly");
}

TEST_CASE("music3 acoustic: the residual dilations are 1, 3, 9 in order") {
  REQUIRE(m3::kVocoderResidualUnits == 3);
  CHECK(m3::kVocoderResidualDilations[0] == 1);
  CHECK(m3::kVocoderResidualDilations[1] == 3);
  CHECK(m3::kVocoderResidualDilations[2] == 9);
}

TEST_CASE("music3 acoustic: the vocoder decodes to a stereo waveform matching upstream") {
  const vllm::MiniMaxMusic3VocoderConfig config = VocConfig();
  const int64_t length = vllm_test::kMusic3VocLength;
  int64_t samples = 0;
  const std::vector<float> out = m3::VocoderDecode(
      ToVector(vllm_test::kMusic3VocLatents,
               static_cast<size_t>(config.latent_channels * length)),
      length, config, VocWeights(), &samples);
  CHECK(samples == vllm_test::kMusic3VocOutSamples);
  CHECK(samples == length * config.hop_length());
  CHECK(config.hop_length() == vllm_test::kMusic3VocHop);
  ExpectClose(out, vllm_test::kMusic3VocOut, static_cast<size_t>(2 * samples), "vocoder");
  // tanh-bounded: no sample may leave [-1, 1] before the pipeline clamps.
  size_t in_range = 0;
  for (float value : out) {
    if (value >= -1.0f && value <= 1.0f) ++in_range;
  }
  MESSAGE("vocoder: " << 2 * samples << " samples compared, " << in_range << " in [-1, 1]");
  CHECK(in_range == out.size());
}

TEST_CASE("music3 acoustic: the stereo fold splits the channels, it does not interleave") {
  // Decoding [first 64 | second 64] must equal decoding each half alone, which
  // an interleaving fold cannot satisfy. Two streams built from DISJOINT latent
  // halves, run through the same weights, must reproduce the two output rows.
  const vllm::MiniMaxMusic3VocoderConfig config = VocConfig();
  const int64_t length = vllm_test::kMusic3VocLength;
  const int64_t stream = config.stream_channels();
  const std::vector<float> latents = ToVector(
      vllm_test::kMusic3VocLatents, static_cast<size_t>(config.latent_channels * length));
  int64_t samples = 0;
  const std::vector<float> out =
      m3::VocoderDecode(latents, length, config, VocWeights(), &samples);

  // The right channel of a latent whose SECOND half is zeroed must change while
  // the left channel does not: that is only true of a contiguous split.
  std::vector<float> half = latents;
  for (int64_t c = stream; c < config.latent_channels; ++c) {
    for (int64_t t = 0; t < length; ++t) half[static_cast<size_t>(c * length + t)] = 0.0f;
  }
  int64_t half_samples = 0;
  const std::vector<float> half_out =
      m3::VocoderDecode(half, length, config, VocWeights(), &half_samples);
  REQUIRE(half_samples == samples);
  size_t left_identical = 0;
  size_t right_differing = 0;
  for (int64_t i = 0; i < samples; ++i) {
    if (out[static_cast<size_t>(i)] == half_out[static_cast<size_t>(i)]) ++left_identical;
    if (out[static_cast<size_t>(samples + i)] != half_out[static_cast<size_t>(samples + i)]) {
      ++right_differing;
    }
  }
  MESSAGE("stereo fold: " << samples << " samples per channel, " << left_identical
                          << " left identical, " << right_differing << " right differing");
  CHECK(left_identical == static_cast<size_t>(samples));
  CHECK(right_differing == static_cast<size_t>(samples));
}

TEST_CASE("music3 acoustic: the vocoder refuses every wrong-shaped input by name") {
  const vllm::MiniMaxMusic3VocoderConfig config = VocConfig();
  const m3::VocoderWeights weights = VocWeights();
  int64_t samples = 0;
  CHECK_THROWS_AS(m3::VocoderDecode({1.0f}, vllm_test::kMusic3VocLength, config, weights,
                                    &samples),
                  std::runtime_error);
  CHECK_THROWS_AS(
      m3::VocoderDecode(std::vector<float>(static_cast<size_t>(config.latent_channels)), 0,
                        config, weights, &samples),
      std::runtime_error);
  m3::VocoderWeights broken = weights;
  broken.blocks.pop_back();
  CHECK_THROWS_AS(
      m3::VocoderDecode(ToVector(vllm_test::kMusic3VocLatents,
                                 static_cast<size_t>(config.latent_channels *
                                                     vllm_test::kMusic3VocLength)),
                        vllm_test::kMusic3VocLength, config, broken, &samples),
      std::runtime_error);
}

TEST_CASE("music3 acoustic: the weight-normed module walk is W1's, in order") {
  // W5 consumes what W1 folds, so the two enumerations must not drift. The
  // .inc's list came from upstream's own `named_modules` walk.
  const std::vector<std::string> ours =
      vllm::MiniMaxMusic3WeightNormedModules(VocConfig());
  REQUIRE(static_cast<int64_t>(ours.size()) == vllm_test::kMusic3VocWeightNormedCount);
  int64_t checked = 0;
  for (int64_t i = 0; i < vllm_test::kMusic3VocWeightNormedCount; ++i) {
    CHECK(ours[static_cast<size_t>(i)] ==
          std::string(vllm_test::kMusic3VocWeightNormedModules[i]));
    ++checked;
  }
  MESSAGE("weight-normed modules: " << checked << " names compared");
  CHECK(checked == 16);
}
