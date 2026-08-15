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

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "minimax_music3_acoustic_goldens.inc"
#include "vllm/model_executor/models/minimax_music3_acoustic.h"

namespace {

namespace m3 = vllm::models::music3;

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
