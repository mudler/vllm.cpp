// LTX-2.5 DiT — the DEVICE-RESIDENT forward gate (phase L8, issue #435).
//
// WHAT MAKES THIS A REAL COMPARISON, and not two arms through one helper. Every
// case below runs `Ltx2DitForwardDevice` and compares it against
// `ltx2_goldens.inc` — the SAME frozen upstream numbers `test_ltx2.cpp` holds the
// CPU forward to, produced by importing and executing the upstream Lightricks
// LTX-2 modules at reduced dimensions (scripts/gen-ltx2-goldens.py). The device
// arm and the host arm share no code below `vt::`: the host one computes into
// `std::vector<float>` with double-accumulated norms, the device one into pooled
// device buffers through vt::MatmulBT / vt::RmsNorm / vt::LayerNorm /
// vt::Attention / vt::AttentionCross and the kLtx2 glue table. A defect in either
// moves it away from a golden that neither produced.
//
// THE FIXTURE IS DUPLICATED FROM test_ltx2.cpp ON PURPOSE, and it is
// self-checking. Every weight and every input is drawn from the same
// deterministic FNV-1a + splitmix64 stream keyed by the parameter's own NAME, so
// a copy that drifted from the original would produce different weights and fail
// against the goldens immediately. Sharing it through a header would instead mean
// editing test_ltx2.cpp, whose case and assertion counts are a recorded baseline.
//
// WHY A SEPARATE BINARY. `test_ltx2` is the L2 parity gate at a fixed baseline
// (29 cases / 1615 assertions). This phase adds a second residency, not more L2
// cases, so it gets its own target and leaves that baseline where it is.
#include "vllm/model_executor/models/ltx2_device.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ltx2_goldens.inc"

#include <nlohmann/json.hpp>

#include "support/max_abs_diff.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using vllm::BindLtx2DitWeights;
using vllm::EnumerateLtx2DitTensors;
using vllm::Ltx2DitDeviceWeights;
using vllm::Ltx2DitForward;
using vllm::Ltx2DitForwardDevice;
using vllm::Ltx2DitParams;
using vllm::Ltx2DitTensorIsTable;
using vllm::Ltx2DitWeights;
using vllm::Ltx2ModalityInput;
using vllm::Ltx2PromptKvCache;
using vllm::Ltx2RopeType;
using vllm::Ltx2StageDitWeightsToDevice;
using vllm::Ltx2TensorSpec;

namespace {

vt::Device Cpu() { return vt::Device{}; }

// ---------------------------------------------------------------------------
// Ltx2Rand — the exact mirror of the generator's deterministic stream
// (scripts/gen-ltx2-goldens.py :: ltx2_rand), as test_ltx2.cpp carries it.
// ---------------------------------------------------------------------------

uint64_t Fnv1a64(const std::string& name) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char byte : name) {
    h ^= static_cast<uint64_t>(byte);
    h *= 0x100000001B3ULL;
  }
  return h;
}

uint64_t Splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

std::vector<float> MakeParam(const std::string& name, int64_t count, double scale,
                             double offset) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const uint64_t u = Splitmix64(seed + static_cast<uint64_t>(i));
    const double unit = static_cast<double>(u >> 11) * 0x1p-53;
    out[static_cast<size_t>(i)] = static_cast<float>((unit * 2.0 - 1.0) * scale + offset);
  }
  return out;
}

bool EndsWith(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<float> MakeNamedParam(const std::string& name, int64_t count) {
  if (EndsWith(name, "q_norm.weight") || EndsWith(name, "k_norm.weight")) {
    return MakeParam(name, count, 0.1, 1.0);
  }
  if (EndsWith(name, ".bias")) return MakeParam(name, count, 0.02, 0.0);
  return MakeParam(name, count, 0.05, 0.0);
}

Ltx2DitParams ReducedParams(Ltx2RopeType rope_type, bool double_precision) {
  Ltx2DitParams p;
  p.num_attention_heads = vllm_test::kLtx2Arch_num_attention_heads;
  p.attention_head_dim = vllm_test::kLtx2Arch_attention_head_dim;
  p.in_channels = vllm_test::kLtx2Arch_in_channels;
  p.out_channels = vllm_test::kLtx2Arch_out_channels;
  p.num_layers = vllm_test::kLtx2Arch_num_layers;
  p.cross_attention_dim = vllm_test::kLtx2Arch_cross_attention_dim;
  p.audio_num_attention_heads = vllm_test::kLtx2Arch_audio_num_attention_heads;
  p.audio_attention_head_dim = vllm_test::kLtx2Arch_audio_attention_head_dim;
  p.audio_in_channels = vllm_test::kLtx2Arch_audio_in_channels;
  p.audio_out_channels = vllm_test::kLtx2Arch_audio_out_channels;
  p.audio_cross_attention_dim = vllm_test::kLtx2Arch_audio_cross_attention_dim;
  p.timestep_scale_multiplier = vllm_test::kLtx2Arch_timestep_scale_multiplier;
  p.av_ca_timestep_scale_multiplier = vllm_test::kLtx2Arch_av_ca_timestep_scale_multiplier;
  p.norm_eps = 1e-6;
  p.positional_embedding_theta = 10000.0;
  p.positional_embedding_max_pos = {20, 2048, 2048};
  p.audio_positional_embedding_max_pos = {20};
  p.use_middle_indices_grid = true;
  p.rope_type = rope_type;
  p.double_precision_rope = double_precision;
  p.apply_gated_attention = true;
  p.cross_attention_adaln = true;
  p.use_prompt_adaln_single = false;
  p.ff_bias = false;
  p.audio_ff_bias = true;
  return p;
}

struct WeightSet {
  std::map<std::string, std::vector<float>> storage;
  std::map<std::string, vt::Tensor> views;
  Ltx2DitWeights weights;
};

WeightSet BuildWeights(const Ltx2DitParams& p) {
  WeightSet set;
  for (const Ltx2TensorSpec& spec : EnumerateLtx2DitTensors(p)) {
    int64_t count = 1;
    for (int64_t d : spec.shape) count *= d;
    set.storage[spec.name] = MakeNamedParam(spec.name, count);
  }
  for (const Ltx2TensorSpec& spec : EnumerateLtx2DitTensors(p)) {
    std::vector<float>& buffer = set.storage[spec.name];
    vt::Tensor t;
    if (spec.shape.size() == 1) {
      t = vt::Tensor::Contiguous(buffer.data(), vt::DType::kF32, Cpu(), {spec.shape[0]});
    } else {
      t = vt::Tensor::Contiguous(buffer.data(), vt::DType::kF32, Cpu(),
                                 {spec.shape[0], spec.shape[1]});
    }
    set.views[spec.name] = t;
  }
  set.weights = BindLtx2DitWeights(p, set.views);
  return set;
}

std::vector<float> Input(const std::string& name, int64_t count, double scale, double offset) {
  return MakeParam(name, count, scale, offset);
}

using vllm_test::MaxAbsDiff;

// The f32 DEVICE arm's floor against the same upstream goldens the CPU arm meets
// at 2e-6. It is looser than the host one for reasons that are structural, not
// slack: vt::RmsNorm and vt::LayerNorm reduce in f32 where the host helpers
// accumulate in double, and MatmulBT / the attention ops carry their own
// accumulation orders. f32 is what upstream torch does, so this arm is arguably
// the closer mirror. The bound is the same 2e-5 MiniMax-H3's device forward is
// held to against ITS upstream goldens (test_minimax_h3.cpp:1312-1313).
//
// A structural regression is orders of magnitude larger than this bound, and that
// was MEASURED rather than assumed. Five guarantees were mutated one at a time in
// a scratch tree and this suite was re-run; every one went red, and the tree
// restored byte-for-byte green afterwards:
//
//   ada_value reads the mirrored AdaLN slice          -> 13 assertions fail
//   add_gated shifts the gate row by one              -> 13 assertions fail
//   split RoPE pairs neighbours instead of halves     -> 10 assertions fail
//   gate_heads drops PytorchGatedAttention's factor 2 -> 12 assertions fail
//   the staging predicate reverts to a SUFFIX match   ->  2 fail, and the
//       assertion COUNT drops 498 -> 491 because CheckTableF32 throws mid-case,
//       which is the doctest signature to watch for rather than the pass count
//
// A tolerance is only meaningful next to evidence that something fails it.
constexpr double kDeviceRoundOff = 2e-5;

// The bf16 arm's floor. bf16 carries 8 explicit mantissa bits, so a ~1.0
// activation through two blocks lands within a few 1e-3 of the f32 result; this
// is the same 5e-3 H3's bf16 device arm uses (test_minimax_h3.cpp:1347-1348).
constexpr double kBf16RoundOff = 5e-3;

struct Modalities {
  std::vector<float> video_latent, video_timesteps, video_sigma, video_context;
  std::vector<float> audio_latent, audio_timesteps, audio_sigma, audio_context;
  std::vector<float> video_self_mask, audio_self_mask;
  std::vector<int32_t> video_context_mask, audio_context_mask;
  std::vector<double> video_positions, audio_positions;
  Ltx2ModalityInput video, audio;
};

void BuildModalities(Modalities* m, bool masked, bool dense_self_mask = false) {
  const int64_t b = vllm_test::kLtx2Batch;
  const int64_t tv = vllm_test::kLtx2VideoTokens;
  const int64_t ta = vllm_test::kLtx2AudioTokens;
  const int64_t sv = vllm_test::kLtx2VideoContext;
  const int64_t sa = vllm_test::kLtx2AudioContext;
  const int64_t vin = vllm_test::kLtx2Arch_in_channels;
  const int64_t ain = vllm_test::kLtx2Arch_audio_in_channels;
  const int64_t vctx = vllm_test::kLtx2Arch_cross_attention_dim;
  const int64_t actx = vllm_test::kLtx2Arch_audio_cross_attention_dim;

  m->video_latent = Input("input.video.latent", b * tv * vin, 0.5, 0.0);
  m->video_sigma = Input("input.video.sigma", b, 0.25, 0.5);
  m->video_timesteps = Input("input.video.timesteps", b * tv, 0.25, 0.5);
  m->video_context = Input("input.video.context", b * sv * vctx, 0.5, 0.0);
  m->audio_latent = Input("input.audio.latent", b * ta * ain, 0.5, 0.0);
  m->audio_sigma = Input("input.audio.sigma", b, 0.25, 0.5);
  m->audio_timesteps = Input("input.audio.timesteps", b * ta, 0.25, 0.5);
  m->audio_context = Input("input.audio.context", b * sa * actx, 0.5, 0.0);

  m->video_positions.assign(vllm_test::kLtx2VideoPositions,
                            vllm_test::kLtx2VideoPositions + b * 3 * tv * 2);
  m->audio_positions.assign(vllm_test::kLtx2AudioPositions,
                            vllm_test::kLtx2AudioPositions + b * 1 * ta * 2);

  m->video.batch = b;
  m->video.tokens = tv;
  m->video.context_tokens = sv;
  m->video.latent = m->video_latent.data();
  m->video.timesteps = m->video_timesteps.data();
  m->video.sigma = m->video_sigma.data();
  m->video.positions = m->video_positions.data();
  m->video.context = m->video_context.data();

  m->audio.batch = b;
  m->audio.tokens = ta;
  m->audio.context_tokens = sa;
  m->audio.latent = m->audio_latent.data();
  m->audio.timesteps = m->audio_timesteps.data();
  m->audio.sigma = m->audio_sigma.data();
  m->audio.positions = m->audio_positions.data();
  m->audio.context = m->audio_context.data();

  if (!masked) return;
  m->video_context_mask.assign(vllm_test::kLtx2VideoContextMask,
                               vllm_test::kLtx2VideoContextMask + b * sv);
  m->audio_context_mask.assign(vllm_test::kLtx2AudioContextMask,
                               vllm_test::kLtx2AudioContextMask + b * sa);
  if (dense_self_mask) {
    m->video_self_mask.assign(vllm_test::kLtx2VideoSelfMaskDense,
                              vllm_test::kLtx2VideoSelfMaskDense + b * tv * tv);
    m->audio_self_mask.assign(vllm_test::kLtx2AudioSelfMaskDense,
                              vllm_test::kLtx2AudioSelfMaskDense + b * ta * ta);
  } else {
    m->video_self_mask.assign(vllm_test::kLtx2VideoSelfMask,
                              vllm_test::kLtx2VideoSelfMask + b * tv);
    m->audio_self_mask.assign(vllm_test::kLtx2AudioSelfMask,
                              vllm_test::kLtx2AudioSelfMask + b * ta);
  }
  m->video.context_mask = m->video_context_mask.data();
  m->audio.context_mask = m->audio_context_mask.data();
  m->video.attention_mask = m->video_self_mask.data();
  m->video.attention_mask_rows = dense_self_mask ? tv : 1;
  m->audio.attention_mask = m->audio_self_mask.data();
  m->audio.attention_mask_rows = dense_self_mask ? ta : 1;
}

// The text of the refusal `run` throws, or "" when it does not throw at all.
template <typename Fn>
std::string RefusalMessage(Fn run) {
  try {
    run();
  } catch (const std::exception& e) {
    return std::string(e.what());
  }
  return std::string();
}

// One forward case, run device-resident and compared against the upstream golden
// that the SAME case's CPU arm is compared against in test_ltx2.cpp.
struct CaseResult {
  double video = 0.0, audio = 0.0;
};

CaseResult RunDeviceCase(vt::Queue& q, vt::DType stream, Ltx2RopeType rope_type,
                         bool double_precision, bool masked, const float* want_video,
                         const float* want_audio, bool audio_enabled = true,
                         bool dense_self_mask = false, bool prompt_adaln = false) {
  Ltx2DitParams p = ReducedParams(rope_type, double_precision);
  p.use_prompt_adaln_single = prompt_adaln;
  WeightSet set = BuildWeights(p);
  // Staged at the SAME dtype the stream computes in. Staging f32 weights under a
  // bf16 stream would compare a DIFFERENT MODEL, not a different dtype policy.
  const Ltx2DitDeviceWeights staged = Ltx2StageDitWeightsToDevice(q, p, set.views, stream);
  Modalities m;
  BuildModalities(&m, masked, dense_self_mask);
  m.audio.enabled = audio_enabled;

  const vllm::Ltx2DitOutputs out =
      Ltx2DitForwardDevice(q, p, staged.weights, &m.video, &m.audio, stream);
  const size_t vcount = static_cast<size_t>(m.video.batch * m.video.tokens * p.out_channels);
  const size_t acount = static_cast<size_t>(m.audio.batch * m.audio.tokens * p.audio_out_channels);
  REQUIRE(out.video.size() == vcount);
  REQUIRE(out.audio.size() == acount);
  return CaseResult{MaxAbsDiff(out.video, want_video, vcount),
                    MaxAbsDiff(out.audio, want_audio, acount)};
}

// The six forward cases test_ltx2.cpp gates the CPU arm on, run device-resident.
// Naming them here rather than looping keeps a failure attributable to ONE case.
void CheckAllForwardCases(vt::Queue& q, const char* label) {
  {
    INFO(label << " / split RoPE");
    const CaseResult r = RunDeviceCase(q, vt::DType::kF32, Ltx2RopeType::kSplit, false, false,
                                       vllm_test::kLtx2ForwardSplitVideo,
                                       vllm_test::kLtx2ForwardSplitAudio);
    MESSAGE("split: max|diff| video=" << r.video << " audio=" << r.audio);
    CHECK(r.video < kDeviceRoundOff);
    CHECK(r.audio < kDeviceRoundOff);
  }
  {
    INFO(label << " / interleaved RoPE");
    const CaseResult r = RunDeviceCase(q, vt::DType::kF32, Ltx2RopeType::kInterleaved, false,
                                       false, vllm_test::kLtx2ForwardInterleavedVideo,
                                       vllm_test::kLtx2ForwardInterleavedAudio);
    MESSAGE("interleaved: max|diff| video=" << r.video << " audio=" << r.audio);
    CHECK(r.video < kDeviceRoundOff);
    CHECK(r.audio < kDeviceRoundOff);
  }
  {
    INFO(label << " / the float64 frequency ladder");
    const CaseResult r = RunDeviceCase(q, vt::DType::kF32, Ltx2RopeType::kSplit, true, false,
                                       vllm_test::kLtx2ForwardDoubleVideo,
                                       vllm_test::kLtx2ForwardDoubleAudio);
    MESSAGE("float64 freqs: max|diff| video=" << r.video << " audio=" << r.audio);
    CHECK(r.video < kDeviceRoundOff);
    CHECK(r.audio < kDeviceRoundOff);
  }
  {
    // The masked case is what puts every cross-attention on the BIASED path, so
    // it is the one that exercises vt::AttentionCross's bias argument for the
    // self-attentions too.
    INFO(label << " / prompt mask + self-attention strength mask");
    const CaseResult r = RunDeviceCase(q, vt::DType::kF32, Ltx2RopeType::kSplit, false, true,
                                       vllm_test::kLtx2ForwardMaskedVideo,
                                       vllm_test::kLtx2ForwardMaskedAudio);
    MESSAGE("masked: max|diff| video=" << r.video << " audio=" << r.audio);
    CHECK(r.video < kDeviceRoundOff);
    CHECK(r.audio < kDeviceRoundOff);
  }
  {
    // The DENSE (B, T, T) mask is the case that makes the PER-QUERY bias row
    // index observable: with a key-only mask there is exactly one row, and a
    // kernel that read row 0 for every query would be indistinguishable.
    INFO(label << " / a DENSE (B, T, T) self-attention mask");
    const CaseResult r = RunDeviceCase(q, vt::DType::kF32, Ltx2RopeType::kSplit, false, true,
                                       vllm_test::kLtx2ForwardDenseMaskVideo,
                                       vllm_test::kLtx2ForwardDenseMaskAudio, true, true);
    MESSAGE("dense mask: max|diff| video=" << r.video << " audio=" << r.audio);
    CHECK(r.video < kDeviceRoundOff);
    CHECK(r.audio < kDeviceRoundOff);
  }
  {
    INFO(label << " / a disabled audio stream still feeds audio->video");
    const CaseResult r = RunDeviceCase(q, vt::DType::kF32, Ltx2RopeType::kSplit, false, false,
                                       vllm_test::kLtx2ForwardAudioOffVideo,
                                       vllm_test::kLtx2ForwardAudioOffAudio, false);
    MESSAGE("audio disabled: max|diff| video=" << r.video << " audio=" << r.audio);
    CHECK(r.video < kDeviceRoundOff);
    CHECK(r.audio < kDeviceRoundOff);
  }
}

// The bf16 PRODUCTION stream: held to the same goldens at a bf16 band, and then
// required to actually DIFFER from the f32 arm. Without that second assertion the
// case would pass with the dtype policy silently not applied.
void CheckBf16Stream(vt::Queue& q, const char* label) {
  INFO(label);
  const CaseResult bf16 =
      RunDeviceCase(q, vt::DType::kBF16, Ltx2RopeType::kSplit, false, false,
                    vllm_test::kLtx2ForwardSplitVideo, vllm_test::kLtx2ForwardSplitAudio);
  MESSAGE("bf16 vs upstream: max|diff| video=" << bf16.video << " audio=" << bf16.audio);
  CHECK(bf16.video < kBf16RoundOff);
  CHECK(bf16.audio < kBf16RoundOff);
  // ... and it must NOT be the f32 result. A bf16 stream that matched f32 to
  // round-off would mean the staging or the kernels silently kept f32, which is
  // exactly the too-WIDE dtype a golden gate cannot catch on its own.
  CHECK(bf16.video > kDeviceRoundOff);
}

// The prompt-side AdaLN arm — upstream's DEFAULT and what the shipped DiT runs
// (.agents/specs/ltx25-prompt-adaln.md, issue #644). The device path forms
// `kv_modulation = table + prompt_timestep` (transformer.py:441-443) through
// `ada_value` and then broadcasts one row per BATCH element over that element's
// prompt tokens, which is a DIFFERENT `modulate` call shape from the static
// arm — so it needs its own case rather than riding on the loop above.
void CheckPromptAdalnCases(vt::Queue& q, const char* label) {
  {
    INFO(label << " / prompt AdaLN, f32");
    const CaseResult r =
        RunDeviceCase(q, vt::DType::kF32, Ltx2RopeType::kSplit, false, false,
                      vllm_test::kLtx2ForwardPromptAdalnVideo,
                      vllm_test::kLtx2ForwardPromptAdalnAudio, /*audio_enabled=*/true,
                      /*dense_self_mask=*/false, /*prompt_adaln=*/true);
    MESSAGE("prompt AdaLN f32: max|diff| video=" << r.video << " audio=" << r.audio);
    CHECK(r.video < kDeviceRoundOff);
    CHECK(r.audio < kDeviceRoundOff);
  }
  {
    // The masked case: the prompt mask and the prompt modulation act on the SAME
    // context tensor, and a per-batch broadcast written as a per-token one would
    // survive the unmasked case at batch 1.
    INFO(label << " / prompt AdaLN + masks, f32");
    const CaseResult r =
        RunDeviceCase(q, vt::DType::kF32, Ltx2RopeType::kSplit, false, true,
                      vllm_test::kLtx2ForwardPromptAdalnMaskedVideo,
                      vllm_test::kLtx2ForwardPromptAdalnMaskedAudio, /*audio_enabled=*/true,
                      /*dense_self_mask=*/false, /*prompt_adaln=*/true);
    MESSAGE("prompt AdaLN masked f32: max|diff| video=" << r.video << " audio=" << r.audio);
    CHECK(r.video < kDeviceRoundOff);
    CHECK(r.audio < kDeviceRoundOff);
  }
  {
    // AND IT IS LOAD-BEARING. The flag-ON and flag-OFF goldens share every common
    // weight bit-for-bit, so a device path that bound the module and never added
    // its output would reproduce the flag-OFF numbers exactly.
    //
    // The bound is `kDeviceRoundOff` itself, and that is the precise statement:
    // the flag-ON run must miss the flag-OFF golden by MORE than the tolerance
    // the flag-OFF case is held to, or a dropped term would pass that gate.
    // Measured here: video 1.46e-4 (7.3x the bound), audio 7.37e-5 (3.7x).
    INFO(label << " / prompt AdaLN is load-bearing");
    const CaseResult r =
        RunDeviceCase(q, vt::DType::kF32, Ltx2RopeType::kSplit, false, false,
                      vllm_test::kLtx2ForwardSplitVideo, vllm_test::kLtx2ForwardSplitAudio,
                      /*audio_enabled=*/true, /*dense_self_mask=*/false,
                      /*prompt_adaln=*/true);
    MESSAGE("prompt AdaLN vs flag-OFF golden: video=" << r.video << " audio=" << r.audio);
    CHECK(r.video > kDeviceRoundOff);
    CHECK(r.audio > kDeviceRoundOff);
  }
  {
    // The bf16 PRODUCTION stream on the same arm: `ada_value` stores the
    // table+timestep sum at the stream dtype, so this is where a bf16 store of the
    // new sum is exercised at all.
    INFO(label << " / prompt AdaLN, bf16");
    const CaseResult r =
        RunDeviceCase(q, vt::DType::kBF16, Ltx2RopeType::kSplit, false, false,
                      vllm_test::kLtx2ForwardPromptAdalnVideo,
                      vllm_test::kLtx2ForwardPromptAdalnAudio, /*audio_enabled=*/true,
                      /*dense_self_mask=*/false, /*prompt_adaln=*/true);
    MESSAGE("prompt AdaLN bf16: max|diff| video=" << r.video << " audio=" << r.audio);
    CHECK(r.video < kBf16RoundOff);
    CHECK(r.audio < kBf16RoundOff);
    CHECK(r.video > kDeviceRoundOff);
  }
}

vt::Backend* TryCuda() { return vt::TryGetBackend(vt::DeviceType::kCUDA); }

}  // namespace

// ---------------------------------------------------------------------------
// The seam
// ---------------------------------------------------------------------------

TEST_CASE("ltx2 device: the kLtx2 glue table is registered for the CPU backend") {
  // Registering on kCPU is what lets the whole device-forward CODE PATH be
  // covered without a GPU, so a GPU gates the KERNELS and not the port.
  CHECK(vllm::ltx2::Ltx2DeviceKernelsAvailable(vt::DeviceType::kCPU));
  const vllm::ltx2::Ltx2DeviceKernels* k = vllm::ltx2::Ltx2Device(vt::DeviceType::kCPU);
  REQUIRE(k != nullptr);
  CHECK(k->ada_value != nullptr);
  CHECK(k->modulate != nullptr);
  CHECK(k->add_gated != nullptr);
  CHECK(k->gate_heads != nullptr);
  CHECK(k->rope != nullptr);
  CHECK(k->output_modulate != nullptr);
  CHECK(k->silu != nullptr);
}

TEST_CASE("ltx2 device: vt::AttentionCross has a CUDA kernel, which this row owed") {
  // ops.h:2174-2182 recorded the native CUDA kernel as owed "alongside the
  // LTX-2.5 device-resident forward, which is the first caller that would need
  // it". Without it, GB10's unified memory would let RegisterReferenceTier serve
  // the CPU kernel to a CUDA queue and every cross-attention in the DiT would
  // have run on the HOST while every gate stayed green.
  if (TryCuda() == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  CHECK(vt::OpRegistered(vt::OpId::kAttentionCross, vt::DeviceType::kCUDA));
}

// ---------------------------------------------------------------------------
// Staging: the dtype policy
// ---------------------------------------------------------------------------

TEST_CASE("ltx2 device: staging narrows the projections and leaves the TABLES f32") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  vt::Queue q{Cpu(), nullptr};

  const Ltx2DitDeviceWeights bf16 =
      Ltx2StageDitWeightsToDevice(q, p, set.views, vt::DType::kBF16);
  int64_t tables = 0, projections = 0;
  for (const auto& kv : bf16.views) {
    if (Ltx2DitTensorIsTable(kv.first)) {
      ++tables;
      // The CHECKPOINT stores these F32 (ltx2_loader.h:64-66). Narrowing a
      // tensor the file itself widened would be the dtype rule applied
      // backwards, so the bf16 stream must NOT touch them.
      CHECK(kv.second.dtype == vt::DType::kF32);
    } else {
      ++projections;
      CHECK(kv.second.dtype == vt::DType::kBF16);
    }
  }
  // Both classes must be non-empty, or the assertions above are vacuous.
  CHECK(tables > 0);
  CHECK(projections > 0);
  MESSAGE("staged " << projections << " projections bf16, " << tables << " tables f32");

  // The f32 parity arm leaves everything f32.
  const Ltx2DitDeviceWeights f32 = Ltx2StageDitWeightsToDevice(q, p, set.views, vt::DType::kF32);
  for (const auto& kv : f32.views) CHECK(kv.second.dtype == vt::DType::kF32);

  // The predicate itself, on ALL SIX names upstream uses (ltx2.cpp:268-269,
  // :294-300). The two `_a2v_ca_*` ones are the reason this is a SUBSTRING match
  // and not a suffix match: they do not END in `scale_shift_table`, a suffix
  // predicate silently staged them bf16, and `ada_value` then read them through
  // its `const float*` table parameter — which drove the audio<->video cross gate
  // to 2.85e32 and the video stream to 6.89e30 after one block, with the f32 arm
  // (where the mismatch cannot arise) still green at 1e-7.
  CHECK(Ltx2DitTensorIsTable("scale_shift_table"));
  CHECK(Ltx2DitTensorIsTable("audio_scale_shift_table"));
  CHECK(Ltx2DitTensorIsTable("transformer_blocks.0.scale_shift_table"));
  CHECK(Ltx2DitTensorIsTable("transformer_blocks.0.audio_scale_shift_table"));
  CHECK(Ltx2DitTensorIsTable("transformer_blocks.0.prompt_scale_shift_table"));
  CHECK(Ltx2DitTensorIsTable("transformer_blocks.0.audio_prompt_scale_shift_table"));
  CHECK(Ltx2DitTensorIsTable("transformer_blocks.0.scale_shift_table_a2v_ca_video"));
  CHECK(Ltx2DitTensorIsTable("transformer_blocks.0.scale_shift_table_a2v_ca_audio"));
  CHECK(Ltx2DitTensorIsTable("patchify_proj.weight") == false);
  CHECK(Ltx2DitTensorIsTable("transformer_blocks.0.attn1.to_q.weight") == false);

  // And the guard that makes a future miss LOUD instead of silent: a table staged
  // at the stream dtype is refused by name rather than reinterpreted.
  const Ltx2DitParams p2 = ReducedParams(Ltx2RopeType::kSplit, false);
  Ltx2DitDeviceWeights broken =
      Ltx2StageDitWeightsToDevice(q, p2, set.views, vt::DType::kBF16);
  broken.weights.blocks[0].scale_shift_table_a2v_ca_video.dtype = vt::DType::kBF16;
  Modalities mm;
  BuildModalities(&mm, false);
  const std::string msg = RefusalMessage([&] {
    (void)Ltx2DitForwardDevice(q, p2, broken.weights, &mm.video, &mm.audio, vt::DType::kBF16);
  });
  INFO(msg);
  CHECK(msg.find("must be F32") != std::string::npos);
}

// ---------------------------------------------------------------------------
// The forward, against the SAME goldens as the CPU arm
// ---------------------------------------------------------------------------

TEST_CASE("ltx2 device: the DEVICE-RESIDENT forward matches upstream (CPU backend)") {
  vt::Queue q{Cpu(), nullptr};
  CheckAllForwardCases(q, "cpu-backend");
}

TEST_CASE("ltx2 device: the DEVICE-RESIDENT forward matches upstream on CUDA") {
  vt::Backend* cuda = TryCuda();
  if (cuda == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  vt::Queue q = cuda->CreateQueue();
  CheckAllForwardCases(q, "cuda");
}

TEST_CASE("ltx2 device: the prompt-side AdaLN arm matches upstream (CPU backend)") {
  vt::Queue q{Cpu(), nullptr};
  CheckPromptAdalnCases(q, "cpu-backend");
}

TEST_CASE("ltx2 device: the prompt-side AdaLN arm matches upstream on CUDA") {
  vt::Backend* cuda = TryCuda();
  if (cuda == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  vt::Queue q = cuda->CreateQueue();
  CheckPromptAdalnCases(q, "cuda");
}

TEST_CASE("ltx2 device: the bf16 PRODUCTION stream matches upstream (CPU backend)") {
  vt::Queue q{Cpu(), nullptr};
  CheckBf16Stream(q, "cpu-backend bf16");
}

TEST_CASE("ltx2 device: the bf16 PRODUCTION stream matches upstream on CUDA") {
  vt::Backend* cuda = TryCuda();
  if (cuda == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  vt::Queue q = cuda->CreateQueue();
  CheckBf16Stream(q, "cuda bf16");
}

TEST_CASE("ltx2 device: CUDA tracks the HOST forward, not just the golden") {
  // The golden comparison above would pass on both arms even if the two device
  // backends disagreed with each other within the band. This one pins the CUDA
  // kernels directly to the trusted host forward on the identical inputs, which
  // is what would catch a CUDA-only defect (a launch-bound off-by-one, a
  // shared-memory tile edge) that the band happens to absorb.
  vt::Backend* cuda = TryCuda();
  if (cuda == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  vt::Queue q = cuda->CreateQueue();
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  BuildModalities(&m, true);
  const vllm::Ltx2DitOutputs host =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32);
  const Ltx2DitDeviceWeights staged =
      Ltx2StageDitWeightsToDevice(q, p, set.views, vt::DType::kF32);
  const vllm::Ltx2DitOutputs dev =
      Ltx2DitForwardDevice(q, p, staged.weights, &m.video, &m.audio, vt::DType::kF32);
  const double dv = MaxAbsDiff(dev.video, host.video.data(), host.video.size());
  const double da = MaxAbsDiff(dev.audio, host.audio.data(), host.audio.size());
  INFO("CUDA-vs-host: video max|diff| = " << dv << ", audio max|diff| = " << da);
  // f32 summation-order slack only (the two arms differ in the norm reduction
  // width and in the softmax recurrence); a structural CUDA-kernel regression
  // would be orders of magnitude larger.
  CHECK(dv < kDeviceRoundOff);
  CHECK(da < kDeviceRoundOff);

  // ── the bf16 arm, BACKEND AGAINST BACKEND ─────────────────────────────────
  //
  // WHY THIS IS HERE. The landing commit said the bf16 CUDA numbers were
  // "bit-identical to the CPU backend's". What was actually measured was that the
  // two arms print the SAME max|diff| against the goldens at six significant
  // figures, and equal printed maxima are consistent with bit-identity without
  // establishing it: nothing compared the two backends' OUTPUTS to each other at
  // bf16 at all. This is that comparison, and it is stated as what it measures.
  //
  // It is deliberately not asserted as equality. The two arms run different
  // softmax algorithms (online recurrence on CUDA, explicit three-pass on the
  // CPU) and different reduction orders, so a bf16 band is the honest bound; the
  // MEASURED value is printed so the claim can be made from a number rather than
  // from an adjective.
  const Ltx2DitDeviceWeights staged_bf16 =
      Ltx2StageDitWeightsToDevice(q, p, set.views, vt::DType::kBF16);
  const vllm::Ltx2DitOutputs cuda_bf16 =
      Ltx2DitForwardDevice(q, p, staged_bf16.weights, &m.video, &m.audio, vt::DType::kBF16);

  // THE CPU-BACKEND ARM RUNS RIGHT AFTER THE CUDA ONE, ON PURPOSE, AND WITH NO
  // POOL SCOPING. That ordering is the only thing in the tree that reaches the
  // #516 hazard from the SIGSEGV side, and it used to need a per-case
  // `DevicePool` to survive: `vllm::Pool()` was a process-wide singleton keyed
  // by SIZE CLASS ONLY, so a block `cudaMalloc`ed for the CUDA arm three lines
  // up was handed straight back to a CPU-backend `DBuf` of the same size class,
  // and the CPU backend's `Copy` is a host `memcpy` on what is a device pointer.
  // MEASURED, not theorised: the case SIGSEGV'd on GB10 in
  // `__memcpy_sve <- UploadStream <- PrepareStreamDev`, and compute-sanitizer
  // reported ZERO device errors, because the fault is host-side.
  //
  // The pool is now one-per-device (device_pool.h, .agents/specs/
  // pool-device-key.md), so the scope is gone and this case is again a DETECTOR
  // for the hazard rather than a caller that was scoped away from it. Do not
  // re-introduce an `ActivePoolScope` here: it would pass whether or not the
  // pool is correct.
  vt::Queue cpuq{Cpu(), nullptr};
  const Ltx2DitDeviceWeights host_bf16 =
      Ltx2StageDitWeightsToDevice(cpuq, p, set.views, vt::DType::kBF16);
  const vllm::Ltx2DitOutputs cpu_bf16 =
      Ltx2DitForwardDevice(cpuq, p, host_bf16.weights, &m.video, &m.audio, vt::DType::kBF16);
  REQUIRE(cuda_bf16.video.size() == cpu_bf16.video.size());
  const double bv = MaxAbsDiff(cuda_bf16.video, cpu_bf16.video.data(), cpu_bf16.video.size());
  const double ba = MaxAbsDiff(cuda_bf16.audio, cpu_bf16.audio.data(), cpu_bf16.audio.size());
  MESSAGE("bf16 CUDA-vs-CPU-BACKEND max|diff|: video=" << bv << " audio=" << ba
          << " (a BOUND, not a bit-identity claim -- nothing here measures bit-identity)");
  CHECK(bv < kBf16RoundOff);
  CHECK(ba < kBf16RoundOff);
}

// ---------------------------------------------------------------------------
// The refusals, each of which would otherwise render confidently and wrongly
// ---------------------------------------------------------------------------

TEST_CASE("ltx2 device: a stream dtype the forward does not carry is REFUSED") {
  vt::Queue q{Cpu(), nullptr};
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  const Ltx2DitDeviceWeights staged =
      Ltx2StageDitWeightsToDevice(q, p, set.views, vt::DType::kF32);
  Modalities m;
  BuildModalities(&m, false);
  const std::string msg = RefusalMessage([&] {
    (void)Ltx2DitForwardDevice(q, p, staged.weights, &m.video, &m.audio, vt::DType::kI8);
  });
  INFO(msg);
  CHECK(msg.find("bf16") != std::string::npos);
}

TEST_CASE("ltx2 device: a prompt K/V cache is REFUSED, never silently ignored") {
  // An ignored cache recomputes CORRECTLY and quietly loses the reuse — the kind
  // of divergence that is found a phase later, when a timing does not move.
  vt::Queue q{Cpu(), nullptr};
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  const Ltx2DitDeviceWeights staged =
      Ltx2StageDitWeightsToDevice(q, p, set.views, vt::DType::kF32);
  Modalities m;
  BuildModalities(&m, false);
  Ltx2PromptKvCache cache;
  const std::string msg = RefusalMessage([&] {
    (void)Ltx2DitForwardDevice(q, p, staged.weights, &m.video, &m.audio, vt::DType::kF32, &cache);
  });
  INFO(msg);
  CHECK(msg.find("cache") != std::string::npos);
  CHECK(msg.find("REFUSED") != std::string::npos);
}

TEST_CASE("ltx2 device: HOST weights on a device queue are REFUSED by name") {
  // This is the failure GB10 would otherwise hide: unified memory addresses a
  // host pointer from a CUDA kernel, so the model would run at host bandwidth
  // with every gate green and every later timing meaningless.
  vt::Backend* cuda = TryCuda();
  if (cuda == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered (a host view IS the device view on CPU)");
    return;
  }
  vt::Queue q = cuda->CreateQueue();
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);  // host-resident
  Modalities m;
  BuildModalities(&m, false);
  const std::string msg = RefusalMessage([&] {
    (void)Ltx2DitForwardDevice(q, p, set.weights, &m.video, &m.audio, vt::DType::kF32);
  });
  INFO(msg);
  CHECK(msg.find("not resident") != std::string::npos);
}

// ---------------------------------------------------------------------------
// The SHIPPED 21.00B checkpoint
// ---------------------------------------------------------------------------

// Everything above runs at reduced dimensions, which is what makes it comparable
// against upstream. This case is the opposite claim and the one the phase is
// actually for: a SHIPPED 21.00B DiT — 48 layers, inner_dim 4096, audio_inner
// 2048, head_dim 128 — staged onto the GPU and pushed through
// `Ltx2DitForwardDevice`.
//
// WHICH CHECKPOINT, because the two shipped files are NOT interchangeable and
// this was MEASURED on GB10 (2026-08-12):
//
//   * vonkaiser `ltx-2.5-22b-distilled-fp8.safetensors` (21.0 GB, 6124 tensors)
//     STAGES AND RUNS. F8_E4M3 plus an F32 scalar is the dequant path phase L6
//     already implements.
//   * The first-party `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`
//     (18.72 GB, 7876 tensors) ALSO stages and runs, since phase L9a
//     (.agents/specs/nvfp4-nibble-order.md).
//
//     It used to be refused, and the refusal carried a WRONG diagnosis worth
//     recording: it said the file stored a LINEAR [N, K/16] group scale. It does
//     not. The bytes were SWIZZLED all along, declared in the cuBLAS-padded
//     framing [4096, 256] rather than the `to_blocked` [1024, 1024] the loader
//     knew — and for every layer in that file the padded framing is NUMERICALLY
//     IDENTICAL to the linear shape, which is exactly why a shape test could not
//     tell them apart and why the wrong diagnosis looked right.
//
//     The file also packs element 2j in the HIGH nibble, where torchao, ModelOpt
//     and our default put it in the LOW one. Nothing about the SHAPES could have
//     revealed that: it was found by correlating the dequantized weights against
//     the vonkaiser FP8 DiT of the same base weights (0.9956 correct vs 0.032
//     wrong), which is now a committed gate in test_ltx2_loader rather than a
//     one-off measurement.
//
// WHAT IT DOES AND DOES NOT CLAIM. There is no golden at this geometry and there
// is no oracle: vLLM-Omni carries no native LTX-2.5 path, so nothing here is a
// parity result. What it establishes is that the shipped weights load, stage, and
// RUN on the device, and that the result is finite and non-degenerate — the two
// ways a 21B forward fails silently. Any wall-clock printed is for sizing only
// and is NOT a speed result; the spec's §0 says why no denominator exists.
//
// AND UNDER WHICH CONFIGURATION, which this case used to leave unstated. That is
// not a footnote here: `Ltx2StreamDitToDevice` resolves geometry from SHAPES, and
// shapes cannot see `frequencies_precision` or
// `av_ca_timestep_scale_multiplier`. Only ONE of the two shipped DiTs declares
// them — read from the NAS 2026-08-12:
//
//   first-party NVFP4  __metadata__ = ['config','gemma_source_checkpoint',
//                                      'license','model_version']
//   vonkaiser FP8      has __metadata__ key: FALSE
//
// So the FP8 arm — the one this case actually runs — took the parser defaults
// `double_precision_rope = false` and `av_ca_timestep_scale_multiplier = 1`,
// against LTX-2.5's declared float64 and 1000, and nothing said so. The engine
// now REFUSES that (ltx2_video.cpp, the `dit_config_path` extra); this case sits
// BELOW the engine, so it resolves the configuration the same way through
// `Ltx2AdoptDeclaredDitParams` and ASSERTS what it ended up with either way.
// Set LTX2_SHIPPED_DIT_CONFIG to a `{"transformer": {...}}` JSON file to supply
// one for a checkpoint that declares none; without it the case still runs, and
// says in its own assertions that it is NOT running LTX-2.5's declared config.
//
// Off by default: it needs the checkpoint AND ~42 GB of device memory, and GB10's
// memory is unified, so an unattended run of this alongside anything else is how
// the box gets rebooted. Set LTX2_SHIPPED_DIT to the .safetensors path.
TEST_CASE("ltx2 device: a SHIPPED 21.00B DiT stages and runs on the GPU") {
  const char* path_env = std::getenv("LTX2_SHIPPED_DIT");
  if (path_env == nullptr) {
    MESSAGE("SKIPPED: set LTX2_SHIPPED_DIT to a shipped 21.00B DiT .safetensors");
    return;
  }
  vt::Backend* cuda = TryCuda();
  if (cuda == nullptr) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  vt::Queue q = cuda->CreateQueue();

  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path_env);
  vllm::Ltx2DitQuant quant = vllm::Ltx2DitQuant::kFp8;
  (void)vllm::Ltx2ParseDitParamsFromCheckpoint(file, &quant);
  // `std::string`, NOT a bare `const char*` ternary. doctest's MESSAGE streams
  // through an ostream whose overload set makes `cond ? "NVFP4" : "FP8"` decay to
  // `bool` and print `1`, so this line reported `quant=1` for BOTH shipped files
  // and could not do the one job spec §3.1 gives it: NAME which DiT produced each
  // artifact, every time. The two are not interchangeable quantizations of one
  // model — they differ in a TRAINED `keyframes_abs_pos_embedding` — so a report
  // that cannot say which one ran is not evidence about either.
  // EVERY `const char*` streamed here is wrapped, not just the ternary. A bare
  // `path_env` decays through the SAME overload and printed `(path 1)` on the
  // first GB10 run of this fix -- the identical defect, reintroduced one token
  // away from where it was being repaired.
  const std::string quant_name = quant == vllm::Ltx2DitQuant::kNvfp4 ? "NVFP4" : "FP8";
  MESSAGE("shipped DiT: " << file.Names().size() << " tensors, quant=" << quant_name
                          << " (path " << std::string(path_env) << ")");

  // Stage tensor-by-tensor. This is the production path: it dequantizes and
  // uploads ONE tensor at a time and frees each host buffer before the next, so
  // peak residency is the device copy plus one tensor rather than two whole
  // models (ltx2_loader.h:74-80).
  vllm::Ltx2DitLoadOptions opt;
  opt.allow_unported_modules = true;  // the shipped file carries four unported families
  const auto t0 = std::chrono::steady_clock::now();
  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2StreamDitToDevice(q, file, opt);
  const auto t1 = std::chrono::steady_clock::now();
  const double stage_s = std::chrono::duration<double>(t1 - t0).count();

  // The geometry the FILE describes, at full scale.
  vllm::Ltx2DitParams p = ck.params;
  MESSAGE("staged in " << stage_s << " s (NOT a speed result): layers=" << p.num_layers
                       << " inner=" << p.inner_dim() << " audio_inner=" << p.audio_inner_dim()
                       << " head_dim=" << p.attention_head_dim
                       << " unported=" << ck.unported.size());
  CHECK(p.num_layers == 48);
  CHECK(p.inner_dim() == 4096);
  CHECK(p.attention_head_dim == 128);

  // ── WHICH CONFIGURATION THIS FORWARD RUNS UNDER, resolved and then ASSERTED ──
  //
  // Three cases, and each one ends in an assertion rather than a hope. The
  // adoption rule is `Ltx2AdoptDeclaredDitParams`, the SAME function the engine
  // calls, so the two cannot answer differently.
  const bool declares_config = file.Metadata().count("config") != 0;
  const char* config_env = std::getenv("LTX2_SHIPPED_DIT_CONFIG");
  // `std::string`, for the reason F8 records: a `const char*` ternary decays to
  // `bool` through doctest's ostream and prints `1`, which is what this very line
  // did on its first GB10 run.
  const std::string config_source =
      declares_config ? std::string("the checkpoint's own __metadata__[\"config\"]")
      : config_env != nullptr ? std::string("LTX2_SHIPPED_DIT_CONFIG=") + config_env
                              : std::string("NONE (manifest shapes only)");
  MESSAGE("config source: " << config_source);
  if (declares_config || config_env != nullptr) {
    nlohmann::json config;
    std::string source;
    if (declares_config) {
      // Both present is the same ambiguity the engine refuses; here it is simply
      // not exercised, and the checkpoint's own config wins by being named.
      config = vllm::Ltx2ReadCheckpointConfig(file);
      source = "the DiT checkpoint's own __metadata__[\"config\"][\"transformer\"]";
    } else {
      std::ifstream in(config_env, std::ios::binary);
      REQUIRE_MESSAGE(in.good(), "cannot open LTX2_SHIPPED_DIT_CONFIG ",
                      std::string(config_env));
      config = nlohmann::json::parse(
          std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
      source = std::string("the LTX2_SHIPPED_DIT_CONFIG file '") + config_env + "'";
    }
    // Refuses by name if it describes a different weight contract, so a config
    // from the OTHER shipped DiT cannot be bound to these tensors unnoticed.
    p = vllm::Ltx2AdoptDeclaredDitParams(config, ck.params, opt.allow_unported_modules, source);
    MESSAGE("adopted config: double_precision_rope=" << p.double_precision_rope
            << " av_ca_timestep_scale_multiplier=" << p.av_ca_timestep_scale_multiplier
            << " timestep_scale_multiplier=" << p.timestep_scale_multiplier);
    // LTX-2.5's declared values. A config that reached here and did NOT carry
    // them is not an LTX-2.5 config, and saying so is the point of asserting.
    CHECK(p.double_precision_rope);
    CHECK(p.av_ca_timestep_scale_multiplier == 1000);
    // The geometry must survive adoption, or the config was bound to the wrong
    // checkpoint and the contract check did not do its job.
    CHECK(p.num_layers == ck.params.num_layers);
    CHECK(p.inner_dim() == ck.params.inner_dim());
  } else {
    // THE HONEST BRANCH, and it is the one the shipped vonkaiser FP8 DiT takes.
    // The forward below runs under the MANIFEST DEFAULTS. That is a different
    // configuration from LTX-2.5's declared one, and the assertions state exactly
    // which, so no later reader can take this run for an LTX-2.5-configured
    // render. It does not invalidate what this case claims — the weights stage,
    // the forward executes on the device, the output is finite and
    // non-degenerate — because none of those depends on the RoPE precision.
    MESSAGE("NO CONFIG DECLARED OR SUPPLIED. This forward runs under the MANIFEST "
            "defaults: double_precision_rope=false, av_ca_timestep_scale_multiplier=1. "
            "LTX-2.5 declares float64 and 1000, so this is NOT LTX-2.5's declared "
            "configuration. Set LTX2_SHIPPED_DIT_CONFIG to run under one.");
    CHECK_FALSE(p.double_precision_rope);
    CHECK(p.av_ca_timestep_scale_multiplier == 1);
  }
  // Every weight must be device-resident, or the forward would run at host
  // bandwidth on unified memory with nothing complaining.
  REQUIRE(!ck.views.empty());
  for (const auto& kv : ck.views) {
    REQUIRE(kv.second.device.type == vt::DeviceType::kCUDA);
  }

  // A SMALL token count at the REAL width. The point is the geometry the kernels
  // see — head_dim 128, inner 4096, 48 layers — not the sequence length; a full
  // 512x768x121 latent is ~2.6e14 FLOPs per step and is a different claim.
  const int64_t tv = 8, ta = 4, s = 8;
  std::vector<float> vlat(static_cast<size_t>(tv * p.in_channels), 0.05f);
  std::vector<float> alat(static_cast<size_t>(ta * p.audio_in_channels), 0.05f);
  std::vector<float> vts(static_cast<size_t>(tv), 0.5f), ats(static_cast<size_t>(ta), 0.5f);
  float vsig = 0.5f, asig = 0.5f;
  std::vector<float> vctx(static_cast<size_t>(s * p.cross_attention_dim), 0.02f);
  std::vector<float> actx(static_cast<size_t>(s * p.audio_cross_attention_dim), 0.02f);
  // The middle-indices grid: [batch, n_pos_dims, tokens, 2] holding each patch's
  // [start, end) bounds (ltx2.h:240-244).
  std::vector<double> vpos(static_cast<size_t>(3 * tv * 2)), apos(static_cast<size_t>(1 * ta * 2));
  for (int64_t a = 0; a < 3; ++a) {
    for (int64_t t = 0; t < tv; ++t) {
      vpos[static_cast<size_t>((a * tv + t) * 2)] = static_cast<double>(t);
      vpos[static_cast<size_t>((a * tv + t) * 2 + 1)] = static_cast<double>(t + 1);
    }
  }
  for (int64_t t = 0; t < ta; ++t) {
    apos[static_cast<size_t>(t * 2)] = static_cast<double>(t);
    apos[static_cast<size_t>(t * 2 + 1)] = static_cast<double>(t + 1);
  }

  Ltx2ModalityInput vin;
  vin.batch = 1; vin.tokens = tv; vin.context_tokens = s;
  vin.latent = vlat.data(); vin.timesteps = vts.data(); vin.sigma = &vsig;
  vin.positions = vpos.data(); vin.context = vctx.data();
  Ltx2ModalityInput ain;
  ain.batch = 1; ain.tokens = ta; ain.context_tokens = s;
  ain.latent = alat.data(); ain.timesteps = ats.data(); ain.sigma = &asig;
  ain.positions = apos.data(); ain.context = actx.data();

  const auto t2 = std::chrono::steady_clock::now();
  const vllm::Ltx2DitOutputs out =
      Ltx2DitForwardDevice(q, p, ck.weights, &vin, &ain, vt::DType::kBF16);
  cuda->Synchronize(q);
  const auto t3 = std::chrono::steady_clock::now();
  MESSAGE("one 21.00B denoise-step forward at tv=" << tv << " ta=" << ta << ": "
          << std::chrono::duration<double>(t3 - t2).count()
          << " s -- SIZING ONLY, not a speed result (spec section 0: no "
             "production-configuration denominator exists for this family)");

  REQUIRE(out.video.size() == static_cast<size_t>(tv * p.out_channels));
  REQUIRE(out.audio.size() == static_cast<size_t>(ta * p.audio_out_channels));
  // The two ways a 21B forward fails without saying so: non-finite, or all-zero
  // (which is what a host pointer handed to a device GEMM produces).
  double vmax = 0.0, amax = 0.0;
  for (float v : out.video) {
    REQUIRE(std::isfinite(v));
    vmax = std::max(vmax, static_cast<double>(std::fabs(v)));
  }
  for (float v : out.audio) {
    REQUIRE(std::isfinite(v));
    amax = std::max(amax, static_cast<double>(std::fabs(v)));
  }
  MESSAGE("shipped-DiT output absmax: video=" << vmax << " audio=" << amax);
  CHECK(vmax > 1e-6);
  CHECK(amax > 1e-6);
}

TEST_CASE("ltx2 device: a single-stream model type is REFUSED") {
  vt::Queue q{Cpu(), nullptr};
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  const Ltx2DitDeviceWeights staged =
      Ltx2StageDitWeightsToDevice(q, p, set.views, vt::DType::kF32);
  Modalities m;
  BuildModalities(&m, false);
  CHECK_THROWS(
      (void)Ltx2DitForwardDevice(q, p, staged.weights, &m.video, nullptr, vt::DType::kF32));
  CHECK_THROWS(
      (void)Ltx2DitForwardDevice(q, p, staged.weights, nullptr, &m.audio, vt::DType::kF32));
}
