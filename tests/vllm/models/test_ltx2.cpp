// LTX-2.5 DiT parity gate (phase L2, .agents/specs/ltx-2-5.md, issue #435).
//
// Every assertion here compares our port against the UPSTREAM Lightricks LTX-2
// modules (packages/ltx-core/src/ltx_core/model/transformer/), IMPORTED AND
// EXECUTED at reduced dimensions by scripts/gen-ltx2-goldens.py and frozen into
// ltx2_goldens.inc. Both sides rebuild every weight and every input from the same
// deterministic FNV-1a + splitmix64 stream keyed by the parameter's own NAME, so
// no weight byte is checked in and the weight CONTRACT is part of the gate: a
// name either side invents that the other lacks changes the numbers.
//
// WHY A REDUCED-DIMENSION GATE. The shipped LTX-2.5 DiT is ~19 GB even in NVFP4
// and vLLM-Omni carries no native 2.5 path (spec section 3), so nothing here
// claims an end-to-end video result. What it does claim is exact: the parameter
// layout, the RoPE tables in both flavours and both frequency precisions, the
// per-head gated attention, the FFN bias asymmetry, the asymmetric cross-modal
// projections, and the full dual-stream DiT forward all reproduce upstream's
// numbers to f32 round-off.
#include "vllm/model_executor/models/ltx2.h"

#include "vllm/model_executor/models/ltx2_conditioning.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ltx2_goldens.inc"

#include "support/max_abs_diff.h"
#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using vllm::BindLtx2DitWeights;
using vllm::EnumerateLtx2DitTensors;
using vllm::Ltx2DitForward;
using vllm::Ltx2DitParams;
using vllm::Ltx2DitWeights;
using vllm::Ltx2ModalityInput;
using vllm::Ltx2PromptKvCache;
using vllm::Ltx2RopeType;
using vllm::Ltx2TensorSpec;
using vllm::ParseLtx2DitParams;
using vllm::ParseLtx2DitParamsFromManifest;

namespace {

vt::Device Cpu() { return vt::Device{}; }

// ---------------------------------------------------------------------------
// Ltx2Rand — the exact mirror of the generator's deterministic stream
// (scripts/gen-ltx2-goldens.py :: ltx2_rand).
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

// make_param: ltx2_rand * scale + offset in float64, then rounded to f32 (the
// generator's `.astype(np.float32)`).
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

// param_spec (gen-ltx2-goldens.py): keyed ONLY by the parameter's name.
std::vector<float> MakeNamedParam(const std::string& name, int64_t count) {
  if (EndsWith(name, "q_norm.weight") || EndsWith(name, "k_norm.weight")) {
    return MakeParam(name, count, 0.1, 1.0);
  }
  if (EndsWith(name, ".bias")) return MakeParam(name, count, 0.02, 0.0);
  return MakeParam(name, count, 0.05, 0.0);
}

// ---------------------------------------------------------------------------
// The reduced architecture, mirroring gen-ltx2-goldens.py :: ARCH.
// ---------------------------------------------------------------------------

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

// Upstream's DEFAULT arm (model.py:77, model_configurator.py:76/:138, diffusers
// transformer_ltx2.py:1185) and the one the shipped LTX-2.5 DiT carries: the
// prompt-side AdaLN MLP is built, so the cross-attention K/V modulation carries a
// timestep term on top of the static per-block table (transformer.py:441-443).
Ltx2DitParams ReducedParamsPromptAdaln(Ltx2RopeType rope_type, bool double_precision) {
  Ltx2DitParams p = ReducedParams(rope_type, double_precision);
  p.use_prompt_adaln_single = true;
  return p;
}

// A materialized weight set: owned f32 storage plus the views the forward takes.
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

// The generator's rand_input, by name.
std::vector<float> Input(const std::string& name, int64_t count, double scale, double offset) {
  return MakeParam(name, count, scale, offset);
}

// The text of the refusal `run` throws, or "" when it does not throw at all. A
// refusal is only useful if it NAMES what went wrong, so the message is part of
// the contract and is asserted, not just the fact that something was thrown.
template <typename Fn>
std::string RefusalMessage(Fn run) {
  try {
    run();
  } catch (const std::exception& e) {
    return std::string(e.what());
  }
  return std::string();
}

// The shared, NaN-hardened reduction. The local copy this replaces used
// `if (d > worst) worst = d;`, and NaN is never > anything, so an all-NaN result
// against a correct golden reduced to 0.0 and passed every bound (issue #449).
using vllm_test::MaxAbsDiff;

// Round-off floor for this gate. The oracle runs torch float32 and the port runs
// f32 GEMMs with f64 norm accumulation, so the two differ only in the last f32
// ulps of a ~1.0-magnitude activation carried through two blocks. Nothing here
// uses doctest's Approx: its `scale` defaults to 1.0, which puts a 1.19e-5
// ABSOLUTE floor under every comparison and would accept a broken forward.
constexpr double kRoundOff = 2e-6;

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
  // The DENSE (B, T, T) form (transformer_args.py:212-215) gives every QUERY its
  // own row of key strengths; the key-only (B, 1, T) broadcast has a single row
  // and so cannot tell a per-query bias index from a constant row-0 read.
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

nlohmann::json ReducedConfig() {
  nlohmann::json t;
  t["dropout"] = 0.0;
  t["attention_bias"] = true;
  t["num_vector_embeds"] = nullptr;
  t["activation_fn"] = "gelu-approximate";
  t["num_embeds_ada_norm"] = 1000;
  t["use_linear_projection"] = false;
  t["only_cross_attention"] = false;
  t["cross_attention_norm"] = true;
  t["double_self_attention"] = false;
  t["upcast_attention"] = false;
  t["standardization_norm"] = "rms_norm";
  t["norm_elementwise_affine"] = false;
  t["qk_norm"] = "rms_norm";
  t["positional_embedding_type"] = "rope";
  t["use_audio_video_cross_attention"] = true;
  t["share_ff"] = false;
  t["av_cross_ada_norm"] = true;
  t["use_middle_indices_grid"] = true;
  t["caption_proj_before_connector"] = true;
  t["num_attention_heads"] = vllm_test::kLtx2Arch_num_attention_heads;
  t["attention_head_dim"] = vllm_test::kLtx2Arch_attention_head_dim;
  t["in_channels"] = vllm_test::kLtx2Arch_in_channels;
  t["out_channels"] = vllm_test::kLtx2Arch_out_channels;
  t["num_layers"] = vllm_test::kLtx2Arch_num_layers;
  t["cross_attention_dim"] = vllm_test::kLtx2Arch_cross_attention_dim;
  t["audio_num_attention_heads"] = vllm_test::kLtx2Arch_audio_num_attention_heads;
  t["audio_attention_head_dim"] = vllm_test::kLtx2Arch_audio_attention_head_dim;
  t["audio_in_channels"] = vllm_test::kLtx2Arch_audio_in_channels;
  t["audio_out_channels"] = vllm_test::kLtx2Arch_audio_out_channels;
  t["audio_cross_attention_dim"] = vllm_test::kLtx2Arch_audio_cross_attention_dim;
  t["apply_gated_attention"] = true;
  t["cross_attention_adaln"] = true;
  t["use_prompt_adaln_single"] = false;
  t["ff_bias"] = false;
  t["audio_ff_bias"] = true;
  t["rope_type"] = "split";
  nlohmann::json metadata;
  metadata["config"]["transformer"] = t;
  return metadata;
}

}  // namespace

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

TEST_CASE("ltx2 layout: the weight contract matches upstream named_parameters()") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  const std::vector<Ltx2TensorSpec> manifest = EnumerateLtx2DitTensors(p);
  REQUIRE(static_cast<int64_t>(manifest.size()) == vllm_test::kLtx2ParamCount);
  size_t dim_cursor = 0;
  for (size_t i = 0; i < manifest.size(); ++i) {
    CAPTURE(i);
    CAPTURE(manifest[i].name);
    CHECK(manifest[i].name == std::string(vllm_test::kLtx2ParamNames[i]));
    const int64_t rank = vllm_test::kLtx2ParamRanks[i];
    REQUIRE(static_cast<int64_t>(manifest[i].shape.size()) == rank);
    for (int64_t d = 0; d < rank; ++d) {
      CHECK(manifest[i].shape[static_cast<size_t>(d)] == vllm_test::kLtx2ParamDims[dim_cursor]);
      ++dim_cursor;
    }
  }
}

TEST_CASE("ltx2 config: ParseLtx2DitParams mirrors LTXModelConfigurator") {
  const Ltx2DitParams parsed = ParseLtx2DitParams(ReducedConfig());
  const Ltx2DitParams expected = ReducedParams(Ltx2RopeType::kSplit, false);
  CHECK(parsed.num_attention_heads == expected.num_attention_heads);
  CHECK(parsed.attention_head_dim == expected.attention_head_dim);
  CHECK(parsed.in_channels == expected.in_channels);
  CHECK(parsed.out_channels == expected.out_channels);
  CHECK(parsed.num_layers == expected.num_layers);
  CHECK(parsed.cross_attention_dim == expected.cross_attention_dim);
  CHECK(parsed.audio_num_attention_heads == expected.audio_num_attention_heads);
  CHECK(parsed.audio_attention_head_dim == expected.audio_attention_head_dim);
  CHECK(parsed.audio_in_channels == expected.audio_in_channels);
  CHECK(parsed.audio_out_channels == expected.audio_out_channels);
  CHECK(parsed.audio_cross_attention_dim == expected.audio_cross_attention_dim);
  CHECK(parsed.apply_gated_attention);
  CHECK(parsed.cross_attention_adaln);
  CHECK_FALSE(parsed.use_prompt_adaln_single);
  CHECK_FALSE(parsed.ff_bias);
  CHECK(parsed.audio_ff_bias);
  CHECK(parsed.rope_type == Ltx2RopeType::kSplit);
  CHECK_FALSE(parsed.double_precision_rope);

  // THE INVISIBLE-CONSTANT CLASS, in the DiT. `norm_eps` feeds the q/k RMSNorm
  // (attention.py:505-506) and every AdaLN, but every arm in this suite passes it
  // EXPLICITLY through ReducedParams, so nothing here reads the FIELD DEFAULT and
  // a 100x mutation of it left all six LTX suites green. The default is not dead
  // code: `ReducedConfig()` carries no `norm_eps` key, which is the shape of a
  // checkpoint that omits it, and upstream's own fallback is
  // `config.get("norm_eps", 1e-06)` (transformer/model_configurator.py:54, 124,
  // 181). So the parse below is exactly the path the default binds on, and this
  // pins it there rather than in a list far from its use.
  CHECK(parsed.norm_eps == doctest::Approx(1e-6).epsilon(1e-12).scale(0.0));
  {
    // Not a SUBCASE deliberately: doctest re-enters the whole case body once per
    // subcase, so adding one here would multiply every assertion above it and
    // move this suite's recorded count for a reason unrelated to coverage.
    nlohmann::json explicit_eps = ReducedConfig();
    explicit_eps["config"]["transformer"]["norm_eps"] = 1e-5;
    CHECK(ParseLtx2DitParams(explicit_eps).norm_eps ==
          doctest::Approx(1e-5).epsilon(1e-12).scale(0.0));
  }

  SUBCASE("frequencies_precision selects the float64 ladder") {
    nlohmann::json cfg = ReducedConfig();
    cfg["config"]["transformer"]["frequencies_precision"] = "float64";
    CHECK(ParseLtx2DitParams(cfg).double_precision_rope);
  }
  SUBCASE("interleaved is accepted, anything else is refused") {
    nlohmann::json cfg = ReducedConfig();
    cfg["config"]["transformer"]["rope_type"] = "interleaved";
    CHECK(ParseLtx2DitParams(cfg).rope_type == Ltx2RopeType::kInterleaved);
    cfg["config"]["transformer"]["rope_type"] = "spiral";
    CHECK_THROWS(ParseLtx2DitParams(cfg));
  }
  SUBCASE("a check_config_value the configurator asserts is enforced here too") {
    nlohmann::json cfg = ReducedConfig();
    cfg["config"]["transformer"]["activation_fn"] = "gelu";
    CHECK_THROWS(ParseLtx2DitParams(cfg));
    nlohmann::json missing = ReducedConfig();
    missing["config"]["transformer"].erase("qk_norm");
    CHECK_THROWS(ParseLtx2DitParams(missing));
  }
  SUBCASE("the 19B caption-projection form is REFUSED, not silently ignored") {
    nlohmann::json cfg = ReducedConfig();
    cfg["config"]["transformer"]["caption_proj_before_connector"] = false;
    CHECK_THROWS(ParseLtx2DitParams(cfg));
  }
  // REPLACES "keyframe absolute-position embeddings are REFUSED". That refusal is
  // retired by row LTX25-KEYFRAMES-ABS-POS (issue #658): the flag is now READ, and
  // this asserts the value it produces in both directions plus its default, which
  // is strictly more than the refusal asserted.
  SUBCASE("use_keyframes_abs_pos_embedding is READ, not refused") {
    nlohmann::json on = ReducedConfig();
    on["config"]["transformer"]["use_keyframes_abs_pos_embedding"] = true;
    CHECK(ParseLtx2DitParams(on).use_keyframes_abs_pos_embedding);

    nlohmann::json off = ReducedConfig();
    off["config"]["transformer"]["use_keyframes_abs_pos_embedding"] = false;
    CHECK_FALSE(ParseLtx2DitParams(off).use_keyframes_abs_pos_embedding);

    // model_configurator.py:82/:142 — `config.get(..., False)`. A config that
    // omits the key means the model has no such parameter, which is what every
    // pre-2.5 checkpoint is.
    nlohmann::json absent = ReducedConfig();
    absent["config"]["transformer"].erase("use_keyframes_abs_pos_embedding");
    CHECK_FALSE(ParseLtx2DitParams(absent).use_keyframes_abs_pos_embedding);
  }
}

TEST_CASE("ltx2 dit: Ltx2AttentionArgs::norm_eps is a LATENT default, so it is pinned") {
  // The sixth instance the constant sweep turned up, and the most inert of them.
  // `Ltx2AttentionArgs::norm_eps` is the eps of the q/k RMSNorm — upstream's
  // `Attention.__init__` declares `norm_eps: float = 1e-6` (attention.py:485) and
  // hands it to both `torch.nn.RMSNorm`s (attention.py:505-506) — but EVERY
  // construction of the struct assigns it before use: ltx2_dit.cpp:188, :244,
  // :280, :338, :366 from `Ltx2DitParams::norm_eps`, ltx2_connector.cpp:253 from
  // `kLtx2ConnectorRmsNormEps`, and each of this suite's own arms from its
  // ReducedParams.
  //
  // Measured, not assumed: mutating this default 1e-6 -> 1.0, a 10^6 change,
  // leaves every suite green. That is not the invisible-epsilon story the other
  // five tell — those are read and merely never bind. This one is never READ, so
  // no fixture, however scaled, can reach it. It is a latent trap: the value a
  // future call site inherits on the day someone adds one and forgets the
  // assignment, at which point 1.0 would be silently applied inside an RMSNorm.
  // A pin is the only instrument that can hold it, and this records that limit
  // rather than dressing it up as coverage.
  CHECK(vllm::Ltx2AttentionArgs{}.norm_eps == doctest::Approx(1e-6).epsilon(1e-12).scale(0.0));
  // ...and it must agree with the DiT parameter that every real call site feeds
  // it from, so the two cannot drift apart unnoticed.
  CHECK(vllm::Ltx2AttentionArgs{}.norm_eps ==
        doctest::Approx(vllm::Ltx2DitParams{}.norm_eps).epsilon(1e-12).scale(0.0));
}

TEST_CASE("ltx2 layout: the shapes recover the geometry") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  const Ltx2DitParams derived = ParseLtx2DitParamsFromManifest(EnumerateLtx2DitTensors(p));
  CHECK(derived.num_layers == p.num_layers);
  CHECK(derived.num_attention_heads == p.num_attention_heads);
  CHECK(derived.attention_head_dim == p.attention_head_dim);
  CHECK(derived.in_channels == p.in_channels);
  CHECK(derived.out_channels == p.out_channels);
  CHECK(derived.cross_attention_dim == p.cross_attention_dim);
  CHECK(derived.audio_num_attention_heads == p.audio_num_attention_heads);
  CHECK(derived.audio_attention_head_dim == p.audio_attention_head_dim);
  CHECK(derived.audio_in_channels == p.audio_in_channels);
  CHECK(derived.audio_out_channels == p.audio_out_channels);
  CHECK(derived.audio_cross_attention_dim == p.audio_cross_attention_dim);
  CHECK(derived.apply_gated_attention == p.apply_gated_attention);
  CHECK(derived.cross_attention_adaln == p.cross_attention_adaln);
  CHECK(derived.use_prompt_adaln_single == p.use_prompt_adaln_single);
  // The FFN bias asymmetry is recovered from the SHAPES alone.
  CHECK(derived.ff_bias == false);
  CHECK(derived.audio_ff_bias == true);
}

TEST_CASE("ltx2 layout: a missing tensor throws by name rather than reading as zeros") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  std::map<std::string, vt::Tensor> views = set.views;
  views.erase("transformer_blocks.1.audio_to_video_attn.to_out.0.weight");
  CHECK_THROWS(BindLtx2DitWeights(p, views));
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

TEST_CASE("ltx2 rope: the frequency ladder is bit-for-bit torch's") {
  const int64_t dim = vllm_test::kLtx2Arch_num_attention_heads *
                      vllm_test::kLtx2Arch_attention_head_dim;
  const int64_t adim = vllm_test::kLtx2Arch_audio_num_attention_heads *
                       vllm_test::kLtx2Arch_audio_attention_head_dim;
  // The ladder is where a last-ulp linspace difference becomes a 1.5e-4
  // frequency error: the audio row spans four orders of magnitude, so BIT
  // equality is the only honest bar here, not a tolerance.
  const std::vector<float> video_f32 = vllm::Ltx2FreqGrid(10000.0, 3, dim, false);
  const std::vector<float> video_f64 = vllm::Ltx2FreqGrid(10000.0, 3, dim, true);
  const std::vector<float> audio_f32 = vllm::Ltx2FreqGrid(10000.0, 1, adim, false);
  const std::vector<float> audio_f64 = vllm::Ltx2FreqGrid(10000.0, 1, adim, true);
  REQUIRE(video_f32.size() == 5);
  REQUIRE(audio_f32.size() == 8);
  for (size_t i = 0; i < video_f32.size(); ++i) {
    CAPTURE(i);
    CHECK(video_f32[i] == vllm_test::kLtx2FreqGridVideoF32[i]);
    CHECK(video_f64[i] == vllm_test::kLtx2FreqGridVideoF64[i]);
  }
  for (size_t i = 0; i < audio_f32.size(); ++i) {
    CAPTURE(i);
    CHECK(audio_f32[i] == vllm_test::kLtx2FreqGridAudioF32[i]);
    CHECK(audio_f64[i] == vllm_test::kLtx2FreqGridAudioF64[i]);
  }
}

TEST_CASE("ltx2 rope: the frequency tables match precompute_freqs_cis") {
  const int64_t b = vllm_test::kLtx2Batch;
  const int64_t tv = vllm_test::kLtx2VideoTokens;
  const int64_t ta = vllm_test::kLtx2AudioTokens;
  std::vector<double> vpos(vllm_test::kLtx2VideoPositions,
                           vllm_test::kLtx2VideoPositions + b * 3 * tv * 2);
  std::vector<double> apos(vllm_test::kLtx2AudioPositions,
                           vllm_test::kLtx2AudioPositions + b * 1 * ta * 2);
  const int64_t dim = vllm_test::kLtx2Arch_num_attention_heads *
                      vllm_test::kLtx2Arch_attention_head_dim;
  const int64_t adim = vllm_test::kLtx2Arch_audio_num_attention_heads *
                       vllm_test::kLtx2Arch_audio_attention_head_dim;
  const int64_t heads = vllm_test::kLtx2Arch_num_attention_heads;
  const int64_t aheads = vllm_test::kLtx2Arch_audio_num_attention_heads;
  const int64_t cross_dim = vllm_test::kLtx2Arch_audio_cross_attention_dim;

  struct Case {
    const char* name;
    Ltx2RopeType type;
    bool double_precision;
    const float* vcos;
    const float* vsin;
    const float* acos;
    const float* asin;
    const float* ccos;
    const float* csin;
  };
  const Case cases[] = {
      {"split", Ltx2RopeType::kSplit, false, vllm_test::kLtx2RopeSplitVideoCos,
       vllm_test::kLtx2RopeSplitVideoSin, vllm_test::kLtx2RopeSplitAudioCos,
       vllm_test::kLtx2RopeSplitAudioSin, vllm_test::kLtx2RopeSplitCrossCos,
       vllm_test::kLtx2RopeSplitCrossSin},
      {"interleaved", Ltx2RopeType::kInterleaved, false,
       vllm_test::kLtx2RopeInterleavedVideoCos, vllm_test::kLtx2RopeInterleavedVideoSin,
       vllm_test::kLtx2RopeInterleavedAudioCos, vllm_test::kLtx2RopeInterleavedAudioSin,
       vllm_test::kLtx2RopeInterleavedCrossCos, vllm_test::kLtx2RopeInterleavedCrossSin},
      {"float64 frequencies", Ltx2RopeType::kSplit, true, vllm_test::kLtx2RopeDoubleVideoCos,
       vllm_test::kLtx2RopeDoubleVideoSin, vllm_test::kLtx2RopeDoubleAudioCos,
       vllm_test::kLtx2RopeDoubleAudioSin, vllm_test::kLtx2RopeDoubleCrossCos,
       vllm_test::kLtx2RopeDoubleCrossSin},
  };

  for (const Case& c : cases) {
    INFO("rope case: " << std::string(c.name));
    const vllm::Ltx2FreqsCis video = vllm::Ltx2PrecomputeFreqsCis(
        vpos.data(), b, tv, 3, 3, true, dim, {20, 2048, 2048}, 10000.0, heads, c.type,
        c.double_precision);
    CHECK(MaxAbsDiff(video.cos, c.vcos, video.cos.size()) < kRoundOff);
    CHECK(MaxAbsDiff(video.sin, c.vsin, video.sin.size()) < kRoundOff);

    const vllm::Ltx2FreqsCis audio = vllm::Ltx2PrecomputeFreqsCis(
        apos.data(), b, ta, 1, 1, true, adim, {20}, 10000.0, aheads, c.type,
        c.double_precision);
    CHECK(MaxAbsDiff(audio.cos, c.acos, audio.cos.size()) < kRoundOff);
    CHECK(MaxAbsDiff(audio.sin, c.asin, audio.sin.size()) < kRoundOff);

    // The audio<->video cross table: the TIME axis of the VIDEO grid only,
    // built at audio_cross_attention_dim (transformer_args.py:364-371).
    const vllm::Ltx2FreqsCis cross = vllm::Ltx2PrecomputeFreqsCis(
        vpos.data(), b, tv, 1, 3, true, cross_dim, {20}, 10000.0, heads, c.type,
        c.double_precision);
    CHECK(MaxAbsDiff(cross.cos, c.ccos, cross.cos.size()) < kRoundOff);
    CHECK(MaxAbsDiff(cross.sin, c.csin, cross.sin.size()) < kRoundOff);
  }
}

// ---------------------------------------------------------------------------
// Leaf bricks
// ---------------------------------------------------------------------------

TEST_CASE("ltx2 brick: AdaLayerNormSingle") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  const int64_t count = vllm_test::kLtx2Batch * vllm_test::kLtx2VideoTokens;
  const int64_t dim = p.inner_dim();
  std::vector<float> ts(vllm_test::kLtx2AdalnTimesteps,
                        vllm_test::kLtx2AdalnTimesteps + count);
  const vllm::Ltx2AdalnOut out =
      vllm::Ltx2AdaLayerNormSingle(Cpu(), set.weights.adaln_single, ts.data(), count, dim);
  CHECK(MaxAbsDiff(out.modulation, vllm_test::kLtx2AdalnModulation,
                   static_cast<size_t>(count * 9 * dim)) < kRoundOff);
  CHECK(MaxAbsDiff(out.embedded, vllm_test::kLtx2AdalnEmbedded,
                   static_cast<size_t>(count * dim)) < kRoundOff);
}

TEST_CASE("ltx2 brick: per-head gated attention") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  const int64_t b = vllm_test::kLtx2Batch;
  const int64_t tv = vllm_test::kLtx2VideoTokens;
  const int64_t dim = p.inner_dim();
  std::vector<float> x(vllm_test::kLtx2AttnGatedInput,
                       vllm_test::kLtx2AttnGatedInput + b * tv * dim);

  vllm::Ltx2AttentionArgs a;
  a.batch = b;
  a.tokens = tv;
  a.context_tokens = tv;
  a.query_dim = dim;
  a.context_dim = dim;
  a.heads = p.num_attention_heads;
  a.dim_head = p.attention_head_dim;
  a.norm_eps = p.norm_eps;
  const std::vector<float> out =
      vllm::Ltx2Attention(Cpu(), set.weights.blocks[0].attn1, x.data(), nullptr, a);
  CHECK(MaxAbsDiff(out, vllm_test::kLtx2AttnGatedOutput, static_cast<size_t>(b * tv * dim)) <
        kRoundOff);
}

// Which shared op a call site dispatches must follow from what the call MEANS,
// not from a coincidence in its numbers. A cross-attention whose prompt happens
// to carry exactly `tokens` keys is still a cross-attention; routing it to
// vt::Attention made the op — and therefore, on a device that has kAttention but
// no kAttentionCross, the SUCCESS OR FAILURE of the call — depend on the prompt
// length, so the same code path would serve one request and throw on the next.
TEST_CASE("ltx2 attention: a CROSS call routes to vt::AttentionCross at ANY prompt length") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  const int64_t b = vllm_test::kLtx2Batch;
  const int64_t tv = vllm_test::kLtx2VideoTokens;
  const int64_t dim = p.inner_dim();
  std::vector<float> x(vllm_test::kLtx2AttnGatedInput,
                       vllm_test::kLtx2AttnGatedInput + b * tv * dim);
  // The degenerate length: a prompt with EXACTLY as many keys as there are
  // queries. `attn2` is the text cross-attention, so context_dim == query_dim.
  std::vector<float> context(vllm_test::kLtx2A2vQuery, vllm_test::kLtx2A2vQuery + b * tv * dim);

  vllm::Ltx2AttentionArgs a;
  a.batch = b;
  a.tokens = tv;
  a.context_tokens = tv;  // S == Tq, and no mask: numerically square
  a.query_dim = dim;
  a.context_dim = dim;
  a.heads = p.num_attention_heads;
  a.dim_head = p.attention_head_dim;
  a.norm_eps = p.norm_eps;

  vt::EnableOpProviderCallStats(true);
  const unsigned long long cross_before =
      vt::GetOpProviderStats(vt::OpId::kAttentionCross, vt::DeviceType::kCPU).selections;
  const unsigned long long self_before =
      vt::GetOpProviderStats(vt::OpId::kAttention, vt::DeviceType::kCPU).selections;
  const std::vector<float> cross_out =
      vllm::Ltx2Attention(Cpu(), set.weights.blocks[0].attn2, x.data(), context.data(), a);
  const unsigned long long cross_after =
      vt::GetOpProviderStats(vt::OpId::kAttentionCross, vt::DeviceType::kCPU).selections;
  const unsigned long long self_after =
      vt::GetOpProviderStats(vt::OpId::kAttention, vt::DeviceType::kCPU).selections;
  CHECK(cross_after == cross_before + static_cast<unsigned long long>(b));
  CHECK(self_after == self_before);

  // SELF-attention with no bias keeps its shared seam: vt::Attention, as the
  // header says. The routing key is `context == nullptr`, not `S == Tq`.
  const unsigned long long self_mid = self_after;
  const std::vector<float> self_out =
      vllm::Ltx2Attention(Cpu(), set.weights.blocks[0].attn1, x.data(), nullptr, a);
  CHECK(vt::GetOpProviderStats(vt::OpId::kAttention, vt::DeviceType::kCPU).selections ==
        self_mid + static_cast<unsigned long long>(b));
  vt::EnableOpProviderCallStats(false);

  REQUIRE(cross_out.size() == static_cast<size_t>(b * tv * dim));
  REQUIRE(self_out.size() == cross_out.size());
}

// Routing is a DISPATCH choice and never an arithmetic one — which is what makes
// it safe to route on the call's meaning instead of on its numbers. On the
// unbiased square problem the two ops both accept, they must agree BIT-FOR-BIT.
TEST_CASE("vt::Attention and vt::AttentionCross agree BIT-for-BIT on a square unbiased call") {
  vt::Queue q{Cpu(), nullptr};
  const int64_t t = 6, h = 2, d = 4;
  std::vector<float> qb = Input("route.q", t * h * d, 0.5, 0.0);
  std::vector<float> kb = Input("route.k", t * h * d, 0.5, 0.0);
  std::vector<float> vb = Input("route.v", t * h * d, 0.5, 0.0);
  std::vector<float> dense_out(static_cast<size_t>(t * h * d), 0.0f);
  std::vector<float> cross_out(static_cast<size_t>(t * h * d), 0.0f);
  vt::Tensor tq = vt::Tensor::Contiguous(qb.data(), vt::DType::kF32, Cpu(), {t, h, d});
  vt::Tensor tk = vt::Tensor::Contiguous(kb.data(), vt::DType::kF32, Cpu(), {t, h, d});
  vt::Tensor tv = vt::Tensor::Contiguous(vb.data(), vt::DType::kF32, Cpu(), {t, h, d});
  vt::Tensor td = vt::Tensor::Contiguous(dense_out.data(), vt::DType::kF32, Cpu(), {t, h, d});
  vt::Tensor tc = vt::Tensor::Contiguous(cross_out.data(), vt::DType::kF32, Cpu(), {t, h, d});
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d)));
  vt::AttentionArgs dense;
  dense.scale = scale;
  dense.causal = false;
  vt::Attention(q, td, tq, tk, tv, dense);
  vt::AttentionCrossArgs cross;
  cross.scale = scale;
  vt::AttentionCross(q, tc, tq, tk, tv, nullptr, cross);
  for (size_t i = 0; i < dense_out.size(); ++i) {
    CAPTURE(i);
    CHECK(cross_out[i] == dense_out[i]);
  }
}

TEST_CASE("ltx2 brick: the asymmetric audio->video cross attention") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  const int64_t b = vllm_test::kLtx2Batch;
  const int64_t tv = vllm_test::kLtx2VideoTokens;
  const int64_t ta = vllm_test::kLtx2AudioTokens;
  const int64_t dim = p.inner_dim();
  const int64_t adim = p.audio_inner_dim();
  std::vector<float> q(vllm_test::kLtx2A2vQuery, vllm_test::kLtx2A2vQuery + b * tv * dim);
  std::vector<float> kv(vllm_test::kLtx2A2vContext,
                        vllm_test::kLtx2A2vContext + b * ta * adim);
  std::vector<double> vpos(vllm_test::kLtx2VideoPositions,
                           vllm_test::kLtx2VideoPositions + b * 3 * tv * 2);
  std::vector<double> apos(vllm_test::kLtx2AudioPositions,
                           vllm_test::kLtx2AudioPositions + b * 1 * ta * 2);

  const vllm::Ltx2FreqsCis vpe = vllm::Ltx2PrecomputeFreqsCis(
      vpos.data(), b, tv, 1, 3, true, p.audio_cross_attention_dim, {20}, 10000.0,
      p.num_attention_heads, Ltx2RopeType::kSplit, false);
  const vllm::Ltx2FreqsCis ape = vllm::Ltx2PrecomputeFreqsCis(
      apos.data(), b, ta, 1, 1, true, p.audio_cross_attention_dim, {20}, 10000.0,
      p.audio_num_attention_heads, Ltx2RopeType::kSplit, false);

  vllm::Ltx2AttentionArgs a;
  a.batch = b;
  a.tokens = tv;
  a.context_tokens = ta;
  a.query_dim = dim;    // Q from the VIDEO stream...
  a.context_dim = adim;  // ...K/V from the AUDIO stream
  a.heads = p.audio_num_attention_heads;
  a.dim_head = p.audio_attention_head_dim;
  a.norm_eps = p.norm_eps;
  a.pe = &vpe;
  a.k_pe = &ape;
  const std::vector<float> out = vllm::Ltx2Attention(
      Cpu(), set.weights.blocks[0].audio_to_video_attn, q.data(), kv.data(), a);
  CHECK(MaxAbsDiff(out, vllm_test::kLtx2A2vOutput, static_cast<size_t>(b * tv * dim)) <
        kRoundOff);
}

TEST_CASE("ltx2 brick: the feed-forward bias asymmetry") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  const int64_t b = vllm_test::kLtx2Batch;
  const int64_t tv = vllm_test::kLtx2VideoTokens;
  const int64_t ta = vllm_test::kLtx2AudioTokens;
  const int64_t dim = p.inner_dim();
  const int64_t adim = p.audio_inner_dim();

  // `ff` has NO bias on LTX-2.5 ...
  CHECK(set.weights.blocks[0].ff.proj_in.bias.data == nullptr);
  std::vector<float> x(vllm_test::kLtx2FfInput, vllm_test::kLtx2FfInput + b * tv * dim);
  const std::vector<float> out =
      vllm::Ltx2FeedForward(Cpu(), set.weights.blocks[0].ff, x.data(), b * tv, dim, 4 * dim);
  CHECK(MaxAbsDiff(out, vllm_test::kLtx2FfOutput, static_cast<size_t>(b * tv * dim)) < kRoundOff);

  // ... while `audio_ff` does.
  CHECK(set.weights.blocks[0].audio_ff.proj_in.bias.data != nullptr);
  std::vector<float> ax(vllm_test::kLtx2AudioFfInput,
                        vllm_test::kLtx2AudioFfInput + b * ta * adim);
  const std::vector<float> aout = vllm::Ltx2FeedForward(Cpu(), set.weights.blocks[0].audio_ff,
                                                        ax.data(), b * ta, adim, 4 * adim);
  CHECK(MaxAbsDiff(aout, vllm_test::kLtx2AudioFfOutput, static_cast<size_t>(b * ta * adim)) <
        kRoundOff);
}

// ---------------------------------------------------------------------------
// The full DiT forward
// ---------------------------------------------------------------------------

namespace {

void CheckForward(Ltx2RopeType rope_type, bool double_precision, bool masked,
                  const float* want_video, const float* want_audio, const char* label,
                  bool audio_enabled = true, bool dense_self_mask = false) {
  INFO("forward case: " << std::string(label));
  const Ltx2DitParams p = ReducedParams(rope_type, double_precision);
  WeightSet set = BuildWeights(p);
  Modalities m;
  BuildModalities(&m, masked, dense_self_mask);
  m.audio.enabled = audio_enabled;
  const vllm::Ltx2DitOutputs out =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32);
  const size_t vcount =
      static_cast<size_t>(m.video.batch * m.video.tokens * p.out_channels);
  const size_t acount =
      static_cast<size_t>(m.audio.batch * m.audio.tokens * p.audio_out_channels);
  const double vdiff = MaxAbsDiff(out.video, want_video, vcount);
  const double adiff = MaxAbsDiff(out.audio, want_audio, acount);
  MESSAGE("max|diff| video=" << vdiff << " audio=" << adiff);
  CHECK(vdiff < kRoundOff);
  CHECK(adiff < kRoundOff);
}

}  // namespace

TEST_CASE("ltx2 forward: split RoPE") {
  CheckForward(Ltx2RopeType::kSplit, false, false, vllm_test::kLtx2ForwardSplitVideo,
               vllm_test::kLtx2ForwardSplitAudio, "split");
}

TEST_CASE("ltx2 forward: interleaved RoPE") {
  CheckForward(Ltx2RopeType::kInterleaved, false, false,
               vllm_test::kLtx2ForwardInterleavedVideo,
               vllm_test::kLtx2ForwardInterleavedAudio, "interleaved");
}

TEST_CASE("ltx2 forward: the float64 frequency ladder") {
  CheckForward(Ltx2RopeType::kSplit, true, false, vllm_test::kLtx2ForwardDoubleVideo,
               vllm_test::kLtx2ForwardDoubleAudio, "float64 frequencies");
}

TEST_CASE("ltx2 forward: prompt mask + self-attention strength mask") {
  CheckForward(Ltx2RopeType::kSplit, false, true, vllm_test::kLtx2ForwardMaskedVideo,
               vllm_test::kLtx2ForwardMaskedAudio, "masked");
}

// The DENSE `(B, T, T)` self-attention mask — upstream's documented dense form
// (transformer_args.py:212-215), which the port accepts (`attention_mask_rows ==
// tokens`) and `vt::AttentionCross` validates as a `[Tq, S]` bias. Every query
// reads its OWN bias row here, so this case is what makes the per-query row
// index observable: with a key-only mask there is exactly one row, and a kernel
// that read row 0 for every query would be indistinguishable.
TEST_CASE("ltx2 forward: a DENSE (B, T, T) self-attention mask, per-query rows") {
  CheckForward(Ltx2RopeType::kSplit, false, true, vllm_test::kLtx2ForwardDenseMaskVideo,
               vllm_test::kLtx2ForwardDenseMaskAudio, "dense self-attention mask",
               /*audio_enabled=*/true, /*dense_self_mask=*/true);
}

TEST_CASE("ltx2 forward: a disabled audio stream still feeds audio->video") {
  // Modality.enabled=False skips the audio blocks (transformer.py:266) while the
  // audio->video cross attention keeps reading the audio state (:268) and the
  // audio head still runs over the untouched patchified latent.
  CheckForward(Ltx2RopeType::kSplit, false, false, vllm_test::kLtx2ForwardAudioOffVideo,
               vllm_test::kLtx2ForwardAudioOffAudio, "audio disabled",
               /*audio_enabled=*/false);
}

// ---------------------------------------------------------------------------
// The prompt-side AdaLN arm — upstream's DEFAULT
// (.agents/specs/ltx25-prompt-adaln.md, issue #644)
// ---------------------------------------------------------------------------

namespace {

// The same forward, run with `use_prompt_adaln_single = true`. Kept separate from
// CheckForward rather than folded into it, because the two arms have DIFFERENT
// weight contracts (12 extra parameters) and sharing one helper would hide which
// contract a case ran under.
vllm::Ltx2DitOutputs RunPromptAdalnForward(const Ltx2DitParams& p, WeightSet& set,
                                           Modalities* m, bool masked) {
  BuildModalities(m, masked);
  return Ltx2DitForward(Cpu(), p, set.weights, &m->video, &m->audio, vt::DType::kF32);
}

void CheckPromptAdalnForward(bool masked, const float* want_video, const float* want_audio,
                             const char* label) {
  INFO("prompt-AdaLN forward case: " << std::string(label));
  const Ltx2DitParams p = ReducedParamsPromptAdaln(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  const vllm::Ltx2DitOutputs out = RunPromptAdalnForward(p, set, &m, masked);
  const size_t vcount = static_cast<size_t>(m.video.batch * m.video.tokens * p.out_channels);
  const size_t acount =
      static_cast<size_t>(m.audio.batch * m.audio.tokens * p.audio_out_channels);
  const double vdiff = MaxAbsDiff(out.video, want_video, vcount);
  const double adiff = MaxAbsDiff(out.audio, want_audio, acount);
  MESSAGE("max|diff| video=" << vdiff << " audio=" << adiff);
  CHECK(vdiff < kRoundOff);
  CHECK(adiff < kRoundOff);
}

}  // namespace

// The 12 parameters the flag adds (6 per stream: three linears x weight+bias),
// which the shipped FP8 checkpoint carries as 18 manifest entries because each
// quantized weight also has a `weight_scale`. Order matters as much as presence:
// upstream registers `prompt_adaln_single` between `adaln_single` and `proj_out`
// inside `_init_video` (model.py:222-232), and the audio twin in `_init_audio`
// (:252-262).
TEST_CASE("ltx2 layout: the flag-ON contract matches upstream named_parameters()") {
  const Ltx2DitParams p = ReducedParamsPromptAdaln(Ltx2RopeType::kSplit, false);
  const std::vector<Ltx2TensorSpec> manifest = EnumerateLtx2DitTensors(p);
  REQUIRE(static_cast<int64_t>(manifest.size()) == vllm_test::kLtx2PromptAdalnParamCount);
  // The flag is the ONLY difference, so the count delta is exactly the module.
  CHECK(vllm_test::kLtx2PromptAdalnParamCount - vllm_test::kLtx2ParamCount == 12);
  size_t dim_cursor = 0;
  for (size_t i = 0; i < manifest.size(); ++i) {
    CAPTURE(i);
    CAPTURE(manifest[i].name);
    CHECK(manifest[i].name == std::string(vllm_test::kLtx2PromptAdalnParamNames[i]));
    const int64_t rank = vllm_test::kLtx2PromptAdalnParamRanks[i];
    REQUIRE(static_cast<int64_t>(manifest[i].shape.size()) == rank);
    for (int64_t d = 0; d < rank; ++d) {
      CHECK(manifest[i].shape[static_cast<size_t>(d)] ==
            vllm_test::kLtx2PromptAdalnParamDims[dim_cursor]);
      ++dim_cursor;
    }
  }
}

// The MLP on its own, so a failure localizes here rather than in the threading.
// The input is the modality's SIGMA scaled by timestep_scale_multiplier
// (transformer_args.py:274-277 -> :173-186) — one scalar per batch element, NOT
// the per-token `timesteps` the main AdaLN consumes.
TEST_CASE("ltx2 brick: the prompt AdaLN MLP runs on sigma, not on the per-token timesteps") {
  const Ltx2DitParams p = ReducedParamsPromptAdaln(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  const int64_t b = vllm_test::kLtx2Batch;
  const int64_t dim = p.inner_dim();
  const int64_t adim = p.audio_inner_dim();

  std::vector<float> vts(vllm_test::kLtx2PromptAdalnVideoTimesteps,
                         vllm_test::kLtx2PromptAdalnVideoTimesteps + b);
  const vllm::Ltx2AdalnOut vout = vllm::Ltx2AdaLayerNormSingle(
      Cpu(), set.weights.prompt_adaln_single, vts.data(), b, dim);
  REQUIRE(vout.modulation.size() == static_cast<size_t>(b * 2 * dim));
  CHECK(MaxAbsDiff(vout.modulation, vllm_test::kLtx2PromptAdalnVideoModulation,
                   static_cast<size_t>(b * 2 * dim)) < kRoundOff);

  std::vector<float> ats(vllm_test::kLtx2PromptAdalnAudioTimesteps,
                         vllm_test::kLtx2PromptAdalnAudioTimesteps + b);
  const vllm::Ltx2AdalnOut aout = vllm::Ltx2AdaLayerNormSingle(
      Cpu(), set.weights.audio_prompt_adaln_single, ats.data(), b, adim);
  REQUIRE(aout.modulation.size() == static_cast<size_t>(b * 2 * adim));
  CHECK(MaxAbsDiff(aout.modulation, vllm_test::kLtx2PromptAdalnAudioModulation,
                   static_cast<size_t>(b * 2 * adim)) < kRoundOff);
}

TEST_CASE("ltx2 forward: the prompt-side AdaLN arm") {
  CheckPromptAdalnForward(false, vllm_test::kLtx2ForwardPromptAdalnVideo,
                          vllm_test::kLtx2ForwardPromptAdalnAudio, "prompt AdaLN");
}

TEST_CASE("ltx2 forward: the prompt-side AdaLN arm, with both masks") {
  CheckPromptAdalnForward(true, vllm_test::kLtx2ForwardPromptAdalnMaskedVideo,
                          vllm_test::kLtx2ForwardPromptAdalnMaskedAudio,
                          "prompt AdaLN + masks");
}

// THE INSTRUMENT THAT MAKES THE ARM ABOVE MEAN SOMETHING.
//
// Both goldens come from the same deterministic weight stream, keyed by parameter
// NAME, so every weight the two arms share is bit-identical and the ONLY thing
// separating `kLtx2ForwardPromptAdaln*` from `kLtx2ForwardSplit*` is the timestep
// term. A port that accepted the flag, bound the 12 tensors and then never added
// their output would reproduce the flag-OFF numbers exactly and pass nothing here.
//
// WHAT THIS FIXTURE'S NUMBERS ARE, AND ARE NOT. The generator's stderr and the
// comment at the end of ltx2_goldens.inc report the VIDEO stream's term at 51.7%
// of its static per-block table — that ratio is emitted for the video stream ONLY,
// and the audio stream's own value on the same fixture is 40.6%, so the 51.7% is
// not a denominator for anything audio — the block-0 prompt K/V moving 5.82%, and
// the DiT output moving
// 1.46e-4 (73x kRoundOff). ALL FOUR are GATE-FLOOR numbers from SYNTHETIC weights,
// not a claim about the trained checkpoint: the table and the prompt-AdaLN MLP are
// both drawn at `param_spec`'s scale=0.05 (gen-ltx2-goldens.py:100-106), so every
// ratio is a property of THIS FIXTURE and moves with the init scale.
//
// On the SHIPPED DiT the term DOMINATES the table rather than halving it —
// rms|term|/rms|table| = 1347% video, 1583% audio, measured through upstream's own
// AdaLayerNormSingle on the real weights (.agents/specs/ltx25-prompt-adaln.md
// §Outcome). So this fixture UNDERSTATES the defect; it does not bound it.
//
// The bound below is set at 20x kRoundOff: comfortably inside the signal this
// fixture does produce, and comfortably outside f32 noise.
TEST_CASE("ltx2 forward: the prompt-AdaLN term is LOAD-BEARING, not decoration") {
  const Ltx2DitParams p = ReducedParamsPromptAdaln(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  const vllm::Ltx2DitOutputs out = RunPromptAdalnForward(p, set, &m, false);
  const size_t vcount = static_cast<size_t>(m.video.batch * m.video.tokens * p.out_channels);
  const size_t acount =
      static_cast<size_t>(m.audio.batch * m.audio.tokens * p.audio_out_channels);
  const double vdiff = MaxAbsDiff(out.video, vllm_test::kLtx2ForwardSplitVideo, vcount);
  const double adiff = MaxAbsDiff(out.audio, vllm_test::kLtx2ForwardSplitAudio, acount);
  MESSAGE("flag-ON vs flag-OFF: video=" << vdiff << " audio=" << adiff);
  CHECK(vdiff > 20.0 * kRoundOff);
  CHECK(adiff > 20.0 * kRoundOff);
}

// ---------------------------------------------------------------------------
// The keyframe absolute-position embedding
// (.agents/specs/ltx25-keyframes-abs-pos.md, issue #658)
// ---------------------------------------------------------------------------
//
// Upstream: `apply_keyframes_absolute_embedding` (transformer_args.py:23-43),
// called ONCE at :269 over the parameter `_init_video` builds at model.py:217-219.
// On a TRAINED checkpoint — which the shipped vonkaiser FP8 DiT is, 4096 of 4096
// bytes non-zero — it adds a learned per-token bias to every token the keyframes
// mask marks, on every forward.

namespace {

Ltx2DitParams ReducedParamsKeyframes(Ltx2RopeType rope_type, bool double_precision) {
  Ltx2DitParams p = ReducedParams(rope_type, double_precision);
  p.use_keyframes_abs_pos_embedding = true;
  return p;
}

// `_first_frame_keyframes_mask` at the fixture's geometry, taken from the GOLDEN
// rather than recomputed here: the golden is `zeros_like(denoise_mask)` with
// `[:, :tokens_per_latent_frame] = 1.0` as upstream wrote it (tools.py:194-195),
// and re-deriving it in the test would let both sides be wrong together.
std::vector<float> KeyframesMaskFromGolden() {
  const int64_t b = vllm_test::kLtx2Batch;
  const int64_t tv = vllm_test::kLtx2VideoTokens;
  std::vector<float> mask(static_cast<size_t>(b * tv));
  for (int64_t i = 0; i < b * tv; ++i) {
    // The golden is (B, T, 1); the port's field is [batch, tokens], the same
    // values with the trailing broadcast axis dropped.
    mask[static_cast<size_t>(i)] = vllm_test::kLtx2KeyframesMask[i];
  }
  return mask;
}

}  // namespace

// The parameter the flag adds, and WHERE it sits. Registration order is the half
// no shape encodes: `keyframes_abs_pos_embedding` is a module-OWN parameter
// created at model.py:217, BEFORE `scale_shift_table` at :230, so it leads
// `named_parameters()`.
TEST_CASE("ltx2 layout: the keyframes contract matches upstream named_parameters()") {
  const Ltx2DitParams p = ReducedParamsKeyframes(Ltx2RopeType::kSplit, false);
  const std::vector<Ltx2TensorSpec> manifest = EnumerateLtx2DitTensors(p);
  REQUIRE(static_cast<int64_t>(manifest.size()) == vllm_test::kLtx2KeyframesParamCount);
  // Exactly ONE parameter, which is what makes the count delta the module.
  CHECK(vllm_test::kLtx2KeyframesParamCount - vllm_test::kLtx2ParamCount == 1);
  size_t dim_cursor = 0;
  for (size_t i = 0; i < manifest.size(); ++i) {
    CAPTURE(i);
    CAPTURE(manifest[i].name);
    CHECK(manifest[i].name == std::string(vllm_test::kLtx2KeyframesParamNames[i]));
    const int64_t rank = vllm_test::kLtx2KeyframesParamRanks[i];
    REQUIRE(static_cast<int64_t>(manifest[i].shape.size()) == rank);
    for (int64_t d = 0; d < rank; ++d) {
      CHECK(manifest[i].shape[static_cast<size_t>(d)] ==
            vllm_test::kLtx2KeyframesParamDims[dim_cursor]);
      ++dim_cursor;
    }
  }
  // And the flag-OFF contract must not name it at all — with a positive control
  // in the same loop, so "found none" cannot be an artefact of looking for the
  // wrong string.
  const std::vector<Ltx2TensorSpec> off =
      EnumerateLtx2DitTensors(ReducedParams(Ltx2RopeType::kSplit, false));
  int64_t named = 0;
  int64_t control = 0;
  for (const Ltx2TensorSpec& spec : off) {
    if (spec.name == "keyframes_abs_pos_embedding") ++named;
    if (spec.name == "scale_shift_table") ++control;
  }
  CHECK(named == 0);
  CHECK(control == 1);
}

// THE MASK RULE, and the half most likely to be ported wrong. Upstream marks the
// target's first latent frame UNCONDITIONALLY — its own comment says so in terms
// (tools.py:190-191) — so a generation with NO keyframe supplied still carries the
// marker. A port that made this conditional would render every clip missing a
// trained term while every shape, every token count and every finite-value check
// stayed green.
TEST_CASE("ltx2 keyframes: the first latent frame is marked with NO keyframe supplied") {
  vllm::Ltx2VideoLatentShape shape;
  shape.batch = 1;
  shape.channels = 4;
  shape.frames = 3;
  shape.height = 2;
  shape.width = 2;

  // No keyframe, no conditioning item, no image — the plainest possible request.
  const std::vector<float> mask = vllm::Ltx2FirstFrameKeyframesMask(shape, /*patch_size=*/1);
  const int64_t tokens = vllm::Ltx2VideoTokenCount(shape, 1);
  REQUIRE(static_cast<int64_t>(mask.size()) == tokens);

  vllm::Ltx2VideoLatentShape one = shape;
  one.frames = 1;
  const int64_t per_frame = vllm::Ltx2VideoTokenCount(one, 1);
  REQUIRE(per_frame > 0);
  REQUIRE(per_frame < tokens);  // or "first frame" and "every token" coincide
  int64_t marked = 0;
  for (int64_t i = 0; i < tokens; ++i) {
    CAPTURE(i);
    CHECK(mask[static_cast<size_t>(i)] == (i < per_frame ? 1.0F : 0.0F));
    if (mask[static_cast<size_t>(i)] > 0.0F) ++marked;
  }
  CHECK(marked == per_frame);

  // The same rule the state builder applies, so the two cannot drift.
  vllm::Ltx2ScaleFactors factors;
  std::vector<float> from_state;
  (void)vllm::Ltx2CreateVideoLatentState(shape, /*patch_size=*/1, factors, /*fps=*/24.0,
                                         /*causal_fix=*/true, /*initial_latent=*/nullptr,
                                         &from_state);
  CHECK(from_state == mask);
}

// The three lines of maths on their own, against upstream's own function run on
// the identical `patchify_proj(latent)` rows.
TEST_CASE("ltx2 brick: apply_keyframes_absolute_embedding") {
  const Ltx2DitParams p = ReducedParamsKeyframes(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  const int64_t b = vllm_test::kLtx2Batch;
  const int64_t tv = vllm_test::kLtx2VideoTokens;
  const int64_t dim = p.inner_dim();

  // The parameter itself must be bound, and bound to the right tensor.
  REQUIRE(set.weights.keyframes_abs_pos_embedding.data != nullptr);
  REQUIRE(set.weights.keyframes_abs_pos_embedding.rank == 2);
  CHECK(set.weights.keyframes_abs_pos_embedding.shape[0] == 1);
  CHECK(set.weights.keyframes_abs_pos_embedding.shape[1] == dim);
  const std::vector<float> bound(set.weights.keyframes_abs_pos_embedding.Ptr<float>(),
                                 set.weights.keyframes_abs_pos_embedding.Ptr<float>() + dim);
  CHECK(MaxAbsDiff(bound, vllm_test::kLtx2KeyframesEmbedding, static_cast<size_t>(dim)) <
        kRoundOff);

  const std::vector<float> mask = KeyframesMaskFromGolden();
  const float* emb = vllm_test::kLtx2KeyframesEmbedding;

  // `hidden_states + mask * embedding`, computed the way the port computes it,
  // against upstream's own output for the same input.
  std::vector<float> got(vllm_test::kLtx2KeyframesHidden,
                         vllm_test::kLtx2KeyframesHidden + b * tv * dim);
  for (int64_t r = 0; r < b * tv; ++r) {
    if (!(mask[static_cast<size_t>(r)] > 0.0F)) continue;
    for (int64_t c = 0; c < dim; ++c) got[static_cast<size_t>(r * dim + c)] += emb[c];
  }
  CHECK(MaxAbsDiff(got, vllm_test::kLtx2KeyframesApplied, static_cast<size_t>(b * tv * dim)) <
        kRoundOff);

  // WHICH tokens moved, stated per row rather than as an aggregate: a marked row
  // must differ from its input by EXACTLY the bias, and an unmarked row must be
  // untouched. A port that applied the bias to the wrong frame passes an
  // aggregate norm and fails here.
  int64_t moved = 0;
  for (int64_t r = 0; r < b * tv; ++r) {
    CAPTURE(r);
    const bool marked = mask[static_cast<size_t>(r)] > 0.0F;
    for (int64_t c = 0; c < dim; ++c) {
      const size_t i = static_cast<size_t>(r * dim + c);
      const double want = static_cast<double>(vllm_test::kLtx2KeyframesHidden[i]) +
                          (marked ? static_cast<double>(emb[c]) : 0.0);
      CHECK(std::fabs(static_cast<double>(vllm_test::kLtx2KeyframesApplied[i]) - want) <
            kRoundOff);
    }
    if (marked) ++moved;
  }
  CHECK(moved == vllm_test::kLtx2KeyframesTokensPerLatentFrame * b);

  // The `keyframes_mask is None` exit (:37-38): upstream returns `hidden_states`
  // itself, so this is a BIT-for-BIT identity, not a tolerance.
  for (int64_t i = 0; i < b * tv * dim; ++i) {
    CAPTURE(i);
    CHECK(vllm_test::kLtx2KeyframesUnmarked[i] == vllm_test::kLtx2KeyframesHidden[i]);
  }
}

TEST_CASE("ltx2 forward: the keyframe marker arm") {
  const Ltx2DitParams p = ReducedParamsKeyframes(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  BuildModalities(&m, false);
  const std::vector<float> mask = KeyframesMaskFromGolden();
  m.video.keyframes_mask = mask.data();
  const vllm::Ltx2DitOutputs out =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32);
  const size_t vcount = static_cast<size_t>(m.video.batch * m.video.tokens * p.out_channels);
  const size_t acount =
      static_cast<size_t>(m.audio.batch * m.audio.tokens * p.audio_out_channels);
  const double vdiff = MaxAbsDiff(out.video, vllm_test::kLtx2ForwardKeyframesVideo, vcount);
  const double adiff = MaxAbsDiff(out.audio, vllm_test::kLtx2ForwardKeyframesAudio, acount);
  MESSAGE("max|diff| video=" << vdiff << " audio=" << adiff);
  CHECK(vdiff < kRoundOff);
  CHECK(adiff < kRoundOff);
}

// Flag ON, marker ABSENT — upstream's `keyframes_mask is None` early return
// carried all the way through a full forward. A port that applied the bias
// unconditionally (to every token, or whenever the parameter exists) reproduces
// the MARKED numbers here and fails.
TEST_CASE("ltx2 forward: the parameter alone applies NOTHING without the marker") {
  const Ltx2DitParams p = ReducedParamsKeyframes(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  BuildModalities(&m, false);
  REQUIRE(m.video.keyframes_mask == nullptr);
  const vllm::Ltx2DitOutputs out =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32);
  const size_t vcount = static_cast<size_t>(m.video.batch * m.video.tokens * p.out_channels);
  const size_t acount =
      static_cast<size_t>(m.audio.batch * m.audio.tokens * p.audio_out_channels);
  CHECK(MaxAbsDiff(out.video, vllm_test::kLtx2ForwardKeyframesNoMaskVideo, vcount) < kRoundOff);
  CHECK(MaxAbsDiff(out.audio, vllm_test::kLtx2ForwardKeyframesNoMaskAudio, acount) < kRoundOff);
}

// THE INSTRUMENT THAT MAKES THE ARM ABOVE MEAN SOMETHING, and the reason the
// generator TRAINS the parameter rather than leaving it at upstream's
// zero-initialization: a zero bias is an exact no-op, because the term is ADDED.
//
// Measured on this fixture (generator stderr, and the comment block at the end of
// ltx2_goldens.inc): supplying the marker moves the DiT's video output by 0.278,
// which is 71.5% of max|unmarked| and five orders of magnitude above kRoundOff.
// The audio row moves too, by 1.49%, and only through the audio<->video cross
// attention — nothing adds this bias to the audio stream.
TEST_CASE("ltx2 forward: the keyframe marker is LOAD-BEARING, not decoration") {
  const Ltx2DitParams p = ReducedParamsKeyframes(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  const size_t vcount = static_cast<size_t>(vllm_test::kLtx2Batch *
                                            vllm_test::kLtx2VideoTokens * p.out_channels);
  const size_t acount = static_cast<size_t>(vllm_test::kLtx2Batch *
                                            vllm_test::kLtx2AudioTokens * p.audio_out_channels);
  // Both operands are goldens, so the vector-taking overload does not apply;
  // `MaxAbsDiffScan` is the same reduction with the same non-finite polarity.
  const vllm_test::MaxAbsDiffScanResult vscan = vllm_test::MaxAbsDiffScan(
      vllm_test::kLtx2ForwardKeyframesNoMaskVideo, vllm_test::kLtx2ForwardKeyframesVideo, vcount);
  const vllm_test::MaxAbsDiffScanResult ascan = vllm_test::MaxAbsDiffScan(
      vllm_test::kLtx2ForwardKeyframesNoMaskAudio, vllm_test::kLtx2ForwardKeyframesAudio, acount);
  REQUIRE(vscan.ok());
  REQUIRE(ascan.ok());
  const double vdiff = vscan.worst;
  const double adiff = ascan.worst;
  MESSAGE("marked vs unmarked: video=" << vdiff << " audio=" << adiff);
  CHECK(vdiff > 20.0 * kRoundOff);
  CHECK(adiff > 20.0 * kRoundOff);

  // And the flag-OFF arm: with the parameter absent entirely, upstream's provider
  // yields None and the marker cannot apply, so the no-mask forward must equal
  // the flag-OFF forward BIT-for-BIT over the shared weight stream.
  for (size_t i = 0; i < vcount; ++i) {
    CAPTURE(i);
    CHECK(vllm_test::kLtx2ForwardKeyframesNoMaskVideo[i] == vllm_test::kLtx2ForwardSplitVideo[i]);
  }
}

// A marker handed to a stream that has no parameter is REFUSED, not dropped.
// Upstream builds the AUDIO args preprocessor with no keyframes_embedding_provider
// at all (model.py:333 against :314), so an audio marker cannot mean anything;
// and a model whose checkpoint omitted the parameter leaves it on `meta`
// (supports_keyframes_abs_pos_embedding, model.py:166-173).
TEST_CASE("ltx2 keyframes: a marker with no parameter is REFUSED, not silently dropped") {
  const std::vector<float> vmask = KeyframesMaskFromGolden();

  SUBCASE("the AUDIO stream never carries one") {
    const Ltx2DitParams p = ReducedParamsKeyframes(Ltx2RopeType::kSplit, false);
    WeightSet set = BuildWeights(p);
    Modalities m;
    BuildModalities(&m, false);
    std::vector<float> amask(
        static_cast<size_t>(vllm_test::kLtx2Batch * vllm_test::kLtx2AudioTokens), 1.0F);
    m.audio.keyframes_mask = amask.data();
    CHECK_THROWS(Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32));
  }

  SUBCASE("a flag-OFF model handed a video marker") {
    const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
    WeightSet set = BuildWeights(p);
    Modalities m;
    BuildModalities(&m, false);
    m.video.keyframes_mask = vmask.data();
    CHECK_THROWS(Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32));
  }

  SUBCASE("the flag set with the view UNBOUND, which is not a zero bias") {
    // The pairing IS `supports_keyframes_abs_pos_embedding` (model.py:166-173):
    // a config that says the parameter exists and a weight map that does not
    // carry it cannot both be believed. A default-constructed `vt::Tensor` here
    // would read as a zero-length bias — no bias at all — on a model whose
    // config says it has one.
    const Ltx2DitParams p = ReducedParamsKeyframes(Ltx2RopeType::kSplit, false);
    WeightSet set = BuildWeights(p);
    Modalities m;
    BuildModalities(&m, false);
    m.video.keyframes_mask = vmask.data();
    // Bound is fine — the positive control for the swap below.
    REQUIRE(set.weights.keyframes_abs_pos_embedding.data != nullptr);
    CHECK_NOTHROW(Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32));
    set.weights.keyframes_abs_pos_embedding = vt::Tensor{};
    CHECK_THROWS(Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32));
  }
}

// THE MEMORY-FORMAT CHECK. Upstream casts BOTH operands to `hidden_states.dtype`
// (transformer_args.py:42-43) and there is no wider accumulator anywhere on this
// path. `hidden_states` is f32 throughout this TU (the DTYPE note at the top of
// ltx2.h), so the addend must be f32 too — and the FP8 arm materializes this
// tensor as BF16 (MaterializeDitTensor's F8_E4M3 arm), reaching f32 only through
// `Ltx2WidenDitToF32`. A bf16 view read as f32 would consume two lanes per float:
// finite, plausible, and invisible to any output check. So it is asserted.
TEST_CASE("ltx2 keyframes: a bf16 embedding view is REFUSED by the f32 forward") {
  const Ltx2DitParams p = ReducedParamsKeyframes(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  BuildModalities(&m, false);
  const std::vector<float> mask = KeyframesMaskFromGolden();
  m.video.keyframes_mask = mask.data();

  // Same bytes, same shape, wrong DTYPE — which is exactly the shape of the
  // defect: a checkpoint loaded without `widen_to_f32`.
  const int64_t dim = p.inner_dim();
  std::vector<uint16_t> narrow(static_cast<size_t>(dim), 0x3F80);
  set.weights.keyframes_abs_pos_embedding =
      vt::Tensor::Contiguous(narrow.data(), vt::DType::kBF16, vt::Device{}, {1, dim});
  CHECK_THROWS(Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32));
}

// REWRITTEN by row LTX25-T2A-ONE-STAGE (#1005), and the rewrite is the point
// rather than an accommodation.
//
// This case used to assert that a single-stream call THROWS, pinning a refusal
// whose stated reason was "LTXModelType.VideoOnly and LTXModelType.AudioOnly
// carry a different weight contract". That reason is about the CHECKPOINT, and
// it does not describe `T2AOneStagePipeline`: upstream loads the ordinary
// AudioVideo file and restricts which keys it reads
// (LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP, model_configurator.py:228-239),
// then calls `LTXModel.forward(video=None, ...)` (t2a_one_stage.py:167).
//
// So the assertion is not widened, it is REPLACED with the contract upstream
// actually has (transformer.py:259-260, "At least one of video or audio must be
// provided") — and the new form is strictly stronger, because it also pins what
// a one-stream call RETURNS. The old one could not tell a served one-stream
// forward from a broken one; both threw.
TEST_CASE("ltx2 forward: ONE stream runs, and both-null is refused") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  BuildModalities(&m, false);

  // Upstream's own refusal, and the only one left.
  CHECK_THROWS(Ltx2DitForward(Cpu(), p, set.weights, nullptr, nullptr, vt::DType::kF32));

  // VIDEO ALONE. `run_a2v` is false because there is no audio state, so the
  // video output must still be the full sequence and the audio one EMPTY —
  // `Ltx2DitOutputs` carries two vectors and a build that filled both would be
  // reporting a stream it never ran.
  const vllm::Ltx2DitOutputs v_only =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, nullptr, vt::DType::kF32);
  CHECK(v_only.audio.empty());
  REQUIRE(v_only.video.size() ==
          static_cast<size_t>(m.video.batch * m.video.tokens * p.out_channels));
  for (const float x : v_only.video) REQUIRE(std::isfinite(x));

  // AUDIO ALONE — the shape `T2AOneStagePipeline` runs.
  const vllm::Ltx2DitOutputs a_only =
      Ltx2DitForward(Cpu(), p, set.weights, nullptr, &m.audio, vt::DType::kF32);
  CHECK(a_only.video.empty());
  REQUIRE(a_only.audio.size() ==
          static_cast<size_t>(m.audio.batch * m.audio.tokens * p.audio_out_channels));
  for (const float x : a_only.audio) REQUIRE(std::isfinite(x));

  // AND ONE STREAM IS NOT THE JOINT FORWARD WITH THE OTHER IGNORED. This is the
  // assertion the old case had no way to make, and it is the whole reason
  // `video = nullptr` is not `video->enabled = false`: upstream's `run_v2a` is
  // `run_ax and (video is not None and vx.numel() > 0)` (transformer.py:269), so
  // a present video stream — enabled or not — still feeds video->audio cross
  // attention. If these were equal, the cross-modal path would be dead on the
  // joint arm instead.
  const vllm::Ltx2DitOutputs joint =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32);
  REQUIRE(joint.audio.size() == a_only.audio.size());
  bool audio_differs = false;
  for (size_t i = 0; i < joint.audio.size(); ++i) {
    if (joint.audio[i] != a_only.audio[i]) audio_differs = true;
  }
  CHECK_MESSAGE(audio_differs,
                "the audio-only forward equals the joint one, so video->audio cross attention "
                "contributed nothing on the joint arm");
}

// ---------------------------------------------------------------------------
// The prompt-K/V cache (spec section 1.2)
// ---------------------------------------------------------------------------

TEST_CASE("ltx2 prompt K/V: cached and recomputed are BIT-IDENTICAL") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  BuildModalities(&m, false);

  const vllm::Ltx2DitOutputs fresh =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32);

  Ltx2PromptKvCache cache;
  const vllm::Ltx2DitOutputs first =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32, &cache);
  REQUIRE(cache.valid);
  REQUIRE(cache.video.size() == static_cast<size_t>(p.num_layers));
  REQUIRE(!cache.video[0].k.empty());
  // The SECOND call takes the cached K/V for every block and every stream.
  const vllm::Ltx2DitOutputs reused =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32, &cache);

  REQUIRE(fresh.video.size() == reused.video.size());
  REQUIRE(fresh.audio.size() == reused.audio.size());
  for (size_t i = 0; i < fresh.video.size(); ++i) {
    CAPTURE(i);
    // BIT-identical, not close: the cache reuses the exact bytes the recompute
    // would have produced, so any difference at all is a divergence.
    CHECK(first.video[i] == fresh.video[i]);
    CHECK(reused.video[i] == fresh.video[i]);
  }
  for (size_t i = 0; i < fresh.audio.size(); ++i) {
    CAPTURE(i);
    CHECK(first.audio[i] == fresh.audio[i]);
    CHECK(reused.audio[i] == fresh.audio[i]);
  }

  // The identity above would hold VACUOUSLY if the second call quietly recomputed
  // instead of reading the cache. Poison the cached bytes and require the output
  // to move: that is the only thing that proves they are actually consumed.
  for (auto& kv : cache.video) {
    for (float& value : kv.k) value += 1.0f;
  }
  const vllm::Ltx2DitOutputs poisoned =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32, &cache);
  bool moved = false;
  for (size_t i = 0; i < fresh.video.size(); ++i) {
    if (poisoned.video[i] != fresh.video[i]) moved = true;
  }
  CHECK(moved);
}

// The failure a SIZE check cannot see. Two requests whose prompts differ but
// whose token counts agree — the ordinary case for a server that reuses one
// cache — would otherwise render request 1's prompt for request 2, silently: no
// shape mismatch, no non-finite value, no error. The cache carries a prompt
// FINGERPRINT and refuses by name instead.
TEST_CASE("ltx2 prompt K/V: a CHANGED prompt of the SAME length is REFUSED, not served") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  BuildModalities(&m, false);

  const vllm::Ltx2DitOutputs first_prompt =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32);
  Ltx2PromptKvCache cache;
  Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32, &cache);
  REQUIRE(cache.valid);

  // A DIFFERENT prompt with the IDENTICAL geometry: negate every context value.
  for (float& value : m.video_context) value = -value;
  for (float& value : m.audio_context) value = -value;

  // GROUND TRUTH: the new prompt really does move the output, so a reuse that
  // reproduces the old numbers is a stale render and not a coincidence.
  const vllm::Ltx2DitOutputs second_prompt =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32);
  bool ground_truth_moved = false;
  for (size_t i = 0; i < first_prompt.video.size(); ++i) {
    if (second_prompt.video[i] != first_prompt.video[i]) ground_truth_moved = true;
  }
  REQUIRE(ground_truth_moved);

  CHECK_THROWS(
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32, &cache));
  // Refusing is half of it; the refusal has to say WHAT changed and what to do.
  CHECK(RefusalMessage([&] {
          Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32, &cache);
        }).find("video prompt's CONTENTS") != std::string::npos);
  CHECK(RefusalMessage([&] {
          Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32, &cache);
        }).find("Reset()") != std::string::npos);

  // Reset() is the documented repair: it rebinds the cache to the NEW prompt and
  // the result is bit-identical to a cache-free forward on that prompt.
  cache.Reset();
  const vllm::Ltx2DitOutputs rebound =
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32, &cache);
  REQUIRE(rebound.video.size() == second_prompt.video.size());
  for (size_t i = 0; i < second_prompt.video.size(); ++i) {
    CAPTURE(i);
    CHECK(rebound.video[i] == second_prompt.video[i]);
  }
  for (size_t i = 0; i < second_prompt.audio.size(); ++i) {
    CAPTURE(i);
    CHECK(rebound.audio[i] == second_prompt.audio[i]);
  }

  // The audio prompt alone is enough, and the prompt MASK counts as part of the
  // prompt: neither stream can hide behind the other.
  Modalities audio_only;
  BuildModalities(&audio_only, false);
  Ltx2PromptKvCache audio_cache;
  Ltx2DitForward(Cpu(), p, set.weights, &audio_only.video, &audio_only.audio, vt::DType::kF32,
                 &audio_cache);
  for (float& value : audio_only.audio_context) value = -value;
  CHECK_THROWS(Ltx2DitForward(Cpu(), p, set.weights, &audio_only.video, &audio_only.audio,
                              vt::DType::kF32, &audio_cache));
  CHECK(RefusalMessage([&] {
          Ltx2DitForward(Cpu(), p, set.weights, &audio_only.video, &audio_only.audio,
                         vt::DType::kF32, &audio_cache);
        }).find("audio prompt's CONTENTS") != std::string::npos);

  // A shorter prompt is caught too — by NAME, not by the size check inside the
  // attention, which reports geometry rather than identity.
  Modalities shorter;
  BuildModalities(&shorter, false);
  Ltx2PromptKvCache shorter_cache;
  Ltx2DitForward(Cpu(), p, set.weights, &shorter.video, &shorter.audio, vt::DType::kF32,
                 &shorter_cache);
  shorter.video.context_tokens -= 1;
  CHECK(RefusalMessage([&] {
          Ltx2DitForward(Cpu(), p, set.weights, &shorter.video, &shorter.audio, vt::DType::kF32,
                         &shorter_cache);
        }).find("video prompt's token count") != std::string::npos);
}

TEST_CASE("ltx2 prompt K/V: the prompt MASK is part of the prompt identity") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  BuildModalities(&m, true);
  Ltx2PromptKvCache cache;
  Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32, &cache);
  // Same context bytes, same lengths, one key unmasked: a different prompt.
  m.video_context_mask[0] = m.video_context_mask[0] != 0 ? 0 : 1;
  CHECK(RefusalMessage([&] {
          Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32, &cache);
        }).find("video prompt mask") != std::string::npos);
}

TEST_CASE("ltx2 prompt K/V: caching is REFUSED when the prompt AdaLN MLP is on") {
  Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  BuildModalities(&m, false);
  p.use_prompt_adaln_single = true;
  Ltx2PromptKvCache cache;
  CHECK_THROWS(
      Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kF32, &cache));
}

TEST_CASE("ltx2 forward: a non-f32 stream dtype is REFUSED, not silently widened") {
  const Ltx2DitParams p = ReducedParams(Ltx2RopeType::kSplit, false);
  WeightSet set = BuildWeights(p);
  Modalities m;
  BuildModalities(&m, false);
  CHECK_THROWS(Ltx2DitForward(Cpu(), p, set.weights, &m.video, &m.audio, vt::DType::kBF16));
}

// ---------------------------------------------------------------------------
// The shared seam this row added
// ---------------------------------------------------------------------------

TEST_CASE("vt::AttentionCross refuses what it cannot serve") {
  vt::Queue q{Cpu(), nullptr};
  std::vector<float> qb(4 * 2 * 3, 0.1f), kb(5 * 2 * 3, 0.2f), vb(5 * 2 * 3, 0.3f);
  std::vector<float> ob(4 * 2 * 3, 0.0f);
  vt::Tensor tq = vt::Tensor::Contiguous(qb.data(), vt::DType::kF32, Cpu(), {4, 2, 3});
  vt::Tensor tk = vt::Tensor::Contiguous(kb.data(), vt::DType::kF32, Cpu(), {5, 2, 3});
  vt::Tensor tv = vt::Tensor::Contiguous(vb.data(), vt::DType::kF32, Cpu(), {5, 2, 3});
  vt::Tensor to = vt::Tensor::Contiguous(ob.data(), vt::DType::kF32, Cpu(), {4, 2, 3});
  vt::AttentionCrossArgs args;
  args.scale = 0.5f;
  // Differing query and key token counts are the whole point of this op.
  CHECK_NOTHROW(vt::AttentionCross(q, to, tq, tk, tv, nullptr, args));
  // A bias whose column count does not match the key count is refused.
  std::vector<float> bad(4 * 4, 0.0f);
  vt::Tensor bias = vt::Tensor::Contiguous(bad.data(), vt::DType::kF32, Cpu(), {4, 4});
  CHECK_THROWS(vt::AttentionCross(q, to, tq, tk, tv, &bias, args));
  // An unset scale is refused rather than silently defaulting.
  vt::AttentionCrossArgs unscaled;
  CHECK_THROWS(vt::AttentionCross(q, to, tq, tk, tv, nullptr, unscaled));
}

// include/vt/ops.h documents that a device with no `kAttentionCross` provider
// REFUSES BY NAME rather than falling back to a host kernel over device memory.
// This is that claim, gated: a device type nothing registers this op on (and
// which `ReferenceTierEligible` rejects, since no backend in this build claims
// it) must throw, and the throw must name the op.
TEST_CASE("vt::AttentionCross: a device with no provider refuses BY NAME") {
  CHECK_FALSE(vt::OpRegistered(vt::OpId::kAttentionCross, vt::DeviceType::kXPU));
  const std::string msg = RefusalMessage(
      [] { (void)vt::GetOp(vt::OpId::kAttentionCross, vt::DeviceType::kXPU); });
  CHECK(msg.find("AttentionCross") != std::string::npos);
  CHECK(msg.find("xpu") != std::string::npos);
}

TEST_CASE("vt::AttentionCross: a DENSE [Tq, S] bias gives every query its OWN row") {
  vt::Queue q{Cpu(), nullptr};
  // Two IDENTICAL queries over two identical keys: the only thing that can make
  // their outputs differ is the bias row each one reads. A kernel that read row 0
  // for every query would return value row 0 twice.
  std::vector<float> qb = {1.0f, 1.0f};
  std::vector<float> kb = {1.0f, 1.0f};
  std::vector<float> vb = {10.0f, 20.0f};
  std::vector<float> ob(2, 0.0f);
  vt::Tensor tq = vt::Tensor::Contiguous(qb.data(), vt::DType::kF32, Cpu(), {2, 1, 1});
  vt::Tensor tk = vt::Tensor::Contiguous(kb.data(), vt::DType::kF32, Cpu(), {2, 1, 1});
  vt::Tensor tv = vt::Tensor::Contiguous(vb.data(), vt::DType::kF32, Cpu(), {2, 1, 1});
  vt::Tensor to = vt::Tensor::Contiguous(ob.data(), vt::DType::kF32, Cpu(), {2, 1, 1});
  // Row 0 keeps key 0 and drops key 1; row 1 does the opposite.
  std::vector<float> bias = {0.0f, -3.4028235e38f, -3.4028235e38f, 0.0f};
  vt::Tensor tb = vt::Tensor::Contiguous(bias.data(), vt::DType::kF32, Cpu(), {2, 2});
  vt::AttentionCrossArgs args;
  args.scale = 1.0f;
  vt::AttentionCross(q, to, tq, tk, tv, &tb, args);
  CHECK(std::fabs(ob[0] - 10.0f) < 1e-6f);
  CHECK(std::fabs(ob[1] - 20.0f) < 1e-6f);
}

// GQA broadcast (`g = h / (Hq / Hkv)`): a declared capability of this op that
// every LTX attention happens not to exercise, because Hq == Hkv everywhere in
// the model. Gate it here rather than ship it unreached.
TEST_CASE("vt::AttentionCross: Hq > Hkv broadcasts each kv-head to its query group") {
  vt::Queue q{Cpu(), nullptr};
  // Tq=1, Hq=4, Hkv=2, D=1, S=1. Query heads 0,1 must read kv-head 0 and query
  // heads 2,3 must read kv-head 1, so the output names its own group.
  std::vector<float> qb = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> kb = {1.0f, 1.0f};
  std::vector<float> vb = {7.0f, -3.0f};
  std::vector<float> ob(4, 0.0f);
  vt::Tensor tq = vt::Tensor::Contiguous(qb.data(), vt::DType::kF32, Cpu(), {1, 4, 1});
  vt::Tensor tk = vt::Tensor::Contiguous(kb.data(), vt::DType::kF32, Cpu(), {1, 2, 1});
  vt::Tensor tv = vt::Tensor::Contiguous(vb.data(), vt::DType::kF32, Cpu(), {1, 2, 1});
  vt::Tensor to = vt::Tensor::Contiguous(ob.data(), vt::DType::kF32, Cpu(), {1, 4, 1});
  vt::AttentionCrossArgs args;
  args.scale = 1.0f;
  vt::AttentionCross(q, to, tq, tk, tv, nullptr, args);
  // A single key means the softmax is 1.0, so each output IS its kv-head's value.
  CHECK(std::fabs(ob[0] - 7.0f) < 1e-6f);
  CHECK(std::fabs(ob[1] - 7.0f) < 1e-6f);
  CHECK(std::fabs(ob[2] + 3.0f) < 1e-6f);
  CHECK(std::fabs(ob[3] + 3.0f) < 1e-6f);
  // Hq that is not a multiple of Hkv has no defined grouping and is refused.
  std::vector<float> q3(3, 1.0f), o3(3, 0.0f);
  vt::Tensor tq3 = vt::Tensor::Contiguous(q3.data(), vt::DType::kF32, Cpu(), {1, 3, 1});
  vt::Tensor to3 = vt::Tensor::Contiguous(o3.data(), vt::DType::kF32, Cpu(), {1, 3, 1});
  CHECK_THROWS(vt::AttentionCross(q, to3, tq3, tk, tv, nullptr, args));
}

TEST_CASE("vt::AttentionCross: a fully masked key drops out of the softmax") {
  vt::Queue q{Cpu(), nullptr};
  // Two keys; the second is masked out, so the result must equal value row 0.
  std::vector<float> qb = {1.0f, 0.0f};
  std::vector<float> kb = {1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> vb = {3.0f, 5.0f, -7.0f, 11.0f};
  std::vector<float> ob(2, 0.0f);
  vt::Tensor tq = vt::Tensor::Contiguous(qb.data(), vt::DType::kF32, Cpu(), {1, 1, 2});
  vt::Tensor tk = vt::Tensor::Contiguous(kb.data(), vt::DType::kF32, Cpu(), {2, 1, 2});
  vt::Tensor tv = vt::Tensor::Contiguous(vb.data(), vt::DType::kF32, Cpu(), {2, 1, 2});
  vt::Tensor to = vt::Tensor::Contiguous(ob.data(), vt::DType::kF32, Cpu(), {1, 1, 2});
  std::vector<float> bias = {0.0f, -3.4028235e38f};
  vt::Tensor tb = vt::Tensor::Contiguous(bias.data(), vt::DType::kF32, Cpu(), {1, 2});
  vt::AttentionCrossArgs args;
  args.scale = 1.0f;
  vt::AttentionCross(q, to, tq, tk, tv, &tb, args);
  CHECK(std::fabs(ob[0] - 3.0f) < 1e-6f);
  CHECK(std::fabs(ob[1] - 5.0f) < 1e-6f);
}
