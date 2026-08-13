// LTX-2.5 — config parse, weight contract, RoPE, and the leaf bricks.
// Port of Lightricks LTX-2, packages/ltx-core/src/ltx_core/model/transformer/.
// See include/vllm/model_executor/models/ltx2.h for the full upstream map and for
// what phase L2 does NOT port.
//
// Everything here computes in f32: that is the PARITY dtype of the L2 gate (the
// oracle runs torch float32), not a widening of a bf16 path. Upstream resolves ONE
// model dtype for the whole DiT, so the production bf16/FP8/NVFP4 arms are a
// single stream-dtype choice and belong to phase L6.
#include "vllm/model_executor/models/ltx2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::DType;
using vt::Tensor;

// vt::MatmulBT + optional bias. `weight` is [out_features, in_features], torch's
// own nn.Linear layout, so y = x @ W^T + b reads straight off the module.
void Linear(vt::Queue& q, const float* in, int64_t rows, int64_t in_features,
            const Ltx2LinearWeight& w, float* out) {
  VT_CHECK(w.weight.rank == 2 && w.weight.shape[1] == in_features,
           "ltx2 linear: weight shape does not match input width");
  VT_CHECK(w.weight.dtype == DType::kF32, "ltx2 linear: L2 forward expects f32 weights");
  const int64_t out_features = w.weight.shape[0];
  Tensor a = Tensor::Contiguous(const_cast<float*>(in), DType::kF32, w.weight.device,
                                {rows, in_features});
  Tensor o = Tensor::Contiguous(out, DType::kF32, w.weight.device, {rows, out_features});
  vt::MatmulBT(q, o, a, w.weight);
  if (w.bias.data != nullptr) {
    const float* b = w.bias.Ptr<float>();
    for (int64_t r = 0; r < rows; ++r) {
      float* dst = out + r * out_features;
      for (int64_t i = 0; i < out_features; ++i) dst[i] += b[i];
    }
  }
}

// torch.nn.functional.rms_norm over the last dim (ltx_core/utils.py:7-12), with
// an optional elementwise weight (torch.nn.RMSNorm, attention.py:505-506).
void RmsNormRows(const float* in, const float* weight, float* out, int64_t rows, int64_t width,
                 double eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* src = in + r * width;
    double sum = 0.0;
    for (int64_t i = 0; i < width; ++i) sum += static_cast<double>(src[i]) * src[i];
    const float inv = static_cast<float>(1.0 / std::sqrt(sum / static_cast<double>(width) + eps));
    float* dst = out + r * width;
    if (weight == nullptr) {
      for (int64_t i = 0; i < width; ++i) dst[i] = src[i] * inv;
    } else {
      for (int64_t i = 0; i < width; ++i) dst[i] = src[i] * inv * weight[i];
    }
  }
}

float Silu(float x) { return x / (1.0f + std::exp(-x)); }

// torch.nn.functional.gelu(..., approximate="tanh"), the activation
// `activation_fn="gelu-approximate"` selects (gelu_approx.py:10).
float GeluTanh(float x) {
  const float kBeta = static_cast<float>(std::sqrt(2.0 / M_PI));
  const float kKappa = 0.044715f;
  const float inner = kBeta * (x + kKappa * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

const nlohmann::json& TransformerConfig(const nlohmann::json& metadata) {
  const nlohmann::json* cfg = &metadata;
  if (cfg->is_object() && cfg->contains("config")) cfg = &cfg->at("config");
  if (cfg->is_object() && cfg->contains("transformer")) cfg = &cfg->at("transformer");
  VT_CHECK(cfg->is_object(), "ltx2: could not locate the transformer config object");
  return *cfg;
}

// ltx_core/utils.py:15-18 check_config_value: the key must be PRESENT and equal.
// Upstream reads `config.get(key)`, so a missing key compares as None and raises;
// mirroring that keeps a truncated config from silently taking our defaults.
template <typename T>
void RequireConfigValue(const nlohmann::json& cfg, const char* key, const T& expected) {
  VT_CHECK(cfg.contains(key), "ltx2 config: required key is missing");
  VT_CHECK(cfg.at(key).is_null() == false, "ltx2 config: required key is null");
  VT_CHECK(cfg.at(key).get<T>() == expected, "ltx2 config: value does not match the expected one");
}

}  // namespace

// ---------------------------------------------------------------------------
// Config (model_configurator.py:19-83)
// ---------------------------------------------------------------------------

Ltx2DitParams ParseLtx2DitParams(const nlohmann::json& metadata) {
  const nlohmann::json& cfg = TransformerConfig(metadata);
  Ltx2DitParams p;

  // model_configurator.py:26-44 — every check_config_value the AV configurator runs.
  RequireConfigValue<double>(cfg, "dropout", 0.0);
  RequireConfigValue<bool>(cfg, "attention_bias", true);
  VT_CHECK(!cfg.contains("num_vector_embeds") || cfg.at("num_vector_embeds").is_null(),
           "ltx2 config: num_vector_embeds must be null");
  RequireConfigValue<std::string>(cfg, "activation_fn", "gelu-approximate");
  RequireConfigValue<int64_t>(cfg, "num_embeds_ada_norm", 1000);
  RequireConfigValue<bool>(cfg, "use_linear_projection", false);
  RequireConfigValue<bool>(cfg, "only_cross_attention", false);
  RequireConfigValue<bool>(cfg, "cross_attention_norm", true);
  RequireConfigValue<bool>(cfg, "double_self_attention", false);
  RequireConfigValue<bool>(cfg, "upcast_attention", false);
  RequireConfigValue<std::string>(cfg, "standardization_norm", "rms_norm");
  RequireConfigValue<bool>(cfg, "norm_elementwise_affine", false);
  RequireConfigValue<std::string>(cfg, "qk_norm", "rms_norm");
  RequireConfigValue<std::string>(cfg, "positional_embedding_type", "rope");
  RequireConfigValue<bool>(cfg, "use_audio_video_cross_attention", true);
  RequireConfigValue<bool>(cfg, "share_ff", false);
  RequireConfigValue<bool>(cfg, "av_cross_ada_norm", true);
  RequireConfigValue<bool>(cfg, "use_middle_indices_grid", true);

  auto get_int = [&](const char* key, int64_t& slot) {
    if (cfg.contains(key) && cfg.at(key).is_number()) slot = cfg.at(key).get<int64_t>();
  };
  auto get_double = [&](const char* key, double& slot) {
    if (cfg.contains(key) && cfg.at(key).is_number()) slot = cfg.at(key).get<double>();
  };
  auto get_bool = [&](const char* key, bool& slot) {
    if (cfg.contains(key) && cfg.at(key).is_boolean()) slot = cfg.at(key).get<bool>();
  };
  auto get_int_list = [&](const char* key, std::vector<int64_t>& slot) {
    if (!cfg.contains(key) || !cfg.at(key).is_array()) return;
    slot.clear();
    for (const auto& v : cfg.at(key)) slot.push_back(v.get<int64_t>());
  };

  get_int("num_attention_heads", p.num_attention_heads);
  get_int("attention_head_dim", p.attention_head_dim);
  get_int("in_channels", p.in_channels);
  get_int("out_channels", p.out_channels);
  get_int("num_layers", p.num_layers);
  get_int("cross_attention_dim", p.cross_attention_dim);
  get_double("norm_eps", p.norm_eps);
  get_double("positional_embedding_theta", p.positional_embedding_theta);
  get_int_list("positional_embedding_max_pos", p.positional_embedding_max_pos);
  get_int("timestep_scale_multiplier", p.timestep_scale_multiplier);
  get_int("audio_num_attention_heads", p.audio_num_attention_heads);
  get_int("audio_attention_head_dim", p.audio_attention_head_dim);
  get_int("audio_in_channels", p.audio_in_channels);
  get_int("audio_out_channels", p.audio_out_channels);
  get_int("audio_cross_attention_dim", p.audio_cross_attention_dim);
  get_int_list("audio_positional_embedding_max_pos", p.audio_positional_embedding_max_pos);
  get_int("av_ca_timestep_scale_multiplier", p.av_ca_timestep_scale_multiplier);
  get_bool("apply_gated_attention", p.apply_gated_attention);
  get_bool("cross_attention_adaln", p.cross_attention_adaln);
  get_bool("use_prompt_adaln_single", p.use_prompt_adaln_single);
  get_bool("ff_bias", p.ff_bias);
  get_bool("audio_ff_bias", p.audio_ff_bias);

  // model_configurator.py:44 — the AV configurator asserts the two head counts agree.
  VT_CHECK(p.num_attention_heads == p.audio_num_attention_heads,
           "ltx2 config: num_attention_heads must equal audio_num_attention_heads");

  // model_configurator.py:67 — LTXRopeType(config.get("rope_type", "split")).
  if (cfg.contains("rope_type")) {
    const std::string rope = cfg.at("rope_type").get<std::string>();
    if (rope == "split") {
      p.rope_type = Ltx2RopeType::kSplit;
    } else if (rope == "interleaved") {
      p.rope_type = Ltx2RopeType::kInterleaved;
    } else {
      VT_CHECK(false, "ltx2 config: rope_type must be \"split\" or \"interleaved\"");
    }
  }
  // model_configurator.py:68 — frequencies_precision == "float64".
  p.double_precision_rope = cfg.contains("frequencies_precision") &&
                            cfg.at("frequencies_precision").is_string() &&
                            cfg.at("frequencies_precision").get<std::string>() == "float64";

  // The two arms this phase does not carry are REFUSED by name rather than
  // silently ignored. Both are real upstream configurations.
  VT_CHECK(cfg.contains("caption_proj_before_connector") &&
               cfg.at("caption_proj_before_connector").is_boolean() &&
               cfg.at("caption_proj_before_connector").get<bool>(),
           "ltx2: caption_proj_before_connector=false puts the caption projections inside the DiT "
           "(19B form, text_projection.py:31-38); phase L3 ports them");
  VT_CHECK(!cfg.contains("use_keyframes_abs_pos_embedding") ||
               !cfg.at("use_keyframes_abs_pos_embedding").is_boolean() ||
               !cfg.at("use_keyframes_abs_pos_embedding").get<bool>(),
           "ltx2: use_keyframes_abs_pos_embedding is not ported (transformer_args.py:23-43); "
           "the LTX-2.5 checkpoint does not carry keyframes_abs_pos_embedding");

  VT_CHECK(p.num_attention_heads > 0 && p.attention_head_dim > 0 &&
               p.audio_num_attention_heads > 0 && p.audio_attention_head_dim > 0 &&
               p.num_layers > 0,
           "ltx2 config: geometry scalars must be positive");
  VT_CHECK(p.positional_embedding_max_pos.size() == 3,
           "ltx2 config: positional_embedding_max_pos must hold three values (t, h, w)");
  VT_CHECK(p.audio_positional_embedding_max_pos.size() == 1,
           "ltx2 config: audio_positional_embedding_max_pos must hold one value (t)");
  // transformer_args.py:364-371 builds the audio<->video cross RoPE at
  // audio_cross_attention_dim and applies it to heads of audio_attention_head_dim,
  // so the two widths have to agree.
  VT_CHECK(p.audio_cross_attention_dim == p.audio_inner_dim(),
           "ltx2 config: audio_cross_attention_dim must equal the audio stream width");
  return p;
}

// ---------------------------------------------------------------------------
// Weight contract
// ---------------------------------------------------------------------------

namespace {

void PushLinear(std::vector<Ltx2TensorSpec>& out, const std::string& prefix, int64_t out_features,
                int64_t in_features, bool bias) {
  out.push_back({prefix + ".weight", {out_features, in_features}});
  if (bias) out.push_back({prefix + ".bias", {out_features}});
}

// attention.py:505-518, in torch's own child-registration order (q_norm, k_norm,
// to_q, to_k, to_v, to_gate_logits, to_out) so the enumeration matches upstream's
// named_parameters() one for one.
void PushAttention(std::vector<Ltx2TensorSpec>& out, const std::string& prefix, int64_t query_dim,
                   int64_t context_dim, int64_t heads, int64_t dim_head, bool gated) {
  const int64_t inner = heads * dim_head;
  out.push_back({prefix + ".q_norm.weight", {inner}});
  out.push_back({prefix + ".k_norm.weight", {inner}});
  PushLinear(out, prefix + ".to_q", inner, query_dim, true);
  PushLinear(out, prefix + ".to_k", inner, context_dim, true);
  PushLinear(out, prefix + ".to_v", inner, context_dim, true);
  if (gated) PushLinear(out, prefix + ".to_gate_logits", heads, query_dim, true);
  PushLinear(out, prefix + ".to_out.0", query_dim, inner, true);
}

// feed_forward.py:9-12 — mult is fixed at 4 upstream.
void PushFeedForward(std::vector<Ltx2TensorSpec>& out, const std::string& prefix, int64_t dim,
                     bool bias) {
  PushLinear(out, prefix + ".net.0.proj", 4 * dim, dim, bias);
  PushLinear(out, prefix + ".net.2", dim, 4 * dim, bias);
}

// adaln.py:31-37 — the PixArt combined embedder is hard-wired to a 256-wide
// sinusoidal projection (timestep_embedding.py:133), whatever the model width is.
void PushAdaLayerNormSingle(std::vector<Ltx2TensorSpec>& out, const std::string& prefix,
                            int64_t dim, int64_t coefficient) {
  PushLinear(out, prefix + ".emb.timestep_embedder.linear_1", dim, 256, true);
  PushLinear(out, prefix + ".emb.timestep_embedder.linear_2", dim, dim, true);
  PushLinear(out, prefix + ".linear", coefficient * dim, dim, true);
}

}  // namespace

std::vector<Ltx2TensorSpec> EnumerateLtx2DitTensors(const Ltx2DitParams& p) {
  std::vector<Ltx2TensorSpec> out;
  const int64_t dim = p.inner_dim();
  const int64_t adim = p.audio_inner_dim();
  const int64_t coefficient = p.adaln_embedding_coefficient();
  const bool gated = p.apply_gated_attention;

  // torch lists a module's OWN parameters before its children, so the two output
  // tables come first (model.py:230, :260).
  out.push_back({"scale_shift_table", {2, dim}});
  out.push_back({"audio_scale_shift_table", {2, adim}});

  // _init_video (model.py:202-232), in child-registration order.
  PushLinear(out, "patchify_proj", dim, p.in_channels, true);
  PushAdaLayerNormSingle(out, "adaln_single", dim, coefficient);
  VT_CHECK(!p.use_prompt_adaln_single,
           "ltx2: use_prompt_adaln_single=true adds a prompt AdaLN MLP (model.py:223-227) whose "
           "timestep term makes the cross-attention K/V uncacheable; not ported in phase L2");
  PushLinear(out, "proj_out", p.out_channels, dim, true);

  // _init_audio (model.py:234-262).
  PushLinear(out, "audio_patchify_proj", adim, p.audio_in_channels, true);
  PushAdaLayerNormSingle(out, "audio_adaln_single", adim, coefficient);
  PushLinear(out, "audio_proj_out", p.audio_out_channels, adim, true);

  // _init_audio_video (model.py:264-287); num_scale_shift_values is 4 (:133).
  PushAdaLayerNormSingle(out, "av_ca_video_scale_shift_adaln_single", dim, 4);
  PushAdaLayerNormSingle(out, "av_ca_audio_scale_shift_adaln_single", adim, 4);
  PushAdaLayerNormSingle(out, "av_ca_a2v_gate_adaln_single", dim, 1);
  PushAdaLayerNormSingle(out, "av_ca_v2a_gate_adaln_single", adim, 1);

  for (int64_t i = 0; i < p.num_layers; ++i) {
    const std::string b = "transformer_blocks." + std::to_string(i);
    // The block's own parameters, in construction order (transformer.py:125, 150,
    // 177, 178, 185, 187).
    out.push_back({b + ".scale_shift_table", {coefficient, dim}});
    out.push_back({b + ".audio_scale_shift_table", {coefficient, adim}});
    out.push_back({b + ".scale_shift_table_a2v_ca_audio", {5, adim}});
    out.push_back({b + ".scale_shift_table_a2v_ca_video", {5, dim}});
    if (p.cross_attention_adaln) {
      out.push_back({b + ".prompt_scale_shift_table", {2, dim}});
      out.push_back({b + ".audio_prompt_scale_shift_table", {2, adim}});
    }
    // Children, in registration order (transformer.py:103-175).
    PushAttention(out, b + ".attn1", dim, dim, p.num_attention_heads, p.attention_head_dim, gated);
    PushAttention(out, b + ".attn2", dim, p.cross_attention_dim, p.num_attention_heads,
                  p.attention_head_dim, gated);
    PushFeedForward(out, b + ".ff", dim, p.ff_bias);
    PushAttention(out, b + ".audio_attn1", adim, adim, p.audio_num_attention_heads,
                  p.audio_attention_head_dim, gated);
    PushAttention(out, b + ".audio_attn2", adim, p.audio_cross_attention_dim,
                  p.audio_num_attention_heads, p.audio_attention_head_dim, gated);
    PushFeedForward(out, b + ".audio_ff", adim, p.audio_ff_bias);
    // The ASYMMETRIC pair: query/out width from the VIDEO stream, head geometry —
    // and therefore key/value width — from the AUDIO stream (transformer.py:154-175).
    PushAttention(out, b + ".audio_to_video_attn", dim, adim, p.audio_num_attention_heads,
                  p.audio_attention_head_dim, gated);
    PushAttention(out, b + ".video_to_audio_attn", adim, dim, p.audio_num_attention_heads,
                  p.audio_attention_head_dim, gated);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Shapes-as-config
// ---------------------------------------------------------------------------

namespace {

const std::vector<int64_t>* FindShape(const std::vector<Ltx2TensorSpec>& manifest,
                                      const std::string& name) {
  for (const Ltx2TensorSpec& spec : manifest) {
    if (spec.name == name) return &spec.shape;
  }
  return nullptr;
}

std::vector<int64_t> RequireShape(const std::vector<Ltx2TensorSpec>& manifest,
                                  const std::string& name, int64_t rank) {
  const std::vector<int64_t>* shape = FindShape(manifest, name);
  VT_CHECK(shape != nullptr,
           "ltx2 manifest: the tensor '" + name + "', which the geometry is derived from, "
           "is missing");
  VT_CHECK(static_cast<int64_t>(shape->size()) == rank,
           "ltx2 manifest: '" + name + "' has rank " + std::to_string(shape->size()) +
               ", expected " + std::to_string(rank));
  return *shape;
}

}  // namespace

Ltx2DitParams ParseLtx2DitParamsFromManifest(const std::vector<Ltx2TensorSpec>& manifest) {
  Ltx2DitParams p;

  const std::vector<int64_t> patch = RequireShape(manifest, "patchify_proj.weight", 2);
  const int64_t dim = patch[0];
  p.in_channels = patch[1];
  p.out_channels = RequireShape(manifest, "proj_out.weight", 2)[0];

  const std::vector<int64_t> apatch = RequireShape(manifest, "audio_patchify_proj.weight", 2);
  const int64_t adim = apatch[0];
  p.audio_in_channels = apatch[1];
  p.audio_out_channels = RequireShape(manifest, "audio_proj_out.weight", 2)[0];

  int64_t layers = 0;
  for (const Ltx2TensorSpec& spec : manifest) {
    const std::string prefix = "transformer_blocks.";
    if (spec.name.compare(0, prefix.size(), prefix) != 0) continue;
    const size_t dot = spec.name.find('.', prefix.size());
    if (dot == std::string::npos) continue;
    const int64_t index = std::stoll(spec.name.substr(prefix.size(), dot - prefix.size()));
    layers = std::max(layers, index + 1);
  }
  VT_CHECK(layers > 0, "ltx2 manifest: no transformer_blocks.* tensors");
  p.num_layers = layers;

  // The HEAD COUNT is only recoverable from a shape through the per-head gate
  // logits (attention.py:513-514, one row per head). Without gating the manifest
  // alone underdetermines it, and guessing would silently reshape every attention.
  const std::vector<int64_t>* gate = FindShape(manifest, "transformer_blocks.0.attn1.to_gate_logits.weight");
  VT_CHECK(gate != nullptr,
           "ltx2 manifest: the head count is only shape-derivable through to_gate_logits "
           "(attention.py:513-514); an ungated checkpoint must supply its transformer config");
  p.apply_gated_attention = true;
  p.num_attention_heads = (*gate)[0];
  const std::vector<int64_t>* agate =
      FindShape(manifest, "transformer_blocks.0.audio_attn1.to_gate_logits.weight");
  VT_CHECK(agate != nullptr, "ltx2 manifest: audio_attn1.to_gate_logits is missing");
  p.audio_num_attention_heads = (*agate)[0];

  VT_CHECK(p.num_attention_heads > 0 && dim % p.num_attention_heads == 0,
           "ltx2 manifest: the video width is not a whole number of heads");
  VT_CHECK(p.audio_num_attention_heads > 0 && adim % p.audio_num_attention_heads == 0,
           "ltx2 manifest: the audio width is not a whole number of heads");
  p.attention_head_dim = dim / p.num_attention_heads;
  p.audio_attention_head_dim = adim / p.audio_num_attention_heads;

  p.cross_attention_dim = RequireShape(manifest, "transformer_blocks.0.attn2.to_k.weight", 2)[1];
  p.audio_cross_attention_dim =
      RequireShape(manifest, "transformer_blocks.0.audio_attn2.to_k.weight", 2)[1];

  // adaln.py:14-16 — the block table's row count IS the AdaLN coefficient, and 9
  // rows is exactly the cross-attention-AdaLN form.
  const int64_t sst_rows = RequireShape(manifest, "transformer_blocks.0.scale_shift_table", 2)[0];
  VT_CHECK(sst_rows == 6 || sst_rows == 9,
           "ltx2 manifest: the block scale_shift_table must have 6 or 9 rows");
  p.cross_attention_adaln = sst_rows == 9;
  VT_CHECK(p.cross_attention_adaln ==
               (FindShape(manifest, "transformer_blocks.0.prompt_scale_shift_table") != nullptr),
           "ltx2 manifest: prompt_scale_shift_table presence disagrees with the AdaLN row count");

  p.use_prompt_adaln_single = FindShape(manifest, "prompt_adaln_single.linear.weight") != nullptr;
  p.ff_bias = FindShape(manifest, "transformer_blocks.0.ff.net.0.proj.bias") != nullptr;
  p.audio_ff_bias = FindShape(manifest, "transformer_blocks.0.audio_ff.net.0.proj.bias") != nullptr;

  // Everything below is NOT encoded in any shape — the RoPE flavour, the
  // frequency precision, the position ceilings, the timestep scaling and the
  // norm epsilon. They keep the LTX-2.5 defaults and must come from the config
  // when a checkpoint disagrees.
  return p;
}

// ---------------------------------------------------------------------------
// Weight binding
// ---------------------------------------------------------------------------

namespace {

Tensor Lookup(const std::map<std::string, Tensor>& tensors, const std::string& name) {
  auto it = tensors.find(name);
  // BY NAME, which is what ltx2.h:228-232 promises. The name is the whole point:
  // a caller assembling its own map has no other way to tell WHICH of 4078
  // parameters it forgot, and the alternative to a named refusal is a
  // zero-filled tensor that renders a plausible wrong video.
  VT_CHECK(it != tensors.end(),
           "ltx2: the weight map is missing the required tensor '" + name + "'");
  return it->second;
}

Ltx2LinearWeight BindLinear(const std::map<std::string, Tensor>& t, const std::string& prefix,
                            bool bias) {
  Ltx2LinearWeight w;
  w.weight = Lookup(t, prefix + ".weight");
  if (bias) w.bias = Lookup(t, prefix + ".bias");
  return w;
}

Ltx2AttentionWeights BindAttention(const std::map<std::string, Tensor>& t,
                                   const std::string& prefix, bool gated) {
  Ltx2AttentionWeights w;
  w.q_norm = Lookup(t, prefix + ".q_norm.weight");
  w.k_norm = Lookup(t, prefix + ".k_norm.weight");
  w.to_q = BindLinear(t, prefix + ".to_q", true);
  w.to_k = BindLinear(t, prefix + ".to_k", true);
  w.to_v = BindLinear(t, prefix + ".to_v", true);
  if (gated) w.to_gate_logits = BindLinear(t, prefix + ".to_gate_logits", true);
  w.to_out = BindLinear(t, prefix + ".to_out.0", true);
  return w;
}

Ltx2FeedForwardWeights BindFeedForward(const std::map<std::string, Tensor>& t,
                                       const std::string& prefix, bool bias) {
  Ltx2FeedForwardWeights w;
  w.proj_in = BindLinear(t, prefix + ".net.0.proj", bias);
  w.proj_out = BindLinear(t, prefix + ".net.2", bias);
  return w;
}

Ltx2AdaLayerNormSingleWeights BindAdaln(const std::map<std::string, Tensor>& t,
                                        const std::string& prefix) {
  Ltx2AdaLayerNormSingleWeights w;
  w.linear_1 = BindLinear(t, prefix + ".emb.timestep_embedder.linear_1", true);
  w.linear_2 = BindLinear(t, prefix + ".emb.timestep_embedder.linear_2", true);
  w.linear = BindLinear(t, prefix + ".linear", true);
  return w;
}

}  // namespace

Ltx2DitWeights BindLtx2DitWeights(const Ltx2DitParams& p,
                                  const std::map<std::string, Tensor>& t) {
  Ltx2DitWeights w;
  w.scale_shift_table = Lookup(t, "scale_shift_table");
  w.audio_scale_shift_table = Lookup(t, "audio_scale_shift_table");
  w.patchify_proj = BindLinear(t, "patchify_proj", true);
  w.adaln_single = BindAdaln(t, "adaln_single");
  w.proj_out = BindLinear(t, "proj_out", true);
  w.audio_patchify_proj = BindLinear(t, "audio_patchify_proj", true);
  w.audio_adaln_single = BindAdaln(t, "audio_adaln_single");
  w.audio_proj_out = BindLinear(t, "audio_proj_out", true);
  w.av_ca_video_scale_shift = BindAdaln(t, "av_ca_video_scale_shift_adaln_single");
  w.av_ca_audio_scale_shift = BindAdaln(t, "av_ca_audio_scale_shift_adaln_single");
  w.av_ca_a2v_gate = BindAdaln(t, "av_ca_a2v_gate_adaln_single");
  w.av_ca_v2a_gate = BindAdaln(t, "av_ca_v2a_gate_adaln_single");

  w.blocks.resize(static_cast<size_t>(p.num_layers));
  for (int64_t i = 0; i < p.num_layers; ++i) {
    Ltx2BlockWeights& b = w.blocks[static_cast<size_t>(i)];
    const std::string prefix = "transformer_blocks." + std::to_string(i);
    b.scale_shift_table = Lookup(t, prefix + ".scale_shift_table");
    b.audio_scale_shift_table = Lookup(t, prefix + ".audio_scale_shift_table");
    b.scale_shift_table_a2v_ca_audio = Lookup(t, prefix + ".scale_shift_table_a2v_ca_audio");
    b.scale_shift_table_a2v_ca_video = Lookup(t, prefix + ".scale_shift_table_a2v_ca_video");
    if (p.cross_attention_adaln) {
      b.prompt_scale_shift_table = Lookup(t, prefix + ".prompt_scale_shift_table");
      b.audio_prompt_scale_shift_table = Lookup(t, prefix + ".audio_prompt_scale_shift_table");
    }
    const bool gated = p.apply_gated_attention;
    b.attn1 = BindAttention(t, prefix + ".attn1", gated);
    b.attn2 = BindAttention(t, prefix + ".attn2", gated);
    b.ff = BindFeedForward(t, prefix + ".ff", p.ff_bias);
    b.audio_attn1 = BindAttention(t, prefix + ".audio_attn1", gated);
    b.audio_attn2 = BindAttention(t, prefix + ".audio_attn2", gated);
    b.audio_ff = BindFeedForward(t, prefix + ".audio_ff", p.audio_ff_bias);
    b.audio_to_video_attn = BindAttention(t, prefix + ".audio_to_video_attn", gated);
    b.video_to_audio_attn = BindAttention(t, prefix + ".video_to_audio_attn", gated);
  }
  return w;
}

// ---------------------------------------------------------------------------
// RoPE (rope.py)
// ---------------------------------------------------------------------------

namespace {

// generate_freq_grid_pytorch (rope.py:110-131): theta ** linspace(0, 1, n) * pi/2,
// the whole ladder in float32.
//
// The linspace itself is reproduced EXACTLY, not approximated: ATen's CPU
// `linspace_kernel` computes `step = (end - start) / (steps - 1)` in the tensor's
// OWN dtype and then walks the first half forward as `start + step * i` and the
// second half backward as `end - step * (steps - 1 - i)`, so both endpoints land
// exactly. Computing `i / (n - 1)` in double instead moves two of the eight audio
// samples by one f32 ulp, which the audio ladder multiplies up to a 1.5e-4
// frequency error and RoPE turns into a 1.2e-4 error in cos/sin — verified
// against `torch.linspace(0, 1, 8, dtype=torch.float32)` bit pattern by bit
// pattern.
std::vector<float> FreqGridPytorch(double theta, int64_t n_pos_dims, int64_t dim) {
  const int64_t n = dim / (2 * n_pos_dims);
  VT_CHECK(n > 0, "ltx2 rope: dim // (2 * n_pos_dims) must be positive");
  std::vector<float> out(static_cast<size_t>(n));
  const float step = n > 1 ? 1.0f / static_cast<float>(n - 1) : 0.0f;
  const int64_t halfway = n / 2;
  for (int64_t i = 0; i < n; ++i) {
    const float t = i < halfway ? step * static_cast<float>(i)
                                : 1.0f - step * static_cast<float>(n - 1 - i);
    out[static_cast<size_t>(i)] =
        std::pow(static_cast<float>(theta), t) * static_cast<float>(M_PI / 2.0);
  }
  return out;
}

// generate_freq_grid_np (rope.py:87-107): the SAME ladder built in numpy float64
// and only then cast to float32. Selected by frequencies_precision == "float64".
std::vector<float> FreqGridNumpy(double theta, int64_t n_pos_dims, int64_t dim) {
  const int64_t n = dim / (2 * n_pos_dims);
  VT_CHECK(n > 0, "ltx2 rope: dim // (2 * n_pos_dims) must be positive");
  std::vector<float> out(static_cast<size_t>(n));
  const double step = n > 1 ? 1.0 / static_cast<double>(n - 1) : 0.0;
  for (int64_t i = 0; i < n; ++i) {
    // numpy's linspace is arange(n) * step, with the final sample forced to `stop`.
    const double t = (i == n - 1) ? 1.0 : step * static_cast<double>(i);
    out[static_cast<size_t>(i)] = static_cast<float>(std::pow(theta, t) * (M_PI / 2.0));
  }
  return out;
}

}  // namespace

std::vector<float> Ltx2FreqGrid(double theta, int64_t n_pos_dims, int64_t dim,
                                bool double_precision) {
  return double_precision ? FreqGridNumpy(theta, n_pos_dims, dim)
                          : FreqGridPytorch(theta, n_pos_dims, dim);
}

Ltx2FreqsCis Ltx2PrecomputeFreqsCis(const double* positions, int64_t batch, int64_t tokens,
                                    int64_t n_pos_dims, int64_t source_n_pos_dims,
                                    bool use_middle_indices_grid, int64_t dim,
                                    const std::vector<int64_t>& max_pos, double theta,
                                    int64_t num_attention_heads, Ltx2RopeType rope_type,
                                    bool double_precision) {
  VT_CHECK(static_cast<int64_t>(max_pos.size()) == n_pos_dims,
           "ltx2 rope: max_pos length must equal the position dimension count");
  VT_CHECK(source_n_pos_dims >= n_pos_dims,
           "ltx2 rope: the position buffer carries fewer axes than the ladder consumes");
  const std::vector<float> indices = Ltx2FreqGrid(theta, n_pos_dims, dim, double_precision);
  const int64_t n = static_cast<int64_t>(indices.size());
  const int64_t stride = use_middle_indices_grid ? 2 : 1;

  // generate_freqs (rope.py:146-161): freqs[b, t, j * n_pos + i] =
  // indices[j] * (2 * position_fraction[b, t, i] - 1). The transpose before the
  // flatten puts the FREQUENCY on the slow axis and the position AXIS on the fast
  // one; swapping them silently rotates a different pair of channels.
  const int64_t width = n * n_pos_dims;
  std::vector<float> freqs(static_cast<size_t>(batch * tokens * width));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t t = 0; t < tokens; ++t) {
      for (int64_t i = 0; i < n_pos_dims; ++i) {
        const double* src = positions + (((b * source_n_pos_dims) + i) * tokens + t) * stride;
        const float mid = use_middle_indices_grid
                              ? static_cast<float>((src[0] + src[1]) / 2.0)
                              : static_cast<float>(src[0]);
        const float frac = mid / static_cast<float>(max_pos[static_cast<size_t>(i)]);
        const float scaled = frac * 2.0f - 1.0f;
        for (int64_t j = 0; j < n; ++j) {
          freqs[static_cast<size_t>((b * tokens + t) * width + j * n_pos_dims + i)] =
              indices[static_cast<size_t>(j)] * scaled;
        }
      }
    }
  }

  Ltx2FreqsCis out;
  if (rope_type == Ltx2RopeType::kSplit) {
    // split_freqs_cis (rope.py:164-184): left-pad cos with ones and sin with
    // zeros up to dim/2, then split into heads and move the head axis in front.
    const int64_t expected = dim / 2;
    const int64_t pad = expected - width;
    VT_CHECK(pad >= 0, "ltx2 rope: split RoPE frequency width exceeds dim/2");
    VT_CHECK(expected % num_attention_heads == 0,
             "ltx2 rope: dim/2 must split evenly across the heads");
    const int64_t per_head = expected / num_attention_heads;
    out.shape = {batch, num_attention_heads, tokens, per_head};
    out.cos.assign(static_cast<size_t>(batch * num_attention_heads * tokens * per_head), 0.0f);
    out.sin.assign(out.cos.size(), 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t c = 0; c < expected; ++c) {
          const float cos_v = c < pad ? 1.0f
                                      : std::cos(freqs[static_cast<size_t>((b * tokens + t) * width +
                                                                           (c - pad))]);
          const float sin_v = c < pad ? 0.0f
                                      : std::sin(freqs[static_cast<size_t>((b * tokens + t) * width +
                                                                           (c - pad))]);
          const int64_t h = c / per_head;
          const int64_t r = c % per_head;
          const size_t o = static_cast<size_t>(((b * num_attention_heads + h) * tokens + t) * per_head + r);
          out.cos[o] = cos_v;
          out.sin[o] = sin_v;
        }
      }
    }
  } else {
    // interleaved_freqs_cis (rope.py:187-195): each frequency is repeated twice,
    // then the whole row is left-padded to `dim`.
    const int64_t n_elem = 2 * n_pos_dims;
    const int64_t pad = dim % n_elem;
    VT_CHECK(2 * width + pad == dim, "ltx2 rope: interleaved RoPE width does not fill dim");
    out.shape = {batch, tokens, dim};
    out.cos.assign(static_cast<size_t>(batch * tokens * dim), 0.0f);
    out.sin.assign(out.cos.size(), 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t c = 0; c < dim; ++c) {
          const size_t o = static_cast<size_t>((b * tokens + t) * dim + c);
          if (c < pad) {
            out.cos[o] = 1.0f;
            out.sin[o] = 0.0f;
            continue;
          }
          const float f = freqs[static_cast<size_t>((b * tokens + t) * width + (c - pad) / 2)];
          out.cos[o] = std::cos(f);
          out.sin[o] = std::sin(f);
        }
      }
    }
  }
  return out;
}

void Ltx2ApplyRotaryEmb(float* x, int64_t batch, int64_t tokens, int64_t dim, int64_t heads,
                        const Ltx2FreqsCis& pe, Ltx2RopeType rope_type) {
  if (rope_type == Ltx2RopeType::kInterleaved) {
    // apply_interleaved_rotary_emb (rope.py:30-40): rotate the (even, odd) pairs.
    VT_CHECK(pe.shape.size() == 3 && pe.shape[2] == dim,
             "ltx2 rope: interleaved tables must be [batch, tokens, dim]");
    for (int64_t r = 0; r < batch * tokens; ++r) {
      float* row = x + r * dim;
      const float* cos = pe.cos.data() + r * dim;
      const float* sin = pe.sin.data() + r * dim;
      for (int64_t c = 0; c < dim; c += 2) {
        const float a = row[c], b = row[c + 1];
        row[c] = a * cos[c] + (-b) * sin[c];
        row[c + 1] = b * cos[c + 1] + a * sin[c + 1];
      }
    }
    return;
  }
  // apply_split_rotary_emb (rope.py:43-84). The input is [batch, tokens, dim] and
  // the tables are [batch, heads, tokens, per_head], so upstream reshapes the
  // input to [batch, heads, tokens, head_dim] first (:65). Within one head the
  // channels split into HALVES — the first half_dim channels pair with the second
  // half_dim, NOT with their neighbour (:67, d=2).
  VT_CHECK(pe.shape.size() == 4 && pe.shape[1] == heads,
           "ltx2 rope: split tables must be [batch, heads, tokens, per_head]");
  const int64_t head_dim = dim / heads;
  const int64_t per_head = pe.shape[3];
  VT_CHECK(head_dim == 2 * per_head, "ltx2 rope: split table width must be head_dim / 2");
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t t = 0; t < tokens; ++t) {
      float* row = x + (b * tokens + t) * dim;
      for (int64_t h = 0; h < heads; ++h) {
        float* v = row + h * head_dim;
        const size_t base = static_cast<size_t>(((b * heads + h) * tokens + t) * per_head);
        for (int64_t r = 0; r < per_head; ++r) {
          const float cos = pe.cos[base + static_cast<size_t>(r)];
          const float sin = pe.sin[base + static_cast<size_t>(r)];
          const float lo = v[r], hi = v[per_head + r];
          v[r] = lo * cos - hi * sin;
          v[per_head + r] = hi * cos + lo * sin;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Masks (transformer_args.py:199-237)
// ---------------------------------------------------------------------------

std::vector<float> Ltx2PrepareContextMask(const int32_t* mask, int64_t batch,
                                          int64_t context_tokens) {
  std::vector<float> out(static_cast<size_t>(batch * context_tokens));
  const float kMax = std::numeric_limits<float>::max();
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = (static_cast<float>(mask[i]) - 1.0f) * kMax;
  }
  return out;
}

std::vector<float> Ltx2PrepareSelfAttentionMask(const float* mask, int64_t count) {
  std::vector<float> out(static_cast<size_t>(count));
  const float kMin = std::numeric_limits<float>::lowest();
  const float kTiny = std::numeric_limits<float>::min();
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] =
        mask[i] > 0.0f ? std::log(std::max(mask[i], kTiny)) : kMin;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Leaf bricks
// ---------------------------------------------------------------------------

Ltx2AdalnOut Ltx2AdaLayerNormSingle(vt::Device device, const Ltx2AdaLayerNormSingleWeights& w,
                                    const float* timesteps, int64_t count, int64_t dim) {
  vt::Queue q{device, nullptr};
  // get_timestep_embedding (timestep_embedding.py:6-54) with num_channels=256,
  // flip_sin_to_cos=True, downscale_freq_shift=0, scale=1, max_period=10000
  // (:133), so the row is [cos(...) | sin(...)].
  const int64_t kChannels = 256;
  const int64_t half = kChannels / 2;
  std::vector<float> proj(static_cast<size_t>(count * kChannels));
  for (int64_t i = 0; i < half; ++i) {
    const float exponent =
        static_cast<float>(-std::log(10000.0)) * static_cast<float>(i) / static_cast<float>(half);
    const float freq = std::exp(exponent);
    for (int64_t r = 0; r < count; ++r) {
      const float v = timesteps[r] * freq;
      proj[static_cast<size_t>(r * kChannels + i)] = std::cos(v);
      proj[static_cast<size_t>(r * kChannels + half + i)] = std::sin(v);
    }
  }
  // TimestepEmbedding.forward (timestep_embedding.py:84-96): linear_1 -> SiLU -> linear_2.
  std::vector<float> hidden(static_cast<size_t>(count * dim));
  Linear(q, proj.data(), count, kChannels, w.linear_1, hidden.data());
  for (float& v : hidden) v = Silu(v);
  Ltx2AdalnOut out;
  out.embedded.resize(static_cast<size_t>(count * dim));
  Linear(q, hidden.data(), count, dim, w.linear_2, out.embedded.data());
  // AdaLayerNormSingle.forward (adaln.py:44-45): linear(silu(embedded_timestep)).
  std::vector<float> activated(out.embedded.size());
  for (size_t i = 0; i < activated.size(); ++i) activated[i] = Silu(out.embedded[i]);
  const int64_t coefficient_dim = w.linear.weight.shape[0];
  out.modulation.resize(static_cast<size_t>(count * coefficient_dim));
  Linear(q, activated.data(), count, dim, w.linear, out.modulation.data());
  return out;
}

std::vector<float> Ltx2FeedForward(vt::Device device, const Ltx2FeedForwardWeights& w,
                                   const float* x, int64_t rows, int64_t dim, int64_t inner) {
  vt::Queue q{device, nullptr};
  std::vector<float> hidden(static_cast<size_t>(rows * inner));
  Linear(q, x, rows, dim, w.proj_in, hidden.data());
  for (float& v : hidden) v = GeluTanh(v);
  std::vector<float> out(static_cast<size_t>(rows * dim));
  Linear(q, hidden.data(), rows, inner, w.proj_out, out.data());
  return out;
}

std::vector<float> Ltx2Attention(vt::Device device, const Ltx2AttentionWeights& w, const float* x,
                                 const float* context, const Ltx2AttentionArgs& args) {
  vt::Queue q{device, nullptr};
  const int64_t batch = args.batch;
  const int64_t tq = args.tokens;
  const int64_t heads = args.heads;
  const int64_t dim_head = args.dim_head;
  const int64_t inner = heads * dim_head;
  // attention.py:556 — `context = x if context is None else context`.
  const float* ctx = context != nullptr ? context : x;
  const int64_t s = context != nullptr ? args.context_tokens : tq;
  const int64_t ctx_dim = context != nullptr ? args.context_dim : args.query_dim;

  // attention.py:559-565: v first, then q and k. The K/V half is exactly what the
  // prompt cache holds, so `kv_in` skips all three of to_v / to_k / k_norm.
  const bool reuse_kv = args.kv_in != nullptr;
  std::vector<float> v;
  std::vector<float> kn;
  if (reuse_kv) {
    VT_CHECK(args.kv_in->k.size() == static_cast<size_t>(batch * s * inner) &&
                 args.kv_in->v.size() == static_cast<size_t>(batch * s * inner),
             "ltx2 attention: cached prompt K/V does not match this call's geometry");
    v = args.kv_in->v;
    kn = args.kv_in->k;
  } else {
    v.resize(static_cast<size_t>(batch * s * inner));
    Linear(q, ctx, batch * s, ctx_dim, w.to_v, v.data());
  }
  std::vector<float> qb(static_cast<size_t>(batch * tq * inner));
  Linear(q, x, batch * tq, args.query_dim, w.to_q, qb.data());

  // PytorchPreAttention (ops.py:22-37): the q/k RMSNorm runs over the FULL inner
  // width, before the head split, and RoPE follows it.
  std::vector<float> qn(qb.size());
  RmsNormRows(qb.data(), w.q_norm.Ptr<float>(), qn.data(), batch * tq, inner, args.norm_eps);
  if (!reuse_kv) {
    std::vector<float> kb(static_cast<size_t>(batch * s * inner));
    Linear(q, ctx, batch * s, ctx_dim, w.to_k, kb.data());
    kn.resize(kb.size());
    RmsNormRows(kb.data(), w.k_norm.Ptr<float>(), kn.data(), batch * s, inner, args.norm_eps);
  }
  if (args.pe != nullptr) {
    Ltx2ApplyRotaryEmb(qn.data(), batch, tq, inner, heads, *args.pe, args.rope_type);
    if (!reuse_kv) {
      const Ltx2FreqsCis& kpe = args.k_pe != nullptr ? *args.k_pe : *args.pe;
      Ltx2ApplyRotaryEmb(kn.data(), batch, s, inner, heads, kpe, args.rope_type);
    }
  }
  if (args.kv_out != nullptr) {
    args.kv_out->k = kn;
    args.kv_out->v = v;
  }

  std::vector<float> attn(static_cast<size_t>(batch * tq * inner));
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim_head)));
  for (int64_t b = 0; b < batch; ++b) {
    Tensor tq_t = Tensor::Contiguous(qn.data() + b * tq * inner, DType::kF32, device,
                                     {tq, heads, dim_head});
    Tensor tk_t = Tensor::Contiguous(kn.data() + b * s * inner, DType::kF32, device,
                                     {s, heads, dim_head});
    Tensor tv_t = Tensor::Contiguous(v.data() + b * s * inner, DType::kF32, device,
                                     {s, heads, dim_head});
    Tensor to_t = Tensor::Contiguous(attn.data() + b * tq * inner, DType::kF32, device,
                                     {tq, heads, dim_head});
    // Route on what the call MEANS, never on what its numbers happen to be.
    // `context == nullptr` is upstream's own self-attention marker
    // (attention.py:556), and an unbiased self-attention is exactly what the
    // shared dense op expresses. Keying this on `s == tq` instead made the
    // DISPATCHED OP depend on the prompt length — and on a device that carries a
    // kAttention kernel but no kAttentionCross one, that is the difference
    // between a call that runs and a call that refuses, from one request to the
    // next. The two ops agree bit-for-bit on the square unbiased problem, so this
    // is a dispatch decision and not an arithmetic one.
    if (context == nullptr && args.bias == nullptr) {
      vt::AttentionArgs a;
      a.scale = scale;
      a.causal = false;
      vt::Attention(q, to_t, tq_t, tk_t, tv_t, a);
    } else {
      vt::AttentionCrossArgs a;
      a.scale = scale;
      Tensor bias;
      if (args.bias != nullptr) {
        bias = Tensor::Contiguous(const_cast<float*>(args.bias) + b * args.bias_rows * s,
                                  DType::kF32, device, {args.bias_rows, s});
      }
      vt::AttentionCross(q, to_t, tq_t, tk_t, tv_t, args.bias != nullptr ? &bias : nullptr, a);
    }
  }

  // PytorchGatedAttention (ops.py:94-106), applied to the attention output BEFORE
  // `to_out` (attention.py:576-579) and driven by the RAW input `x`, not by the
  // attention output. Gating after `to_out` would be a different model.
  if (w.to_gate_logits.weight.data != nullptr) {
    std::vector<float> logits(static_cast<size_t>(batch * tq * heads));
    Linear(q, x, batch * tq, args.query_dim, w.to_gate_logits, logits.data());
    for (int64_t r = 0; r < batch * tq; ++r) {
      for (int64_t h = 0; h < heads; ++h) {
        const float gate = 2.0f / (1.0f + std::exp(-logits[static_cast<size_t>(r * heads + h)]));
        float* dst = attn.data() + r * inner + h * dim_head;
        for (int64_t e = 0; e < dim_head; ++e) dst[e] *= gate;
      }
    }
  }

  std::vector<float> out(static_cast<size_t>(batch * tq * args.query_dim));
  Linear(q, attn.data(), batch * tq, inner, w.to_out, out.data());
  return out;
}

}  // namespace vllm
