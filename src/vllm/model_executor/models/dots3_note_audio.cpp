// dots3-note AUDIO tower (W7a, #2703). Ported from
// `vllm/models/dots3_note/nvidia/audio_encoder.py` and `nvidia/audio.py` read
// in `~/_git/vllm` at `9035151d6`. See the header for the full provenance, for
// the ONE tracked exception to the `vt::Conv2d` seam, and for the one
// deliberate RMSNorm rounding difference.
#include "vllm/model_executor/models/dots3_note_audio.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/layers/linear.h"  // MlpGateUpMethodBase seam
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/dots3_note.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {

using vt::DType;
using vt::Tensor;
using namespace dense_attn;  // Dev / DBuf / MakeTensor / ResidentWeight

namespace {

using dense_loaders::LoadBf16Direct;

// ── config resolution ───────────────────────────────────────────────────────
//
// Same reader shape as `dots3_note_vision.cpp`'s: a MISSING key gets upstream's
// own default; a key that is PRESENT with the wrong json type is a refusal
// rather than a silent fallback, because a config that says
// `"use_causal": "false"` means something and it is not `false`.
int64_t ReadIntOr(const nlohmann::json& j, const char* key, int64_t dflt) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return dflt;
  VT_CHECK(it->is_number_integer(),
           std::string("dots3-note audio_config: `") + key +
               "` must be an integer, got " + it->dump());
  return it->get<int64_t>();
}
double ReadNumOr(const nlohmann::json& j, const char* key, double dflt) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return dflt;
  VT_CHECK(it->is_number(),
           std::string("dots3-note audio_config: `") + key +
               "` must be a number, got " + it->dump());
  return it->get<double>();
}
bool ReadBoolOr(const nlohmann::json& j, const char* key, bool dflt) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return dflt;
  VT_CHECK(it->is_boolean(),
           std::string("dots3-note audio_config: `") + key +
               "` must be a boolean, got " + it->dump());
  return it->get<bool>();
}
std::string ReadStrOr(const nlohmann::json& j, const char* key,
                      const std::string& dflt) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return dflt;
  VT_CHECK(it->is_string(),
           std::string("dots3-note audio_config: `") + key +
               "` must be a string, got " + it->dump());
  return it->get<std::string>();
}

std::string ShapeOf(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

void RequireAudioShape(const OwnedTensor& t, const std::string& name,
                       const std::vector<int64_t>& want) {
  std::vector<int64_t> got(t.shape, t.shape + t.rank);
  VT_CHECK(got == want, "dots3-note audio tower: '" + name + "' ships " +
                            ShapeOf(got) + ", the config implies " +
                            ShapeOf(want));
}

}  // namespace

Dots3NoteAudioParams ParseDots3NoteAudioParams(const HfConfig& config) {
  Dots3NoteAudioParams a;
  const auto ac = config.raw.find("audio_config");
  if (ac == config.raw.end() || !ac->is_object()) return a;
  a.present = true;
  const nlohmann::json& j = *ac;

  a.encoder_type = ReadStrOr(j, "encoder_type", "dots");
  a.use_conv2d_stem = ReadBoolOr(j, "use_conv2d_stem", true);
  a.use_rope = ReadBoolOr(j, "use_rope", true);
  a.use_rms_norm = ReadBoolOr(j, "use_rms_norm", true);
  a.use_causal = ReadBoolOr(j, "use_causal", false);
  a.downsample_hidden_size = ReadIntOr(j, "downsample_hidden_size", 480);
  a.merge_factor = ReadIntOr(j, "merge_factor", 1);
  a.chunk_seconds = ReadIntOr(j, "chunk_seconds", 60);
  // `whisper_adapter_in_dim` falls back to `adapter_in_dim` upstream
  // (`audio.py:30-35`), so both spellings are read here.
  a.adapter_in_dim = ReadIntOr(j, "whisper_adapter_in_dim",
                               ReadIntOr(j, "adapter_in_dim", 1280));
  a.adapter_out_dim = ReadIntOr(j, "whisper_adapter_out_dim",
                                ReadIntOr(j, "adapter_out_dim", 2048));
  a.conv_chunksize = ReadIntOr(j, "conv_chunksize", 500);
  a.conv_bucket_step = ReadIntOr(j, "conv_bucket_step", 0);
  a.conv_bucket_max_elements = ReadIntOr(j, "conv_bucket_max_elements", 0);

  const auto rp = j.find("rope_parameters");
  if (rp != j.end() && rp->is_object()) {
    a.partial_rotary_factor = ReadNumOr(*rp, "partial_rotary_factor", 1.0);
    a.rope_theta = ReadNumOr(*rp, "rope_theta", 10000.0);
    a.rope_type = ReadStrOr(*rp, "rope_type", "default");
  }

  const auto wc = j.find("whisper_config");
  if (wc != j.end() && wc->is_object()) {
    a.d_model = ReadIntOr(*wc, "d_model", 1280);
    a.num_heads = ReadIntOr(*wc, "encoder_attention_heads", 20);
    a.num_layers = ReadIntOr(*wc, "encoder_layers", 32);
    a.ffn_dim = ReadIntOr(*wc, "encoder_ffn_dim", 5120);
    a.num_mel_bins = ReadIntOr(*wc, "num_mel_bins", 128);
    a.max_source_positions = ReadIntOr(*wc, "max_source_positions", 6000);
    a.activation_function = ReadStrOr(*wc, "activation_function", "gelu");
    // `getattr(config, "use_latent_input", False)` (`audio_encoder.py:442`).
    // `Dots3NoteAudioConfig` never sets it, so a checkpoint can only reach it
    // through `whisper_config`, which is where it is read from.
    a.use_latent_input = ReadBoolOr(*wc, "use_latent_input", false);
    a.layer_norm_eps = ReadNumOr(*wc, "layer_norm_eps", 1e-5);
  }
  a.text_hidden_size = config.hidden_size;

  // Geometry that cannot be true of ANY dots3-note audio tower, checked where
  // the key name is still in hand. These are not owed to a brick; they are
  // configs upstream's own constructors would raise on.
  VT_CHECK(a.num_heads > 0 && a.d_model > 0 && a.d_model % a.num_heads == 0,
           "dots3-note audio_config: `whisper_config.d_model` " +
               std::to_string(a.d_model) +
               " is not a whole multiple of `encoder_attention_heads` " +
               std::to_string(a.num_heads) +
               " (audio_encoder.py:213-217 @ 9035151d6 raises on this)");
  VT_CHECK(a.num_layers > 0,
           "dots3-note audio_config: `whisper_config.encoder_layers` must be "
           "positive, got " + std::to_string(a.num_layers));
  VT_CHECK(a.num_mel_bins > 0,
           "dots3-note audio_config: `whisper_config.num_mel_bins` must be "
           "positive, got " + std::to_string(a.num_mel_bins));
  VT_CHECK(a.downsample_hidden_size > 0,
           "dots3-note audio_config: `downsample_hidden_size` must be "
           "positive, got " + std::to_string(a.downsample_hidden_size));
  // `use_conv2d_stem and use_latent_input` raises upstream
  // (`audio_encoder.py:450-453`), so it is a parse refusal rather than a brick.
  VT_CHECK(!(a.use_conv2d_stem && a.use_latent_input),
           "dots3-note audio_config: `use_conv2d_stem` and `use_latent_input` "
           "are mutually exclusive (audio_encoder.py:450-453 @ 9035151d6)");
  return a;
}

std::string Dots3NoteAudioRefusal(
    const Dots3NoteAudioParams& a, const std::string& quant_method,
    const std::vector<int64_t>& weight_block_size) {
  if (!a.present) {
    return "the checkpoint's `config.json` carries no `audio_config`, so this "
           "load has no audio tower to build (multimodal.py:119-126 @ "
           "9035151d6)";
  }
  // ORDER IS BRICK ORDER, and the message names ONE thing.
  if (!weight_block_size.empty() || quant_method == "fp8") {
    return "the checkpoint is BLOCKWISE-QUANTIZED "
           "(`quantization_config.weight_block_size`), and no blockwise-FP8 "
           "audio arm is ported. NOTE: the released "
           "`dots-studio/dots3-note-prev` ships all 430 `audio_encoder.*` "
           "tensors as BF16 and not one as FP8, measured in the committed "
           "shard index, so this branch is about a checkpoint nobody has "
           "published rather than about the released one. W9";
  }
  if (a.encoder_type != "dots") {
    return "`audio_config.encoder_type` is '" + a.encoder_type +
           "'; upstream itself supports only 'dots' and raises on anything "
           "else (nvidia/audio.py:255-256 @ 9035151d6)";
  }
  if (a.activation_function != "swiglu") {
    return "`whisper_config.activation_function` is '" + a.activation_function +
           "'; only 'swiglu' is ported. Upstream selects `ACT2FN[...]` and a "
           "SINGLE-width `fc1` for anything else (audio_encoder.py:325-334 @ "
           "9035151d6), which is a different state dict — `fc1` would be "
           "[ffn_dim, d_model] rather than [2 * ffn_dim, d_model] — so this is "
           "not a swap of one activation. The released checkpoint sets swiglu";
  }
  if (a.use_latent_input) {
    return "`use_latent_input` is true, which selects the LATENT stem "
           "(`_forward_latent_stem`, audio_encoder.py:598-609 @ 9035151d6): a "
           "GLU after conv1, stride 1, no downsampling at all, and a "
           "`latent_dim` input width instead of the mel bins. Unshipped arm";
  }
  if (!a.use_conv2d_stem) {
    return "`audio_config.use_conv2d_stem` is false, which selects the Conv1d "
           "stem (`_forward_conv1d_stem`, audio_encoder.py:585-596 @ "
           "9035151d6): two Conv1d layers at 2x downsampling instead of three "
           "Conv2d layers at 8x, a `conv1`/`conv2` state dict instead of "
           "`conv2d1`/`conv2d2`/`conv2d3`/`conv_out`, and a token stride of 320 "
           "instead of 1280. No published checkpoint sets it; refused rather "
           "than served from the conv2d path. Unshipped arm";
  }
  if (a.use_causal) {
    return "`audio_config.use_causal` is true. It changes THREE things at once "
           "— the conv padding becomes (1, 0), the stem left-pads time by 14, "
           "and every block's attention becomes causal "
           "(audio_encoder.py:465, :540-543, :319 @ 9035151d6) — so it is not "
           "a flag on the attention. The released `dots-studio/dots3-note-prev` "
           "sets it false and `Dots3NoteAudioConfig`'s own default is false. "
           "Unshipped arm";
  }
  if (!a.use_rms_norm) {
    return "`audio_config.use_rms_norm` is false, which makes every norm an "
           "`nn.LayerNorm` (audio_encoder.py:322, :336, :515 @ 9035151d6) — a "
           "different state dict, because a LayerNorm ships a BIAS the released "
           "checkpoint does not carry for any of its 65 norms. Unshipped arm";
  }
  if (!a.use_rope) {
    return "`audio_config.use_rope` is false, which builds a learned "
           "`embed_positions` table of [max_source_positions, d_model] "
           "(audio_encoder.py:509-510 @ 9035151d6) that the released "
           "checkpoint does not ship — its 430 tensors contain no positional "
           "embedding at all. Unshipped arm";
  }
  if (a.merge_factor != 1) {
    return "`audio_config.merge_factor` is " + std::to_string(a.merge_factor) +
           "; only 1 is ported. Upstream reshapes the encoder output to "
           "[B, T / merge, D * merge] before the adapter (nvidia/audio.py:"
           "269-276 @ 9035151d6), which multiplies the adapter's INPUT width, "
           "and the released `whisper_adapter_in_dim` 1280 is the unmerged "
           "one. Unshipped arm";
  }
  if (a.rotary_dim() <= 0 || a.rotary_dim() > a.head_dim()) {
    return "`rope_parameters.partial_rotary_factor` " +
           std::to_string(a.partial_rotary_factor) + " gives a rotary_dim of " +
           std::to_string(a.rotary_dim()) + " against a head_dim of " +
           std::to_string(a.head_dim()) +
           ", which rotates nothing or more than a head "
           "(audio_encoder.py:60-62 @ 9035151d6)";
  }
  if (a.adapter_in_dim != a.d_model) {
    return "`whisper_adapter_in_dim` " + std::to_string(a.adapter_in_dim) +
           " is not the encoder's `d_model` " + std::to_string(a.d_model) +
           ", so the adapter would normalize a width the trunk does not "
           "produce (nvidia/audio.py:258-261 @ 9035151d6)";
  }
  // ── THE ONE THE ENCODER ASSERTS ON ────────────────────────────────────────
  //
  // It names NO brick because nothing is owed: an adapter that does not land in
  // the TEXT hidden space emits rows that cannot be scattered into the prompt
  // at all. It is here, and not only in `EncodeMmDots3NoteForCausalLM`, because
  // a refusal predicate that is a strict SUBSET of the request-time asserts is
  // not a refusal — the request-time throw happens inside the engine's busy
  // loop and stops `AsyncLLM` for the life of the process. This row has already
  // found that defect twice (§4.11.3, and the sparse-routing entry under
  // `## Owed`); the two predicates are kept identical on purpose.
  if (a.adapter_out_dim != a.text_hidden_size) {
    return "`whisper_adapter_out_dim` " + std::to_string(a.adapter_out_dim) +
           " is not the TEXT tower's `hidden_size` " +
           std::to_string(a.text_hidden_size) +
           ", so `audio_adapter.proj.3` emits rows that cannot be scattered "
           "into the prompt at all. This is the comparison "
           "`EncodeMmDots3NoteForCausalLM` makes on a served request";
  }
  return "";
}

std::string Dots3NoteAudioRefusalFor(const HfConfig& config) {
  // The `quantization_config` block is read HERE rather than through
  // `ParseDots3NoteParams`, because this overload must answer for a checkpoint
  // whose LANGUAGE config the caller has not validated — the chat seam runs at
  // server start and holds only a path. Same shape as
  // `Dots3NoteVisionRefusalFor`.
  std::string quant_method;
  std::vector<int64_t> weight_block_size;
  const auto qc = config.raw.find("quantization_config");
  if (qc != config.raw.end() && qc->is_object()) {
    const auto qm = qc->find("quant_method");
    if (qm != qc->end() && qm->is_string()) quant_method = qm->get<std::string>();
    const auto wb = qc->find("weight_block_size");
    if (wb != qc->end() && wb->is_array()) {
      for (const nlohmann::json& e : *wb)
        if (e.is_number_integer()) weight_block_size.push_back(e.get<int64_t>());
    }
  }
  return Dots3NoteAudioRefusal(ParseDots3NoteAudioParams(config), quant_method,
                               weight_block_size);
}

std::vector<Dots3NoteTensor> EnumerateDots3NoteAudioTensors(
    const Dots3NoteAudioParams& a) {
  std::vector<Dots3NoteTensor> out;
  if (!a.present) return out;
  const std::string p = "audio_encoder.";
  const std::string se = p + "dots_encoder.speech_encoder.";
  out.push_back({se + "conv2d1.weight", "audio.stem.conv2d1"});
  out.push_back({se + "conv2d1.bias", "audio.stem.conv2d1"});
  out.push_back({se + "conv2d2.weight", "audio.stem.conv2d2"});
  out.push_back({se + "conv2d2.bias", "audio.stem.conv2d2"});
  out.push_back({se + "conv2d3.weight", "audio.stem.conv2d3"});
  out.push_back({se + "conv2d3.bias", "audio.stem.conv2d3"});
  // `bias=False` (audio_encoder.py:479), so NO `conv_out.bias` is claimed.
  out.push_back({se + "conv_out.weight", "audio.stem.conv_out"});
  for (int64_t l = 0; l < a.num_layers; ++l) {
    const std::string pre = se + "layers." + std::to_string(l) + ".";
    out.push_back({pre + "self_attn_layer_norm.weight", "audio.block.attn_norm"});
    out.push_back({pre + "final_layer_norm.weight", "audio.block.final_norm"});
    out.push_back({pre + "self_attn.q_proj.weight", "audio.block.attn.q"});
    out.push_back({pre + "self_attn.q_proj.bias", "audio.block.attn.q"});
    // NO `k_proj.bias`: `nn.Linear(embed_dim, embed_dim, bias=False)`
    // (audio_encoder.py:221). Claiming one would refuse every real checkpoint.
    out.push_back({pre + "self_attn.k_proj.weight", "audio.block.attn.k"});
    out.push_back({pre + "self_attn.v_proj.weight", "audio.block.attn.v"});
    out.push_back({pre + "self_attn.v_proj.bias", "audio.block.attn.v"});
    out.push_back({pre + "self_attn.out_proj.weight", "audio.block.attn.out"});
    out.push_back({pre + "self_attn.out_proj.bias", "audio.block.attn.out"});
    out.push_back({pre + "fc1.weight", "audio.block.mlp.gate_up"});
    out.push_back({pre + "fc1.bias", "audio.block.mlp.gate_up"});
    out.push_back({pre + "fc2.weight", "audio.block.mlp.down"});
    out.push_back({pre + "fc2.bias", "audio.block.mlp.down"});
  }
  out.push_back({se + "layer_norm.weight", "audio.final_norm"});
  const std::string ad = p + "audio_adapter.proj.";
  out.push_back({ad + "0.weight", "audio.adapter.ln"});
  out.push_back({ad + "0.bias", "audio.adapter.ln"});
  out.push_back({ad + "1.weight", "audio.adapter.fc1"});
  out.push_back({ad + "1.bias", "audio.adapter.fc1"});
  out.push_back({ad + "3.weight", "audio.adapter.fc2"});
  out.push_back({ad + "3.bias", "audio.adapter.fc2"});
  return out;
}

Dots3NoteAudioWeights MaterializeDots3NoteAudio(
    const std::vector<SafetensorsFile>& shards,
    const Dots3NoteAudioParams& a) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& f : shards) {
    for (const std::string& n : f.Names()) where.emplace(n, &f);
  }
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(),
             "dots3-note audio tower: tensor not found: " + name);
    return it->second->Get(name);
  };

  const int64_t D = a.d_model, F = a.ffn_dim, dhs = a.downsample_hidden_size;
  const std::string p = "audio_encoder.";
  const std::string se = p + "dots_encoder.speech_encoder.";
  Dots3NoteAudioWeights w;

  // The stem. Every conv weight ships rank-4 [out, in, 3, 3] as torch stores it
  // and is READ AS [out, in*9]: the im2col composition multiplies a
  // [rows, in*9] column matrix by it with `vt::MatmulBT`, and that
  // reinterpretation is exactly the one `conv2d` performs internally. The
  // ON-DISK rank-4 shape is checked first so a checkpoint with a different
  // kernel size refuses by name instead of being reshaped into agreement.
  const auto load_conv = [&](const std::string& name, int64_t out_ch,
                             int64_t in_ch) {
    const StTensor& t = get(name);
    const std::vector<int64_t> want{out_ch, in_ch, 3, 3};
    VT_CHECK(t.shape == want, "dots3-note audio tower: '" + name +
                                  "' ships " + ShapeOf(t.shape) +
                                  ", the config implies " + ShapeOf(want) +
                                  " (kernel_size=3, audio_encoder.py:466-474 @ "
                                  "9035151d6)");
    OwnedTensor o = LoadBf16Direct(get, name, {out_ch, in_ch * 9});
    RequireAudioShape(o, name, {out_ch, in_ch * 9});
    return o;
  };
  w.conv1_w = load_conv(se + "conv2d1.weight", dhs, 1);
  w.conv1_b = LoadBf16Direct(get, se + "conv2d1.bias");
  RequireAudioShape(w.conv1_b, se + "conv2d1.bias", {dhs});
  w.conv2_w = load_conv(se + "conv2d2.weight", dhs, dhs);
  w.conv2_b = LoadBf16Direct(get, se + "conv2d2.bias");
  RequireAudioShape(w.conv2_b, se + "conv2d2.bias", {dhs});
  w.conv3_w = load_conv(se + "conv2d3.weight", dhs, dhs);
  w.conv3_b = LoadBf16Direct(get, se + "conv2d3.bias");
  RequireAudioShape(w.conv3_b, se + "conv2d3.bias", {dhs});
  w.conv_out_w = LoadBf16Direct(get, se + "conv_out.weight");
  RequireAudioShape(w.conv_out_w, se + "conv_out.weight",
                    {D, a.conv_out_in_dim()});

  w.layers.reserve(static_cast<size_t>(a.num_layers));
  for (int64_t l = 0; l < a.num_layers; ++l) {
    const std::string pre = se + "layers." + std::to_string(l) + ".";
    Dots3NoteAudioLayerWeights lw;
    lw.attn_norm = LoadBf16Direct(get, pre + "self_attn_layer_norm.weight");
    RequireAudioShape(lw.attn_norm, pre + "self_attn_layer_norm.weight", {D});
    lw.final_norm = LoadBf16Direct(get, pre + "final_layer_norm.weight");
    RequireAudioShape(lw.final_norm, pre + "final_layer_norm.weight", {D});
    const auto proj = [&](const char* which, OwnedTensor* wt) {
      *wt = LoadBf16Direct(get, pre + "self_attn." + which + ".weight");
      RequireAudioShape(*wt, pre + "self_attn." + which + ".weight", {D, D});
    };
    const auto bias = [&](const char* which, OwnedTensor* bt) {
      *bt = LoadBf16Direct(get, pre + "self_attn." + which + ".bias");
      RequireAudioShape(*bt, pre + "self_attn." + which + ".bias", {D});
    };
    proj("q_proj", &lw.q_w);
    bias("q_proj", &lw.q_b);
    proj("k_proj", &lw.k_w);  // and NO k bias — audio_encoder.py:221
    proj("v_proj", &lw.v_w);
    bias("v_proj", &lw.v_b);
    proj("out_proj", &lw.out_w);
    bias("out_proj", &lw.out_b);
    lw.fc1_w = LoadBf16Direct(get, pre + "fc1.weight");
    RequireAudioShape(lw.fc1_w, pre + "fc1.weight", {a.fc1_out(), D});
    lw.fc1_b = LoadBf16Direct(get, pre + "fc1.bias");
    RequireAudioShape(lw.fc1_b, pre + "fc1.bias", {a.fc1_out()});
    lw.fc2_w = LoadBf16Direct(get, pre + "fc2.weight");
    RequireAudioShape(lw.fc2_w, pre + "fc2.weight", {D, F});
    lw.fc2_b = LoadBf16Direct(get, pre + "fc2.bias");
    RequireAudioShape(lw.fc2_b, pre + "fc2.bias", {D});
    w.layers.push_back(std::move(lw));
  }

  w.final_norm = LoadBf16Direct(get, se + "layer_norm.weight");
  RequireAudioShape(w.final_norm, se + "layer_norm.weight", {D});

  const std::string ad = p + "audio_adapter.proj.";
  const int64_t AI = a.adapter_in_dim, AO = a.adapter_out_dim;
  w.adapter_ln_w = LoadBf16Direct(get, ad + "0.weight");
  RequireAudioShape(w.adapter_ln_w, ad + "0.weight", {AI});
  w.adapter_ln_b = LoadBf16Direct(get, ad + "0.bias");
  RequireAudioShape(w.adapter_ln_b, ad + "0.bias", {AI});
  w.adapter_fc1_w = LoadBf16Direct(get, ad + "1.weight");
  RequireAudioShape(w.adapter_fc1_w, ad + "1.weight", {AO, AI});
  w.adapter_fc1_b = LoadBf16Direct(get, ad + "1.bias");
  RequireAudioShape(w.adapter_fc1_b, ad + "1.bias", {AO});
  w.adapter_fc2_w = LoadBf16Direct(get, ad + "3.weight");
  RequireAudioShape(w.adapter_fc2_w, ad + "3.weight", {AO, AO});
  w.adapter_fc2_b = LoadBf16Direct(get, ad + "3.bias");
  RequireAudioShape(w.adapter_fc2_b, ad + "3.bias", {AO});

  w.present = true;
  return w;
}

std::vector<float> Dots3NoteAudioRopeCache(int64_t num_positions,
                                           const Dots3NoteAudioParams& a) {
  // `RotaryEmbedding.__init__` (`audio_encoder.py:69-79`):
  //   inv_freq = 1 / theta ** (arange(0, rotary_dim, 2) / rotary_dim)
  // then `get_cos_sin` (`:126-131`) builds `emb = cat((freqs, freqs))` and
  // takes cos/sin of it. Because the second half of `emb` REPEATS the first,
  // and `rotate_half` (`:140-143`) splits at `rotary_dim / 2`, that is exactly
  // the `[cos(rd/2) | sin(rd/2)]` cache `vt::RopeFromCache` consumes at
  // `rotary_dim = rd` with `is_neox_style = true`.
  //
  // The DENOMINATOR is `rotary_dim`, NOT `head_dim`. On the released geometry
  // those are 32 and 64, so using the wrong one would change every frequency
  // and produce a well-shaped wrong answer.
  const int64_t rd = a.rotary_dim();
  const int64_t nf = rd / 2;
  std::vector<float> cache(static_cast<size_t>(num_positions * rd));
  for (int64_t t = 0; t < num_positions; ++t) {
    for (int64_t i = 0; i < nf; ++i) {
      const double inv =
          1.0 / std::pow(a.rope_theta, static_cast<double>(2 * i) /
                                           static_cast<double>(rd));
      const double ang = static_cast<double>(t) * inv;
      cache[static_cast<size_t>(t * rd + i)] = static_cast<float>(std::cos(ang));
      cache[static_cast<size_t>(t * rd + nf + i)] =
          static_cast<float>(std::sin(ang));
    }
  }
  return cache;
}

namespace {

// out[M,N] = x[M,K] @ W[N,K]^T (+ bias[N]). bf16 throughout, the projection
// shape the rest of this tree spells.
void LinearBias(Dev d, DBuf& out, const Tensor& x, const Tensor& w,
                const Tensor* bias) {
  vt::MatmulBT(d.q, out.t(), x, w);
  if (bias != nullptr) vt::Add(d.q, out.t(), out.t(), *bias);
}

std::vector<float> DownloadF32(Dev d, DBuf& buf, int64_t n) {
  std::vector<uint16_t> bits(static_cast<size_t>(n));
  buf.Download(d, bits.data());
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    out[static_cast<size_t>(i)] = vt::BF16ToF32(bits[static_cast<size_t>(i)]);
  return out;
}

// ONE stride-2, padding-1, 3x3 Conv2d layer, composed as im2col +
// `vt::MatmulBT` + bias + GELU-erf. See the header for why this is one exact
// tracked exception to the `vt::Conv2d` seam (#2709) rather than a call to it.
//
// The activation between stem layers lives on the HOST as
// `[freq * time, channels]` bf16 bits, which is what makes the FOUR TEMPORAL
// MASK STAGES a memset rather than a fifth kernel. That layout is not a
// convenience: `x.permute(0, 3, 1, 2).reshape(batch, time, channels * freq)`
// (`audio_encoder.py:579-580`) is the final regroup, and doing it from a
// channel-last spatial buffer is one index arithmetic instead of two.
//
// `out[fo, to, co] = sum_{ci, kf, kt} x[fi, ti, ci] * W[co, ci, kf, kt]`
// with `fi = 2*fo - 1 + kf`, `ti = 2*to - 1 + kt`, and zero outside the input.
std::vector<uint16_t> ConvStemLayer(Dev d, const std::vector<uint16_t>& x_bits,
                                    int64_t in_ch, int64_t Fin, int64_t Tin,
                                    const OwnedTensor& weight,
                                    const OwnedTensor& bias, int64_t out_ch,
                                    int64_t* Fout, int64_t* Tout) {
  // `(n + 2*padding - kernel) / stride + 1` at padding 1, kernel 3, stride 2.
  const int64_t Fo = (Fin + 2 - 3) / 2 + 1;
  const int64_t To = (Tin + 2 - 3) / 2 + 1;
  VT_CHECK(Fo > 0 && To > 0,
           "dots3-note audio tower: a stem layer over " + std::to_string(Fin) +
               "x" + std::to_string(Tin) + " produces no output");
  const int64_t K = in_ch * 9;
  const int64_t rows = Fo * To;
  std::vector<uint16_t> col(static_cast<size_t>(rows * K), 0);
  for (int64_t fo = 0; fo < Fo; ++fo) {
    for (int64_t to = 0; to < To; ++to) {
      uint16_t* dst = col.data() + static_cast<size_t>((fo * To + to) * K);
      for (int64_t ci = 0; ci < in_ch; ++ci) {
        for (int64_t kf = 0; kf < 3; ++kf) {
          const int64_t fi = 2 * fo - 1 + kf;
          if (fi < 0 || fi >= Fin) continue;
          for (int64_t kt = 0; kt < 3; ++kt) {
            const int64_t ti = 2 * to - 1 + kt;
            if (ti < 0 || ti >= Tin) continue;
            dst[ci * 9 + kf * 3 + kt] =
                x_bits[static_cast<size_t>((fi * Tin + ti) * in_ch + ci)];
          }
        }
      }
    }
  }
  DBuf cb(d, DType::kBF16, {rows, K}, col.data());
  DBuf ob(d, DType::kBF16, {rows, out_ch});
  Tensor cw = ResidentWeight(d, weight, {out_ch, K});
  Tensor cbias = ResidentWeight(d, bias, {out_ch});
  LinearBias(d, ob, cb.t(), cw, &cbias);
  // `nn.functional.gelu` with the default `approximate='none'`, i.e. the exact
  // erf form (`audio_encoder.py:547`, `:552`, `:557`).
  vt::GeluErf(d.q, ob.t(), ob.t());
  std::vector<uint16_t> out(static_cast<size_t>(rows * out_ch));
  ob.Download(d, out.data());
  *Fout = Fo;
  *Tout = To;
  return out;
}

// `_temporal_mask` (`audio_encoder.py:528-533`): zero every position at or past
// `valid` along the TIME axis, over a `[freq * time, channels]` host buffer.
//
// THIS IS NOT OPTIONAL AND A SHAPE CHECK CANNOT SEE IT. The mel of a
// zero-padded tail is NOT zero: the front end's `-8` global-max floor pushed
// through `(x + 4) / 4` is a nonzero constant, so without the mask that
// constant leaks through each 3x3 receptive field into the LAST VALID tokens.
void MaskTime(std::vector<uint16_t>* x, int64_t Fdim, int64_t Tdim,
              int64_t channels, int64_t valid) {
  if (valid >= Tdim) return;
  const int64_t lo = std::max<int64_t>(valid, 0);
  for (int64_t f = 0; f < Fdim; ++f) {
    for (int64_t t = lo; t < Tdim; ++t) {
      uint16_t* row = x->data() + static_cast<size_t>((f * Tdim + t) * channels);
      for (int64_t c = 0; c < channels; ++c) row[c] = 0;
    }
  }
}

}  // namespace

std::vector<float> Dots3NoteAudioForward(const std::vector<float>& mel,
                                         int64_t num_samples,
                                         int64_t num_tokens, int64_t hop_length,
                                         const Dots3NoteAudioWeights& w,
                                         const Dots3NoteAudioParams& a,
                                         vt::Backend& backend,
                                         Dots3NoteAudioCapture* cap) {
  VT_CHECK(w.present,
           "dots3-note audio tower: the weights were never materialized. The "
           "loader only materializes a tower Dots3NoteAudioRefusal accepted.");
  // `assert mel.shape[1] == self.chunk_mel_frames` (`nvidia/audio.py:215` @
  // `9035151d6`), made executable HERE by W7b (#2797) rather than only in the
  // processor. Upstream can only ever hand `_forward_speech_encoder` a STACK of
  // `chunk_mel_frames`-wide segments (`torch.stack`, `:220`), and this function
  // is one element of that stack. A caller that hands it the WHOLE stack — the
  // shape the code had before `Dots3NoteAudioForwardChunks` existed — gets a
  // correctly-shaped answer of the right row count off a mel of the wrong
  // width, and this width is the only number that can tell the two apart.
  VT_CHECK(a.num_mel_bins > 0 &&
               static_cast<int64_t>(mel.size()) ==
                   a.num_mel_bins * a.chunk_mel_frames(),
           "dots3-note audio tower: the mel holds " +
               std::to_string(mel.size()) + " values, which is not ONE chunk of "
               + std::to_string(a.num_mel_bins) + " x " +
               std::to_string(a.chunk_mel_frames()) +
               " (nvidia/audio.py:215 @ 9035151d6). Pass one chunk; "
               "`Dots3NoteAudioForwardChunks` is what takes the stack.");
  const int64_t D = a.d_model, F = a.ffn_dim;
  const int64_t nh = a.num_heads, hd = a.head_dim();
  const int64_t Fmel = a.num_mel_bins;
  VT_CHECK(hop_length > 0, "dots3-note audio tower: hop_length must be positive");
  VT_CHECK(Fmel > 0 && !mel.empty() &&
               static_cast<int64_t>(mel.size()) % Fmel == 0,
           "dots3-note audio tower: the mel buffer holds " +
               std::to_string(mel.size()) + " values, which is not a whole "
               "number of " + std::to_string(Fmel) + "-bin frames");
  const int64_t Tmel = static_cast<int64_t>(mel.size()) / Fmel;
  VT_CHECK(num_tokens > 0,
           "dots3-note audio tower: a placeholder span of " +
               std::to_string(num_tokens) + " tokens");

  vt::Queue q = backend.CreateQueue();
  Dev d{backend, q};
  const vt::RmsNormArgs rms{static_cast<float>(a.rms_norm_eps), /*gemma=*/false};

  // ── the stem (`_forward_conv2d_stem`, audio_encoder.py:564-583) ────────────
  //
  // `valid_mel_lens = audio_sample_lens // hop_length` (`:570-574`), then
  // `(n + 1) // 2` at each stride-2 layer (`:550`, `:555`, `:560`). The FOUR
  // stages are upstream's own count: one on the mel itself and one after each
  // GELU.
  std::vector<int64_t> valid(4);
  valid[0] = num_samples / hop_length;
  valid[1] = (valid[0] + 1) / 2;
  valid[2] = (valid[1] + 1) / 2;
  valid[3] = (valid[2] + 1) / 2;
  if (cap != nullptr) cap->valid_lens = valid;

  // `[B, 1, n_mels, T]` (`:568`) as a `[freq * time, channels=1]` host buffer.
  std::vector<uint16_t> x(static_cast<size_t>(Fmel * Tmel));
  for (int64_t f = 0; f < Fmel; ++f) {
    for (int64_t t = 0; t < Tmel; ++t) {
      x[static_cast<size_t>(f * Tmel + t)] =
          vt::F32ToBF16(mel[static_cast<size_t>(f * Tmel + t)]);
    }
  }
  // Step 0 (`:545-546`): mask the MEL. `silence_mel(-1.5) -> 0` is upstream's
  // own comment, and -1.5 is exactly the `-8` floor through `(x + 4) / 4`.
  MaskTime(&x, Fmel, Tmel, 1, valid[0]);
  if (cap != nullptr) {
    cap->masked_mel.resize(static_cast<size_t>(Fmel * Tmel));
    for (size_t i = 0; i < cap->masked_mel.size(); ++i)
      cap->masked_mel[i] = vt::BF16ToF32(x[i]);
  }

  const int64_t dhs = a.downsample_hidden_size;
  int64_t Fcur = Fmel, Tcur = Tmel;
  int64_t Fn = 0, Tn = 0;
  x = ConvStemLayer(d, x, /*in_ch=*/1, Fcur, Tcur, w.conv1_w, w.conv1_b, dhs,
                    &Fn, &Tn);
  Fcur = Fn; Tcur = Tn;
  MaskTime(&x, Fcur, Tcur, dhs, valid[1]);  // step 1, :549-551
  x = ConvStemLayer(d, x, dhs, Fcur, Tcur, w.conv2_w, w.conv2_b, dhs, &Fn, &Tn);
  Fcur = Fn; Tcur = Tn;
  MaskTime(&x, Fcur, Tcur, dhs, valid[2]);  // step 2, :554-556
  x = ConvStemLayer(d, x, dhs, Fcur, Tcur, w.conv3_w, w.conv3_b, dhs, &Fn, &Tn);
  Fcur = Fn; Tcur = Tn;
  MaskTime(&x, Fcur, Tcur, dhs, valid[3]);  // step 3, :559-561

  VT_CHECK(Fcur == a.freq_after(),
           "dots3-note audio tower: the stem left " + std::to_string(Fcur) +
               " mel bins, the config implies " +
               std::to_string(a.freq_after()) +
               " (audio_encoder.py:476-478 @ 9035151d6)");
  VT_CHECK(num_tokens <= Tcur,
           "dots3-note audio tower: the placeholder span is " +
               std::to_string(num_tokens) + " tokens and the stem produced " +
               std::to_string(Tcur) +
               ". The processor's `ceil(num_samples / token_stride)` and the "
               "stem's 8x downsampling of the padded chunk disagree, and the "
               "varlen pack would read past the stem output.");

  // `x.permute(0, 3, 1, 2).reshape(batch, time, channels * frequency)`
  // (`:579-580`): the regrouped row for time `t` is CHANNEL-MAJOR over
  // frequency, `c * Fcur + f`. Getting that order backwards is a permutation of
  // the same 7680 numbers and produces a well-shaped wrong answer.
  const int64_t CF = dhs * Fcur;
  std::vector<uint16_t> regrouped(static_cast<size_t>(Tcur * CF));
  for (int64_t t = 0; t < Tcur; ++t) {
    for (int64_t c = 0; c < dhs; ++c) {
      for (int64_t f = 0; f < Fcur; ++f) {
        regrouped[static_cast<size_t>(t * CF + c * Fcur + f)] =
            x[static_cast<size_t>((f * Tcur + t) * dhs + c)];
      }
    }
  }

  // `conv_out` (`:479`, `:581`) — a bias-free Linear over the regrouped row.
  // Only the first `num_tokens` rows survive the varlen pack below
  // (`audio_encoder.py:679-681` with B == 1), so the projection is applied to
  // exactly those rows rather than to all 750 and then sliced. That is a saving
  // and not a divergence: the GEMM is row-independent.
  DBuf hidden(d, DType::kBF16, {num_tokens, D});
  {
    DBuf rg(d, DType::kBF16, {num_tokens, CF}, regrouped.data());
    Tensor cw = ResidentWeight(d, w.conv_out_w, {D, CF});
    LinearBias(d, hidden, rg.t(), cw, nullptr);
  }
  if (cap != nullptr) cap->stem_out = DownloadF32(d, hidden, num_tokens * D);

  // ── RoPE (`audio_encoder.py:642-650`, `:679-685`) ──────────────────────────
  //
  // `position_ids = arange(inputs_embeds.shape[1])` over the FULL stem output,
  // and the varlen pack then gathers the first `input_seq_lens[0]` of them for
  // a single item — so the positions the kept rows carry are 0..num_tokens-1,
  // which is what this cache holds.
  const int64_t rd = a.rotary_dim();
  const std::vector<float> cache_f = Dots3NoteAudioRopeCache(num_tokens, a);
  if (cap != nullptr) cap->rope_cache = cache_f;
  std::vector<uint16_t> cache_bits(cache_f.size());
  for (size_t i = 0; i < cache_f.size(); ++i)
    cache_bits[i] = vt::F32ToBF16(cache_f[i]);
  DBuf cache(d, DType::kBF16, {num_tokens, rd}, cache_bits.data());
  std::vector<int32_t> pos_idx(static_cast<size_t>(num_tokens));
  for (int64_t i = 0; i < num_tokens; ++i)
    pos_idx[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  DBuf posb(d, DType::kI32, {num_tokens}, pos_idx.data());
  vt::RopeArgs ra;
  ra.base = static_cast<float>(a.rope_theta);
  // PARTIAL: 32 of a 64-wide head on the released geometry. A port that rotated
  // the whole head would still emit [T, 1280].
  ra.rotary_dim = static_cast<int>(rd);
  ra.is_neox_style = true;

  // `self.scaling = head_dim ** -0.5` (`audio_encoder.py:218`), passed to
  // `flash_attn_varlen_func` as `softmax_scale` (`:276`). `causal=is_decoder`
  // (`:277`) and `is_decoder = use_causal` (`:319`), which the refusal pins
  // false — written out rather than hard-coded so the two say the same thing.
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const vt::AttentionArgs aargs{scale, /*causal=*/a.use_causal};

  const int64_t T = num_tokens;
  DBuf n1(d, DType::kBF16, {T, D});
  DBuf qb(d, DType::kBF16, {T, D});
  DBuf kb(d, DType::kBF16, {T, D});
  DBuf vb(d, DType::kBF16, {T, D});
  DBuf ao(d, DType::kBF16, {T, nh, hd});
  DBuf attn(d, DType::kBF16, {T, D});
  DBuf n2(d, DType::kBF16, {T, D});
  DBuf mlp_out(d, DType::kBF16, {T, D});

  for (size_t l = 0; l < w.layers.size(); ++l) {
    const Dots3NoteAudioLayerWeights& lw = w.layers[l];
    // `WhisperEncoderLayer.forward` (`:360-391`): PRE-norm, and the residual is
    // the UN-normalized stream.
    vt::RmsNorm(d.q, n1.t(), hidden.t(), ResidentWeight(d, lw.attn_norm, {D}),
                rms);
    {
      Tensor qw = ResidentWeight(d, lw.q_w, {D, D});
      Tensor qbi = ResidentWeight(d, lw.q_b, {D});
      Tensor kw = ResidentWeight(d, lw.k_w, {D, D});
      Tensor vw = ResidentWeight(d, lw.v_w, {D, D});
      Tensor vbi = ResidentWeight(d, lw.v_b, {D});
      LinearBias(d, qb, n1.t(), qw, &qbi);
      // NO BIAS ON K, and this line is the whole convention
      // (`audio_encoder.py:221-224`). Passing `&qbi` here by copy-paste would
      // be arithmetically plausible and silently wrong.
      LinearBias(d, kb, n1.t(), kw, nullptr);
      LinearBias(d, vb, n1.t(), vw, &vbi);
    }
    Tensor q3 = qb.t();
    q3.rank = 3; q3.shape[0] = T; q3.shape[1] = nh; q3.shape[2] = hd;
    q3.stride[0] = nh * hd; q3.stride[1] = hd; q3.stride[2] = 1;
    Tensor k3 = kb.t();
    k3.rank = 3; k3.shape[0] = T; k3.shape[1] = nh; k3.shape[2] = hd;
    k3.stride[0] = nh * hd; k3.stride[1] = hd; k3.stride[2] = 1;
    Tensor v3 = vb.t();
    v3.rank = 3; v3.shape[0] = T; v3.shape[1] = nh; v3.shape[2] = hd;
    v3.stride[0] = nh * hd; v3.stride[1] = hd; v3.stride[2] = 1;
    // `apply_rotary_pos_emb` on the VARLEN path (`:264-267`), which rotates q
    // and k and leaves v alone.
    vt::RopeFromCache(d.q, q3, &k3, posb.t(), cache.t(), ra);
    // ONE window: this port serves exactly one audio item per request, so
    // `cu_seqlens` is `[0, T]` and the whole item is a single bidirectional
    // block.
    vt::AttentionDenseFlash(d.q, ao.t(), q3, k3, v3, aargs);
    {
      Tensor ao2 = ao.t();
      ao2.rank = 2; ao2.shape[0] = T; ao2.shape[1] = D;
      ao2.stride[0] = D; ao2.stride[1] = 1;
      Tensor ow = ResidentWeight(d, lw.out_w, {D, D});
      Tensor ob = ResidentWeight(d, lw.out_b, {D});
      LinearBias(d, attn, ao2, ow, &ob);
    }
    vt::Add(d.q, hidden.t(), hidden.t(), attn.t());

    // `residual + fc2(swiglu(fc1(final_layer_norm(x))))` (`:378-391`).
    vt::RmsNorm(d.q, n2.t(), hidden.t(), ResidentWeight(d, lw.final_norm, {D}),
                rms);
    {
      // THE SHARED SEAM, through the BIAS arm W7a added to it. `fc1` is already
      // the merged gate|up pair and `swiglu` chunks it gate-then-up
      // (`:42-44`), which is `vt::SiluAndMul`'s own order — so this is
      // `layers::MlpGateUpMethodBase` and not two hand-written GEMMs
      // (AGENTS.md, "Shared seams").
      DBuf act = layers::UnquantizedMlpGateUpBiasMethod(&lw.fc1_w, &lw.fc1_b, F)
                     .Apply(d, n2.t());
      Tensor dw = ResidentWeight(d, lw.fc2_w, {D, F});
      Tensor db = ResidentWeight(d, lw.fc2_b, {D});
      LinearBias(d, mlp_out, act.t(), dw, &db);
    }
    vt::Add(d.q, hidden.t(), hidden.t(), mlp_out.t());
    if (cap != nullptr && l == 0) cap->block0_out = DownloadF32(d, hidden, T * D);
  }

  // `hidden_states = self.layer_norm(hidden_states)` (`:721`).
  DBuf trunk(d, DType::kBF16, {T, D});
  vt::RmsNorm(d.q, trunk.t(), hidden.t(), ResidentWeight(d, w.final_norm, {D}),
              rms);
  if (cap != nullptr) cap->trunk_out = DownloadF32(d, trunk, T * D);

  // ── the adapter (`AudioAdapter`, audio.py:237-248) ─────────────────────────
  //
  // `Sequential(LayerNorm(in), Linear(in, out), GELU(), Linear(out, out))`.
  // `proj.0` is a LAYERNORM — mean-subtracting, with a bias — and NOT the
  // RMSNorm the encoder blocks use. `nn.GELU()` with no argument is the exact
  // erf form, not the tanh approximation.
  const int64_t AO = a.adapter_out_dim;
  DBuf aln(d, DType::kBF16, {T, D});
  {
    Tensor lw_ = ResidentWeight(d, w.adapter_ln_w, {D});
    Tensor lb_ = ResidentWeight(d, w.adapter_ln_b, {D});
    const vt::LayerNormArgs ln{static_cast<float>(a.layer_norm_eps)};
    Tensor out = aln.t();
    vt::LayerNorm(d.q, out, trunk.t(), &lw_, &lb_, ln);
  }
  DBuf a1(d, DType::kBF16, {T, AO});
  {
    Tensor fw = ResidentWeight(d, w.adapter_fc1_w, {AO, D});
    Tensor fb = ResidentWeight(d, w.adapter_fc1_b, {AO});
    LinearBias(d, a1, aln.t(), fw, &fb);
  }
  vt::GeluErf(d.q, a1.t(), a1.t());
  DBuf a2(d, DType::kBF16, {T, AO});
  {
    Tensor fw = ResidentWeight(d, w.adapter_fc2_w, {AO, AO});
    Tensor fb = ResidentWeight(d, w.adapter_fc2_b, {AO});
    LinearBias(d, a2, a1.t(), fw, &fb);
  }
  return DownloadF32(d, a2, T * AO);
}

// ── W7b (#2797): the CHUNK LOOP, `nvidia/audio.py:220-234` @ `9035151d6` ────
//
// See the header for why a loop over the single-chunk tower IS upstream's one
// batched varlen call: the pack gives each chunk its own `cu_seqlens` window
// and restarts its rope positions at 0, so no chunk can reach another.
std::vector<float> Dots3NoteAudioForwardChunks(
    const std::vector<float>& mels,
    const std::vector<int64_t>& chunk_num_samples,
    const std::vector<int64_t>& chunk_num_tokens, int64_t hop_length,
    const Dots3NoteAudioWeights& w, const Dots3NoteAudioParams& a,
    vt::Backend& backend, std::vector<Dots3NoteAudioCapture>* captures) {
  const int64_t k = static_cast<int64_t>(chunk_num_tokens.size());
  VT_CHECK(k > 0,
           "dots3-note audio tower: a waveform with no chunks. The processor's "
           "segment loop emits at least one for any non-empty waveform "
           "(nvidia/audio.py:196-203 @ 9035151d6).");
  VT_CHECK(static_cast<int64_t>(chunk_num_samples.size()) == k,
           "dots3-note audio tower: " +
               std::to_string(chunk_num_samples.size()) +
               " chunk sample lengths against " + std::to_string(k) +
               " chunk token counts. They are upstream's `audio_sample_lens` "
               "and `token_lens` and are built in the same loop "
               "(nvidia/audio.py:217-218 @ 9035151d6).");
  const int64_t Fmel = a.num_mel_bins;
  VT_CHECK(Fmel > 0 && !mels.empty() &&
               static_cast<int64_t>(mels.size()) % (Fmel * k) == 0,
           "dots3-note audio tower: the stacked mel holds " +
               std::to_string(mels.size()) + " values, which is not " +
               std::to_string(k) + " chunks of " + std::to_string(Fmel) +
               " bins");
  // One chunk's mel is the whole buffer divided by the chunk count: upstream
  // stacks EQUAL-length padded mels (`pad_or_trim` to `chunk_samples`,
  // `:213`), which is what makes `torch.stack` legal there and this division
  // legal here.
  const int64_t per = static_cast<int64_t>(mels.size()) / k;

  if (captures != nullptr) captures->assign(static_cast<size_t>(k), {});

  std::vector<float> out;
  for (int64_t i = 0; i < k; ++i) {
    // `mel_features[idx]` — one padded chunk, at its OWN valid sample length.
    // Handing `chunk_num_samples[i]` rather than the padded length is what
    // makes the four-stage temporal mask fire on the SHORT final chunk;
    // `valid_mel_lens = audio_sample_lens // hop_length` is per batch element
    // upstream too (`audio_encoder.py:570-577`).
    const std::vector<float> one(
        mels.begin() + static_cast<std::ptrdiff_t>(i * per),
        mels.begin() + static_cast<std::ptrdiff_t>((i + 1) * per));
    const std::vector<float> rows = Dots3NoteAudioForward(
        one, chunk_num_samples[static_cast<size_t>(i)],
        chunk_num_tokens[static_cast<size_t>(i)], hop_length, w, a, backend,
        captures != nullptr ? &(*captures)[static_cast<size_t>(i)] : nullptr);
    // `audio_embedding[idx, : token_len * merge_factor, :]` (`:229-233`): the
    // single-chunk tower already returns exactly those rows, because the varlen
    // pack it mirrors keeps the first `num_tokens` of the stem output and
    // everything after that is row-wise. `merge_factor != 1` is refused by
    // name, so `token_len * merge_factor` is `token_len`.
    VT_CHECK(static_cast<int64_t>(rows.size()) ==
                 chunk_num_tokens[static_cast<size_t>(i)] * a.adapter_out_dim,
             "dots3-note audio tower: chunk " + std::to_string(i) +
                 " produced " + std::to_string(rows.size()) +
                 " floats for " +
                 std::to_string(chunk_num_tokens[static_cast<size_t>(i)]) +
                 " rows of " + std::to_string(a.adapter_out_dim));
    // `torch.cat(chunk_embeddings, dim=0)` (`:234`) — IN ORDER. Concatenating
    // the same chunks in another order produces a correctly-shaped answer, so
    // the order is asserted in the gate rather than only written here.
    out.insert(out.end(), rows.begin(), rows.end());
  }
  return out;
}

}  // namespace vllm
