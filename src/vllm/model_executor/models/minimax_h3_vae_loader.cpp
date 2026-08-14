// MiniMax-H3 VAE checkpoint loaders — materializing the shipped safetensors files
// into the weight structs the (already gated) VAE forwards consume.
//
// The forwards were gated against the checkpoint's OWN remote code at reduced
// dimensions, with weights rebuilt from the shared PRNG. That proved the MATH.
// What it could not prove is that the SHIPPED file's tensors bind onto those
// structs — and for the audio VAE they do not, not directly. Both mismatches were
// found by reading the real 1087-tensor header (an HTTP range request over the
// file's first 2 MiB, no payload downloaded) and are gated against it:
//
//   1. WEIGHT-NORM SPELLING. The checkpoint ships torch's LEGACY weight_norm pair
//      `weight_g` / `weight_v`. The decoder reads the PARAMETRIZATION spelling
//      `parametrizations.weight.original0` / `original1`, because the generator
//      that produced its goldens ran the checkpoint's remote code under a modern
//      torch, where weight_norm is a parametrization. Same tensors, different era.
//   2. PREFIX. Every BigVGAN tensor lives under `decoder.`, but `dec_in_proj.*` —
//      the Conv1d that runs BEFORE BigVGAN — sits at the top level.
//
// Either mismatch alone yields a loader that throws by name (best case) or, if a
// future refactor made lookups lenient, a decoder reading zeros. The mapping is
// therefore asserted against the real manifest in the test, not just exercised.
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/model_executor/models/vocoder1d.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

float Bf16ToF32(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float out;
  std::memcpy(&out, &widened, sizeof(out));
  return out;
}

float F16ToF32(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exp = (bits >> 10) & 0x1Fu;
  const uint32_t mant = bits & 0x3FFu;
  uint32_t out_bits;
  if (exp == 0) {
    if (mant == 0) {
      out_bits = sign;  // +/- zero
    } else {
      // Subnormal: renormalize into the f32 exponent range.
      uint32_t e = 0;
      uint32_t m = mant;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++e;
      }
      m &= 0x3FFu;
      out_bits = sign | ((127 - 15 - e) << 23) | (m << 13);
    }
  } else if (exp == 0x1Fu) {
    out_bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    out_bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &out_bits, sizeof(out));
  return out;
}

}  // namespace

std::vector<float> MiniMaxH3ReadSafetensorF32(const StTensor& tensor) {
  int64_t numel = 1;
  for (int64_t d : tensor.shape) numel *= d;
  std::vector<float> out(static_cast<size_t>(numel));
  if (tensor.dtype == "F32") {
    VT_CHECK(tensor.nbytes == static_cast<size_t>(numel) * 4,
             "minimax_h3: F32 tensor span does not match its shape");
    std::memcpy(out.data(), tensor.data, tensor.nbytes);
  } else if (tensor.dtype == "BF16" || tensor.dtype == "F16") {
    VT_CHECK(tensor.nbytes == static_cast<size_t>(numel) * 2,
             "minimax_h3: 16-bit tensor span does not match its shape");
    // Byte-wise load, NOT `reinterpret_cast<const uint16_t*>(tensor.data)[i]`.
    // safetensors puts the payload immediately after a JSON header of ARBITRARY
    // length, so a tensor's first byte is only 2-byte aligned if the writer
    // happened to pad; the format does not require it. The cast was UB on such a
    // file and UBSan caught it ("load of misaligned address ... requires 2 byte
    // alignment") the first time a checkpoint with an odd header reached this
    // path. memcpy has no alignment precondition and compiles to the same load
    // where the address does happen to be aligned.
    const auto* bytes = static_cast<const unsigned char*>(tensor.data);
    const bool bf16 = (tensor.dtype == "BF16");
    for (int64_t i = 0; i < numel; ++i) {
      uint16_t bits;
      std::memcpy(&bits, bytes + static_cast<size_t>(i) * 2, sizeof(bits));
      out[static_cast<size_t>(i)] = bf16 ? Bf16ToF32(bits) : F16ToF32(bits);
    }
  } else {
    VT_CHECK(false, "minimax_h3: unsupported tensor dtype (expected F32/BF16/F16)");
  }
  return out;
}

MiniMaxH3AudioVaeWeights LoadMiniMaxH3AudioVaeWeights(const SafetensorsFile& file) {
  MiniMaxH3AudioVaeWeights out;
  for (const std::string& name : file.Names()) {
    // The audio ENCODER shares this file, as do the VAE's mean/logvar heads and
    // pre_block in repackaged copies. Generation only decodes, so none are loaded.
    if (name.rfind("encoder.", 0) == 0) continue;
    if (name.rfind("mean_proj.", 0) == 0 || name.rfind("logs_proj.", 0) == 0) continue;
    if (name.rfind("pre_block.", 0) == 0) continue;

    std::string key = name;
    if (key.rfind("decoder.", 0) == 0) key = key.substr(std::strlen("decoder."));

    // The kaiser-sinc anti-aliasing filters are COMPUTED at load
    // (vocoder1d::KaiserSincFilter1d), never read.
    if (key.size() >= 7 && key.compare(key.size() - 7, 7, ".filter") == 0) continue;

    const StTensor& tensor = file.Get(name);

    // --- the three weight-norm spellings this decoder must accept ---
    // (1) LEGACY  `weight_g` / `weight_v`  — the OFFICIAL MiniMax-H3 checkpoint.
    // (2) MODERN  `parametrizations.weight.original0/1` — what the decoder reads,
    //     and what the generator produced (it ran the remote code under a torch
    //     where weight_norm is a parametrization).
    // (3) MATERIALIZED plain `weight` — repackaged community bundles, which folded
    //     the norm at conversion time.
    const std::string g = ".weight_g", v = ".weight_v";
    if (key.size() > g.size() && key.compare(key.size() - g.size(), g.size(), g) == 0) {
      key = key.substr(0, key.size() - g.size()) + ".parametrizations.weight.original0";
    } else if (key.size() > v.size() && key.compare(key.size() - v.size(), v.size(), v) == 0) {
      key = key.substr(0, key.size() - v.size()) + ".parametrizations.weight.original1";
    } else if (key.size() > 7 && key.compare(key.size() - 7, 7, ".weight") == 0 &&
               key != "dec_in_proj.weight" && tensor.shape.size() == 3) {
      // (3) Reconstruct an EXACT pair rather than an approximate one: with v = w and
      // g = per-dim-0-slice ||w||, the decoder's own g * v / ||v|| returns w.
      // dec_in_proj is a PLAIN Conv1d (not weight-normalized), so it is excluded.
      const std::string base = key.substr(0, key.size() - 7);
      std::vector<float> w = MiniMaxH3ReadSafetensorF32(tensor);
      const int64_t rows = tensor.shape[0];
      VT_CHECK(rows > 0 && static_cast<int64_t>(w.size()) % rows == 0,
               "minimax_h3 audio vae: materialized conv weight has an implausible shape");
      const int64_t per_row = static_cast<int64_t>(w.size()) / rows;
      std::vector<float> mag(static_cast<size_t>(rows));
      for (int64_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int64_t i = 0; i < per_row; ++i) {
          const double e = w[static_cast<size_t>(r * per_row + i)];
          sum += e * e;
        }
        mag[static_cast<size_t>(r)] = static_cast<float>(std::sqrt(sum));
      }
      VT_CHECK(out.tensors.count(base + ".parametrizations.weight.original1") == 0,
               "minimax_h3 audio vae: two checkpoint tensors map to the same name");
      out.tensors[base + ".parametrizations.weight.original1"] = std::move(w);
      out.tensors[base + ".parametrizations.weight.original0"] = std::move(mag);
      continue;
    }

    VT_CHECK(out.tensors.count(key) == 0,
             "minimax_h3 audio vae: two checkpoint tensors map to the same name");
    out.tensors[key] = MiniMaxH3ReadSafetensorF32(tensor);
  }
  VT_CHECK(!out.tensors.empty(), "minimax_h3 audio vae: checkpoint contained no decoder tensors");
  return out;
}

// The ENCODER half of the SAME audio-VAE file — the half the decoder loader above
// skips outright. ref2va needs it: an audio reference is a waveform, and the rows
// the DiT is conditioned on come out of this stack.
//
// It carries the decoder loader's rules over unchanged, because the file is the
// same file:
//   * PREFIX. The DAC encoder is under `encoder.`, stripped so the forward reads
//     bare `block.N...`. `pre_block.*` and `mean_proj.*` sit at the TOP level and
//     are kept verbatim — the same split `dec_in_proj` has on the decoder side.
//   * WEIGHT-NORM SPELLING. All three: legacy `weight_g`/`weight_v` (what the
//     official checkpoint ships), the modern `parametrizations.weight.originalN`
//     (what the forward reads, and what a modern-torch generator produces), and a
//     plain MATERIALIZED `weight` (repackaged bundles).
//
// `logs_proj.*` is deliberately NOT loaded. Conditioning takes the distribution
// MEAN; carrying the log-variance head would imply something is sampled, and a
// sampled reference would condition differently on every run. `decoder.*` and
// `dec_in_proj.*` are the synthesis half and are skipped.
MiniMaxH3AudioVaeWeights LoadMiniMaxH3AudioVaeEncoderWeights(const SafetensorsFile& file) {
  MiniMaxH3AudioVaeWeights out;
  for (const std::string& name : file.Names()) {
    const bool is_encoder = name.rfind("encoder.", 0) == 0;
    const bool is_pre_block = name.rfind("pre_block.", 0) == 0;
    const bool is_mean = name.rfind("mean_proj.", 0) == 0;
    if (!is_encoder && !is_pre_block && !is_mean) continue;

    std::string key = name;
    if (is_encoder) key = key.substr(std::strlen("encoder."));

    const StTensor& tensor = file.Get(name);

    // `mean_proj` and every `pre_block` Linear are PLAIN modules, so their bare
    // `.weight` must NOT be reinterpreted as a materialized weight-norm pair. Only
    // the encoder's own convs are weight-normalized.
    const std::string g = ".weight_g", v = ".weight_v";
    if (key.size() > g.size() && key.compare(key.size() - g.size(), g.size(), g) == 0) {
      key = key.substr(0, key.size() - g.size()) + ".parametrizations.weight.original0";
    } else if (key.size() > v.size() && key.compare(key.size() - v.size(), v.size(), v) == 0) {
      key = key.substr(0, key.size() - v.size()) + ".parametrizations.weight.original1";
    } else if (is_encoder && key.size() > 7 && key.compare(key.size() - 7, 7, ".weight") == 0 &&
               tensor.shape.size() == 3) {
      // Reconstruct an EXACT (g, v) pair: with v = w and g = per-output-channel
      // ||w||, the forward's own g * v / ||v|| returns w.
      const std::string base = key.substr(0, key.size() - 7);
      std::vector<float> w = MiniMaxH3ReadSafetensorF32(tensor);
      const int64_t rows = tensor.shape[0];
      VT_CHECK(rows > 0 && static_cast<int64_t>(w.size()) % rows == 0,
               "minimax_h3 audio encoder: materialized conv weight has an implausible shape");
      const int64_t per_row = static_cast<int64_t>(w.size()) / rows;
      std::vector<float> mag(static_cast<size_t>(rows));
      for (int64_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int64_t i = 0; i < per_row; ++i) {
          const double e = w[static_cast<size_t>(r * per_row + i)];
          sum += e * e;
        }
        mag[static_cast<size_t>(r)] = static_cast<float>(std::sqrt(sum));
      }
      VT_CHECK(out.tensors.count(base + ".parametrizations.weight.original1") == 0,
               "minimax_h3 audio encoder: two checkpoint tensors map to the same name");
      out.tensors[base + ".parametrizations.weight.original1"] = std::move(w);
      out.tensors[base + ".parametrizations.weight.original0"] = std::move(mag);
      continue;
    }

    VT_CHECK(out.tensors.count(key) == 0,
             "minimax_h3 audio encoder: two checkpoint tensors map to the same name");
    out.tensors[key] = MiniMaxH3ReadSafetensorF32(tensor);
  }
  VT_CHECK(out.tensors.count("block.0.parametrizations.weight.original1") != 0,
           "minimax_h3 audio encoder: checkpoint contained no encoder tensors");
  VT_CHECK(out.tensors.count("mean_proj.weight") != 0,
           "minimax_h3 audio encoder: checkpoint is missing mean_proj (the latent MEAN head)");
  return out;
}

MiniMaxH3AudioVaeWeights LoadMiniMaxH3VideoVaeDecoderWeights(const SafetensorsFile& file) {
  MiniMaxH3AudioVaeWeights out;
  for (const std::string& name : file.Names()) {
    // The 3D-CNN ENCODER shares this file (conditioning only), and `quant_conv`
    // is its output stage. Generation decodes, so neither is loaded.
    if (name.rfind("encoder.", 0) == 0) continue;
    if (name.rfind("quant_conv.", 0) == 0) continue;

    std::string key = name;
    if (key.rfind("decoder.", 0) == 0) key = key.substr(std::strlen("decoder."));
    VT_CHECK(out.tensors.count(key) == 0,
             "minimax_h3 video vae: two checkpoint tensors map to the same name");
    out.tensors[key] = MiniMaxH3ReadSafetensorF32(file.Get(name));
  }
  VT_CHECK(!out.tensors.empty(), "minimax_h3 video vae: checkpoint contained no decoder tensors");
  return out;
}

// The ENCODER half of the same file — the piece the decoder loader deliberately
// skips. Needed for CONDITIONING (fl2va keyframes, ref2va references): turning a
// supplied image or clip into the latent the DiT is conditioned on.
//
// Mirrors the decoder loader's one mapping rule in reverse: strip `encoder.`,
// because MiniMaxH3EncoderFcn3dForward reads bare names (`conv_in.weight`,
// `down.N.block.M...`). `quant_conv` IS kept — it is the encoder's output stage,
// projecting conv_out's 48 channels to the 48 that split into mean|logvar.
MiniMaxH3AudioVaeWeights LoadMiniMaxH3VideoVaeEncoderWeights(const SafetensorsFile& file) {
  MiniMaxH3AudioVaeWeights out;
  for (const std::string& name : file.Names()) {
    const bool is_encoder = name.rfind("encoder.", 0) == 0;
    const bool is_quant = name.rfind("quant_conv.", 0) == 0;
    if (!is_encoder && !is_quant) continue;  // the ViT decoder half
    std::string key = name;
    if (is_encoder) key = key.substr(std::strlen("encoder."));
    VT_CHECK(out.tensors.count(key) == 0,
             "minimax_h3 video vae encoder: two checkpoint tensors map to the same name");
    out.tensors[key] = MiniMaxH3ReadSafetensorF32(file.Get(name));
  }
  VT_CHECK(!out.tensors.empty(),
           "minimax_h3 video vae: checkpoint contained no encoder tensors");
  return out;
}

// Encode frames [in_channels, T, H, W] in [-1, 1] to the conditioning latent.
//
// The 3D CNN emits 2*z_channels (mean | logvar); `quant_conv` mixes those 48
// channels, and conditioning takes the DISTRIBUTION MEAN rather than a sample --
// a sample would make the same reference image produce a different conditioning
// on every run, which is not what a reference is for.
std::vector<float> MiniMaxH3VideoVaeEncodeToLatent(const MiniMaxH3EncoderFcn3dConfig& config,
                                                   const MiniMaxH3AudioVaeWeights& weights,
                                                   const std::vector<float>& frames,
                                                   MiniMaxH3VideoFrameShape* out_shape) {
  MiniMaxH3VideoFrameShape shape{};
  std::vector<float> moments = MiniMaxH3EncoderFcn3dForward(config, weights, frames, &shape);
  // `z_channels` is the MOMENTS width, not the latent width: the encoder's final
  // conv emits it directly, and the real checkpoint has conv_out -> 48 with a
  // 24-channel latent. So the latent is half of it (mean | logvar).
  const int64_t two_z = config.z_channels;
  VT_CHECK(two_z % 2 == 0,
           "minimax_h3 vae encode: z_channels must be even (it carries mean|logvar)");
  const int64_t latent_ch = two_z / 2;
  const int64_t per_channel = shape.t * shape.h * shape.w;
  VT_CHECK(static_cast<int64_t>(moments.size()) == two_z * per_channel,
           "minimax_h3 vae encode: encoder output is not [2 * z_channels, T, H, W]");

  // quant_conv: a 1x1x1 Conv3d over the 2*z channels, same shape as the decoder's
  // post_quant_conv but on the encoder side.
  if (weights.Has("quant_conv.weight")) {
    const std::vector<float>& w = weights.Get("quant_conv.weight");
    const std::vector<float>& b = weights.Get("quant_conv.bias");
    VT_CHECK(static_cast<int64_t>(w.size()) == two_z * two_z &&
                 static_cast<int64_t>(b.size()) == two_z,
             "minimax_h3 vae encode: quant_conv must be [2Z, 2Z, 1, 1, 1] with a 2Z bias");
    std::vector<float> mixed(moments.size());
    for (int64_t o = 0; o < two_z; ++o) {
      for (int64_t p = 0; p < per_channel; ++p) {
        double acc = b[static_cast<size_t>(o)];
        for (int64_t i = 0; i < two_z; ++i) {
          acc += static_cast<double>(w[static_cast<size_t>(o * two_z + i)]) *
                 moments[static_cast<size_t>(i * per_channel + p)];
        }
        mixed[static_cast<size_t>(o * per_channel + p)] = static_cast<float>(acc);
      }
    }
    moments = std::move(mixed);
  }

  // DiagonalGaussianDistribution.mean == the first half of the moments.
  std::vector<float> latent(static_cast<size_t>(latent_ch * per_channel));
  std::copy(moments.begin(), moments.begin() + latent.size(), latent.begin());
  if (out_shape != nullptr) *out_shape = shape;
  return latent;
}

std::vector<float> MiniMaxH3VideoVaePostQuantConv(const MiniMaxH3AudioVaeWeights& weights,
                                                  const std::vector<float>& latent,
                                                  int64_t channels, int64_t elems_per_channel) {
  const std::vector<float>& w = weights.Get("post_quant_conv.weight");
  const std::vector<float>& b = weights.Get("post_quant_conv.bias");
  VT_CHECK(static_cast<int64_t>(w.size()) == channels * channels,
           "minimax_h3 post_quant_conv: weight must be [C, C, 1, 1, 1]");
  VT_CHECK(static_cast<int64_t>(b.size()) == channels,
           "minimax_h3 post_quant_conv: bias must have one value per channel");
  VT_CHECK(static_cast<int64_t>(latent.size()) == channels * elems_per_channel,
           "minimax_h3 post_quant_conv: latent size does not match [C, ...]");

  // A 1x1x1 Conv3d over a CHANNEL-MAJOR latent: out[o, p] = sum_i w[o, i] * in[i, p]
  // + b[o]. The accumulation is f32 in input-channel order, matching torch's
  // contraction over a length-C reduction.
  std::vector<float> out(latent.size());
  for (int64_t o = 0; o < channels; ++o) {
    const float bias = b[static_cast<size_t>(o)];
    float* dst = out.data() + o * elems_per_channel;
    for (int64_t p = 0; p < elems_per_channel; ++p) dst[p] = bias;
    for (int64_t i = 0; i < channels; ++i) {
      const float coeff = w[static_cast<size_t>(o * channels + i)];
      const float* src = latent.data() + i * elems_per_channel;
      for (int64_t p = 0; p < elems_per_channel; ++p) dst[p] += coeff * src[p];
    }
  }
  return out;
}


// --- H3-Encoder (FL2VA/text_encoder) ---------------------------------------
// The one loader that TRANSFORMS rather than renames: HF ships q/k/v and gate/up
// SEPARATE, the port (like vLLM) consumes them FUSED. Row-concatenation order is
// load-bearing — the forward slices qkv_proj at [0, q_width), [q_width,
// q_width+kv_width), [q_width+kv_width, ...), so any other order silently feeds
// keys into the query path.
MiniMaxH3AudioVaeWeights LoadMiniMaxH3EncoderWeights(const std::vector<SafetensorsFile>& shards,
                                                     int64_t max_layers) {
  // One index over every shard, so a tensor is found wherever it lives.
  std::map<std::string, std::pair<size_t, const StTensor*>> index;
  for (size_t s = 0; s < shards.size(); ++s) {
    for (const std::string& name : shards[s].Names()) {
      VT_CHECK(index.count(name) == 0, "minimax_h3 encoder: tensor appears in two shards");
      index.emplace(name, std::make_pair(s, &shards[s].Get(name)));
    }
  }
  VT_CHECK(!index.empty(), "minimax_h3 encoder: no shards contained any tensor");

  MiniMaxH3AudioVaeWeights out;
  auto read = [&](const std::string& name) -> std::vector<float> {
    const auto it = index.find(name);
    VT_CHECK(it != index.end(), "minimax_h3 encoder: checkpoint is missing a required tensor");
    return MiniMaxH3ReadSafetensorF32(*it->second.second);
  };
  auto has = [&](const std::string& name) { return index.count(name) != 0; };

  const std::string lm = "model.language_model.";
  const std::string vis = "model.visual.";

  // --- text tower: strip the prefix, FUSE q/k/v and gate/up ---
  for (int64_t layer = 0;; ++layer) {
    const std::string src = lm + "layers." + std::to_string(layer) + ".";
    if (!has(src + "input_layernorm.weight")) break;
    if (max_layers > 0 && layer >= max_layers) break;
    const std::string dst = "layers." + std::to_string(layer) + ".";

    out.tensors[dst + "input_layernorm.weight"] = read(src + "input_layernorm.weight");
    out.tensors[dst + "post_attention_layernorm.weight"] =
        read(src + "post_attention_layernorm.weight");
    out.tensors[dst + "self_attn.q_norm.weight"] = read(src + "self_attn.q_norm.weight");
    out.tensors[dst + "self_attn.k_norm.weight"] = read(src + "self_attn.k_norm.weight");
    out.tensors[dst + "self_attn.o_proj.weight"] = read(src + "self_attn.o_proj.weight");
    out.tensors[dst + "mlp.down_proj.weight"] = read(src + "mlp.down_proj.weight");

    // [q_all | k_all | v_all], the order the forward slices.
    std::vector<float> q = read(src + "self_attn.q_proj.weight");
    const std::vector<float> k = read(src + "self_attn.k_proj.weight");
    const std::vector<float> v = read(src + "self_attn.v_proj.weight");
    q.reserve(q.size() + k.size() + v.size());
    q.insert(q.end(), k.begin(), k.end());
    q.insert(q.end(), v.begin(), v.end());
    out.tensors[dst + "self_attn.qkv_proj.weight"] = std::move(q);

    // [gate | up], matching MergedColumnParallelLinear.
    std::vector<float> gate = read(src + "mlp.gate_proj.weight");
    const std::vector<float> up = read(src + "mlp.up_proj.weight");
    gate.reserve(gate.size() + up.size());
    gate.insert(gate.end(), up.begin(), up.end());
    out.tensors[dst + "mlp.gate_up_proj.weight"] = std::move(gate);
  }
  VT_CHECK(out.tensors.count("layers.0.self_attn.qkv_proj.weight") != 0,
           "minimax_h3 encoder: no text-tower layers were loaded");

  if (has(lm + "embed_tokens.weight")) {
    // Kept: the text forward takes inputs_embeds, so a caller needs this to embed.
    out.tensors["embed_tokens.weight"] = read(lm + "embed_tokens.weight");
  }
  // `model.language_model.norm.weight` is deliberately NOT loaded — H3 reads the
  // UNNORMALIZED truncated output, and carrying the tensor would imply otherwise.

  // --- vision tower: prefix strip only; HF already ships qkv fused ---
  for (const auto& kv : index) {
    const std::string& name = kv.first;
    if (name.rfind(vis, 0) != 0) continue;
    const std::string dst = name.substr(vis.size());
    VT_CHECK(out.tensors.count(dst) == 0, "minimax_h3 encoder: vision name collides");
    out.tensors[dst] = MiniMaxH3ReadSafetensorF32(*kv.second.second);
  }
  return out;
}


// --- config.json parsing ----------------------------------------------------
namespace {

void ReadStats(const nlohmann::json& config, MiniMaxH3LatentStats* stats) {
  if (stats == nullptr) return;
  auto read = [&](const char* key, std::vector<float>& into) {
    if (!config.contains(key) || !config.at(key).is_array()) return;
    for (const auto& v : config.at(key)) into.push_back(v.get<float>());
  };
  read("latents_mean", stats->mean);
  read("latents_std", stats->std_dev);
  VT_CHECK(stats->mean.size() == stats->std_dev.size(),
           "minimax_h3 config: latents_mean and latents_std must have equal length");
}

std::vector<int64_t> ReadIntArray(const nlohmann::json& config, const char* key,
                                  const std::vector<int64_t>& fallback) {
  if (!config.contains(key) || !config.at(key).is_array()) return fallback;
  std::vector<int64_t> out;
  for (const auto& v : config.at(key)) out.push_back(v.get<int64_t>());
  return out;
}

}  // namespace

MiniMaxH3AudioVaeConfig ParseMiniMaxH3AudioVaeConfig(const nlohmann::json& config,
                                                     MiniMaxH3LatentStats* stats) {
  MiniMaxH3AudioVaeConfig out;
  // `latent_dim` is the mel count BigVGAN consumes (2048), NOT `latent_channels`
  // (32) — that one is the VAE's own latent width, which dec_in_proj maps FROM.
  if (config.contains("latent_dim")) out.num_mels = config.at("latent_dim").get<int64_t>();
  if (config.contains("decoder_dim")) {
    out.upsample_initial_channel = config.at("decoder_dim").get<int64_t>();
  }
  out.upsample_rates = ReadIntArray(config, "decoder_rates", out.upsample_rates);
  out.upsample_kernel_sizes =
      ReadIntArray(config, "decoder_kernel_sizes", out.upsample_kernel_sizes);
  out.resblock_kernel_sizes =
      ReadIntArray(config, "resblock_kernel_sizes", out.resblock_kernel_sizes);
  if (config.contains("resblock_dilation_sizes") &&
      config.at("resblock_dilation_sizes").is_array()) {
    out.resblock_dilation_sizes.clear();
    for (const auto& row : config.at("resblock_dilation_sizes")) {
      std::vector<int64_t> dil;
      for (const auto& v : row) dil.push_back(v.get<int64_t>());
      out.resblock_dilation_sizes.push_back(std::move(dil));
    }
  }
  VT_CHECK(out.upsample_rates.size() == out.upsample_kernel_sizes.size(),
           "minimax_h3 audio vae config: decoder_rates and decoder_kernel_sizes differ in length");
  VT_CHECK(out.resblock_kernel_sizes.size() == out.resblock_dilation_sizes.size(),
           "minimax_h3 audio vae config: resblock kernels and dilations differ in length");
  ReadStats(config, stats);
  return out;
}

MiniMaxH3VideoVaeDecoderConfig ParseMiniMaxH3VideoVaeDecoderConfig(const nlohmann::json& config,
                                                                   MiniMaxH3LatentStats* stats) {
  MiniMaxH3VideoVaeDecoderConfig out;
  if (config.contains("decoder_num_layers")) {
    out.num_layers = config.at("decoder_num_layers").get<int64_t>();
  }
  if (config.contains("latent_channels")) {
    out.in_channels = config.at("latent_channels").get<int64_t>();
  }
  if (config.contains("out_channels")) out.out_channels = config.at("out_channels").get<int64_t>();
  if (config.contains("decoder_num_register_tokens")) {
    out.num_register_tokens = config.at("decoder_num_register_tokens").get<int64_t>();
  }
  if (config.contains("decoder_rope_theta")) {
    out.rope_theta = config.at("decoder_rope_theta").get<double>();
  }
  const int64_t heads = config.contains("decoder_num_attention_heads")
                            ? config.at("decoder_num_attention_heads").get<int64_t>()
                            : out.block.heads;
  const int64_t dim_head = config.contains("decoder_attention_head_dim")
                               ? config.at("decoder_attention_head_dim").get<int64_t>()
                               : out.block.dim_head;
  out.block.heads = heads;
  out.block.dim_head = dim_head;
  out.block.dim = heads * dim_head;
  const int64_t ffn_mult = config.contains("decoder_ffn_mult")
                               ? config.at("decoder_ffn_mult").get<int64_t>()
                               : 4;
  out.block.ff_inner = out.block.dim * ffn_mult;
  if (config.contains("clip_length")) out.clip_length = config.at("clip_length").get<int64_t>();
  if (config.contains("token_drop")) out.token_drop = config.at("token_drop").get<int64_t>();
  if (config.contains("temporal_downsample_factors")) {
    // vae_ratio_t = prod(time_down) (klvae.py:1148).
    int64_t prod = 1;
    for (const auto& v : config.at("temporal_downsample_factors")) prod *= v.get<int64_t>();
    if (prod > 0) out.vae_ratio_t = prod;
  }
  if (config.contains("decoder_norm_eps")) {
    out.block.eps = config.at("decoder_norm_eps").get<double>();
  }
  // rope_apply_dim = int(dim_head * rope_dim_ratio), the checkpoint's own formula.
  const double ratio = config.contains("decoder_rope_dim_ratio")
                           ? config.at("decoder_rope_dim_ratio").get<double>()
                           : 0.75;
  out.rope_apply_dim = static_cast<int64_t>(static_cast<double>(dim_head) * ratio);
  ReadStats(config, stats);
  return out;
}

}  // namespace vllm
