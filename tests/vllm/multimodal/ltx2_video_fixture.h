// A REDUCED-DIMENSION LTX-2.5 CHECKPOINT SET, written in the SHIPPED FORMAT.
//
// Phase L7's gate is that the whole path runs: detection by tensor name, the
// phase-L6 quantized DiT loader, the metadata-borne VAE configs, the recipe, the
// denoise loop, both decoders, the vocoder, and the artifacts. Every one of
// those reads a FILE, so a fixture that hands the engine pre-built structs would
// gate none of them.
//
// So this writes real `.safetensors`: the DiT ComfyUI-prefixed and F8_E4M3
// quantized with per-tensor F32 `_scale` sidecars, exactly as
// `vonkaiser/LTX-2.5-FP8-NVFP4` is; the two VAEs bf16 with their config in
// `__metadata__["config"]`, exactly as `Lightricks/LTX-2.5` ships them.
//
// ─── WHY THE GEOMETRY IS WHAT IT IS ──────────────────────────────────────────
//
// Three numbers are NOT free, and a fixture that moved them would gate a
// pipeline the shipped one is not:
//
//   * The video VAE must downscale by exactly (8, 32, 32). `VIDEO_SCALE_FACTORS`
//     is a CONSTANT for the conv arm upstream (ltx-core types.py:70, reached
//     through utils/helpers.py:66-72), not something derived from the
//     checkpoint, so the decoder's own block list must multiply out to it:
//     patch_size 4 x compress_space 2 x compress_all 2 x compress_all 2 = 32
//     spatially, and compress_time 2 x compress_all 2 x compress_all 2 = 8
//     temporally.
//   * The audio latent is 8 channels x 16 mel bins (`AudioLatentShape`
//     defaults, types.py:184-200), and 8 x 16 = 128 IS the DiT's audio input
//     width. The audio VAE's `z_channels` must therefore be 8.
//   * The vocoder's `conv_pre` is hardcoded to 128 = 2 channels x 64 mel bins
//     upstream (vocoder.py:350-358), so the audio decoder must emit 64 mel bins,
//     which fixes `ch_mult` at three levels (16 x 2 x 2 = 64).
//
// Everything else — layer counts, widths, channel bases — is reduced.
//
// ─── ON DUPLICATING test_ltx2_vae.cpp's REDUCED BUILDERS ─────────────────────
//
// The two VAE parameter builders below express the same module contracts as the
// ones in tests/vllm/models/test_ltx2_vae.cpp, at a different scale, and they are
// deliberately INDEPENDENT rather than shared. This project has already recorded
// what sharing costs: a gate written against a shared helper proves the two arms
// AGREE, not that either is right (max|diff| == 0 over pure noise, because both
// arms called the same wrong dequant). A second, independently written statement
// of "which tensors does this module need, and at what shapes" is a real check on
// the first; one helper called twice is not.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_audio_vae.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/model_executor/models/ltx2_upsampler.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"

namespace ltx2_fixture {

// ── the deterministic stream ────────────────────────────────────────────────
// Per-tensor FNV-1a seed + splitmix64, the same shape every fixture in this tree
// uses: no weight byte is checked in, and a tensor's values depend only on its
// NAME, so adding one cannot perturb another.
inline uint64_t Fnv1a(const std::string& s) {
  uint64_t h = 1469598103934665603ULL;
  for (const char c : s) {
    h ^= static_cast<unsigned char>(c);
    h *= 1099511628211ULL;
  }
  return h;
}

inline std::vector<float> Param(const std::string& name, int64_t numel, double scale,
                                double offset = 0.0) {
  uint64_t state = Fnv1a(name);
  std::vector<float> out(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const double u = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    out[static_cast<size_t>(i)] = static_cast<float>((u * 2.0 - 1.0) * scale + offset);
  }
  return out;
}

// ── dtype encoders ─────────────────────────────────────────────────────────
inline uint16_t F32ToBf16(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  // Round-to-nearest-even on the truncated 16 low bits, which is what every
  // producer of a bf16 checkpoint does.
  const uint32_t rounded = bits + 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(rounded >> 16);
}

// IEEE fp8 E4M3FN (no infinities; 0x7F/0xFF are the only NaNs), the encoding
// `DequantFp8ToBf16` decodes (nvfp4_dequant.h:63-77). Saturating, RNE.
inline uint8_t F32ToE4M3(float v) {
  if (!(v == v)) return 0x7F;  // NaN in, NaN out
  const uint8_t sign = v < 0.0F ? 0x80U : 0x00U;
  double a = v < 0.0F ? -static_cast<double>(v) : static_cast<double>(v);
  constexpr double kMax = 448.0;
  if (a > kMax) a = kMax;
  constexpr double kSubnormalStep = 1.0 / 512.0;  // 2^-9
  constexpr double kMinNormal = 1.0 / 64.0;       // 2^-6, exponent field 1
  if (a < 0.5 * kSubnormalStep) return sign;      // rounds to zero
  // SUBNORMALS ARE A SEPARATE BRANCH AND MUST NOT FALL THROUGH. An earlier
  // revision of this encoder let a value below 2^-6 reach the normal path with a
  // mantissa still under 1.0, producing a NEGATIVE 3-bit fraction whose `|` into
  // the byte set the high bits — and 0xFF/0x7F are E4M3's only NaNs. 1.1% of a
  // uniform [-0.16, 0.16] draw lands below 2^-6, so the fixture's DiT came back
  // with 460 NaN weights and the whole render was NaN. The engine's own gate
  // caught it, which is the argument for having the gate at all.
  if (a < kMinNormal) {
    int q = static_cast<int>(a / kSubnormalStep + 0.5);
    if (q >= 8) return static_cast<uint8_t>(sign | 0x08U);  // rounded up to 2^-6
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(q));
  }
  int exp = 0;
  double mant = a;
  while (mant >= 2.0) {
    mant /= 2.0;
    ++exp;
  }
  while (mant < 1.0) {
    mant *= 2.0;
    --exp;
  }
  int frac = static_cast<int>((mant - 1.0) * 8.0 + 0.5);
  if (frac >= 8) {
    frac = 0;
    ++exp;
  }
  if (exp > 8 || (exp == 8 && frac > 6)) {
    return static_cast<uint8_t>(sign | 0x7EU);  // saturate at 448, never 0x7F (NaN)
  }
  const int biased = exp + 7;
  return static_cast<uint8_t>(sign | static_cast<uint8_t>(biased << 3) |
                              static_cast<uint8_t>(frac));
}

// ── a safetensors writer that can emit four dtypes ─────────────────────────
//
// "U8" is the ONE that does not go through `values`: it writes `raw` verbatim.
// The text encoder needs it for three things a float stream cannot express —
// NVFP4-packed projection weights (two 4-bit values per byte), the
// `torchao_nvfp4` marker's JSON payload, and the embedded tokenizer/asset pack,
// which the shipped file stores AS TENSORS (ltx2_text_encoder.h:336-346).
struct Entry {
  std::string name;
  std::string dtype;  // "F32" | "BF16" | "F8_E4M3" | "U8"
  std::vector<int64_t> shape;
  std::vector<float> values;
  // "U8" only. A default member initializer rather than a bare declaration
  // because every existing brace-initializer in this file supplies four fields,
  // and `-Wmissing-field-initializers` is an ERROR here: an NSDMI is what makes
  // the field genuinely optional instead of demanding a `{}` at 30 call sites
  // that have nothing to do with the text encoder.
  std::string raw = {};
};

inline void WriteFileBytes(const std::string& path, const std::string& bytes) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("ltx2 fixture: cannot write " + path);
  std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
}

inline void WriteSafetensors(const std::vector<Entry>& entries, const std::string& metadata_json,
                             const std::string& path) {
  std::string header = "{";
  bool first = true;
  if (!metadata_json.empty()) {
    header += "\"__metadata__\":" + metadata_json;
    first = false;
  }
  size_t offset = 0;
  std::string payload;
  for (const Entry& e : entries) {
    size_t bytes = 0;
    // `raw` is "these bytes are already encoded", and it wins over `values` for
    // ANY dtype rather than only for U8. The text encoder needs exactly that for
    // its `weight_scale`: the tensor is F8_E4M3, but its bytes are a SWIZZLED
    // permutation of E4M3 codes, so they cannot be produced by encoding a float
    // stream element-wise the way every other F8_E4M3 entry here is.
    if (!e.raw.empty()) {
      bytes = e.raw.size();
      payload += e.raw;
    } else if (e.dtype == "F32") {
      bytes = e.values.size() * sizeof(float);
      const size_t at = payload.size();
      payload.resize(at + bytes);
      std::memcpy(&payload[at], e.values.data(), bytes);
    } else if (e.dtype == "BF16") {
      bytes = e.values.size() * sizeof(uint16_t);
      for (const float v : e.values) {
        const uint16_t b = F32ToBf16(v);
        payload.append(reinterpret_cast<const char*>(&b), sizeof(b));
      }
    } else {
      bytes = e.values.size();
      for (const float v : e.values) payload.push_back(static_cast<char>(F32ToE4M3(v)));
    }
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i != 0) header += ",";
      header += std::to_string(e.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";
  std::string file;
  const uint64_t n = header.size();
  file.append(reinterpret_cast<const char*>(&n), sizeof(n));
  file += header;
  file += payload;
  WriteFileBytes(path, file);
}

// ── the DiT ────────────────────────────────────────────────────────────────

inline vllm::Ltx2DitParams ReducedDitParams() {
  vllm::Ltx2DitParams p;
  p.num_layers = 2;
  p.num_attention_heads = 2;
  p.attention_head_dim = 8;  // inner_dim 16
  p.in_channels = 4;
  p.out_channels = 4;
  p.cross_attention_dim = 16;  // must equal inner_dim (transformer_args.py:197)
  p.audio_num_attention_heads = 2;
  p.audio_attention_head_dim = 4;  // audio inner_dim 8
  // 8 latent channels x 16 mel bins — NOT free, see the header note.
  p.audio_in_channels = 128;
  p.audio_out_channels = 128;
  p.audio_cross_attention_dim = 8;
  p.rope_type = vllm::Ltx2RopeType::kSplit;
  p.use_middle_indices_grid = true;
  p.apply_gated_attention = true;   // every LTX-2.5 attention is gated
  p.cross_attention_adaln = true;   // the shipped config sets it
  p.use_prompt_adaln_single = false;
  p.ff_bias = false;                // LTX-2.5 (gemma4) sets ff_bias=false
  p.audio_ff_bias = true;
  return p;
}

// ── the embeddings connector ───────────────────────────────────────────────
//
// The two `*_embeddings_connector` families the shipped DiTs carry (129 tensors
// each) and that phase L9c wires into the render path. Reduced in every axis the
// shipped one is reduced in, and NOT reduced in the one that decides the
// substitution: `num_learnable_registers` tiles across the sequence, so the
// prompt row count must be a multiple of it.
struct ReducedConnectorOptions {
  bool present = true;
  int64_t num_layers = 2;
  int64_t num_learnable_registers = 2;
  bool gated = true;
  bool ff_bias = true;
  // Seeds the connector's own parameter stream. A second value writes a DIFFERENT
  // connector into an otherwise byte-identical checkpoint, which is how a test
  // proves the render actually READS these weights.
  std::string tag = "a";
  // Write the audio family too. `false` is the half-a-connector checkpoint the
  // engine refuses.
  bool audio = true;
};

// The connector configuration the fixture's own tensors are written from, so the
// config the engine parses and the shapes it finds can never disagree by
// accident — only when a test makes them.
inline vllm::Ltx2ConnectorConfig ReducedConnectorConfig(const vllm::Ltx2DitParams& params,
                                                        const ReducedConnectorOptions& options,
                                                        vllm::Ltx2ConnectorStream stream) {
  vllm::Ltx2ConnectorConfig c;
  c.prefix = vllm::Ltx2ConnectorCheckpointPrefix(stream);
  const bool video = stream == vllm::Ltx2ConnectorStream::kVideo;
  c.num_attention_heads = video ? params.num_attention_heads : params.audio_num_attention_heads;
  c.attention_head_dim = video ? params.attention_head_dim : params.audio_attention_head_dim;
  c.num_layers = options.num_layers;
  c.num_learnable_registers = options.num_learnable_registers;
  c.apply_gated_attention = options.gated;
  c.ff_bias = options.ff_bias;
  c.rope_type = vllm::Ltx2RopeType::kSplit;
  c.double_precision_rope = true;  // the shipped config's frequencies_precision
  c.positional_embedding_max_pos = {4096};
  return c;
}

// The DiT's `{"transformer": {...}}` object, exactly as the first-party
// `Lightricks/LTX-2.5` DiT carries it in `__metadata__["config"]`. Built
// separately from the file writer because two callers need it: the writer, and a
// test that has to hand the SAME config to the engine through the
// `dit_config_path` extra for a checkpoint that declares none.
inline nlohmann::json ReducedDitTransformerConfig(
    const vllm::Ltx2DitParams& params,
    const ReducedConnectorOptions& connector = ReducedConnectorOptions{}) {
  nlohmann::json transformer;
  transformer["_class_name"] = "AVTransformer3DModel";
  // Every key `LTXModelConfigurator.from_metadata` runs `check_config_value` on
  // (model_configurator.py:26-44). They are all fixed values, so a fixture that
  // omitted one would be refused — which is the point: the shipped checkpoints
  // carry them and a config that does not is not an LTX-2.5 config.
  transformer["dropout"] = 0.0;
  transformer["attention_bias"] = true;
  transformer["num_vector_embeds"] = nullptr;
  transformer["activation_fn"] = "gelu-approximate";
  transformer["num_embeds_ada_norm"] = 1000;
  transformer["use_linear_projection"] = false;
  transformer["only_cross_attention"] = false;
  transformer["cross_attention_norm"] = true;
  transformer["double_self_attention"] = false;
  transformer["upcast_attention"] = false;
  transformer["standardization_norm"] = "rms_norm";
  transformer["norm_elementwise_affine"] = false;
  transformer["qk_norm"] = "rms_norm";
  transformer["positional_embedding_type"] = "rope";
  transformer["use_audio_video_cross_attention"] = true;
  transformer["share_ff"] = false;
  transformer["av_cross_ada_norm"] = true;
  // The 22B form: the caption projections live in the TEXT ENCODER, not in the
  // DiT (model_configurator.py:199-219). LTX-2.5 is that form.
  //
  // ALL FOUR of `Ltx2SelectTextFeatureVariant`'s markers, not just the first.
  // Upstream selects V1 vs V2 from the presence AND the exact values of the set
  // (encoder_configurator.py:163-209), and a PARTIAL set is refused as config
  // drift rather than resolved either way. An earlier revision of this fixture
  // carried only `caption_proj_before_connector`, which is the partial case —
  // `Ltx2SelectTextFeatureVariant` REFUSES it (ltx2_text_encoder.cpp:184-192).
  // The reason that refusal never fired is NOT that the partial set gated
  // nothing: it is that no production path CALLED the selector before L13. The
  // four marker keys and the engine's first call to the selector landed in the
  // same commit, so there was no earlier run for the partial set to refuse.
  //
  // THE OBSERVED HEADER, recorded so the next reader does not need the 18.72 GB
  // checkpoint mounted to check these four values. Read 2026-08-13 from the
  // safetensors `__metadata__` of the first-party NVFP4 DiT — header JSON only,
  // no tensor data:
  //
  //   /mnt/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/diffusion_models/
  //       ltx-2.5-22b-distilled-transformer-nvfp4.safetensors
  //   18,721,432,024 bytes; `Lightricks/LTX-2.5` revision
  //   8a4ff96f581e72bedc1b44367581c49d544a05f1 (HF download record; the LFS oid
  //   is the upstream sha256 and was NOT re-verified locally).
  //
  //   __metadata__["config"]["transformer"]:
  //       caption_proj_before_connector     true
  //       caption_projection_first_linear   false
  //       caption_proj_input_norm           false
  //       caption_projection_second_linear  false
  //       num_attention_heads 32, attention_head_dim 128        -> video 4096
  //       audio_num_attention_heads 32, audio_attention_head_dim 64 -> audio 2048
  //       text_encoder_norm_type "PER_TOKEN_RMS", model_version "2.5.0"
  //
  // So this checkpoint resolves to V2 with no key missing and no value drifted.
  // `text_encoder_norm_type` independently corroborates that, and the selector
  // deliberately does NOT read it — it mirrors upstream's four-marker test.
  //
  // The vonkaiser FP8 DiT that the render arms actually load carries NO
  // `__metadata__` block at all, so this first-party file is the only on-disk
  // source for these four values. The four below are READ from it, not assumed
  // from the class defaults.
  transformer["caption_proj_before_connector"] = true;
  transformer["caption_projection_first_linear"] = false;
  transformer["caption_proj_input_norm"] = false;
  transformer["caption_projection_second_linear"] = false;
  // The values the SHAPES cannot see, which is why the engine reads this config
  // at all — they are the shipped ones.
  transformer["norm_eps"] = 1e-6;
  transformer["positional_embedding_theta"] = 10000.0;
  transformer["positional_embedding_max_pos"] = std::vector<int64_t>{20, 2048, 2048};
  transformer["audio_positional_embedding_max_pos"] = std::vector<int64_t>{20};
  transformer["timestep_scale_multiplier"] = 1000;
  transformer["av_ca_timestep_scale_multiplier"] = 1000.0;
  transformer["frequencies_precision"] = "float64";
  transformer["num_layers"] = params.num_layers;
  transformer["num_attention_heads"] = params.num_attention_heads;
  transformer["attention_head_dim"] = params.attention_head_dim;
  transformer["in_channels"] = params.in_channels;
  transformer["out_channels"] = params.out_channels;
  transformer["cross_attention_dim"] = params.cross_attention_dim;
  transformer["audio_num_attention_heads"] = params.audio_num_attention_heads;
  transformer["audio_attention_head_dim"] = params.audio_attention_head_dim;
  transformer["audio_out_channels"] = params.audio_out_channels;
  transformer["audio_cross_attention_dim"] = params.audio_cross_attention_dim;
  transformer["apply_gated_attention"] = params.apply_gated_attention;
  transformer["cross_attention_adaln"] = params.cross_attention_adaln;
  transformer["ff_bias"] = params.ff_bias;
  transformer["rope_type"] = "split";
  transformer["use_middle_indices_grid"] = params.use_middle_indices_grid;
  // The `connector_*` keys the two Embeddings1DConnector configurators read
  // (embeddings_connector.py:194-256). `connector_positional_embedding_max_pos`
  // is the one the shipped config moves OFF its class default of [1], and it
  // divides every token index, so it is written at the shipped [4096].
  transformer["connector_num_attention_heads"] = params.num_attention_heads;
  transformer["connector_attention_head_dim"] = params.attention_head_dim;
  transformer["audio_connector_num_attention_heads"] = params.audio_num_attention_heads;
  transformer["audio_connector_attention_head_dim"] = params.audio_attention_head_dim;
  transformer["connector_num_layers"] = connector.num_layers;
  transformer["connector_num_learnable_registers"] = connector.num_learnable_registers;
  transformer["connector_apply_gated_attention"] = connector.gated;
  transformer["connector_ff_bias"] = connector.ff_bias;
  transformer["connector_positional_embedding_max_pos"] = std::vector<int64_t>{4096};
  return transformer;
}

// How a fixture DiT declares itself. Every field mirrors a shape a SHIPPED file
// is actually in, measured 2026-08-12 — none of them is a hypothetical:
//
//   declare_config = true  + declare_model_version = true
//       the first-party `Lightricks/LTX-2.5` NVFP4 DiT
//       (`__metadata__` = config, gemma_source_checkpoint, license, model_version)
//   declare_config = false + declare_model_version = false
//       the `vonkaiser/LTX-2.5-FP8-NVFP4` FP8 DiT, which carries NO
//       `__metadata__` key at all
//
// `transformer_overrides` is merged into the config AFTER it is built and AFTER
// the tensors have been written from `params`, which is how a test makes a
// declared config DISAGREE with the shapes beside it.
struct ReducedDitOptions {
  std::string model_version = "2.5.0";
  bool declare_config = true;
  bool declare_model_version = true;
  nlohmann::json transformer_overrides = nlohmann::json::object();
  ReducedConnectorOptions connector;
};

// Write the DiT in the shipped ComfyUI + FP8 shape: every rank-2 `*.weight`
// becomes F8_E4M3 with an F32 `<name>_scale` sidecar, the scale-shift tables stay
// F32 (the shipped file stores them F32), and everything else is BF16 — which is
// exactly the dtype split `vonkaiser/LTX-2.5-FP8-NVFP4` carries.
inline void WriteReducedDit(const vllm::Ltx2DitParams& params, const std::string& path,
                            const ReducedDitOptions& options) {
  std::vector<Entry> entries;
  const std::string prefix = vllm::kLtx2DitCheckpointPrefix;
  for (const vllm::Ltx2TensorSpec& spec : vllm::EnumerateLtx2DitTensors(params)) {
    int64_t numel = 1;
    for (const int64_t d : spec.shape) numel *= d;
    const std::string full = prefix + spec.name;
    const bool table = spec.name.find("scale_shift_table") != std::string::npos;
    const bool quantizable = spec.shape.size() == 2 && !table &&
                             spec.name.size() > 7 &&
                             spec.name.compare(spec.name.size() - 7, 7, ".weight") == 0;
    std::vector<float> values = Param("ltx2.dit." + spec.name, numel, 0.08);
    if (table) {
      entries.push_back({full, "F32", spec.shape, std::move(values)});
    } else if (quantizable) {
      // The FP8 arm's own convention: bytes carry the SHAPE of the weight and the
      // per-tensor F32 scale carries its magnitude.
      constexpr float kScale = 0.5F;
      for (float& v : values) v /= kScale;
      entries.push_back({full, "F8_E4M3", spec.shape, std::move(values)});
      entries.push_back({full + "_scale", "F32", {}, {kScale}});
    } else {
      entries.push_back({full, "BF16", spec.shape, std::move(values)});
    }
  }
  // The connector families, written in the SAME FP8/BF16 split. They sit beside
  // the DiT contract, not inside it — upstream loads them into the text
  // encoder's EmbeddingsProcessor (encoder_configurator.py:331-346) — which is
  // why they are enumerated from their own contract rather than the DiT's.
  if (options.connector.present) {
    std::vector<vllm::Ltx2ConnectorStream> streams = {vllm::Ltx2ConnectorStream::kVideo};
    if (options.connector.audio) streams.push_back(vllm::Ltx2ConnectorStream::kAudio);
    for (const vllm::Ltx2ConnectorStream stream : streams) {
      const vllm::Ltx2ConnectorConfig c =
          ReducedConnectorConfig(params, options.connector, stream);
      for (const vllm::Ltx2ConnectorTensorSpec& spec : vllm::EnumerateLtx2ConnectorTensors(c)) {
        int64_t numel = 1;
        for (const int64_t d : spec.shape) numel *= d;
        const std::string full = prefix + spec.name;
        std::vector<float> values =
            Param("ltx2.conn." + options.connector.tag + "." + spec.name, numel, 0.08);
        // MEASURED from the shipped `vonkaiser` FP8 DiT header: every rank-2
        // tensor of this family is F8_E4M3 with an F32 sidecar, `learnable_registers`
        // INCLUDED (`...learnable_registers` + `...learnable_registers_scale`).
        // That is not the DiT's rule — there the scale-shift tables stay F32 —
        // so it is stated from the file rather than inherited.
        const bool quantizable = spec.shape.size() == 2;
        if (quantizable) {
          constexpr float kScale = 0.5F;
          for (float& v : values) v /= kScale;
          entries.push_back({full, "F8_E4M3", spec.shape, std::move(values)});
          entries.push_back({full + "_scale", "F32", {}, {kScale}});
        } else {
          entries.push_back({full, "BF16", spec.shape, std::move(values)});
        }
      }
    }
  }
  nlohmann::json metadata = nlohmann::json::object();
  if (options.declare_config) {
    nlohmann::json transformer = ReducedDitTransformerConfig(params, options.connector);
    transformer.update(options.transformer_overrides);
    nlohmann::json config;
    config["transformer"] = transformer;
    metadata["config"] = config.dump();
  }
  if (options.declare_model_version) metadata["model_version"] = options.model_version;
  // An EMPTY object writes no `__metadata__` key at all, which is the shape the
  // shipped vonkaiser FP8 DiT is in — not an empty one.
  WriteSafetensors(entries, metadata.empty() ? std::string() : metadata.dump(), path);
}

inline void WriteReducedDit(const vllm::Ltx2DitParams& params, const std::string& path,
                            const std::string& model_version = "2.5.0") {
  ReducedDitOptions options;
  options.model_version = model_version;
  WriteReducedDit(params, path, options);
}

// The same `{"transformer": {...}}` object as a standalone JSON FILE, which is
// what the engine's `dit_config_path` extra reads.
inline void WriteDitConfigJson(const vllm::Ltx2DitParams& params, const std::string& path,
                               const nlohmann::json& transformer_overrides =
                                   nlohmann::json::object(),
                               const ReducedConnectorOptions& connector =
                                   ReducedConnectorOptions{}) {
  nlohmann::json transformer = ReducedDitTransformerConfig(params, connector);
  transformer.update(transformer_overrides);
  nlohmann::json config;
  config["transformer"] = transformer;
  WriteFileBytes(path, config.dump());
}

// ── the Conv video VAE ─────────────────────────────────────────────────────

inline vllm::Ltx2ConvVideoDecoderConfig ReducedVideoDecoderConfig(int64_t latent_channels) {
  vllm::Ltx2ConvVideoDecoderConfig cfg;
  cfg.in_channels = latent_channels;
  cfg.out_channels = 3;
  cfg.patch_size = 4;
  cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  cfg.causal = false;                // the shipped config: causal_decoder false
  cfg.timestep_conditioning = false; // the shipped config
  cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kZeros;
  cfg.base_channels = 4;
  // The shipped block list with every res_x reduced to one layer. The four
  // compress blocks are what make the scale factors (8, 32, 32) — see the header.
  cfg.decoder_blocks = {
      {"res_x", 1, 0, false, false},        {"compress_space", 1, 2, false, false},
      {"res_x", 1, 0, false, false},        {"compress_time", 1, 2, false, false},
      {"res_x", 1, 0, false, false},        {"compress_all", 1, 1, false, false},
      {"res_x", 1, 0, false, false},        {"compress_all", 1, 2, false, false},
      {"res_x", 1, 0, false, false},
  };
  return cfg;
}

inline void WriteReducedVideoVae(const vllm::Ltx2ConvVideoDecoderConfig& cfg,
                                 const std::string& path) {
  std::vector<Entry> entries;
  auto put = [&](const std::string& name, const std::vector<int64_t>& shape, double scale,
                 double offset = 0.0) {
    int64_t numel = 1;
    for (const int64_t d : shape) numel *= d;
    entries.push_back(
        {"decoder." + name, "BF16", shape, Param("ltx2.vvae." + name, numel, scale, offset)});
  };

  int64_t multiplier = 1;
  for (const vllm::Ltx2VideoDecoderBlock& b : cfg.decoder_blocks) {
    if (b.name == "compress_time" || b.name == "compress_space" || b.name == "compress_all") {
      multiplier *= b.multiplier != 0 ? b.multiplier : 1;
    } else if (b.name == "res_x_y") {
      multiplier *= b.multiplier != 0 ? b.multiplier : 2;
    }
  }
  int64_t channels = cfg.base_channels * multiplier;

  entries.push_back({"per_channel_statistics.std-of-means", "BF16", {cfg.in_channels},
                     Param("ltx2.vvae.std", cfg.in_channels, 0.1, 1.0)});
  entries.push_back({"per_channel_statistics.mean-of-means", "BF16", {cfg.in_channels},
                     Param("ltx2.vvae.mean", cfg.in_channels, 0.1)});
  put("conv_in.conv.weight", {channels, cfg.in_channels, 3, 3, 3}, 0.1);
  put("conv_in.conv.bias", {channels}, 0.05);

  auto put_resnet3d = [&](const std::string& p, int64_t in_ch, int64_t out_ch) {
    put(p + ".conv1.conv.weight", {out_ch, in_ch, 3, 3, 3}, 0.1);
    put(p + ".conv1.conv.bias", {out_ch}, 0.05);
    put(p + ".conv2.conv.weight", {out_ch, out_ch, 3, 3, 3}, 0.1);
    put(p + ".conv2.conv.bias", {out_ch}, 0.05);
    if (in_ch != out_ch) {
      put(p + ".conv_shortcut.weight", {out_ch, in_ch, 1, 1, 1}, 0.1);
      put(p + ".conv_shortcut.bias", {out_ch}, 0.05);
      put(p + ".norm3.weight", {in_ch}, 0.1, 1.0);
      put(p + ".norm3.bias", {in_ch}, 0.05);
    }
  };

  int64_t index = 0;
  for (auto it = cfg.decoder_blocks.rbegin(); it != cfg.decoder_blocks.rend(); ++it, ++index) {
    const vllm::Ltx2VideoDecoderBlock& block = *it;
    const std::string bp = "up_blocks." + std::to_string(index);
    if (block.name == "res_x") {
      for (int64_t i = 0; i < block.num_layers; ++i) {
        put_resnet3d(bp + ".res_blocks." + std::to_string(i), channels, channels);
      }
    } else {
      int64_t stride_product = 2;
      if (block.name == "compress_space") stride_product = 4;
      if (block.name == "compress_all") stride_product = 8;
      const int64_t reduction = block.multiplier != 0 ? block.multiplier : 1;
      const int64_t conv_out = stride_product * channels / reduction;
      put(bp + ".conv.conv.weight", {conv_out, channels, 3, 3, 3}, 0.1);
      put(bp + ".conv.conv.bias", {conv_out}, 0.05);
      channels /= reduction;
    }
  }
  const int64_t patch_out = cfg.out_channels * cfg.patch_size * cfg.patch_size;
  put("conv_out.conv.weight", {patch_out, channels, 3, 3, 3}, 0.1);
  put("conv_out.conv.bias", {patch_out}, 0.05);

  nlohmann::json vae;
  vae["_class_name"] = "CausalVideoAutoencoder";
  vae["dims"] = 3;
  vae["in_channels"] = 3;
  vae["out_channels"] = cfg.out_channels;
  vae["latent_channels"] = cfg.in_channels;
  vae["patch_size"] = cfg.patch_size;
  vae["norm_layer"] = "pixel_norm";
  vae["causal_decoder"] = cfg.causal;
  vae["timestep_conditioning"] = cfg.timestep_conditioning;
  vae["spatial_padding_mode"] = "zeros";
  vae["decoder_base_channels"] = cfg.base_channels;
  nlohmann::json blocks = nlohmann::json::array();
  for (const vllm::Ltx2VideoDecoderBlock& b : cfg.decoder_blocks) {
    nlohmann::json params = nlohmann::json::object();
    if (b.name == "res_x") {
      params["num_layers"] = b.num_layers;
    } else {
      params["multiplier"] = b.multiplier;
    }
    blocks.push_back(nlohmann::json::array({b.name, params}));
  }
  vae["decoder_blocks"] = blocks;
  nlohmann::json config;
  config["vae"] = vae;
  nlohmann::json metadata;
  metadata["config"] = config.dump();
  metadata["model_version"] = "2.5.0";
  WriteSafetensors(entries, metadata.dump(), path);
}

// ── the audio VAE + its BWE vocoder ────────────────────────────────────────

inline vllm::Ltx2AudioDecoderConfig ReducedAudioDecoderConfig() {
  vllm::Ltx2AudioDecoderConfig cfg;
  cfg.ch = 4;
  cfg.out_ch = 2;
  cfg.ch_mult = {1, 2, 4};  // two upsample stages: 16 latent mel bins -> 64
  cfg.num_res_blocks = 1;
  cfg.attn_resolutions = {};  // the shipped config carries none
  cfg.resolution = 256;
  cfg.z_channels = 8;  // == the audio latent's channel count
  cfg.norm_type = vllm::Ltx2NormType::kPixel;
  cfg.causality_axis = vllm::Ltx2CausalityAxis::kHeight;
  cfg.mid_block_add_attention = false;  // the shipped config
  cfg.mel_bins = 64;
  return cfg;
}

inline vllm::Ltx2VocoderBweConfig ReducedVocoderBweConfig() {
  vllm::Ltx2VocoderBweConfig cfg;
  // The RATES are the shipped ones — they set the sample-rate arithmetic
  // (160 = 16000/100 frames, 240/80 = the 3x band extension), and moving them
  // would make the fixture's audio timeline a different one from production's.
  // Only the WIDTHS are reduced.
  cfg.vocoder.resblock_kernel_sizes = {3};
  cfg.vocoder.upsample_rates = {5, 2, 2, 2, 2, 2};
  cfg.vocoder.upsample_kernel_sizes = {11, 4, 4, 4, 4, 4};
  cfg.vocoder.resblock_dilation_sizes = {{1, 3, 5}};
  cfg.vocoder.upsample_initial_channel = 128;  // 6 halvings -> 2 output channels
  cfg.vocoder.amp = true;
  cfg.vocoder.snakebeta = true;
  cfg.vocoder.use_tanh_at_final = false;
  cfg.vocoder.apply_final_activation = true;
  cfg.vocoder.use_bias_at_final = false;
  cfg.vocoder.output_sampling_rate = 16000;
  cfg.vocoder.prefix = "vocoder.";

  cfg.bwe_generator = cfg.vocoder;
  cfg.bwe_generator.upsample_rates = {6, 5, 2, 2, 2};
  cfg.bwe_generator.upsample_kernel_sizes = {12, 11, 4, 4, 4};
  cfg.bwe_generator.upsample_initial_channel = 64;  // 5 halvings -> 2
  cfg.bwe_generator.apply_final_activation = false;
  cfg.bwe_generator.output_sampling_rate = 48000;
  cfg.bwe_generator.prefix = "bwe_generator.";

  cfg.filter_length = 512;
  cfg.hop_length = 80;
  cfg.win_length = 512;
  cfg.n_mel_channels = 64;
  cfg.input_sampling_rate = 16000;
  cfg.output_sampling_rate = 48000;
  return cfg;
}

inline void WriteReducedAudioVae(const vllm::Ltx2AudioDecoderConfig& cfg,
                                 const vllm::Ltx2VocoderBweConfig& voc,
                                 const std::string& path) {
  std::vector<Entry> entries;
  auto put = [&](const std::string& name, const std::vector<int64_t>& shape, double scale,
                 double offset = 0.0) {
    int64_t numel = 1;
    for (const int64_t d : shape) numel *= d;
    entries.push_back({name, "BF16", shape, Param("ltx2.avae." + name, numel, scale, offset)});
  };

  // --- the decoder ---
  const std::string dp = "audio_vae.decoder.";
  const int64_t levels = static_cast<int64_t>(cfg.ch_mult.size());
  const int64_t base = cfg.ch * cfg.ch_mult[static_cast<size_t>(levels - 1)];
  const int64_t patched = cfg.z_channels * 16;  // channels x latent mel bins
  put("audio_vae.per_channel_statistics.std-of-means", {patched}, 0.05, 1.0);
  put("audio_vae.per_channel_statistics.mean-of-means", {patched}, 0.05);
  put(dp + "conv_in.conv.weight", {base, cfg.z_channels, 3, 3}, 0.1);
  put(dp + "conv_in.conv.bias", {base}, 0.05);

  auto put_resnet = [&](const std::string& p, int64_t in_ch, int64_t out_ch) {
    put(p + ".conv1.conv.weight", {out_ch, in_ch, 3, 3}, 0.1);
    put(p + ".conv1.conv.bias", {out_ch}, 0.05);
    put(p + ".conv2.conv.weight", {out_ch, out_ch, 3, 3}, 0.1);
    put(p + ".conv2.conv.bias", {out_ch}, 0.05);
    if (in_ch != out_ch) {
      put(p + ".nin_shortcut.conv.weight", {out_ch, in_ch, 1, 1}, 0.1);
      put(p + ".nin_shortcut.conv.bias", {out_ch}, 0.05);
    }
  };
  put_resnet(dp + "mid.block_1", base, base);
  put_resnet(dp + "mid.block_2", base, base);

  int64_t block_in = base;
  struct Stage {
    std::vector<std::pair<int64_t, int64_t>> blocks;
    int64_t upsample = 0;
  };
  std::vector<Stage> stages(static_cast<size_t>(levels));
  for (int64_t level = levels - 1; level >= 0; --level) {
    Stage& stage = stages[static_cast<size_t>(level)];
    const int64_t block_out = cfg.ch * cfg.ch_mult[static_cast<size_t>(level)];
    for (int64_t i = 0; i < cfg.num_res_blocks + 1; ++i) {
      stage.blocks.emplace_back(block_in, block_out);
      block_in = block_out;
    }
    if (level != 0) stage.upsample = block_in;
  }
  for (int64_t level = 0; level < levels; ++level) {
    const Stage& stage = stages[static_cast<size_t>(level)];
    const std::string sp = dp + "up." + std::to_string(level);
    for (size_t i = 0; i < stage.blocks.size(); ++i) {
      put_resnet(sp + ".block." + std::to_string(i), stage.blocks[i].first,
                 stage.blocks[i].second);
    }
    if (stage.upsample != 0) {
      put(sp + ".upsample.conv.conv.weight", {stage.upsample, stage.upsample, 3, 3}, 0.1);
      put(sp + ".upsample.conv.conv.bias", {stage.upsample}, 0.05);
    }
  }
  put(dp + "conv_out.conv.weight", {cfg.out_ch, block_in, 3, 3}, 0.1);
  put(dp + "conv_out.conv.bias", {cfg.out_ch}, 0.05);

  // --- the two vocoder arms, under `vocoder.<arm>.` ---
  auto put_vocoder = [&](const vllm::Ltx2VocoderConfig& v) {
    const std::string p = "vocoder." + v.prefix;
    const int64_t initial = v.upsample_initial_channel;
    const int64_t num_kernels = static_cast<int64_t>(v.resblock_kernel_sizes.size());
    put(p + "conv_pre.weight", {initial, 128, 7}, 0.1);
    put(p + "conv_pre.bias", {initial}, 0.05);
    for (size_t i = 0; i < v.upsample_rates.size(); ++i) {
      const int64_t in_ch = initial / (int64_t{1} << i);
      const int64_t out_ch = initial / (int64_t{1} << (i + 1));
      put(p + "ups." + std::to_string(i) + ".weight",
          {in_ch, out_ch, v.upsample_kernel_sizes[i]}, 0.1);
      put(p + "ups." + std::to_string(i) + ".bias", {out_ch}, 0.05);
    }
    for (size_t i = 0; i < v.upsample_rates.size(); ++i) {
      const int64_t ch = initial / (int64_t{1} << (i + 1));
      for (int64_t j = 0; j < num_kernels; ++j) {
        const std::string block =
            p + "resblocks." + std::to_string(static_cast<int64_t>(i) * num_kernels + j);
        const int64_t kernel = v.resblock_kernel_sizes[static_cast<size_t>(j)];
        for (const char* group : {"convs1", "convs2"}) {
          for (int64_t d = 0; d < 3; ++d) {
            put(block + "." + group + "." + std::to_string(d) + ".weight", {ch, ch, kernel}, 0.1);
            put(block + "." + group + "." + std::to_string(d) + ".bias", {ch}, 0.05);
          }
        }
        for (const char* group : {"acts1", "acts2"}) {
          for (int64_t d = 0; d < 3; ++d) {
            const std::string act = block + "." + group + "." + std::to_string(d) + ".act.";
            put(act + "alpha", {ch}, 0.1);
            put(act + "beta", {ch}, 0.1);
          }
        }
      }
    }
    const int64_t final_channels = initial / (int64_t{1} << v.upsample_rates.size());
    // act_post is UNCONDITIONALLY SnakeBeta (vocoder.py:388), so it always
    // carries `.beta` — the trap test_ltx2_vae.cpp records.
    put(p + "act_post.act.alpha", {final_channels}, 0.1);
    put(p + "act_post.act.beta", {final_channels}, 0.1);
    put(p + "conv_post.weight", {2, final_channels, 7}, 0.1);
    if (v.use_bias_at_final) put(p + "conv_post.bias", {2}, 0.05);
  };
  put_vocoder(voc.vocoder);
  put_vocoder(voc.bwe_generator);

  // The MelSTFT's two bases are BUFFERS in the checkpoint, not computed.
  const int64_t n_freqs = voc.filter_length / 2 + 1;
  put("vocoder.mel_stft.stft_fn.forward_basis", {2 * n_freqs, 1, voc.filter_length}, 0.05);
  put("vocoder.mel_stft.stft_fn.inverse_basis", {2 * n_freqs, 1, voc.filter_length}, 0.05);
  // Non-negative, like a real mel filterbank.
  put("vocoder.mel_stft.mel_basis", {voc.n_mel_channels, n_freqs}, 0.02, 0.03);

  nlohmann::json dd;
  dd["ch"] = cfg.ch;
  dd["out_ch"] = cfg.out_ch;
  dd["ch_mult"] = cfg.ch_mult;
  dd["num_res_blocks"] = cfg.num_res_blocks;
  dd["attn_resolutions"] = nlohmann::json::array();
  dd["resolution"] = cfg.resolution;
  dd["z_channels"] = cfg.z_channels;
  dd["norm_type"] = "pixel";
  dd["causality_axis"] = "height";
  dd["mid_block_add_attention"] = cfg.mid_block_add_attention;
  dd["mel_bins"] = cfg.mel_bins;
  nlohmann::json audio_vae;
  audio_vae["model"]["params"]["ddconfig"] = dd;
  audio_vae["model"]["params"]["sampling_rate"] = 16000;
  audio_vae["preprocessing"]["stft"]["hop_length"] = 160;
  audio_vae["preprocessing"]["stft"]["causal"] = true;
  audio_vae["preprocessing"]["mel"]["n_mel_channels"] = cfg.mel_bins;

  auto vocoder_json = [](const vllm::Ltx2VocoderConfig& v) {
    nlohmann::json j;
    j["resblock"] = "AMP1";
    j["stereo"] = true;
    j["activation"] = "snakebeta";
    j["resblock_kernel_sizes"] = v.resblock_kernel_sizes;
    j["upsample_rates"] = v.upsample_rates;
    j["upsample_kernel_sizes"] = v.upsample_kernel_sizes;
    j["resblock_dilation_sizes"] = v.resblock_dilation_sizes;
    j["upsample_initial_channel"] = v.upsample_initial_channel;
    j["use_tanh_at_final"] = v.use_tanh_at_final;
    j["use_bias_at_final"] = v.use_bias_at_final;
    return j;
  };
  nlohmann::json vocoder;
  vocoder["vocoder"] = vocoder_json(voc.vocoder);
  vocoder["bwe"] = vocoder_json(voc.bwe_generator);
  vocoder["bwe"]["input_sampling_rate"] = voc.input_sampling_rate;
  vocoder["bwe"]["output_sampling_rate"] = voc.output_sampling_rate;
  vocoder["bwe"]["hop_length"] = voc.hop_length;
  vocoder["bwe"]["n_fft"] = voc.filter_length;
  vocoder["bwe"]["num_mels"] = voc.n_mel_channels;
  vocoder["bwe"]["apply_final_activation"] = false;

  nlohmann::json config;
  config["audio_vae"] = audio_vae;
  config["vocoder"] = vocoder;
  nlohmann::json metadata;
  metadata["config"] = config.dump();
  metadata["model_version"] = "2.5.0";
  WriteSafetensors(entries, metadata.dump(), path);
}

// ── the latent spatial x2 upsampler (the two-stage recipe's phase 2) ────────

inline vllm::Ltx2UpsamplerConfig ReducedUpsamplerConfig(int64_t latent_channels) {
  vllm::Ltx2UpsamplerConfig cfg;
  cfg.in_channels = latent_channels;
  // GroupNorm(32, mid_channels) is a LITERAL at all three construction sites
  // (upsampler res_block.py:24,26 and model.py:50), so the width is not free to
  // reduce below 32 — the fixture reduces the DEPTH instead.
  cfg.mid_channels = 32;
  cfg.num_blocks_per_stage = 1;
  cfg.dims = 3;
  cfg.spatial_upsample = true;
  cfg.temporal_upsample = false;
  cfg.spatial_scale = 2.0;
  cfg.rational_resampler = false;
  return cfg;
}

inline void WriteReducedUpsampler(const vllm::Ltx2UpsamplerConfig& cfg, const std::string& path) {
  std::vector<Entry> entries;
  for (const vllm::Ltx2UpsamplerTensorSpec& spec : vllm::EnumerateLtx2UpsamplerTensors(cfg)) {
    int64_t numel = 1;
    for (const int64_t d : spec.shape) numel *= d;
    entries.push_back(
        {spec.name, "BF16", spec.shape, Param("ltx2.ups." + spec.name, numel, 0.1)});
  }
  nlohmann::json config;
  config["_class_name"] = "LatentUpsampler";
  config["in_channels"] = cfg.in_channels;
  config["mid_channels"] = cfg.mid_channels;
  config["num_blocks_per_stage"] = cfg.num_blocks_per_stage;
  config["dims"] = cfg.dims;
  config["spatial_upsample"] = cfg.spatial_upsample;
  config["temporal_upsample"] = cfg.temporal_upsample;
  config["spatial_scale"] = cfg.spatial_scale;
  config["rational_resampler"] = cfg.rational_resampler;
  nlohmann::json metadata;
  metadata["config"] = config.dump();
  WriteSafetensors(entries, metadata.dump(), path);
}

// ── the Gemma-4 TEXT ENCODER (phase L13) ───────────────────────────────────
//
// The shipped `vonkaiser/LTX-2.5-FP8-NVFP4` text encoder is ONE safetensors file
// carrying four separable things, and this writes all four in their shipped
// forms because the engine reads all four:
//
//   1. the Gemma-4 tower  `model.embed_tokens`, `model.norm`, `model.layers.{i}.*`
//   2. the two caption projections `text_embedding_projection.*_aggregate_embed`
//   3. the embedded tokenizer/asset pack, stored AS TENSORS
//   4. NO `__metadata__` at all — which is why the Gemma config is an INPUT
//
// TWO dtype rules, both measured from the shipped file rather than chosen:
//
//   * the TOWER may be BF16 (`Ltx2LoadGemmaTowerFromSafetensors` takes either
//     form), so the fixture writes it BF16 — the NVFP4 tower arm is gated
//     against the SHIPPED checkpoint in tests/vllm/models/test_ltx2_text_encoder.cpp
//     and duplicating it here would gate the same dequant twice.
//   * the PROJECTIONS have no BF16 form at all: `LoadProjection`
//     (ltx2_loader.cpp:583-623) requires `weight_scale` + `weight_scale_2` and
//     goes through `Ltx2DequantTorchaoNvfp4ToBf16` unconditionally. So they are
//     written torchao-NVFP4, swizzled, with a marker — anything else does not
//     load, and a fixture that dodged that would gate a file shape no shipped
//     checkpoint is in.

// e2m1 -> the byte pairs `DequantNvfp4ToBf16` reads. Deterministic per NAME, in
// the same shape every fixture in this tree uses.
inline std::string Nvfp4PackedBytes(const std::string& name, size_t count) {
  uint64_t state = Fnv1a(name);
  std::string out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    // Every one of the 256 byte values is a pair of VALID e2m1 codes — the
    // format has no NaN and no infinity — so unlike the E4M3 scale below this
    // needs no exclusion.
    out.push_back(static_cast<char>(static_cast<uint8_t>(z & 0xFFU)));
  }
  return out;
}

// The E4M3 group scales, LINEAR [rows, groups], before the swizzle. Confined to
// 0x30..0x3F, i.e. [0.5, 1.875): E4M3's only NaNs are 0x7F/0xFF, and a scale of
// 448 against an e2m1 6.0 would put the fixture's weights four orders above the
// DiT's — finite, and nothing like a checkpoint.
inline std::vector<uint8_t> Nvfp4LinearScale(const std::string& name, size_t count) {
  uint64_t state = Fnv1a(name);
  std::vector<uint8_t> out(count);
  for (size_t i = 0; i < count; ++i) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    out[i] = static_cast<uint8_t>(0x30U | static_cast<uint8_t>(z & 0x0FU));
  }
  return out;
}

// The FORWARD cuBLAS block-scale swizzle, PADDED — the inverse of
// `Ltx2UnswizzleNvfp4BlockScale` (ltx2_loader.h:160-176), transcribed from the
// same source it cites (vLLM `qutlass_utils.py:177-179`).
//
// The padding is the part that is NOT optional here and IS optional in
// tests/vllm/models/test_ltx2_loader.cpp's copy, which `REQUIRE`s rows % 128 == 0.
// A reduced fixture's projection is 16 and 8 rows wide, so it never satisfies
// that; the loader nonetheless demands the buffer be exactly
// round_up(rows,128) * round_up(cols,4) bytes, and the pad rows are never read.
inline std::string SwizzleBlockScalePadded(const std::vector<uint8_t>& linear, int64_t rows,
                                           int64_t cols) {
  const int64_t rows_padded = ((rows + 127) / 128) * 128;
  const int64_t cols_padded = ((cols + 3) / 4) * 4;
  const int64_t ctiles = cols_padded / 4;
  std::string out(static_cast<size_t>(rows_padded * cols_padded), '\0');
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t rt = r / 128, rem = r % 128, a = rem / 32, s = rem % 32;
      const int64_t ct = c / 4, q = c % 4;
      out[static_cast<size_t>(((((rt * ctiles) + ct) * 32 + s) * 4 + a) * 4 + q)] =
          static_cast<char>(linear[static_cast<size_t>(r * cols + c)]);
    }
  }
  return out;
}

// The `__metadata__`-less shipped encoder's geometry, reduced. `heads`/`head_dim`
// and their `global_*` twins are separate on purpose: the shipped tower's full
// layers are 512-wide with ONE kv head against the sliding layers' 256 and 8, and
// a fixture with one uniform geometry would never bind that split.
struct ReducedTextEncoderOptions {
  int64_t hidden = 64;
  int64_t layers = 3;
  int64_t heads = 4;
  int64_t head_dim = 16;         // the SLIDING layers
  int64_t kv_heads = 2;          // the SLIDING layers
  int64_t global_head_dim = 32;  // the FULL layers
  int64_t global_kv_heads = 1;   // the FULL layers
  int64_t intermediate = 128;
  int64_t vocab = 16;  // >= every id ReducedTokenizerJson can emit
  // Seeds the tower's own parameter stream. A second value writes a DIFFERENT
  // tower into an otherwise identically shaped file, which is how a test proves
  // the render actually READS these weights.
  std::string tag = "a";
  // The shipped `vonkaiser` build carries NO `__metadata__`, so this defaults
  // false and the Gemma config has to come through the engine's own extra —
  // exactly the shape the real checkpoint puts a caller in.
  bool declare_gemma_config = false;
};

// The Gemma-4 config, in the OFFICIAL bf16 encoder's `__metadata__["gemma_config"]`
// shape (tests/vllm/models/ltx2_gemma4_text_config.json), reduced. `layer_types`
// carries one `full_attention` so the mixed geometry above is actually bound.
inline nlohmann::json ReducedGemmaConfig(const ReducedTextEncoderOptions& o) {
  nlohmann::json text;
  text["model_type"] = "gemma4_unified_text";
  text["hidden_size"] = o.hidden;
  text["num_hidden_layers"] = o.layers;
  text["num_attention_heads"] = o.heads;
  text["num_key_value_heads"] = o.kv_heads;
  text["head_dim"] = o.head_dim;
  text["global_head_dim"] = o.global_head_dim;
  text["num_global_key_value_heads"] = o.global_kv_heads;
  text["intermediate_size"] = o.intermediate;
  text["vocab_size"] = o.vocab;
  text["rms_norm_eps"] = 1e-6;
  text["sliding_window"] = 8;
  text["hidden_size_per_layer_input"] = 0;
  text["num_kv_shared_layers"] = 0;
  // The shipped tower's full layers alias V onto K and ship no `v_proj`
  // (ltx2_text_encoder.h:485-486). Declared here AND expressed in the tensor set
  // below, because the loader cross-checks the two against each other.
  text["attention_k_eq_v"] = true;
  text["tie_word_embeddings"] = true;
  std::vector<std::string> types;
  for (int64_t l = 0; l < o.layers; ++l) {
    types.push_back(l + 1 == o.layers ? "full_attention" : "sliding_attention");
  }
  text["layer_types"] = types;
  nlohmann::json rope;
  rope["sliding_attention"] = {{"rope_type", "default"}, {"rope_theta", 10000.0}};
  rope["full_attention"] = {
      {"rope_type", "default"}, {"rope_theta", 1000000.0}, {"partial_rotary_factor", 0.25}};
  text["rope_parameters"] = rope;

  nlohmann::json doc;
  doc["model_type"] = "gemma4_unified";
  doc["architectures"] = std::vector<std::string>{"Gemma4UnifiedForConditionalGeneration"};
  doc["gemma_version"] = "gemma4-ltx-fixture";
  doc["tie_word_embeddings"] = true;
  doc["text_config"] = text;
  return doc;
}

// A tiny SentencePiece-flavoured `tokenizer.json` in the SHIPPED tokenizer's own
// form — `Replace(" " -> U+2581)` normalizer, `Split(" ", MergedWithPrevious)`
// pre-tokenizer, an EMPTY `post_processor.special_tokens` map (measured on the
// shipped file: it adds nothing, which is why `ltx_core` prepends BOS itself).
// Written independently of the one in test_ltx2_text_encoder.cpp rather than
// shared: it is the INPUT to a different gate, and this file's header records
// why two independent statements beat one helper called twice.
inline std::string ReducedTokenizerJson() {
  return R"JSON({
    "version": "1.0",
    "added_tokens": [
      {"id": 0, "content": "<pad>", "special": true},
      {"id": 1, "content": "<eos>", "special": true},
      {"id": 2, "content": "<bos>", "special": true}
    ],
    "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
    "pre_tokenizer": {"type": "Split", "pattern": {"String": " "},
                      "behavior": "MergedWithPrevious", "invert": false},
    "post_processor": {"type": "TemplateProcessing",
                       "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                       "special_tokens": {}},
    "decoder": {"type": "Sequence", "decoders": [
      {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
      {"type": "ByteFallback"}, {"type": "Fuse"}]},
    "model": {"type": "BPE", "byte_fallback": true,
              "vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2,
                        "▁a": 3, "▁b": 4, "▁c": 5,
                        "a": 6, "b": 7, "c": 8, "▁": 9,
                        "▁ab": 10, "ab": 11},
              "merges": ["▁ a", "▁ b", "▁ c", "a b", "▁a b"]}
  })JSON";
}

// The reduced text encoder. `dit` supplies the two projection WIDTHS, because
// `Ltx2SelectTextFeatureVariant` resolves them from the DiT's own
// `num_attention_heads * attention_head_dim` (encoder_configurator.py:203-209) —
// so a fixture that picked its own would write a file the engine then refuses.
inline void WriteReducedTextEncoder(const vllm::Ltx2DitParams& dit, const std::string& path,
                                    const ReducedTextEncoderOptions& o =
                                        ReducedTextEncoderOptions{}) {
  std::vector<Entry> entries;
  const std::string seed = "ltx2.te." + o.tag + ".";
  auto bf16 = [&](const std::string& name, const std::vector<int64_t>& shape, double scale,
                  double offset = 0.0) {
    int64_t numel = 1;
    for (const int64_t d : shape) numel *= d;
    entries.push_back({name, "BF16", shape, Param(seed + name, numel, scale, offset), {}});
  };

  // 1. THE TOWER, bf16.
  bf16("model.embed_tokens.weight", {o.vocab, o.hidden}, 0.1);
  bf16("model.norm.weight", {o.hidden}, 0.1, 1.0);
  for (int64_t l = 0; l < o.layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    const bool full = l + 1 == o.layers;
    const int64_t dh = full ? o.global_head_dim : o.head_dim;
    const int64_t kv = full ? o.global_kv_heads : o.kv_heads;
    bf16(b + "self_attn.q_proj.weight", {o.heads * dh, o.hidden}, 0.1);
    bf16(b + "self_attn.k_proj.weight", {kv * dh, o.hidden}, 0.1);
    // The FULL layer ships NO `v_proj` — that is what `attention_k_eq_v` MEANS,
    // and the loader refuses an absent one that the config does not declare.
    if (!full) bf16(b + "self_attn.v_proj.weight", {kv * dh, o.hidden}, 0.1);
    bf16(b + "self_attn.o_proj.weight", {o.hidden, o.heads * dh}, 0.1);
    bf16(b + "self_attn.q_norm.weight", {dh}, 0.1, 1.0);
    bf16(b + "self_attn.k_norm.weight", {dh}, 0.1, 1.0);
    bf16(b + "mlp.gate_proj.weight", {o.intermediate, o.hidden}, 0.1);
    bf16(b + "mlp.up_proj.weight", {o.intermediate, o.hidden}, 0.1);
    bf16(b + "mlp.down_proj.weight", {o.hidden, o.intermediate}, 0.1);
    bf16(b + "input_layernorm.weight", {o.hidden}, 0.1, 1.0);
    bf16(b + "post_attention_layernorm.weight", {o.hidden}, 0.1, 1.0);
    bf16(b + "pre_feedforward_layernorm.weight", {o.hidden}, 0.1, 1.0);
    bf16(b + "post_feedforward_layernorm.weight", {o.hidden}, 0.1, 1.0);
    bf16(b + "layer_scalar", {1}, 0.1, 1.0);
  }

  // 2. THE TWO CAPTION PROJECTIONS, torchao-NVFP4, at the DiT's own widths.
  const int64_t in_features = o.hidden * (o.layers + 1);
  const int64_t groups = in_features / 16;
  const char* marker =
      R"({"format": "torchao_nvfp4", "block_size": 16, "scope": "full", )"
      R"("config": "NVFP4DynamicActivationNVFP4WeightConfig", "is_swizzled_scales": true, )"
      R"("use_triton_kernel": true, "use_dynamic_activation": true, )"
      R"("use_dynamic_per_tensor_scale": true})";
  struct Projection {
    const char* module;
    int64_t out;
  };
  const Projection projections[] = {
      {"text_embedding_projection.video_aggregate_embed", dit.cross_attention_dim},
      {"text_embedding_projection.audio_aggregate_embed", dit.audio_cross_attention_dim},
  };
  for (const Projection& p : projections) {
    const std::string m = p.module;
    entries.push_back({m + ".weight",
                       "U8",
                       {p.out, in_features / 2},
                       {},
                       Nvfp4PackedBytes(seed + m + ".w",
                                        static_cast<size_t>(p.out * in_features / 2))});
    entries.push_back(
        {m + ".weight_scale",
         "F8_E4M3",
         {((p.out + 127) / 128) * 128 / 4, ((groups + 3) / 4) * 4 * 4},
         {},
         SwizzleBlockScalePadded(
             Nvfp4LinearScale(seed + m + ".s", static_cast<size_t>(p.out * groups)), p.out,
             groups)});
    const float scale_2 = 0.0078125F;  // 2^-7 — see Nvfp4LinearScale on magnitude
    entries.push_back({m + ".weight_scale_2", "F32", {}, {scale_2}, {}});
    entries.push_back({m + ".torchao_nvfp4",
                       "U8",
                       {static_cast<int64_t>(std::strlen(marker))},
                       {},
                       std::string(marker)});
    // V2 declares `aggregate_bias = true`, and the bias is BF16 on its own
    // unpack path — the exact split `RequireDeclaredProjection` refuses a loader
    // for getting half right.
    bf16(m + ".bias", {p.out}, 0.05);
  }

  // 3. THE ASSET PACK, stored as tensors. `tokenizer_config.json` and
  //    `processor_config.json` are REQUIRED (gemma_assets.py:38-41);
  //    `generation_config.json` is not, and is what states the two ids.
  auto raw = [&](const std::string& name, const std::string& bytes) {
    entries.push_back(
        {name, "U8", {static_cast<int64_t>(bytes.size())}, {}, bytes});
  };
  raw("tokenizer_json", ReducedTokenizerJson());
  raw("hf_asset__tokenizer_config.json", R"({"tokenizer_class":"PreTrainedTokenizerFast"})");
  raw("hf_asset__processor_config.json", R"({"processor_class":"Gemma4Processor"})");
  raw("hf_asset__generation_config.json", R"({"bos_token_id":2,"pad_token_id":0})");

  // 4. `__metadata__`, or the shipped file's LACK of one.
  std::string metadata;
  if (o.declare_gemma_config) {
    nlohmann::json md;
    md["gemma_config"] = ReducedGemmaConfig(o).dump();
    metadata = md.dump();
  }
  WriteSafetensors(entries, metadata, path);
}

// The Gemma config as a standalone JSON FILE, which is what the engine's
// `encoder_config_path` extra reads for a checkpoint that declares none.
inline void WriteGemmaConfigJson(const std::string& path,
                                 const ReducedTextEncoderOptions& o =
                                     ReducedTextEncoderOptions{}) {
  WriteFileBytes(path, ReducedGemmaConfig(o).dump());
}

// ── prompt embeds ──────────────────────────────────────────────────────────
inline void WritePromptEmbeds(const std::string& path, const std::string& tag, int64_t rows,
                              int64_t width) {
  const std::vector<float> values = Param(tag, rows * width, 0.2);
  std::string bytes(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
  WriteFileBytes(path, bytes);
}

// The whole set, as an engine would be pointed at it.
struct Paths {
  std::string dit, video_vae, audio_vae, upsampler, video_embeds, audio_embeds;
  // Phase L13: the text tower, and the Gemma config the shipped encoder does not
  // carry. Written by every fixture; POINTING the engine at them is opt-in,
  // because a load that materializes a tower is not what most cases here gate.
  std::string encoder, encoder_config;
};

// `prompt_tokens` defaults to 4, not 3: the connector's register table is TILED
// across the sequence (embeddings_connector.py:144), so the row count must be a
// multiple of `ReducedConnectorOptions::num_learnable_registers` (2), and 4 makes
// the tiling repeat twice rather than once — a single repeat cannot separate a
// tiled table from an indexed one.
inline Paths WriteFixture(const std::string& dir, int64_t prompt_tokens = 4) {
  ::mkdir(dir.c_str(), 0755);
  const vllm::Ltx2DitParams dit = ReducedDitParams();
  Paths p;
  p.dit = dir + "/dit.safetensors";
  p.video_vae = dir + "/video_vae.safetensors";
  p.audio_vae = dir + "/audio_vae.safetensors";
  p.upsampler = dir + "/upsampler.safetensors";
  p.video_embeds = dir + "/video_prompt_embeds.f32";
  p.audio_embeds = dir + "/audio_prompt_embeds.f32";
  p.encoder = dir + "/text_encoder.safetensors";
  p.encoder_config = dir + "/gemma_config.json";
  WriteReducedTextEncoder(dit, p.encoder);
  WriteGemmaConfigJson(p.encoder_config);
  WriteReducedDit(dit, p.dit);
  WriteReducedVideoVae(ReducedVideoDecoderConfig(dit.in_channels), p.video_vae);
  WriteReducedAudioVae(ReducedAudioDecoderConfig(), ReducedVocoderBweConfig(), p.audio_vae);
  WriteReducedUpsampler(ReducedUpsamplerConfig(dit.in_channels), p.upsampler);
  WritePromptEmbeds(p.video_embeds, "ltx2.embeds.video", prompt_tokens, dit.cross_attention_dim);
  WritePromptEmbeds(p.audio_embeds, "ltx2.embeds.audio", prompt_tokens,
                    dit.audio_cross_attention_dim);
  return p;
}

}  // namespace ltx2_fixture
