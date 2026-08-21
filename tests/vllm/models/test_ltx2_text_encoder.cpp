// LTX-2.5 TEXT CONDITIONING parity gate — the Gemma-4 multi-layer feature
// aggregation, both normalization variants, the two caption projections, the
// encoder -> conditioning hand-off and the embedded asset pack, each compared
// against the UPSTREAM `ltx_core` module executed at reduced dimensions on CPU by
// scripts/gen-ltx2-text-goldens.py.
//
// Both sides rebuild every weight and input from ONE deterministic stream, so no
// weight byte is checked in, and each extractor also asserts its PARAMETER
// MANIFEST — name and shape, in named_parameters() order — against upstream's, so
// a parameter one side builds and the other does not is a failure rather than a
// silent no-op.
//
// The traps this file exists to catch, each of which yields a finite,
// correctly-shaped, WRONG conditioning vector rather than an error:
//
//   * the layer axis concatenated layer-major instead of hidden-major;
//   * the wrong normalization variant (both are "a normalization" and both zero
//     the pads, so only the VALUES differ);
//   * a mask reduction over the wrong axes, or a padding side assumed;
//   * V1's bias-free projection given a bias, or V2's bias dropped.
//
// No doctest::Approx appears here: every tensor comparison is an explicit max
// absolute difference against a stated bound. Approx's default scale of 1.0 puts
// a 1.19e-5 ABSOLUTE floor under any epsilon, which would let a tight tolerance
// silently accept anything (including zero).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "support/max_abs_diff.h"
#include "support/process_id.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/model_executor/models/ltx2_text_encoder.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

#include "ltx2_text_goldens.inc"
#include "ltx2_gemma_tower_goldens.inc"
#include "ltx2_prompt_tokens_goldens.inc"

namespace fs = std::filesystem;

using vllm::Ltx2TextFeatureConfig;
using vllm::Ltx2TextHiddenStates;
using vllm::Ltx2TextNormVariant;

namespace {

// ---------------------------------------------------------------------------
// Ltx2Rand — the exact mirror of the generator's stream
// (scripts/gen-ltx2-text-goldens.py :: ltx2_rand), identical to the one L2's
// suite uses: a per-tensor FNV-1a seed plus a splitmix64 counter, so both sides
// build identical tensors from a NAME alone and cannot drift by reordering their
// parameter construction.
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

std::vector<float> Ltx2Make(const std::string& name, int64_t count, double scale,
                            double offset) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const uint64_t u = Splitmix64(seed + static_cast<uint64_t>(i));
    const double unit = (static_cast<double>(u >> 11) * 0x1p-53) * 2.0 - 1.0;
    out[static_cast<size_t>(i)] = static_cast<float>(unit * scale + offset);
  }
  return out;
}

bool EndsWith(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The generator's `param_spec` rule, mirrored EXACTLY (and it is L2's rule
// verbatim, so the three LTX-2.5 suites share one weight stream).
std::vector<float> Ltx2Param(const std::string& name,
                             const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) count *= dim;
  double scale = 0.05;
  double offset = 0.0;
  if (EndsWith(name, "q_norm.weight") || EndsWith(name, "k_norm.weight")) {
    scale = 0.1;
    offset = 1.0;
  } else if (EndsWith(name, ".bias")) {
    scale = 0.02;
  }
  return Ltx2Make(name, count, scale, offset);
}

// ---------------------------------------------------------------------------
// The reduced fixture the generator built.
// ---------------------------------------------------------------------------

constexpr int64_t kBatch = vllm_test::kLtxTeBatch;
constexpr int64_t kSeq = vllm_test::kLtxTeSeq;
constexpr int64_t kHidden = vllm_test::kLtxTeGemmaHidden;
constexpr int64_t kLayers = vllm_test::kLtxTeNumLayers;

std::vector<std::vector<float>> HiddenStateBuffers() {
  std::vector<std::vector<float>> states;
  for (int64_t l = 0; l < kLayers; ++l) {
    states.push_back(Ltx2Make("input.hidden." + std::to_string(l),
                              kBatch * kSeq * kHidden, 0.5, 0.0));
  }
  return states;
}

Ltx2TextHiddenStates MakeStates(const std::vector<std::vector<float>>& buffers) {
  Ltx2TextHiddenStates states;
  for (const std::vector<float>& b : buffers) states.layers.push_back(b.data());
  states.batch = kBatch;
  states.seq = kSeq;
  states.hidden = kHidden;
  return states;
}

// The masks the generator emits, as int32 (our mask dtype).
std::vector<int32_t> MaskFrom(const int64_t* golden) {
  std::vector<int32_t> mask(static_cast<size_t>(kBatch * kSeq));
  for (size_t i = 0; i < mask.size(); ++i) mask[i] = static_cast<int32_t>(golden[i]);
  return mask;
}

// The shared, NaN-hardened reduction. The local copy this replaces used
// `std::max(worst, ...)`, which is `a < b ? b : a`; `a < NaN` is false, so an
// all-NaN result against a correct golden reduced to 0.0 (issue #449).
using vllm_test::MaxAbsDiff;

// `additive_mask` holds -FLT_MAX, whose absolute difference saturates any bound;
// compare it EXACTLY instead, which is also what upstream produces (one multiply
// by finfo.max, no accumulation).
void CheckExact(const std::vector<float>& got, const float* want, size_t count) {
  REQUIRE(got.size() == count);
  for (size_t i = 0; i < count; ++i) CHECK(got[i] == want[i]);
}

void CheckExactI(const std::vector<int32_t>& got, const int64_t* want, size_t count) {
  REQUIRE(got.size() == count);
  for (size_t i = 0; i < count; ++i) CHECK(static_cast<int64_t>(got[i]) == want[i]);
}

// The tolerance every f32 brick is held to. The generator runs upstream in torch
// float32 and every value we compare is an f32 store, so anything above this is an
// algorithm difference, not round-off. (The reductions accumulate in f64 and round
// once, which is stated and justified where they are written; the SCALARS are f32
// because upstream's are.)
constexpr double kTol = 1e-5;

Ltx2TextFeatureConfig V2Config() {
  Ltx2TextFeatureConfig cfg;
  cfg.variant = Ltx2TextNormVariant::kPerTokenRmsV2;
  cfg.embedding_dim = kHidden;
  cfg.num_layers = kLayers;
  cfg.video_out_features = vllm_test::kLtxTeVideoInner;
  cfg.audio_out_features = vllm_test::kLtxTeAudioInner;
  cfg.aggregate_bias = true;
  cfg.is_av = false;
  return cfg;
}

Ltx2TextFeatureConfig V1Config() {
  Ltx2TextFeatureConfig cfg;
  cfg.variant = Ltx2TextNormVariant::kPaddedBatchV1;
  cfg.embedding_dim = kHidden;
  cfg.num_layers = kLayers;
  cfg.video_out_features = kHidden;  // V1 projects back to the Gemma width
  cfg.audio_out_features = 0;
  cfg.aggregate_bias = false;
  cfg.is_av = true;
  return cfg;
}

vllm::Ltx2TextEncoderWeights V2Weights() {
  vllm::Ltx2TextEncoderWeights w;
  const int64_t flat = vllm_test::kLtxTeFlatDim;
  w.video.out_features = vllm_test::kLtxTeVideoInner;
  w.video.in_features = flat;
  w.video.weight = Ltx2Param("video_aggregate_embed.weight", {w.video.out_features, flat});
  w.video.bias = Ltx2Param("video_aggregate_embed.bias", {w.video.out_features});
  w.audio.out_features = vllm_test::kLtxTeAudioInner;
  w.audio.in_features = flat;
  w.audio.weight = Ltx2Param("audio_aggregate_embed.weight", {w.audio.out_features, flat});
  w.audio.bias = Ltx2Param("audio_aggregate_embed.bias", {w.audio.out_features});
  return w;
}

vllm::Ltx2TextEncoderWeights V1Weights() {
  vllm::Ltx2TextEncoderWeights w;
  const int64_t flat = vllm_test::kLtxTeFlatDim;
  w.video.out_features = kHidden;
  w.video.in_features = flat;
  w.video.weight = Ltx2Param("aggregate_embed.weight", {kHidden, flat});
  // bias stays EMPTY: encoder_configurator.py:187 builds V1's Linear with bias=False.
  return w;
}

nlohmann::json V2TransformerConfig() {
  return nlohmann::json{
      {"caption_proj_before_connector", true},
      {"caption_projection_first_linear", false},
      {"caption_proj_input_norm", false},
      {"caption_projection_second_linear", false},
      {"num_attention_heads", vllm_test::kLtxTeVideoHeads},
      {"attention_head_dim", vllm_test::kLtxTeVideoHeadDim},
      {"audio_num_attention_heads", vllm_test::kLtxTeAudioHeads},
      {"audio_attention_head_dim", vllm_test::kLtxTeAudioHeadDim},
  };
}

nlohmann::json V1TransformerConfig() {
  return nlohmann::json{
      {"num_attention_heads", vllm_test::kLtxTeVideoHeads},
      {"attention_head_dim", vllm_test::kLtxTeVideoHeadDim},
  };
}

// --- a synthetic single-file text-encoder pack, for the asset gate -----------

void AppendU64(std::string& out, uint64_t value) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

// One tensor in a synthetic pack, at its real dtype and shape.
struct PackTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string bytes;
};

// Writes a minimal .safetensors. `metadata` is written verbatim as the
// `__metadata__` object when non-empty.
std::string WriteTypedPack(const fs::path& path, const std::vector<PackTensor>& tensors,
                           const std::string& metadata_json) {
  nlohmann::json header = nlohmann::json::object();
  if (!metadata_json.empty()) header["__metadata__"] = nlohmann::json::parse(metadata_json);
  uint64_t offset = 0;
  std::string payload;
  for (const PackTensor& t : tensors) {
    header[t.name] = nlohmann::json{
        {"dtype", t.dtype},
        {"shape", t.shape},
        {"data_offsets",
         nlohmann::json::array({offset, offset + static_cast<uint64_t>(t.bytes.size())})}};
    offset += t.bytes.size();
    payload += t.bytes;
  }
  const std::string head = header.dump();
  std::string out;
  AppendU64(out, head.size());
  out += head;
  out += payload;
  std::ofstream file(path, std::ios::binary);
  file.write(out.data(), static_cast<std::streamsize>(out.size()));
  file.close();
  return path.string();
}

// The U8-asset form, in the shape gemma_assets.py:335-386 packs them.
std::string WritePack(const fs::path& path,
                      const std::vector<std::pair<std::string, std::string>>& tensors,
                      const std::string& metadata_json) {
  std::vector<PackTensor> typed;
  typed.reserve(tensors.size());
  for (const auto& [name, bytes] : tensors)
    typed.push_back(
        PackTensor{name, "U8", {static_cast<int64_t>(bytes.size())}, bytes});
  return WriteTypedPack(path, typed, metadata_json);
}

std::string Bf16Bytes(const std::vector<float>& values) {
  std::string out(values.size() * sizeof(uint16_t), '\0');
  auto* p = reinterpret_cast<uint16_t*>(out.data());
  for (size_t i = 0; i < values.size(); ++i) p[i] = vt::F32ToBF16(values[i]);
  return out;
}

std::string BytesToString(const std::vector<uint8_t>& bytes) {
  return std::string(bytes.begin(), bytes.end());
}

// --- the Gemma-4 per-layer hidden-state seam --------------------------------
//
// LTX-2.5 needs EVERY hidden state, so `Gemma4Model::ForwardHiddenStates` had to
// exist. A tiny CPU-synthetic Gemma-4 gates the ORDER of what it returns, which
// is the part that fails silently: 49 finite tensors of the right shape carrying
// the wrong states condition on the wrong thing and still render.
//
// The reduced config deliberately mirrors the SHIPPED LTX text tower's shape
// rather than unsloth's E4B: `model.layers.N.{input,post_attention,pre_feedforward,
// post_feedforward}_layernorm` + `layer_scalar`, uniform head_dim, GQA, and NO
// per-layer-embedding tensors at all — measured on
// gemma4-12b-with-proj-nvfp4-torchao.safetensors, whose 1688 tensors contain no
// `embed_tokens_per_layer` and no `per_layer_*`.

vt::Queue Qcpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

vllm::OwnedTensor MakeBf16(const std::vector<int64_t>& shape, bool nk, uint32_t seed,
                           float scale = 0.08f) {
  vllm::OwnedTensor o;
  o.dtype = vt::DType::kBF16;
  o.nk = nk;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  // The deterministic stream again, so the fixture is reproducible from a seed.
  for (int64_t i = 0; i < numel; ++i) {
    const uint64_t u = Splitmix64(static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ULL +
                                  static_cast<uint64_t>(i));
    const double unit = (static_cast<double>(u >> 11) * 0x1p-53) * 2.0 - 1.0;
    p[i] = vt::F32ToBF16(static_cast<float>(unit * scale));
  }
  return o;
}

vllm::HfConfig TinyGemma4Config() {
  vllm::HfConfig c;
  c.num_hidden_layers = 3;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.intermediate_size = 128;
  c.rms_norm_eps = 1e-6;
  c.vocab_size = 96;
  c.sliding_window = 8;
  // No hidden_size_per_layer_input -> ple_dim 0, no global_head_dim -> one head
  // width, no final_logit_softcapping -> the lm_head product is a plain matmul,
  // which is what lets the last-state invariant below be checked directly.
  c.raw = nlohmann::json{{"tie_word_embeddings", true}};
  return c;
}

vllm::Gemma4Weights TinyGemma4Weights(const vllm::HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads;
  const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
  const int64_t I = c.intermediate_size, V = c.vocab_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  vllm::Gemma4Weights w;
  w.tie_word_embeddings = true;
  w.embed_tokens = MakeBf16({V, H}, /*nk=*/false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.3f);
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    vllm::Gemma4LayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.pre_feedforward_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.post_feedforward_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.layer_scalar = MakeBf16({1}, false, seed++, 0.5f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, /*nk=*/true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, /*nk=*/true, seed++);
    lw.attn.q_norm = MakeBf16({Dh}, false, seed++, 0.3f);
    lw.attn.k_norm = MakeBf16({Dh}, false, seed++, 0.3f);
    lw.mlp.gate_up_proj = MakeBf16({2 * I, H}, /*nk=*/true, seed++);
    lw.mlp.down_proj = MakeBf16({H, I}, /*nk=*/true, seed++);
    lw.head_dim = Dh;
    lw.num_kv_heads = Hkv;
    lw.is_full_attention = false;
    lw.is_kv_shared = false;
    lw.kv_target_layer = -1;
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// ---------------------------------------------------------------------------
// The Gemma-4 TOWER fixture — the mirror of
// scripts/gen-ltx2-gemma-tower-goldens.py.
//
// Every tensor is rebuilt from the deterministic stream keyed by the same
// HuggingFace parameter NAME the oracle filled, so this side and the oracle
// cannot drift by reordering, and a port that reads the wrong tensor name reads
// different numbers rather than the same numbers in the wrong slot.
// ---------------------------------------------------------------------------

std::string TowerParam(int64_t layer, const char* suffix) {
  return "language_model.layers." + std::to_string(layer) + "." + suffix;
}

// The generator's `gemma_param_spec`, verbatim. The OFFSETS are the load-bearing
// part: Gemma-4's RMSNorm multiplies by `weight` directly rather than by
// `1 + weight` (modeling_gemma4_unified.py:181-185) and initializes it to ones,
// and `layer_scalar` is a buffer initialized to ones and applied as
// `hidden_states *= layer_scalar` (:501, :535). Centring either on 0 would make a
// port that ignores it pass.
std::vector<float> TowerRand(const std::string& name, int64_t count) {
  double scale = 0.05;
  double offset = 0.0;
  if (EndsWith(name, "layer_scalar") || EndsWith(name, "_layernorm.weight") ||
      EndsWith(name, "norm.weight")) {
    scale = 0.1;
    offset = 1.0;
  } else if (EndsWith(name, ".bias")) {
    scale = 0.02;
  }
  return Ltx2Make(name, count, scale, offset);
}

vllm::OwnedTensor TowerTensor(const std::string& name,
                              const std::vector<int64_t>& shape, bool nk) {
  int64_t numel = 1;
  for (int64_t d : shape) numel *= d;
  const std::vector<float> values = TowerRand(name, numel);
  vllm::OwnedTensor o;
  o.dtype = vt::DType::kBF16;
  o.nk = nk;
  o.rank = static_cast<int>(shape.size());
  for (int i = 0; i < o.rank; ++i) o.shape[i] = shape[static_cast<size_t>(i)];
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t i = 0; i < numel; ++i)
    p[i] = vt::F32ToBF16(values[static_cast<size_t>(i)]);
  return o;
}

// Concatenate several named [rows_i, H] tensors into one raw-NK block, which is
// how `Gemma4Weights` carries q/k/v and gate/up (gemma4_weights.cpp:290-296).
vllm::OwnedTensor TowerConcatNk(
    const std::vector<std::pair<std::string, int64_t>>& parts, int64_t H) {
  int64_t rows = 0;
  for (const auto& part : parts) rows += part.second;
  vllm::OwnedTensor o;
  o.dtype = vt::DType::kBF16;
  o.nk = true;
  o.rank = 2;
  o.shape[0] = rows;
  o.shape[1] = H;
  o.bytes.resize(static_cast<size_t>(rows * H) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  int64_t at = 0;
  for (const auto& part : parts) {
    const std::vector<float> values = TowerRand(part.first, part.second * H);
    for (int64_t i = 0; i < part.second * H; ++i)
      p[at + i] = vt::F32ToBF16(values[static_cast<size_t>(i)]);
    at += part.second * H;
  }
  return o;
}

vllm::HfConfig TowerConfig() {
  // PARSED from the very JSON the oracle was configured with, rather than
  // reconstructed field by field. `attention_k_eq_v`, `final_logit_softcapping`,
  // both `rope_parameters` entries and `rms_norm_eps` all reach the forward
  // through `cfg.raw`, and a hand-built config would silently disagree with the
  // run that produced the goldens about any field nobody remembered.
  return vllm::ParseHfConfig(nlohmann::json::parse(vllm_test::kLtxTowerTextConfigJson),
                             "ltx2_gemma_tower_goldens.inc");
}

vllm::Gemma4Weights TowerWeights(const vllm::HfConfig& c) {
  const int64_t H = c.hidden_size;
  const int64_t I = c.intermediate_size, V = c.vocab_size;
  const int64_t Hq = vllm_test::kLtxTowerNumHeads;
  vllm::Gemma4Weights w;
  w.tie_word_embeddings = true;
  w.embed_tokens = TowerTensor("language_model.embed_tokens.weight", {V, H}, false);
  w.final_norm = TowerTensor("language_model.norm.weight", {H}, false);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const bool full = vllm_test::kLtxTowerLayerIsFull[static_cast<size_t>(l)] != 0;
    // The two geometries the shipped tower actually mixes.
    const int64_t Dh = full ? vllm_test::kLtxTowerGlobalHeadDim
                            : vllm_test::kLtxTowerHeadDim;
    const int64_t Hkv = full ? vllm_test::kLtxTowerNumGlobalKvHeads
                             : vllm_test::kLtxTowerNumKvHeads;
    vllm::Gemma4LayerWeights lw;
    lw.is_full_attention = full;
    lw.is_kv_shared = false;
    lw.kv_target_layer = -1;
    lw.head_dim = Dh;
    lw.num_kv_heads = Hkv;
    // `attention_k_eq_v` is true and the FULL layers are where it bites: they
    // ship no `v_proj` at all, so V aliases K. Duplicating the K rows is exactly
    // what gemma4_weights.cpp:289-292 does for that case.
    lw.k_eq_v = full;
    const std::string q = TowerParam(l, "self_attn.q_proj.weight");
    const std::string k = TowerParam(l, "self_attn.k_proj.weight");
    const std::string v = TowerParam(l, "self_attn.v_proj.weight");
    lw.attn.qkv_proj =
        full ? TowerConcatNk({{q, Hq * Dh}, {k, Hkv * Dh}, {k, Hkv * Dh}}, H)
             : TowerConcatNk({{q, Hq * Dh}, {k, Hkv * Dh}, {v, Hkv * Dh}}, H);
    lw.attn.o_proj =
        TowerTensor(TowerParam(l, "self_attn.o_proj.weight"), {H, Hq * Dh}, true);
    lw.attn.q_norm = TowerTensor(TowerParam(l, "self_attn.q_norm.weight"), {Dh}, false);
    lw.attn.k_norm = TowerTensor(TowerParam(l, "self_attn.k_norm.weight"), {Dh}, false);
    lw.mlp.gate_up_proj = TowerConcatNk({{TowerParam(l, "mlp.gate_proj.weight"), I},
                                         {TowerParam(l, "mlp.up_proj.weight"), I}},
                                        H);
    lw.mlp.down_proj = TowerTensor(TowerParam(l, "mlp.down_proj.weight"), {H, I}, true);
    lw.input_layernorm =
        TowerTensor(TowerParam(l, "input_layernorm.weight"), {H}, false);
    lw.post_attention_layernorm =
        TowerTensor(TowerParam(l, "post_attention_layernorm.weight"), {H}, false);
    lw.pre_feedforward_layernorm =
        TowerTensor(TowerParam(l, "pre_feedforward_layernorm.weight"), {H}, false);
    lw.post_feedforward_layernorm =
        TowerTensor(TowerParam(l, "post_feedforward_layernorm.weight"), {H}, false);
    lw.layer_scalar = TowerTensor(TowerParam(l, "layer_scalar"), {1}, false);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// A KV pool whose layers do NOT share one geometry. The shipped tower's sliding
// layers are 8 kv heads x 256 and its full layers 1 x 512, and the runner's
// single-uniform-head_dim allocation (gemma4.h, G1 HONEST STATUS) is exactly
// what cannot express that — so the LTX text path builds its own, and so does
// this gate.
struct TowerCachePool {
  std::vector<std::vector<float>> buf;
  std::vector<vllm::PagedKvCache> attn_kv;
  TowerCachePool(const vllm::HfConfig& c, int64_t num_blocks, int64_t block_size) {
    buf.reserve(static_cast<size_t>(c.num_hidden_layers));
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      const bool full = vllm_test::kLtxTowerLayerIsFull[static_cast<size_t>(l)] != 0;
      const int64_t Dh = full ? vllm_test::kLtxTowerGlobalHeadDim
                              : vllm_test::kLtxTowerHeadDim;
      const int64_t Hkv = full ? vllm_test::kLtxTowerNumGlobalKvHeads
                               : vllm_test::kLtxTowerNumKvHeads;
      buf.emplace_back(
          static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0.0f);
      vllm::PagedKvCache kv;
      kv.data = buf.back().data();
      kv.dtype = vt::DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
  }
};

const float* TowerGoldenF32(int64_t state) {
  static const float* const kStates[] = {
      vllm_test::kLtxTowerStateF32_0,  vllm_test::kLtxTowerStateF32_1,
      vllm_test::kLtxTowerStateF32_2,  vllm_test::kLtxTowerStateF32_3,
      vllm_test::kLtxTowerStateF32_4,  vllm_test::kLtxTowerStateF32_5,
      vllm_test::kLtxTowerStateF32_6,  vllm_test::kLtxTowerStateF32_7,
      vllm_test::kLtxTowerStateF32_8,  vllm_test::kLtxTowerStateF32_9,
      vllm_test::kLtxTowerStateF32_10, vllm_test::kLtxTowerStateF32_11,
      vllm_test::kLtxTowerStateF32_12,
  };
  return kStates[static_cast<size_t>(state)];
}

const float* TowerGoldenBf16(int64_t state) {
  static const float* const kStates[] = {
      vllm_test::kLtxTowerStateBf16_0,  vllm_test::kLtxTowerStateBf16_1,
      vllm_test::kLtxTowerStateBf16_2,  vllm_test::kLtxTowerStateBf16_3,
      vllm_test::kLtxTowerStateBf16_4,  vllm_test::kLtxTowerStateBf16_5,
      vllm_test::kLtxTowerStateBf16_6,  vllm_test::kLtxTowerStateBf16_7,
      vllm_test::kLtxTowerStateBf16_8,  vllm_test::kLtxTowerStateBf16_9,
      vllm_test::kLtxTowerStateBf16_10, vllm_test::kLtxTowerStateBf16_11,
      vllm_test::kLtxTowerStateBf16_12,
  };
  return kStates[static_cast<size_t>(state)];
}

const float* TowerGoldenPadded(int64_t state) {
  static const float* const kStates[] = {
      vllm_test::kLtxTowerPaddedStateF32_0,  vllm_test::kLtxTowerPaddedStateF32_1,
      vllm_test::kLtxTowerPaddedStateF32_2,  vllm_test::kLtxTowerPaddedStateF32_3,
      vllm_test::kLtxTowerPaddedStateF32_4,  vllm_test::kLtxTowerPaddedStateF32_5,
      vllm_test::kLtxTowerPaddedStateF32_6,  vllm_test::kLtxTowerPaddedStateF32_7,
      vllm_test::kLtxTowerPaddedStateF32_8,  vllm_test::kLtxTowerPaddedStateF32_9,
      vllm_test::kLtxTowerPaddedStateF32_10, vllm_test::kLtxTowerPaddedStateF32_11,
      vllm_test::kLtxTowerPaddedStateF32_12,
  };
  return kStates[static_cast<size_t>(state)];
}

const float* TowerGoldenPaddedBf16(int64_t state) {
  static const float* const kStates[] = {
      vllm_test::kLtxTowerPaddedStateBf16_0,  vllm_test::kLtxTowerPaddedStateBf16_1,
      vllm_test::kLtxTowerPaddedStateBf16_2,  vllm_test::kLtxTowerPaddedStateBf16_3,
      vllm_test::kLtxTowerPaddedStateBf16_4,  vllm_test::kLtxTowerPaddedStateBf16_5,
      vllm_test::kLtxTowerPaddedStateBf16_6,  vllm_test::kLtxTowerPaddedStateBf16_7,
      vllm_test::kLtxTowerPaddedStateBf16_8,  vllm_test::kLtxTowerPaddedStateBf16_9,
      vllm_test::kLtxTowerPaddedStateBf16_10, vllm_test::kLtxTowerPaddedStateBf16_11,
      vllm_test::kLtxTowerPaddedStateBf16_12,
  };
  return kStates[static_cast<size_t>(state)];
}

// ─────────── the reduced tower as a FILE, so the LOADER is CI-reachable ───────
//
// `Ltx2LoadGemmaTowerFromSafetensors` and every refusal it documents were
// reachable from exactly one place — the opt-in real-checkpoint case — so none of
// them had CI coverage. The tower fixture above already carries the weight stream
// the ORACLE ran, keyed by HuggingFace parameter NAME; writing that same stream
// into a safetensors file under the CHECKPOINT's tensor names (`model.*`, which
// is what the loader reads) makes the loader answerable to it. The two must agree
// byte for byte, which is a real gate on the parts that have no shape to check
// them: the q|k|v concat ORDER, the `k_eq_v` aliasing of V onto K on the full
// layers, and the per-layer 8-vs-16 head_dim / 2-vs-1 kv-head split.
//
// SCOPE, because "the concat order is gated" is broader than what this reaches.
// The only concat under this case is the LTX tower loader's own `TowerConcat`
// (ltx2_text_encoder.cpp:951-953, both qkv arms; the helper is at :835).
// `gemma4_weights.cpp` assembles its qkv through a SEPARATE implementation
// (gemma4_weights.cpp:281-295) that nothing in this suite loads — MEASURED:
// mutating it leaves every case in this file green. That path owes its own
// gate; this one does not stand in for it.
std::vector<PackTensor> TowerPackTensors(const vllm::HfConfig& c) {
  const int64_t H = c.hidden_size, I = c.intermediate_size, V = c.vocab_size;
  const int64_t Hq = vllm_test::kLtxTowerNumHeads;
  std::vector<PackTensor> ts;
  // `file` is the CHECKPOINT name the loader reads; `stream` is the HuggingFace
  // parameter name the deterministic weight stream is keyed by. They differ
  // (`model.` vs `language_model.`) and conflating them would make the fixture
  // agree with the loader about bytes nobody generated from the same seed.
  auto add = [&ts](const std::string& file, const std::string& stream,
                   std::vector<int64_t> shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    ts.push_back(PackTensor{file, "BF16", std::move(shape), Bf16Bytes(TowerRand(stream, n))});
  };
  add("model.embed_tokens.weight", "language_model.embed_tokens.weight", {V, H});
  add("model.norm.weight", "language_model.norm.weight", {H});
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const bool full = vllm_test::kLtxTowerLayerIsFull[static_cast<size_t>(l)] != 0;
    const int64_t Dh =
        full ? vllm_test::kLtxTowerGlobalHeadDim : vllm_test::kLtxTowerHeadDim;
    const int64_t Hkv = full ? vllm_test::kLtxTowerNumGlobalKvHeads
                             : vllm_test::kLtxTowerNumKvHeads;
    const std::string b = "model.layers." + std::to_string(l) + ".";
    auto layer = [&](const char* suffix, std::vector<int64_t> shape) {
      add(b + suffix, TowerParam(l, suffix), std::move(shape));
    };
    layer("self_attn.q_proj.weight", {Hq * Dh, H});
    layer("self_attn.k_proj.weight", {Hkv * Dh, H});
    // The full layers ship NO v_proj at all: `attention_k_eq_v` aliases V onto K.
    if (!full) layer("self_attn.v_proj.weight", {Hkv * Dh, H});
    layer("self_attn.o_proj.weight", {H, Hq * Dh});
    layer("self_attn.q_norm.weight", {Dh});
    layer("self_attn.k_norm.weight", {Dh});
    layer("mlp.gate_proj.weight", {I, H});
    layer("mlp.up_proj.weight", {I, H});
    layer("mlp.down_proj.weight", {H, I});
    layer("input_layernorm.weight", {H});
    layer("post_attention_layernorm.weight", {H});
    layer("pre_feedforward_layernorm.weight", {H});
    layer("post_feedforward_layernorm.weight", {H});
    layer("layer_scalar", {1});
  }
  return ts;
}

bool SameTensor(const vllm::OwnedTensor& a, const vllm::OwnedTensor& b) {
  if (a.dtype != b.dtype || a.rank != b.rank || a.nk != b.nk) return false;
  for (int i = 0; i < a.rank; ++i)
    if (a.shape[i] != b.shape[i]) return false;
  return a.bytes == b.bytes;
}

// A tokenizer over the reduced tower's 64-id vocabulary whose ADDED tokens spell
// one id each, so a prompt can be written that tokenizes to EXACTLY the token
// sequence the oracle ran. Without that the conditioning path has no committed
// oracle to be held to and only its shapes can be checked.
std::string TowerTokenizerJson() {
  nlohmann::json added = nlohmann::json::array();
  nlohmann::json vocab = nlohmann::json::object();
  const char* kSpecial[] = {"<pad>", "<eos>", "<bos>"};
  for (int64_t i = 0; i < 3; ++i) {
    vocab[kSpecial[i]] = i;
    added.push_back({{"id", i}, {"content", kSpecial[i]}, {"special", true}});
  }
  for (int64_t i = 3; i < vllm_test::kLtxTowerVocab; ++i) {
    const std::string t = "<t" + std::to_string(i) + ">";
    vocab[t] = i;
    added.push_back({{"id", i}, {"content", t}, {"special", false}});
  }
  // Normalizer / pre-tokenizer / decoder are the SHIPPED tokenizer's own forms,
  // carried over from the wrapper fixture above. They are inert on a prompt made
  // only of added tokens (it has no spaces), which is the point: the prompt's
  // token ids come from the added-token table and not from BPE luck.
  nlohmann::json j;
  j["version"] = "1.0";
  j["added_tokens"] = added;
  j["normalizer"] = {{"type", "Replace"},
                     {"pattern", {{"String", " "}}},
                     {"content", "▁"}};
  j["pre_tokenizer"] = {{"type", "Split"},
                        {"pattern", {{"String", " "}}},
                        {"behavior", "MergedWithPrevious"},
                        {"invert", false}};
  j["post_processor"] = {
      {"type", "TemplateProcessing"},
      {"single", nlohmann::json::array({{{"Sequence", {{"id", "A"}, {"type_id", 0}}}}})},
      {"special_tokens", nlohmann::json::object()}};
  j["decoder"] = {{"type", "Sequence"},
                  {"decoders", nlohmann::json::array(
                                   {{{"type", "Replace"},
                                     {"pattern", {{"String", "▁"}}},
                                     {"content", " "}},
                                    {{"type", "Fuse"}}})}};
  j["model"] = {{"type", "BPE"},
                {"byte_fallback", false},
                {"vocab", vocab},
                {"merges", nlohmann::json::array()}};
  return j.dump();
}

// The caption projections for the reduced tower's 32-wide, 13-state stack. Widths
// are the fixture's own; what matters is that BOTH streams are non-degenerate so
// a conditioning defect cannot hide in an unused one.
vllm::Ltx2TextFeatureConfig TowerFeatureConfig() {
  vllm::Ltx2TextFeatureConfig cfg;
  cfg.variant = vllm::Ltx2TextNormVariant::kPerTokenRmsV2;
  cfg.embedding_dim = vllm_test::kLtxTowerHidden;
  cfg.num_layers = vllm_test::kLtxTowerNumStates;
  cfg.video_out_features = 24;
  cfg.audio_out_features = 12;
  cfg.aggregate_bias = true;
  cfg.is_av = false;
  return cfg;
}

vllm::Ltx2TextEncoderWeights TowerProjections() {
  const int64_t flat = vllm_test::kLtxTowerHidden * vllm_test::kLtxTowerNumStates;
  vllm::Ltx2TextEncoderWeights w;
  w.video.out_features = 24;
  w.video.in_features = flat;
  w.video.weight = Ltx2Param("tower.video_aggregate_embed.weight", {24, flat});
  w.video.bias = Ltx2Param("tower.video_aggregate_embed.bias", {24});
  w.audio.out_features = 12;
  w.audio.in_features = flat;
  w.audio.weight = Ltx2Param("tower.audio_aggregate_embed.weight", {12, flat});
  w.audio.bias = Ltx2Param("tower.audio_aggregate_embed.bias", {12});
  return w;
}

// The ORACLE's left-padded states, laid out exactly as the conditioning path lays
// out its own: full padded width, pad rows ZERO.
std::vector<std::vector<float>> OraclePaddedStates(const float* (*golden)(int64_t)) {
  const int64_t H = vllm_test::kLtxTowerHidden;
  const int64_t PT = vllm_test::kLtxTowerPaddedSeq;
  const int64_t P = vllm_test::kLtxTowerNumPad;
  std::vector<std::vector<float>> out(
      static_cast<size_t>(vllm_test::kLtxTowerNumStates));
  for (int64_t s = 0; s < vllm_test::kLtxTowerNumStates; ++s) {
    std::vector<float>& dst = out[static_cast<size_t>(s)];
    dst.assign(static_cast<size_t>(PT * H), 0.0f);
    const float* src = golden(s);
    for (int64_t t = P; t < PT; ++t)
      for (int64_t h = 0; h < H; ++h)
        dst[static_cast<size_t>(t * H + h)] = src[static_cast<size_t>(t * H + h)];
  }
  return out;
}

struct Gemma4CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<vllm::PagedKvCache> attn_kv;
  Gemma4CachePool(const vllm::HfConfig& c, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l)
      buf.emplace_back(static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0.0f);
    for (auto& b : buf) {
      vllm::PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = vt::DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
  }
};

vllm::v1::CommonAttentionMetadata PrefillMeta(int64_t T, int64_t block_size) {
  vllm::v1::CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t % block_size);
  m.causal = true;
  return m;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ltx2 text: the hidden-state stack is HIDDEN-major, layer-minor") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const std::vector<float> stacked = Ltx2StackHiddenStates(MakeStates(buffers));

  const size_t count = static_cast<size_t>(kBatch * kSeq * kHidden * kLayers);
  const double worst = MaxAbsDiff(stacked, vllm_test::kLtxTeStacked, count);
  MESSAGE("ltx2 text stack max|diff| = " << worst);
  CHECK(worst == 0.0);  // a pure permutation: it must be BIT-equal

  // The trap, made explicit: the layer-major alternative is a different tensor,
  // so a port that concatenates `l * D + d` cannot pass the check above.
  std::vector<float> layer_major(count);
  for (int64_t b = 0; b < kBatch; ++b)
    for (int64_t t = 0; t < kSeq; ++t)
      for (int64_t d = 0; d < kHidden; ++d)
        for (int64_t l = 0; l < kLayers; ++l) {
          const size_t src = static_cast<size_t>(((b * kSeq + t) * kHidden + d) * kLayers + l);
          const size_t dst = static_cast<size_t>(((b * kSeq + t) * kLayers + l) * kHidden + d);
          layer_major[dst] = stacked[src];
        }
  bool differs = false;
  for (size_t i = 0; i < count; ++i)
    if (layer_major[i] != stacked[i]) differs = true;
  CHECK(differs);
}

TEST_CASE("ltx2 text: the hidden-state contract is 48 layers + the embedding output") {
  // encoder_configurator.py:182 and base_encoder.py:68-71 — the shipped Gemma-4
  // has 48 decoder layers and LTX consumes 49 states, which is what makes the
  // caption projections 3840 x 49 = 188160 wide.
  CHECK(vllm::Ltx2GemmaHiddenStateContract::Count(48) == 49);
  CHECK(vllm::Ltx2GemmaHiddenStateContract::Count(48) * 3840 == 188160);
  CHECK(vllm::Ltx2GemmaHiddenStateContract::Count(vllm_test::kLtxTeGemmaHiddenLayers) ==
        kLayers);
}

TEST_CASE("ltx2 text: the normalization variant is SELECTED from config, never guessed") {
  const Ltx2TextFeatureConfig v2 = vllm::Ltx2SelectTextFeatureVariant(
      V2TransformerConfig(), vllm_test::kLtxTeGemmaHidden,
      vllm_test::kLtxTeGemmaHiddenLayers);
  CHECK(static_cast<int>(v2.variant == Ltx2TextNormVariant::kPerTokenRmsV2) ==
        static_cast<int>(vllm_test::kLtxTeSelectedV2IsV2));
  CHECK(v2.embedding_dim == vllm_test::kLtxTeV2EmbeddingDim);
  CHECK(v2.FlatDim() == vllm_test::kLtxTeV2VideoIn);
  CHECK(v2.video_out_features == vllm_test::kLtxTeV2VideoOut);
  CHECK(v2.FlatDim() == vllm_test::kLtxTeV2AudioIn);
  CHECK(v2.audio_out_features == vllm_test::kLtxTeV2AudioOut);
  CHECK(static_cast<int64_t>(v2.aggregate_bias) == vllm_test::kLtxTeV2VideoHasBias);
  CHECK(static_cast<int64_t>(v2.aggregate_bias) == vllm_test::kLtxTeV2AudioHasBias);
  CHECK_FALSE(v2.is_av);

  const Ltx2TextFeatureConfig v1 = vllm::Ltx2SelectTextFeatureVariant(
      V1TransformerConfig(), vllm_test::kLtxTeGemmaHidden,
      vllm_test::kLtxTeGemmaHiddenLayers);
  CHECK(static_cast<int>(v1.variant == Ltx2TextNormVariant::kPerTokenRmsV2) ==
        static_cast<int>(vllm_test::kLtxTeSelectedV1IsV2));
  CHECK(v1.variant == Ltx2TextNormVariant::kPaddedBatchV1);
  CHECK(v1.FlatDim() == vllm_test::kLtxTeV1AggregateIn);
  CHECK(v1.video_out_features == vllm_test::kLtxTeV1AggregateOut);
  CHECK(static_cast<int64_t>(v1.aggregate_bias) == vllm_test::kLtxTeV1AggregateHasBias);
  CHECK(static_cast<int64_t>(v1.is_av) == vllm_test::kLtxTeV1IsAv);
  CHECK(v1.audio_out_features == 0);

  // encoder_configurator.py:190-192 — a PARTIAL V2 marker set is
  // NotImplementedError, not a fall-back to V1.
  nlohmann::json partial = V2TransformerConfig();
  partial.erase("caption_proj_input_norm");
  CHECK_THROWS_AS(vllm::Ltx2SelectTextFeatureVariant(partial, kHidden,
                                                     vllm_test::kLtxTeGemmaHiddenLayers),
                  std::runtime_error);

  // encoder_configurator.py:194-201 — a marker key present with the WRONG value
  // is config drift and is refused too.
  nlohmann::json drifted = V2TransformerConfig();
  drifted["caption_projection_first_linear"] = true;
  CHECK_THROWS_AS(vllm::Ltx2SelectTextFeatureVariant(drifted, kHidden,
                                                     vllm_test::kLtxTeGemmaHiddenLayers),
                  std::runtime_error);
}

TEST_CASE("ltx2 text: the weight manifest matches upstream named_parameters()") {
  // V2 (LTX-2.5): two biased projections, in this order.
  const std::vector<std::string> v2_names = {
      "video_aggregate_embed.weight", "video_aggregate_embed.bias",
      "audio_aggregate_embed.weight", "audio_aggregate_embed.bias"};
  const std::vector<std::vector<int64_t>> v2_shapes = {
      {vllm_test::kLtxTeVideoInner, vllm_test::kLtxTeFlatDim},
      {vllm_test::kLtxTeVideoInner},
      {vllm_test::kLtxTeAudioInner, vllm_test::kLtxTeFlatDim},
      {vllm_test::kLtxTeAudioInner}};
  REQUIRE(static_cast<int64_t>(v2_names.size()) == vllm_test::kLtxTeV2ParamCount);
  size_t dim = 0;
  for (size_t i = 0; i < v2_names.size(); ++i) {
    CHECK(v2_names[i] == std::string(vllm_test::kLtxTeV2ParamNames[i]));
    CHECK(static_cast<int64_t>(v2_shapes[i].size()) == vllm_test::kLtxTeV2ParamRanks[i]);
    for (int64_t d : v2_shapes[i]) CHECK(d == vllm_test::kLtxTeV2ParamDims[dim++]);
  }

  // V1: ONE projection and NO bias.
  REQUIRE(vllm_test::kLtxTeV1ParamCount == 1);
  CHECK(std::string(vllm_test::kLtxTeV1ParamNames[0]) == "aggregate_embed.weight");
  CHECK(vllm_test::kLtxTeV1ParamRanks[0] == 2);
  CHECK(vllm_test::kLtxTeV1ParamDims[0] == vllm_test::kLtxTeV1AggregateOut);
  CHECK(vllm_test::kLtxTeV1ParamDims[1] == vllm_test::kLtxTeFlatDim);
}

TEST_CASE("ltx2 text: `_norm_and_concat_padded_batch`, both padding sides") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const std::vector<float> stacked = Ltx2StackHiddenStates(MakeStates(buffers));
  const size_t count = static_cast<size_t>(kBatch * kSeq * kHidden * kLayers);

  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* want;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeNormV1Left},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeNormV1Right},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const std::vector<float> got = vllm::Ltx2NormAndConcatPaddedBatch(
        stacked.data(), mask.data(), kBatch, kSeq, kHidden, kLayers);
    const double worst = MaxAbsDiff(got, c.want, count);
    MESSAGE("ltx2 text norm V1 (" << std::string(c.tag) << ") max|diff| = " << worst);
    CHECK(worst < kTol);

    // feature_extractor.py:44-45 — the padded positions are ZEROED, exactly.
    for (int64_t b = 0; b < kBatch; ++b)
      for (int64_t t = 0; t < kSeq; ++t) {
        if (mask[static_cast<size_t>(b * kSeq + t)] != 0) continue;
        for (int64_t f = 0; f < kHidden * kLayers; ++f)
          CHECK(got[static_cast<size_t>((b * kSeq + t) * kHidden * kLayers + f)] == 0.0f);
      }
  }
}

TEST_CASE("ltx2 text: `norm_and_concat_per_token_rms`, both padding sides") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const std::vector<float> stacked = Ltx2StackHiddenStates(MakeStates(buffers));
  const size_t count = static_cast<size_t>(kBatch * kSeq * kHidden * kLayers);

  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* want;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeNormV2Left},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeNormV2Right},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const std::vector<float> got = vllm::Ltx2NormAndConcatPerTokenRms(
        stacked.data(), mask.data(), kBatch, kSeq, kHidden, kLayers);
    const double worst = MaxAbsDiff(got, c.want, count);
    MESSAGE("ltx2 text norm V2 (" << std::string(c.tag) << ") max|diff| = " << worst);
    CHECK(worst < kTol);

    for (int64_t b = 0; b < kBatch; ++b)
      for (int64_t t = 0; t < kSeq; ++t) {
        if (mask[static_cast<size_t>(b * kSeq + t)] != 0) continue;
        for (int64_t f = 0; f < kHidden * kLayers; ++f)
          CHECK(got[static_cast<size_t>((b * kSeq + t) * kHidden * kLayers + f)] == 0.0f);
      }
  }

  // The two variants are NOT interchangeable — proven here rather than assumed,
  // so a config-selection bug cannot hide behind "it is still a normalization".
  const std::vector<int32_t> mask = MaskFrom(vllm_test::kLtxTeMaskLeft);
  const std::vector<float> v1 = vllm::Ltx2NormAndConcatPaddedBatch(
      stacked.data(), mask.data(), kBatch, kSeq, kHidden, kLayers);
  const std::vector<float> v2 = vllm::Ltx2NormAndConcatPerTokenRms(
      stacked.data(), mask.data(), kBatch, kSeq, kHidden, kLayers);
  double gap = 0.0;
  for (size_t i = 0; i < count; ++i)
    gap = std::max(gap, std::abs(static_cast<double>(v1[i]) - static_cast<double>(v2[i])));
  MESSAGE("ltx2 text: V1 vs V2 normalization max|diff| = " << gap);
  CHECK(gap > 1.0);
}

TEST_CASE("ltx2 text: `_rescale_norm` uses each projection's OWN width") {
  const double video = vllm::Ltx2RescaleNorm(vllm_test::kLtxTeVideoInner, kHidden);
  const double audio = vllm::Ltx2RescaleNorm(vllm_test::kLtxTeAudioInner, kHidden);
  MESSAGE("ltx2 text rescale video = " << video << " audio = " << audio);
  CHECK(std::abs(video - vllm_test::kLtxTeRescaleVideo) < 1e-12);
  CHECK(std::abs(audio - vllm_test::kLtxTeRescaleAudio) < 1e-12);
  // The shipped ratios sit on OPPOSITE sides of 1 (4096 > 3840 > 2048), so a
  // swapped numerator/denominator cannot pass both arms.
  CHECK(video > 1.0);
  CHECK(audio < 1.0);
  CHECK(vllm::Ltx2RescaleNorm(4096, 3840) > 1.0);
  CHECK(vllm::Ltx2RescaleNorm(2048, 3840) < 1.0);
}

TEST_CASE("ltx2 text: FeatureExtractorV2 — the two caption projections") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const Ltx2TextHiddenStates states = MakeStates(buffers);
  const Ltx2TextFeatureConfig cfg = V2Config();
  const vllm::Ltx2TextEncoderWeights weights = V2Weights();

  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* video;
    const float* audio;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeV2VideoLeft,
       vllm_test::kLtxTeV2AudioLeft},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeV2VideoRight,
       vllm_test::kLtxTeV2AudioRight},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const vllm::Ltx2TextFeatures got =
        vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), weights, cfg);
    const double wv = MaxAbsDiff(got.video, c.video,
                                 static_cast<size_t>(kBatch * kSeq * cfg.video_out_features));
    const double wa = MaxAbsDiff(got.audio, c.audio,
                                 static_cast<size_t>(kBatch * kSeq * cfg.audio_out_features));
    MESSAGE("ltx2 text V2 extractor (" << std::string(c.tag) << ") max|diff| video = " << wv
                                       << " audio = " << wa);
    CHECK(wv < kTol);
    CHECK(wa < kTol);

    // The norm zeroes a padded position, so its PROJECTED value is exactly the
    // Linear's bias — NOT zero. A port that force-zeroes projected pads diverges
    // from upstream on every padded row while still looking "masked".
    for (int64_t b = 0; b < kBatch; ++b)
      for (int64_t t = 0; t < kSeq; ++t) {
        if (mask[static_cast<size_t>(b * kSeq + t)] != 0) continue;
        for (int64_t f = 0; f < cfg.video_out_features; ++f) {
          const size_t idx =
              static_cast<size_t>((b * kSeq + t) * cfg.video_out_features + f);
          CHECK(std::abs(static_cast<double>(got.video[idx]) -
                         static_cast<double>(weights.video.bias[static_cast<size_t>(f)])) <
                kTol);
        }
      }
  }
}

// LTX25-TEXT-LINEAR-SEAM (#1208). The caption projection IS the shared vt GEMM
// seam, not a hand-rolled loop beside it, and this case is what says so
// executably.
//
// It runs at flat = 8192 rather than at the goldens' 24, because the two
// implementations this discriminates between — an f64 accumulator and the seam's
// f32 one — agree to well under any bound at 24. The width is still 23x smaller
// than the shipped 188160, so the fixture stays a few megabytes.
//
// TWO assertions, and the second is what stops the first from being vacuous:
//
//   1. the projection equals `vt::MatmulBT` + bias BYTE for byte. No tolerance:
//      if the production path is the seam then there is nothing to round
//      differently, and a bound here would accept an implementation that merely
//      lands nearby.
//   2. an f64-accumulating reference — the algorithm this row replaced — is
//      measurably FURTHER away than that. Without it a test that passed under
//      both implementations would be indistinguishable from one that
//      discriminates, which is the failure AGENTS.md names for a gate that
//      cannot say how.
//
// Correctness against upstream is NOT this case's job and it does not claim it:
// this is a consistency gate between two of our own paths. The goldens above hold
// the projection to the executed `ltx_core` module at 1e-5, and neither case
// substitutes for the other.
TEST_CASE("ltx2 text: the caption projection is the vt::MatmulBT seam") {
  constexpr int64_t kSeamHidden = 128;
  constexpr int64_t kSeamLayers = 64;                       // num_hidden_layers + 1
  constexpr int64_t kSeamFlat = kSeamHidden * kSeamLayers;  // 8192
  constexpr int64_t kSeamOut = 32;
  constexpr int64_t kSeamBatch = 1;
  constexpr int64_t kSeamSeq = 3;
  constexpr int64_t kSeamRows = kSeamBatch * kSeamSeq;

  // One deterministic stream, so the fixture is reproducible without a checked-in
  // weight byte and without <random>'s implementation-defined engines.
  uint64_t bits = 0x9e3779b97f4a7c15ull;
  auto next = [&bits]() {
    bits ^= bits << 13;
    bits ^= bits >> 7;
    bits ^= bits << 17;
    return static_cast<float>(static_cast<double>(bits >> 40) / 16777216.0 - 0.5);
  };

  std::vector<std::vector<float>> layers(static_cast<size_t>(kSeamLayers));
  std::vector<const float*> ptrs;
  for (int64_t l = 0; l < kSeamLayers; ++l) {
    std::vector<float>& buf = layers[static_cast<size_t>(l)];
    buf.resize(static_cast<size_t>(kSeamRows * kSeamHidden));
    for (float& v : buf) v = next();
    ptrs.push_back(buf.data());
  }
  Ltx2TextHiddenStates states;
  states.layers = ptrs;
  states.batch = kSeamBatch;
  states.seq = kSeamSeq;
  states.hidden = kSeamHidden;

  // Every position valid: a padded row projects to the bias alone, which no
  // accumulator touches, and would only dilute the comparison.
  const std::vector<int32_t> mask(static_cast<size_t>(kSeamRows), 1);

  vllm::Ltx2TextEncoderWeights weights;
  weights.video.in_features = kSeamFlat;
  weights.video.out_features = kSeamOut;
  weights.video.weight.resize(static_cast<size_t>(kSeamOut * kSeamFlat));
  for (float& v : weights.video.weight) v = next() * 0.05f;
  weights.video.bias.resize(static_cast<size_t>(kSeamOut));
  for (float& v : weights.video.bias) v = next() * 0.02f;

  Ltx2TextFeatureConfig cfg;
  cfg.variant = Ltx2TextNormVariant::kPerTokenRmsV2;
  cfg.embedding_dim = kSeamHidden;
  cfg.num_layers = kSeamLayers;
  cfg.video_out_features = kSeamOut;
  cfg.audio_out_features = 0;  // the audio arm is not what this case is about
  cfg.aggregate_bias = true;
  cfg.is_av = false;

  const vllm::Ltx2TextFeatures got =
      vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), weights, cfg);
  const size_t count = static_cast<size_t>(kSeamRows * kSeamOut);
  REQUIRE(got.video.size() == count);

  // The projection's INPUT, rebuilt from the same exported helpers the extractor
  // itself calls, so nothing here re-implements a step: the stack, the V2 norm,
  // and the per-projection rescale (feature_extractor.py:121-129).
  const std::vector<float> stacked = vllm::Ltx2StackHiddenStates(states);
  std::vector<float> scaled = vllm::Ltx2NormAndConcatPerTokenRms(
      stacked.data(), mask.data(), kSeamBatch, kSeamSeq, kSeamHidden, kSeamLayers);
  const float rescale =
      static_cast<float>(vllm::Ltx2RescaleNorm(kSeamOut, kSeamHidden));
  for (float& v : scaled) v *= rescale;

  const vt::Device dev{vt::DeviceType::kCPU, 0};
  vt::Queue q{dev, nullptr};
  std::vector<float> seam(count, 0.0f);
  vt::Tensor a = vt::Tensor::Contiguous(scaled.data(), vt::DType::kF32, dev,
                                        {kSeamRows, kSeamFlat});
  vt::Tensor b = vt::Tensor::Contiguous(weights.video.weight.data(), vt::DType::kF32,
                                        dev, {kSeamOut, kSeamFlat});
  vt::Tensor o =
      vt::Tensor::Contiguous(seam.data(), vt::DType::kF32, dev, {kSeamRows, kSeamOut});
  vt::MatmulBT(q, o, a, b);
  for (int64_t r = 0; r < kSeamRows; ++r)
    for (int64_t f = 0; f < kSeamOut; ++f)
      seam[static_cast<size_t>(r * kSeamOut + f)] +=
          weights.video.bias[static_cast<size_t>(f)];

  // 1 — byte equality with the seam.
  size_t mismatched = 0;
  double worst_seam = 0.0;
  for (size_t i = 0; i < count; ++i) {
    if (got.video[i] != seam[i]) ++mismatched;
    worst_seam = std::max(worst_seam, std::abs(static_cast<double>(got.video[i]) -
                                               static_cast<double>(seam[i])));
  }
  MESSAGE("ltx2 text seam: elements differing from vt::MatmulBT = "
          << mismatched << " / " << count << ", max|diff| = " << worst_seam);
  CHECK(mismatched == 0);

  // 2 — the discrimination proof. An f64 accumulator over the SAME input lands a
  // measurable distance away at this width, so passing assertion 1 is a statement
  // about the implementation rather than about the fixture.
  double worst_f64 = 0.0;
  for (int64_t r = 0; r < kSeamRows; ++r) {
    const float* xr = scaled.data() + static_cast<size_t>(r * kSeamFlat);
    for (int64_t f = 0; f < kSeamOut; ++f) {
      const float* wr =
          weights.video.weight.data() + static_cast<size_t>(f * kSeamFlat);
      double acc = static_cast<double>(weights.video.bias[static_cast<size_t>(f)]);
      for (int64_t i = 0; i < kSeamFlat; ++i)
        acc += static_cast<double>(xr[i]) * static_cast<double>(wr[i]);
      worst_f64 = std::max(
          worst_f64,
          std::abs(
              static_cast<double>(got.video[static_cast<size_t>(r * kSeamOut + f)]) -
              acc));
    }
  }
  MESSAGE("ltx2 text seam: max|diff| vs an f64-accumulating reference = " << worst_f64);
  // The floor sits between two MEASURED values on this deterministic fixture:
  // 5.28e-08 when the projection accumulates in f64 (all that separates it from
  // the reference is the store rounding), and 4.11e-06 when it is the seam. Both
  // are constants of the fixture, so the band is a discrimination statement and
  // not a tolerance.
  CHECK(worst_f64 > 1e-6);
}

TEST_CASE("ltx2 text: FeatureExtractorV1 — one bias-free projection, is_av") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const Ltx2TextHiddenStates states = MakeStates(buffers);
  const Ltx2TextFeatureConfig cfg = V1Config();
  const vllm::Ltx2TextEncoderWeights weights = V1Weights();

  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* video;
    const float* audio;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeV1VideoLeft,
       vllm_test::kLtxTeV1AudioLeft},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeV1VideoRight,
       vllm_test::kLtxTeV1AudioRight},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const vllm::Ltx2TextFeatures got =
        vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), weights, cfg);
    const size_t count = static_cast<size_t>(kBatch * kSeq * cfg.video_out_features);
    const double wv = MaxAbsDiff(got.video, c.video, count);
    const double wa = MaxAbsDiff(got.audio, c.audio, count);
    MESSAGE("ltx2 text V1 extractor (" << std::string(c.tag) << ") max|diff| video = " << wv
                                       << " audio = " << wa);
    CHECK(wv < kTol);
    CHECK(wa < kTol);
    // feature_extractor.py:95-96 — `is_av` returns the SAME tensor twice.
    CHECK(got.audio == got.video);

    // No bias: a padded position projects to exactly ZERO under V1.
    for (int64_t b = 0; b < kBatch; ++b)
      for (int64_t t = 0; t < kSeq; ++t) {
        if (mask[static_cast<size_t>(b * kSeq + t)] != 0) continue;
        for (int64_t f = 0; f < cfg.video_out_features; ++f)
          CHECK(got.video[static_cast<size_t>((b * kSeq + t) * cfg.video_out_features + f)] ==
                0.0f);
      }
  }
}

TEST_CASE("ltx2 text: additive mask, right-pad ordering and the binary mask") {
  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* additive;
    const int64_t* sort_idx;
    const float* reordered_mask;
    const int64_t* binary;
    const int64_t* binary_registers;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeAdditiveMaskLeft,
       vllm_test::kLtxTeSortIdxLeft, vllm_test::kLtxTeReorderedMaskLeft,
       vllm_test::kLtxTeBinaryMaskLeft, vllm_test::kLtxTeBinaryMaskFromRegistersLeft},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeAdditiveMaskRight,
       vllm_test::kLtxTeSortIdxRight, vllm_test::kLtxTeReorderedMaskRight,
       vllm_test::kLtxTeBinaryMaskRight, vllm_test::kLtxTeBinaryMaskFromRegistersRight},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const std::vector<float> additive =
        vllm::Ltx2ConvertToAdditiveMask(mask.data(), kBatch, kSeq);
    CheckExact(additive, c.additive, static_cast<size_t>(kBatch * kSeq));

    std::vector<int32_t> sort_index;
    std::vector<float> reordered;
    vllm::Ltx2ComputeRightPadOrder(additive.data(), kBatch, kSeq, sort_index, reordered);
    CheckExactI(sort_index, c.sort_idx, static_cast<size_t>(kBatch * kSeq));
    CheckExact(reordered, c.reordered_mask, static_cast<size_t>(kBatch * kSeq));

    const std::vector<int32_t> binary =
        vllm::Ltx2ToBinaryMask(reordered.data(), kBatch, kSeq);
    CheckExactI(binary, c.binary, static_cast<size_t>(kBatch * kSeq));
    const std::vector<float> zeros(static_cast<size_t>(kBatch * kSeq), 0.0f);
    const std::vector<int32_t> from_registers =
        vllm::Ltx2ToBinaryMask(zeros.data(), kBatch, kSeq);
    CheckExactI(from_registers, c.binary_registers, static_cast<size_t>(kBatch * kSeq));
  }

  // embeddings_processor.py:26-27 — idempotent on already right-padded input.
  const std::vector<int32_t> right = MaskFrom(vllm_test::kLtxTeMaskRight);
  const std::vector<float> additive =
      vllm::Ltx2ConvertToAdditiveMask(right.data(), kBatch, kSeq);
  std::vector<int32_t> sort_index;
  std::vector<float> reordered;
  vllm::Ltx2ComputeRightPadOrder(additive.data(), kBatch, kSeq, sort_index, reordered);
  for (int64_t b = 0; b < kBatch; ++b)
    for (int64_t t = 0; t < kSeq; ++t)
      CHECK(sort_index[static_cast<size_t>(b * kSeq + t)] == static_cast<int32_t>(t));
}

TEST_CASE("ltx2 text: the encoder -> conditioning hand-off") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const Ltx2TextHiddenStates states = MakeStates(buffers);
  const Ltx2TextFeatureConfig cfg = V2Config();
  const vllm::Ltx2TextEncoderWeights weights = V2Weights();

  struct Case {
    const char* tag;
    const int64_t* mask;
    const float* video;
    const float* audio;
    const float* reordered_mask;
    const int64_t* sort_idx;
  };
  const Case cases[] = {
      {"left", vllm_test::kLtxTeMaskLeft, vllm_test::kLtxTeReorderedVideoLeft,
       vllm_test::kLtxTeReorderedAudioLeft, vllm_test::kLtxTeReorderedMaskLeft,
       vllm_test::kLtxTeSortIdxLeft},
      {"right", vllm_test::kLtxTeMaskRight, vllm_test::kLtxTeReorderedVideoRight,
       vllm_test::kLtxTeReorderedAudioRight, vllm_test::kLtxTeReorderedMaskRight,
       vllm_test::kLtxTeSortIdxRight},
  };
  for (const Case& c : cases) {
    const std::vector<int32_t> mask = MaskFrom(c.mask);
    const vllm::Ltx2TextConditioning got =
        vllm::Ltx2TextEncoderConditioning(states, mask.data(), weights, cfg);
    const double wv = MaxAbsDiff(got.video, c.video,
                                 static_cast<size_t>(kBatch * kSeq * cfg.video_out_features));
    const double wa = MaxAbsDiff(got.audio, c.audio,
                                 static_cast<size_t>(kBatch * kSeq * cfg.audio_out_features));
    MESSAGE("ltx2 text conditioning (" << std::string(c.tag) << ") max|diff| video = " << wv
                                       << " audio = " << wa);
    CHECK(wv < kTol);
    CHECK(wa < kTol);
    CheckExact(got.additive_mask, c.reordered_mask, static_cast<size_t>(kBatch * kSeq));
    CheckExactI(got.sort_index, c.sort_idx, static_cast<size_t>(kBatch * kSeq));
  }
}

TEST_CASE("ltx2 text: the DECLARED weight contract is checked against the WEIGHTS") {
  // `aggregate_bias` and `*_out_features` are what the SELECTOR read out of the
  // checkpoint config (ltx2_text_encoder.cpp:130, :166). `w.bias.empty()`,
  // `w.out_features` and `w.in_features` are what the LOADER actually supplied.
  // Nothing compared the two, so both of the following ran silently:
  //
  //   * bias-less weights under aggregate_bias=true — the loader reads
  //     `video_aggregate_embed.weight` (U8/NVFP4) and misses `.bias` (BF16, a
  //     DIFFERENT dtype on a different unpack path). Every conditioning row is
  //     then shifted by the missing bias and every padded row projects to 0
  //     instead of to the bias: finite, correctly shaped, WRONG PROMPT. Measured
  //     drift on this fixture before the refusal existed: 0.0194063.
  //   * an out_features mismatch — the config claimed 40 while the video tensor
  //     came out 80 wide, because the width that runs comes from `w.out_features`
  //     and the config's copy is only used for `_rescale_norm`.
  //
  // This is the header's fourth "fails silently" entry (ltx2_text_encoder.h:57-60)
  // enforced rather than merely named, and it is phase L6's most likely mistake.
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const Ltx2TextHiddenStates states = MakeStates(buffers);
  const std::vector<int32_t> mask = MaskFrom(vllm_test::kLtxTeMaskLeft);

  SUBCASE("the CORRECT weights still go through the same door") {
    CHECK_NOTHROW(vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), V2Weights(),
                                                        V2Config()));
    CHECK_NOTHROW(vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), V1Weights(),
                                                        V1Config()));
    CHECK_NOTHROW(vllm::Ltx2TextEncoderConditioning(states, mask.data(), V2Weights(),
                                                    V2Config()));
    CHECK_NOTHROW(vllm::Ltx2TextEncoderConditioning(states, mask.data(), V1Weights(),
                                                    V1Config()));
  }

  SUBCASE("aggregate_bias=true with a bias-less projection is REFUSED") {
    vllm::Ltx2TextEncoderWeights video_dropped = V2Weights();
    video_dropped.video.bias.clear();
    CHECK_THROWS_AS(vllm::Ltx2TextFeatureExtractorForward(states, mask.data(),
                                                          video_dropped, V2Config()),
                    std::runtime_error);
    CHECK_THROWS_AS(vllm::Ltx2TextEncoderConditioning(states, mask.data(), video_dropped,
                                                      V2Config()),
                    std::runtime_error);

    vllm::Ltx2TextEncoderWeights audio_dropped = V2Weights();
    audio_dropped.audio.bias.clear();
    CHECK_THROWS_AS(vllm::Ltx2TextFeatureExtractorForward(states, mask.data(),
                                                          audio_dropped, V2Config()),
                    std::runtime_error);
    CHECK_THROWS_AS(vllm::Ltx2TextEncoderConditioning(states, mask.data(), audio_dropped,
                                                      V2Config()),
                    std::runtime_error);
  }

  SUBCASE("aggregate_bias=false with a bias that appeared anyway is REFUSED") {
    // encoder_configurator.py:187 builds V1's Linear with bias=False, so a bias
    // here means the loader bound a tensor upstream does not have.
    vllm::Ltx2TextEncoderWeights invented = V1Weights();
    invented.video.bias.assign(static_cast<size_t>(kHidden), 0.5f);
    CHECK_THROWS_AS(vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), invented,
                                                          V1Config()),
                    std::runtime_error);
    CHECK_THROWS_AS(vllm::Ltx2TextEncoderConditioning(states, mask.data(), invented,
                                                      V1Config()),
                    std::runtime_error);
  }

  SUBCASE("an out_features that disagrees with the config is REFUSED") {
    Ltx2TextFeatureConfig narrowed = V2Config();
    narrowed.video_out_features = vllm_test::kLtxTeVideoInner / 2;
    CHECK_THROWS_AS(vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), V2Weights(),
                                                          narrowed),
                    std::runtime_error);
    CHECK_THROWS_AS(vllm::Ltx2TextEncoderConditioning(states, mask.data(), V2Weights(),
                                                      narrowed),
                    std::runtime_error);

    Ltx2TextFeatureConfig audio_narrowed = V2Config();
    audio_narrowed.audio_out_features = vllm_test::kLtxTeAudioInner / 2;
    CHECK_THROWS_AS(vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), V2Weights(),
                                                          audio_narrowed),
                    std::runtime_error);

    Ltx2TextFeatureConfig v1_narrowed = V1Config();
    v1_narrowed.video_out_features = kHidden - 1;
    CHECK_THROWS_AS(vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), V1Weights(),
                                                          v1_narrowed),
                    std::runtime_error);
  }

  SUBCASE("an in_features that is not the FLAT width is REFUSED") {
    // The "+1 is the embedding layer" trap from the other side: a projection
    // built for 48 layers instead of 49 is exactly `flat - embedding_dim` wide.
    vllm::Ltx2TextEncoderWeights short_flat = V2Weights();
    short_flat.video.in_features = vllm_test::kLtxTeFlatDim - kHidden;
    short_flat.video.weight.resize(
        static_cast<size_t>(short_flat.video.out_features * short_flat.video.in_features));
    CHECK_THROWS_AS(vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), short_flat,
                                                          V2Config()),
                    std::runtime_error);
    CHECK_THROWS_AS(vllm::Ltx2TextEncoderConditioning(states, mask.data(), short_flat,
                                                      V2Config()),
                    std::runtime_error);
  }
}

TEST_CASE("ltx2 text: the normalization EPSILONS are PINNED and load-bearing") {
  // The epsilon limit is a CLASS, not one instance: a constant that only matters
  // on a degenerate input is invisible to any golden built from random values,
  // and this fixture's values are random. Measured on THIS suite before the pins
  // below existed: `range_ + eps` -> `range_ + 0.0f` moved norm V1's max|diff|
  // from 4.77e-07 to 5.24521e-06 and the gate still said SUCCESS, and
  // `denom + eps` -> `denom + 0.0` moved nothing at all.
  //
  // That second measurement is the reason the degenerate goldens below are ARRAYS
  // rather than `isfinite` booleans. The generator was already running each of
  // these inputs through upstream and already had the output tensor; reducing it
  // to one boolean is what made `denom + 0.0` invisible, and a float64 mean
  // denominator invisible with it. Re-measured against the arrays:
  // `denom + eps` -> `denom + 0.0` now fails three of them, at 0.476837 (constant
  // slice), 0.715256 (one valid token) and 0.476837 (near-constant).
  //
  // So they are held three ways: the VALUE against upstream measured by probe, the
  // ARITHMETIC WIDTH the value is used at, and the DEGENERATE INPUT that makes
  // each one the only thing between the port and a division by zero.

  // 1. The values. The generator recovers them from upstream numerically
  //    (gen-ltx2-text-goldens.py :: emit_epsilons) rather than restating ours.
  MESSAGE("ltx2 text eps V1 upstream = " << vllm_test::kLtxTeNormV1EpsUpstream
                                         << " ours = " << vllm::kLtx2TextNormV1Eps);
  MESSAGE("ltx2 text eps V2 upstream = " << vllm_test::kLtxTeNormV2EpsUpstream
                                         << " ours = " << vllm::kLtx2TextNormV2Eps);
  CHECK(std::abs(vllm::kLtx2TextNormV1Eps - vllm_test::kLtxTeNormV1EpsUpstream) < 1e-8);
  CHECK(std::abs(static_cast<double>(vllm::kLtx2TextNormV2Eps) -
                 vllm_test::kLtxTeNormV2EpsUpstream) < 1e-8);
  // Both probes must land ON 1e-6, not merely near each other.
  CHECK(vllm_test::kLtxTeNormV1EpsUpstream > 0.0);
  CHECK(vllm_test::kLtxTeNormV2EpsUpstream > 0.0);

  const size_t count = static_cast<size_t>(kBatch * kSeq * kHidden * kLayers);
  const std::vector<int32_t> mask = MaskFrom(vllm_test::kLtxTeMaskRight);

  // 2. V1 `range_ + eps` (feature_extractor.py:41). A CONSTANT (batch, layer)
  //    slice makes `range_` EXACTLY zero, which is the only input on which THAT
  //    epsilon is reachable — and because it then divides the whole expression,
  //    it also multiplies any error in `mean` by 8/eps = 8e6. So this input gates
  //    both epsilons at once, against upstream's VALUES.
  REQUIRE(vllm_test::kLtxTeNormV1ConstantSliceFinite == 1);
  const std::vector<float> constant(count, 0.5f);
  const std::vector<float> const_out = vllm::Ltx2NormAndConcatPaddedBatch(
      constant.data(), mask.data(), kBatch, kSeq, kHidden, kLayers);
  REQUIRE(const_out.size() == count);
  for (float x : const_out) CHECK(std::isfinite(x));
  {
    const double worst = MaxAbsDiff(const_out, vllm_test::kLtxTeNormV1ConstantSlice, count);
    MESSAGE("ltx2 text norm V1 constant slice max|diff| = " << worst);
    CHECK(worst < kTol);
  }

  // 3. V1 `denom + eps` (feature_extractor.py:34-35), the mean's denominator, and
  //    the DTYPE it is added in. Upstream's `denom` is an int64 tensor and `eps` a
  //    python float, so `denom + eps` promotes to the DEFAULT dtype and the add
  //    happens in float32: 18 + 1e-6 is 18.000001907348633, where a float64 add
  //    gives 18.000001. That is one f32 ULP in `mean`, and case 2 above amplifies
  //    it by 8/eps.
  //
  //    Stated precisely, because the previous revision of this comment claimed the
  //    opposite and was wrong: this epsilon IS observable at the output. On the
  //    constant slice under `mask_right`, row 0 (3 valid tokens, denom 18) reads
  //    0.476837158 upstream and 0.238418579 from a float64 denominator — 23842x
  //    kTol. It is unobservable only on an all-pad row, checked immediately below,
  //    where :44-45 zeroes every position before anything can escape. Two further
  //    degenerate inputs are gated because neither of them can see the defect on
  //    its own: ONE valid token in row 0 (denom 6) agrees to the bit either way,
  //    and so does row 1 of the constant slice (denom 24). A gate built from only
  //    those would be green on a defect worth 23842x its own tolerance.
  //
  //    On a realistic Gemma workload none of this is catastrophic: `range_` is
  //    O(1) rather than 0, so the same one-ULP mean error lands around 2e-12 at
  //    the output. It is gated here because the degenerate arm is where it is
  //    visible, not because the degenerate arm is where it matters.
  REQUIRE(vllm_test::kLtxTeNormV1ZeroLenFinite == 1);
  REQUIRE(vllm_test::kLtxTeNormV1ZeroLenRowIsZero == 1);
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const std::vector<float> stacked = Ltx2StackHiddenStates(MakeStates(buffers));
  std::vector<int32_t> zero_len(static_cast<size_t>(kBatch * kSeq), 1);
  for (int64_t t = 0; t < kSeq; ++t) zero_len[static_cast<size_t>(t)] = 0;
  const std::vector<float> zero_len_out = vllm::Ltx2NormAndConcatPaddedBatch(
      stacked.data(), zero_len.data(), kBatch, kSeq, kHidden, kLayers);
  REQUIRE(zero_len_out.size() == count);
  for (float x : zero_len_out) CHECK(std::isfinite(x));
  for (int64_t i = 0; i < kSeq * kHidden * kLayers; ++i)
    CHECK(zero_len_out[static_cast<size_t>(i)] == 0.0f);
  {
    const double worst = MaxAbsDiff(zero_len_out, vllm_test::kLtxTeNormV1ZeroLen, count);
    MESSAGE("ltx2 text norm V1 zero-length row max|diff| = " << worst);
    CHECK(worst < kTol);
  }

  // 3b. ONE valid token in row 0, so `sequence_lengths == 1` and denom == d. The
  //     mask travels with the golden so the two sides cannot disagree about which
  //     positions are valid.
  {
    const std::vector<int32_t> one_mask = MaskFrom(vllm_test::kLtxTeNormV1OneValidTokenMask);
    const std::vector<float> one_out = vllm::Ltx2NormAndConcatPaddedBatch(
        constant.data(), one_mask.data(), kBatch, kSeq, kHidden, kLayers);
    const double worst = MaxAbsDiff(one_out, vllm_test::kLtxTeNormV1OneValidToken, count);
    MESSAGE("ltx2 text norm V1 one-valid-token max|diff| = " << worst);
    CHECK(worst < kTol);
  }

  // 3c. A range of exactly ONE f32 ULP, so `range_` (2^-24) and `eps` (1e-6) are
  //     the same order and neither dominates the other.
  {
    std::vector<float> near = constant;
    near[0] = std::nextafterf(0.5f, 1.0f);
    const std::vector<float> near_out = vllm::Ltx2NormAndConcatPaddedBatch(
        near.data(), mask.data(), kBatch, kSeq, kHidden, kLayers);
    const double worst = MaxAbsDiff(near_out, vllm_test::kLtxTeNormV1NearConstant, count);
    MESSAGE("ltx2 text norm V1 near-constant max|diff| = " << worst);
    CHECK(worst < kTol);
  }

  // 4. V2 `variance + 1e-6` (feature_extractor.py:61). A token whose whole hidden
  //    slice is zero has variance 0, so `rsqrt(0)` is the alternative.
  REQUIRE(vllm_test::kLtxTeNormV2ZeroVarianceFinite == 1);
  std::vector<float> zero_var = stacked;
  for (int64_t d = 0; d < kHidden; ++d)
    for (int64_t l = 0; l < kLayers; ++l)
      zero_var[static_cast<size_t>(d * kLayers + l)] = 0.0f;
  const std::vector<float> zero_var_out = vllm::Ltx2NormAndConcatPerTokenRms(
      zero_var.data(), mask.data(), kBatch, kSeq, kHidden, kLayers);
  REQUIRE(zero_var_out.size() == count);
  for (float x : zero_var_out) CHECK(std::isfinite(x));
  {
    const double worst = MaxAbsDiff(zero_var_out, vllm_test::kLtxTeNormV2ZeroVariance, count);
    MESSAGE("ltx2 text norm V2 zero-variance max|diff| = " << worst);
    CHECK(worst < kTol);
  }
}

TEST_CASE("ltx2 text: a non-f32 compute dtype is REFUSED, never silently widened") {
  const std::vector<std::vector<float>> buffers = HiddenStateBuffers();
  const Ltx2TextHiddenStates states = MakeStates(buffers);
  const std::vector<int32_t> mask = MaskFrom(vllm_test::kLtxTeMaskLeft);
  CHECK_THROWS_AS(vllm::Ltx2TextFeatureExtractorForward(states, mask.data(), V2Weights(),
                                                        V2Config(), vt::DType::kBF16),
                  std::runtime_error);
  CHECK_THROWS_AS(vllm::Ltx2TextEncoderConditioning(states, mask.data(), V2Weights(),
                                                    V2Config(), vt::DType::kBF16),
                  std::runtime_error);
}

TEST_CASE("ltx2 text: the tokenizer and HF sidecars come out of TENSORS, not files") {
  const fs::path dir =
      fs::temp_directory_path() / ("ltx2_text_assets_" + std::to_string(vllm_test::ProcessId()));
  fs::create_directories(dir);

  const std::string tokenizer = R"({"version":"1.0","model":{"type":"BPE"}})";
  const std::string tok_cfg = R"({"tokenizer_class":"PreTrainedTokenizerFast"})";
  const std::string proc_cfg = R"({"processor_class":"Gemma4Processor"})";
  const std::string chat = "{{ messages }}";

  SUBCASE("a complete pack round-trips, config included") {
    const std::string path = WritePack(
        dir / "full.safetensors",
        {{"tokenizer_json", tokenizer},
         {"hf_asset__tokenizer_config.json", tok_cfg},
         {"hf_asset__processor_config.json", proc_cfg},
         {"hf_asset__chat_template.jinja", chat}},
        R"({"format":"pt","gemma_config":"{\"model_type\":\"gemma4_unified\",\"hidden_size\":3840}"})");
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    const vllm::Ltx2GemmaAssets assets = vllm::Ltx2LoadGemmaAssets(file);
    CHECK(BytesToString(assets.tokenizer_json) == tokenizer);
    CHECK(assets.sidecars.size() == 3);
    CHECK(BytesToString(assets.SidecarBytes("tokenizer_config.json")) == tok_cfg);
    CHECK(BytesToString(assets.SidecarBytes("chat_template.jinja")) == chat);
    CHECK(assets.SidecarJson("processor_config.json")["processor_class"] ==
          "Gemma4Processor");
    REQUIRE(assets.has_config);
    CHECK(assets.config["model_type"] == "gemma4_unified");
    CHECK(assets.config["hidden_size"] == 3840);
    // gemma_assets.py:148 — an absent sidecar throws, it does not return empty.
    CHECK_THROWS_AS(assets.SidecarBytes("nope.json"), std::runtime_error);
  }

  SUBCASE("a missing tokenizer tensor is refused by NAME") {
    const std::string path =
        WritePack(dir / "notok.safetensors",
                  {{"hf_asset__tokenizer_config.json", tok_cfg},
                   {"hf_asset__processor_config.json", proc_cfg}},
                  R"({"gemma_config":"{}"})");
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaAssets(file), std::runtime_error);
  }

  SUBCASE("a missing REQUIRED sidecar is refused") {
    // gemma_assets.py:38-41 — tokenizer_config.json and processor_config.json are
    // required; chat_template and generation_config are not.
    const std::string path =
        WritePack(dir / "nosidecar.safetensors",
                  {{"tokenizer_json", tokenizer},
                   {"hf_asset__tokenizer_config.json", tok_cfg}},
                  R"({"gemma_config":"{}"})");
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaAssets(file), std::runtime_error);
  }

  SUBCASE("the SHIPPED checkpoint's missing __metadata__ is refused, not invented") {
    // MEASURED: vonkaiser/LTX-2.5-FP8-NVFP4's
    // gemma4-12b-with-proj-nvfp4-torchao.safetensors has 1688 tensors and NO
    // __metadata__ block, so upstream's from_single_file raises on it
    // (gemma_assets.py:110-114). We refuse identically by default, and a caller
    // that sources the config elsewhere opts out explicitly.
    const std::string path = WritePack(dir / "nometa.safetensors",
                                       {{"tokenizer_json", tokenizer},
                                        {"hf_asset__tokenizer_config.json", tok_cfg},
                                        {"hf_asset__processor_config.json", proc_cfg}},
                                       "");
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaAssets(file), std::runtime_error);
    const vllm::Ltx2GemmaAssets assets = vllm::Ltx2LoadGemmaAssets(file, false);
    CHECK_FALSE(assets.has_config);
    CHECK(BytesToString(assets.tokenizer_json) == tokenizer);
  }

  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ─────────────── the Gemma-4 seam LTX needs, and its ORDER ───────────────────

TEST_CASE("gemma4: ForwardHiddenStates returns L+1 states in transformers' order") {
  const vllm::HfConfig cfg = TinyGemma4Config();
  const vllm::Gemma4Weights weights = TinyGemma4Weights(cfg);
  const int64_t T = 5;
  const int64_t H = cfg.hidden_size;
  const int64_t L = cfg.num_hidden_layers;
  Gemma4CachePool pool(cfg, /*num_blocks=*/2, /*block_size=*/8);
  const vllm::v1::CommonAttentionMetadata meta = PrefillMeta(T, 8);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Qcpu();

  const vllm::Gemma4HiddenStatesResult got = vllm::Gemma4Model::ForwardHiddenStates(
      tokens, positions, meta, pool.attn_kv, weights, cfg, q);

  // The count IS the contract: 48 + 1 on the shipped tower, L + 1 here.
  REQUIRE(static_cast<int64_t>(got.hidden_states.size()) == L + 1);
  for (const std::vector<float>& state : got.hidden_states) {
    REQUIRE(static_cast<int64_t>(state.size()) == T * H);
    for (float x : state) REQUIRE(std::isfinite(x));
  }

  // [0] is the EMBEDDING output, sqrt(hidden)-scaled — not the first layer's
  // output. Reconstructed here from the table, so a port that starts the tuple at
  // layer 0's output fails.
  const auto* table = reinterpret_cast<const uint16_t*>(weights.embed_tokens.bytes.data());
  const double normalizer =
      static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(std::sqrt(static_cast<float>(H)))));
  double worst_embed = 0.0;
  double scale_embed = 0.0;
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < H; ++h) {
      const double want =
          static_cast<double>(vt::BF16ToF32(
              table[static_cast<size_t>(tokens[static_cast<size_t>(t)] * H + h)])) *
          normalizer;
      const double have =
          static_cast<double>(got.hidden_states[0][static_cast<size_t>(t * H + h)]);
      worst_embed = std::max(worst_embed, std::abs(have - want));
      scale_embed = std::max(scale_embed, std::abs(want));
    }
  MESSAGE("gemma4 hidden_states[0] vs embed*sqrt(H): max|diff| = "
          << worst_embed << " over max|value| = " << scale_embed);
  // The state is stored bf16, so equality is bf16 rounding of the product.
  CHECK(worst_embed < 0.01 * scale_embed);

  // [L] is the FINAL-NORMED state, not the raw output of the last decoder layer.
  // Proven by the invariant only the final-normed state satisfies: the logits are
  // exactly that state through the tied lm_head. A port that stored the raw last
  // layer output here would be off by the whole RMSNorm.
  const int64_t V = cfg.vocab_size;
  REQUIRE(static_cast<int64_t>(got.logits.size()) == T * V);
  double worst_logit = 0.0;
  double scale_logit = 0.0;
  for (int64_t t = 0; t < T; ++t)
    for (int64_t v = 0; v < V; ++v) {
      double acc = 0.0;
      for (int64_t h = 0; h < H; ++h)
        acc += static_cast<double>(
                   got.hidden_states[static_cast<size_t>(L)][static_cast<size_t>(t * H + h)]) *
               static_cast<double>(vt::BF16ToF32(table[static_cast<size_t>(v * H + h)]));
      const double have = static_cast<double>(got.logits[static_cast<size_t>(t * V + v)]);
      worst_logit = std::max(worst_logit, std::abs(have - acc));
      scale_logit = std::max(scale_logit, std::abs(acc));
    }
  MESSAGE("gemma4 hidden_states[L] @ lm_head vs logits: max|diff| = "
          << worst_logit << " over max|value| = " << scale_logit);
  CHECK(worst_logit < 0.01 * scale_logit);

  // Every state genuinely differs from its neighbour: a capture that pushed the
  // same buffer L+1 times would satisfy every check above except this one.
  for (int64_t i = 0; i + 1 <= L; ++i) {
    double gap = 0.0;
    for (size_t k = 0; k < got.hidden_states[static_cast<size_t>(i)].size(); ++k)
      gap = std::max(gap, std::abs(static_cast<double>(
                                       got.hidden_states[static_cast<size_t>(i)][k]) -
                                   static_cast<double>(
                                       got.hidden_states[static_cast<size_t>(i + 1)][k])));
    CHECK(gap > 0.0);
  }

  // The plain forward is unchanged by the capture: same logits, bit for bit.
  Gemma4CachePool pool2(cfg, 2, 8);
  const std::vector<float> plain = vllm::Gemma4Model::Forward(
      tokens, positions, meta, pool2.attn_kv, weights, cfg, q);
  REQUIRE(plain.size() == got.logits.size());
  for (size_t i = 0; i < plain.size(); ++i) CHECK(plain[i] == got.logits[i]);
}

TEST_CASE("gemma4 -> ltx2: the captured stack feeds the conditioning path") {
  // The seam end to end at reduced dims: every Gemma-4 hidden state goes straight
  // into the LTX feature extractor, which is the whole reason the capture exists.
  const vllm::HfConfig cfg = TinyGemma4Config();
  const vllm::Gemma4Weights weights = TinyGemma4Weights(cfg);
  const int64_t T = 5;
  Gemma4CachePool pool(cfg, 2, 8);
  const vllm::v1::CommonAttentionMetadata meta = PrefillMeta(T, 8);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Qcpu();
  const vllm::Gemma4HiddenStatesResult run = vllm::Gemma4Model::ForwardHiddenStates(
      tokens, positions, meta, pool.attn_kv, weights, cfg, q);

  Ltx2TextFeatureConfig fcfg = vllm::Ltx2SelectTextFeatureVariant(
      V2TransformerConfig(), cfg.hidden_size, cfg.num_hidden_layers);
  REQUIRE(fcfg.num_layers == static_cast<int64_t>(run.hidden_states.size()));

  vllm::Ltx2TextEncoderWeights w;
  w.video.out_features = fcfg.video_out_features;
  w.video.in_features = fcfg.FlatDim();
  w.video.weight = Ltx2Param("video_aggregate_embed.weight",
                             {w.video.out_features, w.video.in_features});
  w.video.bias = Ltx2Param("video_aggregate_embed.bias", {w.video.out_features});
  w.audio.out_features = fcfg.audio_out_features;
  w.audio.in_features = fcfg.FlatDim();
  w.audio.weight = Ltx2Param("audio_aggregate_embed.weight",
                             {w.audio.out_features, w.audio.in_features});
  w.audio.bias = Ltx2Param("audio_aggregate_embed.bias", {w.audio.out_features});

  Ltx2TextHiddenStates states;
  for (const std::vector<float>& s : run.hidden_states) states.layers.push_back(s.data());
  states.batch = 1;
  states.seq = T;
  states.hidden = cfg.hidden_size;

  // One padded position, to keep the mask path live on the real stack too.
  const std::vector<int32_t> mask = {1, 1, 1, 1, 0};
  const vllm::Ltx2TextConditioning cond =
      vllm::Ltx2TextEncoderConditioning(states, mask.data(), w, fcfg);
  REQUIRE(cond.video.size() == static_cast<size_t>(T * fcfg.video_out_features));
  REQUIRE(cond.audio.size() == static_cast<size_t>(T * fcfg.audio_out_features));
  for (float x : cond.video) REQUIRE(std::isfinite(x));
  for (float x : cond.audio) REQUIRE(std::isfinite(x));

  // A stack missing the embedding state is REFUSED by count, not silently padded.
  Ltx2TextHiddenStates short_stack = states;
  short_stack.layers.pop_back();
  CHECK_THROWS_AS(
      vllm::Ltx2TextEncoderConditioning(short_stack, mask.data(), w, fcfg),
      std::runtime_error);
}

// ─────────── the TOWER, against the oracle phase L3 could not run ────────────
//
// L3 recorded that this comparison was impossible: the `transformers` on the box
// had no `gemma4_unified` in CONFIG_MAPPING, so the tower could not be built at
// reduced dims and there was nothing to compare against. It is possible now, and
// these cases are the first time `Gemma4Model::ForwardHiddenStates` is held to a
// RUNNING upstream rather than to invariants derived from its own output.
//
// The fixture keeps every geometry the shipped 12B mixes — a (sliding x 5, full)
// pattern twice over, two head widths, 2 kv heads against 1, `attention_k_eq_v`
// on the full layers only, two rope types at two thetas — because a fixture with
// one uniform layer type cannot separate a port that handles both from one that
// handles the first and applies it twice.

TEST_CASE("gemma4 tower: the config the oracle ran PARSES into HfConfig") {
  const vllm::HfConfig c = TowerConfig();
  CHECK(c.hidden_size == vllm_test::kLtxTowerHidden);
  CHECK(c.num_hidden_layers == vllm_test::kLtxTowerNumLayers);
  CHECK(c.head_dim == vllm_test::kLtxTowerHeadDim);
  CHECK(c.num_attention_heads == vllm_test::kLtxTowerNumHeads);
  CHECK(c.num_key_value_heads == vllm_test::kLtxTowerNumKvHeads);
  CHECK(c.intermediate_size == vllm_test::kLtxTowerIntermediate);
  CHECK(c.vocab_size == vllm_test::kLtxTowerVocab);
  REQUIRE(c.sliding_window.has_value());
  CHECK(*c.sliding_window == vllm_test::kLtxTowerSlidingWindow);

  // The fields that reach the forward through `raw` and that NO shape encodes.
  // Each one moves every hidden state while leaving every tensor byte identical,
  // which is why they are read from the config rather than defaulted.
  CHECK(c.raw["global_head_dim"] == vllm_test::kLtxTowerGlobalHeadDim);
  CHECK(c.raw["num_global_key_value_heads"] == vllm_test::kLtxTowerNumGlobalKvHeads);
  CHECK(c.raw["attention_k_eq_v"] == true);
  CHECK(c.raw["num_kv_shared_layers"] == 0);
  CHECK(c.raw["hidden_size_per_layer_input"] == 0);  // no PLE on this tower
  CHECK(c.raw["tie_word_embeddings"] == true);
  CHECK(c.raw["enable_moe_block"] == false);

  // `layer_types` is what decides which of the two geometries each layer gets,
  // and it must agree with the pattern the goldens were produced under.
  REQUIRE(static_cast<int64_t>(c.layer_types.size()) == vllm_test::kLtxTowerNumLayers);
  int64_t full_count = 0;
  for (int64_t l = 0; l < vllm_test::kLtxTowerNumLayers; ++l) {
    const bool want_full = vllm_test::kLtxTowerLayerIsFull[static_cast<size_t>(l)] != 0;
    CHECK((c.layer_types[static_cast<size_t>(l)] == "full_attention") == want_full);
    if (want_full) ++full_count;
  }
  // Both kinds are present, so neither branch can be dead in this fixture.
  CHECK(full_count == 2);
  CHECK(full_count < vllm_test::kLtxTowerNumLayers);

  // The nested per-layer-type rope: two thetas, and partial rotary on the full
  // arm only (the shipped config's `rope_parameters`, carried over unreduced).
  const nlohmann::json& rope = c.raw["rope_parameters"];
  CHECK(rope["full_attention"]["rope_type"] == "proportional");
  CHECK(rope["full_attention"]["rope_theta"] == 1000000.0);
  CHECK(rope["full_attention"]["partial_rotary_factor"] == 0.25);
  CHECK(rope["sliding_attention"]["rope_type"] == "default");
  CHECK(rope["sliding_attention"]["rope_theta"] == 10000.0);
}

TEST_CASE("gemma4 tower: EVERY hidden state matches the RUNNING upstream tower") {
  const vllm::HfConfig cfg = TowerConfig();
  const vllm::Gemma4Weights weights = TowerWeights(cfg);
  const int64_t T = vllm_test::kLtxTowerSeq;
  const int64_t H = vllm_test::kLtxTowerHidden;
  const int64_t S = vllm_test::kLtxTowerNumStates;

  std::vector<int32_t> tokens(static_cast<size_t>(T));
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    tokens[static_cast<size_t>(t)] = vllm_test::kLtxTowerTokens[static_cast<size_t>(t)];
    positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  TowerCachePool pool(cfg, /*num_blocks=*/4, /*block_size=*/8);
  const vllm::v1::CommonAttentionMetadata meta = PrefillMeta(T, 8);
  vt::Queue q = Qcpu();
  const vllm::Gemma4HiddenStatesResult got = vllm::Gemma4Model::ForwardHiddenStates(
      tokens, positions, meta, pool.attn_kv, weights, cfg, q);

  REQUIRE(static_cast<int64_t>(got.hidden_states.size()) == S);

  // THE TOLERANCE IS A MEASUREMENT, NOT A NUMBER SOMEBODY PICKED.
  //
  // Our stream carries bf16 and widens only on the way out (gemma4.h,
  // Gemma4HiddenStatesResult); bf16 is also upstream's own resolved model dtype
  // (base_encoder.py:41). So the dtype-MATCHED arm is bf16-vs-bf16, and the
  // question "is a difference a defect?" has a measurable answer: how much does
  // UPSTREAM's own answer move when the same upstream code runs at bf16 instead
  // of f32? That is `kLtxTowerDtypeNoise`, emitted by the generator from the two
  // oracle runs, and it is the smallest difference this comparison can resolve.
  //
  // A port inside it is indistinguishable from upstream at upstream's arithmetic
  // width. A port outside it has something bf16 rounding does not explain. And
  // the bound cannot be loosened to rescue a failure, because loosening it means
  // regenerating it, which means the oracle moved.
  //
  // The f32 arm is gated too, at the triangle-inequality bound (our distance to
  // the bf16 oracle plus the bf16 oracle's distance to the f32 one, so 2x the
  // floor). A bf16 store is exactly what absorbs a reduction-order defect, so the
  // wider arm is the one that would show one moving; reporting it without gating
  // it would be the "report-only is not a result" failure.
  double worst_bf16_ratio = 0.0;
  double worst_f32_ratio = 0.0;
  for (int64_t s = 0; s < S; ++s) {
    REQUIRE(static_cast<int64_t>(got.hidden_states[static_cast<size_t>(s)].size()) ==
            T * H);
    const double d_bf16 = MaxAbsDiff(got.hidden_states[static_cast<size_t>(s)],
                                     TowerGoldenBf16(s), static_cast<size_t>(T * H));
    const double d_f32 = MaxAbsDiff(got.hidden_states[static_cast<size_t>(s)],
                                    TowerGoldenF32(s), static_cast<size_t>(T * H));
    const double floor =
        static_cast<double>(vllm_test::kLtxTowerDtypeNoise[static_cast<size_t>(s)]);
    const double state_scale =
        static_cast<double>(vllm_test::kLtxTowerStateScale[static_cast<size_t>(s)]);
    REQUIRE(floor > 0.0);
    MESSAGE("gemma4 tower state "
            << static_cast<int>(s) << ": max|diff| vs bf16 oracle = " << d_bf16
            << " (" << (d_bf16 / floor) << "x the oracle's own f32-vs-bf16 floor "
            << floor << "), vs f32 oracle = " << d_f32 << ", over max|value| = "
            << state_scale);
    CHECK(d_bf16 <= floor);
    CHECK(d_f32 <= 2.0 * floor);
    worst_bf16_ratio = std::max(worst_bf16_ratio, d_bf16 / floor);
    worst_f32_ratio = std::max(worst_f32_ratio, d_f32 / (2.0 * floor));
  }
  MESSAGE("gemma4 tower: WORST over all "
          << static_cast<int>(S) << " states — bf16 arm reached "
          << worst_bf16_ratio << "x its floor, f32 arm " << worst_f32_ratio
          << "x its floor (1.0 = at the bound)");

  // The states are not all the same buffer pushed S times — the check every
  // other assertion here would still pass under.
  for (int64_t s = 0; s + 1 < S; ++s) {
    double gap = 0.0;
    for (size_t i = 0; i < got.hidden_states[static_cast<size_t>(s)].size(); ++i)
      gap = std::max(gap,
                     std::abs(static_cast<double>(
                                  got.hidden_states[static_cast<size_t>(s)][i]) -
                              static_cast<double>(
                                  got.hidden_states[static_cast<size_t>(s + 1)][i])));
    CHECK(gap > 0.0);
  }
}

TEST_CASE("gemma4 tower: LEFT PADDING is equivalent to running the valid tokens") {
  // What this buys, stated because it is the difference between a prompt costing
  // 1024 tower rows and costing its own length. Upstream pads EVERY prompt to
  // 1024 (gemma_assets.py:162, base_encoder.py:231-236) and runs all 1024 rows
  // through a 12B tower. Pads are masked out of attention and sit causally
  // BEFORE every valid token, so a valid row cannot depend on one; the feature
  // extractor then zeroes the pad rows anyway (feature_extractor.py:63-64).
  //
  // That is an argument, and an argument is not a measurement. Here the oracle
  // runs the FULL left-padded sequence and our short run's rows are held to its
  // VALID rows. If the equivalence is ever false — a mask that leaks, a position
  // that is derived from the mask rather than from the cache position, a sliding
  // window that counts pads — this is what says so.
  const vllm::HfConfig cfg = TowerConfig();
  const vllm::Gemma4Weights weights = TowerWeights(cfg);
  const int64_t T = vllm_test::kLtxTowerSeq;
  const int64_t H = vllm_test::kLtxTowerHidden;
  const int64_t S = vllm_test::kLtxTowerNumStates;
  const int64_t P = vllm_test::kLtxTowerNumPad;
  const int64_t PT = vllm_test::kLtxTowerPaddedSeq;

  // The pads are real pad ids in a real left-padded batch, so the fixture is the
  // layout the tokenizer produces rather than a convenient one.
  REQUIRE(P > 0);
  REQUIRE(P + T == PT);
  for (int64_t i = 0; i < P; ++i) {
    CHECK(vllm_test::kLtxTowerPaddedTokens[static_cast<size_t>(i)] ==
          vllm_test::kLtxTowerPadId);
    CHECK(vllm_test::kLtxTowerPaddedMask[static_cast<size_t>(i)] == 0);
  }

  // Positions are the ORIGINAL absolute ones — P..P+T-1, not 0..T-1. That is
  // what transformers does: with no explicit position_ids it uses the cache
  // position, which counts pad rows (modeling_gemma4_unified.py, the
  // `cache_position` default).
  //
  // This comment used to end "so a port that renumbers from zero after dropping
  // the pads rotates every query by the wrong angle", and that was an OVERCLAIM.
  // MEASURED in the oracle: the same 8 tokens told 12..19 and told 0..7 differ by
  // 5.11e-05 in f32 on values of magnitude 14.35 — 3.6e-06 relative, f32
  // round-off — because rotary position embedding is RELATIVE and the pads are
  // masked out of attention, so a uniform shift of every position cancels. At
  // bf16 the two differ by 1.09, which is 0.65-1.70x the per-state dtype floor,
  // i.e. the rounding of different absolute angles rather than a different
  // answer. So renumbering is a MIRRORING divergence, not a rotation error, and
  // no numeric gate in this suite separates the two cleanly. It is mirrored
  // because upstream does it; the reason is fidelity, not arithmetic.
  std::vector<int32_t> tokens(static_cast<size_t>(T));
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    tokens[static_cast<size_t>(t)] =
        vllm_test::kLtxTowerPaddedTokens[static_cast<size_t>(P + t)];
    positions[static_cast<size_t>(t)] = static_cast<int32_t>(P + t);
  }

  // Only the T VALID tokens are given to the tower, and their KV occupies slots
  // 0..T-1. The pads get no cache entry at all, which is the point: a run that
  // declared seq_len = P + T while writing only T keys would have attention read
  // P slots of zeroes and call them keys — the same wrong answer as attending to
  // the pads, arrived at by a different route. Measured, when this gate was
  // first written that way: max|diff| 17.97 against a max|value| of 14.35.
  TowerCachePool pool(cfg, /*num_blocks=*/4, /*block_size=*/8);
  const vllm::v1::CommonAttentionMetadata meta = PrefillMeta(T, 8);

  vt::Queue q = Qcpu();
  const vllm::Gemma4HiddenStatesResult got = vllm::Gemma4Model::ForwardHiddenStates(
      tokens, positions, meta, pool.attn_kv, weights, cfg, q);
  REQUIRE(static_cast<int64_t>(got.hidden_states.size()) == S);

  // CLAIM 1, and it is UPSTREAM'S, not ours: dropping the pads is free. The
  // generator measured it inside the oracle in f32 — the same valid tokens told
  // their absolute positions, run with and without the pads — so this number
  // carries no dtype noise at all. Asserting it here keeps the claim from
  // quietly becoming untrue if the oracle ever moves.
  double worst_oracle = 0.0;
  double scale = 0.0;
  for (int64_t s = 0; s < S; ++s) {
    worst_oracle = std::max(
        worst_oracle,
        static_cast<double>(vllm_test::kLtxTowerPadEquivalence[static_cast<size_t>(s)]));
    scale = std::max(scale,
                     static_cast<double>(
                         vllm_test::kLtxTowerStateScale[static_cast<size_t>(s)]));
  }
  MESSAGE("gemma4 tower: UPSTREAM's own padded-vs-short f32 spread = "
          << worst_oracle << " over max|value| = " << scale << " ("
          << (worst_oracle / scale) << " relative)");
  // f32 round-off on values of this magnitude, and nothing more.
  CHECK(worst_oracle < 1e-5 * scale);

  // CLAIM 2, and this one IS ours: our short run reproduces the padded run's
  // valid rows. Held to twice the dtype floor, because our stream is bf16 and
  // the padded golden is f32 — the same triangle bound the parity case uses.
  double worst = 0.0;
  double worst_ratio = 0.0;
  for (int64_t s = 0; s < S; ++s) {
    const std::vector<float>& mine = got.hidden_states[static_cast<size_t>(s)];
    REQUIRE(static_cast<int64_t>(mine.size()) == T * H);
    const float* want = TowerGoldenPadded(s);
    double d = 0.0;
    for (int64_t t = 0; t < T; ++t)
      for (int64_t h = 0; h < H; ++h) {
        const double w = static_cast<double>(want[static_cast<size_t>((P + t) * H + h)]);
        const double m = static_cast<double>(mine[static_cast<size_t>(t * H + h)]);
        d = std::max(d, std::abs(m - w));
      }
    const double floor =
        static_cast<double>(vllm_test::kLtxTowerDtypeNoise[static_cast<size_t>(s)]);
    REQUIRE(floor > 0.0);
    CHECK(d <= 2.0 * floor);
    worst = std::max(worst, d);
    worst_ratio = std::max(worst_ratio, d / (2.0 * floor));
  }
  MESSAGE("gemma4 tower left-pad equivalence: OUR max|diff| over the VALID rows of all "
          << static_cast<int>(S) << " states = " << worst << ", worst state reached "
          << worst_ratio << "x its bound");
}

TEST_CASE("gemma4 tower: partial_rotary_factor, on the ROPE TABLE the states cannot see") {
  // WHY THIS IS NOT GATED THROUGH THE HIDDEN STATES, stated as a measurement.
  //
  // `partial_rotary_factor` decides how many of a full-attention layer's angle
  // pairs are rotated and how many are zero-padded to identity. It is
  // config-carried and shape-invisible, and the case above — which does hold
  // every state to the oracle — CANNOT resolve it. Forcing the factor to 1.0 in
  // `MakeLayout` (gemma4.cpp) displaces the worst state by 1.09e-01 against that
  // state's measured bf16 noise floor of 9.99e-02: a ratio of 1.09, i.e. inside
  // the tolerance, and the whole suite stays green at 23 cases / 3583
  // assertions. Enlarging the fixture does not rescue it — MEASURED against this
  // oracle, worst signal-to-floor falls from 1.09 at the committed dims to 0.65
  // at (head_dim 16/32, seq 32) and 0.26 at (16/32, seq 8), because bf16
  // accumulation noise grows at least as fast as the rope contribution does.
  //
  // So the instrument is the TABLE, in f32, with no accumulation in it, against
  // the oracle's own `Gemma4UnifiedTextRotaryEmbedding` routed through
  // `_compute_proportional_rope_parameters` — the code that actually decides the
  // zero padding (modeling_rope_utils.py:187-254; the `torch.zeros` pad is :246).
  //
  // SCOPE, so the instrument is not read as covering more than it does: this is
  // the FULL-ATTENTION table only (`kLtxTowerGlobalHeadDim`, the `proportional`
  // arm at theta 1e6 — the one `partial_rotary_factor` applies to). The SLIDING
  // layers' `default` rope at theta 1e4 has NO equivalent f32 instrument; it is
  // reached only through the hidden states, where the same bf16 noise-floor
  // argument above works against resolving a config-carried angle defect.
  const vllm::HfConfig cfg = TowerConfig();
  const int64_t Dh = vllm_test::kLtxTowerGlobalHeadDim;
  const int64_t P = vllm_test::kLtxTowerRopePositions;
  const int64_t pairs = Dh / 2;

  const std::vector<float> got = vllm::Gemma4ProportionalRopeCosSin(cfg, Dh, P - 1);
  REQUIRE(static_cast<int64_t>(got.size()) == P * Dh);

  double worst = 0.0;
  for (int64_t i = 0; i < P * Dh; ++i)
    worst = std::max(worst, std::abs(static_cast<double>(got[static_cast<size_t>(i)]) -
                                     static_cast<double>(
                                         vllm_test::kLtxTowerRopeFullCosSin[
                                             static_cast<size_t>(i)])));
  MESSAGE("gemma4 tower rope: max|ours - oracle| over "
          << static_cast<int>(P) << " x " << static_cast<int>(Dh) << " = " << worst);
  // Both sides evaluate cos/sin once and store f32. Nothing accumulates, so the
  // only difference admissible here is the last-bit disagreement between two
  // libm implementations of cos/sin — NOT a tolerance that a rotated pair could
  // hide inside. A pair that should be identity and is not moves this by up to
  // 1.0, five orders of magnitude above the bound.
  CHECK(worst <= 1e-6);

  // The STRUCTURE, exactly, because it is the thing the factor controls and an
  // exact check cannot be absorbed by any noise argument. `rope_angles` is
  // upstream's `int(partial * head_dim // 2)` (modeling_rope_utils.py:234).
  const int64_t rotated = static_cast<int64_t>(0.25 * static_cast<double>(Dh)) / 2;
  REQUIRE(rotated == 2);
  REQUIRE(rotated < pairs);
  for (int64_t t = 0; t < P; ++t) {
    for (int64_t j = rotated; j < pairs; ++j) {
      CHECK(got[static_cast<size_t>(t * Dh + j)] == 1.0f);           // cos
      CHECK(got[static_cast<size_t>(t * Dh + pairs + j)] == 0.0f);   // sin
    }
  }
  // ...and the pairs BELOW the cut really do rotate, so "every pair is identity"
  // — a factor read as 0 — is not what just passed.
  double moved = 0.0;
  for (int64_t t = 0; t < P; ++t)
    for (int64_t j = 0; j < rotated; ++j)
      moved = std::max(moved,
                       std::abs(static_cast<double>(got[static_cast<size_t>(t * Dh + pairs + j)])));
  CHECK(moved > 0.5);
}

// ─────────────── the LOADER, and the whole prompt path, in CI ────────────────

TEST_CASE("ltx2 tower loader: the reduced tower READ BACK OFF A FILE") {
  const vllm::HfConfig cfg = TowerConfig();
  const nlohmann::json gemma_config =
      nlohmann::json::parse(vllm_test::kLtxTowerTextConfigJson);
  const fs::path dir = fs::temp_directory_path() / "vllm_ltx2_tower_loader";
  fs::create_directories(dir);
  const std::string path =
      WriteTypedPack(dir / "tower.safetensors", TowerPackTensors(cfg), "");

  SUBCASE("every layer matches the fixture the ORACLE ran, byte for byte") {
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    const vllm::Ltx2GemmaTower tower =
        vllm::Ltx2LoadGemmaTowerFromSafetensors(file, gemma_config);
    const vllm::Gemma4Weights want = TowerWeights(cfg);

    CHECK(tower.config.hidden_size == vllm_test::kLtxTowerHidden);
    CHECK(tower.config.num_hidden_layers == vllm_test::kLtxTowerNumLayers);
    // Everything here is bf16 already, so nothing was dequantized. The NVFP4 arm
    // is the shipped checkpoint's and stays on the opt-in case.
    CHECK(tower.dequantized_modules.empty());
    REQUIRE(tower.weights.layers.size() == want.layers.size());
    CHECK(SameTensor(tower.weights.embed_tokens, want.embed_tokens));
    CHECK(SameTensor(tower.weights.final_norm, want.final_norm));
    for (size_t l = 0; l < want.layers.size(); ++l) {
      const vllm::Gemma4LayerWeights& g = tower.weights.layers[l];
      const vllm::Gemma4LayerWeights& w = want.layers[l];
      CHECK(g.is_full_attention == w.is_full_attention);
      CHECK(g.head_dim == w.head_dim);
      CHECK(g.num_kv_heads == w.num_kv_heads);
      // The full layers ship no `v_proj`; V aliases K. That is a 1-vs-2 kv-head,
      // 16-vs-8 head_dim difference that every shape check downstream passes.
      CHECK(g.k_eq_v == w.k_eq_v);
      CHECK(SameTensor(g.attn.qkv_proj, w.attn.qkv_proj));
      CHECK(SameTensor(g.attn.o_proj, w.attn.o_proj));
      CHECK(SameTensor(g.attn.q_norm, w.attn.q_norm));
      CHECK(SameTensor(g.attn.k_norm, w.attn.k_norm));
      CHECK(SameTensor(g.mlp.gate_up_proj, w.mlp.gate_up_proj));
      CHECK(SameTensor(g.mlp.down_proj, w.mlp.down_proj));
      CHECK(SameTensor(g.input_layernorm, w.input_layernorm));
      CHECK(SameTensor(g.post_attention_layernorm, w.post_attention_layernorm));
      CHECK(SameTensor(g.pre_feedforward_layernorm, w.pre_feedforward_layernorm));
      CHECK(SameTensor(g.post_feedforward_layernorm, w.post_feedforward_layernorm));
      CHECK(SameTensor(g.layer_scalar, w.layer_scalar));
    }
  }

  SUBCASE("a PLE tower is REFUSED, not loaded without its PLE tensors") {
    nlohmann::json bad = gemma_config;
    bad["hidden_size_per_layer_input"] = 8;
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaTowerFromSafetensors(file, bad),
                    std::runtime_error);
  }

  SUBCASE("num_kv_shared_layers != 0 is REFUSED, not silently unshared") {
    nlohmann::json bad = gemma_config;
    bad["num_kv_shared_layers"] = 2;
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaTowerFromSafetensors(file, bad),
                    std::runtime_error);
  }

  SUBCASE("a missing v_proj without attention_k_eq_v is REFUSED, not aliased") {
    // The pack's full layers carry no `v_proj` — that is the real checkpoint's
    // shape. Withdrawing the DECLARATION must not turn a missing tensor into a
    // deliberate K-aliased layer.
    nlohmann::json bad = gemma_config;
    bad["attention_k_eq_v"] = false;
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaTowerFromSafetensors(file, bad),
                    std::runtime_error);
  }

  SUBCASE("layer_types that does not cover every layer is REFUSED") {
    nlohmann::json bad = gemma_config;
    bad["layer_types"] = nlohmann::json::array({"full_attention", "sliding_attention"});
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaTowerFromSafetensors(file, bad),
                    std::runtime_error);
  }

  SUBCASE("a module in NEITHER the bf16 nor the torchao-NVFP4 form is REFUSED") {
    // F16 parses as a tensor and has the right element count, so a loader that
    // read whichever of the two forms happened to fit would build a tower out of
    // reinterpreted bytes. The refusal names the module.
    std::vector<PackTensor> ts = TowerPackTensors(cfg);
    int hits = 0;
    for (PackTensor& t : ts) {
      if (t.name == "model.layers.0.self_attn.q_proj.weight") {
        t.dtype = "F16";
        ++hits;
      }
    }
    REQUIRE(hits == 1);
    const std::string bad_path =
        WriteTypedPack(dir / "tower_f16.safetensors", ts, "");
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(bad_path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaTowerFromSafetensors(file, gemma_config),
                    std::runtime_error);
  }

  SUBCASE("a NON-bf16 norm vector is REFUSED rather than reinterpreted") {
    // F16 is the same WIDTH as bf16, so the file parses and the element count is
    // right; only the exponent layout differs. Nothing downstream of a norm
    // weight read that way would look wrong — it would just scale every
    // activation by a different number.
    std::vector<PackTensor> ts = TowerPackTensors(cfg);
    int hits = 0;
    for (PackTensor& t : ts) {
      if (t.name == "model.norm.weight") {
        t.dtype = "F16";
        ++hits;
      }
    }
    REQUIRE(hits == 1);
    const std::string bad_path =
        WriteTypedPack(dir / "tower_f16norm.safetensors", ts, "");
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(bad_path);
    CHECK_THROWS_AS(vllm::Ltx2LoadGemmaTowerFromSafetensors(file, gemma_config),
                    std::runtime_error);
  }
}

TEST_CASE("ltx2 prompt -> conditioning: the VALUES, against the left-padded oracle") {
  // THE HEADLINE PATH, AND UNTIL NOW ITS ONLY GATE WAS OPT-IN.
  //
  // `Ltx2EncodePromptToConditioning` is what the row exists to ship, and the
  // only case that called it needed a 24 GB checkpoint. Everything it does that
  // nothing else does — tokenize, drop the pads, run the tower at the surviving
  // tokens' own positions, scatter the rows back to the full padded width, and
  // hand the stack to the extractor — was therefore gated on nothing but shapes,
  // isfinite, max|v| > 1e-3, sort_index and "two prompts differ".
  //
  // What makes a VALUE gate possible in CI is that the committed left-padded
  // oracle run is a run of this exact situation: 12 pad rows then 8 valid tokens
  // at absolute positions 12..19. Give the path a tokenizer that produces those
  // 8 tokens and the oracle's own rows are the answer it owes.
  //
  // ONE THING THIS DOES NOT SHARPLY GATE, said plainly rather than implied.
  // Renumbering the positions from zero (`positions[i] = i`) does red this case,
  // but only at 1.10x the audio floor — and that is a property of the DEFECT, not
  // a weakness here: rotary embedding is relative and the pads are masked, so
  // upstream's own f32 answers for 12..19 and 0..7 agree to 3.6e-06 relative.
  // The numbering is mirrored for fidelity to transformers, and the note at the
  // left-pad case above carries the measurement.
  const vllm::HfConfig cfg = TowerConfig();
  const nlohmann::json gemma_config =
      nlohmann::json::parse(vllm_test::kLtxTowerTextConfigJson);
  const fs::path dir = fs::temp_directory_path() / "vllm_ltx2_tower_prompt";
  fs::create_directories(dir);
  const std::string path =
      WriteTypedPack(dir / "tower.safetensors", TowerPackTensors(cfg), "");
  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
  const vllm::Ltx2GemmaTower tower =
      vllm::Ltx2LoadGemmaTowerFromSafetensors(file, gemma_config);

  const int64_t T = vllm_test::kLtxTowerSeq;
  const int64_t PT = vllm_test::kLtxTowerPaddedSeq;
  const int64_t P = vllm_test::kLtxTowerNumPad;
  const int64_t S = vllm_test::kLtxTowerNumStates;

  const vllm::tok::Tokenizer tokenizer =
      vllm::tok::Tokenizer::FromHfJsonBytes(TowerTokenizerJson(), "<tower fixture>");
  vllm::Ltx2GemmaSpecialIds ids;
  ids.bos_id = vllm_test::kLtxTowerTokens[0];
  ids.pad_id = static_cast<int32_t>(vllm_test::kLtxTowerPadId);

  // The prompt spells tokens 1..T-1; the BOS at index 0 is what the wrapper
  // prepends. Built from the golden rather than typed out, so it cannot drift.
  std::string prompt;
  for (int64_t i = 1; i < T; ++i)
    prompt += "<t" + std::to_string(vllm_test::kLtxTowerTokens[static_cast<size_t>(i)]) +
              ">";

  const vllm::Ltx2TextFeatureConfig fcfg = TowerFeatureConfig();
  const vllm::Ltx2TextEncoderWeights proj = TowerProjections();
  vt::Queue q = Qcpu();
  const vllm::Ltx2PromptConditioning out = vllm::Ltx2EncodePromptToConditioning(
      tower, tokenizer, ids, proj, fcfg, prompt, q, PT);

  // The tokenizer really did reproduce the oracle's batch: same ids, same pad
  // run, same mask. If this ever stops holding, the value comparison below is
  // comparing two different prompts and has to fail loudly here first.
  REQUIRE(out.seq == PT);
  REQUIRE(out.tokens.num_valid == T);
  REQUIRE(out.tokens.first_valid == P);
  for (int64_t i = 0; i < PT; ++i) {
    CHECK(out.tokens.input_ids[static_cast<size_t>(i)] ==
          vllm_test::kLtxTowerPaddedTokens[static_cast<size_t>(i)]);
    CHECK(out.mask[static_cast<size_t>(i)] ==
          vllm_test::kLtxTowerPaddedMask[static_cast<size_t>(i)]);
  }

  // The reference: the ORACLE's left-padded states through the SAME projection,
  // at both of the oracle's own arithmetic widths. The projection is shared with
  // the path under test — deliberately, because it is separately gated against
  // its own goldens above; what is NOT shared is the hidden states, which is
  // exactly the part `first_valid + i` decides.
  const std::vector<std::vector<float>> f32_states =
      OraclePaddedStates(&TowerGoldenPadded);
  const std::vector<std::vector<float>> bf16_states =
      OraclePaddedStates(&TowerGoldenPaddedBf16);
  auto as_states = [&](const std::vector<std::vector<float>>& bufs) {
    vllm::Ltx2TextHiddenStates st;
    for (const std::vector<float>& b : bufs) st.layers.push_back(b.data());
    st.batch = 1;
    st.seq = PT;
    st.hidden = vllm_test::kLtxTowerHidden;
    return st;
  };
  REQUIRE(static_cast<int64_t>(f32_states.size()) == S);
  const vllm::Ltx2TextConditioning want_f32 = vllm::Ltx2TextEncoderConditioning(
      as_states(f32_states), out.mask.data(), proj, fcfg);
  const vllm::Ltx2TextConditioning want_bf16 = vllm::Ltx2TextEncoderConditioning(
      as_states(bf16_states), out.mask.data(), proj, fcfg);

  // The floor, propagated rather than borrowed: it is the oracle's OWN
  // f32-vs-bf16 spread carried through the identical projection, so it is the
  // smallest difference this comparison can resolve. It cannot be widened to
  // rescue a failure without regenerating the oracle; what the two CHECKs below
  // carry is a derivation over it, stated where they stand.
  auto max_diff = [](const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
      d = std::max(d, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return d;
  };
  const double video_floor = max_diff(want_f32.video, want_bf16.video);
  const double audio_floor = max_diff(want_f32.audio, want_bf16.audio);
  REQUIRE(video_floor > 0.0);
  REQUIRE(audio_floor > 0.0);

  const double video_bf16 = max_diff(out.conditioning.video, want_bf16.video);
  const double video_f32 = max_diff(out.conditioning.video, want_f32.video);
  const double audio_bf16 = max_diff(out.conditioning.audio, want_bf16.audio);
  const double audio_f32 = max_diff(out.conditioning.audio, want_f32.audio);
  MESSAGE("ltx2 prompt->conditioning: video max|diff| vs bf16 oracle = "
          << video_bf16 << " (" << (video_bf16 / video_floor)
          << "x the propagated floor " << video_floor << "), vs f32 oracle = "
          << video_f32 << "; audio " << audio_bf16 << " ("
          << (audio_bf16 / audio_floor) << "x " << audio_floor << ") / " << audio_f32);
  // BOTH arms carry the SAME bound. The reasoning is the triangle step the
  // state-level parity case above spells out; the NUMBER comes from this case's
  // own measurements, cited below. `out.conditioning` is OUR bf16 realization,
  // `want_bf16` is the ORACLE's, and neither is a rounding of the other. Both depart from the shared f32 trajectory `want_f32`, so
  //
  //   |ours - oracle_bf16| <= |ours - oracle_f32| + |oracle_f32 - oracle_bf16|
  //
  // and the second term IS the propagated floor. The first term is `video_f32`,
  // which THIS case measures a line above: 0.79x the floor on video and 0.475x
  // on audio, so taking it at one floor is the measurement, not an assumption.
  // Hence 2x on the bf16 arm as well as on the f32 one.
  //
  // WHY NOT 3x, since the sibling assertion below gates that first term at 2x.
  // Chaining the two ASSERTED bounds would give 3x, and 3x is a bound that can
  // never fire: `|ours - oracle_bf16| <= |ours - oracle_f32| + floor` is the
  // triangle inequality itself, so once the f32 arm passes at 2x the bf16 arm at
  // 3x is arithmetic rather than a gate. 2x is the tighter value the measured
  // premise supports, and it is the one that can still say something.
  //
  // WHY IT USED TO SAY 1x, AND WHY THAT WAS NEVER DERIVED. The state-level case
  // gates `d_bf16 <= floor` and derives its f32 arm from that primitive. This
  // case IMPORTED the 1x across `Ltx2TextEncoderConditioning`, which stacks all
  // 13 states and combines them. A per-state relation that holds elementwise
  // does not survive that: the floor is the projection of ONE error vector and
  // this quantity is the projection of a DIFFERENT one, and the projection can
  // amplify the second relative to the first. It held until the seam's bf16
  // rounding polarity was corrected to upstream's in `4712dac40`, which re-rolled
  // both error directions and moved the video arm from 0.56x to 1.21x and the
  // audio arm from 0.69x to 1.31x — with our distance to the f32 oracle
  // IMPROVING at 11 of the 13 states. The old constant broke, not the port.
  // #1458.
  //
  // WHAT THIS COSTS, MEASURED rather than estimated, and it is not this bound
  // that spent it. The position-renumbering note at the top of this case claims
  // detection at 1.10x the audio floor. Ratios to the propagated floor, one
  // build directory, every source restored sha256-verified:
  //
  //   cpu_ops.cpp        code          video    audio
  //   before 4712dac40   correct       0.565    0.688   -> pass at 1.0x
  //   before 4712dac40   renumbered    0.831    1.099   -> RED at 1.0x
  //   at aeba0de6f       correct       1.209    1.313   -> RED at 1.0x
  //   at aeba0de6f       renumbered    0.683    0.931   -> pass at 1.0x
  //
  // Read the bottom two rows together: post-`4712dac40` the instrument is
  // INVERTED, reding the correct code and passing the mutant, and the mutant is
  // measurably CLOSER to the oracle than the port is. The 2x here restores a
  // functioning instrument; it does not recover that detection, and no constant
  // can, because the ordering of the two has reversed. It is also a property of
  // the defect — upstream's own f32 answers for positions 12..19 and 0..7 agree
  // to 3.6e-06 relative — and `gen-ltx2-gemma-tower-goldens.py:363-375` already
  // records that the end-to-end states are the wrong instrument for this class
  // and the f32 rope table is the right one. Owed as #1467.
  CHECK(video_bf16 <= 2.0 * video_floor);
  CHECK(video_f32 <= 2.0 * video_floor);
  CHECK(audio_bf16 <= 2.0 * audio_floor);
  CHECK(audio_f32 <= 2.0 * audio_floor);

  // The reorder and the mask, compared EXACTLY — but read what that does and does
  // NOT prove, because the label oversells it. `out.conditioning` and `want_f32`
  // reach `sort_index` / `additive_mask` through the SAME
  // `Ltx2ComputeRightPadOrder`, over the SAME `out.mask`, so a defect INSIDE that
  // function cancels and these assertions cannot fire. MEASURED: replacing its
  // stable partition with the identity permutation leaves this case entirely
  // green. What they DO establish is that the end-to-end path hands the mask
  // through unaltered — `out.mask` itself is held to the golden above — so the
  // conditioning is ordered consistently with the tokens this case just checked.
  // The ORDERING ITSELF is gated by the two cases that red on that same mutation,
  // both against committed upstream goldens rather than against a second call of
  // the code under test: "ltx2 text: additive mask, right-pad ordering and the
  // binary mask" and "ltx2 text: the encoder -> conditioning hand-off".
  REQUIRE(out.conditioning.sort_index.size() == want_f32.sort_index.size());
  for (size_t i = 0; i < want_f32.sort_index.size(); ++i)
    CHECK(out.conditioning.sort_index[i] == want_f32.sort_index[i]);
  REQUIRE(out.conditioning.additive_mask.size() == want_f32.additive_mask.size());
  for (size_t i = 0; i < want_f32.additive_mask.size(); ++i)
    CHECK(out.conditioning.additive_mask[i] == want_f32.additive_mask[i]);

  // And the conditioning is not a wall of zeros that every bound above would
  // also accept.
  double vmax = 0.0;
  for (float x : out.conditioning.video)
    vmax = std::max(vmax, std::abs(static_cast<double>(x)));
  CHECK(vmax > 1e-3);
}

// ──────────────────── a real prompt becomes real tokens ──────────────────────

namespace {

// The env-gated path to the shipped text encoder. The tokenizer it carries is a
// 32 MB TENSOR, so this cannot be a committed fixture; the GOLDENS are committed
// and the asset is opt-in.
std::string Ltx2TextEncoderPathOrEmpty() {
  const char* explicit_path = std::getenv("VLLM_CPP_LTX2_TEXT_ENCODER");
  if (explicit_path != nullptr && *explicit_path != '\0') return explicit_path;
  const char* root = std::getenv("CHECKPOINT_ROOT");
  if (root == nullptr || *root == '\0') return {};
  const fs::path p = fs::path(root) / "ltx-2.5" / "vonkaiser-fp8-nvfp4" /
                     "text_encoders" / "gemma4-12b-with-proj-nvfp4-torchao.safetensors";
  std::error_code ec;
  return fs::exists(p, ec) ? p.string() : std::string();
}

const int32_t* PromptIds(int64_t i) {
  static const int32_t* const kIds[] = {
      vllm_test::kLtxPromptIds_0, vllm_test::kLtxPromptIds_1,
      vllm_test::kLtxPromptIds_2, vllm_test::kLtxPromptIds_3};
  return kIds[static_cast<size_t>(i)];
}

const int32_t* PromptMask(int64_t i) {
  static const int32_t* const kMask[] = {
      vllm_test::kLtxPromptMask_0, vllm_test::kLtxPromptMask_1,
      vllm_test::kLtxPromptMask_2, vllm_test::kLtxPromptMask_3};
  return kMask[static_cast<size_t>(i)];
}

}  // namespace

TEST_CASE("ltx2 prompt: the tokenizer WRAPPER, on a fixture that needs no asset") {
  // A tiny SentencePiece-flavoured tokenizer.json in the shipped tokenizer's own
  // form — `Replace(" " -> U+2581)` normalizer plus `Split(" ",
  // MergedWithPrevious)` pre-tokenizer — so the wrapper's behaviour is gated in
  // CI without the 32 MB tensor. What is checked here is the WRAPPER
  // (tokenizer.py:31-59); the real vocab is the next case.
  const std::string tokenizer_json = R"JSON({
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

  const vllm::tok::Tokenizer tok =
      vllm::tok::Tokenizer::FromHfJsonBytes(tokenizer_json, "<inline fixture>");
  const int32_t kBos = 2, kPad = 0;
  const int64_t kMax = 8;

  SUBCASE("the BOS is PREPENDED and the padding is on the LEFT") {
    const vllm::Ltx2GemmaPromptTokens t =
        vllm::Ltx2TokenizeGemmaPrompt(tok, "a b c", kBos, kPad, kMax);
    REQUIRE(static_cast<int64_t>(t.input_ids.size()) == kMax);
    REQUIRE(static_cast<int64_t>(t.attention_mask.size()) == kMax);
    // LEFT padding (base_encoder.py:235): the valid run is the TAIL, and its
    // first index is the pad count — which is why the caller cannot assume the
    // prompt starts at position 0.
    CHECK(t.first_valid == kMax - t.num_valid);
    CHECK(t.num_valid > 1);
    CHECK(t.input_ids[static_cast<size_t>(t.first_valid)] == kBos);
    CHECK_FALSE(t.truncated);
    for (int64_t i = 0; i < kMax; ++i) {
      const bool valid = i >= t.first_valid;
      CHECK(t.attention_mask[static_cast<size_t>(i)] == (valid ? 1 : 0));
      if (!valid) CHECK(t.input_ids[static_cast<size_t>(i)] == kPad);
    }
  }

  SUBCASE("the prompt is STRIPPED before tokenizing") {
    // tokenizer.py:33, and diffusers strips too (pipeline_ltx2.py:333). Without
    // it the leading run becomes real metaspace tokens and the prompt differs.
    const vllm::Ltx2GemmaPromptTokens bare =
        vllm::Ltx2TokenizeGemmaPrompt(tok, "a b", kBos, kPad, kMax);
    const vllm::Ltx2GemmaPromptTokens padded =
        vllm::Ltx2TokenizeGemmaPrompt(tok, "  \n a b \t ", kBos, kPad, kMax);
    CHECK(bare.num_valid == padded.num_valid);
    CHECK(bare.input_ids == padded.input_ids);
    CHECK(bare.attention_mask == padded.attention_mask);
  }

  SUBCASE("an EMPTY prompt still carries a BOS, and exactly one token") {
    const vllm::Ltx2GemmaPromptTokens t =
        vllm::Ltx2TokenizeGemmaPrompt(tok, "   ", kBos, kPad, kMax);
    CHECK(t.num_valid == 1);
    CHECK(t.input_ids[static_cast<size_t>(kMax - 1)] == kBos);
    CHECK(t.attention_mask[static_cast<size_t>(kMax - 1)] == 1);
  }

  SUBCASE("a BOS already present is NOT doubled") {
    // Upstream's guard is `if not input_ids or input_ids[0] != bos_id`
    // (tokenizer.py:44), i.e. conditional. A port that prepends unconditionally
    // produces two BOS and drops the prompt's last token to fit.
    const vllm::Ltx2GemmaPromptTokens t =
        vllm::Ltx2TokenizeGemmaPrompt(tok, "<bos> a b", kBos, kPad, kMax);
    REQUIRE(t.num_valid >= 2);
    CHECK(t.input_ids[static_cast<size_t>(t.first_valid)] == kBos);
    CHECK(t.input_ids[static_cast<size_t>(t.first_valid + 1)] != kBos);
  }

  SUBCASE("truncation costs the LAST token, never the BOS") {
    // tokenizer.py:39-46 truncates, prepends, then re-slices, so the BOS wins.
    const int64_t tiny = 3;
    const vllm::Ltx2GemmaPromptTokens t =
        vllm::Ltx2TokenizeGemmaPrompt(tok, "a b c a b c a b c", kBos, kPad, tiny);
    REQUIRE(static_cast<int64_t>(t.input_ids.size()) == tiny);
    CHECK(t.num_valid == tiny);
    CHECK(t.first_valid == 0);
    CHECK(t.input_ids[0] == kBos);
    CHECK(t.truncated);
    for (int64_t i = 0; i < tiny; ++i)
      CHECK(t.attention_mask[static_cast<size_t>(i)] == 1);
  }

  SUBCASE("RIGHT padding puts the valid run at the FRONT") {
    const vllm::Ltx2GemmaPromptTokens t = vllm::Ltx2TokenizeGemmaPrompt(
        tok, "a b", kBos, kPad, kMax, vllm::Ltx2GemmaPaddingSide::kRight);
    CHECK(t.first_valid == 0);
    CHECK(t.input_ids[0] == kBos);
    CHECK(t.attention_mask[static_cast<size_t>(kMax - 1)] == 0);
  }

  SUBCASE("a missing BOS id is REFUSED, not defaulted") {
    CHECK_THROWS_AS(vllm::Ltx2TokenizeGemmaPrompt(tok, "a b", -1, kPad, kMax),
                    std::runtime_error);
  }
}

TEST_CASE("ltx2 prompt: REAL prompts through the SHIPPED 262144-entry vocab") {
  const std::string path = Ltx2TextEncoderPathOrEmpty();
  if (path.empty()) {
    MESSAGE(
        "SKIPPED: set CHECKPOINT_ROOT (or VLLM_CPP_LTX2_TEXT_ENCODER) to gate the "
        "real vocab. The tokenizer ships as a 32 MB TENSOR inside the checkpoint, "
        "so it cannot be a committed fixture; the goldens beside this test are.");
    return;
  }

  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
  // require_config=false: the shipped vonkaiser build has NO __metadata__ at all,
  // which is exactly the case ltx2_text_encoder.h:366-373 records.
  const vllm::Ltx2GemmaAssets assets = vllm::Ltx2LoadGemmaAssets(file, false);
  CHECK_FALSE(assets.has_config);
  REQUIRE(!assets.tokenizer_json.empty());

  const vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromHfJsonBytes(
      std::string_view(reinterpret_cast<const char*>(assets.tokenizer_json.data()),
                       assets.tokenizer_json.size()),
      path + "::tokenizer_json");

  // The ids come from the checkpoint's own generation_config, not from a
  // hardcoded pair, and they must be the ones the goldens were built with.
  const vllm::Ltx2GemmaSpecialIds ids = vllm::Ltx2ResolveGemmaSpecialIds(assets, tok);
  CHECK(ids.bos_id == vllm_test::kLtxPromptBosId);
  CHECK(ids.pad_id == vllm_test::kLtxPromptPadId);

  for (int64_t p = 0; p < vllm_test::kLtxPromptCount; ++p) {
    const vllm::Ltx2GemmaPromptTokens t = vllm::Ltx2TokenizeGemmaPrompt(
        tok, vllm_test::kLtxPromptText[p], ids.bos_id, ids.pad_id,
        vllm_test::kLtxPromptMaxLength);
    REQUIRE(static_cast<int64_t>(t.input_ids.size()) ==
            vllm_test::kLtxPromptMaxLength);
    const int64_t want_valid = vllm_test::kLtxPromptValidCount[static_cast<size_t>(p)];
    // The count first, because a count mismatch makes every id comparison
    // meaningless and its message is the readable one.
    CHECK(t.num_valid == want_valid);
    int64_t first_bad = -1;
    for (int64_t i = 0; i < vllm_test::kLtxPromptMaxLength && first_bad < 0; ++i) {
      if (t.input_ids[static_cast<size_t>(i)] !=
              PromptIds(p)[static_cast<size_t>(i)] ||
          t.attention_mask[static_cast<size_t>(i)] !=
              PromptMask(p)[static_cast<size_t>(i)])
        first_bad = i;
    }
    MESSAGE("ltx2 prompt " << static_cast<int>(p) << " ("
                           << static_cast<int>(want_valid)
                           << " valid tokens): first mismatching index = "
                           << static_cast<int>(first_bad) << " (-1 = EXACT)");
    // Token ids are EXACT or they are wrong; there is no tolerance here.
    CHECK(first_bad == -1);
  }
}

// ────────────── a real prompt, the real 12B tower, real conditioning ─────────
//
// The whole point of phase L10, and the one case that cannot be run from a
// fixture: 7.4 GB of NVFP4 off the NAS, dequantized to ~24 GB of bf16, a real
// English sentence tokenized with the embedded 262144-entry vocab, 49 hidden
// states, both caption projections, and conditioning the DiT's two streams
// accept. Env-gated on the checkpoint and on an explicit opt-in, because it
// wants ~33 GB of host memory and minutes of CPU — a gate nobody can run by
// accident is worse than useless, and one that runs by accident in CI is worse
// still.

TEST_CASE("ltx2 e2e: a PROMPT drives the real Gemma-4 12B tower to conditioning") {
  const std::string path = Ltx2TextEncoderPathOrEmpty();
  if (path.empty() || std::getenv("VLLM_CPP_LTX2_TOWER_E2E") == nullptr) {
    MESSAGE(
        "SKIPPED: needs CHECKPOINT_ROOT (or VLLM_CPP_LTX2_TEXT_ENCODER) AND "
        "VLLM_CPP_LTX2_TOWER_E2E=1. It dequantizes a 12B NVFP4 tower to ~24 GB of "
        "host bf16 and runs it, so it is opt-in rather than checkpoint-presence "
        "gated.");
    return;
  }

  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);

  // 1. THE ASSETS. `require_config=false` because the shipped vonkaiser build
  //    has no `__metadata__` at all — upstream's own `from_single_file` raises
  //    here (gemma_assets.py:110-114), which is precisely why the Gemma config
  //    has to be supplied out of band.
  const vllm::Ltx2GemmaAssets assets = vllm::Ltx2LoadGemmaAssets(file, false);
  CHECK_FALSE(assets.has_config);

  const vllm::tok::Tokenizer tokenizer = vllm::tok::Tokenizer::FromHfJsonBytes(
      std::string_view(reinterpret_cast<const char*>(assets.tokenizer_json.data()),
                       assets.tokenizer_json.size()),
      path + "::tokenizer_json");
  const vllm::Ltx2GemmaSpecialIds ids =
      vllm::Ltx2ResolveGemmaSpecialIds(assets, tokenizer);

  // 2. THE CONFIG, out of band and committed with its provenance: it is the
  //    `__metadata__["gemma_config"]` of the OFFICIAL bf16 text encoder.
  const fs::path config_path =
      fs::path(__FILE__).parent_path() / "ltx2_gemma4_text_config.json";
  std::ifstream cfg_in(config_path);
  REQUIRE(cfg_in.good());
  const nlohmann::json gemma_config = nlohmann::json::parse(cfg_in);
  CHECK(gemma_config["gemma_version"] == "gemma4-12b-ltx-v1");

  // 3. THE TOWER. This is the ~24 GB step.
  const vllm::Ltx2GemmaTower tower =
      vllm::Ltx2LoadGemmaTowerFromSafetensors(file, gemma_config);
  CHECK(tower.config.hidden_size == 3840);
  CHECK(tower.config.num_hidden_layers == 48);
  CHECK(tower.config.vocab_size == 262144);
  CHECK(static_cast<int64_t>(tower.weights.layers.size()) == 48);
  // Every one of the 329 quantized modules on the text path arrived NVFP4 and
  // was dequantized — a tower that silently found bf16 tensors instead would be
  // a different checkpoint. The count is arithmetic, not a memory: the 40
  // sliding layers carry q, k, v, o, gate, up, down; the 8 full ones carry the
  // same minus `v_proj`, which they do not ship; and `embed_tokens` is the one
  // module outside the layers. 40*7 + 8*6 + 1 = 329.
  //
  // PINNED exactly rather than `> 300`. A lower bound is satisfied by a loader
  // that quietly took, say, 40 of the 48 layers' modules from a bf16 fallback,
  // which is the failure the line is here to catch.
  MESSAGE("ltx2 e2e: dequantized " << tower.dequantized_modules.size()
                                   << " torchao-NVFP4 modules");
  CHECK(tower.dequantized_modules.size() == 40 * 7 + 8 * 6 + 1);

  // The mixed geometry, READ BACK off what was actually built rather than off
  // the config that asked for it.
  int64_t full_layers = 0;
  for (const vllm::Gemma4LayerWeights& lw : tower.weights.layers) {
    if (lw.is_full_attention) {
      ++full_layers;
      CHECK(lw.head_dim == 512);
      CHECK(lw.num_kv_heads == 1);
      CHECK(lw.k_eq_v);  // no v_proj: V aliases K
      CHECK(lw.attn.qkv_proj.shape[0] == 16 * 512 + 2 * 1 * 512);
    } else {
      CHECK(lw.head_dim == 256);
      CHECK(lw.num_kv_heads == 8);
      CHECK_FALSE(lw.k_eq_v);
      CHECK(lw.attn.qkv_proj.shape[0] == 16 * 256 + 2 * 8 * 256);
    }
    CHECK(lw.attn.qkv_proj.shape[1] == 3840);
    CHECK(lw.mlp.gate_up_proj.shape[0] == 2 * 15360);
  }
  CHECK(full_layers == 8);

  // 4. THE CAPTION PROJECTIONS and the V1/V2 selection.
  const vllm::Ltx2TextEncoderCheckpoint te =
      vllm::Ltx2LoadTextEncoderFromSafetensors(file);
  CHECK(te.gemma_hidden_size == 3840);
  CHECK(te.gemma_num_hidden_layers == 48);
  CHECK(te.video.out_features == 4096);
  CHECK(te.audio.out_features == 2048);
  // 188160 = 3840 x 49 — the width §1.4 had to correct once, confirmed here off
  // the file rather than off the spec.
  CHECK(te.video.in_features == 3840 * 49);
  CHECK(te.audio.in_features == 3840 * 49);

  vllm::Ltx2TextFeatureConfig fcfg;
  fcfg.variant = vllm::Ltx2TextNormVariant::kPerTokenRmsV2;
  fcfg.embedding_dim = te.gemma_hidden_size;
  fcfg.num_layers = te.gemma_num_hidden_layers + 1;
  fcfg.video_out_features = te.video.out_features;
  fcfg.audio_out_features = te.audio.out_features;
  fcfg.aggregate_bias = true;  // V2: both Linears carry one (encoder_configurator.py:206-208)
  const vllm::Ltx2TextEncoderWeights proj = vllm::Ltx2WidenTextProjectionsToF32(te);
  REQUIRE(!proj.video.bias.empty());
  REQUIRE(!proj.audio.bias.empty());

  // 5. THE PROMPT.
  const std::string prompt = "a red fox running through deep snow at sunrise";
  vt::Queue q = Qcpu();
  const vllm::Ltx2PromptConditioning out = vllm::Ltx2EncodePromptToConditioning(
      tower, tokenizer, ids, proj, fcfg, prompt, q);

  // The tokens: the same 10 the HF oracle produced for this exact prompt.
  CHECK(out.tokens.num_valid == vllm_test::kLtxPromptValidCount[0]);
  CHECK(out.seq == vllm_test::kLtxPromptMaxLength);
  for (int64_t i = 0; i < out.seq; ++i)
    CHECK(out.tokens.input_ids[static_cast<size_t>(i)] ==
          vllm_test::kLtxPromptIds_0[static_cast<size_t>(i)]);

  // The conditioning: two streams at the two widths the DiT cross-attends over.
  REQUIRE(out.conditioning.video.size() ==
          static_cast<size_t>(out.seq * fcfg.video_out_features));
  REQUIRE(out.conditioning.audio.size() ==
          static_cast<size_t>(out.seq * fcfg.audio_out_features));
  REQUIRE(out.conditioning.additive_mask.size() == static_cast<size_t>(out.seq));
  REQUIRE(out.conditioning.sort_index.size() == static_cast<size_t>(out.seq));

  double vmax = 0.0, amax = 0.0, vsum = 0.0;
  for (float x : out.conditioning.video) {
    REQUIRE(std::isfinite(x));
    vmax = std::max(vmax, std::abs(static_cast<double>(x)));
    vsum += std::abs(static_cast<double>(x));
  }
  for (float x : out.conditioning.audio) {
    REQUIRE(std::isfinite(x));
    amax = std::max(amax, std::abs(static_cast<double>(x)));
  }
  MESSAGE("ltx2 e2e: prompt \"" << prompt << "\" -> " << out.tokens.num_valid
                                << " tokens -> video ["
                                << out.seq << ", " << fcfg.video_out_features
                                << "] max|v| = " << vmax << " mean|v| = "
                                << (vsum / static_cast<double>(out.conditioning.video.size()))
                                << ", audio [" << out.seq << ", "
                                << fcfg.audio_out_features << "] max|v| = " << amax);
  // VALUES, not just isfinite: a conditioning tensor that reduced to zeros would
  // satisfy every shape and finiteness check above and render an unconditioned
  // clip. §7.0(3) — a golden reduced to `isfinite` once hid a 23842x error.
  CHECK(vmax > 1e-3);
  CHECK(amax > 1e-3);

  // The RIGHT-PAD REORDER put the valid tokens at the FRONT. Upstream's
  // connector assumes right-padded input (embeddings_processor.py:82-92) while
  // the tokenizer LEFT-pads, so this reordering is load-bearing: without it every
  // valid token would sit behind 1014 pad rows.
  for (int64_t i = 0; i < out.tokens.num_valid; ++i)
    CHECK(out.conditioning.sort_index[static_cast<size_t>(i)] ==
          static_cast<int32_t>(out.tokens.first_valid + i));

  // A DIFFERENT prompt must produce DIFFERENT conditioning. Without this the
  // whole case would pass for a tower that ignored its input entirely — which is
  // exactly the failure "the model advertises text-to-video and we cannot type
  // text" would become if it were fixed carelessly.
  const vllm::Ltx2PromptConditioning other = vllm::Ltx2EncodePromptToConditioning(
      tower, tokenizer, ids, proj, fcfg, "a blue whale diving under thick pack ice",
      q, vllm_test::kLtxPromptMaxLength);
  REQUIRE(other.conditioning.video.size() == out.conditioning.video.size());
  double gap = 0.0;
  for (size_t i = 0; i < out.conditioning.video.size(); ++i)
    gap = std::max(gap, std::abs(static_cast<double>(out.conditioning.video[i]) -
                                 static_cast<double>(other.conditioning.video[i])));
  MESSAGE("ltx2 e2e: two different prompts differ by max|diff| = " << gap);
  CHECK(gap > 1e-3);
}
