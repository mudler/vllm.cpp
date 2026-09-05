// N1b-close (task #230): run-gate for the NVFP4 Laguna safetensors loader
// (LoadLagunaForCausalLMWeights). Builds a tiny synthetic 2-layer NVFP4 checkpoint
// (L0 dense, L1 MoE with 2 W4A4 experts) matching the verified
// poolside/Laguna-S-2.1-NVFP4 name-map + dtypes, loads it through the REAL loader,
// and asserts every field round-trips byte-identically. RED-first: a wrong
// scale-global reciprocal and a missing tensor both fail.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/laguna.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits carrier
#include "vllm/v1/attention/backend.h"

namespace vllm {
void StageLagunaGraphEmbedding(const OwnedTensor& embed, int32_t token,
                               int64_t hidden_size, int64_t vocab_size,
                               float* destination);
}  // namespace vllm

using vllm::HfConfig;
using vllm::LagunaWeights;
using vllm::LoadLagunaForCausalLMWeights;
using vllm::SafetensorsFile;

namespace {

// ── synthetic safetensors builder ──────────────────────────────────────────
struct Fx {
  std::string name;
  std::string dtype;                 // "BF16" / "F32" / "U8" / "F8_E4M3"
  std::vector<int64_t> shape;
  std::string bytes;
};

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

// Deterministic filler: byte b at index i = (seed*31 + i) & 0xff.
std::string Fill(size_t n, int seed) {
  std::string s(n, '\0');
  for (size_t i = 0; i < n; ++i) s[i] = static_cast<char>((seed * 31 + static_cast<int>(i)) & 0xff);
  return s;
}
// f32 scalar as 4 raw bytes.
std::string F32Bytes(float v) {
  std::string s(4, '\0');
  std::memcpy(s.data(), &v, 4);
  return s;
}

std::string BuildSt(const std::vector<Fx>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Fx& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {{"dtype", t.dtype},
                   {"shape", t.shape},
                   {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  return U64Le(header.size()) + header + data;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("laguna_nvfp4_loader_" + std::to_string(counter++) + ".safetensors"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// ── tiny model geometry ─────────────────────────────────────────────────────
constexpr int H = 32, Dh = 16, NH = 2, KV = 1, DENSE_I = 64, MOE_I = 16, V = 8, E = 2;

HfConfig TinyConfig() {
  HfConfig c;
  c.architectures = {"LagunaForCausalLM"};
  c.model_type = "laguna";
  c.hidden_size = H;
  c.num_hidden_layers = 2;
  c.vocab_size = V;
  c.num_attention_heads = NH;
  c.num_key_value_heads = KV;
  c.head_dim = Dh;
  c.intermediate_size = DENSE_I;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 4096;
  c.sliding_window = 8;
  c.num_experts = E;
  c.num_experts_per_tok = 1;
  c.moe_intermediate_size = MOE_I;
  nlohmann::json layer_types = nlohmann::json::array();
  nlohmann::json heads = nlohmann::json::array();
  for (int l = 0; l < 2; ++l) {
    const bool global = (l == 0);
    layer_types.push_back(global ? "full_attention" : "sliding_attention");
    heads.push_back(NH);
    c.layer_types.push_back(global ? "full_attention" : "sliding_attention");
  }
  c.raw = {
      {"hidden_size", H}, {"num_hidden_layers", 2}, {"vocab_size", V},
      {"num_attention_heads", NH}, {"num_key_value_heads", KV}, {"head_dim", Dh},
      {"intermediate_size", DENSE_I}, {"rms_norm_eps", 1e-6},
      {"max_position_embeddings", 4096}, {"tie_word_embeddings", false},
      {"sliding_window", 8}, {"layer_types", layer_types},
      {"num_attention_heads_per_layer", heads}, {"num_experts", E},
      {"num_experts_per_tok", 1}, {"moe_intermediate_size", MOE_I},
      {"shared_expert_intermediate_size", MOE_I}, {"norm_topk_prob", true},
      {"moe_routed_scaling_factor", 2.5}, {"mlp_only_layers", nlohmann::json::array({0})},
      {"rope_parameters",
       {{"full_attention", {{"rope_type", "yarn"}, {"rope_theta", 500000.0},
                            {"factor", 8.0}, {"original_max_position_embeddings", 512},
                            {"beta_slow", 1.0}, {"beta_fast", 32.0},
                            {"attention_factor", 1.0}, {"partial_rotary_factor", 0.5}}},
        {"sliding_attention", {{"rope_type", "default"}, {"rope_theta", 10000.0},
                               {"partial_rotary_factor", 1.0}}}}},
  };
  return c;
}

// BF16 tensor (2 bytes/elem).
Fx Bf16(const std::string& n, std::vector<int64_t> shape, int seed) {
  int64_t ne = 1;
  for (int64_t d : shape) ne *= d;
  return {n, "BF16", std::move(shape), Fill(static_cast<size_t>(ne) * 2, seed)};
}
void AttnBf16(std::vector<Fx>& v, const std::string& b, int& s) {
  v.push_back(Bf16(b + "input_layernorm.weight", {H}, s++));
  v.push_back(Bf16(b + "post_attention_layernorm.weight", {H}, s++));
  v.push_back(Bf16(b + "self_attn.q_proj.weight", {NH * Dh, H}, s++));
  v.push_back(Bf16(b + "self_attn.k_proj.weight", {KV * Dh, H}, s++));
  v.push_back(Bf16(b + "self_attn.v_proj.weight", {KV * Dh, H}, s++));
  v.push_back(Bf16(b + "self_attn.o_proj.weight", {H, NH * Dh}, s++));
  v.push_back(Bf16(b + "self_attn.g_proj.weight", {NH, H}, s++));
  v.push_back(Bf16(b + "self_attn.q_norm.weight", {Dh}, s++));
  v.push_back(Bf16(b + "self_attn.k_norm.weight", {Dh}, s++));
}
// One W4A4 NVFP4 projection: weight_packed U8 [N,K/2] + weight_scale F8 [N,K/16]
// + weight_global_scale/input_global_scale F32 scalars.
void ExpertProj(std::vector<Fx>& v, const std::string& p, int64_t n, int64_t k, int& s,
                float wgs, float igs) {
  v.push_back({p + ".weight_packed", "U8", {n, k / 2}, Fill(static_cast<size_t>(n * k / 2), s++)});
  v.push_back({p + ".weight_scale", "F8_E4M3", {n, k / 16}, Fill(static_cast<size_t>(n * k / 16), s++)});
  v.push_back({p + ".weight_global_scale", "F32", {}, F32Bytes(wgs)});
  v.push_back({p + ".input_global_scale", "F32", {}, F32Bytes(igs)});
}

std::vector<Fx> BuildTensors() {
  std::vector<Fx> v;
  int s = 1;
  v.push_back(Bf16("model.embed_tokens.weight", {V, H}, s++));
  v.push_back(Bf16("model.norm.weight", {H}, s++));
  v.push_back(Bf16("lm_head.weight", {V, H}, s++));
  // layer 0 dense
  AttnBf16(v, "model.layers.0.", s);
  v.push_back(Bf16("model.layers.0.mlp.gate_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16("model.layers.0.mlp.up_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16("model.layers.0.mlp.down_proj.weight", {H, DENSE_I}, s++));
  // layer 1 MoE
  AttnBf16(v, "model.layers.1.", s);
  v.push_back(Bf16("model.layers.1.mlp.gate.weight", {E, H}, s++));  // router BF16
  v.push_back({"model.layers.1.mlp.experts.e_score_correction_bias", "F32", {E},
               Fill(static_cast<size_t>(E) * 4, s++)});
  for (int e = 0; e < E; ++e) {
    const std::string ep = "model.layers.1.mlp.experts." + std::to_string(e) + ".";
    ExpertProj(v, ep + "gate_proj", MOE_I, H, s, 2.0F + e, 3.0F + e);
    ExpertProj(v, ep + "up_proj", MOE_I, H, s, 4.0F + e, 5.0F + e);
    ExpertProj(v, ep + "down_proj", H, MOE_I, s, 6.0F + e, 7.0F + e);
  }
  v.push_back(Bf16("model.layers.1.mlp.shared_expert.gate_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16("model.layers.1.mlp.shared_expert.up_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16("model.layers.1.mlp.shared_expert.down_proj.weight", {H, MOE_I}, s++));
  return v;
}

// ── controlled-FINITE builders (for the forward run-gate) ───────────────────
// The byte-identity fixtures above use arbitrary Fill() bytes, which decode to
// inf/NaN for BF16 (0x7F80…) and fp8-e4m3 (0x7F/0xFF). A forward through those
// would produce NaN that is a fixture artifact, not a real bug. These builders
// emit only finite, bounded values so `std::isfinite` is a meaningful assertion.
uint16_t Bf16Bits(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  return static_cast<uint16_t>(bits >> 16);  // truncate f32 -> bf16 (finite in, finite out)
}
// BF16 tensor with small deterministic values in [-0.06, 0.06].
//
// #2834: the period of this sequence MUST NOT divide the row stride of the
// tensors it fills. It was 16, and `model.embed_tokens.weight` and
// `lm_head.weight` are both `[V,H] = [8,32]`, so a row advanced the sequence by
// `32*7 = 224 == 0 (mod 16)` and every row came out byte-identical to every
// other row. The forward then returned one constant for all 24 logits, and no
// gate on this fixture could see a token id, a position, a causal mask or an
// embedding gather. 13 is coprime with 7 and with the strides here: a row
// advances the phase by `224 == 3 (mod 13)`, so the eight vocabulary rows take
// eight distinct phases. The case at the foot of this file holds that.
Fx Bf16Finite(const std::string& n, std::vector<int64_t> shape, int seed) {
  int64_t ne = 1;
  for (int64_t d : shape) ne *= d;
  std::string s(static_cast<size_t>(ne) * 2, '\0');
  for (int64_t i = 0; i < ne; ++i) {
    const float v = static_cast<float>(((i * 7 + seed) % 13) - 6) * 0.01F;
    const uint16_t bf = Bf16Bits(v);
    std::memcpy(&s[static_cast<size_t>(i) * 2], &bf, 2);
  }
  return {n, "BF16", std::move(shape), std::move(s)};
}
void AttnBf16Finite(std::vector<Fx>& v, const std::string& b, int& s) {
  v.push_back(Bf16Finite(b + "input_layernorm.weight", {H}, s++));
  v.push_back(Bf16Finite(b + "post_attention_layernorm.weight", {H}, s++));
  v.push_back(Bf16Finite(b + "self_attn.q_proj.weight", {NH * Dh, H}, s++));
  v.push_back(Bf16Finite(b + "self_attn.k_proj.weight", {KV * Dh, H}, s++));
  v.push_back(Bf16Finite(b + "self_attn.v_proj.weight", {KV * Dh, H}, s++));
  v.push_back(Bf16Finite(b + "self_attn.o_proj.weight", {H, NH * Dh}, s++));
  v.push_back(Bf16Finite(b + "self_attn.g_proj.weight", {NH, H}, s++));
  v.push_back(Bf16Finite(b + "self_attn.q_norm.weight", {Dh}, s++));
  v.push_back(Bf16Finite(b + "self_attn.k_norm.weight", {Dh}, s++));
}
// FINITE W4A4 projection: fp8-e4m3 scale byte 0x38 (== 1.0), fp4-e2m1 packed byte
// 0x11 (both nibbles code-1 == 0.5), unit globals -> bounded, NaN-free dequant.
void ExpertProjFinite(std::vector<Fx>& v, const std::string& p, int64_t n, int64_t k) {
  v.push_back({p + ".weight_packed", "U8", {n, k / 2},
               std::string(static_cast<size_t>(n * k / 2), '\x11')});
  v.push_back({p + ".weight_scale", "F8_E4M3", {n, k / 16},
               std::string(static_cast<size_t>(n * k / 16), '\x38')});
  v.push_back({p + ".weight_global_scale", "F32", {}, F32Bytes(1.0F)});
  v.push_back({p + ".input_global_scale", "F32", {}, F32Bytes(1.0F)});
}
std::vector<Fx> BuildFiniteTensors() {
  std::vector<Fx> v;
  int s = 1;
  v.push_back(Bf16Finite("model.embed_tokens.weight", {V, H}, s++));
  v.push_back(Bf16Finite("model.norm.weight", {H}, s++));
  v.push_back(Bf16Finite("lm_head.weight", {V, H}, s++));
  AttnBf16Finite(v, "model.layers.0.", s);
  v.push_back(Bf16Finite("model.layers.0.mlp.gate_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16Finite("model.layers.0.mlp.up_proj.weight", {DENSE_I, H}, s++));
  v.push_back(Bf16Finite("model.layers.0.mlp.down_proj.weight", {H, DENSE_I}, s++));
  AttnBf16Finite(v, "model.layers.1.", s);
  v.push_back(Bf16Finite("model.layers.1.mlp.gate.weight", {E, H}, s++));  // router BF16
  v.push_back({"model.layers.1.mlp.experts.e_score_correction_bias", "F32", {E},
               std::string(static_cast<size_t>(E) * 4, '\0')});  // zeros
  for (int e = 0; e < E; ++e) {
    const std::string ep = "model.layers.1.mlp.experts." + std::to_string(e) + ".";
    ExpertProjFinite(v, ep + "gate_proj", MOE_I, H);
    ExpertProjFinite(v, ep + "up_proj", MOE_I, H);
    ExpertProjFinite(v, ep + "down_proj", H, MOE_I);
  }
  v.push_back(Bf16Finite("model.layers.1.mlp.shared_expert.gate_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16Finite("model.layers.1.mlp.shared_expert.up_proj.weight", {MOE_I, H}, s++));
  v.push_back(Bf16Finite("model.layers.1.mlp.shared_expert.down_proj.weight", {H, MOE_I}, s++));
  return v;
}

}  // namespace

TEST_CASE("laguna nvfp4 loader: synthetic checkpoint round-trips byte-identically") {
  const std::vector<Fx> ts = BuildTensors();
  TempFile f(BuildSt(ts));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));

  const LagunaWeights w = LoadLagunaForCausalLMWeights(shards, TinyConfig());

  // structure
  CHECK(w.params.num_hidden_layers == 2);
  REQUIRE(w.layers.size() == 2u);
  CHECK(w.layers[0].is_dense == true);
  CHECK(w.layers[1].is_dense == false);

  // model-level BF16 round-trip (embed bytes == the fixture's embed bytes)
  auto find = [&](const std::string& n) -> const Fx& {
    for (const Fx& t : ts) if (t.name == n) return t;
    FAIL("fixture tensor missing: " << n);
    return ts[0];
  };
  const Fx& embed = find("model.embed_tokens.weight");
  REQUIRE(w.embed.bytes.size() == embed.bytes.size());
  CHECK(std::memcmp(w.embed.bytes.data(), embed.bytes.data(), embed.bytes.size()) == 0);

  // dense L0 MLP present, MoE fp4 fields empty on the dense layer
  CHECK(w.layers[0].mlp.gate_proj.bytes.size() ==
        find("model.layers.0.mlp.gate_proj.weight").bytes.size());

  // MoE layer: 2 W4A4 experts loaded
  const auto& moe = w.layers[1].moe;
  REQUIRE(moe.experts_gate_fp4.size() == static_cast<size_t>(E));
  REQUIRE(moe.experts_up_fp4.size() == static_cast<size_t>(E));
  REQUIRE(moe.experts_down_fp4.size() == static_cast<size_t>(E));

  // expert-0 gate: shapes + packed/scale byte-identity + global-scale math
  const auto& g0 = moe.experts_gate_fp4[0];
  CHECK(g0.n == MOE_I);
  CHECK(g0.k == H);
  const Fx& g0p = find("model.layers.1.mlp.experts.0.gate_proj.weight_packed");
  const Fx& g0s = find("model.layers.1.mlp.experts.0.gate_proj.weight_scale");
  REQUIRE(g0.packed.bytes.size() == g0p.bytes.size());
  CHECK(std::memcmp(g0.packed.bytes.data(), g0p.bytes.data(), g0p.bytes.size()) == 0);
  REQUIRE(g0.scale.bytes.size() == g0s.bytes.size());
  CHECK(std::memcmp(g0.scale.bytes.data(), g0s.bytes.data(), g0s.bytes.size()) == 0);
  // gate_proj expert 0: wgs=2.0, igs=3.0 (from ExpertProj call)
  CHECK(g0.weight_global_scale_inv == doctest::Approx(2.0));
  CHECK(g0.scale2 == doctest::Approx(1.0 / 2.0));              // reciprocal of the divisor
  CHECK(g0.input_global_scale_inv == doctest::Approx(3.0));
  CHECK(g0.alpha == doctest::Approx((1.0 / 2.0) * (1.0 / 3.0)));  // scale2 * 1/igs

  // expert-1 down: distinct globals prove per-expert indexing (wgs=7.0, igs=8.0)
  const auto& d1 = moe.experts_down_fp4[1];
  CHECK(d1.n == H);
  CHECK(d1.k == MOE_I);
  CHECK(d1.scale2 == doctest::Approx(1.0 / 7.0));
  CHECK(d1.alpha == doctest::Approx((1.0 / 7.0) * (1.0 / 8.0)));

  // F32 bias round-trip
  const Fx& bias = find("model.layers.1.mlp.experts.e_score_correction_bias");
  REQUIRE(moe.e_score_correction_bias.bytes.size() == bias.bytes.size());
  CHECK(std::memcmp(moe.e_score_correction_bias.bytes.data(), bias.bytes.data(),
                    bias.bytes.size()) == 0);

  // shared expert is BF16 (not fp4)
  CHECK(moe.shared_gate.bytes.size() ==
        find("model.layers.1.mlp.shared_expert.gate_proj.weight").bytes.size());
  CHECK(moe.shared_gate_fp4.Empty());  // the fp4 shared field stays unused
}

TEST_CASE("laguna nvfp4 loader: missing tensor throws (RED-first)") {
  std::vector<Fx> ts = BuildTensors();
  // drop expert-0 gate weight_scale -> the loader must throw, not silently succeed
  ts.erase(std::remove_if(ts.begin(), ts.end(),
                          [](const Fx& t) {
                            return t.name ==
                                   "model.layers.1.mlp.experts.0.gate_proj.weight_scale";
                          }),
           ts.end());
  TempFile f(BuildSt(ts));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  CHECK_THROWS(LoadLagunaForCausalLMWeights(shards, TinyConfig()));
}

TEST_CASE("laguna graph staging reads a borrowed BF16 row at an odd byte offset") {
  constexpr int64_t kRows = 2;
  constexpr int64_t kCols = 4;
  const uint16_t values[] = {
      Bf16Bits(0.0F), Bf16Bits(1.0F), Bf16Bits(2.0F), Bf16Bits(3.0F),
      Bf16Bits(10.0F), Bf16Bits(11.0F), Bf16Bits(12.0F), Bf16Bits(13.0F),
  };
  auto storage = std::make_shared<std::vector<uint8_t>>(
      1 + sizeof(values));
  std::memcpy(storage->data() + 1, values, sizeof(values));
  REQUIRE(reinterpret_cast<uintptr_t>(storage->data() + 1) %
              alignof(uint16_t) ==
          1);

  vllm::OwnedTensor embed;
  embed.bytes = vllm::OwnedBytes::Borrow(storage->data() + 1, sizeof(values),
                                          storage);
  embed.dtype = vt::DType::kBF16;
  embed.rank = 2;
  embed.shape[0] = kRows;
  embed.shape[1] = kCols;

  float destination[kCols] = {};
  vllm::StageLagunaGraphEmbedding(embed, 1, kCols, kRows, destination);
  CHECK(std::vector<float>(destination, destination + kCols) ==
        std::vector<float>{10.0F, 11.0F, 12.0F, 13.0F});
}

// N2 CPU RUN-GATE (task #230): the NVFP4 fp4 MoE branch in LagunaFfnBlock actually
// EXECUTES through the real LagunaForwardGguf entry (the same path examples/laguna_gen
// loops for the DGX benchmark) and produces finite, deterministic logits. This is the
// pre-DGX safety proof that the fp4 activation-quant -> MatmulNvfp4Fp4 -> alpha combine
// wires up correctly on CPU (ScaledFp4QuantKernel + MatmulNvfp4Fp4Kernel run here) before
// the real GB10 near-tie gate vs the recorded vLLM golden.
TEST_CASE("laguna nvfp4 forward: fp4 MoE branch runs finite + deterministic (CPU run-gate)") {
  const std::vector<Fx> ts = BuildFiniteTensors();
  TempFile f(BuildSt(ts));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  const LagunaWeights w = LoadLagunaForCausalLMWeights(shards, TinyConfig());
  REQUIRE(w.has_nvfp4_weights);
  // Layer 1 is MoE and its experts are the true-W4A4 fp4 tensors -> the fp4 branch fires.
  REQUIRE(!w.layers[1].moe.experts_gate_fp4.empty());
  REQUIRE(w.layers[1].moe.experts_gate_fp4[0].IsTrueW4A4());

  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<int32_t> tokens = {1, 3, 2};
  const std::vector<int32_t> positions = {0, 1, 2};
  const std::vector<int32_t> logits_indices = {0, 1, 2};

  const std::vector<float> out1 =
      vllm::LagunaForwardGguf(w, cpuq, tokens, positions, logits_indices);
  REQUIRE(static_cast<int64_t>(out1.size()) == 3 * V);
  for (float x : out1) CHECK(std::isfinite(x));

  // Deterministic: identical inputs -> byte-identical logits across runs.
  const std::vector<float> out2 =
      vllm::LagunaForwardGguf(w, cpuq, tokens, positions, logits_indices);
  REQUIRE(out2.size() == out1.size());
  CHECK(std::memcmp(out1.data(), out2.data(), out1.size() * sizeof(float)) == 0);

  // The routed experts are actually CONSUMED: zeroing EVERY routed expert's gate packed
  // codes (silu(gate)=silu(0)=0 -> routed contribution collapses to 0, regardless of which
  // expert top-1 picks per token) shifts the MoE-layer output, hence the logits. Proves the
  // fp4 GEMM result flows into the residual stream.
  std::vector<Fx> ts2 = BuildFiniteTensors();
  for (Fx& t : ts2)
    if (t.name.find("mlp.experts.") != std::string::npos &&
        t.name.find("gate_proj.weight_packed") != std::string::npos)
      std::fill(t.bytes.begin(), t.bytes.end(), '\x00');  // e2m1 code 0 == 0.0
  TempFile f2(BuildSt(ts2));
  std::vector<SafetensorsFile> shards2;
  shards2.push_back(SafetensorsFile::Open(f2.path()));
  const LagunaWeights w2 = LoadLagunaForCausalLMWeights(shards2, TinyConfig());
  const std::vector<float> out3 =
      vllm::LagunaForwardGguf(w2, cpuq, tokens, positions, logits_indices);
  REQUIRE(out3.size() == out1.size());
  bool differs = false;
  for (size_t i = 0; i < out1.size(); ++i)
    if (out1[i] != out3[i]) { differs = true; break; }
  CHECK(differs);
}

// ═══════════════════════════════════════════════════════════════════════════
// #2618 — the REGISTRY step. Everything above enters `LagunaForwardGguf` by
// name; nothing above proves that the PRODUCTION entry point reaches it. It did
// not: `ForwardLagunaForCausalLM` sent both of its branches to
// `LagunaModel::Forward`, the f32 reference, whose `moe.experts_*` this
// safetensors loader leaves EMPTY (it fills `experts_*_fp4` instead), so the
// reference sliced `exp_g.begin() + id*gu_stride` out of an empty vector and the
// process died inside `memcpy`. The GGUF arm did not crash — its
// `moe.experts_*` ARE filled, with Q4_K/Q5_K blocks that `ReadF32` refuses by
// name — so the two arms failed differently and only one of them failed loudly.
//
// These cases enter ONLY through `ModelRegistry::Load` + `ModelRegistry::Forward`.
// See `.agents/specs/laguna-registry-forward-2618.md`.
namespace {

// A `ModelForwardInput` over one sequence, built the way the runner builds one.
// `multi_kv` stays NULL: `kLagunaFactory` leaves `consumes_multi_kv` false, so a
// topology here would be refused by the engine guard before dispatch and the
// case would measure that guard instead of this forward.
struct RegistryStep {
  std::vector<int32_t> token_ids;
  std::vector<int32_t> positions;
  std::vector<int32_t> logits_indices;
  vllm::v1::CommonAttentionMetadata attn_meta{};
  vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  vllm::HfConfig config = TinyConfig();
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  explicit RegistryStep(std::vector<int32_t> ids, std::vector<int32_t> want)
      : token_ids(std::move(ids)), logits_indices(std::move(want)) {
    const int T = static_cast<int>(token_ids.size());
    positions.resize(token_ids.size());
    for (int i = 0; i < T; ++i) positions[static_cast<size_t>(i)] = i;
    attn_meta.num_reqs = 1;
    attn_meta.num_actual_tokens = T;
    attn_meta.num_computed_tokens_cpu = {0};
    attn_meta.seq_lens_cpu = {T};
    attn_meta.seq_lens = attn_meta.seq_lens_cpu;
    attn_meta.query_start_loc = {0, T};
    attn_meta.query_start_loc_cpu = attn_meta.query_start_loc;
  }

  vllm::ModelForwardInput Get() {
    vllm::ModelForwardInput in{.token_ids = token_ids,
                               .positions = positions,
                               .attn_meta = attn_meta,
                               .gdn_meta = gdn_meta,
                               .attn_kv = attn_kv,
                               .gdn_state = gdn_state,
                               .config = config,
                               .queue = queue,
                               .logits_indices = logits_indices,
                               .num_reqs = 1};
    return in;
  }
};

std::unique_ptr<vllm::LoadedModel> LoadThroughRegistry(
    const std::vector<SafetensorsFile>& shards) {
  return vllm::ModelRegistry::Load(TinyConfig(),
                                   vllm::ModelSource::FromSafetensors(shards));
}

}  // namespace

// (1) The step RUNS through the production entry point and produces finite
// logits of the right shape. RED on `ca07f6e94` is a SIGSEGV, not an assertion:
// the process dies inside `LagunaModel::Forward`, so doctest prints no counts.
TEST_CASE("laguna registry forward: ModelRegistry::Forward runs the NVFP4 arm") {
  TempFile f(BuildSt(BuildFiniteTensors()));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(shards);
  REQUIRE(model != nullptr);

  RegistryStep step({1, 3, 2}, {0, 1, 2});
  const vllm::ForwardLogits out = vllm::ModelRegistry::Forward(*model, step.Get());
  CHECK(out.vocab == V);
  CHECK(out.rows == 3);
  REQUIRE(out.host.size() == static_cast<size_t>(3 * V));
  for (float x : out.host) CHECK(std::isfinite(x));
}

// (2) WHICH forward it reached. (1) alone passes for any forward that returns
// finite numbers of the right shape, so it cannot tell `LagunaForwardGguf` from
// a zero-filled stub. These logits must be BYTE-IDENTICAL to the direct
// `LagunaForwardGguf` call the loader run-gate above already trusts.
TEST_CASE("laguna registry forward: registry logits == LagunaForwardGguf logits") {
  TempFile f(BuildSt(BuildFiniteTensors()));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(shards);
  REQUIRE(model != nullptr);

  const LagunaWeights w = LoadLagunaForCausalLMWeights(shards, TinyConfig());
  REQUIRE(w.has_nvfp4_weights);
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<int32_t> tokens = {1, 3, 2};
  const std::vector<int32_t> positions = {0, 1, 2};
  const std::vector<int32_t> want = {0, 1, 2};
  const std::vector<float> direct =
      vllm::LagunaForwardGguf(w, cpuq, tokens, positions, want);

  RegistryStep step(tokens, want);
  const vllm::ForwardLogits out = vllm::ModelRegistry::Forward(*model, step.Get());
  REQUIRE(out.host.size() == direct.size());
  CHECK(std::memcmp(out.host.data(), direct.data(),
                    direct.size() * sizeof(float)) == 0);
}

// (3) The routed experts are CONSUMED on the registry path, not only on the
// direct one: zeroing every routed expert's packed gate codes (e2m1 code 0 ==
// 0.0, so silu(gate) collapses the routed contribution regardless of which
// expert top-1 picks) must move the registry's logits.
TEST_CASE("laguna registry forward: routed experts reach the registry step") {
  RegistryStep step({1, 3, 2}, {0, 1, 2});

  TempFile f(BuildSt(BuildFiniteTensors()));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(shards);
  const vllm::ForwardLogits base = vllm::ModelRegistry::Forward(*model, step.Get());

  std::vector<Fx> ts2 = BuildFiniteTensors();
  for (Fx& t : ts2)
    if (t.name.find("mlp.experts.") != std::string::npos &&
        t.name.find("gate_proj.weight_packed") != std::string::npos)
      std::fill(t.bytes.begin(), t.bytes.end(), '\x00');
  TempFile f2(BuildSt(ts2));
  std::vector<SafetensorsFile> shards2;
  shards2.push_back(SafetensorsFile::Open(f2.path()));
  std::unique_ptr<vllm::LoadedModel> model2 = LoadThroughRegistry(shards2);
  const vllm::ForwardLogits zeroed =
      vllm::ModelRegistry::Forward(*model2, step.Get());

  REQUIRE(base.host.size() == zeroed.host.size());
  bool differs = false;
  for (size_t i = 0; i < base.host.size(); ++i)
    if (base.host[i] != zeroed.host[i]) { differs = true; break; }
  CHECK(differs);
}

// (4) #2834 — THE FIXTURE ITSELF MUST DISCRIMINATE.
//
// (1) to (3) above, and the two direct-entry cases before them, all run on
// `BuildFiniteTensors()`. None of them can fail while that fixture returns one
// constant for every logit, which is what it did until this case existed:
// `Bf16Finite` had period 16 and the `[V,H] = [8,32]` row stride advanced it by
// `32*7 = 224 == 0 (mod 16)`, so every row of `model.embed_tokens.weight` and of
// `lm_head.weight` was byte-identical to every other row. A constant output is
// finite, is deterministic, and is byte-identical to itself across two entry
// points, so the finite/deterministic run-gate and the registry byte-identity
// case both held trivially. `MODEL-LAGUNA-REGISTRY-FORWARD-2618`'s M5 mutation
// (zeroing `positions`) SURVIVED for exactly this reason.
//
// This case asserts the three properties those gates need and could not have:
// the logits move with the TOKEN ID, they move with the POSITION, and they are
// not all the same number. It enters through `ModelRegistry::Forward`, so it
// measures the production path rather than the fixture in isolation.
TEST_CASE("laguna registry forward: the fixture discriminates token, position and row") {
  TempFile f(BuildSt(BuildFiniteTensors()));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(f.path()));
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(shards);
  REQUIRE(model != nullptr);

  auto run = [&](const std::vector<int32_t>& ids,
                 const std::vector<int32_t>& pos) {
    RegistryStep step(ids, {0, 1, 2});
    step.positions = pos;
    const vllm::ForwardLogits out = vllm::ModelRegistry::Forward(*model, step.Get());
    REQUIRE(out.host.size() == static_cast<size_t>(3 * V));
    return out.host;
  };
  auto maxdiff = [](const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    float m = 0.0F;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
  };

  const std::vector<float> base = run({1, 3, 2}, {0, 1, 2});

  // T1 — the TOKEN ID reaches the logits. A fixture whose embedding rows are all
  // equal cannot fail this by any amount, so it is the embedding gather's gate.
  const float tok_diff = maxdiff(base, run({5, 5, 5}, {0, 1, 2}));
  MESSAGE("laguna fixture: maxdiff tokens {1,3,2} vs {5,5,5} = " << tok_diff);
  CHECK(tok_diff > 0.0F);

  // T2 — the POSITION reaches the logits, which is RoPE plus the position
  // plumbing. {0,0,0} leaves RoPE at the identity for every row.
  const float pos_diff = maxdiff(base, run({1, 3, 2}, {0, 0, 0}));
  MESSAGE("laguna fixture: maxdiff positions {0,1,2} vs {0,0,0} = " << pos_diff);
  CHECK(pos_diff > 0.0F);

  const float far_diff = maxdiff(base, run({1, 3, 2}, {0, 2000, 4000}));
  MESSAGE("laguna fixture: maxdiff positions {0,1,2} vs {0,2000,4000} = " << far_diff);
  CHECK(far_diff > 0.0F);

  // T3 — the output is not one repeated number. Two ways it could be: every row
  // equal to every other row (the causal mask and the row plumbing), and every
  // vocabulary entry within a row equal (the `lm_head` rows).
  float row_spread = 0.0F;
  for (int v = 0; v < V; ++v)
    row_spread = std::max(row_spread, std::fabs(base[static_cast<size_t>(v)] -
                                                base[static_cast<size_t>(V + v)]));
  MESSAGE("laguna fixture: maxdiff row0 vs row1 = " << row_spread);
  CHECK(row_spread > 0.0F);

  float lo = base[0], hi = base[0];
  for (int v = 0; v < V; ++v) {
    lo = std::min(lo, base[static_cast<size_t>(v)]);
    hi = std::max(hi, base[static_cast<size_t>(v)]);
  }
  MESSAGE("laguna fixture: row0 logit min " << lo << " max " << hi);
  CHECK(hi > lo);
}
