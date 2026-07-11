// CPU correctness scaffold for the DENSE Qwen3.6-27B text gate (notes §5
// steps 2-3). No GPU, no checkpoint, no oracle — three CPU gates:
//   1. Loader ROUTING: IsQwen27QuantizedLinear encodes the config.json `ignore`
//      list (notes §3.6) — the quantized set vs the bf16 set.
//   2. W4A4 MATERIALIZATION: MaterializeCtNvfp4Bf16Transposed dequants a packed
//      NVFP4 W4A4 tensor (via the CT reference) to bf16 in Matmul-B layout,
//      against the hand-computed block from the CT-emulation unit test.
//   3. Dense FORWARD wiring: Qwen3_5DenseModel::ForwardDense runs a small
//      synthetic hybrid (GDN + full-attn) dense model and returns finite,
//      deterministic [T,vocab] logits; perturbing an MLP weight moves the output
//      (proves the dense SwiGLU MLP is actually in the forward path).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::DenseMlpWeights;
using vllm::HfConfig;
using vllm::IsQwen27QuantizedLinear;
using vllm::MaterializeCtNvfp4Bf16Transposed;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5DenseModel;
using vllm::Qwen3_5DenseWeights;
using vllm::StTensor;
using vllm::TensorResolver;
using vt::DType;

namespace {

uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u = static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}

OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {  // f32
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

// 27B-shaped small dense config: layer_types [LA, LA, LA, FA], no experts.
HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = 40;
  c.num_attention_heads = 6;   // GQA ratio 3 (Hv/Hk analogue) — 6:2
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.intermediate_size = 16;  // dense SwiGLU MLP
  c.num_experts = 0;         // DENSE — no MoE
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 6;  // GQA ratio 3 vs the 35B's 2
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  return c;
}

DenseMlpWeights MakeMlp(const HfConfig& c, uint64_t s) {
  DenseMlpWeights m;
  const int64_t H = c.hidden_size, I = c.intermediate_size;
  m.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 1);
  m.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 2);
  m.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 3);
  return m;
}

Qwen3_5DenseWeights MakeWeights(const HfConfig& c) {
  Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads, Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention = (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.mlp = MakeMlp(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

struct PlainWeightFixture {
  std::unordered_map<std::string, std::vector<uint8_t>> storage;
  std::unordered_map<std::string, StTensor> tensors;

  void AddBf16(const std::string& name, std::vector<int64_t> shape,
               uint16_t first = 1) {
    int64_t numel = 1;
    for (int64_t dim : shape) numel *= dim;
    std::vector<uint8_t>& bytes = storage[name];
    bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
    auto* values = reinterpret_cast<uint16_t*>(bytes.data());
    for (int64_t i = 0; i < numel; ++i)
      values[i] = vt::F32ToBF16(static_cast<float>(first + i));
    tensors[name] = StTensor{"BF16", std::move(shape), bytes.data(), bytes.size()};
  }

  void AddF32(const std::string& name, std::vector<int64_t> shape,
              float first = 1.0F) {
    int64_t numel = 1;
    for (int64_t dim : shape) numel *= dim;
    std::vector<uint8_t>& bytes = storage[name];
    bytes.resize(static_cast<size_t>(numel) * sizeof(float));
    auto* values = reinterpret_cast<float*>(bytes.data());
    for (int64_t i = 0; i < numel; ++i) values[i] = first + static_cast<float>(i);
    tensors[name] = StTensor{"F32", std::move(shape), bytes.data(), bytes.size()};
  }

  TensorResolver Resolver() {
    return [this](const std::string& name) -> const StTensor& {
      return tensors.at(name);
    };
  }
};

void AddPlainLayerCommon(PlainWeightFixture& f, const std::string& base) {
  f.AddBf16(base + "input_layernorm.weight", {3});
  f.AddBf16(base + "post_attention_layernorm.weight", {3});
  f.AddBf16(base + "mlp.gate_proj.weight", {2, 3}, 20);
  f.AddBf16(base + "mlp.up_proj.weight", {2, 3}, 30);
  f.AddBf16(base + "mlp.down_proj.weight", {3, 2}, 40);
}

}  // namespace

TEST_CASE("qwen27 loader routing: IsQwen27QuantizedLinear encodes §3.6") {
  const std::string L = "model.language_model.layers.0.";
  // Quantized (W4A4): dense-MLP + self_attn q/k/v/o + GDN out_proj.
  CHECK(IsQwen27QuantizedLinear(L + "mlp.gate_proj"));
  CHECK(IsQwen27QuantizedLinear(L + "mlp.up_proj"));
  CHECK(IsQwen27QuantizedLinear(L + "mlp.down_proj"));
  CHECK(IsQwen27QuantizedLinear(L + "self_attn.q_proj"));
  CHECK(IsQwen27QuantizedLinear(L + "self_attn.k_proj"));
  CHECK(IsQwen27QuantizedLinear(L + "self_attn.v_proj"));
  CHECK(IsQwen27QuantizedLinear(L + "self_attn.o_proj"));
  CHECK(IsQwen27QuantizedLinear(L + "linear_attn.out_proj"));
  // NOT quantized (bf16): GDN in-projs, conv, norms, embed, lm_head, mtp, visual.
  CHECK_FALSE(IsQwen27QuantizedLinear(L + "linear_attn.in_proj_qkv"));
  CHECK_FALSE(IsQwen27QuantizedLinear(L + "linear_attn.in_proj_z"));
  CHECK_FALSE(IsQwen27QuantizedLinear(L + "linear_attn.in_proj_a"));
  CHECK_FALSE(IsQwen27QuantizedLinear(L + "linear_attn.in_proj_b"));
  CHECK_FALSE(IsQwen27QuantizedLinear(L + "linear_attn.conv1d"));
  CHECK_FALSE(IsQwen27QuantizedLinear(L + "input_layernorm"));
  CHECK_FALSE(IsQwen27QuantizedLinear("model.language_model.embed_tokens"));
  CHECK_FALSE(IsQwen27QuantizedLinear("lm_head"));
  CHECK_FALSE(IsQwen27QuantizedLinear("mtp.layers.0.mlp.gate_proj"));
  CHECK_FALSE(IsQwen27QuantizedLinear("model.visual.blocks.0.mlp.gate_proj"));
}

TEST_CASE("qwen35 plain dense loader accepts BF16 projections and F32 GDN weights") {
  const std::string base = "model.language_model.layers.0.";

  SUBCASE("full attention and MLP use ordinary BF16 weights") {
    PlainWeightFixture f;
    AddPlainLayerCommon(f, base);
    f.AddBf16(base + "self_attn.q_proj.weight", {2, 3}, 1);
    f.AddBf16(base + "self_attn.k_proj.weight", {2, 3}, 10);
    f.AddBf16(base + "self_attn.v_proj.weight", {2, 3}, 20);
    f.AddBf16(base + "self_attn.o_proj.weight", {3, 2}, 30);
    f.AddBf16(base + "self_attn.q_norm.weight", {2});
    f.AddBf16(base + "self_attn.k_norm.weight", {2});

    const Qwen3_5DenseLayerWeights layer =
        vllm::LoadQwen3_5DenseLayer(f.Resolver(), "full_attention", 0);
    CHECK_FALSE(layer.is_linear_attention);
    CHECK(layer.attn.q_proj_fp4.Empty());
    CHECK(layer.attn.q_proj.nk);
    CHECK(layer.attn.q_proj.shape[0] == 2);
    CHECK(layer.attn.q_proj.shape[1] == 3);
    const auto* q = reinterpret_cast<const uint16_t*>(layer.attn.q_proj.bytes.data());
    CHECK(q[0] == vt::F32ToBF16(1.0F));
    CHECK(q[1] == vt::F32ToBF16(2.0F));
    CHECK(layer.mlp.gate_proj_fp4.Empty());
    CHECK_FALSE(layer.mlp.gate_proj.Empty());
    CHECK(layer.mlp.gate_proj.nk);
    CHECK(layer.mlp.gate_up_proj.nk);
    CHECK(layer.mlp.gate_up_proj.shape[0] == 4);
    CHECK(layer.mlp.gate_up_proj.shape[1] == 3);
    const auto* gate_up = reinterpret_cast<const uint16_t*>(
        layer.mlp.gate_up_proj.bytes.data());
    CHECK(gate_up[0] == vt::F32ToBF16(20.0F));
    CHECK(gate_up[6] == vt::F32ToBF16(30.0F));
  }

  SUBCASE("linear attention accepts plain out_proj and F32 state parameters") {
    PlainWeightFixture f;
    AddPlainLayerCommon(f, base);
    const std::string la = base + "linear_attn.";
    f.AddBf16(la + "in_proj_qkv.weight", {4, 3}, 1);
    f.AddBf16(la + "in_proj_z.weight", {4, 3}, 20);
    f.AddBf16(la + "in_proj_b.weight", {2, 3}, 40);
    f.AddBf16(la + "in_proj_a.weight", {2, 3}, 50);
    f.AddBf16(la + "out_proj.weight", {3, 4});
    f.AddBf16(la + "conv1d.weight", {4, 1, 3});
    f.AddF32(la + "A_log", {2}, -2.0F);
    f.AddF32(la + "dt_bias", {2}, 0.25F);
    f.AddF32(la + "norm.weight", {4}, 0.5F);

    const Qwen3_5DenseLayerWeights layer =
        vllm::LoadQwen3_5DenseLayer(f.Resolver(), "linear_attention", 0);
    CHECK(layer.is_linear_attention);
    CHECK(layer.gdn.out_proj_fp4.Empty());
    CHECK_FALSE(layer.gdn.out_proj.Empty());
    CHECK(layer.gdn.out_proj.nk);
    CHECK(layer.gdn.in_proj_ba.nk);
    CHECK(layer.gdn.in_proj_ba.shape[0] == 4);
    CHECK(layer.gdn.in_proj_ba.shape[1] == 3);
    const auto* ba = reinterpret_cast<const uint16_t*>(
        layer.gdn.in_proj_ba.bytes.data());
    CHECK(ba[0] == vt::F32ToBF16(40.0F));
    CHECK(ba[6] == vt::F32ToBF16(50.0F));
    CHECK(layer.gdn.a_log.dtype == DType::kF32);
    CHECK(layer.gdn.dt_bias.dtype == DType::kF32);
    CHECK(layer.gdn.norm_weight.dtype == DType::kF32);
    const auto* a_log = reinterpret_cast<const float*>(layer.gdn.a_log.bytes.data());
    CHECK(a_log[0] == doctest::Approx(-2.0F));
    CHECK(a_log[1] == doctest::Approx(-1.0F));
  }
}

TEST_CASE("qwen27 W4A4 materialize: CT dequant + bf16 + transpose to [in,out]") {
  // Same hand-computed block as test_ct_nvfp4_emulation: out=1, in=16.
  //   nibbles: 0=+0.5 1=+6 2=-6 3=0 4=+1 5=-1 6..15=0
  //   group scale fp8 0x3A = 1.25; global divisor 0.5 -> mult 2.0 -> gs 2.5.
  std::vector<uint8_t> packed = {0x71, 0x0F, 0xA2, 0x00, 0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> scale = {0x3A};
  float wgs = 0.5F;
  const float expected[16] = {1.25F, 15.0F, -15.0F, 0.0F, 2.5F, -2.5F, 0.0F, 0.0F,
                              0.0F,  0.0F,  0.0F,    0.0F, 0.0F, 0.0F,  0.0F, 0.0F};

  std::unordered_map<std::string, StTensor> tensors;
  auto add = [&](const std::string& n, const std::string& dt,
                 std::vector<int64_t> shape, const void* data, size_t nbytes) {
    StTensor t;
    t.dtype = dt;
    t.shape = std::move(shape);
    t.data = reinterpret_cast<const uint8_t*>(data);
    t.nbytes = nbytes;
    tensors[n] = t;
  };
  add("p.weight_packed", "U8", {1, 8}, packed.data(), packed.size());
  add("p.weight_scale", "F8_E4M3", {1, 1}, scale.data(), scale.size());
  add("p.weight_global_scale", "F32", {1}, &wgs, sizeof(float));
  const TensorResolver get = [&tensors](const std::string& n) -> const StTensor& {
    return tensors.at(n);
  };

  const OwnedTensor o = MaterializeCtNvfp4Bf16Transposed(get, "p");
  // Matmul-B layout: [in=16, out=1].
  REQUIRE(o.rank == 2);
  CHECK(o.shape[0] == 16);
  CHECK(o.shape[1] == 1);
  const auto* v = reinterpret_cast<const uint16_t*>(o.bytes.data());
  for (int i = 0; i < 16; ++i)
    CHECK(v[i] == vt::F32ToBF16(expected[i]));  // transposed[i,0] == dequant[0,i]
}

TEST_CASE("qwen27 dense forward: finite, deterministic [T,vocab] logits") {
  const HfConfig c = MakeConfig();
  const Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t T = 6, vocab = c.vocab_size;
  std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};
  std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5};

  const std::vector<float> a = Qwen3_5DenseModel::ForwardDense(ids, pos, w, c, q);
  REQUIRE(a.size() == static_cast<size_t>(T * vocab));
  for (float x : a) REQUIRE(std::isfinite(x));

  // Determinism: identical inputs -> identical logits.
  const std::vector<float> b = Qwen3_5DenseModel::ForwardDense(ids, pos, w, c, q);
  double maxd = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    maxd = std::max(maxd, std::abs(static_cast<double>(a[i]) - b[i]));
  CHECK(maxd == 0.0);
}

TEST_CASE("qwen27 dense forward: the dense MLP is wired into the forward path") {
  const HfConfig c = MakeConfig();
  Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};
  std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5};

  const std::vector<float> base = Qwen3_5DenseModel::ForwardDense(ids, pos, w, c, q);

  // Perturb one dense-MLP down-proj weight in layer 0; the output must change.
  auto* dp = reinterpret_cast<uint16_t*>(w.layers[0].mlp.down_proj.bytes.data());
  dp[0] = vt::F32ToBF16(0.5F);
  const std::vector<float> moved = Qwen3_5DenseModel::ForwardDense(ids, pos, w, c, q);

  double maxd = 0.0;
  for (size_t i = 0; i < base.size(); ++i)
    maxd = std::max(maxd, std::abs(static_cast<double>(base[i]) - moved[i]));
  MESSAGE("dense-MLP perturbation moved logits by max|diff| = " << maxd);
  CHECK(maxd > 0.0);
}
