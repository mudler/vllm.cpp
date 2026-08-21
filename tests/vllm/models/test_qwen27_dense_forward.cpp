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
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::DenseMlpWeights;
using vllm::HfConfig;
using vllm::IsQwen27QuantizedLinear;
using vllm::LoadMergedBf16RawNK;
using vllm::LoadQwen3_5DenseGdn;
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

// Ported from vllm/model_executor/models/qwen3_5.py:200-210's
// stacked_params_mapping and linear.py::MergedColumnParallelLinear.weight_loader:
// physical in_proj_ba owns exact [b,a] output-row order in one raw [N,K]
// parameter. This focused loader contract also pins one-owner byte accounting.
TEST_CASE("qwen27 loader packs GDN in_proj_ba in exact b,a row order") {
  std::vector<uint16_t> b = {
      0x3f80, 0x4000, 0x4040, 0x4080,
      0x40a0, 0x40c0, 0x40e0, 0x4100,
  };  // [2,4]
  std::vector<uint16_t> a = {
      0xbf80, 0xc000, 0xc040, 0xc080,
      0xc0a0, 0xc0c0, 0xc0e0, 0xc100,
      0x3e80, 0x3f00, 0x3f40, 0x3f80,
  };  // [3,4]
  std::unordered_map<std::string, StTensor> tensors;
  auto add = [&](const std::string& name, const std::string& dtype,
                 std::vector<int64_t> shape, const void* data, size_t bytes) {
    StTensor tensor;
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    tensor.data = static_cast<const uint8_t*>(data);
    tensor.nbytes = bytes;
    tensors[name] = tensor;
  };
  add("b", "BF16", {2, 4}, b.data(), b.size() * sizeof(uint16_t));
  add("a", "BF16", {3, 4}, a.data(), a.size() * sizeof(uint16_t));
  const TensorResolver get = [&tensors](const std::string& name) -> const StTensor& {
    const auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("missing tensor: " + name);
    return it->second;
  };

  const OwnedTensor merged = LoadMergedBf16RawNK(get, {"b", "a"});
  REQUIRE(merged.rank == 2);
  CHECK(merged.shape[0] == 5);
  CHECK(merged.shape[1] == 4);
  CHECK(merged.dtype == DType::kBF16);
  CHECK(merged.nk);
  REQUIRE(merged.bytes.size() == (b.size() + a.size()) * sizeof(uint16_t));
  CHECK(std::memcmp(merged.bytes.data(), b.data(), b.size() * sizeof(uint16_t)) == 0);
  CHECK(std::memcmp(merged.bytes.data() + b.size() * sizeof(uint16_t), a.data(),
                    a.size() * sizeof(uint16_t)) == 0);

  CHECK_THROWS_WITH_AS(LoadMergedBf16RawNK(get, {"b", "missing"}),
                       doctest::Contains("missing tensor"), std::runtime_error);
  tensors["a"].dtype = "F32";
  CHECK_THROWS_WITH_AS(LoadMergedBf16RawNK(get, {"b", "a"}),
                       doctest::Contains("unsupported dtype 'F32'"),
                       std::runtime_error);
  tensors["a"].dtype = "BF16";
  tensors["a"].shape = {3, 5};
  CHECK_THROWS_WITH_AS(LoadMergedBf16RawNK(get, {"b", "a"}),
                       doctest::Contains("share input width"), std::runtime_error);
  CHECK_THROWS_WITH_AS(LoadMergedBf16RawNK(get, {}),
                       doctest::Contains("at least one shard"), std::runtime_error);

  // nvidia/Qwen3.6-27B-NVFP4 publishes the GDN in-projections as per-tensor FP8
  // beside BF16 siblings, so one merged parameter may MIX dtypes. The FP8 shard
  // is materialized first (nvfp4_dequant.h:83 -- `bf16(f8(w) * scale)`), and the
  // b,a row order plus the nk=true orientation stay exactly as the all-BF16 case
  // above. E4M3: 0x38 = 1.0, 0x40 = 2.0, 0x3c = 1.5.
  const std::vector<uint8_t> a_f8 = {
      0x38, 0x40, 0x3c, 0x38,
      0x40, 0x3c, 0x38, 0x40,
      0x3c, 0x38, 0x40, 0x3c,
  };  // [3,4]
  const float a_scale = 2.0F;
  tensors["a"].dtype = "F8_E4M3";
  tensors["a"].shape = {3, 4};
  tensors["a"].data = a_f8.data();
  tensors["a"].nbytes = a_f8.size();
  add("a_scale", "F32", {1}, &a_scale, sizeof(a_scale));

  const OwnedTensor mixed = LoadMergedBf16RawNK(get, {"b", "a"});
  REQUIRE(mixed.rank == 2);
  CHECK(mixed.shape[0] == 5);
  CHECK(mixed.shape[1] == 4);
  CHECK(mixed.dtype == DType::kBF16);
  REQUIRE(mixed.bytes.size() == (b.size() + a_f8.size()) * sizeof(uint16_t));
  CHECK(std::memcmp(mixed.bytes.data(), b.data(), b.size() * sizeof(uint16_t)) == 0);
  const uint16_t* merged_rows =
      reinterpret_cast<const uint16_t*>(mixed.bytes.data());
  // 1.0,2.0,1.5 * 2.0 -> 2.0,4.0,3.0 -> bf16 0x4000,0x4080,0x4040.
  const uint16_t want[3] = {0x4000, 0x4080, 0x4040};
  for (size_t i = 0; i < a_f8.size(); ++i) {
    CHECK(merged_rows[b.size() + i] == want[i % 3]);
  }

  // A per-output-channel scale read as per-tensor would be silently wrong, so
  // any other element count is REJECTED rather than reinterpreted.
  const float bad_scale[2] = {2.0F, 2.0F};
  add("a_scale", "F32", {2}, bad_scale, sizeof(bad_scale));
  CHECK_THROWS_WITH_AS(LoadMergedBf16RawNK(get, {"b", "a"}),
                       doctest::Contains("per-tensor or one value per output row"),
                       std::runtime_error);
}

TEST_CASE("qwen27 GDN loader retains one merged BA owner and no split copies") {
  constexpr int64_t kHidden = 16;
  constexpr int64_t kHeads = 2;
  std::vector<uint16_t> qkv(16 * kHidden, 0x3f80);
  std::vector<uint16_t> z(16 * kHidden, 0x3f00);
  std::vector<uint16_t> b(kHeads * kHidden);
  std::vector<uint16_t> a(kHeads * kHidden);
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<uint16_t>(0x4000 + i);
    a[i] = static_cast<uint16_t>(0x4100 + i);
  }
  std::vector<uint16_t> conv(16 * 4, 0x3e80);
  std::vector<uint16_t> alog(kHeads, 0x3f80);
  std::vector<uint16_t> dtb(kHeads, 0x3f00);
  std::vector<uint16_t> norm(8, 0x3f80);
  std::vector<uint8_t> out_packed(16 * (16 / 2), 0);
  std::vector<uint8_t> out_scale(16 * (16 / 16), 0x38);
  float weight_divisor = 1.0f;
  float input_divisor = 1.0f;

  std::unordered_map<std::string, StTensor> tensors;
  auto add = [&](const std::string& name, const std::string& dtype,
                 std::vector<int64_t> shape, const void* data, size_t bytes) {
    StTensor tensor;
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    tensor.data = static_cast<const uint8_t*>(data);
    tensor.nbytes = bytes;
    tensors[name] = tensor;
  };
  const std::string p = "layer.linear_attn.";
  add(p + "in_proj_qkv.weight", "BF16", {16, kHidden}, qkv.data(),
      qkv.size() * sizeof(uint16_t));
  add(p + "in_proj_z.weight", "BF16", {16, kHidden}, z.data(),
      z.size() * sizeof(uint16_t));
  add(p + "in_proj_b.weight", "BF16", {kHeads, kHidden}, b.data(),
      b.size() * sizeof(uint16_t));
  add(p + "in_proj_a.weight", "BF16", {kHeads, kHidden}, a.data(),
      a.size() * sizeof(uint16_t));
  add(p + "conv1d.weight", "BF16", {16, 1, 4}, conv.data(),
      conv.size() * sizeof(uint16_t));
  add(p + "A_log", "BF16", {kHeads}, alog.data(),
      alog.size() * sizeof(uint16_t));
  add(p + "dt_bias", "BF16", {kHeads}, dtb.data(),
      dtb.size() * sizeof(uint16_t));
  add(p + "norm.weight", "BF16", {8}, norm.data(),
      norm.size() * sizeof(uint16_t));
  add(p + "out_proj.weight_packed", "U8", {16, 8}, out_packed.data(),
      out_packed.size());
  add(p + "out_proj.weight_scale", "F8_E4M3", {16, 1},
      out_scale.data(), out_scale.size());
  add(p + "out_proj.weight_global_scale", "F32", {1}, &weight_divisor,
      sizeof(weight_divisor));
  add(p + "out_proj.input_global_scale", "F32", {1}, &input_divisor,
      sizeof(input_divisor));
  const TensorResolver get = [&tensors](const std::string& name) -> const StTensor& {
    const auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("missing tensor: " + name);
    return it->second;
  };

  const vllm::GdnLayerWeights gdn = LoadQwen3_5DenseGdn(get, "layer.");
  CHECK(gdn.in_proj_b.Empty());
  CHECK(gdn.in_proj_a.Empty());
  REQUIRE_FALSE(gdn.in_proj_ba.Empty());
  CHECK(gdn.in_proj_ba.nk);
  CHECK(gdn.in_proj_ba.rank == 2);
  CHECK(gdn.in_proj_ba.shape[0] == 2 * kHeads);
  CHECK(gdn.in_proj_ba.shape[1] == kHidden);
  CHECK(gdn.in_proj_ba.bytes.size() ==
        (b.size() + a.size()) * sizeof(uint16_t));
  CHECK(std::memcmp(gdn.in_proj_ba.bytes.data(), b.data(),
                    b.size() * sizeof(uint16_t)) == 0);
  CHECK(std::memcmp(gdn.in_proj_ba.bytes.data() + b.size() * sizeof(uint16_t),
                    a.data(), a.size() * sizeof(uint16_t)) == 0);

  // W2: one physical in_proj_qkvz owner in exact [q,k,v,z] row order — the
  // checkpoint's in_proj_qkv already stacks q|k|v rows (vLLM stacked mapping
  // (0,1,2), qwen3_5.py:203-207); in_proj_z appends the z rows (shard 3). The
  // split fields stay empty (one-owner rule, no duplicate resident weights).
  CHECK(gdn.in_proj_qkv.Empty());
  CHECK(gdn.in_proj_z.Empty());
  REQUIRE_FALSE(gdn.in_proj_qkvz.Empty());
  CHECK(gdn.in_proj_qkvz.nk);
  CHECK(gdn.in_proj_qkvz.rank == 2);
  CHECK(gdn.in_proj_qkvz.shape[0] == 16 + 16);  // conv_dim + value_dim rows
  CHECK(gdn.in_proj_qkvz.shape[1] == kHidden);
  CHECK(gdn.in_proj_qkvz.bytes.size() ==
        (qkv.size() + z.size()) * sizeof(uint16_t));
  CHECK(std::memcmp(gdn.in_proj_qkvz.bytes.data(), qkv.data(),
                    qkv.size() * sizeof(uint16_t)) == 0);
  CHECK(std::memcmp(gdn.in_proj_qkvz.bytes.data() +
                        qkv.size() * sizeof(uint16_t),
                    z.data(), z.size() * sizeof(uint16_t)) == 0);
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

namespace {

// [in,out] Matmul-B bf16 -> raw torch Linear [out,in] with nk=true (the real
// 27B loader orientation; consumed via vt::MatmulBT).
OwnedTensor ToRawNK(const OwnedTensor& w) {
  OwnedTensor r;
  r.dtype = w.dtype;
  r.rank = 2;
  r.shape[0] = w.shape[1];
  r.shape[1] = w.shape[0];
  r.bytes.resize(w.bytes.size());
  const auto* src = reinterpret_cast<const uint16_t*>(w.bytes.data());
  auto* dst = reinterpret_cast<uint16_t*>(r.bytes.data());
  for (int64_t i = 0; i < w.shape[0]; ++i)
    for (int64_t o = 0; o < w.shape[1]; ++o)
      dst[o * w.shape[0] + i] = src[i * w.shape[1] + o];
  r.nk = true;
  return r;
}

// Concatenate raw-NK owners along their output rows (the merged owner layout).
OwnedTensor ConcatRawNK(const OwnedTensor& top, const OwnedTensor& bottom) {
  OwnedTensor r;
  r.dtype = top.dtype;
  r.rank = 2;
  r.shape[0] = top.shape[0] + bottom.shape[0];
  r.shape[1] = top.shape[1];
  r.bytes.resize(top.bytes.size() + bottom.bytes.size());
  std::memcpy(r.bytes.data(), top.bytes.data(), top.bytes.size());
  std::memcpy(r.bytes.data() + top.bytes.size(), bottom.bytes.data(),
              bottom.bytes.size());
  r.nk = true;
  return r;
}

}  // namespace

// The merged in_proj_qkvz owner must be a pure ownership change: on CPU (and
// with VT_GDN_MERGED_QKVZ=0 on CUDA) the forward slices the one owner into the
// exact q|k|v and z row ranges and issues the same two MatmulBT GEMMs as the
// split raw-NK fields — bit-identical logits, no duplicate resident weights.
// Mirrors qwen_gdn_linear_attn.py:929-936 (one physical projection, logical
// mixed_qkv/z views) with the split arm as the byte-exact reference.
TEST_CASE("qwen27 dense forward: merged qkvz owner equals split raw-NK fields") {
  const HfConfig c = MakeConfig();
  vt::Queue q = Q();
  std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};
  std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5};

  // Split arm: the same bytes as the packed arm, in raw-NK split fields.
  Qwen3_5DenseWeights split = MakeWeights(c);
  for (auto& layer : split.layers) {
    if (!layer.is_linear_attention) continue;
    layer.gdn.in_proj_qkv = ToRawNK(layer.gdn.in_proj_qkv);
    layer.gdn.in_proj_z = ToRawNK(layer.gdn.in_proj_z);
  }
  const std::vector<float> base =
      Qwen3_5DenseModel::ForwardDense(ids, pos, split, c, q);

  // Packed arm: one owner per layer, split fields EMPTY (one-owner rule).
  Qwen3_5DenseWeights packed = MakeWeights(c);
  for (auto& layer : packed.layers) {
    if (!layer.is_linear_attention) continue;
    layer.gdn.in_proj_qkvz = ConcatRawNK(ToRawNK(layer.gdn.in_proj_qkv),
                                         ToRawNK(layer.gdn.in_proj_z));
    layer.gdn.in_proj_qkv = OwnedTensor{};
    layer.gdn.in_proj_z = OwnedTensor{};
  }
  const std::vector<float> got =
      Qwen3_5DenseModel::ForwardDense(ids, pos, packed, c, q);

  REQUIRE(got.size() == base.size());
  double maxd = 0.0;
  for (size_t i = 0; i < base.size(); ++i)
    maxd = std::max(maxd, std::abs(static_cast<double>(base[i]) - got[i]));
  CHECK(maxd == 0.0);
}

// The gate ACTIVATION must reach the kernel, not just the config struct.
//
// Upstream qwen_gdn_linear_attn.py:452-464 @555967922 resolves
// `output_gate_type` and hands it to RMSNormGated as `activation=`; our
// vt::RmsNormGatedArgs::sigmoid_gate is the same switch. Every gate checkpoint
// we own today resolves to silu, so a silu-only corpus cannot see this wiring
// being absent -- a config-field assertion would pass with the model layer
// still hard-coding silu. This drives the sigmoid arm through the real dense
// forward (GdnBlock -> vt::RmsNormGated) and requires the logits to MOVE.
//
// Sensitivity note: silu(z) = z*sigmoid(z) and sigmoid(z) differ for every
// z != 1, and the synthetic z gate spans a non-degenerate range, so the two
// arms cannot coincide by construction.
TEST_CASE("qwen27 dense forward: output_gate_type=sigmoid reaches the GDN gate") {
  const HfConfig silu = MakeConfig();
  // The struct default is the upstream default; MakeConfig never sets it.
  REQUIRE(silu.output_gate_type == "silu");
  HfConfig sigmoid = MakeConfig();
  sigmoid.output_gate_type = "sigmoid";

  const Qwen3_5DenseWeights w = MakeWeights(silu);
  vt::Queue q = Q();
  const std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};
  const std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5};

  const std::vector<float> silu_logits =
      Qwen3_5DenseModel::ForwardDense(ids, pos, w, silu, q);
  const std::vector<float> sigmoid_logits =
      Qwen3_5DenseModel::ForwardDense(ids, pos, w, sigmoid, q);
  REQUIRE(sigmoid_logits.size() == silu_logits.size());

  double maxd = 0.0;
  for (size_t i = 0; i < silu_logits.size(); ++i) {
    CHECK(std::isfinite(sigmoid_logits[i]));
    maxd = std::max(maxd,
                    std::abs(static_cast<double>(silu_logits[i]) - sigmoid_logits[i]));
  }
  MESSAGE("silu vs sigmoid GDN gate moved logits by max|diff| = " << maxd);
  CHECK(maxd > 0.0);

  // Same config twice is deterministic: the difference above is the gate, not
  // run-to-run noise.
  const std::vector<float> silu_again =
      Qwen3_5DenseModel::ForwardDense(ids, pos, w, silu, q);
  double repeat = 0.0;
  for (size_t i = 0; i < silu_logits.size(); ++i)
    repeat = std::max(repeat,
                      std::abs(static_cast<double>(silu_logits[i]) - silu_again[i]));
  CHECK(repeat == 0.0);
}

namespace {

// A zero-filled copy of an existing weight: same dtype, rank and shape, all
// bytes 0 (so every bf16 element is +0.0f exactly).
OwnedTensor ZeroLike(const OwnedTensor& src) {
  OwnedTensor t;
  t.dtype = src.dtype;
  t.rank = src.rank;
  for (int i = 0; i < src.rank; ++i) t.shape[i] = src.shape[i];
  t.nk = src.nk;
  t.bytes.assign(src.bytes.size(), 0);
  return t;
}

}  // namespace

// WHICH activation, not merely "a different one".
//
// The case above proves the two arms DIFFER. It would pass just as happily
// with the boolean INVERTED -- a silu checkpoint driving the sigmoid kernel,
// which is exactly this row's bug class turned inside out. So the polarity
// needs a reference that does not go through our gate at all.
//
// It comes from arithmetic: silu(0) = 0 * sigmoid(0) = 0 EXACTLY, while
// sigmoid(0) = 0.5. The GDN gate input is z = h @ in_proj_z (a plain GEMM, no
// bias -- qwen3_5.cpp ProjectGdnQkvz), so zeroing in_proj_z makes z
// identically zero. Under a SILU gate the tail then computes
// norm(core) * 0 = 0 and the block returns 0 @ out_proj = 0, i.e. the GDN
// layers contribute exactly nothing -- indistinguishable from having zeroed
// out_proj itself, which is the independent reference this case compares
// against. Under a SIGMOID gate the same tail computes 0.5 * norm(core) and
// the layer contributes, so the two models must NOT agree.
//
// The reference model never consults output_gate_type, and the assertion is a
// bit-identity, not "they differ": inverting the resolution makes the silu arm
// diverge from it and the sigmoid arm coincide with it, so both halves flip.
TEST_CASE("qwen27 dense forward: the silu arm is silu, not merely 'not sigmoid'") {
  const HfConfig silu = MakeConfig();
  REQUIRE(silu.output_gate_type == "silu");
  HfConfig sigmoid = MakeConfig();
  sigmoid.output_gate_type = "sigmoid";

  // z ≡ 0 on every GDN layer: silu annihilates the gate, sigmoid does not.
  Qwen3_5DenseWeights zero_z = MakeWeights(silu);
  // The INDEPENDENT reference: the GDN blocks contribute nothing because their
  // out_proj is zero, whatever the gate does.
  Qwen3_5DenseWeights no_gdn = MakeWeights(silu);
  size_t gdn_layers = 0;
  for (size_t l = 0; l < zero_z.layers.size(); ++l) {
    if (!zero_z.layers[l].is_linear_attention) continue;
    ++gdn_layers;
    zero_z.layers[l].gdn.in_proj_z = ZeroLike(zero_z.layers[l].gdn.in_proj_z);
    no_gdn.layers[l].gdn.in_proj_z = ZeroLike(no_gdn.layers[l].gdn.in_proj_z);
    no_gdn.layers[l].gdn.out_proj = ZeroLike(no_gdn.layers[l].gdn.out_proj);
  }
  REQUIRE(gdn_layers == 3);  // the fixture's [LA, LA, LA, FA]

  vt::Queue q = Q();
  const std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};
  const std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5};

  const std::vector<float> ref =
      Qwen3_5DenseModel::ForwardDense(ids, pos, no_gdn, silu, q);
  const std::vector<float> silu_out =
      Qwen3_5DenseModel::ForwardDense(ids, pos, zero_z, silu, q);
  const std::vector<float> sigmoid_out =
      Qwen3_5DenseModel::ForwardDense(ids, pos, zero_z, sigmoid, q);
  REQUIRE(silu_out.size() == ref.size());
  REQUIRE(sigmoid_out.size() == ref.size());

  // silu(0) == 0: BIT-identical to the GDN-contributes-nothing reference.
  size_t silu_differs = 0;
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::memcmp(&ref[i], &silu_out[i], sizeof(float)) != 0) ++silu_differs;
  CAPTURE(silu_differs);
  CHECK(silu_differs == 0);

  // sigmoid(0) == 0.5: the same weights must NOT reduce to that reference, or
  // the fixture would be degenerate and the check above would prove nothing.
  double sigmoid_gap = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    sigmoid_gap = std::max(sigmoid_gap,
                           std::abs(static_cast<double>(ref[i]) - sigmoid_out[i]));
  MESSAGE("z=0 sigmoid arm departs from the no-GDN reference by " << sigmoid_gap);
  CHECK(sigmoid_gap > 0.0);
}

// FIX-PROBE-CANNOT-SAY-NO (#1258) ------------------------------------------
//
// #1256 named TWO always-true `TensorExists` stubs; #1257 fixed the
// `LoadQwen3_5DenseGdn` one. The other lives in the resolver-only overload of
// `LoadQwen3_5DenseLayer`, whose live caller is the `qwen36_gdn_layer_27b`
// isolated-layer golden (`tests/parity/test_op_parity.cpp`), SKIPped on a host
// without the pinned 27B snapshot and therefore invisible to every CPU lane.
//
// This bag builds an all-BF16 `linear_attention` layer: nothing in it is
// quantized, nothing carries a `weight_scale_inv`, and nothing disagrees with
// anything. A truthful probe routes it down the bf16 arm. A probe that cannot
// say no reports a block-wise FP8 scale on every projection and the load is
// refused over a tensor that does not exist.
namespace {

struct DenseLayerBag {
  std::unordered_map<std::string, StTensor> tensors;
  std::vector<std::vector<uint16_t>> storage;

  void AddBf16(const std::string& name, std::vector<int64_t> shape,
               uint16_t fill) {
    int64_t n = 1;
    for (const int64_t d : shape) n *= d;
    storage.emplace_back(static_cast<size_t>(n), fill);
    const std::vector<uint16_t>& v = storage.back();
    StTensor t;
    t.dtype = "BF16";
    t.shape = std::move(shape);
    t.data = reinterpret_cast<const uint8_t*>(v.data());
    t.nbytes = v.size() * sizeof(uint16_t);
    tensors[name] = t;
  }

  // Absence is signalled by THROWING, which is the whole contract a resolver
  // has: it returns a reference and so has no other way to say "not here"
  // (`SafetensorsFile::Get` documents the same, safetensors_reader.h).
  TensorResolver Resolver() const {
    return [this](const std::string& name) -> const StTensor& {
      const auto it = tensors.find(name);
      if (it == tensors.end()) throw std::runtime_error("missing tensor: " + name);
      return it->second;
    };
  }

  std::function<bool(const std::string&)> Has() const {
    return [this](const std::string& name) {
      return tensors.find(name) != tensors.end();
    };
  }
};

// An all-BF16 `linear_attention` layer under `{prefix}layers.{idx}.`.
DenseLayerBag Bf16LinearAttentionLayer(const std::string& base) {
  constexpr int64_t kH = 16, kI = 8, kHeads = 2, kConv = 16;
  DenseLayerBag bag;
  const std::string la = base + "linear_attn.";
  bag.AddBf16(base + "input_layernorm.weight", {kH}, 0x3f80);
  bag.AddBf16(base + "post_attention_layernorm.weight", {kH}, 0x3f80);
  bag.AddBf16(la + "in_proj_qkv.weight", {kConv, kH}, 0x3f80);
  bag.AddBf16(la + "in_proj_z.weight", {kConv, kH}, 0x3f00);
  bag.AddBf16(la + "in_proj_b.weight", {kHeads, kH}, 0x4000);
  bag.AddBf16(la + "in_proj_a.weight", {kHeads, kH}, 0x4100);
  bag.AddBf16(la + "conv1d.weight", {kConv, 1, 4}, 0x3e80);
  bag.AddBf16(la + "A_log", {kHeads}, 0x3f80);
  bag.AddBf16(la + "dt_bias", {kHeads}, 0x3f00);
  bag.AddBf16(la + "norm.weight", {8}, 0x3f80);
  bag.AddBf16(la + "out_proj.weight", {kH, kH}, 0x3f80);
  bag.AddBf16(base + "mlp.gate_proj.weight", {kI, kH}, 0x3f80);
  bag.AddBf16(base + "mlp.up_proj.weight", {kI, kH}, 0x3f00);
  bag.AddBf16(base + "mlp.down_proj.weight", {kH, kI}, 0x3f80);
  return bag;
}

bool Names(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// `e.what()`, or "" when the call returned. The M3 loader gate
// (test_fp8_block_weight_load.cpp) reports refusals the same way: "threw
// something" and "threw THIS" are different results and only one of them is
// evidence.
std::string LayerLoadFailure(const DenseLayerBag& bag,
                             const std::string& prefix) {
  try {
    const Qwen3_5DenseLayerWeights layer =
        vllm::LoadQwen3_5DenseLayer(bag.Resolver(), "linear_attention", 0, prefix);
    (void)layer;
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

}  // namespace

// T1 -------------------------------------------------------------------------
TEST_CASE("qwen27 dense layer: the resolver-only seam loads a bf16 layer without refusing") {
  const DenseLayerBag bag = Bf16LinearAttentionLayer("m.layers.0.");

  const std::string failure = LayerLoadFailure(bag, "m.");
  INFO("refusal was: " << failure);
  REQUIRE(failure.empty());

  const Qwen3_5DenseLayerWeights layer =
      vllm::LoadQwen3_5DenseLayer(bag.Resolver(), "linear_attention", 0, "m.");
  CHECK(layer.is_linear_attention);
  // Routed BF16, which is what this checkpoint is: one merged qkvz owner, one
  // merged gate_up owner, and every quantized slot left empty.
  CHECK_FALSE(layer.gdn.in_proj_qkvz.Empty());
  CHECK_FALSE(layer.gdn.out_proj.Empty());
  CHECK_FALSE(layer.mlp.gate_up_proj.Empty());
  CHECK(layer.gdn.in_proj_qkv_fp8_block.scale.Empty());
  CHECK(layer.gdn.out_proj_fp8_block.scale.Empty());
  CHECK(layer.mlp.gate_proj_fp8_block.scale.Empty());
  CHECK(layer.gdn.out_proj_fp4.packed.Empty());
  CHECK(layer.mlp.gate_proj_fp4.packed.Empty());

  // The SAME layer through the explicit-probe overload with a truthful probe
  // must land in the same arm, which is what makes the resolver-only seam a
  // shorthand rather than a second routing policy.
  const Qwen3_5DenseLayerWeights via_probe = vllm::LoadQwen3_5DenseLayer(
      bag.Resolver(), bag.Has(), "linear_attention", 0, "m.");
  CHECK(via_probe.gdn.in_proj_qkvz.bytes == layer.gdn.in_proj_qkvz.bytes);
  CHECK(via_probe.mlp.gate_up_proj.bytes == layer.mlp.gate_up_proj.bytes);
}

// T2 -------------------------------------------------------------------------
//
// The mutation of the chosen guard, run as a test rather than by hand: hand the
// explicit-probe overload the exact stub #1256 found twice, and require that it
// is refused for BEING a stub. Before this row it was refused too — naming
// `in_proj_qkv.weight_scale_inv`, a tensor the fixture does not have and never
// had, which is a report about the checkpoint for a defect in the caller.
TEST_CASE("qwen27 dense layer: a presence probe that cannot answer no is refused") {
  const DenseLayerBag bag = Bf16LinearAttentionLayer("m.layers.0.");
  const std::function<bool(const std::string&)> cannot_say_no =
      [](const std::string&) { return true; };

  std::string message;
  try {
    const Qwen3_5DenseLayerWeights layer = vllm::LoadQwen3_5DenseLayer(
        bag.Resolver(), cannot_say_no, "linear_attention", 0, "m.");
    (void)layer;
  } catch (const std::exception& e) {
    message = e.what();
  }
  INFO("refusal was: " << message);
  REQUIRE_FALSE(message.empty());
  // Refused for the right reason: it names the probe, the seam and the sentinel.
  CHECK(Names(message, "tensor-presence probe"));
  CHECK(Names(message, "LoadQwen3_5DenseLayer"));
  CHECK(Names(message, vllm::dense_loaders::kAbsentProbeSentinel));
  // And NOT for the wrong one — the old report blamed a checkpoint that is fine.
  CHECK_FALSE(Names(message, "weight_scale_inv"));

  // The same bag with a truthful probe loads, which is what makes the assertion
  // above a statement about the PROBE rather than about the fixture.
  const Qwen3_5DenseLayerWeights ok = vllm::LoadQwen3_5DenseLayer(
      bag.Resolver(), bag.Has(), "linear_attention", 0, "m.");
  CHECK_FALSE(ok.gdn.in_proj_qkvz.Empty());
}

// T3 -------------------------------------------------------------------------
//
// The negative control for both helpers, at the unit level: a resolver-derived
// probe answers BOTH ways, and a truthful probe passes the guard. Without this,
// T2 is satisfied by a guard that refuses every probe.
TEST_CASE("dense loaders: a resolver-derived presence probe answers both ways") {
  const DenseLayerBag bag = Bf16LinearAttentionLayer("m.layers.0.");
  const std::function<bool(const std::string&)> probe =
      vllm::dense_loaders::ProbeThroughResolver(bag.Resolver());

  CHECK(probe("m.layers.0.linear_attn.in_proj_qkv.weight"));
  CHECK_FALSE(probe("m.layers.0.linear_attn.in_proj_qkv.weight_scale_inv"));
  CHECK_FALSE(probe(vllm::dense_loaders::kAbsentProbeSentinel));

  // The probe OUTLIVES the temporary `bag.Resolver()` it was built from, which
  // it can only do because `ProbeThroughResolver` captures by VALUE. Stated
  // honestly: a by-reference capture would be undefined behaviour here rather
  // than a deterministic failure, so this line is a detector only under the
  // `sanitize-cpu` lanes. It is kept because that is exactly where a lifetime
  // bug in a loader helper should be caught.
  CHECK(probe("m.layers.0.mlp.gate_proj.weight"));

  vllm::dense_loaders::CheckProbeCanAnswerNo(probe, "T3 resolver-derived");
  vllm::dense_loaders::CheckProbeCanAnswerNo(bag.Has(), "T3 map-backed");
  CHECK(true);  // reached, so neither call above threw
}
