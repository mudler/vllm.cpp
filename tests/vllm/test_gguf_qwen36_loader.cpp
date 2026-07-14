// M0.10 Task 2: GGUF -> Qwen3_5MoeWeights loader + HfConfigFromGguf.
//
// Builds tiny SYNTHETIC qwen35moe GGUFs (gguf_builder.h) with the llama.cpp
// tensor names + metadata keys and asserts the loader (a) reads the right
// HfConfig, (b) maps every GGUF name to the right OwnedTensor with the same
// transpose the safetensors loader applies, (c) splits the stacked expert
// tensors per expert, (d) inverts the convert-time value transforms (norm w+1,
// ssm_a = -exp(A_log)) and (e) inverts the V-head grouped->tiled reorder when
// num_v_heads != num_k_heads. The real APEX GGUF end-to-end greedy parity vs
// the 35B safetensors is dgx-pending.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vt/dtype.h"

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

namespace {

// Little-endian f32 bytes for `n` values from fill(idx).
template <typename F>
std::string F32Data(int64_t n, F fill) {
  std::string s;
  s.reserve(static_cast<size_t>(n) * 4);
  for (int64_t i = 0; i < n; ++i) {
    const float v = fill(i);
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    for (int k = 0; k < 4; ++k) s.push_back(static_cast<char>((bits >> (8 * k)) & 0xff));
  }
  return s;
}

int64_t Prod(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

// Add an F32 tensor from a torch shape [out,in] (or [E,out,in]); ggml dims are
// the reverse, and torch-row-major data == ggml storage order.
template <typename F>
void AddF32(GgufModelBuilder& b, const std::string& name,
            const std::vector<int64_t>& torch_shape, F fill) {
  std::vector<uint64_t> ggml_dims;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it) {
    ggml_dims.push_back(static_cast<uint64_t>(*it));
  }
  b.AddTensor(name, ggml_dims, /*F32=*/0, F32Data(Prod(torch_shape), fill));
}

uint16_t Bf16(const vllm::OwnedTensor& t, int64_t i) {
  return reinterpret_cast<const uint16_t*>(t.bytes.data())[i];
}
float F32(const vllm::OwnedTensor& t, int64_t i) {
  return reinterpret_cast<const float*>(t.bytes.data())[i];
}

struct Dims {
  int64_t H = 8, vocab = 10;
  int64_t n_head = 2, n_head_kv = 1, head_dim = 4;
  int64_t E = 3, used = 2, I = 6, Is = 6;
  int64_t num_k = 2, num_v = 2, state = 4, conv_k = 3;
  int64_t n_layer = 2, full_interval = 2;
};

// A base value per tensor keeps cross-tensor values distinct; fill = base+idx.
void AddMoe(GgufModelBuilder& b, int64_t il, const Dims& d, float base) {
  const std::string p = "blk." + std::to_string(il) + ".";
  AddF32(b, p + "ffn_gate_inp.weight", {d.E, d.H}, [=](int64_t i) { return float(i); });
  AddF32(b, p + "ffn_gate_inp_shexp.weight", {d.H}, [=](int64_t i) { return base + i; });
  AddF32(b, p + "ffn_gate_exps.weight", {d.E, d.I, d.H}, [=](int64_t i) { return float(i); });
  AddF32(b, p + "ffn_up_exps.weight", {d.E, d.I, d.H}, [=](int64_t i) { return float(i) + 0.5F; });
  AddF32(b, p + "ffn_down_exps.weight", {d.E, d.H, d.I}, [=](int64_t i) { return float(i); });
  AddF32(b, p + "ffn_gate_shexp.weight", {d.Is, d.H}, [=](int64_t i) { return base + i; });
  AddF32(b, p + "ffn_up_shexp.weight", {d.Is, d.H}, [=](int64_t i) { return base + i; });
  AddF32(b, p + "ffn_down_shexp.weight", {d.H, d.Is}, [=](int64_t i) { return base + i; });
}

std::string BuildGguf(const Dims& d) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen35moe"));
  b.AddKv(U32Kv("qwen35moe.embedding_length", d.H));
  b.AddKv(U32Kv("qwen35moe.block_count", d.n_layer));
  b.AddKv(U32Kv("qwen35moe.attention.head_count", d.n_head));
  b.AddKv(U32Kv("qwen35moe.attention.head_count_kv", d.n_head_kv));
  b.AddKv(U32Kv("qwen35moe.attention.key_length", d.head_dim));
  b.AddKv(U32Kv("qwen35moe.expert_count", d.E));
  b.AddKv(U32Kv("qwen35moe.expert_used_count", d.used));
  b.AddKv(U32Kv("qwen35moe.expert_feed_forward_length", d.I));
  b.AddKv(U32Kv("qwen35moe.expert_shared_feed_forward_length", d.Is));
  b.AddKv(U32Kv("qwen35moe.ssm.group_count", d.num_k));
  b.AddKv(U32Kv("qwen35moe.ssm.time_step_rank", d.num_v));
  b.AddKv(U32Kv("qwen35moe.ssm.state_size", d.state));
  b.AddKv(U32Kv("qwen35moe.ssm.conv_kernel", d.conv_k));
  b.AddKv(F32Kv("qwen35moe.rope.freq_base", 1000000.0F));
  b.AddKv(F32Kv("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(U32Kv("qwen35moe.full_attention_interval", d.full_interval));
  b.AddKv(U32Kv("qwen35moe.context_length", 40960));

  const int64_t key_dim = d.num_k * d.state;
  const int64_t value_dim = d.num_v * d.state;
  const int64_t conv_dim = 2 * key_dim + value_dim;

  // Model level. token_embd raw; output_norm/output convert-transformed.
  AddF32(b, "token_embd.weight", {d.vocab, d.H}, [](int64_t i) { return float(i); });
  AddF32(b, "output_norm.weight", {d.H}, [](int64_t i) { return 5.0F + i; });  // raw w = 4+i
  AddF32(b, "output.weight", {d.vocab, d.H}, [](int64_t i) { return float(i); });

  for (int64_t il = 0; il < d.n_layer; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const bool linear = ((il + 1) % d.full_interval) != 0;
    AddF32(b, p + "attn_norm.weight", {d.H}, [](int64_t i) { return 5.0F + i; });
    AddF32(b, p + "post_attention_norm.weight", {d.H}, [](int64_t i) { return 3.0F + i; });
    if (linear) {
      AddF32(b, p + "attn_qkv.weight", {conv_dim, d.H}, [](int64_t i) { return float(i); });
      AddF32(b, p + "attn_gate.weight", {value_dim, d.H}, [](int64_t i) { return float(i); });
      AddF32(b, p + "ssm_beta.weight", {d.num_v, d.H}, [](int64_t i) { return float(i); });
      AddF32(b, p + "ssm_alpha.weight", {d.num_v, d.H}, [](int64_t i) { return float(i); });
      AddF32(b, p + "ssm_conv1d.weight", {conv_dim, d.conv_k}, [](int64_t i) { return float(i); });
      AddF32(b, p + "ssm_out.weight", {d.H, value_dim}, [](int64_t i) { return float(i); });
      // ssm_a = -exp(known); known = i (recover a_log = i).
      AddF32(b, p + "ssm_a", {d.num_v}, [](int64_t i) { return -std::exp(float(i)); });
      AddF32(b, p + "ssm_dt.bias", {d.num_v}, [](int64_t i) { return 2.0F + i; });
      AddF32(b, p + "ssm_norm.weight", {d.state}, [](int64_t i) { return 7.0F + i; });  // raw
    } else {
      AddF32(b, p + "attn_q.weight", {2 * d.n_head * d.head_dim, d.H}, [](int64_t i) { return float(i); });
      AddF32(b, p + "attn_k.weight", {d.n_head_kv * d.head_dim, d.H}, [](int64_t i) { return float(i); });
      AddF32(b, p + "attn_v.weight", {d.n_head_kv * d.head_dim, d.H}, [](int64_t i) { return float(i); });
      AddF32(b, p + "attn_output.weight", {d.H, d.n_head * d.head_dim}, [](int64_t i) { return float(i); });
      AddF32(b, p + "attn_q_norm.weight", {d.head_dim}, [](int64_t i) { return 5.0F + i; });
      AddF32(b, p + "attn_k_norm.weight", {d.head_dim}, [](int64_t i) { return 6.0F + i; });
    }
    AddMoe(b, il, d, 100.0F);
  }
  return b.Build();
}

}  // namespace

TEST_CASE("HfConfigFromGguf reads the qwen35moe hparams") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);

  CHECK(c.model_type == "qwen35moe");
  REQUIRE(c.architectures.size() == 1);
  CHECK(c.architectures[0] == "Qwen3_5MoeForConditionalGeneration");
  CHECK(c.hidden_size == 8);
  CHECK(c.num_hidden_layers == 2);
  CHECK(c.num_attention_heads == 2);
  CHECK(c.num_key_value_heads == 1);
  CHECK(c.head_dim == 4);
  CHECK(c.vocab_size == 10);  // from token_embd (kv absent)
  CHECK(c.num_experts == 3);
  CHECK(c.num_experts_per_tok == 2);
  CHECK(c.moe_intermediate_size == 6);
  CHECK(c.shared_expert_intermediate_size == 6);
  CHECK(c.linear_num_key_heads == 2);
  CHECK(c.linear_num_value_heads == 2);
  CHECK(c.linear_key_head_dim == 4);
  CHECK(c.linear_value_head_dim == 4);
  CHECK(c.linear_conv_kernel_dim == 3);
  CHECK(c.rope_theta == doctest::Approx(1000000.0));
  CHECK(c.rms_norm_eps == doctest::Approx(1e-6));
  CHECK(c.max_position_embeddings == 40960);
  // full_attention_interval=2 -> layer0 linear (GDN), layer1 full attention.
  REQUIRE(c.layer_types.size() == 2);
  CHECK(c.layer_types[0] == "linear_attention");
  CHECK(c.layer_types[1] == "full_attention");
}

TEST_CASE("LoadQwen3_5MoeFromGguf: names, transposes, expert split, transforms") {
  Dims d;
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
  const vllm::Qwen3_5MoeWeights w = vllm::LoadQwen3_5MoeFromGguf(g, c);

  SUBCASE("model-level tensors") {
    // embed_tokens [vocab, H] direct.
    REQUIRE(w.embed_tokens.rank == 2);
    CHECK(w.embed_tokens.shape[0] == 10);
    CHECK(w.embed_tokens.shape[1] == 8);
    CHECK(Bf16(w.embed_tokens, 13) == vt::F32ToBF16(13.0F));
    // final_norm bf16 [H]: convert stored (w+1), loader returns raw w = 4+i.
    CHECK(w.final_norm.shape[0] == 8);
    CHECK(Bf16(w.final_norm, 0) == vt::F32ToBF16(4.0F));
    CHECK(Bf16(w.final_norm, 3) == vt::F32ToBF16(7.0F));
    // lm_head [H, vocab] transposed from output [vocab, H].
    CHECK(w.lm_head.shape[0] == 8);
    CHECK(w.lm_head.shape[1] == 10);
    // output[vocab=2,H=1] = 2*8+1 = 17 -> lm_head[H=1,vocab=2] index 1*10+2=12.
    CHECK(Bf16(w.lm_head, 1 * 10 + 2) == vt::F32ToBF16(17.0F));
  }

  SUBCASE("GDN layer 0: mapping + transposes + value transforms") {
    const auto& L = w.layers[0];
    REQUIRE(L.is_linear_attention);
    // input/post norms bf16 [H], raw = stored - 1.
    CHECK(Bf16(L.input_layernorm, 2) == vt::F32ToBF16(4.0F + 2));   // stored 5+i
    CHECK(Bf16(L.post_attention_layernorm, 1) == vt::F32ToBF16(2.0F + 1));  // 3+i
    // in_proj_qkv [H=8, conv_dim=24] (from attn_qkv [conv_dim, H]).
    CHECK(L.gdn.in_proj_qkv.shape[0] == 8);
    CHECK(L.gdn.in_proj_qkv.shape[1] == 24);
    // attn_qkv[o=2,i=1] = 2*8+1 = 17 -> transposed[i=1,o=2] index 1*24+2 = 26.
    CHECK(Bf16(L.gdn.in_proj_qkv, 1 * 24 + 2) == vt::F32ToBF16(17.0F));
    // in_proj_z [H, value_dim=8]; in_proj_a/b [H, num_v=2].
    CHECK(L.gdn.in_proj_z.shape[1] == 8);
    // GGUF retains its transformed [H,Hv] split owners; the 27B safetensors
    // packed BA path must remain unselectable for every GGUF model.
    CHECK(L.gdn.in_proj_ba.Empty());
    REQUIRE_FALSE(L.gdn.in_proj_b.Empty());
    REQUIRE_FALSE(L.gdn.in_proj_a.Empty());
    CHECK(L.gdn.in_proj_b.shape[0] == 8);
    CHECK(L.gdn.in_proj_b.shape[1] == 2);
    CHECK(L.gdn.in_proj_a.shape[1] == 2);
    // conv1d [conv_dim=24, K=3] direct (NOT transposed).
    CHECK(L.gdn.conv1d_weight.shape[0] == 24);
    CHECK(L.gdn.conv1d_weight.shape[1] == 3);
    CHECK(Bf16(L.gdn.conv1d_weight, 7) == vt::F32ToBF16(7.0F));
    // out_proj [value_dim=8, H=8] (from ssm_out [H, value_dim]).
    CHECK(L.gdn.out_proj.shape[0] == 8);
    CHECK(L.gdn.out_proj.shape[1] == 8);
    // a_log f32 [num_v=2] = log(-ssm_a) = known index.
    CHECK(L.gdn.a_log.dtype == vt::DType::kF32);
    CHECK(L.gdn.a_log.shape[0] == 2);
    CHECK(F32(L.gdn.a_log, 0) == doctest::Approx(0.0F));
    CHECK(F32(L.gdn.a_log, 1) == doctest::Approx(1.0F));
    // dt_bias f32 [2] direct (2+i).
    CHECK(L.gdn.dt_bias.dtype == vt::DType::kF32);
    CHECK(F32(L.gdn.dt_bias, 1) == doctest::Approx(3.0F));
    // GDN norm_weight bf16 [Dv=4] RAW (convert does NOT add 1): 7+i.
    CHECK(L.gdn.norm_weight.shape[0] == 4);
    CHECK(Bf16(L.gdn.norm_weight, 0) == vt::F32ToBF16(7.0F));
    CHECK(Bf16(L.gdn.norm_weight, 2) == vt::F32ToBF16(9.0F));
  }

  SUBCASE("full-attn layer 1: mapping + transposes") {
    const auto& L = w.layers[1];
    REQUIRE_FALSE(L.is_linear_attention);
    // q_proj [H=8, 2*Hq*Dh=16]; k/v [H, Hkv*Dh=4]; o_proj [Hq*Dh=8, H=8].
    CHECK(L.attn.q_proj.shape[0] == 8);
    CHECK(L.attn.q_proj.shape[1] == 16);
    CHECK(L.attn.k_proj.shape[1] == 4);
    CHECK(L.attn.v_proj.shape[1] == 4);
    CHECK(L.attn.o_proj.shape[0] == 8);
    CHECK(L.attn.o_proj.shape[1] == 8);
    // q_norm/k_norm bf16 [Dh=4], raw = stored - 1 (q: 5+i, k: 6+i).
    CHECK(Bf16(L.attn.q_norm, 0) == vt::F32ToBF16(4.0F));
    CHECK(Bf16(L.attn.k_norm, 1) == vt::F32ToBF16(6.0F));  // (6+1)-1
  }

  SUBCASE("MoE: router, shared gate, per-expert split, transposes") {
    const auto& M = w.layers[0].moe;
    // router_gate [H=8, E=3] (from ffn_gate_inp [E, H]).
    CHECK(M.router_gate.shape[0] == 8);
    CHECK(M.router_gate.shape[1] == 3);
    // ffn_gate_inp[e=2,h=1] = 2*8+1 = 17 -> router[h=1,e=2] index 1*3+2 = 5.
    CHECK(Bf16(M.router_gate, 1 * 3 + 2) == vt::F32ToBF16(17.0F));
    // shared_gate [H, 1].
    CHECK(M.shared_gate.shape[0] == 8);
    CHECK(M.shared_gate.shape[1] == 1);
    // Expert split: E owned tensors each [H, I] (gate/up), [I, H] (down).
    REQUIRE(M.expert_gate.size() == 3);
    REQUIRE(M.expert_up.size() == 3);
    REQUIRE(M.expert_down.size() == 3);
    CHECK(M.expert_gate[0].shape[0] == 8);
    CHECK(M.expert_gate[0].shape[1] == 6);
    CHECK(M.expert_down[0].shape[0] == 6);
    CHECK(M.expert_down[0].shape[1] == 8);
    // ffn_gate_exps torch [E,I,H]; expert e stem [I=6,H=8]. For e=1, [o,i] =
    // e*(I*H) + o*H + i. Pick o=2,i=3 -> 1*48 + 2*8 + 3 = 67. Transposed to
    // [H,I]: [i=3,o=2] index 3*6+2 = 20.
    CHECK(Bf16(M.expert_gate[1], 3 * 6 + 2) == vt::F32ToBF16(67.0F));
    // up uses +0.5 fill: same position for e=1.
    CHECK(Bf16(M.expert_up[1], 3 * 6 + 2) == vt::F32ToBF16(67.5F));
    // shared gate/up/down proj transposed.
    CHECK(M.shared_gate_proj.shape[0] == 8);   // [H, Is]
    CHECK(M.shared_gate_proj.shape[1] == 6);
    CHECK(M.shared_down_proj.shape[0] == 6);   // [Is, H]
    CHECK(M.shared_down_proj.shape[1] == 8);
  }
}

TEST_CASE("LoadQwen3_5MoeFromGguf: V-head reorder when num_v != num_k") {
  // num_k=2, num_v=4 -> num_v_per_k=2. GGUF stores V heads TILED; the loader
  // recovers HF GROUPED order: grouped[g=k*2+r] = tiled[t=r*2+k]:
  //   g0<-t0, g1<-t2, g2<-t1, g3<-t3.
  Dims d;
  d.num_k = 2;
  d.num_v = 4;
  d.state = 2;  // Dk=Dv=2; value_dim=8, key_dim=4, conv_dim=16
  d.n_layer = 1;
  d.full_interval = 4;  // layer 0 -> linear (GDN)
  TempFile f(BuildGguf(d));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig c = vllm::HfConfigFromGguf(g);
  REQUIRE(c.layer_types.size() == 1);
  REQUIRE(c.layer_types[0] == "linear_attention");
  const vllm::Qwen3_5MoeWeights w = vllm::LoadQwen3_5MoeFromGguf(g, c);
  const auto& gdn = w.layers[0].gdn;

  // ssm_a tiled = -exp(t); a_log grouped = [t0,t2,t1,t3] = [0,2,1,3].
  REQUIRE(gdn.a_log.shape[0] == 4);
  CHECK(F32(gdn.a_log, 0) == doctest::Approx(0.0F));
  CHECK(F32(gdn.a_log, 1) == doctest::Approx(2.0F));
  CHECK(F32(gdn.a_log, 2) == doctest::Approx(1.0F));
  CHECK(F32(gdn.a_log, 3) == doctest::Approx(3.0F));
  // dt_bias tiled = 2+t; grouped = [2,4,3,5].
  CHECK(F32(gdn.dt_bias, 1) == doctest::Approx(4.0F));
  CHECK(F32(gdn.dt_bias, 2) == doctest::Approx(3.0F));
}
