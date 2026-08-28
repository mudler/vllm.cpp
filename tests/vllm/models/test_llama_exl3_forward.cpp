// The EXL3 arm REACHED from the dense forward — QUANT-EXL3 W1b (#2181).
//
// WHY THIS FILE EXISTS. A fresh review deleted every production call site of
// the EXL3 arm — both `AttnBlock` branches, both `MlpBlock` factory arms and the
// `lm_head_exl3` branch — and the whole declared gate stayed GREEN. The arm was
// reachable, and a manual `vllm-cli` run proved it, but nothing automated
// measured the capability: exactly what `.agents/reachability.md` says a
// class-level gate does instead of a capability-level one.
//
// The gate is an EQUIVALENCE. The same tiny model is built twice from the SAME
// bytes: once with the trellis in the EXL3 fields, and once with those weights
// DECODED into the bf16 fields. Both go through `Qwen3DenseModel::Forward` and
// must agree. Deleting an EXL3 call site does not merely change the numbers —
// the arm falls through to a bf16 field that an EXL3 load leaves EMPTY, and
// `ResidentWeight` refuses it by name — so the case reds either way.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vllm::Exl3Weight;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::Qwen3DenseWeights;
using vllm::v1::CommonAttentionMetadata;
using vt::DType;

struct Rng {
  uint32_t s = 12345;
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  float unit() { return static_cast<float>(next() % 2000) / 1000.0f - 1.0f; }
};

OwnedTensor MakeBf16(const std::vector<int64_t>& shape, bool nk, uint32_t seed, float scale = 1.0f) {
  Rng r;
  r.s = seed | 1u;
  OwnedTensor t;
  t.dtype = DType::kBF16;
  t.nk = nk;
  t.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    t.shape[i] = shape[i];
    numel *= shape[i];
  }
  std::vector<uint8_t> b(static_cast<size_t>(numel) * 2);
  auto* p = reinterpret_cast<uint16_t*>(b.data());
  for (int64_t i = 0; i < numel; ++i) p[i] = vt::F32ToBF16(r.unit() * scale);
  t.bytes = vllm::OwnedBytes(std::move(b));
  return t;
}

OwnedTensor MakeOwnedFrom(DType dt, const std::vector<int64_t>& shape,
                          const std::vector<uint8_t>& bytes) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
  t.bytes = vllm::OwnedBytes(bytes);
  return t;
}

// One synthetic EXL3 projection [k, n] at `bits`, plus the SAME weights decoded
// into a bf16 raw-NK [n, k] operand. The two are the same bytes read two ways,
// which is what makes the forward comparison an equivalence rather than a
// tolerance on two different models.
struct Pair {
  Exl3Weight exl3;
  OwnedTensor bf16;  // raw-NK [n=out, k=in], what MatmulBT consumes
};

Pair MakePair(int64_t k, int64_t n, int bits, int codebook, uint32_t seed) {
  Rng r;
  r.s = seed | 1u;
  const int64_t words = static_cast<int64_t>(k / 16) * (n / 16) * 16 * bits;
  std::vector<uint16_t> trellis(static_cast<size_t>(words));
  for (auto& w : trellis) w = static_cast<uint16_t>(r.next() & 0xffffu);
  std::vector<uint16_t> suh(static_cast<size_t>(k)), svh(static_cast<size_t>(n));
  // Sign vectors, which is what they are: +-1 in fp16.
  for (auto& v : suh) v = vt::F32ToF16((r.next() & 1u) ? 1.0f : -1.0f);
  for (auto& v : svh) v = vt::F32ToF16((r.next() & 1u) ? 1.0f : -1.0f);

  Pair p;
  const auto as_bytes = [](const std::vector<uint16_t>& v) {
    return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(v.data()),
                                reinterpret_cast<const uint8_t*>(v.data()) + v.size() * 2);
  };
  p.exl3.trellis =
      MakeOwnedFrom(DType::kI8, {k / 16, n / 16, 32 * bits}, as_bytes(trellis));
  p.exl3.suh = MakeOwnedFrom(DType::kF16, {k}, as_bytes(suh));
  p.exl3.svh = MakeOwnedFrom(DType::kF16, {n}, as_bytes(svh));
  p.exl3.codebook = codebook;

  // The decoded twin. `Exl3DequantLinear` yields [k, n]; MatmulBT wants raw-NK
  // [n, k], so this transposes on the way out — the ONE place the two arms'
  // orientations are reconciled, and getting it wrong shows up immediately as a
  // failed equivalence rather than as a plausible wrong number.
  std::vector<float> w(static_cast<size_t>(k) * n, 0.0f);
  vt::Exl3DequantLinear(trellis.data(), suh.data(), svh.data(), k, n, bits, codebook, w.data());
  std::vector<uint8_t> b(static_cast<size_t>(k) * n * 2);
  auto* q = reinterpret_cast<uint16_t*>(b.data());
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < k; ++j)
      q[i * k + j] = vt::F32ToBF16(w[static_cast<size_t>(j) * n + i]);
  p.bf16 = MakeOwnedFrom(DType::kBF16, {n, k}, b);
  p.bf16.nk = true;
  return p;
}

HfConfig TinyConfig() {
  HfConfig c;
  c.num_hidden_layers = 2;
  // EVERY projection's k AND n must be a multiple of 128: each side was
  // Hadamard-128 transformed at quantization time (`exl3_lib/quantize.py:15`),
  // so the reference dequant refuses anything else. That constrains the tiny
  // model more than a bf16 one -- in particular `num_key_value_heads * head_dim`
  // is a projection width and cannot be the usual small GQA number.
  c.hidden_size = 128;
  c.num_attention_heads = 4;   // qdim  = 4 * 64 = 256
  c.num_key_value_heads = 2;   // kvdim = 2 * 64 = 128
  c.head_dim = 64;
  c.rotary_dim = 64;
  c.intermediate_size = 128;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 1000000.0;
  c.vocab_size = 128;
  return c;
}

struct CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<PagedKvCache> attn_kv;
  CachePool(const HfConfig& c, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l)
      buf.emplace_back(static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0.0f);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
  }
};

CommonAttentionMetadata PrefillMeta(int32_t n_tokens) {
  CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = n_tokens;
  am.query_start_loc = {0, n_tokens};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {n_tokens};
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = n_tokens;
  am.max_seq_len = n_tokens;
  am.block_table_num_cols = 1;
  am.block_table_tensor = {0};
  am.slot_mapping.resize(static_cast<size_t>(n_tokens));
  for (int32_t i = 0; i < n_tokens; ++i) am.slot_mapping[i] = i;
  am.causal = true;
  return am;
}

// Both containers, from ONE set of trellis bytes. `exl3` populates only the
// EXL3 fields; `dense` populates only the bf16 ones, from the decode.
void BuildBoth(const HfConfig& c, Qwen3DenseWeights* exl3, Qwen3DenseWeights* dense) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads, Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim, I = c.intermediate_size, V = c.vocab_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;

  for (Qwen3DenseWeights* w : {exl3, dense}) {
    w->tie_word_embeddings = false;
    w->attention_bias = false;
    w->embed_tokens = MakeBf16({V, H}, false, 1);
    w->final_norm = MakeBf16({H}, false, 2, 0.5f);
  }

  uint32_t seed = 700;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    vllm::Qwen3DenseLayerWeights le, ld;
    le.input_layernorm = ld.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    le.post_attention_layernorm = ld.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);

    const Pair q = MakePair(H, qdim, 3, 0, seed++);
    const Pair k = MakePair(H, kdim, 3, 0, seed++);
    const Pair v = MakePair(H, kdim, 3, 0, seed++);
    const Pair o = MakePair(qdim, H, 3, 0, seed++);
    const Pair g = MakePair(H, I, 3, 0, seed++);
    const Pair u = MakePair(H, I, 3, 0, seed++);
    const Pair d = MakePair(I, H, 3, 0, seed++);

    le.attn.q_proj_exl3 = q.exl3;
    le.attn.k_proj_exl3 = k.exl3;
    le.attn.v_proj_exl3 = v.exl3;
    le.attn.o_proj_exl3 = o.exl3;
    le.mlp.gate_proj_exl3 = g.exl3;
    le.mlp.up_proj_exl3 = u.exl3;
    le.mlp.down_proj_exl3 = d.exl3;

    // The dense twin merges q|k|v and gate|up the way the bf16 loader does.
    std::vector<uint8_t> qkv;
    for (const OwnedTensor* t : {&q.bf16, &k.bf16, &v.bf16})
      qkv.insert(qkv.end(), t->bytes.data(), t->bytes.data() + t->bytes.size());
    ld.attn.qkv_proj = MakeOwnedFrom(DType::kBF16, {qdim + 2 * kdim, H}, qkv);
    ld.attn.qkv_proj.nk = true;
    ld.attn.o_proj = o.bf16;
    std::vector<uint8_t> gu;
    for (const OwnedTensor* t : {&g.bf16, &u.bf16})
      gu.insert(gu.end(), t->bytes.data(), t->bytes.data() + t->bytes.size());
    ld.mlp.gate_up_proj = MakeOwnedFrom(DType::kBF16, {2 * I, H}, gu);
    ld.mlp.gate_up_proj.nk = true;
    ld.mlp.down_proj = d.bf16;

    exl3->layers.push_back(std::move(le));
    dense->layers.push_back(std::move(ld));
  }

  const Pair head = MakePair(H, V, 3, 0, 999);
  exl3->lm_head_exl3 = head.exl3;
  // Matmul-B [H, vocab] for the bf16 arm, which is the decode untransposed.
  std::vector<float> hw(static_cast<size_t>(H) * V, 0.0f);
  vt::Exl3DequantLinear(reinterpret_cast<const uint16_t*>(head.exl3.trellis.bytes.data()),
                        reinterpret_cast<const uint16_t*>(head.exl3.suh.bytes.data()),
                        reinterpret_cast<const uint16_t*>(head.exl3.svh.bytes.data()), H, V, 3, 0,
                        hw.data());
  std::vector<uint8_t> hb(static_cast<size_t>(H) * V * 2);
  auto* hp = reinterpret_cast<uint16_t*>(hb.data());
  for (size_t i = 0; i < hw.size(); ++i) hp[i] = vt::F32ToBF16(hw[i]);
  dense->lm_head = MakeOwnedFrom(DType::kBF16, {H, V}, hb);
}

}  // namespace

TEST_CASE("llama exl3 forward: the EXL3 arm is REACHED and agrees with its decoded twin") {
  const HfConfig c = TinyConfig();
  Qwen3DenseWeights wq, wd;
  BuildBoth(c, &wq, &wd);
  REQUIRE(wq.layers[0].attn.IsExl3());
  REQUIRE(wq.layers[0].mlp.IsExl3());
  REQUIRE(wd.layers[0].attn.IsExl3() == false);

  const std::vector<int32_t> tokens = {3, 17, 42, 5};
  const std::vector<int32_t> positions = {0, 1, 2, 3};
  const CommonAttentionMetadata meta = PrefillMeta(4);

  vt::Queue q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue();
  CachePool pe(c, 4, 8), pd(c, 4, 8);
  const std::vector<float> le =
      vllm::Qwen3DenseModel::Forward(tokens, positions, meta, pe.attn_kv, wq, c, q);
  const std::vector<float> ld =
      vllm::Qwen3DenseModel::Forward(tokens, positions, meta, pd.attn_kv, wd, c, q);

  REQUIRE(le.size() == ld.size());
  REQUIRE(!le.empty());
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < le.size(); ++i) {
    const double d = static_cast<double>(le[i]) - ld[i];
    num += d * d;
    den += static_cast<double>(ld[i]) * ld[i];
  }
  const double rel = std::sqrt(num / den);
  MESSAGE("exl3 forward vs decoded-bf16 forward: rel_rms = ", rel);
  // The two arms are the same weights through different kernels: EXL3 rides the
  // Hadamards on the activations while the bf16 twin has them baked in, and the
  // twin rounds the decode to bf16. So this is a bound, not an equality.
  CHECK(rel <= 5.0e-2);
  // NOT VACUOUS: a forward returning zeros would pass any relative bound.
  REQUIRE(den > 0.0);
  for (float x : le) REQUIRE(std::isfinite(x));

  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(q);
}
