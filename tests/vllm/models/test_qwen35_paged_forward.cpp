// M1.8 Task 3 — batched PAGED forward parity (THE central refactor).
//
// Validates Qwen3_5Model::Forward (paged) against the retained M0.9 dense
// reference Qwen3_5Model::ForwardDense on a small SYNTHETIC MoE model (CPU;
// the real 35B paged greedy gate re-runs on dgx — dgx-pending). Three gates:
//   1. PAGED == DENSE anchor: a batch-of-1 full prefill of a short sequence
//      through the paged path (vt::ReshapeAndCache + vt::PagedAttention +
//      GDN over persistent-but-fresh mamba state) equals the dense forward on
//      the same input (per-token final logits).
//   2. DECODE via KV cache: prefill a sequence, then a single-token DECODE step
//      reading the prefilled K/V + persisted GDN state equals the dense forward
//      over the full sequence (last position) — proves the paged READ + the KV/
//      mamba-state growth work.
//   3. GDN fresh-vs-continuing zeroing: a mixed batch of two fresh prefills
//      where one request's mamba block is pre-seeded with GARBAGE; its output
//      must be identical to running it alone — proves the GDN-state ZEROING
//      (qwen_gdn_linear_attn.py:1513-1514) is wired.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::Qwen3_5Model;
using vllm::Qwen3_5MoeWeights;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;

namespace {

// splitmix64-based small deterministic weight values in [-0.08, 0.08).
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

HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // layer_types [LA, LA, LA, FA]
  c.vocab_size = 40;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 16;
  c.shared_expert_intermediate_size = 16;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  return c;
}

vllm::MoeBlockWeights MakeMoe(const HfConfig& c, uint64_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeOwned(DType::kBF16, {H, E}, s + 1);
  m.shared_gate = MakeOwned(DType::kBF16, {H, 1}, s + 2);
  for (int64_t e = 0; e < E; ++e) {
    m.expert_gate.push_back(MakeOwned(DType::kBF16, {H, I}, s + 100 + e * 7));
    m.expert_up.push_back(MakeOwned(DType::kBF16, {H, I}, s + 200 + e * 7));
    m.expert_down.push_back(MakeOwned(DType::kBF16, {I, H}, s + 300 + e * 7));
  }
  m.shared_gate_proj = MakeOwned(DType::kBF16, {H, Is}, s + 3);
  m.shared_up_proj = MakeOwned(DType::kBF16, {H, Is}, s + 4);
  m.shared_down_proj = MakeOwned(DType::kBF16, {Is, H}, s + 5);
  return m;
}

Qwen3_5MoeWeights MakeWeights(const HfConfig& c) {
  Qwen3_5MoeWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
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
    vllm::Qwen3_5MoeLayerWeights lw;
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
    lw.moe = MakeMoe(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// --- KV / mamba-state cache pool for one step through the paged Forward. Owns
// the host buffers and hands out PagedKvCache / GdnStateCache views. ---
struct CachePool {
  const HfConfig& c;
  int64_t num_blocks;
  int64_t block_size;
  std::vector<std::vector<float>> full_attn_buf;   // per full-attn layer
  std::vector<std::vector<float>> gdn_ssm_buf;     // per GDN layer
  std::vector<std::vector<float>> gdn_conv_buf;    // per GDN layer
  std::vector<PagedKvCache> attn_kv;
  std::vector<GdnStateCache> gdn_state;

  CachePool(const HfConfig& cfg, int64_t nb, int64_t bs)
      : c(cfg), num_blocks(nb), block_size(bs) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    const int64_t Hv = c.linear_num_value_heads, Dv = c.linear_value_head_dim,
                  Dk = c.linear_key_head_dim, Kw = c.linear_conv_kernel_dim;
    const int64_t key_dim = c.linear_num_key_heads * Dk, value_dim = Hv * Dv;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      if (c.layer_types[static_cast<size_t>(l)] == "linear_attention") {
        gdn_ssm_buf.emplace_back(static_cast<size_t>(nb * Hv * Dv * Dk), 0.0f);
        gdn_conv_buf.emplace_back(static_cast<size_t>(nb * conv_dim * (Kw - 1)), 0.0f);
      } else {
        full_attn_buf.emplace_back(static_cast<size_t>(nb * 2 * bs * Hkv * Dh), 0.0f);
      }
    }
    Rebind();
  }

  // (Re)build the views after the buffers exist (vectors don't reallocate after
  // construction here, but Rebind keeps the views authoritative).
  void Rebind() {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    const int64_t Hv = c.linear_num_value_heads, Dv = c.linear_value_head_dim,
                  Dk = c.linear_key_head_dim, Kw = c.linear_conv_kernel_dim;
    const int64_t key_dim = c.linear_num_key_heads * Dk, value_dim = Hv * Dv;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    attn_kv.clear();
    gdn_state.clear();
    for (auto& b : full_attn_buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
    for (size_t g = 0; g < gdn_ssm_buf.size(); ++g) {
      GdnStateCache gs;
      gs.ssm_state = vt::Tensor::Contiguous(gdn_ssm_buf[g].data(), DType::kF32,
                                            vt::Device{vt::DeviceType::kCPU, 0},
                                            {num_blocks, Hv, Dv, Dk});
      gs.conv_state = vt::Tensor::Contiguous(gdn_conv_buf[g].data(), DType::kF32,
                                             vt::Device{vt::DeviceType::kCPU, 0},
                                             {num_blocks, conv_dim, Kw - 1});
      gdn_state.push_back(gs);
    }
  }
};

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// Full-attn CommonAttentionMetadata for one contiguous single-request PREFILL of
// T tokens whose blocks are `blocks`.
CommonAttentionMetadata PrefillAttnMeta(int64_t T, const std::vector<int32_t>& blocks,
                                        int64_t block_size, int64_t start_slot) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = static_cast<int>(blocks.size());
  m.block_table_tensor = blocks;
  for (int64_t t = 0; t < T; ++t) {
    const int64_t blk = blocks[static_cast<size_t>(t / block_size)];
    m.slot_mapping.push_back(blk * block_size + (start_slot + t) % block_size);
  }
  m.causal = true;
  return m;
}

// GDN metadata for a single fresh PREFILL request at state block `sidx`.
GDNAttentionMetadata PrefillGdnMeta(int64_t T, int32_t sidx) {
  GDNAttentionMetadata g;
  g.num_prefills = 1;
  g.num_prefill_tokens = static_cast<int>(T);
  g.num_decodes = 0;
  g.num_decode_tokens = 0;
  g.num_actual_tokens = static_cast<int>(T);
  g.has_initial_state = std::vector<uint8_t>{0};
  g.non_spec_state_indices_tensor = std::vector<int32_t>{sidx};
  g.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.prefill_state_indices = std::vector<int32_t>{sidx};
  g.prefill_has_initial_state = std::vector<uint8_t>{0};
  return g;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::abs(static_cast<double>(a[i]) - b[i]));
  return m;
}

}  // namespace

TEST_CASE("qwen35 paged: full-prefill batch-of-1 equals dense forward") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t T = 6, vocab = c.vocab_size;
  std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};
  std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5};

  const std::vector<float> dense =
      Qwen3_5Model::ForwardDense(ids, pos, w, c, q);

  CachePool pool(c, /*num_blocks=*/8, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillAttnMeta(T, {0, 1}, 8, 0);
  const GDNAttentionMetadata gm = PrefillGdnMeta(T, 0);
  const std::vector<float> paged = Qwen3_5Model::Forward(
      ids, pos, am, gm, pool.attn_kv, pool.gdn_state, w, c, q);

  REQUIRE(paged.size() == static_cast<size_t>(T * vocab));
  const double d = MaxAbsDiff(paged, dense, paged.size());
  MESSAGE("paged==dense full-prefill max|diff| = " << d);
  CHECK(d < 1e-2);
}

TEST_CASE("qwen35 paged: decode via KV cache equals dense over full sequence") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t T = 5, vocab = c.vocab_size;
  std::vector<int32_t> ids = {7, 1, 22, 4, 15};
  std::vector<int32_t> pos = {0, 1, 2, 3, 4};
  const int32_t next = 8;

  // Paged: prefill then a single-token decode over the SAME persistent caches.
  CachePool pool(c, 8, 8);
  {
    const CommonAttentionMetadata am = PrefillAttnMeta(T, {0, 1}, 8, 0);
    const GDNAttentionMetadata gm = PrefillGdnMeta(T, 0);
    (void)Qwen3_5Model::Forward(ids, pos, am, gm, pool.attn_kv, pool.gdn_state,
                                w, c, q);
  }
  // Decode step: query_len 1 at absolute position T, seq_len T+1.
  CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = 1;
  am.query_start_loc = {0, 1};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {static_cast<int32_t>(T + 1)};
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = 1;
  am.max_seq_len = static_cast<int>(T + 1);
  am.block_table_num_cols = 2;
  am.block_table_tensor = {0, 1};
  am.slot_mapping = {static_cast<int64_t>(T)};  // next slot in block 0
  am.causal = true;
  GDNAttentionMetadata gm;
  gm.num_prefills = 0;
  gm.num_prefill_tokens = 0;
  gm.num_decodes = 1;
  gm.num_decode_tokens = 1;
  gm.num_actual_tokens = 1;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1};

  const std::vector<float> decode_logits = Qwen3_5Model::Forward(
      {next}, {static_cast<int32_t>(T)}, am, gm, pool.attn_kv, pool.gdn_state,
      w, c, q);
  REQUIRE(decode_logits.size() == static_cast<size_t>(vocab));

  // Dense reference over the full T+1 sequence; take the last position.
  std::vector<int32_t> full_ids = ids;
  full_ids.push_back(next);
  std::vector<int32_t> full_pos = pos;
  full_pos.push_back(static_cast<int32_t>(T));
  const std::vector<float> dense = Qwen3_5Model::ForwardDense(full_ids, full_pos, w, c, q);
  std::vector<float> dense_last(dense.begin() + static_cast<int64_t>(T) * vocab, dense.end());

  const double d = MaxAbsDiff(decode_logits, dense_last, decode_logits.size());
  MESSAGE("paged decode-via-cache vs dense max|diff| = " << d);
  CHECK(d < 2e-2);
}

TEST_CASE("qwen35 paged: GDN state zeroing protects a fresh req in a mixed batch") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t Ta = 4, Tb = 3, vocab = c.vocab_size;
  std::vector<int32_t> a_ids = {6, 19, 2, 27};
  std::vector<int32_t> b_ids = {11, 0, 33};

  // Reference: request A run ALONE, fresh.
  std::vector<int32_t> a_pos = {0, 1, 2, 3};
  const std::vector<float> a_alone = Qwen3_5Model::ForwardDense(a_ids, a_pos, w, c, q);

  // Mixed batch: A (state block 0) then B (state block 1), both fresh prefills.
  // Pre-seed A's mamba blocks (index 0) with GARBAGE in every GDN layer — the
  // zeroing must scrub it so A's output is uncorrupted.
  CachePool pool(c, 8, 8);
  for (auto& ssm : pool.gdn_ssm_buf)
    for (auto& x : ssm) x = 999.0f;  // garbage everywhere (block 0 is A's)
  for (auto& cv : pool.gdn_conv_buf)
    for (auto& x : cv) x = 777.0f;
  pool.Rebind();

  std::vector<int32_t> ids;
  ids.insert(ids.end(), a_ids.begin(), a_ids.end());
  ids.insert(ids.end(), b_ids.begin(), b_ids.end());
  std::vector<int32_t> pos = {0, 1, 2, 3, 0, 1, 2};

  CommonAttentionMetadata am;
  am.num_reqs = 2;
  am.num_actual_tokens = static_cast<int>(Ta + Tb);
  am.query_start_loc = {0, static_cast<int32_t>(Ta), static_cast<int32_t>(Ta + Tb)};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {static_cast<int32_t>(Ta), static_cast<int32_t>(Tb)};
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = static_cast<int>(Ta);
  am.max_seq_len = static_cast<int>(Ta);
  am.block_table_num_cols = 2;
  am.block_table_tensor = {0, 1, 2, 3};  // A -> blocks 0,1 ; B -> blocks 2,3
  for (int64_t t = 0; t < Ta; ++t) am.slot_mapping.push_back(t);            // block 0
  for (int64_t t = 0; t < Tb; ++t) am.slot_mapping.push_back(2 * 8 + t);    // block 2
  am.causal = true;

  GDNAttentionMetadata gm;
  gm.num_prefills = 2;
  gm.num_prefill_tokens = static_cast<int>(Ta + Tb);
  gm.num_decodes = 0;
  gm.num_decode_tokens = 0;
  gm.num_actual_tokens = static_cast<int>(Ta + Tb);
  gm.has_initial_state = std::vector<uint8_t>{0, 0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0, 1};
  gm.non_spec_query_start_loc =
      std::vector<int32_t>{0, static_cast<int32_t>(Ta), static_cast<int32_t>(Ta + Tb)};
  gm.prefill_query_start_loc = gm.non_spec_query_start_loc;
  gm.prefill_state_indices = std::vector<int32_t>{0, 1};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0, 0};

  const std::vector<float> batch = Qwen3_5Model::Forward(
      ids, pos, am, gm, pool.attn_kv, pool.gdn_state, w, c, q);
  REQUIRE(batch.size() == static_cast<size_t>((Ta + Tb) * vocab));

  // A occupies the leading Ta token rows; must equal the standalone-fresh A.
  std::vector<float> a_from_batch(batch.begin(),
                                  batch.begin() + static_cast<int64_t>(Ta) * vocab);
  const double d = MaxAbsDiff(a_from_batch, a_alone, a_from_batch.size());
  MESSAGE("mixed-batch fresh-A vs standalone-A max|diff| = " << d
          << " (garbage-seeded mamba block, zeroing must scrub)");
  CHECK(d < 1e-2);
}
