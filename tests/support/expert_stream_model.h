// The synthetic Qwen3.5-MoE model that `test_expert_stream_wiring` and
// `test_expert_stream_steps` both drive (ENG-EXPERT-STREAM, issue #912, repairs
// #1091).
//
// WHY A HEADER AND NOT A DUPLICATE. `VT_MOE_EXPERT_STREAM` is read ONCE into a
// function-local static, and the store behind it is a process-lifetime
// singleton, so each question about the lane needs its own PROCESS and therefore
// its own test binary. Those two ask different questions of the SAME model — a
// four-layer hybrid MoE whose routed experts are uniform Q8_0 keep-quant STACKED
// towers, which is the shape `KqExpertSlice` slices — and a copy per binary
// would let the copies drift apart from the shape the seam serves.
//
// `test_expert_stream_mixed_slot` is deliberately NOT a client: its whole
// subject is a tower set whose gate/up and down slices differ in size (Q4_0
// against Q8_0), and it must not set `VT_MOE_EXPERT_STREAM_SLOT_BYTES`, because
// that override is what would hide the defect it exists for. Its model is a
// different model, not a copy of this one.
//
// Each client keeps its own environment setup, since which knobs a binary sets
// is part of what it is asking.
#ifndef VLLM_TESTS_SUPPORT_EXPERT_STREAM_MODEL_H_
#define VLLM_TESTS_SUPPORT_EXPERT_STREAM_MODEL_H_

#include <doctest/doctest.h>

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

namespace expert_stream_test {

using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::PagedKvCache;
using vllm::Qwen3_5MoeWeights;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;

inline float RandV(uint64_t s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  s ^= s >> 33;
  return (static_cast<float>((s >> 40) & 0xFFFF) / 32768.0f) - 1.0f;
}

inline vllm::OwnedTensor MakeOwned(DType dt, const std::vector<int64_t>& shape, uint64_t seed) {
  vllm::OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    std::vector<uint8_t> b(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(b.data());
    for (int64_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
    t.bytes = vllm::OwnedBytes(std::move(b));
  } else {
    std::vector<uint8_t> b(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(b.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
    t.bytes = vllm::OwnedBytes(std::move(b));
  }
  return t;
}

// A keep-quant STACKED expert tower: [rows, cols] Q8_0, `nk = true`, exactly the
// shape the GGUF keep-quant loader produces and the only shape KqExpertSlice
// slices.
//
// The blocks are BUILT, not filled with noise. A Q8_0 block is an fp16 scale
// followed by 32 int8 weights, and random bytes put random bit patterns in the
// scale — including the fp16 encodings of inf and NaN, which propagate straight
// through the GEMM and make every later comparison vacuous. The values are
// arbitrary but well-formed, which is all this test needs: every arm decodes the
// SAME bytes, so equality between arms is a real comparison.
inline vllm::OwnedTensor MakeKqTower(int64_t rows, int64_t cols, uint64_t seed) {
  vllm::OwnedTensor t;
  t.dtype = DType::kQ8_0;
  t.nk = true;
  t.rank = 2;
  t.shape[0] = rows;
  t.shape[1] = cols;
  const size_t row_bytes = vt::RowSizeBytes(DType::kQ8_0, cols);
  const int64_t blocks_per_row = cols / 32;
  REQUIRE(cols % 32 == 0);          // Q8_0 is a 32-element block quant
  REQUIRE(row_bytes == static_cast<size_t>(blocks_per_row) * 34);
  std::vector<uint8_t> b(static_cast<size_t>(rows) * row_bytes);
  size_t o = 0;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint16_t d = vt::F32ToF16(0.004f + 0.001f * RandV(seed + static_cast<uint64_t>(r * 131 + blk)));
      std::memcpy(b.data() + o, &d, 2);
      o += 2;
      for (int j = 0; j < 32; ++j) {
        const int8_t q = static_cast<int8_t>(
            static_cast<int>(100.0f * RandV(seed + static_cast<uint64_t>((r * 131 + blk) * 32 + j))));
        std::memcpy(b.data() + o, &q, 1);
        o += 1;
      }
    }
  }
  t.bytes = vllm::OwnedBytes(std::move(b));
  return t;
}

inline HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = 40;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 32;
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

inline vllm::MoeBlockWeights MakeKqMoe(const HfConfig& c, uint64_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeOwned(DType::kBF16, {H, E}, s + 1);
  m.shared_gate = MakeOwned(DType::kBF16, {H, 1}, s + 2);
  // The routed experts are STACKED keep-quant towers, and the per-expert vectors
  // stay empty — the A3 layout the streaming seam is defined against.
  m.expert_gate_kq = MakeKqTower(E * I, H, s + 100);
  m.expert_up_kq = MakeKqTower(E * I, H, s + 200);
  m.expert_down_kq = MakeKqTower(E * H, I, s + 300);
  m.shared_gate_proj = MakeOwned(DType::kBF16, {H, Is}, s + 3);
  m.shared_up_proj = MakeOwned(DType::kBF16, {H, Is}, s + 4);
  m.shared_down_proj = MakeOwned(DType::kBF16, {Is, H}, s + 5);
  return m;
}

inline Qwen3_5MoeWeights MakeWeights(const HfConfig& c, uint64_t base_seed = 0) {
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
    const uint64_t s = base_seed + 1000 + static_cast<uint64_t>(l) * 5000;
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
    lw.moe = MakeKqMoe(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

struct CachePool {
  const HfConfig& c;
  int64_t num_blocks;
  int64_t block_size;
  std::vector<std::vector<float>> full_attn_buf;
  std::vector<std::vector<float>> gdn_ssm_buf;
  std::vector<std::vector<float>> gdn_conv_buf;
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

inline vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

inline CommonAttentionMetadata PrefillAttnMeta(int64_t T, const std::vector<int32_t>& blocks,
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

inline GDNAttentionMetadata PrefillGdnMeta(int64_t T, int32_t sidx) {
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
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
  g.batch_ptr = conv.batch_ptr;
  g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  return g;
}
}  // namespace expert_stream_test

#endif  // VLLM_TESTS_SUPPORT_EXPERT_STREAM_MODEL_H_
