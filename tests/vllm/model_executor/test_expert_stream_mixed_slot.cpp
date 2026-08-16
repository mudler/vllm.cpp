// ENG-EXPERT-STREAM (#912) F7, the size half: the slot store must be sized from
// the LARGEST expert slice, not from whichever one arrived first.
//
// A dynamic (UD) quant keeps `down_proj` at a higher precision than the gate/up
// pair, so within one layer the three slices are NOT the same size. The store is
// built lazily on the first slice taken, and it used to take that slice's size
// as the slot size. Streaming a UD checkpoint therefore sized every slot from a
// gate slice and then refused the very first DOWN slice, by name, in the middle
// of decode -- on exactly the checkpoints this row exists to serve.
//
// `Qwen35ExpertStream::Reserve` is the repair: `ExpertMlpKq` declares
// max(gate, up, down) before it takes any of the three, so the store is sized
// once, correctly, before anything is stored.
//
// THIS NEEDS ITS OWN BINARY, and specifically one that does NOT set
// `VT_MOE_EXPERT_STREAM_SLOT_BYTES`. That variable overrides the computed size,
// so a binary that sets it can never observe this defect; and the store is a
// process-lifetime singleton built on the first slice, so the FIRST model this
// process runs has to be the mixed-precision one.
#include <stdlib.h>

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::PagedKvCache;
using vllm::Qwen3_5MoeWeights;
using vllm::Qwen3_5Model;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;

namespace {

struct EnableExpertStreaming {
  EnableExpertStreaming() {
    ::setenv("VT_MOE_EXPERT_STREAM", "1", 1);
    ::setenv("VT_MOE_EXPERT_STREAM_SLOTS", "64", 1);
    // DELIBERATELY NOT SET: VT_MOE_EXPERT_STREAM_SLOT_BYTES. The whole point is
    // to let the store size itself, which is where the defect lives.
    ::unsetenv("VT_MOE_EXPERT_STREAM_SLOT_BYTES");
    // Quiet under ctest, but overwrite=0 so an operator who sets this var
    // still gets the line -- which is the only way to SEE the statistics
    // this row added, and a gate that suppresses its own evidence is a
    // smaller version of the defect it was written for.
    ::setenv("VT_MOE_EXPERT_STREAM_STATS_EVERY", "0", 0);
    ::setenv("VT_QWEN35_GROUPED_MOE", "0", 1);
  }
};
const EnableExpertStreaming kEnableExpertStreaming;

float RandV(uint64_t s) {
  s = s * 6364136223846793005ULL + 1442695040888963407ULL;
  s ^= s >> 33;
  return (static_cast<float>((s >> 40) & 0xFFFF) / 32768.0f) - 1.0f;
}

vllm::OwnedTensor MakeOwned(DType dt, const std::vector<int64_t>& shape, uint64_t seed) {
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

// A stacked keep-quant tower in either Q4_0 (18 bytes per 32-element block) or
// Q8_0 (34 bytes). Both are built block by block with a well-formed fp16 scale;
// random bytes would put inf/NaN bit patterns in the scale.
vllm::OwnedTensor MakeKqTower(int64_t rows, int64_t cols, uint64_t seed, DType dt) {
  vllm::OwnedTensor t;
  t.dtype = dt;
  t.nk = true;
  t.rank = 2;
  t.shape[0] = rows;
  t.shape[1] = cols;
  const size_t row_bytes = vt::RowSizeBytes(dt, cols);
  const int64_t blocks_per_row = cols / 32;
  REQUIRE(cols % 32 == 0);
  const size_t blk_bytes = (dt == DType::kQ4_0) ? 18u : 34u;
  REQUIRE(row_bytes == static_cast<size_t>(blocks_per_row) * blk_bytes);
  std::vector<uint8_t> b(static_cast<size_t>(rows) * row_bytes);
  size_t o = 0;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
      const uint16_t d = vt::F32ToF16(
          0.004f + 0.001f * RandV(seed + static_cast<uint64_t>(r * 131 + blk)));
      std::memcpy(b.data() + o, &d, 2);
      o += 2;
      if (dt == DType::kQ4_0) {
        for (int j = 0; j < 16; ++j) {  // two 4-bit weights per byte
          const uint8_t lo = static_cast<uint8_t>(
              (static_cast<int>(7.0f + 7.0f * RandV(seed + static_cast<uint64_t>(r * 977 + blk * 31 + j))) & 0x0F));
          const uint8_t hi = static_cast<uint8_t>(
              (static_cast<int>(7.0f + 7.0f * RandV(seed + static_cast<uint64_t>(r * 977 + blk * 31 + j + 512))) & 0x0F));
          b[o++] = static_cast<uint8_t>(lo | (hi << 4));
        }
      } else {
        for (int j = 0; j < 32; ++j) {
          const int8_t q = static_cast<int8_t>(static_cast<int>(
              100.0f * RandV(seed + static_cast<uint64_t>((r * 131 + blk) * 32 + j))));
          std::memcpy(b.data() + o, &q, 1);
          o += 1;
        }
      }
    }
  }
  t.bytes = vllm::OwnedBytes(std::move(b));
  return t;
}

HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 2;
  c.vocab_size = 40;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "full_attention"};
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

// THE UD SHAPE: gate/up at Q4_0, down kept at Q8_0. Per expert that is a 576-byte
// gate slice and a 1088-byte down slice, so the down slice is nearly twice the
// size of the one the store would otherwise have been built from.
vllm::MoeBlockWeights MakeMixedMoe(const HfConfig& c, uint64_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeOwned(DType::kBF16, {H, E}, s + 1);
  m.shared_gate = MakeOwned(DType::kBF16, {H, 1}, s + 2);
  m.expert_gate_kq = MakeKqTower(E * I, H, s + 100, DType::kQ4_0);
  m.expert_up_kq = MakeKqTower(E * I, H, s + 200, DType::kQ4_0);
  m.expert_down_kq = MakeKqTower(E * H, I, s + 300, DType::kQ8_0);
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
    lw.moe = MakeMixedMoe(c, s + 500);
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
    const int64_t Hkv2 = Hkv, Dh2 = Dh;
    for (auto& b : full_attn_buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = nb;
      kv.block_size = bs;
      kv.num_kv_heads = Hkv2;
      kv.head_size = Dh2;
      attn_kv.push_back(kv);
    }
    for (size_t g = 0; g < gdn_ssm_buf.size(); ++g) {
      GdnStateCache gs;
      gs.ssm_state = vt::Tensor::Contiguous(gdn_ssm_buf[g].data(), DType::kF32,
                                            vt::Device{vt::DeviceType::kCPU, 0},
                                            {nb, Hv, Dv, Dk});
      gs.conv_state = vt::Tensor::Contiguous(gdn_conv_buf[g].data(), DType::kF32,
                                             vt::Device{vt::DeviceType::kCPU, 0},
                                             {nb, conv_dim, Kw - 1});
      gdn_state.push_back(gs);
    }
  }
};

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

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
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
  g.batch_ptr = conv.batch_ptr;
  g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  return g;
}

}  // namespace

TEST_CASE("a UD-shaped model streams: the slot fits the LARGEST slice, not the first") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);

  // The sizes this case turns on, asserted rather than assumed: the down slice
  // really is bigger, so sizing from a gate slice really would refuse it.
  const size_t gate_slice =
      static_cast<size_t>(c.moe_intermediate_size) *
      vt::RowSizeBytes(DType::kQ4_0, c.hidden_size);
  const size_t down_slice = static_cast<size_t>(c.hidden_size) *
                            vt::RowSizeBytes(DType::kQ8_0, c.moe_intermediate_size);
  REQUIRE(down_slice > gate_slice);

  const std::vector<int32_t> ids = {5, 9, 2, 31};
  std::vector<int32_t> pos = {0, 1, 2, 3};
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/8);
  const std::vector<int32_t> blocks = {0};

  // The forward must COMPLETE. Without the reservation the store is built from
  // the first gate slice and the first down slice throws a named refusal
  // ("exceeds the slot budget of ...") partway through the first layer.
  vt::Queue q = Q();
  std::vector<float> logits;
  REQUIRE_NOTHROW(logits = Qwen3_5Model::Forward(
                      ids, pos, PrefillAttnMeta(4, blocks, 8, 0),
                      PrefillGdnMeta(4, 0), pool.attn_kv, pool.gdn_state, w, c,
                      q, {}));
  REQUIRE(logits.size() == ids.size() * static_cast<size_t>(c.vocab_size));
  for (float v : logits) REQUIRE(std::isfinite(v));

  // And it streamed rather than quietly falling back to the tower.
  const vllm::detail::ExpertStreamStats s = vllm::detail::ExpertStreamSnapshot();
  CHECK(s.active);
  CHECK(s.fills > 0);
  CHECK(s.steps == 1);
  CHECK(s.exhausted == 0);
}
