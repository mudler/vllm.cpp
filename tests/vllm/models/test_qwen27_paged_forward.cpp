// Batched PAGED dense forward parity for the 27B (notes §5 "Paged dense path +
// runner wiring"). Validates Qwen3_5DenseModel::Forward (paged) against the
// retained single-sequence reference Qwen3_5DenseModel::ForwardDense on a small
// SYNTHETIC hybrid DENSE model (CPU; the real 27B greedy gate is GPU-gated). This
// is the 27B analogue of test_qwen35_paged_forward.cpp — same three gates, dense
// SwiGLU MLP instead of the MoE block, GQA ratio 3 (Hv/Hk = 6/2):
//   1. PAGED == DENSE anchor: a batch-of-1 full prefill of a short sequence
//      through the paged path (vt::ReshapeAndCache + vt::PagedAttention + GDN
//      over persistent-but-fresh mamba state) equals the dense forward.
//   2. Multi-block prefill (block_size < T, non-contiguous blocks) — exercises
//      the block-table indirection + block stride at the model level.
//   3. DECODE via KV cache: prefill then a single-token decode reading the
//      prefilled K/V + persisted GDN state equals the dense forward at the last
//      position — proves the paged READ + KV/mamba-state growth.
//   4. GDN fresh-vs-continuing zeroing: a mixed batch where one request's mamba
//      block is pre-seeded with GARBAGE; its output must match running it alone.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::DenseMlpWeights;
using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::MergeDenseGateUpGlobals;
using vllm::MergeFullAttnQkvGlobals;
using vllm::Nvfp4Weight;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5DenseModel;
using vllm::Qwen3_5DenseWeights;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;

namespace {

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    if (old != nullptr) {
      had_old_ = true;
      old_ = old;
    }
    setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_old_)
      setenv(name_.c_str(), old_.c_str(), 1);
    else
      unsetenv(name_.c_str());
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::string old_;
  bool had_old_ = false;
};

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

// 27B-shaped small dense config: layer_types [LA, LA, LA, FA], no experts,
// GQA ratio 3 (Hv/Hk = 6/2) — matches test_qwen27_dense_forward's MakeConfig.
HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = 40;
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.intermediate_size = 16;
  c.num_experts = 0;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 6;  // GQA ratio 3
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

// KV / mamba-state cache pool for one step through the paged Forward. Owns the
// host buffers and hands out PagedKvCache / GdnStateCache views (mirrors the 35B
// test's CachePool).
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
  return g;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::abs(static_cast<double>(a[i]) - b[i]));
  return m;
}

// Full-attn metadata for ONE chunked-prefill step of a single request: a chunk
// of `qlen` new tokens whose first token sits at absolute position `context`
// (context == num_computed_tokens). seq_lens = context + qlen so the paged
// attention attends causally over the already-cached context + this chunk
// (cpu_paged_attn.cpp: context = seq_lens - query_len). Positions/slots are
// laid out contiguously from `context` in a single block table `blocks`.
CommonAttentionMetadata ChunkAttnMeta(int64_t context, int64_t qlen,
                                      const std::vector<int32_t>& blocks,
                                      int64_t block_size) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(qlen);
  m.query_start_loc = {0, static_cast<int32_t>(qlen)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(context + qlen)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(qlen);
  m.max_seq_len = static_cast<int>(context + qlen);
  m.block_table_num_cols = static_cast<int>(blocks.size());
  m.block_table_tensor = blocks;
  for (int64_t t = 0; t < qlen; ++t) {
    const int64_t abs_pos = context + t;
    const int64_t blk = blocks[static_cast<size_t>(abs_pos / block_size)];
    m.slot_mapping.push_back(blk * block_size + abs_pos % block_size);
  }
  m.causal = true;
  return m;
}

// GDN metadata for ONE chunked-prefill step of a single request. When
// `has_initial` is true (context > 0, a resumed chunk) the GDN layer must
// CONTINUE from the saved recurrent + conv state for state index `sidx`
// (prefill_has_initial_state=1 => the gather is NOT zeroed), mirroring the
// builder's has_initial_state = context_lens > 0 (gdn_attn.cpp build()).
GDNAttentionMetadata ChunkGdnMeta(int64_t qlen, int32_t sidx, bool has_initial) {
  GDNAttentionMetadata g;
  g.num_prefills = 1;
  g.num_prefill_tokens = static_cast<int>(qlen);
  g.num_decodes = 0;
  g.num_decode_tokens = 0;
  g.num_actual_tokens = static_cast<int>(qlen);
  const uint8_t hi = has_initial ? 1 : 0;
  g.has_initial_state = std::vector<uint8_t>{hi};
  g.non_spec_state_indices_tensor = std::vector<int32_t>{sidx};
  g.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(qlen)};
  g.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(qlen)};
  g.prefill_state_indices = std::vector<int32_t>{sidx};
  g.prefill_has_initial_state = std::vector<uint8_t>{hi};
  return g;
}

}  // namespace

// Mirrors vLLM v0.25.0
// qwen_gdn_linear_attn.py::_forward_core:1286-1298. The packed branch is
// selected only for enabled, pure non-spec decode; every prefill/mixed/spec
// shape remains on the standard recurrence. Local W1D2 additionally scopes the
// first model consumer to the real dense packed-BA path and requires compatible
// BF16 storage plus persistent device indices.
TEST_CASE("qwen27 packed GDN selection is pure non-spec decode only") {
  vllm::detail::GdnPackedDecodeEligibility e;
  e.runtime_enabled = true;
  e.cuda = true;
  e.dense_model = true;
  e.has_packed_ba = true;
  e.merged_ba_enabled = true;
  e.dtype_compatible = true;
  e.has_state_indices = true;
  e.num_actual_tokens = 4;
  e.num_decodes = 4;
  e.num_decode_tokens = 4;

  CHECK(vllm::detail::ShouldUsePackedGdnDecode(e));

  auto rejects = [&](const vllm::detail::GdnPackedDecodeEligibility& candidate) {
    CHECK_FALSE(vllm::detail::ShouldUsePackedGdnDecode(candidate));
  };
  {
    auto x = e;
    x.runtime_enabled = false;
    rejects(x);  // VT_GDN_PACKED_DECODE=0 rollback.
  }
  {
    auto x = e;
    x.cuda = false;
    rejects(x);
  }
  {
    auto x = e;
    x.dense_model = false;
    rejects(x);  // 35B/MoE is deliberately inert in W1D2.
  }
  {
    auto x = e;
    x.has_packed_ba = false;
    rejects(x);
  }
  {
    auto x = e;
    x.merged_ba_enabled = false;
    rejects(x);
  }
  {
    auto x = e;
    x.dtype_compatible = false;
    rejects(x);
  }
  {
    auto x = e;
    x.has_state_indices = false;
    rejects(x);
  }
  {
    auto x = e;
    x.num_prefills = 4;
    x.num_prefill_tokens = 4;
    x.num_decodes = 0;
    x.num_decode_tokens = 0;
    rejects(x);  // prefill only.
  }
  {
    auto x = e;
    x.num_prefills = 1;
    x.num_prefill_tokens = 3;
    x.num_decodes = 1;
    x.num_decode_tokens = 1;
    rejects(x);  // mixed decode + prefill.
  }
  {
    auto x = e;
    x.num_spec_decodes = 1;
    x.num_spec_decode_tokens = 2;
    rejects(x);
  }
  {
    auto x = e;
    x.num_decode_tokens = 3;
    rejects(x);  // not one token per decode request.
  }
  {
    auto x = e;
    x.num_actual_tokens = 5;
    rejects(x);  // padded/non-actual rows are never consumed as live tokens.
  }
}

// W2 merged-qkvz dispatch: ONE in_proj_qkvz GEMM is selected only on CUDA, with
// the packed owner resident, the runtime toggles on (VT_GDN_MERGED_PROJ master,
// VT_GDN_MERGED_QKVZ leaf) and one uniform output dtype (mixed_qkv and z come
// out of one GEMM, so GdnInDType must equal GdnOutDType — the 27B default BF16/
// BF16). Every other combination stays on the exact two split GEMMs sliced from
// the same owner. 35B (fp8 qkv/z, no packed owner) and GGUF/synthetic (split
// bf16) are inert by has_packed_qkvz.
TEST_CASE("qwen27 merged qkvz selection requires CUDA, owner, toggle, one dtype") {
  vllm::detail::GdnMergedQkvzEligibility e;
  e.runtime_enabled = true;
  e.cuda = true;
  e.has_packed_qkvz = true;
  e.uniform_dtype = true;

  CHECK(vllm::detail::ShouldUseMergedGdnQkvz(e));

  {
    auto x = e;
    x.runtime_enabled = false;  // VT_GDN_MERGED_QKVZ=0 / VT_GDN_MERGED_PROJ=0.
    CHECK_FALSE(vllm::detail::ShouldUseMergedGdnQkvz(x));
  }
  {
    auto x = e;
    x.cuda = false;  // CPU keeps the reference split arithmetic.
    CHECK_FALSE(vllm::detail::ShouldUseMergedGdnQkvz(x));
  }
  {
    auto x = e;
    x.has_packed_qkvz = false;  // 35B fp8 / GGUF / synthetic split owners.
    CHECK_FALSE(vllm::detail::ShouldUseMergedGdnQkvz(x));
  }
  {
    auto x = e;
    x.uniform_dtype = false;  // diagnostic VT_GDN_IN_BF16 != VT_GDN_OUT_BF16.
    CHECK_FALSE(vllm::detail::ShouldUseMergedGdnQkvz(x));
  }
}

TEST_CASE("qwen27 packed GDN validates engine state slots before upload") {
  CHECK_NOTHROW(vllm::detail::ValidateGdnStateIndices(
      std::vector<int32_t>{0, 1, -1}, /*required=*/3, /*slots=*/2));
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnStateIndices(
          std::vector<int32_t>{0, 0}, /*required=*/2, /*slots=*/2),
      doctest::Contains("duplicate live GDN state index"), std::runtime_error);
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnStateIndices(
          std::vector<int32_t>{0, 2}, /*required=*/2, /*slots=*/2),
      doctest::Contains("GDN state index out of range"), std::runtime_error);
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnStateIndices(
          std::vector<int32_t>{0}, /*required=*/2, /*slots=*/2),
      doctest::Contains("GDN state index metadata is too short"),
      std::runtime_error);
}

TEST_CASE("qwen27 GDN metadata validates complete prefill suffixes before I/O") {
  GDNAttentionMetadata gm;
  gm.num_decodes = 1;
  gm.num_decode_tokens = 1;
  gm.num_prefills = 2;
  gm.num_prefill_tokens = 5;
  gm.num_actual_tokens = 6;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0, 1, 2};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1, 3, 6};
  gm.has_initial_state = std::vector<uint8_t>{1, 0, 1};
  gm.prefill_state_indices = std::vector<int32_t>{1, 2};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, 2, 5};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0, 1};

  CHECK_NOTHROW(vllm::detail::ValidateGdnAttentionMetadata(
      gm, /*state_slots=*/3, /*allow_inert_padding=*/false));

  auto bad = gm;
  (*bad.non_spec_state_indices_tensor)[2] = 3;
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnAttentionMetadata(
          bad, /*state_slots=*/3, /*allow_inert_padding=*/false),
      doctest::Contains("GDN state index out of range"), std::runtime_error);

  bad = gm;
  (*bad.prefill_state_indices)[0] = 2;
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnAttentionMetadata(
          bad, /*state_slots=*/3, /*allow_inert_padding=*/false),
      doctest::Contains("prefill state indices must match"),
      std::runtime_error);

  bad = gm;
  (*bad.prefill_query_start_loc)[1] = 3;
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnAttentionMetadata(
          bad, /*state_slots=*/3, /*allow_inert_padding=*/false),
      doctest::Contains("prefill query offsets must match"),
      std::runtime_error);

  bad = gm;
  (*bad.prefill_has_initial_state)[0] = 1;
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnAttentionMetadata(
          bad, /*state_slots=*/3, /*allow_inert_padding=*/false),
      doctest::Contains("prefill initial-state mask must match"),
      std::runtime_error);
}

TEST_CASE("qwen27 GDN graph padding requires indexed state I/O") {
  CHECK(vllm::detail::CanUseGdnDecodeGraphSize(
      /*real_batch=*/3, /*capture_batch=*/3, /*indexed_state_io=*/false));
  CHECK_FALSE(vllm::detail::CanUseGdnDecodeGraphSize(
      /*real_batch=*/3, /*capture_batch=*/4, /*indexed_state_io=*/false));
  CHECK(vllm::detail::CanUseGdnDecodeGraphSize(
      /*real_batch=*/3, /*capture_batch=*/4, /*indexed_state_io=*/true));

  GDNAttentionMetadata padded;
  padded.num_decodes = 4;
  padded.num_decode_tokens = 4;
  padded.num_actual_tokens = 4;
  padded.non_spec_state_indices_tensor = std::vector<int32_t>{0, 1, 2, -1};
  CHECK_NOTHROW(vllm::detail::ValidateGdnAttentionMetadata(
      padded, /*state_slots=*/3, /*allow_inert_padding=*/true));
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnAttentionMetadata(
          padded, /*state_slots=*/3, /*allow_inert_padding=*/false),
      doctest::Contains("live GDN state index must be non-negative"),
      std::runtime_error);
}

TEST_CASE("qwen27 decode graph preflights every GDN cache before replay") {
  const HfConfig c = MakeConfig();
  CachePool pool(c, /*num_blocks=*/3, /*block_size=*/8);
  GDNAttentionMetadata gm;
  gm.num_prefills = 0;
  gm.num_prefill_tokens = 0;
  gm.num_decodes = 2;
  gm.num_decode_tokens = 2;
  gm.num_actual_tokens = 2;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0, 2};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1, 2};

  CHECK_NOTHROW(vllm::detail::ValidateGdnDecodeGraphState(
      gm, pool.gdn_state, /*real_batch=*/2));

  gm.non_spec_state_indices_tensor = std::vector<int32_t>{1, 1};
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnDecodeGraphState(
          gm, pool.gdn_state, /*real_batch=*/2),
      doctest::Contains("duplicate live GDN state index"), std::runtime_error);

  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0, 3};
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnDecodeGraphState(
          gm, pool.gdn_state, /*real_batch=*/2),
      doctest::Contains("GDN state index out of range"), std::runtime_error);

  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0, 2};
  std::vector<GdnStateCache> inconsistent = pool.gdn_state;
  inconsistent.back().conv_state.shape[0] = 2;
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnDecodeGraphState(
          gm, inconsistent, /*real_batch=*/2),
      doctest::Contains("conv/SSM state slot counts must match"),
      std::runtime_error);

  inconsistent = pool.gdn_state;
  inconsistent.back().ssm_state.shape[0] = 2;
  inconsistent.back().conv_state.shape[0] = 2;
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnDecodeGraphState(
          gm, inconsistent, /*real_batch=*/2),
      doctest::Contains("all GDN layers must use the same state slot count"),
      std::runtime_error);

  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0, 1, 2};
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnDecodeGraphState(
          gm, pool.gdn_state, /*real_batch=*/2),
      doctest::Contains("state index count must equal the real decode batch"),
      std::runtime_error);
}

TEST_CASE("qwen27 eager forward preflights every GDN cache layer") {
  const HfConfig c = MakeConfig();
  const Qwen3_5DenseWeights w = MakeWeights(c);
  CachePool pool(c, /*num_blocks=*/3, /*block_size=*/8);
  std::vector<GdnStateCache> inconsistent = pool.gdn_state;
  REQUIRE(inconsistent.size() >= 2);
  inconsistent.back().ssm_state.shape[0] = 2;
  inconsistent.back().conv_state.shape[0] = 2;

  const CommonAttentionMetadata am = PrefillAttnMeta(1, {0}, 8, 0);
  const GDNAttentionMetadata gm = PrefillGdnMeta(1, 0);
  vt::Queue q = Q();
  CHECK_THROWS_WITH_AS(
      Qwen3_5DenseModel::Forward({5}, {0}, am, gm, pool.attn_kv,
                                 inconsistent, w, c, q),
      doctest::Contains("all GDN layers must use the same state slot count"),
      std::runtime_error);
}

// Port of the fused logical-scale processing contract in pinned vLLM
// compressed_tensors_w4a4_nvfp4.py:95-138. Gate/up checkpoint scalars are
// loaded into arrays by MergedColumnParallelLinear; each maximum is selected
// before its single reciprocal and the shared alpha multiplication.
TEST_CASE("qwen27 dense merged gate-up uses maximum CT logical-shard divisors") {
  Nvfp4Weight gate;
  gate.weight_global_scale_inv = 224.0F;
  gate.input_global_scale_inv = 80.0F;
  gate.scale2 = 1.0F / gate.weight_global_scale_inv;
  gate.alpha = gate.scale2 * (1.0F / gate.input_global_scale_inv);

  Nvfp4Weight up;
  up.weight_global_scale_inv = 448.0F;
  up.input_global_scale_inv = 96.0F;
  up.scale2 = 1.0F / up.weight_global_scale_inv;
  up.alpha = up.scale2 * (1.0F / up.input_global_scale_inv);

  const vllm::DenseGateUpGlobals globals =
      MergeDenseGateUpGlobals(gate, up);
  CHECK(globals.input_global_scale_inv == 96.0F);
  CHECK(globals.weight_global_scale == 1.0F / 448.0F);
  CHECK(globals.alpha == (1.0F / 96.0F) * (1.0F / 448.0F));

  up.weight_global_scale_inv = 0.0F;
  CHECK_THROWS_WITH_AS(MergeDenseGateUpGlobals(gate, up),
                       doctest::Contains("missing CT weight divisor"),
                       std::runtime_error);
}

// Ported from tests/model_executor/model_loader/test_reload.py:150's
// QKVParallelLinear construction, tests/models/test_adapters.py:44-60's
// physical shard placement, and
// compressed_tensors_w4a4_nvfp4.py:95-138. Q/K/V may carry unequal checkpoint
// scalars; the packed physical linear selects each maximum before reciprocal.
TEST_CASE("qwen27 packed QKV uses maximum CT logical-shard divisors") {
  Nvfp4Weight q;
  q.weight_global_scale_inv = 256.0F;
  q.input_global_scale_inv = 72.0F;
  Nvfp4Weight k;
  k.weight_global_scale_inv = 512.0F;
  k.input_global_scale_inv = 80.0F;
  Nvfp4Weight v;
  v.weight_global_scale_inv = 384.0F;
  v.input_global_scale_inv = 96.0F;

  const vllm::FullAttnQkvGlobals globals =
      MergeFullAttnQkvGlobals(q, k, v);
  CHECK(globals.input_global_scale_inv == 96.0F);
  CHECK(globals.weight_global_scale == 1.0F / 512.0F);
  CHECK(globals.alpha == (1.0F / 96.0F) * (1.0F / 512.0F));

  v.input_global_scale_inv = 0.0F;
  CHECK_THROWS_WITH_AS(MergeFullAttnQkvGlobals(q, k, v),
                       doctest::Contains("missing CT input divisor"),
                       std::runtime_error);
}

TEST_CASE("qwen27 dense paged: full-prefill batch-of-1 equals dense forward") {
  const HfConfig c = MakeConfig();
  const Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t T = 6, vocab = c.vocab_size;
  std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};
  std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5};

  const std::vector<float> dense = Qwen3_5DenseModel::ForwardDense(ids, pos, w, c, q);

  CachePool pool(c, /*num_blocks=*/8, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillAttnMeta(T, {0, 1}, 8, 0);
  const GDNAttentionMetadata gm = PrefillGdnMeta(T, 0);
  const std::vector<float> paged = Qwen3_5DenseModel::Forward(
      ids, pos, am, gm, pool.attn_kv, pool.gdn_state, w, c, q);

  REQUIRE(paged.size() == static_cast<size_t>(T * vocab));
  const double d = MaxAbsDiff(paged, dense, paged.size());
  MESSAGE("dense paged==dense full-prefill max|diff| = " << d);
  CHECK(d < 1e-2);
}

TEST_CASE("qwen27 dense paged: multi-block full-prefill (block_size<T) equals dense") {
  const HfConfig c = MakeConfig();
  const Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t T = 6, vocab = c.vocab_size;
  std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};
  std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5};

  const std::vector<float> dense = Qwen3_5DenseModel::ForwardDense(ids, pos, w, c, q);

  // block_size=4: tokens 0-3 -> block 2, tokens 4-5 -> block 0 (non-contiguous).
  CachePool pool(c, /*num_blocks=*/8, /*block_size=*/4);
  const CommonAttentionMetadata am = PrefillAttnMeta(T, {2, 0}, 4, 0);
  const GDNAttentionMetadata gm = PrefillGdnMeta(T, 0);
  const std::vector<float> paged = Qwen3_5DenseModel::Forward(
      ids, pos, am, gm, pool.attn_kv, pool.gdn_state, w, c, q);

  REQUIRE(paged.size() == static_cast<size_t>(T * vocab));
  const double d = MaxAbsDiff(paged, dense, paged.size());
  MESSAGE("dense paged==dense multi-block max|diff| = " << d);
  CHECK(d < 1e-2);
}

TEST_CASE("qwen27 dense paged: decode via KV cache equals dense over full sequence") {
  const HfConfig c = MakeConfig();
  const Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t T = 5, vocab = c.vocab_size;
  std::vector<int32_t> ids = {7, 1, 22, 4, 15};
  std::vector<int32_t> pos = {0, 1, 2, 3, 4};
  const int32_t next = 8;

  CachePool pool(c, 8, 8);
  {
    const CommonAttentionMetadata am = PrefillAttnMeta(T, {0, 1}, 8, 0);
    const GDNAttentionMetadata gm = PrefillGdnMeta(T, 0);
    (void)Qwen3_5DenseModel::Forward(ids, pos, am, gm, pool.attn_kv, pool.gdn_state,
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
  am.slot_mapping = {static_cast<int64_t>(T)};
  am.causal = true;
  GDNAttentionMetadata gm;
  gm.num_prefills = 0;
  gm.num_prefill_tokens = 0;
  gm.num_decodes = 1;
  gm.num_decode_tokens = 1;
  gm.num_actual_tokens = 1;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1};

  const std::vector<float> decode_logits = Qwen3_5DenseModel::Forward(
      {next}, {static_cast<int32_t>(T)}, am, gm, pool.attn_kv, pool.gdn_state, w, c, q);
  REQUIRE(decode_logits.size() == static_cast<size_t>(vocab));

  std::vector<int32_t> full_ids = ids;
  full_ids.push_back(next);
  std::vector<int32_t> full_pos = pos;
  full_pos.push_back(static_cast<int32_t>(T));
  const std::vector<float> dense = Qwen3_5DenseModel::ForwardDense(full_ids, full_pos, w, c, q);
  std::vector<float> dense_last(dense.begin() + static_cast<int64_t>(T) * vocab, dense.end());

  const double d = MaxAbsDiff(decode_logits, dense_last, decode_logits.size());
  MESSAGE("dense paged decode-via-cache vs dense max|diff| = " << d);
  CHECK(d < 2e-2);
}

// Ports the mixed decode+prefill turnover shape from pinned-vLLM
// tests/v1/worker/test_mamba_utils.py:342-358. W1 uploads the complete
// non-spec state-index vector once, while GdnDecode consumes only its leading
// decode rows. This catches passing the full [decode+prefill] view to the
// decode recurrence when a completed request is replaced by a fresh prefill.
TEST_CASE("qwen27 dense paged: indexed GDN mixed turnover matches row-copy fallback") {
  const HfConfig c = MakeConfig();
  const Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t vocab = c.vocab_size;
  CachePool fallback_pool(c, 8, 8);
  CachePool indexed_pool(c, 8, 8);

  const std::vector<int32_t> seed_ids = {7, 1, 22};
  const std::vector<int32_t> seed_pos = {0, 1, 2};
  auto seed = [&](CachePool& pool, const char* toggle) {
    ScopedEnv env("VT_GDN_INDEXED_STATE_IO", toggle);
    const CommonAttentionMetadata am = PrefillAttnMeta(3, {0, 1}, 8, 0);
    const GDNAttentionMetadata gm = PrefillGdnMeta(3, 0);
    (void)Qwen3_5DenseModel::Forward(seed_ids, seed_pos, am, gm, pool.attn_kv,
                                     pool.gdn_state, w, c, q);
  };
  seed(fallback_pool, "0");
  seed(indexed_pool, "1");

  // Request 0 decodes one token from state slot 0; request 1 starts a two-token
  // prefill in state slot 1. Decodes lead the flattened token stream.
  CommonAttentionMetadata am;
  am.num_reqs = 2;
  am.num_actual_tokens = 3;
  am.query_start_loc = {0, 1, 3};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {4, 2};
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = 2;
  am.max_seq_len = 4;
  am.block_table_num_cols = 2;
  am.block_table_tensor = {0, 1, 2, 3};
  am.slot_mapping = {3, 16, 17};
  am.causal = true;

  GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = 2;
  gm.num_decodes = 1;
  gm.num_decode_tokens = 1;
  gm.num_actual_tokens = 3;
  gm.has_initial_state = std::vector<uint8_t>{1, 0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0, 1};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1, 3};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, 2};
  gm.prefill_state_indices = std::vector<int32_t>{1};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};

  const std::vector<int32_t> ids = {4, 11, 0};
  const std::vector<int32_t> pos = {3, 0, 1};
  std::vector<float> fallback;
  {
    ScopedEnv env("VT_GDN_INDEXED_STATE_IO", "0");
    fallback = Qwen3_5DenseModel::Forward(ids, pos, am, gm, fallback_pool.attn_kv,
                                          fallback_pool.gdn_state, w, c, q);
  }
  std::vector<float> indexed;
  {
    ScopedEnv env("VT_GDN_INDEXED_STATE_IO", "1");
    indexed = Qwen3_5DenseModel::Forward(ids, pos, am, gm, indexed_pool.attn_kv,
                                         indexed_pool.gdn_state, w, c, q);
  }

  REQUIRE(indexed.size() == static_cast<size_t>(3 * vocab));
  REQUIRE(fallback.size() == indexed.size());
  const double logits_diff = MaxAbsDiff(indexed, fallback, indexed.size());
  MESSAGE("indexed mixed-turnover logits vs fallback max|diff| = " << logits_diff);
  CHECK(logits_diff < 1e-4);
  for (size_t layer = 0; layer < indexed_pool.gdn_ssm_buf.size(); ++layer) {
    CHECK(MaxAbsDiff(indexed_pool.gdn_ssm_buf[layer],
                     fallback_pool.gdn_ssm_buf[layer],
                     indexed_pool.gdn_ssm_buf[layer].size()) < 1e-4);
    CHECK(MaxAbsDiff(indexed_pool.gdn_conv_buf[layer],
                     fallback_pool.gdn_conv_buf[layer],
                     indexed_pool.gdn_conv_buf[layer].size()) < 1e-4);
  }
}

TEST_CASE("qwen27 dense paged: GDN state zeroing protects a fresh req in a mixed batch") {
  const HfConfig c = MakeConfig();
  const Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t Ta = 4, Tb = 3, vocab = c.vocab_size;
  std::vector<int32_t> a_ids = {6, 19, 2, 27};
  std::vector<int32_t> b_ids = {11, 0, 33};

  std::vector<int32_t> a_pos = {0, 1, 2, 3};
  const std::vector<float> a_alone = Qwen3_5DenseModel::ForwardDense(a_ids, a_pos, w, c, q);

  CachePool pool(c, 8, 8);
  for (auto& ssm : pool.gdn_ssm_buf)
    for (auto& x : ssm) x = 999.0f;  // garbage (block 0 is A's)
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
  for (int64_t t = 0; t < Ta; ++t) am.slot_mapping.push_back(t);          // block 0
  for (int64_t t = 0; t < Tb; ++t) am.slot_mapping.push_back(2 * 8 + t);  // block 2
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

  const std::vector<float> batch = Qwen3_5DenseModel::Forward(
      ids, pos, am, gm, pool.attn_kv, pool.gdn_state, w, c, q);
  REQUIRE(batch.size() == static_cast<size_t>((Ta + Tb) * vocab));

  std::vector<float> a_from_batch(batch.begin(),
                                  batch.begin() + static_cast<int64_t>(Ta) * vocab);
  const double d = MaxAbsDiff(a_from_batch, a_alone, a_from_batch.size());
  MESSAGE("dense mixed-batch fresh-A vs standalone-A max|diff| = " << d
          << " (garbage-seeded mamba block, zeroing must scrub)");
  CHECK(d < 1e-2);
}

// CHUNKED PREFILL state continuity (the 27B OOM fix's correctness gate): a
// GDN-hybrid sequence prefilled in ONE shot must produce logits bit-identical
// to the SAME sequence prefilled in SEVERAL chunked-prefill steps over the
// persistent KV + GDN (ssm + conv) state. Each resumed chunk carries
// has_initial_state=1 so the GDN recurrence continues from the saved state, the
// causal conv1d reads its saved window, and the full-attn layer attends over the
// already-cached context. If any of those re-zeroed / dropped state, the resumed
// chunks would diverge from the one-shot reference. Proves the state-continuity
// machinery the enabled chunked prefill relies on.
TEST_CASE("qwen27 dense paged: one-shot prefill == chunked prefill (state continuity)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t T = 6, vocab = c.vocab_size;
  const std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};
  const std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5};
  const std::vector<int32_t> blocks = {0, 1};  // block_size 8 => all in block 0

  // One-shot reference: full prefill of the whole sequence (fresh state).
  CachePool ref_pool(c, /*num_blocks=*/8, /*block_size=*/8);
  const CommonAttentionMetadata ref_am = PrefillAttnMeta(T, blocks, 8, 0);
  const GDNAttentionMetadata ref_gm = PrefillGdnMeta(T, 0);
  const std::vector<float> one_shot = Qwen3_5DenseModel::Forward(
      ids, pos, ref_am, ref_gm, ref_pool.attn_kv, ref_pool.gdn_state, w, c, q);
  REQUIRE(one_shot.size() == static_cast<size_t>(T * vocab));

  // Drive the SAME sequence split at these cumulative boundaries, resuming each
  // chunk from the persisted state. Two splittings: {3,3} and {2,2,2} (multiple
  // resumptions). state index 0 for all chunks (single request, single block).
  auto run_chunked = [&](const std::vector<int64_t>& chunk_lens) {
    CachePool pool(c, /*num_blocks=*/8, /*block_size=*/8);
    int64_t context = 0;
    std::vector<float> last_logits;
    for (const int64_t qlen : chunk_lens) {
      std::vector<int32_t> cids(ids.begin() + context,
                                ids.begin() + context + qlen);
      std::vector<int32_t> cpos(pos.begin() + context,
                                pos.begin() + context + qlen);
      const CommonAttentionMetadata am = ChunkAttnMeta(context, qlen, blocks, 8);
      const GDNAttentionMetadata gm =
          ChunkGdnMeta(qlen, /*sidx=*/0, /*has_initial=*/context > 0);
      last_logits = Qwen3_5DenseModel::Forward(cids, cpos, am, gm, pool.attn_kv,
                                               pool.gdn_state, w, c, q);
      context += qlen;
    }
    return last_logits;  // logits of the FINAL chunk's tokens.
  };

  for (const std::vector<int64_t>& split :
       std::vector<std::vector<int64_t>>{{3, 3}, {2, 2, 2}}) {
    const std::vector<float> chunked = run_chunked(split);
    const int64_t tail = split.back();
    REQUIRE(chunked.size() == static_cast<size_t>(tail * vocab));
    // Compare the final chunk's logits against the matching one-shot tail rows.
    std::vector<float> ref_tail(
        one_shot.begin() + static_cast<int64_t>(T - tail) * vocab, one_shot.end());
    const double d = MaxAbsDiff(chunked, ref_tail, chunked.size());
    MESSAGE("chunked (split tail=" << tail << ") vs one-shot max|diff| = " << d
            << " (must be ~0 — bit-identical state continuity)");
    CHECK(d < 1e-4);
  }
}
