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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::DenseMlpWeights;
using vllm::ForwardLogits;
using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::ModelForwardInput;
using vllm::ModelRegistry;
using vllm::Qwen3_5MTPHiddenStates;
using vllm::MergeDenseGateUpGlobals;
using vllm::MergeFullAttnQkvGlobals;
using vllm::Nvfp4Weight;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5AuxTaps;
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
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
  g.batch_ptr = conv.batch_ptr;
  g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
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
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
  g.batch_ptr = conv.batch_ptr;
  g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  return g;
}

}  // namespace

// Mirrors vLLM v0.25.0
// qwen_gdn_linear_attn.py::_forward_core:1286-1298. The packed branch is
// selected only for enabled, pure non-spec decode; every prefill/mixed/spec
// shape remains on the standard recurrence. Locally it additionally requires the
// packed-BA owner, compatible BF16 activation dtypes, and persistent device
// indices.
//
// W1D2's `dense_model` term is gone (GDN-MOE-BF16-OUT, #1168). The `rejects`
// case that pinned it is replaced below by the argument for its removal rather
// than deleted: the term is redundant behind the dtype rule, and what still
// keeps a MoE checkpoint off this path is `has_packed_ba` (#1169), which the
// case below now names.
TEST_CASE("qwen27 packed GDN selection is pure non-spec decode only") {
  vllm::detail::GdnPackedDecodeEligibility e;
  e.runtime_enabled = true;
  e.cuda = true;
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
    // W1D2 asserted here that `dense_model = false` deselects, which was the
    // record of f344decf4's day-one staging gate. GDN-MOE-BF16-OUT (#1168)
    // removed that term, so the assertion is replaced by the two terms that
    // ACTUALLY keep a MoE checkpoint off this path, neither of which is a model
    // shape. The merged BA owner below is built only by the dense loader
    // (qwen3_5_dense_weights.cpp:437) — that is #1169, and it is owed. The
    // second is the dtype rule, already pinned by the `dtype_compatible` case
    // further down: `core_out` is `outdt` and GdnPackedDecodeDTypesCompatible
    // pins it to BF16, so an f32 recurrence output deselects on EITHER arm,
    // which is what made the shape term redundant rather than merely unmeasured.
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

// PERF-27B-GDN-PACKED-REACHABLE (#365). CPU tier: these pin DECISIONS, never
// numbers. The vendored FLA cubin the decision unlocks is CUDA-only (and only
// linked in VLLM_CPP_TRITON builds), so nothing here proves it runs, is
// token-exact, or is faster. What IS provable on CPU is which branch the model
// would take, and that is the whole deliverable of this row.

// The eligibility runs BEFORE ProjectGdnQkvz (it feeds ProjectGdnBA's output
// dtype), so `mixed_qkv`'s dtype is predicted, not observed. The prediction has
// to mirror ProjectGdnQkvz's branch order exactly: merged BF16 owner first,
// then the native-FP8 owner, then the split BF16 owner.
TEST_CASE("qwen27 packed GDN predicts the mixed_qkv dtype ProjectGdnQkvz emits") {
  using vllm::detail::GdnMixedQkvDTypeInputs;
  using vllm::detail::GdnProjectedMixedQkvDType;
  using vt::DType;

  // Merged BF16 owner (the 27B bf16 / 4B path): mixed_qkv is GdnInDType().
  CHECK(GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
            true, false, false, DType::kBF16, DType::kF32}) == DType::kBF16);
  CHECK(GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
            true, false, false, DType::kF32, DType::kF32}) == DType::kF32);

  // Split BF16 owner (GGUF/synthetic): also GdnInDType().
  CHECK(GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
            false, false, false, DType::kBF16, DType::kF32}) == DType::kBF16);

  // Native-FP8 owner (modelopt_mixed): the fp8 epilogue's dtype, NOT
  // GdnInDType() — precisely the case the old fp8-weight term stood in for.
  // At the f32 default the epilogue emits F32 even though GdnInDType() is BF16.
  CHECK(GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
            false, true, true, DType::kBF16, DType::kF32}) == DType::kF32);
  // ...and BF16 once the fp8 GEMM is made to emit it (PERF-FP8-ALPHA-FOLD).
  CHECK(GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
            false, true, true, DType::kBF16, DType::kBF16}) == DType::kBF16);

  // PERF-GDN-PACKED-BRIDGE. `VT_GDN_FP8_IN_BF16` narrows the MERGED fp8 arm
  // only: `fp8_indt` reaches MergedFp8QkvzD and nothing else, while the SPLIT
  // fp8 arm still hardcodes F32 on both of its call paths. So the prediction
  // must know which arm runs. Getting this wrong is not symmetric — predicting
  // BF16 on a checkpoint that takes the split path is the UNSAFE direction, in
  // which vt::GdnPackedDecode is handed an f32 mixed_qkv with bf16 a/b/out and
  // THROWS on its uniformity check.
  CHECK(GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
            false, true, false, DType::kBF16, DType::kBF16}) == DType::kF32);
  // ...and the split arm is F32 whatever the epilogue dtype says.
  CHECK(GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
            false, true, false, DType::kF32, DType::kBF16}) == DType::kF32);

  // Branch ORDER: a checkpoint carrying BOTH owners projects through the BF16
  // one, so neither the FP8 epilogue dtype nor the fp8 arm flag may win.
  CHECK(GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
            true, true, true, DType::kBF16, DType::kF32}) == DType::kBF16);
  CHECK(GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
            true, true, false, DType::kBF16, DType::kBF16}) == DType::kBF16);
}

// PERF-GDN-PACKED-BRIDGE (#365). The single source for the dtype the native-FP8
// merged GDN in_proj emits. ONE function answers it for BOTH the producer
// (ProjectGdnQkvz's `fp8_indt`, handed to MergedFp8QkvzD) and the predictor
// (GdnFp8MixedQkvDType, which the packed-decode eligibility reads BEFORE the
// projection has run). They cannot drift because they are the same call.
//
// The three terms are PERF-FP8-ALPHA-FOLD's own, kept verbatim: the opt-in
// toggle, `indt == BF16` (honouring VT_GDN_IN_BF16's documented rollback, since
// that lever is the one being unblocked), and `outdt == BF16`, which keeps the
// chain dtype-uniform.
//
// GDN-MOE-BF16-OUT (#1168), and #521 asked for exactly this correction. That
// third term USED to be described here as what "confines this to the dense 27B —
// the 35B is MoE, so GdnOutDType is f32 there and the whole arm stays inert".
// `GdnOutDType()` no longer branches on model shape, so `outdt == BF16` is now
// unconditional at the default and the term excludes NO checkpoint. What keeps
// this arm inert on the 35B is the DEFAULT-OFF `VT_GDN_FP8_IN_BF16` toggle, and
// that is the only remaining bound — which is why the toggle case below is the
// one a reader must not mistake for a shape guard.
TEST_CASE("qwen27 fp8 merged in_proj dtype is one decision, all three terms") {
  using vllm::detail::GdnFp8MergedMixedQkvDType;
  using vt::DType;

  // All three: BF16. This is the ONLY combination that narrows.
  CHECK(GdnFp8MergedMixedQkvDType(true, DType::kBF16, DType::kBF16) ==
        DType::kBF16);

  // Toggle OFF is the shipped default and must be F32 regardless.
  CHECK(GdnFp8MergedMixedQkvDType(false, DType::kBF16, DType::kBF16) ==
        DType::kF32);

  // Each dtype term is independently necessary.
  CHECK(GdnFp8MergedMixedQkvDType(true, DType::kF32, DType::kBF16) ==
        DType::kF32);  // VT_GDN_IN_BF16=0 rollback.
  CHECK(GdnFp8MergedMixedQkvDType(true, DType::kBF16, DType::kF32) ==
        DType::kF32);  // the VT_GDN_OUT_BF16=0 rollback, on EITHER arm (#1168).
  CHECK(GdnFp8MergedMixedQkvDType(true, DType::kF32, DType::kF32) ==
        DType::kF32);
  CHECK(GdnFp8MergedMixedQkvDType(false, DType::kF32, DType::kF32) ==
        DType::kF32);
}

// PERF-GDN-PACKED-BRIDGE (#365). The composition end to end, as a decision:
// does the packed arm become eligible when — and ONLY when — the fp8 merged
// in_proj actually emits BF16? This is the assertion that was false before the
// bridge: both toggles ON still predicted F32, so the arm stayed inert and any
// A/B measured nothing.
TEST_CASE("qwen27 fp8 tower reaches the packed dtype rule only when it emits BF16") {
  using vllm::detail::GdnFp8MergedMixedQkvDType;
  using vllm::detail::GdnMixedQkvDTypeInputs;
  using vllm::detail::GdnPackedDecodeDTypes;
  using vllm::detail::GdnPackedDecodeDTypesCompatible;
  using vllm::detail::GdnProjectedMixedQkvDType;
  using vt::DType;

  // The real modelopt_mixed 27B: no bf16 qkvz owner, fp8 owner, merged arm.
  // `ba_out`/`core_out` are BF16 at their defaults (MergedGdnBaOutputDType under
  // packed, GdnOutDType dense) and the SSM cache is F32, native to the
  // checkpoint's own `mamba_ssm_dtype: "float32"`.
  auto rule_for = [](bool toggle_on) {
    const DType mixed = GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
        false, true, true, DType::kBF16,
        GdnFp8MergedMixedQkvDType(toggle_on, DType::kBF16, DType::kBF16)});
    return GdnPackedDecodeDTypesCompatible(
        GdnPackedDecodeDTypes{mixed, DType::kBF16, DType::kBF16, DType::kF32});
  };

  CHECK_FALSE(rule_for(false));  // shipped default: fp8 tower stays unpacked.
  CHECK(rule_for(true));         // VT_GDN_FP8_IN_BF16=1: packed becomes eligible.

  // The SPLIT fp8 arm never becomes eligible, toggle or not — it still emits
  // F32, and predicting otherwise is the throw described above.
  auto split_rule_for = [](bool toggle_on) {
    const DType mixed = GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
        false, true, false, DType::kBF16,
        GdnFp8MergedMixedQkvDType(toggle_on, DType::kBF16, DType::kBF16)});
    return GdnPackedDecodeDTypesCompatible(
        GdnPackedDecodeDTypes{mixed, DType::kBF16, DType::kBF16, DType::kF32});
  };
  CHECK_FALSE(split_rule_for(false));
  CHECK_FALSE(split_rule_for(true));
}

// The rule itself: vt::GdnPackedDecode's uniformity contract over the four
// ACTIVATION tensors, pinned to BF16, with an INDEPENDENT state dtype. No term
// keys on how the weights are stored.
TEST_CASE("qwen27 packed GDN dtype rule is uniformity + BF16, state independent") {
  using vllm::detail::GdnPackedDecodeDTypes;
  using vllm::detail::GdnPackedDecodeDTypesCompatible;
  using vt::DType;

  const GdnPackedDecodeDTypes ok{DType::kBF16, DType::kBF16, DType::kBF16,
                                 DType::kF32};
  CHECK(GdnPackedDecodeDTypesCompatible(ok));

  // The state is independent of the activation quartet: every dtype the GDN
  // state cache can hold is accepted (mirrors FLA, which casts it on load).
  {
    auto x = ok;
    x.ssm_state = DType::kBF16;
    CHECK(GdnPackedDecodeDTypesCompatible(x));
  }
  {
    auto x = ok;
    x.ssm_state = DType::kF16;
    CHECK(GdnPackedDecodeDTypesCompatible(x));
  }
  {
    auto x = ok;
    x.ssm_state = DType::kI32;
    CHECK_FALSE(GdnPackedDecodeDTypesCompatible(x));
  }

  // Each single-field activation deviation is rejected on its own.
  {
    auto x = ok;
    x.mixed_qkv = DType::kF32;  // today's fp8 GDN tower.
    CHECK_FALSE(GdnPackedDecodeDTypesCompatible(x));
  }
  {
    auto x = ok;
    x.ba_out = DType::kF32;  // VT_GDN_BA_OUT_BF16=0.
    CHECK_FALSE(GdnPackedDecodeDTypesCompatible(x));
  }
  {
    auto x = ok;
    x.core_out = DType::kF32;  // VT_GDN_OUT_BF16=0.
    CHECK_FALSE(GdnPackedDecodeDTypesCompatible(x));
  }
  // Uniform but not BF16 is still rejected: the model's packed leg is pinned to
  // the dtype the hand kernel and the vendored cubin are exercised at.
  CHECK_FALSE(GdnPackedDecodeDTypesCompatible(GdnPackedDecodeDTypes{
      DType::kF32, DType::kF32, DType::kF32, DType::kF32}));
}

// VT_GDN_PACKED_DECODE_FP8_TOWER: DEFAULT OFF, house '1'-leading convention.
TEST_CASE("qwen27 packed GDN fp8-tower toggle defaults OFF") {
  using vllm::detail::PackedGdnDecodeFp8TowerFlagIsOn;

  CHECK_FALSE(PackedGdnDecodeFp8TowerFlagIsOn(nullptr));  // unset.
  CHECK_FALSE(PackedGdnDecodeFp8TowerFlagIsOn(""));
  CHECK_FALSE(PackedGdnDecodeFp8TowerFlagIsOn("0"));
  CHECK_FALSE(PackedGdnDecodeFp8TowerFlagIsOn("2"));
  CHECK_FALSE(PackedGdnDecodeFp8TowerFlagIsOn("on"));
  CHECK(PackedGdnDecodeFp8TowerFlagIsOn("1"));
  CHECK(PackedGdnDecodeFp8TowerFlagIsOn("1x"));
}

// GDN-MOE-BF16-OUT (#1168), Edit 1. The GDN recurrence-output dtype is resolved
// from the ENVIRONMENT alone, and the resolver has no model-shape input to give
// it a second answer. `GdnOutDType()` used to take a `bool dense_model` and
// default to it, so a reader had to visit every call site to learn what the
// default was and a new call site could reintroduce the dense/MoE split
// silently. vLLM has no such parameter because it has no such decision: it
// resolves ONE model dtype and every layer inherits it
// (qwen_gdn_linear_attn.py:870-873, :843, :459-465 @ 5559679).
//
// The truth table is asked of the pure helper the production resolver reads,
// mirroring PackedGdnDecodeFp8TowerFlagIsOn: `GdnOutDType()` caches its getenv
// in a function-local static, so one process can only ever observe one value of
// it. `nullptr` IS the production default, and it answers bf16 with nothing in
// the signature that could say dense or MoE. Default-ON — anything that is not
// a leading '0' is bf16 — the polarity VT_GDN_IN_BF16 uses and the one
// PackedGdnDecodeEnvSelected already mirrors for VT_GDN_OUT_BF16.
//
// What this case CANNOT say is what the model runs; that is
// `test_qwen35_paged_forward`'s "the GDN recurrence output and z gate are bf16",
// which enters through ModelRegistry::Forward on a MoE config.
TEST_CASE("qwen27 GDN out dtype is bf16 by default and keys on no model shape") {
  using vllm::detail::GdnOutBf16FlagIsOn;

  CHECK(GdnOutBf16FlagIsOn(nullptr));  // unset -> bf16, on EITHER arm.
  CHECK_FALSE(GdnOutBf16FlagIsOn("0"));
  CHECK_FALSE(GdnOutBf16FlagIsOn("0x"));  // leading char decides, as elsewhere.
  CHECK(GdnOutBf16FlagIsOn("1"));
  CHECK(GdnOutBf16FlagIsOn(""));  // not a leading '0'.
}

// GDN-MOE-BF16-OUT (#1168), Edit 2. The eligibility carries NO model-shape term:
// an eligibility whose every remaining term is true selects packed decode, and
// nothing in it can say whether the checkpoint is dense or MoE.
//
// The `dense_model` term this replaces entered at f344decf4 as one of that
// change's "real-model safety gates" and was never revisited. Neither reference
// has an equivalent: VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE defaults True with
// no shape term (vllm/envs.py:124 @ 5559679), and SGLang keys
// `supports_packed_decode` on the platform alone (gdn_triton.py:43 @ f63458b5be).
// It became REDUNDANT rather than merely unsupported once GdnOutDType stopped
// branching on model shape: `core_out` is `outdt` and
// GdnPackedDecodeDTypesCompatible already pins it to BF16, so the dtype rule is
// what excludes an f32 recurrence output on either arm.
//
// This case deliberately never NAMES a model-shape field, so it says the same
// thing before and after the field exists.
TEST_CASE("qwen27 packed GDN selection carries no model-shape term") {
  vllm::detail::GdnPackedDecodeEligibility e;
  e.runtime_enabled = true;
  e.cuda = true;
  e.has_packed_ba = true;
  e.merged_ba_enabled = true;
  e.dtype_compatible = true;
  e.has_state_indices = true;
  e.num_actual_tokens = 4;
  e.num_decodes = 4;
  e.num_decode_tokens = 4;

  CHECK(vllm::detail::ShouldUsePackedGdnDecode(e));
}

// THE ROW. The eligibility must key on the dtypes the packed op needs and NOT
// on "the GDN weights happen to be fp8". Composing the three helpers exactly as
// GdnBlockPaged does, an fp8 tower and a bf16 tower that produce the SAME
// activation dtypes must reach the SAME decision.
TEST_CASE("qwen27 packed GDN selection keys on dtypes, not on fp8 weights") {
  using vllm::detail::GdnMixedQkvDTypeInputs;
  using vllm::detail::GdnPackedDecodeDTypes;
  using vllm::detail::GdnPackedDecodeDTypesCompatible;
  using vllm::detail::GdnPackedDecodeEligibility;
  using vllm::detail::GdnProjectedMixedQkvDType;
  using vllm::detail::PackedGdnDecodeFp8TowerFlagIsOn;
  using vllm::detail::ShouldUsePackedGdnDecode;
  using vt::DType;

  // The exact call-site composition (qwen3_5.cpp GdnBlockPaged): predict
  // mixed_qkv, apply the dtype rule, then the default-OFF fp8-tower clause.
  auto selects = [](bool fp8_tower, DType fp8_out_dtype, DType ssm_state,
                    const char* toggle) {
    const bool bf16_owner = !fp8_tower;
    // PERF-GDN-PACKED-BRIDGE: an fp8 tower here means the MERGED arm, which is
    // the one VT_GDN_FP8_IN_BF16 narrows and the only one that can reach packed.
    const DType mixed = GdnProjectedMixedQkvDType(GdnMixedQkvDTypeInputs{
        bf16_owner, fp8_tower, fp8_tower, DType::kBF16, fp8_out_dtype});
    GdnPackedDecodeEligibility e;
    e.runtime_enabled = true;
    e.cuda = true;
    e.has_packed_ba = true;
    e.merged_ba_enabled = true;
    e.dtype_compatible =
        GdnPackedDecodeDTypesCompatible(GdnPackedDecodeDTypes{
            mixed, DType::kBF16, DType::kBF16, ssm_state}) &&
        (!fp8_tower || PackedGdnDecodeFp8TowerFlagIsOn(toggle));
    e.has_state_indices = true;
    e.num_actual_tokens = 4;
    e.num_decodes = 4;
    e.num_decode_tokens = 4;
    return ShouldUsePackedGdnDecode(e);
  };

  // BF16 tower: selected, and the toggle is irrelevant to it (the fp8 clause
  // can only ever PERMIT, never deselect).
  CHECK(selects(false, DType::kF32, DType::kF32, nullptr));
  CHECK(selects(false, DType::kF32, DType::kF32, "1"));
  CHECK(selects(false, DType::kF32, DType::kBF16, nullptr));

  // FP8 tower, toggle ON, and the fp8 GEMM emits BF16: SELECTED. The weights
  // are fp8 and it no longer matters — this is the case the old
  // `in_proj_qkv_fp8.Empty()` term excluded by construction.
  CHECK(selects(true, DType::kBF16, DType::kF32, "1"));
  // ...reaching the SAME decision as a bf16 tower with the same dtypes.
  CHECK(selects(true, DType::kBF16, DType::kF32, "1") ==
        selects(false, DType::kF32, DType::kF32, "1"));

  // FP8 tower, toggle ON, but the fp8 GEMM still emits F32 (this base SHA):
  // NOT selected — correctly, by the dtype rule rather than by construction.
  CHECK_FALSE(selects(true, DType::kF32, DType::kF32, "1"));
  // A bf16 tower whose mixed_qkv is f32 is rejected identically, which is what
  // "does not key on fp8 per se" means: the SAME dtypes give the SAME answer.
  {
    const DType mixed_f32 = GdnProjectedMixedQkvDType(
        GdnMixedQkvDTypeInputs{true, false, false, DType::kF32, DType::kF32});
    CHECK(mixed_f32 == DType::kF32);
    CHECK_FALSE(GdnPackedDecodeDTypesCompatible(GdnPackedDecodeDTypes{
        mixed_f32, DType::kBF16, DType::kBF16, DType::kF32}));
  }

  // DEFAULT OFF: with the toggle unset the fp8 tower stays excluded even when
  // every dtype lines up, so production selection is byte-identical to before.
  CHECK_FALSE(selects(true, DType::kBF16, DType::kF32, nullptr));
  CHECK_FALSE(selects(true, DType::kBF16, DType::kF32, "0"));
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

// PERF-27B-GDN-FP8-QKVZ — the FP8 leaf's dispatch predicate. vLLM issues ONE
// merged qkvz GEMM per GDN layer on this tower too (the 27B NVFP4 checkpoint is
// `modelopt_mixed`, so its GDN input projections are FP8 W8A8 and the BF16
// merged owner is empty). The merged fp8 GEMM quantizes the shared activation
// ONCE, so a shard pair that does not agree bitwise on `input_scale` MUST stay
// on the exact two legacy GEMMs. Every term is load-invariant and checked once.
TEST_CASE("qwen27 merged FP8 qkvz selection requires every guard") {
  vllm::detail::GdnMergedFp8QkvzEligibility e;
  e.runtime_enabled = true;
  e.fp8_platform = true;
  e.has_fp8_shards = true;
  e.shared_k = true;
  e.shared_input_scale = true;
  e.shard_widths_match = true;

  CHECK(vllm::detail::ShouldUseMergedGdnFp8Qkvz(e));

  {
    auto x = e;
    x.runtime_enabled = false;  // VT_GDN_MERGED_QKVZ_FP8=0 (or the BF16 leaf's
                                // VT_GDN_MERGED_QKVZ=0 / VT_GDN_MERGED_PROJ=0).
    CHECK_FALSE(vllm::detail::ShouldUseMergedGdnFp8Qkvz(x));
  }
  {
    auto x = e;
    x.fp8_platform = false;  // CPU / no fp8 GEMM registered.
    CHECK_FALSE(vllm::detail::ShouldUseMergedGdnFp8Qkvz(x));
  }
  {
    auto x = e;
    x.has_fp8_shards = false;  // 27B BF16 merged owner / GGUF / synthetic.
    CHECK_FALSE(vllm::detail::ShouldUseMergedGdnFp8Qkvz(x));
  }
  {
    auto x = e;
    x.shared_k = false;  // the two shards do not read one [M,K] activation.
    CHECK_FALSE(vllm::detail::ShouldUseMergedGdnFp8Qkvz(x));
  }
  {
    auto x = e;
    x.shared_input_scale = false;  // THE scale-compatibility stop condition.
    CHECK_FALSE(vllm::detail::ShouldUseMergedGdnFp8Qkvz(x));
  }
  {
    auto x = e;
    x.shard_widths_match = false;  // shard N != conv_dim / value_dim.
    CHECK_FALSE(vllm::detail::ShouldUseMergedGdnFp8Qkvz(x));
  }
}

// The load-time scale-compatibility predicate itself. ModelOpt FP8 here is
// PER-TENSOR: the merged GEMM can only reproduce both split GEMMs when the two
// shards quantize the activation with the identical scalar, so the comparison is
// exact float equality — one ulp apart must fall back. This is the same
// predicate `Fp8SharedInputScale` applies to this pair for the fused
// RmsNorm+quant, so the two guards cannot drift.
TEST_CASE("qwen27 GDN fp8 shared input_scale is exact, not approximate") {
  const auto shard = [](int64_t n, int64_t k, float input_scale,
                        float weight_scale) {
    vllm::Fp8Weight w;
    w.n = n;
    w.k = k;
    w.input_scale = input_scale;
    w.weight_scale = weight_scale;
    w.alpha = input_scale * weight_scale;
    w.packed.dtype = DType::kI8;
    w.packed.rank = 2;
    w.packed.shape[0] = n;
    w.packed.shape[1] = k;
    w.packed.bytes.resize(static_cast<size_t>(n * k));
    return w;
  };

  const float base = 0.0078125F;
  const float one_ulp = std::nextafter(base, 1.0F);
  REQUIRE(base != one_ulp);

  {
    vllm::GdnLayerWeights g;
    g.in_proj_qkv_fp8 = shard(10240, 5120, base, 0.0007629395F);
    g.in_proj_z_fp8 = shard(6144, 5120, base, 0.0005340576F);
    float scale = 0.0F;
    CHECK(vllm::detail::GdnFp8SharedInputScale(g, &scale));
    CHECK(scale == base);
  }
  {
    // One ulp apart: NOT mergeable. The activation each split GEMM quantizes
    // would differ, so one merged GEMM cannot reproduce both.
    vllm::GdnLayerWeights g;
    g.in_proj_qkv_fp8 = shard(10240, 5120, base, 0.0007629395F);
    g.in_proj_z_fp8 = shard(6144, 5120, one_ulp, 0.0005340576F);
    float scale = -1.0F;
    CHECK_FALSE(vllm::detail::GdnFp8SharedInputScale(g, &scale));
    CHECK(scale == -1.0F);  // untouched on rejection.
  }
  {
    // Differing WEIGHT scales are fine — each shard's folded alpha is applied
    // per output column, so only the activation scale has to agree.
    vllm::GdnLayerWeights g;
    g.in_proj_qkv_fp8 = shard(10240, 5120, base, 0.0007629395F);
    g.in_proj_z_fp8 = shard(6144, 5120, base, 0.25F);
    CHECK(vllm::detail::GdnFp8SharedInputScale(g, nullptr));
  }
  {
    // A non-FP8 owner (27B BF16 merged / GGUF / synthetic) is never mergeable.
    vllm::GdnLayerWeights g;
    g.in_proj_qkv_fp8 = shard(10240, 5120, base, 0.0007629395F);
    CHECK_FALSE(vllm::detail::GdnFp8SharedInputScale(g, nullptr));
  }
}

// The merged-FP8 leaf's rollback env truth table, pinned on the CPU tier the
// same way PackedGdnDecodeEnvSelected is: VT_GDN_MERGED_QKVZ_FP8 is the leaf
// switch and the BF16 leaf's master/leaf rollbacks also turn it off, so one
// switch can retire the whole merged-input-projection topology.
TEST_CASE("qwen27 merged FP8 qkvz env rollback truth table") {
  using vllm::detail::GdnMergedFp8QkvzEnvConfig;
  using vllm::detail::MergedGdnFp8QkvzEnvSelected;

  CHECK(MergedGdnFp8QkvzEnvSelected(GdnMergedFp8QkvzEnvConfig{}));  // all unset.
  CHECK(MergedGdnFp8QkvzEnvSelected(
      GdnMergedFp8QkvzEnvConfig{"1", "1", "1"}));
  CHECK_FALSE(MergedGdnFp8QkvzEnvSelected(
      GdnMergedFp8QkvzEnvConfig{"0", nullptr, nullptr}));  // master off.
  CHECK_FALSE(MergedGdnFp8QkvzEnvSelected(
      GdnMergedFp8QkvzEnvConfig{nullptr, "0", nullptr}));  // BF16 leaf off.
  CHECK_FALSE(MergedGdnFp8QkvzEnvSelected(
      GdnMergedFp8QkvzEnvConfig{nullptr, nullptr, "0"}));  // FP8 leaf off.
}

// The 27B gate's packed-dispatch-count contract must agree with the engine's
// process-cached env couplings: ShouldUsePackedGdnDecode requires
// merged_ba_enabled (master VT_GDN_MERGED_PROJ + leaf VT_GDN_MERGED_BA) and
// the coupled BF16 dtypes (VT_GDN_IN_BF16 / VT_GDN_OUT_BF16 /
// VT_GDN_BA_OUT_BF16), not just VT_GDN_PACKED_DECODE. DGX gate arm 2b
// (VT_GDN_MERGED_PROJ=0 at baea3ec) proved the old expectation wrong at run
// time: the engine correctly ran the decomposed recurrence (zero packed
// launches, 229/229 token assertions green) while the test still demanded 48.
// This truth table pins the shared helper the gate test now consumes.
TEST_CASE("qwen27 packed GDN env selection mirrors every coupled rollback") {
  using vllm::detail::GdnPackedDecodeEnvConfig;
  using vllm::detail::PackedGdnDecodeEnvSelected;

  // Default production env (all unset): packed is selected (48 launches).
  CHECK(PackedGdnDecodeEnvSelected(GdnPackedDecodeEnvConfig{}));

  auto with = [](const char* packed, const char* proj, const char* ba,
                 const char* in, const char* out, const char* ba_out) {
    GdnPackedDecodeEnvConfig env;
    env.packed_decode = packed;
    env.merged_proj = proj;
    env.merged_ba = ba;
    env.in_bf16 = in;
    env.out_bf16 = out;
    env.ba_out_bf16 = ba_out;
    return env;
  };

  // Direct rollback.
  CHECK_FALSE(PackedGdnDecodeEnvSelected(
      with("0", nullptr, nullptr, nullptr, nullptr, nullptr)));
  // Master merged-projection rollback splits BA -> decomposed recurrence.
  CHECK_FALSE(PackedGdnDecodeEnvSelected(
      with(nullptr, "0", nullptr, nullptr, nullptr, nullptr)));
  // Leaf merged-BA rollback likewise.
  CHECK_FALSE(PackedGdnDecodeEnvSelected(
      with(nullptr, nullptr, "0", nullptr, nullptr, nullptr)));
  // Coupled-dtype overrides break dtype_compatible.
  CHECK_FALSE(PackedGdnDecodeEnvSelected(
      with(nullptr, nullptr, nullptr, "0", nullptr, nullptr)));
  CHECK_FALSE(PackedGdnDecodeEnvSelected(
      with(nullptr, nullptr, nullptr, nullptr, "0", nullptr)));
  CHECK_FALSE(PackedGdnDecodeEnvSelected(
      with(nullptr, nullptr, nullptr, nullptr, nullptr, "0")));

  // Explicit "1" (and non-"0") values keep the default selection.
  CHECK(PackedGdnDecodeEnvSelected(with("1", "1", "1", "1", "1", "1")));
  // The qkvz leaf is NOT coupled to packed decode (DGX arm 2a passed with 48
  // packed launches under VT_GDN_MERGED_QKVZ=0) — no qkvz field exists here.
  // Master off dominates even with the leaf explicitly on.
  CHECK_FALSE(PackedGdnDecodeEnvSelected(
      with(nullptr, "0", "1", nullptr, nullptr, nullptr)));
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

TEST_CASE(
    "qwen27 GDN state-index uniqueness is O(n) and force-full re-verifies") {
  // The default fast path is a single O(n) seen-set pass (no O(n^2) inner
  // scan): it still fails closed on a duplicate that is NOT adjacent, on an
  // out-of-range live slot, and on a non-(-1) negative, while accepting the
  // inert -1 padding sentinel. state_slots (== max_num_reqs) bounds the pass.
  CHECK_NOTHROW(vllm::detail::ValidateGdnStateIndices(
      std::vector<int32_t>{3, 0, 2, 1, -1}, /*required=*/5, /*slots=*/4));
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnStateIndices(
          std::vector<int32_t>{3, 0, 2, 3}, /*required=*/4, /*slots=*/4),
      doctest::Contains("duplicate live GDN state index"), std::runtime_error);
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnStateIndices(
          std::vector<int32_t>{0, -2}, /*required=*/2, /*slots=*/4),
      doctest::Contains("invalid negative GDN state index"),
      std::runtime_error);

  // force_full_uniqueness=true adds the exhaustive O(n^2) pairwise
  // cross-verification (what VT_GDN_VALIDATE=1 drives globally): identical
  // fail-closed verdicts, a redundant paranoid check for debugging.
  CHECK_NOTHROW(vllm::detail::ValidateGdnStateIndices(
      std::vector<int32_t>{0, 1, 2, 3}, /*required=*/4, /*slots=*/4,
      /*force_full_uniqueness=*/true));
  CHECK_THROWS_WITH_AS(
      vllm::detail::ValidateGdnStateIndices(
          std::vector<int32_t>{1, 3, 1}, /*required=*/3, /*slots=*/4,
          /*force_full_uniqueness=*/true),
      doctest::Contains("duplicate live GDN state index"), std::runtime_error);
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
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
  gm.batch_ptr = conv.batch_ptr;
  gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;

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

// SPEC-MTP I5d-pre: the hidden-tap out-field on the type-erased ModelForwardInput.
// When set, ModelRegistry::Forward routes the Qwen3.5 dense forward to
// ForwardDeviceTap, which (a) returns byte-identical logits to the untapped
// forward and (b) captures the full [T,H] post-final-norm hidden into the
// carrier. RED-first: before this increment ModelForwardInput has no hidden_tap
// field and the forward never populates the carrier.
TEST_CASE("qwen27 hidden-tap out-field captures post-final-norm [T,H], inert on logits") {
  const HfConfig c = MakeConfig();
  const Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t T = 6, vocab = c.vocab_size, H = c.hidden_size;
  const std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};
  const std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5};
  const std::vector<int32_t> logits_indices;  // full [T, vocab].

  // Fresh KV/GDN state per forward (each forward writes the caches).
  CachePool pool_ref(c, /*num_blocks=*/8, /*block_size=*/8);
  CachePool pool_tap(c, /*num_blocks=*/8, /*block_size=*/8);
  CachePool pool_plain(c, /*num_blocks=*/8, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillAttnMeta(T, {0, 1}, 8, 0);
  const GDNAttentionMetadata gm = PrefillGdnMeta(T, 0);

  // Reference: the concrete ForwardDeviceTap directly (the I5c primitive this
  // seam routes to).
  Qwen3_5MTPHiddenStates ref_carrier;
  const ForwardLogits ref = Qwen3_5DenseModel::ForwardDeviceTap(
      ids, pos, am, gm, pool_ref.attn_kv, pool_ref.gdn_state, w, c, q,
      &ref_carrier, logits_indices);

  // Type-erased forward WITH the tap set (the new routing).
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(w);
  Qwen3_5MTPHiddenStates tap_carrier;
  ModelForwardInput in_tap{ids,
                           pos,
                           am,
                           gm,
                           pool_tap.attn_kv,
                           pool_tap.gdn_state,
                           c,
                           q,
                           logits_indices};
  in_tap.num_reqs = 1;
  in_tap.hidden_tap = &tap_carrier;
  const ForwardLogits routed_tap = ModelRegistry::Forward(*model, in_tap);

  // Type-erased forward WITHOUT the tap (the current default path).
  ModelForwardInput in_plain{ids,
                             pos,
                             am,
                             gm,
                             pool_plain.attn_kv,
                             pool_plain.gdn_state,
                             c,
                             q,
                             logits_indices};
  in_plain.num_reqs = 1;
  const ForwardLogits routed_plain = ModelRegistry::Forward(*model, in_plain);

  // The tap captured the full [T,H] post-final-norm hidden (bf16).
  REQUIRE(tap_carrier.storage != nullptr);
  REQUIRE(tap_carrier.tensor.data != nullptr);
  REQUIRE(tap_carrier.tensor.rank == 2);
  CHECK(tap_carrier.tensor.shape[0] == T);
  CHECK(tap_carrier.tensor.shape[1] == H);
  CHECK(tap_carrier.tensor.dtype == DType::kBF16);

  // Routing goes to ForwardDeviceTap: the captured hidden is bit-identical to the
  // direct reference carrier (bf16 raw bytes).
  REQUIRE(ref_carrier.tensor.data != nullptr);
  CHECK(std::memcmp(tap_carrier.tensor.data, ref_carrier.tensor.data,
                    static_cast<size_t>(T * H) * sizeof(uint16_t)) == 0);

  // Inert on logits: the tapped forward's logits are byte-identical to the
  // untapped default path (and to the direct reference).
  REQUIRE(routed_tap.on_device());
  REQUIRE(routed_plain.on_device());
  REQUIRE(routed_tap.rows == T);
  REQUIRE(routed_tap.vocab == vocab);
  REQUIRE(routed_plain.rows == routed_tap.rows);
  const size_t nbytes = static_cast<size_t>(T * vocab) * sizeof(float);
  CHECK(std::memcmp(routed_tap.device_tensor.data,
                    routed_plain.device_tensor.data, nbytes) == 0);
  CHECK(std::memcmp(routed_tap.device_tensor.data, ref.device_tensor.data,
                    nbytes) == 0);
}

// SPEC-DFLASH D1 (DF-AUX-TAPS): the multi-layer aux-tap out-field on the type-erased
// ModelForwardInput. ForwardDeviceMultiTap captures (hidden + residual) — the exact
// value vLLM appends via eagle3 _maybe_add_hidden_state (interfaces.py:1382) at aux key
// L+1 — at each configured target_layer_id L into a [T, H×taps] buffer whose column
// order is ascending layer_ids (== cat(aux, dim=-1) fed to the DFlash fc). It returns
// byte-identical logits and is INERT when aux_tap is null.
//
// REFERENCE per tap is an INDEPENDENT truncated-model rebuild: a model with only the
// first L+1 layers (layers > L cannot affect the depth-L residual stream, and
// MakeWeights is deterministic per-layer so layers 0..L are bit-identical) captured at
// its own last layer reproduces the full model's tap for L. This validates the captured
// VALUE, the LAYER index, and the concat SLOT together. RED-first: a wrong tap layer or
// wrong concat order would map a column to the wrong depth and fail (the explicit
// negative check below shows layer 0's column differs from the layer-2 reference).
TEST_CASE("qwen27 DFlash multi-tap captures [T,H×taps] at configured layers, inert on logits") {
  const HfConfig c = MakeConfig();  // 4 layers: lin, lin, lin, full
  const Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t T = 6, vocab = c.vocab_size, H = c.hidden_size;
  const std::vector<int32_t> ids = {5, 9, 2, 31, 17, 3};
  const std::vector<int32_t> pos = {0, 1, 2, 3, 4, 5};
  const std::vector<int32_t> logits_indices;              // full [T, vocab]
  const std::vector<int32_t> layer_ids = {0, 2, 3};       // ascending; intermediate + last
  const int64_t taps = static_cast<int64_t>(layer_ids.size());

  const CommonAttentionMetadata am = PrefillAttnMeta(T, {0, 1}, 8, 0);
  const GDNAttentionMetadata gm = PrefillGdnMeta(T, 0);

  // Full-model multi-tap forward.
  CachePool pool_aux(c, /*num_blocks=*/8, /*block_size=*/8);
  Qwen3_5AuxTaps aux;
  aux.layer_ids = layer_ids;
  const ForwardLogits multi = Qwen3_5DenseModel::ForwardDeviceMultiTap(
      ids, pos, am, gm, pool_aux.attn_kv, pool_aux.gdn_state, w, c, q, &aux,
      logits_indices);

  // Shape / dtype: [T, H×taps] bf16.
  REQUIRE(aux.storage != nullptr);
  REQUIRE(aux.tensor.data != nullptr);
  REQUIRE(aux.tensor.rank == 2);
  CHECK(aux.tensor.shape[0] == T);
  CHECK(aux.tensor.shape[1] == H * taps);
  CHECK(aux.tensor.dtype == DType::kBF16);

  const uint16_t* aux_raw = static_cast<const uint16_t*>(aux.tensor.data);
  auto aux_col = [&](int64_t k, int64_t t, int64_t h) -> uint16_t {
    return aux_raw[t * (H * taps) + k * H + h];  // column block k = tap for layer_ids[k]
  };

  // Independent per-layer reference via a truncated rebuild (first L+1 layers only).
  auto ref_tap_for_layer = [&](int64_t L) {
    HfConfig cL = c;
    cL.num_hidden_layers = static_cast<int>(L + 1);
    cL.layer_types.assign(c.layer_types.begin(),
                          c.layer_types.begin() + static_cast<size_t>(L + 1));
    Qwen3_5DenseWeights wL = MakeWeights(cL);
    CachePool poolL(cL, 8, 8);
    const CommonAttentionMetadata amL = PrefillAttnMeta(T, {0, 1}, 8, 0);
    const GDNAttentionMetadata gmL = PrefillGdnMeta(T, 0);
    auto tapL = std::make_shared<Qwen3_5AuxTaps>();
    tapL->layer_ids = {static_cast<int32_t>(L)};
    Qwen3_5DenseModel::ForwardDeviceMultiTap(ids, pos, amL, gmL, poolL.attn_kv,
                                             poolL.gdn_state, wL, cL, q, tapL.get(),
                                             std::vector<int32_t>{});
    return tapL;  // owns its device storage
  };

  // Every configured tap's column == the independent truncated reference at that layer.
  for (int64_t k = 0; k < taps; ++k) {
    const int64_t L = layer_ids[static_cast<size_t>(k)];
    auto ref = ref_tap_for_layer(L);
    REQUIRE(ref->tensor.shape[0] == T);
    REQUIRE(ref->tensor.shape[1] == H);
    const uint16_t* rb = static_cast<const uint16_t*>(ref->tensor.data);
    for (int64_t t = 0; t < T; ++t)
      for (int64_t h = 0; h < H; ++h)
        CHECK(aux_col(k, t, h) == rb[t * H + h]);
  }

  // RED-first: a wrong tap layer / wrong concat order is DETECTABLE. Column 0
  // (layer 0's residual) must differ from the layer-2 reference — distinct
  // residual-stream values across depth.
  {
    auto ref2 = ref_tap_for_layer(2);
    const uint16_t* rb2 = static_cast<const uint16_t*>(ref2->tensor.data);
    int diffs = 0;
    for (int64_t t = 0; t < T; ++t)
      for (int64_t h = 0; h < H; ++h)
        if (aux_col(0, t, h) != rb2[t * H + h]) ++diffs;
    CHECK(diffs > 0);
  }

  // Inert on logits: the multi-tap forward's logits are byte-identical to the
  // untapped default path routed through ModelRegistry::Forward.
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::BorrowQwen3_5DenseLoadedModel(w);
  CachePool pool_plain(c, 8, 8);
  ModelForwardInput in_plain{ids,  pos,
                             am,   gm,
                             pool_plain.attn_kv,
                             pool_plain.gdn_state,
                             c,    q,
                             logits_indices};
  in_plain.num_reqs = 1;
  const ForwardLogits routed_plain = ModelRegistry::Forward(*model, in_plain);
  REQUIRE(multi.on_device());
  REQUIRE(routed_plain.on_device());
  REQUIRE(multi.rows == T);
  REQUIRE(multi.vocab == vocab);
  CHECK(std::memcmp(multi.device_tensor.data, routed_plain.device_tensor.data,
                    static_cast<size_t>(T * vocab) * sizeof(float)) == 0);

  // Routing parity: ModelRegistry::Forward with aux_tap set reaches ForwardDeviceMultiTap
  // (same logits + a fresh [T,H×taps] capture bit-identical to the direct call).
  CachePool pool_route(c, 8, 8);
  Qwen3_5AuxTaps aux_routed;
  aux_routed.layer_ids = layer_ids;
  ModelForwardInput in_aux{ids,  pos,
                           am,   gm,
                           pool_route.attn_kv,
                           pool_route.gdn_state,
                           c,    q,
                           logits_indices};
  in_aux.num_reqs = 1;
  in_aux.aux_tap = &aux_routed;
  const ForwardLogits routed_aux = ModelRegistry::Forward(*model, in_aux);
  REQUIRE(aux_routed.tensor.data != nullptr);
  CHECK(aux_routed.tensor.shape[1] == H * taps);
  CHECK(std::memcmp(aux_routed.tensor.data, aux.tensor.data,
                    static_cast<size_t>(T * H * taps) * sizeof(uint16_t)) == 0);
  CHECK(std::memcmp(routed_aux.device_tensor.data, routed_plain.device_tensor.data,
                    static_cast<size_t>(T * vocab) * sizeof(float)) == 0);
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
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
  gm.batch_ptr = conv.batch_ptr;
  gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;

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
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
  gm.batch_ptr = conv.batch_ptr;
  gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;

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

// PERF-27B-LMHEAD-FP4 (issue #213). The PAGED forward has TWO lm_head call sites
// — gathered (prefill/mixed) and the non-gathered full [T,vocab] arm — and a
// PACKED head leaves the bf16 owner EMPTY, so reverting either hands the logits
// GEMM an empty tensor. test_qwen27_dense_lmhead_fp4 pins the NUMERICS but runs
// only the EAGER forward; this pins that both PAGED arms SELECT the packed head.
namespace {
Nvfp4Weight MakePackedHead(int64_t n, int64_t k, uint64_t seed) {
  Nvfp4Weight w;
  w.n = n;
  w.k = k;
  w.scale2 = 0.125F;  // ModelOpt weight_scale_2 IS the scale
  w.packed = MakeOwned(DType::kI8, {n, k / 2}, seed);
  w.scale = MakeOwned(DType::kI8, {n, k / 16}, seed + 1);
  // fp4 operands are RAW bytes, and MakeOwned has no kI8 arm — its f32 branch
  // over-allocates 4 B/element — so size and fill them exactly here.
  w.packed.bytes.resize(static_cast<size_t>(n) * static_cast<size_t>(k / 2));
  w.scale.bytes.resize(static_cast<size_t>(n) * static_cast<size_t>(k / 16));
  auto* pb = w.packed.bytes.data();
  for (size_t i = 0; i < w.packed.bytes.size(); ++i)
    pb[i] = static_cast<uint8_t>((i * 37U + 11U) & 0x77U);
  const uint8_t kE4M3PowersOfTwo[4] = {0x34, 0x38, 0x3C, 0x40};  // .25 .5 1 2
  auto* sb = w.scale.bytes.data();
  for (size_t i = 0; i < w.scale.bytes.size(); ++i)
    sb[i] = kE4M3PowersOfTwo[i & 3U];
  return w;
}
}  // namespace

TEST_CASE("qwen27 dense paged: both lm_head arms run a PACKED NVFP4 head") {
  const HfConfig c = MakeConfig();
  const int64_t T = 5, V = c.vocab_size;
  Qwen3_5DenseWeights w = MakeWeights(c);
  w.lm_head_fp4 = MakePackedHead(V, c.hidden_size, 4242);

  const std::vector<int32_t> ids{3, 11, 7, 20, 5}, pos{0, 1, 2, 3, 4};
  vt::Queue q = Q();
  const std::vector<float> eager =
      Qwen3_5DenseModel::ForwardDense(ids, pos, w, c, q);
  REQUIRE(eager.size() == static_cast<size_t>(T * V));

  const CommonAttentionMetadata am = PrefillAttnMeta(T, {0, 1}, 8, 0);
  const GDNAttentionMetadata gm = PrefillGdnMeta(T, 0);

  // Non-gathered arm: empty logits_indices -> the full [T, vocab].
  {
    CachePool pool(c, 4, 8);
    const std::vector<float> full = Qwen3_5DenseModel::Forward(
        ids, pos, am, gm, pool.attn_kv, pool.gdn_state, w, c, q, {});
    REQUIRE(full.size() == eager.size());
    CHECK(MaxAbsDiff(full, eager, full.size()) < 1e-3);
  }
  // Gathered arm: the prefill shape the engine actually runs -> [1, vocab].
  {
    CachePool pool(c, 4, 8);
    const std::vector<float> got =
        Qwen3_5DenseModel::Forward(ids, pos, am, gm, pool.attn_kv,
                                   pool.gdn_state, w, c, q,
                                   {static_cast<int32_t>(T - 1)});
    REQUIRE(got.size() == static_cast<size_t>(V));
    const std::vector<float> tail(eager.end() - V, eager.end());
    CHECK(MaxAbsDiff(got, tail, got.size()) < 1e-3);
  }
}
