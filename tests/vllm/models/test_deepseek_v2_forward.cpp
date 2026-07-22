// vllm.cpp original (MLA campaign W7 forward doctest); no upstream mirror.
//
// The W7 forward + batch-ordering + shared-expert gates for DeepSeek-V2
// (`DeepseekV2ForCausalLM`, src/vllm/model_executor/models/deepseek_v2.cpp).
//
//   1. BATCH-ORDERING GATE. `BuildMlaBatchSplit` is the pure host function that
//      reproduces `split_decodes_and_prefills` (mla_attention.py:1640-1649) and
//      `prefill_tokens_with_context` (:1806-1810). Both orderings MLA depends on
//      are asserted, and both violations are asserted to THROW: W6's gate
//      measured 0.86 relative error from a batch whose with-context prefill was
//      not first, so a silent wrong answer is the failure mode being closed.
//   2. CPU SYNTHETIC FORWARD. A tiny random DeepSeek-V2 (dense layer 0 + MoE
//      layer 1 with 4 routed + 2 shared experts, q_lora_rank NULL — the V2-Lite
//      branch) run end to end on CPU: embed -> MLA block (A-projections, both
//      RMSNorms, decoupled YaRN RoPE, the MLA cache write, the prefill MHA) ->
//      MoE/dense MLP -> final norm -> untied lm_head. Asserts finite logits,
//      bit-exact determinism, and that the fusion-catalog ADOPT path is
//      byte-identical to the hand-call fallback.
//   3. SHARED-EXPERT COVERAGE (new for this model family — Qwen3-Coder has no
//      shared expert, and Qwen3.6's carries a sigmoid gate DeepSeek does not).
//      Two decisive checks:
//        (a) zeroing every ROUTED expert makes the MoE block's output exactly
//            the shared MLP's, so a MoE layer with zeroed routed experts must be
//            BIT-IDENTICAL to a DENSE layer holding the same MLP weights;
//        (b) turning the shared expert off changes the logits — i.e. the path is
//            EXERCISED, not merely compiled.
//   4. REAL-CHECKPOINT PREFILL ARGMAX (dgx-only) — the DeepSeek-V2-Lite analogue
//      of the Qwen3-Coder W3 sanity case. The token-exact bar is W8.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deepseek_v2.h"
#include "vllm/model_executor/models/device_pool.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace fs = std::filesystem;

namespace {

using vllm::BuildMlaBatchSplit;
using vllm::DeepseekV2DenseMlp;
using vllm::DeepseekV2LayerWeights;
using vllm::DeepseekV2Model;
using vllm::DeepseekV2Params;
using vllm::DeepseekV2Weights;
using vllm::HfConfig;
using vllm::MlaBatchSplit;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;
using vt::DType;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

std::string ScratchDir() {
  const char* env = std::getenv("TMPDIR");
  const fs::path base = env != nullptr ? fs::path(env) : fs::temp_directory_path();
  const fs::path dir = base / "vllm_cpp_deepseek_v2_test";
  std::error_code ec;
  fs::create_directories(dir, ec);
  return dir.string();
}

// A tiny DeepSeek-V2 config.json. `first_k_dense_replace` and
// `n_shared_experts` are parameters so the shared-expert equivalence gate can
// build the two models it compares. `intermediate_size` is deliberately equal to
// `moe_intermediate_size * n_shared_experts` (32) so a dense layer and a shared
// expert have the SAME shape.
// `real_mla_dims` switches the MLA head geometry to DeepSeek-V2-Lite's REAL one
// (qk_nope 128 + qk_rope 64 = QK 192, V 128, kv_lora 512). The CUDA MLA prefill
// launcher is instantiated for head_dim 192 ONLY (MLA campaign W5 — upstream's
// `requires_v_padding` path pads V 128 -> 192 and asks FA-2 for a SYMMETRIC
// 192), so any CUDA case must use these dims; the CPU reference kernels are
// dimension-generic and use the smaller ones so the CPU cases stay fast.
std::string WriteTinyConfig(const std::string& name, int first_k_dense_replace,
                            int n_shared_experts, bool real_mla_dims = false) {
  const std::string path = ScratchDir() + "/" + name + ".json";
  std::ofstream f(path);
  f << R"({
  "architectures": ["DeepseekV2ForCausalLM"],
  "model_type": "deepseek_v2",
  "hidden_size": 64,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 4,
  "vocab_size": 100,
  "intermediate_size": 32,
  "moe_intermediate_size": 16,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "n_group": 1,
  "topk_group": 1,
  "norm_topk_prob": false,
  "scoring_func": "softmax",
  "topk_method": "greedy",
  "routed_scaling_factor": 1.0,
  "moe_layer_freq": 1,
  "q_lora_rank": null,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000,
  "max_position_embeddings": 128,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "rope_scaling": {
    "type": "yarn",
    "factor": 4,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 0.707,
    "mscale_all_dim": 0.707,
    "original_max_position_embeddings": 32
  },
  "qk_nope_head_dim": )"
    << (real_mla_dims ? 128 : 16) << R"(,
  "qk_rope_head_dim": )" << (real_mla_dims ? 64 : 8) << R"(,
  "v_head_dim": )" << (real_mla_dims ? 128 : 16) << R"(,
  "kv_lora_rank": )" << (real_mla_dims ? 512 : 24) << R"(,
  "first_k_dense_replace": )"
    << first_k_dense_replace << R"(,
  "n_shared_experts": )" << n_shared_experts << R"(
})";
  f.close();
  return path;
}

OwnedTensor MakeBf16(const std::vector<int64_t>& shape, bool nk, uint32_t seed,
                     float scale = 0.08f) {
  OwnedTensor o;
  o.dtype = DType::kBF16;
  o.nk = nk;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (int64_t i = 0; i < numel; ++i) p[i] = vt::F32ToBF16(dist(rng));
  return o;
}

OwnedTensor ZeroBf16(const std::vector<int64_t>& shape, bool nk) {
  OwnedTensor o = MakeBf16(shape, nk, 1, 0.0f);
  std::fill(o.bytes.begin(), o.bytes.end(), static_cast<uint8_t>(0));
  return o;
}

DeepseekV2DenseMlp MakeMlp(int64_t H, int64_t I, uint32_t seed) {
  DeepseekV2DenseMlp m;
  m.gate_up_proj = MakeBf16({2 * I, H}, /*nk=*/true, seed);
  m.down_proj = MakeBf16({H, I}, /*nk=*/true, seed + 1);
  return m;
}

// The MLA attention weights for one layer, INCLUDING the load-time
// `kv_b_proj -> W_UK/W_UV` absorption split (mla::AbsorbKvBProjBf16) — the same
// transform the real loader performs, so the synthetic path exercises it too.
vllm::DeepseekV2MlaWeights MakeMla(const DeepseekV2Params& p, uint32_t seed) {
  const vllm::mla::MlaBlockDims& d = p.mla;
  const int64_t H = d.hidden_size, N = d.num_heads, L = d.kv_lora_rank;
  const int64_t P = d.qk_nope_head_dim, R = d.qk_rope_head_dim, V = d.v_head_dim;
  const int64_t Dqk = d.qk_head_dim();
  vllm::DeepseekV2MlaWeights w;
  // q_lora_rank NULL -> the direct q_proj branch (deepseek_v2.py:1028-1034).
  w.kv_a_proj_with_mqa = MakeBf16({L + R, H}, /*nk=*/true, seed);
  w.q_proj = MakeBf16({N * Dqk, H}, /*nk=*/true, seed + 1);
  w.kv_a_layernorm = MakeBf16({L}, false, seed + 2, 0.5f);
  w.kv_b_proj = MakeBf16({N * (P + V), L}, /*nk=*/true, seed + 3);
  w.o_proj = MakeBf16({H, N * V}, /*nk=*/true, seed + 4);
  const vllm::mla::AbsorbedKvBProj a = vllm::mla::AbsorbKvBProjBf16(
      reinterpret_cast<const uint16_t*>(w.kv_b_proj.bytes.data()), d);
  w.w_uk_t = MakeBf16({N, P, L}, false, 1, 0.0f);
  std::memcpy(w.w_uk_t.bytes.data(), a.w_uk_t.data(),
              a.w_uk_t.size() * sizeof(uint16_t));
  w.w_uv = MakeBf16({N, L, V}, false, 1, 0.0f);
  std::memcpy(w.w_uv.bytes.data(), a.w_uv.data(), a.w_uv.size() * sizeof(uint16_t));
  return w;
}

// `zero_routed`: every routed expert weight is exactly 0, so the routed
// contribution to the combine is exactly 0 and the block output IS the shared
// MLP's output. `shared` may be supplied so two models can share it byte-for-byte.
DeepseekV2Weights TinyWeights(const DeepseekV2Params& p, bool zero_routed = false,
                              const DeepseekV2DenseMlp* shared_override = nullptr) {
  const int64_t H = p.hidden_size, V = p.vocab_size;
  const int64_t E = p.n_routed_experts, I = p.moe_intermediate_size;
  DeepseekV2Weights w;
  w.params = p;
  w.embed_tokens = MakeBf16({V, H}, false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.5f);
  w.lm_head = MakeBf16({H, V}, false, 3);
  {
    const int64_t rows = p.max_position_embeddings, rot = p.mla.qk_rope_head_dim;
    const std::vector<float> cache =
        vllm::mla::BuildDeepseekRopeCosSinCache(p.rope, rows);
    w.rope_cos_sin_cache = MakeBf16({rows, rot}, false, 1, 0.0f);
    auto* dst = reinterpret_cast<uint16_t*>(w.rope_cos_sin_cache.bytes.data());
    for (size_t i = 0; i < cache.size(); ++i) dst[i] = vt::F32ToBF16(cache[i]);
  }
  uint32_t seed = 100;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    DeepseekV2LayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.attn = MakeMla(p, seed);
    seed += 10;
    lw.is_moe = p.is_moe_layer(l);
    if (lw.is_moe) {
      lw.moe.router_gate = MakeBf16({H, E}, false, seed++);
      for (int64_t e = 0; e < E; ++e) {
        lw.moe.expert_gate.push_back(zero_routed ? ZeroBf16({H, I}, false)
                                                 : MakeBf16({H, I}, false, seed));
        lw.moe.expert_up.push_back(zero_routed ? ZeroBf16({H, I}, false)
                                               : MakeBf16({H, I}, false, seed + 1));
        lw.moe.expert_down.push_back(zero_routed ? ZeroBf16({I, H}, false)
                                                 : MakeBf16({I, H}, false, seed + 2));
        seed += 3;
      }
      if (p.n_shared_experts > 0) {
        lw.moe.shared = shared_override != nullptr
                            ? *shared_override
                            : MakeMlp(H, p.shared_intermediate_size(), seed);
      }
      seed += 2;
    } else {
      lw.dense = shared_override != nullptr && l == 1
                     ? *shared_override
                     : MakeMlp(H, p.intermediate_size, seed);
      seed += 2;
    }
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// One MLA cache per layer: [num_blocks, block_size, kv_lora_rank + qk_rope],
// num_kv_heads == 1, NO separate V (MLAAttentionSpec).
struct MlaCachePool {
  std::vector<std::vector<uint16_t>> buf;
  std::vector<PagedKvCache> attn_kv;
  MlaCachePool(const DeepseekV2Params& p, int64_t num_blocks, int64_t block_size) {
    const int64_t head_size = p.mla.head_size();
    for (int64_t l = 0; l < p.num_hidden_layers; ++l)
      buf.emplace_back(static_cast<size_t>(num_blocks * block_size * head_size), 0);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kBF16;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = 1;
      kv.head_size = head_size;
      attn_kv.push_back(kv);
    }
  }
};

CommonAttentionMetadata PrefillMeta(int64_t T, int64_t block_size) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t % block_size);
  m.causal = true;
  return m;
}

std::vector<float> RunTiny(const DeepseekV2Weights& w) {
  const int64_t T = 5, bs = 8;
  MlaCachePool pool(w.params, /*num_blocks=*/2, bs);
  const CommonAttentionMetadata am = PrefillMeta(T, bs);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Q();
  return DeepseekV2Model::Forward(tokens, positions, am, pool.attn_kv, w, q);
}

bool HasCuda() {
  try {
    vt::GetBackend(vt::DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

// The CUDA sibling of MlaCachePool: device-resident, zero-initialised MLA caches
// (one per layer). Needed because the CUDA forward's `vt::ConcatAndCacheMla`
// writes into the cache on device.
struct CudaMlaCachePool {
  vt::Backend& b;
  std::vector<void*> owned;
  std::vector<PagedKvCache> attn_kv;
  CudaMlaCachePool(vt::Backend& backend, vt::Queue& q, const DeepseekV2Params& p,
                   int64_t num_blocks, int64_t block_size)
      : b(backend) {
    const int64_t head_size = p.mla.head_size();
    const size_t bytes =
        static_cast<size_t>(num_blocks * block_size * head_size) * sizeof(uint16_t);
    const std::vector<uint16_t> zeros(bytes / sizeof(uint16_t), 0);
    for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
      void* d = b.Alloc(bytes);
      b.Copy(q, d, zeros.data(), bytes);
      owned.push_back(d);
      PagedKvCache kv;
      kv.data = d;
      kv.dtype = DType::kBF16;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = 1;
      kv.head_size = head_size;
      attn_kv.push_back(kv);
    }
    b.Synchronize(q);
  }
  ~CudaMlaCachePool() {
    for (void* p : owned) b.Free(p);
  }
  CudaMlaCachePool(const CudaMlaCachePool&) = delete;
  CudaMlaCachePool& operator=(const CudaMlaCachePool&) = delete;
};

std::vector<float> RunTinyCuda(const DeepseekV2Weights& w) {
  const int64_t T = 5, bs = 8;
  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q{vt::Device{vt::DeviceType::kCUDA, 0}, nullptr};
  CudaMlaCachePool pool(b, q, w.params, /*num_blocks=*/2, bs);
  const CommonAttentionMetadata am = PrefillMeta(T, bs);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  return DeepseekV2Model::Forward(tokens, positions, am, pool.attn_kv, w, q);
}

// A batch descriptor for the ordering gate: (query_len, context_len) per request.
CommonAttentionMetadata BatchOf(
    const std::vector<std::pair<int32_t, int32_t>>& reqs) {
  CommonAttentionMetadata m;
  m.num_reqs = static_cast<int>(reqs.size());
  m.query_start_loc.push_back(0);
  for (const auto& r : reqs) {
    m.query_start_loc.push_back(m.query_start_loc.back() + r.first);
    m.seq_lens.push_back(r.first + r.second);
  }
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens_cpu = m.seq_lens;
  m.num_actual_tokens = m.query_start_loc.back();
  m.block_table_num_cols = 1;
  m.block_table_tensor.assign(reqs.size(), 0);
  m.causal = true;
  return m;
}

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// 1. BATCH-ORDERING GATE — the invariant W6 found the hard way.
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("MLA batch split: decodes first, with-context prefills leading the tail") {
  // 2 decodes (q_len 1), then a WITH-CONTEXT prefill, then a context-free one.
  // Upstream shape: split_decodes_and_prefills with decode_threshold 1
  // (mla_attention.py:1420, :1640-1649).
  const MlaBatchSplit s =
      BuildMlaBatchSplit(BatchOf({{1, 9}, {1, 4}, {6, 12}, {3, 0}}));
  CHECK(s.num_decodes == 2);
  CHECK(s.num_decode_tokens == 2);
  CHECK(s.decode_max_seq_len == 10);  // max(1+9, 1+4)
  CHECK(s.num_prefills == 2);
  CHECK(s.num_prefill_tokens == 9);
  CHECK(s.num_prefills_with_context == 1);
  CHECK(s.prefill_max_query_len == 6);
  // RELATIVE to the prefill sub-batch (mla_attention.py:1670-1675).
  REQUIRE(s.prefill_cu_seqlens_q.size() == 3);
  CHECK(s.prefill_cu_seqlens_q[0] == 0);
  CHECK(s.prefill_cu_seqlens_q[1] == 6);
  CHECK(s.prefill_cu_seqlens_q[2] == 9);
  REQUIRE(s.prefill_context_lens.size() == 2);
  CHECK(s.prefill_context_lens[0] == 12);
  CHECK(s.prefill_context_lens[1] == 0);
  // `prefill_tokens_with_context = prefill_query_start_loc[
  //  num_prefills_with_context]` (:1806-1810) — a PREFIX length, here 6.
  CHECK(s.prefill_cu_seqlens_q[static_cast<size_t>(s.num_prefills_with_context)] == 6);
}

TEST_CASE("MLA batch split: pure decode and pure prefill batches") {
  const MlaBatchSplit d = BuildMlaBatchSplit(BatchOf({{1, 3}, {1, 7}, {1, 1}}));
  CHECK(d.num_decodes == 3);
  CHECK(d.num_prefills == 0);
  CHECK(d.num_decode_tokens == 3);

  const MlaBatchSplit p = BuildMlaBatchSplit(BatchOf({{4, 0}, {2, 0}}));
  CHECK(p.num_decodes == 0);
  CHECK(p.num_prefills == 2);
  CHECK(p.num_prefills_with_context == 0);
  CHECK(p.num_prefill_tokens == 6);
}

TEST_CASE("MLA batch split REJECTS a decode that follows a prefill") {
  // Decode tokens are packed FIRST; the block slices q[:num_mqa_tokens]
  // (mla_attention.py:700-737), so an interleaved batch would silently attend
  // the wrong rows.
  CHECK_THROWS(BuildMlaBatchSplit(BatchOf({{4, 0}, {1, 3}})));
}

TEST_CASE("MLA batch split REJECTS a with-context prefill after a context-free one") {
  // THE W6 REGRESSION: this exact ordering produced 0.86 relative error because
  // `prefill_tokens_with_context` is a PREFIX length (:1806-1810).
  CHECK_THROWS(BuildMlaBatchSplit(BatchOf({{3, 0}, {5, 11}})));
  // ... and the same batch with the with-context request FIRST is accepted.
  const MlaBatchSplit ok = BuildMlaBatchSplit(BatchOf({{5, 11}, {3, 0}}));
  CHECK(ok.num_prefills_with_context == 1);
}

// ════════════════════════════════════════════════════════════════════════════
// 2. CPU SYNTHETIC FORWARD
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("deepseek-v2 forward: CPU synthetic runs, finite, deterministic") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig cfg = vllm::LoadHfConfig(WriteTinyConfig("tiny", 1, 2));
  const DeepseekV2Params p = vllm::ParseDeepseekV2Params(cfg);
  // The V2-Lite-shaped branch: q_lora_rank NULL -> the DIRECT q_proj path.
  CHECK_FALSE(p.mla.has_q_lora());
  CHECK(p.mla.head_size() == 32);      // kv_lora_rank 24 + qk_rope 8
  CHECK(p.mla.qk_head_dim() == 24);    // qk_nope 16 + qk_rope 8
  CHECK(p.is_moe_layer(0) == false);   // first_k_dense_replace == 1
  CHECK(p.is_moe_layer(1) == true);
  CHECK(p.shared_intermediate_size() == 32);

  const DeepseekV2Weights w = TinyWeights(p);
  const std::vector<float> a = RunTiny(w);
  REQUIRE(a.size() == static_cast<size_t>(5) * static_cast<size_t>(p.vocab_size));
  for (float x : a) REQUIRE(std::isfinite(x));

  const std::vector<float> b = RunTiny(w);
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

TEST_CASE("deepseek-v2 forward: fusion-catalog ADOPT == hand-call fallback") {
  const HfConfig cfg = vllm::LoadHfConfig(WriteTinyConfig("tiny", 1, 2));
  const DeepseekV2Params p = vllm::ParseDeepseekV2Params(cfg);
  const DeepseekV2Weights w = TinyWeights(p);

  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const std::vector<float> adopt = RunTiny(w);
  setenv("VT_FUSED_CHAIN_ADOPT", "0", 1);
  const std::vector<float> hand = RunTiny(w);
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);

  REQUIRE(adopt.size() == hand.size());
  CHECK(std::memcmp(adopt.data(), hand.data(), adopt.size() * sizeof(float)) == 0);
}

// ════════════════════════════════════════════════════════════════════════════
// 3. SHARED-EXPERT COVERAGE (new for this family)
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("deepseek-v2 MoE: zeroed routed experts == a DENSE layer with the shared MLP") {
  // DeepSeek's shared expert is a PLAIN MLP added to the routed sum
  // (deepseek_v2.py:344-357; moe_runner.py:407) — NO sigmoid gate. With every
  // routed expert weight exactly 0 the routed term is exactly 0, so the MoE
  // block's output IS the shared MLP's, and a MoE layer must be BIT-IDENTICAL to
  // a dense layer holding the same MLP weights (the tiny config sets
  // intermediate_size == moe_intermediate_size * n_shared_experts == 32 so the
  // two shapes match).
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig moe_cfg = vllm::LoadHfConfig(WriteTinyConfig("tiny_moe", 1, 2));
  const HfConfig dense_cfg = vllm::LoadHfConfig(WriteTinyConfig("tiny_dense", 2, 2));
  const DeepseekV2Params pm = vllm::ParseDeepseekV2Params(moe_cfg);
  const DeepseekV2Params pd = vllm::ParseDeepseekV2Params(dense_cfg);
  REQUIRE(pm.is_moe_layer(1));
  REQUIRE_FALSE(pd.is_moe_layer(1));

  const DeepseekV2DenseMlp mlp = MakeMlp(pm.hidden_size, 32, /*seed=*/9001);
  const DeepseekV2Weights wm = TinyWeights(pm, /*zero_routed=*/true, &mlp);
  const DeepseekV2Weights wd = TinyWeights(pd, /*zero_routed=*/true, &mlp);
  REQUIRE(wm.layers[1].is_moe);
  REQUIRE_FALSE(wd.layers[1].is_moe);
  REQUIRE_FALSE(wm.layers[1].moe.shared.Empty());  // the shared path is populated

  const std::vector<float> a = RunTiny(wm);
  const std::vector<float> b = RunTiny(wd);
  REQUIRE(a.size() == b.size());
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

TEST_CASE("deepseek-v2 MoE: the shared expert is EXERCISED (off changes the logits)") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig with_cfg = vllm::LoadHfConfig(WriteTinyConfig("tiny_sh2", 1, 2));
  const HfConfig without_cfg = vllm::LoadHfConfig(WriteTinyConfig("tiny_sh0", 1, 0));
  const DeepseekV2Params pw = vllm::ParseDeepseekV2Params(with_cfg);
  const DeepseekV2Params pn = vllm::ParseDeepseekV2Params(without_cfg);
  CHECK(pw.n_shared_experts == 2);
  CHECK(pn.n_shared_experts == 0);

  // Identical seeds everywhere EXCEPT that the second model has no shared MLP,
  // so any difference is the shared-expert contribution alone.
  const DeepseekV2Weights ww = TinyWeights(pw);
  const DeepseekV2Weights wn = TinyWeights(pn);
  REQUIRE_FALSE(ww.layers[1].moe.shared.Empty());
  REQUIRE(wn.layers[1].moe.shared.Empty());

  const std::vector<float> a = RunTiny(ww);
  const std::vector<float> b = RunTiny(wn);
  REQUIRE(a.size() == b.size());
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    worst = std::max(worst, std::abs(static_cast<double>(a[i] - b[i])));
  CHECK_MESSAGE(worst > 1e-4,
                "the shared-expert path did not affect the logits -> NOT exercised");
}

TEST_CASE("deepseek-v2 forward: CPU at the REAL V2-Lite MLA head geometry") {
  // The cases above run a miniature head geometry so they stay fast; this one
  // uses DeepSeek-V2-Lite's REAL MLA dims (QK 192 = 128 nope + 64 rope, V 128,
  // 576-wide latent) with a tiny hidden/vocab/expert count, so the head-dim
  // constants the kernels branch on are the production ones even in CI.
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig cfg =
      vllm::LoadHfConfig(WriteTinyConfig("tiny_real", 1, 2, /*real_mla_dims=*/true));
  const DeepseekV2Params p = vllm::ParseDeepseekV2Params(cfg);
  REQUIRE(p.mla.qk_head_dim() == 192);
  REQUIRE(p.mla.head_size() == 576);
  const DeepseekV2Weights w = TinyWeights(p);
  const std::vector<float> a = RunTiny(w);
  REQUIRE(a.size() == static_cast<size_t>(5) * static_cast<size_t>(p.vocab_size));
  for (float x : a) REQUIRE(std::isfinite(x));
  const std::vector<float> b = RunTiny(w);
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// ════════════════════════════════════════════════════════════════════════════
// 3b. THE CUDA PATH IS EXERCISED, NOT MERELY COMPILED
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("deepseek-v2 forward: CUDA agrees with CPU and is bit-exact run to run") {
  if (!HasCuda()) {
    MESSAGE("SKIP: no CUDA device (the CUDA forward path is dgx-only)");
    return;
  }
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig cfg =
      vllm::LoadHfConfig(WriteTinyConfig("tiny_cuda", 1, 2, /*real_mla_dims=*/true));
  const DeepseekV2Params p = vllm::ParseDeepseekV2Params(cfg);
  const DeepseekV2Weights w = TinyWeights(p);

  // This run takes the paths a CPU run CANNOT: the CUDA MLA decode/prefill
  // kernels, `vt::ConcatAndCacheMla` on device, `vt::BatchedMatmul`, and — the
  // piece unique to this TU — the GROUPED bf16 MoE GEMM branch
  // (`vt::MoeGroupedGemmBf16` x3 + `vt::MoeSiluMul`), which is CUDA-only and
  // falls back to the per-expert reference loop everywhere else.
  REQUIRE(p.mla.qk_head_dim() == 192);  // the ONLY head_dim the CUDA MLA prefill has
  REQUIRE(p.mla.head_size() == 576);

  // PER-BACKEND SCRATCH POOLS, deliberately. The shared `DevicePool`
  // (device_pool.h) is a process-wide singleton keyed ONLY on a byte size class
  // and is documented "backend-agnostic"; that is safe for the engine, which
  // drives exactly one device per process, but NOT for a test binary that runs a
  // CPU forward and a CUDA forward in the same process — the second backend
  // would be handed the first backend's recycled pointers (observed: SIGSEGV in
  // the CPU arm on a CUDA block). Giving each arm its own pool is a TEST-LOCAL
  // fix; the hazard itself is pre-existing and unrelated to MLA, recorded in the
  // W7 ledger row rather than papered over here.
  static vllm::DevicePool cuda_pool;
  static vllm::DevicePool cpu_pool;
  std::vector<float> cuda, again, cpu;
  {
    const vllm::ActivePoolScope scope(&cuda_pool);
    cuda = RunTinyCuda(w);
    // Bit-exact run to run on device (nothing in the chain is non-deterministic).
    again = RunTinyCuda(w);
  }
  REQUIRE(cuda.size() == static_cast<size_t>(5) * static_cast<size_t>(p.vocab_size));
  for (float x : cuda) REQUIRE(std::isfinite(x));
  CHECK(std::memcmp(cuda.data(), again.data(), cuda.size() * sizeof(float)) == 0);

  // ... and it computes the same function as the CPU reference path. The band is
  // bf16-GEMM-accumulation-order wide (the CPU arm runs the per-expert reference
  // loop and cuBLASLt/grouped kernels reduce in a different order), so this is a
  // NUMERIC agreement check, not a bit check.
  {
    const vllm::ActivePoolScope scope(&cpu_pool);
    cpu = RunTiny(w);
  }
  REQUIRE(cpu.size() == cuda.size());
  double scale = 1e-6, worst = 0.0;
  for (float x : cpu) scale = std::max(scale, std::abs(static_cast<double>(x)));
  for (size_t i = 0; i < cpu.size(); ++i)
    worst = std::max(worst, std::abs(static_cast<double>(cuda[i] - cpu[i])) / scale);
  MESSAGE("deepseek-v2 CUDA-vs-CPU worst relative logit error = " << worst);
  CHECK(worst < 5e-2);
}

// ════════════════════════════════════════════════════════════════════════════
// 4. REAL-CHECKPOINT PREFILL ARGMAX SANITY (dgx-only)
// ════════════════════════════════════════════════════════════════════════════
namespace {
std::string FindV2LiteSnapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec))
    if (fs::exists(e.path() / "model.safetensors.index.json", ec))
      return e.path().string();
  return "";
}

std::vector<vllm::SafetensorsFile> OpenShards(const std::string& snap) {
  const std::map<std::string, std::string> wmap =
      vllm::LoadSafetensorsIndex(snap + "/model.safetensors.index.json");
  std::unordered_set<std::string> files;
  for (const auto& [tensor, file] : wmap) files.insert(file);
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(files.size());
  for (const std::string& f : files)
    shards.push_back(vllm::SafetensorsFile::Open(snap + "/" + f));
  return shards;
}
}  // namespace

// DIAGNOSTIC (dgx-only): load the REAL DeepSeek-V2-Lite and run a single-sequence
// CPU prefill; assert the logits are finite and deterministic and REPORT the
// last-token argmax + top5. This isolates the MLA forward from the paged engine
// and the sampler; the token-exact assertion belongs to W8.
TEST_CASE("deepseek-v2 forward: real DeepSeek-V2-Lite prefill argmax sanity (dgx-only)") {
  const std::string snap = FindV2LiteSnapshot();
  if (snap.empty()) {
    MESSAGE("SKIP: DeepSeek-V2-Lite checkpoint absent (dgx-only forward sanity)");
    return;
  }
  const HfConfig cfg = vllm::LoadHfConfig(snap + "/config.json");
  const std::vector<vllm::SafetensorsFile> shards = OpenShards(snap);
  const DeepseekV2Weights w =
      vllm::LoadDeepseekV2ForCausalLMWeights(shards, cfg);
  const DeepseekV2Params& p = w.params;

  // The prompt token ids are supplied by the harness so this file carries no
  // tokenizer dependency; VT_DEEPSEEK_PROMPT_IDS is a comma-separated list.
  std::vector<int32_t> tokens;
  if (const char* env = std::getenv("VT_DEEPSEEK_PROMPT_IDS"); env != nullptr) {
    std::string s(env), cur;
    for (char c : s + ",") {
      if (c == ',') {
        if (!cur.empty()) tokens.push_back(std::stoi(cur));
        cur.clear();
      } else if (c != ' ') {
        cur.push_back(c);
      }
    }
  }
  if (tokens.empty()) {
    MESSAGE("SKIP: set VT_DEEPSEEK_PROMPT_IDS=<comma-separated prompt ids>");
    return;
  }
  const int64_t T = static_cast<int64_t>(tokens.size());
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  const int64_t bs = 64, head_size = p.mla.head_size();
  const int64_t nb = (T + bs - 1) / bs;
  std::vector<std::vector<uint16_t>> buf;
  std::vector<PagedKvCache> attn_kv;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l)
    buf.emplace_back(static_cast<size_t>(nb * bs * head_size), 0);
  for (auto& b : buf) {
    PagedKvCache kv;
    kv.data = b.data();
    kv.dtype = DType::kBF16;
    kv.num_blocks = nb;
    kv.block_size = bs;
    kv.num_kv_heads = 1;
    kv.head_size = head_size;
    attn_kv.push_back(kv);
  }
  CommonAttentionMetadata m = PrefillMeta(T, bs);
  m.block_table_num_cols = static_cast<int>(nb);
  m.block_table_tensor.assign(static_cast<size_t>(nb), 0);
  for (int64_t i = 0; i < nb; ++i)
    m.block_table_tensor[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  m.slot_mapping.clear();
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t);

  vt::Queue q = Q();
  const std::vector<float> logits =
      DeepseekV2Model::Forward(tokens, positions, m, attn_kv, w, q);
  REQUIRE(logits.size() == static_cast<size_t>(T) * p.vocab_size);
  for (float x : logits) REQUIRE(std::isfinite(x));

  const int64_t V = p.vocab_size;
  const float* last = logits.data() + (T - 1) * V;
  int argmax = 0;
  for (int64_t v = 1; v < V; ++v)
    if (last[v] > last[argmax]) argmax = static_cast<int>(v);
  std::vector<int> idx(static_cast<size_t>(V));
  for (int64_t v = 0; v < V; ++v) idx[static_cast<size_t>(v)] = static_cast<int>(v);
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                    [&](int a, int b) { return last[a] > last[b]; });
  MESSAGE("deepseek-v2 prefill argmax=" << argmax << " top5=[" << idx[0] << ","
          << idx[1] << "," << idx[2] << "," << idx[3] << "," << idx[4] << "]");

  const std::vector<float> again =
      DeepseekV2Model::Forward(tokens, positions, m, attn_kv, w, q);
  CHECK(std::memcmp(logits.data(), again.data(), logits.size() * sizeof(float)) == 0);
}
