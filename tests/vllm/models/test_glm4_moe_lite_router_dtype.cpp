// GLM-4.7-Flash (`Glm4MoeLiteForCausalLM`) MoE ROUTER DTYPE gate — issue #2928,
// row MODEL-TEXT-GLM4-MOE-LITE-ROUTER-F32.
//
// Upstream `Glm4MoeLiteForCausalLM` does NOT use `DeepseekV2MoE`. Its MoE block
// is `Glm4MoeLite`, a bare subclass of `Glm4MoE` (glm4_moe_lite.py:86-87,
// instantiated at :161-165), whose gate is
// `nn.Linear(hidden_size, n_routed_experts, bias=False, dtype=torch.float32)`
// (glm4_moe.py:141-146) fed `hidden_states.to(dtype=torch.float32)` (:218), with
// `router_logits_dtype=torch.float32` (:205). No config key participates: the
// fp32 router is a property of the CLASS. Read at 5559679229, the pin the
// `tests/parity/goldens/glm4_moe_lite_greedy/` artifacts were captured on.
//
// `DeepseekV2MoE` resolves the same dtype from the config instead
// (deepseek_v2.py:308-314 through `_get_moe_router_dtype`, :123-133 — fp32 only
// for `model_type == "glm_moe_dsa"` or an explicit `moe_router_dtype:
// "float32"`). Our port composes the DeepSeek-V2 block for GLM, and the
// published `zai-org/GLM-4.7-Flash` config.json declares NO `moe_router_dtype`,
// so before this row our GLM router logits were rounded to bf16 in front of the
// top-k.
//
// WHY THAT IS NOT THE "merely too wide" HAZARD deepseek_v2.h:164-168 describes.
// GLM routes top-4 of 64 with `topk_method: noaux_tc` — sigmoid scores plus
// `e_score_correction_bias` decide the SELECTION — and `norm_topk_prob: true`.
// bf16 carries an 8-bit mantissa, so the rounding lands in front of a DISCRETE
// rank-4 boundary and the resulting error is bimodal, not a tolerance. The
// vehicle the block was gated on (DeepSeek-V2-Lite) is top-2 of 4, softmax,
// greedy, no bias, which is why `test_deepseek_v2_forward.cpp`'s tripwire case
// measures the two arms as bit-identical there. This file measures them at GLM's
// shape instead of arguing about it.
//
// Host-only: a CPU forward over a tiny synthetic model. No checkpoint, no GPU.
// The SACRED engine gate for this architecture is
// tests/vllm/models/test_glm4_moe_lite_paged_engine.cpp and needs a 58.2 GiB
// snapshot that is on neither this host nor the NAS.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deepseek_v2.h"
#include "vllm/model_executor/models/glm4_moe_lite.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace fs = std::filesystem;

namespace {

using vllm::DeepseekV2DenseMlp;
using vllm::DeepseekV2LayerWeights;
using vllm::DeepseekV2Model;
using vllm::DeepseekV2Params;
using vllm::DeepseekV2Weights;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;
using vt::DType;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

std::string ScratchDir() {
  const char* env = std::getenv("TMPDIR");
  const fs::path base = env != nullptr ? fs::path(env) : fs::temp_directory_path();
  const fs::path dir = base / "vllm_cpp_glm4_moe_lite_router_dtype_test";
  std::error_code ec;
  fs::create_directories(dir, ec);
  return dir.string();
}

// A tiny model carrying GLM-4.7-Flash's ROUTER shape verbatim — 64 routed
// experts, top-4, one shared expert, `topk_method: noaux_tc` with NO
// `scoring_func` key (GLM omits it and its class hardcodes sigmoid),
// `norm_topk_prob: true`, `routed_scaling_factor: 1.8`, `n_group`/`topk_group` 1,
// `first_k_dense_replace: 1`, `num_nextn_predict_layers: 1` — and NO
// `moe_router_dtype` key, exactly as the published config.json does not declare
// one. Everything else is shrunk so the CPU forward is fast; the router shape is
// what this file is about.
std::string WriteGlmTinyConfig(const std::string& name, int num_hidden_layers) {
  const std::string path = ScratchDir() + "/" + name + ".json";
  std::ofstream f(path);
  f << R"({
  "architectures": ["Glm4MoeLiteForCausalLM"],
  "model_type": "glm4_moe_lite",
  "hidden_size": 64,
  "num_hidden_layers": )"
    << num_hidden_layers << R"(,
  "num_attention_heads": 4,
  "num_key_value_heads": 4,
  "vocab_size": 100,
  "intermediate_size": 16,
  "moe_intermediate_size": 16,
  "n_routed_experts": 64,
  "num_experts_per_tok": 4,
  "n_group": 1,
  "topk_group": 1,
  "norm_topk_prob": true,
  "topk_method": "noaux_tc",
  "routed_scaling_factor": 1.8,
  "moe_layer_freq": 1,
  "n_shared_experts": 1,
  "first_k_dense_replace": 1,
  "num_nextn_predict_layers": 1,
  "q_lora_rank": 768,
  "qk_nope_head_dim": 16,
  "qk_rope_head_dim": 8,
  "v_head_dim": 16,
  "kv_lora_rank": 24,
  "rms_norm_eps": 1e-05,
  "rope_theta": 1000000,
  "max_position_embeddings": 128,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16"
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

OwnedTensor MakeF32(const std::vector<int64_t>& shape, uint32_t seed, float scale) {
  OwnedTensor o;
  o.dtype = DType::kF32;
  o.nk = false;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(float));
  auto* p = reinterpret_cast<float*>(o.bytes.data());
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (int64_t i = 0; i < numel; ++i) p[i] = dist(rng);
  return o;
}

DeepseekV2DenseMlp MakeMlp(int64_t H, int64_t I, uint32_t seed) {
  DeepseekV2DenseMlp m;
  m.gate_up_proj = MakeBf16({2 * I, H}, /*nk=*/true, seed);
  m.down_proj = MakeBf16({H, I}, /*nk=*/true, seed + 1);
  return m;
}

// GLM's q_lora branch (`q_lora_rank: 768`, shrunk here): `fused_qkv_a_proj` +
// `q_a_layernorm` + `q_b_proj`, the branch DeepSeek-V2-Lite never takes.
vllm::DeepseekV2MlaWeights MakeMla(const DeepseekV2Params& p, uint32_t seed) {
  const vllm::mla::MlaBlockDims& d = p.mla;
  const int64_t H = d.hidden_size, N = d.num_heads, L = d.kv_lora_rank;
  const int64_t P = d.qk_nope_head_dim, R = d.qk_rope_head_dim, V = d.v_head_dim;
  const int64_t Dqk = d.qk_head_dim();
  const int64_t QL = d.q_lora_rank;
  vllm::DeepseekV2MlaWeights w;
  w.fused_qkv_a_proj = MakeBf16({QL + L + R, H}, /*nk=*/true, seed);
  w.q_a_layernorm = MakeBf16({QL}, false, seed + 1, 0.5f);
  w.q_b_proj = MakeBf16({N * Dqk, QL}, /*nk=*/true, seed + 2);
  w.kv_a_layernorm = MakeBf16({L}, false, seed + 3, 0.5f);
  w.kv_b_proj = MakeBf16({N * (P + V), L}, /*nk=*/true, seed + 4);
  w.o_proj = MakeBf16({H, N * V}, /*nk=*/true, seed + 5);
  const vllm::mla::AbsorbedKvBProj a = vllm::mla::AbsorbKvBProjBf16(
      reinterpret_cast<const uint16_t*>(w.kv_b_proj.bytes.data()), d);
  w.w_uk_t = MakeBf16({N, P, L}, false, 1, 0.0f);
  std::memcpy(w.w_uk_t.bytes.data(), a.w_uk_t.data(),
              a.w_uk_t.size() * sizeof(uint16_t));
  w.w_uv = MakeBf16({N, L, V}, false, 1, 0.0f);
  std::memcpy(w.w_uv.bytes.data(), a.w_uv.data(), a.w_uv.size() * sizeof(uint16_t));
  return w;
}

DeepseekV2Weights GlmTinyWeights(const DeepseekV2Params& p, uint32_t seed0) {
  const int64_t H = p.hidden_size, V = p.vocab_size;
  const int64_t E = p.n_routed_experts, I = p.moe_intermediate_size;
  DeepseekV2Weights w;
  w.params = p;
  w.embed_tokens = MakeBf16({V, H}, false, seed0 + 1);
  w.final_norm = MakeBf16({H}, false, seed0 + 2, 0.5f);
  w.lm_head = MakeBf16({H, V}, false, seed0 + 3);
  {
    const int64_t rows = p.max_position_embeddings, rot = p.mla.qk_rope_head_dim;
    const std::vector<float> cache =
        vllm::mla::BuildDeepseekRopeCosSinCache(p.rope, rows);
    w.rope_cos_sin_cache = MakeBf16({rows, rot}, false, 1, 0.0f);
    auto* dst = reinterpret_cast<uint16_t*>(w.rope_cos_sin_cache.bytes.data());
    for (size_t i = 0; i < cache.size(); ++i) dst[i] = vt::F32ToBF16(cache[i]);
  }
  uint32_t seed = seed0 + 100;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    DeepseekV2LayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.attn = MakeMla(p, seed);
    seed += 10;
    lw.is_moe = p.is_moe_layer(l);
    if (lw.is_moe) {
      lw.moe.router_gate = MakeBf16({H, E}, false, seed++);
      // `topk_method: noaux_tc` — the f32 learned selection bias
      // (deepseek_v2.py:316-318). GLM always carries one.
      lw.moe.e_score_correction_bias = MakeF32({E}, seed++, 0.05f);
      for (int64_t e = 0; e < E; ++e) {
        lw.moe.expert_gate.push_back(MakeBf16({H, I}, false, seed));
        lw.moe.expert_up.push_back(MakeBf16({H, I}, false, seed + 1));
        lw.moe.expert_down.push_back(MakeBf16({I, H}, false, seed + 2));
        seed += 3;
      }
      lw.moe.shared = MakeMlp(H, p.shared_intermediate_size(), seed);
      seed += 2;
    } else {
      lw.dense = MakeMlp(H, p.intermediate_size, seed);
      seed += 2;
    }
    w.layers.push_back(std::move(lw));
  }
  return w;
}

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

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// T1 — the PARSE. GLM's router dtype is a class property upstream, so it cannot
// come from a config key the checkpoint does not ship.
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("glm4-moe-lite router: the GLM parse resolves f32 where the DeepSeek parse does not") {
  const HfConfig cfg = vllm::LoadHfConfig(WriteGlmTinyConfig("router_parse", 2));

  // `Glm4MoeLite` IS `Glm4MoE` (glm4_moe_lite.py:86-87), whose gate is fp32 with
  // no config key in the path (glm4_moe.py:141-146,205,218).
  const DeepseekV2Params glm = vllm::ParseGlm4MoeLiteParams(cfg);
  CHECK(glm.router_dtype_is_f32);

  // The SAME bytes through the DeepSeek-V2 parser resolve upstream's `None`,
  // because `_get_moe_router_dtype` (deepseek_v2.py:123-133) sees no
  // `moe_router_dtype` key and `model_type` is not `glm_moe_dsa`. That parser is
  // correct as it stands and this row does not change it.
  const DeepseekV2Params dsv2 = vllm::ParseDeepseekV2Params(cfg, /*allow_mtp_tail=*/true);
  CHECK_FALSE(dsv2.router_dtype_is_f32);

  // Everything else the two resolve must be identical: this row moves ONE field.
  CHECK(glm.n_routed_experts == dsv2.n_routed_experts);
  CHECK(glm.num_experts_per_tok == dsv2.num_experts_per_tok);
  CHECK(glm.norm_topk_prob == dsv2.norm_topk_prob);
  CHECK(glm.has_e_score_correction_bias == dsv2.has_e_score_correction_bias);
  CHECK(glm.routed_scaling_factor == doctest::Approx(dsv2.routed_scaling_factor));
  CHECK(glm.scoring_func == dsv2.scoring_func);

  // The GLM router shape this file exists for.
  CHECK(glm.n_routed_experts == 64);
  CHECK(glm.num_experts_per_tok == 4);
  CHECK(glm.norm_topk_prob);
  CHECK(glm.has_e_score_correction_bias);
  CHECK(glm.scoring_func == vt::MoeScoringFunc::kSigmoid);
  CHECK(glm.routed_scaling_factor == doctest::Approx(1.8f));
}

// ════════════════════════════════════════════════════════════════════════════
// T2 — the FORWARD. At GLM's router shape the dtype is OBSERVABLE, so the too-
// narrow store is not the invisible-width hazard; and the params GLM's own load
// path resolves select the arm upstream computes.
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("glm4-moe-lite router: bf16 logits CHANGE the forward at GLM's top-4-of-64 noaux_tc shape") {
  const HfConfig cfg = vllm::LoadHfConfig(WriteGlmTinyConfig("router_fwd", 4));
  const DeepseekV2Params glm = vllm::ParseGlm4MoeLiteParams(cfg);
  REQUIRE(glm.router_dtype_is_f32);
  REQUIRE(glm.is_moe_layer(1));
  REQUIRE_FALSE(glm.is_moe_layer(0));  // first_k_dense_replace: 1

  // Fixed seeds, so the count below is a property of the build and not of a
  // random draw. Each seed is a whole synthetic GLM: different attention,
  // different gate, different experts, different bias.
  const std::vector<uint32_t> seeds = {1,   17,  53,  101, 199, 271, 353, 431,
                                       509, 601, 691, 787, 877, 971, 1063, 1153};
  int seeds_differing = 0;
  double worst = 0.0;
  for (uint32_t s : seeds) {
    DeepseekV2Weights w = GlmTinyWeights(glm, s);
    REQUIRE(w.params.router_dtype_is_f32);
    const std::vector<float> f32 = RunTiny(w);
    REQUIRE(!f32.empty());
    for (float x : f32) REQUIRE(std::isfinite(x));

    // The arm this port took before #2928: the DeepSeek-V2 answer for a config
    // that ships no `moe_router_dtype`.
    w.params.router_dtype_is_f32 = false;
    const std::vector<float> bf16 = RunTiny(w);
    REQUIRE(bf16.size() == f32.size());
    for (float x : bf16) REQUIRE(std::isfinite(x));

    bool differs = false;
    for (size_t i = 0; i < f32.size(); ++i) {
      if (f32[i] != bf16[i]) differs = true;
      worst = std::max(worst, std::abs(static_cast<double>(f32[i]) -
                                       static_cast<double>(bf16[i])));
    }
    if (differs) ++seeds_differing;
  }

  MESSAGE("router dtype observable in the GLM forward: ", seeds_differing, "/",
          seeds.size(), " seeds differ, worst |f32 - bf16| logit = ", worst);
  // The claim is bimodal, not a tolerance: rounding the logits in front of a
  // top-4-of-64 `noaux_tc` selection changes WHICH experts run. One seed is
  // enough to falsify "a token gate cannot see this either way"; the count and
  // the margin are printed so a later reader sees how far from the edge it is.
  CHECK(seeds_differing >= 1);
  CHECK(worst > 0.0);
}

// ════════════════════════════════════════════════════════════════════════════
// T2b — determinism, so T2's comparison is a dtype statement and not a
// run-to-run one.
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("glm4-moe-lite router: each arm is deterministic run to run") {
  const HfConfig cfg = vllm::LoadHfConfig(WriteGlmTinyConfig("router_det", 4));
  const DeepseekV2Params glm = vllm::ParseGlm4MoeLiteParams(cfg);
  DeepseekV2Weights w = GlmTinyWeights(glm, 17);

  const std::vector<float> a = RunTiny(w);
  const std::vector<float> b = RunTiny(w);
  REQUIRE(a.size() == b.size());
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);

  w.params.router_dtype_is_f32 = false;
  const std::vector<float> c = RunTiny(w);
  const std::vector<float> d = RunTiny(w);
  REQUIRE(c.size() == d.size());
  CHECK(std::memcmp(c.data(), d.data(), c.size() * sizeof(float)) == 0);
}

// ════════════════════════════════════════════════════════════════════════════
// T3 — the MECHANISM, measured rather than argued. The error a too-narrow
// router store makes is a SELECTION error, so it is bimodal and not a
// tolerance: assert that the selected SET changes, and print the margin.
// ════════════════════════════════════════════════════════════════════════════
namespace {

vt::Device CpuDev() { return vt::Device{vt::DeviceType::kCPU, 0}; }

vt::Tensor Contig(void* data, DType dt, const std::vector<int64_t>& shape) {
  vt::Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = CpuDev();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

// One `vt::MoeRouterTopK` call at GLM-4.7-Flash's REAL router configuration.
std::vector<int32_t> RouteGlm(const std::vector<float>& logits, int64_t T, int64_t E,
                              std::vector<float>* bias) {
  vt::MoeRouterTopKArgs args{};
  args.top_k = 4;                                    // num_experts_per_tok
  args.renormalize = true;                           // norm_topk_prob
  args.scoring_func = vt::MoeScoringFunc::kSigmoid;  // noaux_tc hardcodes sigmoid
  args.num_expert_group = 1;                         // n_group
  args.topk_group = 1;                               // topk_group
  args.routed_scaling_factor = 1.8f;                 // routed_scaling_factor
  std::vector<float> weights(static_cast<size_t>(T * args.top_k), 0.0f);
  std::vector<int32_t> ids(static_cast<size_t>(T * args.top_k), -1);
  vt::Tensor tl = Contig(const_cast<float*>(logits.data()), DType::kF32, {T, E});
  vt::Tensor tw = Contig(weights.data(), DType::kF32, {T, args.top_k});
  vt::Tensor ti = Contig(ids.data(), DType::kI32, {T, args.top_k});
  vt::Tensor tb = Contig(bias->data(), DType::kF32, {E});
  vt::Queue q = Q();
  vt::MoeRouterTopK(q, tw, ti, tl, args, &tb);
  return ids;
}

}  // namespace

TEST_CASE("glm4-moe-lite router: the bf16 store CHANGES WHICH EXPERTS RUN") {
  // GLM-4.7-Flash's real router: 64 routed experts, top-4, `noaux_tc` sigmoid
  // with a learned selection bias, `norm_topk_prob`, `routed_scaling_factor`
  // 1.8, one expert group.
  const int64_t E = 64, K = 4, T = 4096;
  std::mt19937 rng(20260904);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> logits_f32(static_cast<size_t>(T * E));
  for (float& x : logits_f32) x = nd(rng);
  std::vector<float> bias(static_cast<size_t>(E));
  for (float& x : bias) x = 0.05f * nd(rng);

  // The ONLY difference: `deepseek_v2.cpp:363` stores the gate GEMM's f32 result
  // into a bf16 buffer before the top-k reads it.
  std::vector<float> logits_bf16 = logits_f32;
  for (float& x : logits_bf16) x = vt::BF16ToF32(vt::F32ToBF16(x));

  std::vector<float> bias_a = bias, bias_b = bias;
  const std::vector<int32_t> ids_f32 = RouteGlm(logits_f32, T, E, &bias_a);
  const std::vector<int32_t> ids_bf16 = RouteGlm(logits_bf16, T, E, &bias_b);
  REQUIRE(ids_f32.size() == ids_bf16.size());

  int tokens_with_a_different_SET = 0;
  for (int64_t t = 0; t < T; ++t) {
    std::vector<int32_t> a(ids_f32.begin() + static_cast<long>(t * K),
                           ids_f32.begin() + static_cast<long>((t + 1) * K));
    std::vector<int32_t> b(ids_bf16.begin() + static_cast<long>(t * K),
                           ids_bf16.begin() + static_cast<long>((t + 1) * K));
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) ++tokens_with_a_different_SET;
  }

  const double rate = static_cast<double>(tokens_with_a_different_SET) /
                      static_cast<double>(T);
  MESSAGE("bf16 router store changed the top-4 SET for ",
          tokens_with_a_different_SET, "/", T, " tokens (", 100.0 * rate, "%)");
  // A SET difference means a different expert's weights ran. It cannot be
  // absorbed by a tolerance, and it is what `deepseek_v2.h:164-168`'s "a token
  // gate cannot see this either way" does not cover: that sentence is about a
  // store that is too WIDE.
  CHECK(tokens_with_a_different_SET > 0);
}

// ════════════════════════════════════════════════════════════════════════════
// T4 — REACHABILITY. The two cases above prove the value; this one proves that
// the PRODUCTION load-and-forward path carries it. It goes in through
// `ModelRegistry::Load` and `ModelRegistry::Forward` over a synthetic
// GLM-4.7-Flash checkpoint on disk — no hand-built `DeepseekV2Weights`, no
// internal type — and compares the registry's own logits to the two arms.
//
// Delete `weights.params = ParseGlm4MoeLiteParams(config);` from
// `LoadGlm4MoeLiteForCausalLM` and this case reds while T1 and T2 stay green,
// which is exactly what the loader call site owes.
// ════════════════════════════════════════════════════════════════════════════
namespace {

struct StEntry {
  std::string name;
  std::string dtype;  // "BF16" or "F32"
  std::vector<int64_t> shape;
  uint32_t seed = 0;
  float scale = 0.08f;
};

void AppendEntry(std::vector<StEntry>* out, const std::string& name,
                 const std::vector<int64_t>& shape, uint32_t seed,
                 float scale = 0.08f, const char* dtype = "BF16") {
  out->push_back(StEntry{name, dtype, shape, seed, scale});
}

// Exactly the tensors `LoadDeepseekV2ForCausalLMWeights` resolves for this
// config, in checkpoint layout: the q_lora attention branch
// (`q_a_proj` + `kv_a_proj_with_mqa` merged into `fused_qkv_a_proj`,
// deepseek_v2_weights.cpp:160-173), a dense layer 0 (`first_k_dense_replace: 1`)
// and a `noaux_tc` MoE layer 1 with its f32 `e_score_correction_bias`, 64 routed
// experts and one shared expert.
std::vector<StEntry> GlmTinyCheckpointEntries(const DeepseekV2Params& p,
                                             uint32_t seed0) {
  const int64_t H = p.hidden_size, V = p.vocab_size;
  const int64_t N = p.mla.num_heads, L = p.mla.kv_lora_rank;
  const int64_t R = p.mla.qk_rope_head_dim, QL = p.mla.q_lora_rank;
  const int64_t Dqk = p.mla.qk_head_dim(), VD = p.mla.v_head_dim;
  const int64_t P = p.mla.qk_nope_head_dim;
  const int64_t E = p.n_routed_experts, I = p.moe_intermediate_size;
  std::vector<StEntry> e;
  uint32_t s = seed0;
  AppendEntry(&e, "model.embed_tokens.weight", {V, H}, s++);
  AppendEntry(&e, "model.norm.weight", {H}, s++, 0.5f);
  AppendEntry(&e, "lm_head.weight", {V, H}, s++);
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    AppendEntry(&e, b + "input_layernorm.weight", {H}, s++, 0.5f);
    AppendEntry(&e, b + "post_attention_layernorm.weight", {H}, s++, 0.5f);
    AppendEntry(&e, b + "self_attn.q_a_proj.weight", {QL, H}, s++);
    AppendEntry(&e, b + "self_attn.kv_a_proj_with_mqa.weight", {L + R, H}, s++);
    AppendEntry(&e, b + "self_attn.q_a_layernorm.weight", {QL}, s++, 0.5f);
    AppendEntry(&e, b + "self_attn.q_b_proj.weight", {N * Dqk, QL}, s++);
    AppendEntry(&e, b + "self_attn.kv_a_layernorm.weight", {L}, s++, 0.5f);
    AppendEntry(&e, b + "self_attn.kv_b_proj.weight", {N * (P + VD), L}, s++);
    AppendEntry(&e, b + "self_attn.o_proj.weight", {H, N * VD}, s++);
    if (p.is_moe_layer(l)) {
      AppendEntry(&e, b + "mlp.gate.weight", {E, H}, s++);
      AppendEntry(&e, b + "mlp.gate.e_score_correction_bias", {E}, s++, 0.05f,
                  "F32");
      for (int64_t x = 0; x < E; ++x) {
        const std::string ex = b + "mlp.experts." + std::to_string(x) + ".";
        AppendEntry(&e, ex + "gate_proj.weight", {I, H}, s++);
        AppendEntry(&e, ex + "up_proj.weight", {I, H}, s++);
        AppendEntry(&e, ex + "down_proj.weight", {H, I}, s++);
      }
      const int64_t SI = p.shared_intermediate_size();
      AppendEntry(&e, b + "mlp.shared_experts.gate_proj.weight", {SI, H}, s++);
      AppendEntry(&e, b + "mlp.shared_experts.up_proj.weight", {SI, H}, s++);
      AppendEntry(&e, b + "mlp.shared_experts.down_proj.weight", {H, SI}, s++);
    } else {
      AppendEntry(&e, b + "mlp.gate_proj.weight", {p.intermediate_size, H}, s++);
      AppendEntry(&e, b + "mlp.up_proj.weight", {p.intermediate_size, H}, s++);
      AppendEntry(&e, b + "mlp.down_proj.weight", {H, p.intermediate_size}, s++);
    }
  }
  return e;
}

void WriteGlmTinyCheckpoint(const std::vector<StEntry>& entries,
                            const std::string& path) {
  nlohmann::json header = nlohmann::json::object();
  size_t off = 0;
  for (const StEntry& e : entries) {
    size_t n = 1;
    for (int64_t d : e.shape) n *= static_cast<size_t>(d);
    const size_t w = e.dtype == "F32" ? 4u : 2u;
    header[e.name] = {{"dtype", e.dtype},
                      {"shape", e.shape},
                      {"data_offsets", {off, off + n * w}}};
    off += n * w;
  }
  const std::string hs = header.dump();
  std::ofstream out(path, std::ios::binary);
  const uint64_t hlen = hs.size();
  out.write(reinterpret_cast<const char*>(&hlen), 8);
  out.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  for (const StEntry& e : entries) {
    size_t n = 1;
    for (int64_t d : e.shape) n *= static_cast<size_t>(d);
    std::mt19937 rng(e.seed);
    std::uniform_real_distribution<float> dist(-e.scale, e.scale);
    for (size_t i = 0; i < n; ++i) {
      const float v = dist(rng);
      if (e.dtype == "F32") {
        out.write(reinterpret_cast<const char*>(&v), 4);
      } else {
        const uint16_t b = vt::F32ToBF16(v);
        out.write(reinterpret_cast<const char*>(&b), 2);
      }
    }
  }
}

}  // namespace

TEST_CASE("glm4-moe-lite router: the PRODUCTION load+forward carries the f32 router") {
  const std::string cfg_path = WriteGlmTinyConfig("router_reach", 4);
  const HfConfig cfg = vllm::LoadHfConfig(cfg_path);
  const DeepseekV2Params p = vllm::ParseGlm4MoeLiteParams(cfg);
  REQUIRE(p.is_moe_layer(1));
  REQUIRE(p.router_dtype_is_f32);

  // A fixture that cannot SEPARATE the two arms witnesses nothing, so the case
  // searches a fixed seed list for one that does and then uses it. Same
  // population T2 samples; here it also has to survive the safetensors round
  // trip and the real loader. Deterministic: the seeds are written down.
  const std::vector<uint32_t> seeds = {7, 61, 137, 233, 331, 431, 541, 653, 769, 883};
  const std::string ckpt = ScratchDir() + "/router_reach.safetensors";
  std::vector<float> want_f32, want_bf16;
  uint32_t witness = 0;
  for (uint32_t seed : seeds) {
    WriteGlmTinyCheckpoint(GlmTinyCheckpointEntries(p, seed), ckpt);
    std::vector<vllm::SafetensorsFile> probe;
    probe.push_back(vllm::SafetensorsFile::Open(ckpt));
    DeepseekV2Weights w = vllm::LoadDeepseekV2ForCausalLMWeights(
        probe, cfg, /*allow_mtp_tail=*/true);
    w.params.router_dtype_is_f32 = true;
    const std::vector<float> a = RunTiny(w);
    w.params.router_dtype_is_f32 = false;
    const std::vector<float> b = RunTiny(w);
    REQUIRE(a.size() == b.size());
    if (a != b) {
      want_f32 = a;
      want_bf16 = b;
      witness = seed;
      break;
    }
  }
  MESSAGE("separating checkpoint seed = ", witness);
  REQUIRE(witness != 0);
  REQUIRE(!want_f32.empty());
  REQUIRE(want_f32 != want_bf16);
  // `ckpt` on disk is the witness fixture: the loop wrote it last before break.

  // ── the production entry points ──────────────────────────────────────────
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ckpt));
  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
  std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(cfg, source);
  REQUIRE(model != nullptr);
  CHECK(model->registration().architecture == "Glm4MoeLiteForCausalLM");

  const int64_t T = 5, bs = 8;
  MlaCachePool pool(p, /*num_blocks=*/2, bs);
  const CommonAttentionMetadata am = PrefillMeta(T, bs);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  const std::vector<int32_t> logits_indices;
  std::vector<vllm::GdnStateCache> gdn_state;
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  vt::Queue q = Q();
  const vllm::ModelForwardInput input{tokens,      positions, am,  gdn_meta,
                                      pool.attn_kv, gdn_state, cfg, q,
                                      logits_indices};
  const vllm::ForwardLogits got = vllm::ModelRegistry::Forward(*model, input);
  // `gather_logits` defaults TRUE, so the registry's DEFAULT configuration takes
  // the device-resident arm (`ForwardGlm4MoeLiteForCausalLM` ->
  // `DeepseekV2Model::ForwardDevice`). That is the arm to measure; read it back.
  REQUIRE(got.rows * got.vocab == static_cast<int64_t>(want_f32.size()));
  std::vector<float> logits(want_f32.size(), 0.0f);
  if (got.on_device()) {
    vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
    cpu.Copy(q, logits.data(), got.device_tensor.data,
             logits.size() * sizeof(float));
    cpu.Synchronize(q);
  } else {
    REQUIRE(got.host.size() == want_f32.size());
    logits = got.host;
  }

  // The registry's own output IS the f32 arm, and is NOT the bf16 one.
  CHECK(std::memcmp(logits.data(), want_f32.data(),
                    want_f32.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(logits.data(), want_bf16.data(),
                    want_bf16.size() * sizeof(float)) != 0);
}
