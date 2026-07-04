// M1.8 Task 4 — the batched PAGED model runner (GPUModelRunner).
//
// Ported from vllm/v1/worker/gpu/model_runner.py @ e24d1b24 (the execute_model /
// sample_tokens split, KV-cache allocation from KVCacheConfig, the decode-first
// reorder). Drives the runner directly (no scheduler) over a small SYNTHETIC MoE
// model (CPU; the real 35B greedy through the runner on dgx is the milestone
// gate, dgx-pending). Cases:
//   1. KV allocation shape from a fake KVCacheConfig (full-attn buffer + GDN
//      ssm/conv state buffer dims correct, one per layer).
//   2. THE ORDERING IDENTITY GATE (mandatory de-risk): a batch of {1 decode,
//      1 prefill} admitted prefill-first — after the decode-first reorder,
//      logits_indices / the SamplingMetadata row (via the per-req seed) / the
//      attention seq_lens+block_table row / the GDN state index / the write-back
//      slot ALL resolve to the same request.
//   3. Single-request greedy decode over N steps: token sampled + appended, the
//      KV cache grows, the next step reads it (the decode continues).
//   4. A 2-request greedy batch step: each request samples from its OWN logits
//      row (matches the standalone dense argmax).
#include "vllm/v1/worker/gpu/runner.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::Qwen3_5Model;
using vllm::Qwen3_5MoeWeights;
using vllm::SamplingParams;
using vllm::v1::CachedRequestData;
using vllm::v1::FullAttentionSpec;
using vllm::v1::GPUModelRunner;
using vllm::v1::KVCacheConfig;
using vllm::v1::MambaSpec;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::NewRequestData;
using vllm::v1::SchedulerOutput;
using vt::DType;

namespace {

uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
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
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
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
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
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
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
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

constexpr int kBlockSize = 8;
constexpr int kMaxModelLen = 32;
constexpr int kNumBlocks = 8;

// A fake KVCacheConfig with the gate group structure: one full-attn group + one
// mamba (GDN) group, sharing kNumBlocks blocks.
KVCacheConfig MakeKvConfig(const HfConfig& c) {
  const int Hkv = static_cast<int>(c.num_key_value_heads);
  const int Dh = static_cast<int>(c.head_dim);
  const int Hv = static_cast<int>(c.linear_num_value_heads);
  const int Dv = static_cast<int>(c.linear_value_head_dim);
  const int Dk = static_cast<int>(c.linear_key_head_dim);
  const int Kw = static_cast<int>(c.linear_conv_kernel_dim);
  const int key_dim = static_cast<int>(c.linear_num_key_heads) * Dk;
  const int value_dim = Hv * Dv;
  const int conv_dim = 2 * key_dim + value_dim;

  KVCacheConfig kv;
  kv.num_blocks = kNumBlocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa3"},
      std::make_shared<FullAttentionSpec>(kBlockSize, Hkv, Dh, DType::kF32));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn0", "gdn1", "gdn2"},
      std::make_shared<MambaSpec>(
          kMaxModelLen,
          std::vector<std::vector<int64_t>>{{Hv, Dv, Dk}, {conv_dim, Kw - 1}},
          std::vector<DType>{DType::kF32, DType::kF32}));
  return kv;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

SamplingParams Greedy() {
  SamplingParams sp;
  sp.temperature = 0.0;  // greedy (argmax).
  sp.PostInit();
  return sp;
}

// A NewRequestData for the gate group structure (block_ids = {full-attn, gdn}).
NewRequestData MakeNewReq(const std::string& id, std::vector<int32_t> prompt,
                          std::vector<int32_t> output, int num_computed,
                          std::vector<int> fa_blocks, int gdn_block,
                          const SamplingParams& sp) {
  NewRequestData nr;
  nr.req_id = id;
  std::vector<int32_t> all = prompt;
  all.insert(all.end(), output.begin(), output.end());
  nr.prompt_token_ids = std::move(prompt);
  nr.sampling_params = sp;
  nr.block_ids = {std::move(fa_blocks), std::vector<int>{gdn_block}};
  nr.num_computed_tokens = num_computed;
  nr.prefill_token_ids = std::move(all);
  return nr;
}

SchedulerOutput NewStep(std::vector<NewRequestData> new_reqs,
                        std::map<std::string, int> scheduled) {
  SchedulerOutput so;
  so.scheduled_cached_reqs = CachedRequestData::make_empty();
  so.scheduled_new_reqs = std::move(new_reqs);
  int total = 0;
  for (const auto& [id, n] : scheduled) total += n;
  so.num_scheduled_tokens = std::move(scheduled);
  so.total_num_scheduled_tokens = total;
  return so;
}

// A decode step for a single already-admitted request.
SchedulerOutput DecodeStep(const std::string& id, int num_computed,
                           int num_output) {
  SchedulerOutput so;
  CachedRequestData cached;
  cached.req_ids = {id};
  cached.num_computed_tokens = {num_computed};
  cached.num_output_tokens = {num_output};
  cached.new_block_ids.emplace_back(std::nullopt);  // no new blocks this step
  so.scheduled_cached_reqs = std::move(cached);
  so.num_scheduled_tokens = {{id, 1}};
  so.total_num_scheduled_tokens = 1;
  return so;
}

int GreedyArgmax(const std::vector<float>& logits, int64_t row, int64_t vocab) {
  int best = 0;
  float bv = logits[static_cast<size_t>(row * vocab)];
  for (int64_t v = 1; v < vocab; ++v) {
    const float x = logits[static_cast<size_t>(row * vocab + v)];
    if (x > bv) {
      bv = x;
      best = static_cast<int>(v);
    }
  }
  return best;
}

}  // namespace

// ─── 1. KV allocation shape from a fake KVCacheConfig ────────────────────────
TEST_CASE("runner: KV allocation from KVCacheConfig (full-attn + GDN state)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);

  CHECK(runner.full_attn_group_id() == 0);
  CHECK(runner.gdn_group_id() == 1);
  CHECK(runner.num_blocks() == kNumBlocks);

  // One PagedKvCache per full-attn layer (config has exactly 1).
  REQUIRE(runner.attn_kv().size() == 1);
  const PagedKvCache& kv = runner.attn_kv()[0];
  CHECK(kv.num_blocks == kNumBlocks);
  CHECK(kv.block_size == kBlockSize);
  CHECK(kv.num_kv_heads == c.num_key_value_heads);
  CHECK(kv.head_size == c.head_dim);
  CHECK(kv.data != nullptr);

  // One GdnStateCache per GDN layer (config has exactly 3).
  REQUIRE(runner.gdn_state().size() == 3);
  const GdnStateCache& gs = runner.gdn_state()[0];
  // ssm_state [num_blocks, Hv, Dv, Dk].
  CHECK(gs.ssm_state.shape[0] == kNumBlocks);
  CHECK(gs.ssm_state.shape[1] == c.linear_num_value_heads);
  CHECK(gs.ssm_state.shape[2] == c.linear_value_head_dim);
  CHECK(gs.ssm_state.shape[3] == c.linear_key_head_dim);
  // conv_state [num_blocks, conv_dim, K-1].
  const int64_t key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
  const int64_t conv_dim =
      2 * key_dim + c.linear_num_value_heads * c.linear_value_head_dim;
  CHECK(gs.conv_state.shape[0] == kNumBlocks);
  CHECK(gs.conv_state.shape[1] == conv_dim);
  CHECK(gs.conv_state.shape[2] == c.linear_conv_kernel_dim - 1);
}

// ─── 2. THE ORDERING IDENTITY GATE (mandatory de-risk) ───────────────────────
// A batch of {1 decode "D", 1 prefill "P"} admitted PREFILL-FIRST. After the
// decode-first reorder the four seams must agree on ONE order (slot 0 == D,
// slot 1 == P): logits_indices, the SamplingMetadata row (via P's seed), the
// attention seq_lens+block_table row, the GDN state index, and the write-back.
TEST_CASE("runner: four-way ordering identity (mixed decode+prefill)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);

  // P: fresh prefill, 5 tokens (seq_len 5). Random with a distinctive seed +
  // top_k so its SamplingMetadata row is identifiable.
  SamplingParams p_params;
  p_params.temperature = 0.7;
  p_params.top_k = 2;
  p_params.seed = 12345;
  p_params.PostInit();
  NewRequestData p = MakeNewReq("P", {1, 2, 3, 4, 5}, {}, /*num_computed=*/0,
                                /*fa_blocks=*/{2, 3}, /*gdn_block=*/1, p_params);

  // D: decode, prompt 3 + 1 already-produced output token, num_computed 3
  // (seq_len 4). Greedy (no seed).
  NewRequestData d = MakeNewReq("D", {6, 7, 8}, {9}, /*num_computed=*/3,
                                /*fa_blocks=*/{0, 1}, /*gdn_block=*/0, Greedy());

  // Admit PREFILL-FIRST so the reorder must move the decode to the front.
  SchedulerOutput so = NewStep({p, d}, {{"P", 5}, {"D", 1}});

  auto out_opt = runner.execute_model(so);
  CHECK_FALSE(out_opt.has_value());  // MRV2 split: forward done, no output yet.

  // (a) The reorder placed the decode "D" at slot 0, prefill "P" at slot 1.
  const auto& ib = runner.input_batch();
  REQUIRE(ib.num_reqs() == 2);
  REQUIRE(ib.req_ids[0].has_value());
  REQUIRE(ib.req_ids[1].has_value());
  CHECK(*ib.req_ids[0] == "D");
  CHECK(*ib.req_ids[1] == "P");

  // (b) Attention seq_lens + block_table rows: slot 0 == D (seq_len 4, fa block
  // 0), slot 1 == P (seq_len 5, fa block 2).
  const auto& am = runner.last_attn_meta();
  REQUIRE(am.seq_lens.size() == 2);
  CHECK(am.seq_lens[0] == 4);  // D: computed 3 + scheduled 1
  CHECK(am.seq_lens[1] == 5);  // P: computed 0 + scheduled 5
  const int cols = am.block_table_num_cols;
  CHECK(am.block_table_tensor[0] == 0);                        // D fa block 0
  CHECK(am.block_table_tensor[static_cast<size_t>(cols)] == 2);  // P fa block 2

  // (c) logits_indices: D's single token at flat index 0; P's last of 5 tokens
  // at flat index 5 (query_start_loc [0,1,6]).
  const auto& step = runner.last_step();
  REQUIRE(step.logits_indices.size() == 2);
  CHECK(step.logits_indices[0] == 0);  // D last token
  CHECK(step.logits_indices[1] == 5);  // P last token

  // (d) GDN metadata: 1 decode + 1 prefill, decode-first. State indices in the
  // reordered order = [D's gdn block 0, P's gdn block 1]; the prefill sub-batch
  // is P (state 1, fresh -> has_initial_state 0).
  const auto& gm = runner.last_gdn_meta();
  CHECK(gm.num_decodes == 1);
  CHECK(gm.num_prefills == 1);
  REQUIRE(gm.non_spec_state_indices_tensor.has_value());
  CHECK(*gm.non_spec_state_indices_tensor == std::vector<int32_t>{0, 1});
  REQUIRE(gm.prefill_state_indices.has_value());
  CHECK(*gm.prefill_state_indices == std::vector<int32_t>{1});
  REQUIRE(gm.has_initial_state.has_value());
  CHECK(*gm.has_initial_state == std::vector<uint8_t>{1, 0});  // D continues, P fresh

  // (e) SamplingMetadata row alignment: P (seeded) is at slot 1, so its seed
  // surfaces at generators[1]; D (unseeded, greedy) is absent.
  const auto sm = ib.make_sampling_metadata();
  CHECK_FALSE(sm.all_greedy);  // P is random
  REQUIRE(sm.generators.count(1) == 1);
  CHECK(sm.generators.at(1) == 12345u);
  CHECK(sm.generators.count(0) == 0);

  // (f) Write-back slot: sample -> the sampled token lands in the SAME slot the
  // request occupies. D's row grows at slot 0, P's at slot 1.
  const int d_tokens_before = ib.num_tokens_no_spec[0];  // D: prompt3+output1 = 4
  const int p_tokens_before = ib.num_tokens_no_spec[1];  // P: prompt5 = 5
  CHECK(d_tokens_before == 4);
  CHECK(p_tokens_before == 5);

  ModelRunnerOutput mro = runner.sample_tokens(std::nullopt);

  // The output order + index map match the dense (reordered) order.
  REQUIRE(mro.req_ids.size() == 2);
  CHECK(mro.req_ids[0] == "D");
  CHECK(mro.req_ids[1] == "P");
  CHECK(mro.req_id_to_index.at("D") == 0);
  CHECK(mro.req_id_to_index.at("P") == 1);
  REQUIRE(mro.sampled_token_ids.size() == 2);
  REQUIRE(mro.sampled_token_ids[0].size() == 1);
  REQUIRE(mro.sampled_token_ids[1].size() == 1);

  // Write-back appended one token to each request's OWN row.
  const auto& ib2 = runner.input_batch();
  CHECK(ib2.num_tokens_no_spec[0] == d_tokens_before + 1);  // D grew
  CHECK(ib2.num_tokens_no_spec[1] == p_tokens_before + 1);  // P grew
  // The sampled token was written at the request's next free column.
  CHECK(ib2.token_id(0, d_tokens_before) == mro.sampled_token_ids[0][0]);
  CHECK(ib2.token_id(1, p_tokens_before) == mro.sampled_token_ids[1][0]);
}

// A 3-request mixed batch admitted [P0 prefill, P1 prefill, D decode]. The
// decode-first reorder must pull D to the front, moving ≥2 requests. Rather than
// hard-code the exact post-partition permutation (upstream does a MINIMUM-SWAP
// partition, not a stable sort — [P0,P1,D] -> swap(0,2) -> [D,P1,P0]), this asserts
// the order-INDEPENDENT invariant: whatever slot each request lands in, EVERY
// per-slot field (seq_len, fa block, GDN state index, seed, token count) still
// resolves to that SAME request. A field left behind during a swap_states chain
// would desync exactly one of these against the req_id at its slot.
TEST_CASE("runner: 3-request reorder keeps every per-slot field self-consistent") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);

  SamplingParams p0_params;
  p0_params.temperature = 0.7;
  p0_params.top_k = 2;
  p0_params.seed = 111;
  p0_params.PostInit();
  SamplingParams p1_params;
  p1_params.temperature = 0.7;
  p1_params.top_k = 2;
  p1_params.seed = 222;
  p1_params.PostInit();

  // Per-request expected fields, keyed by req_id (order-independent oracle).
  struct Expect {
    int seq_len;
    int fa_block;
    int gdn_state;
    unsigned seed;  // 0 = greedy / no generator
    int tokens;
  };
  const std::map<std::string, Expect> want = {
      {"D", {4, 0, 0, 0, 4}},     // decode: prompt3+out1, fa block 0, gdn 0, greedy
      {"P0", {5, 2, 1, 111, 5}},  // prefill 5, fa block 2, gdn 1, seed 111
      {"P1", {4, 4, 2, 222, 4}},  // prefill 4, fa block 4, gdn 2, seed 222
  };

  NewRequestData p0 = MakeNewReq("P0", {1, 2, 3, 4, 5}, {}, /*num_computed=*/0,
                                 /*fa_blocks=*/{2, 3}, /*gdn_block=*/1, p0_params);
  NewRequestData p1 = MakeNewReq("P1", {10, 11, 12, 13}, {}, /*num_computed=*/0,
                                 /*fa_blocks=*/{4, 5}, /*gdn_block=*/2, p1_params);
  NewRequestData d = MakeNewReq("D", {6, 7, 8}, {9}, /*num_computed=*/3,
                                /*fa_blocks=*/{0, 1}, /*gdn_block=*/0, Greedy());

  SchedulerOutput so = NewStep({p0, p1, d}, {{"P0", 5}, {"P1", 4}, {"D", 1}});
  auto out_opt = runner.execute_model(so);
  CHECK_FALSE(out_opt.has_value());

  const auto& ib = runner.input_batch();
  REQUIRE(ib.num_reqs() == 3);
  // Decode must lead after the reorder.
  CHECK(*ib.req_ids[0] == "D");

  const auto& am = runner.last_attn_meta();
  const auto& gm = runner.last_gdn_meta();
  const auto sm = ib.make_sampling_metadata();
  const int cols = am.block_table_num_cols;
  REQUIRE(gm.non_spec_state_indices_tensor.has_value());

  // For each occupied slot, cross-check ALL five per-slot fields against the
  // oracle for whichever request landed there.
  for (int i = 0; i < ib.num_reqs(); ++i) {
    REQUIRE(ib.req_ids[static_cast<size_t>(i)].has_value());
    const std::string rid = *ib.req_ids[static_cast<size_t>(i)];
    const Expect& e = want.at(rid);
    CHECK(am.seq_lens[static_cast<size_t>(i)] == e.seq_len);
    CHECK(am.block_table_tensor[static_cast<size_t>(i * cols)] == e.fa_block);
    CHECK((*gm.non_spec_state_indices_tensor)[static_cast<size_t>(i)] == e.gdn_state);
    CHECK(ib.num_tokens_no_spec[static_cast<size_t>(i)] == e.tokens);
    if (e.seed == 0) {
      CHECK(sm.generators.count(i) == 0);
    } else {
      REQUIRE(sm.generators.count(i) == 1);
      CHECK(sm.generators.at(i) == e.seed);
    }
  }
}

// ─── 3. Single-request greedy decode over N steps ────────────────────────────
TEST_CASE("runner: single-request greedy decode over N steps (KV grows, feedback)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);

  const std::vector<int32_t> prompt = {5, 9, 2, 31, 17};
  const int P = static_cast<int>(prompt.size());

  // Step 1: prefill.
  SchedulerOutput s1 =
      NewStep({MakeNewReq("A", prompt, {}, 0, {0, 1}, 0, Greedy())}, {{"A", P}});
  CHECK_FALSE(runner.execute_model(s1).has_value());
  ModelRunnerOutput m1 = runner.sample_tokens(std::nullopt);
  REQUIRE(m1.sampled_token_ids.size() == 1);
  REQUIRE(m1.sampled_token_ids[0].size() == 1);
  const int32_t tok1 = m1.sampled_token_ids[0][0];

  // The token was written back at column P (== the decode input next step).
  CHECK(runner.input_batch().num_tokens_no_spec[0] == P + 1);
  CHECK(runner.input_batch().token_id(0, P) == tok1);

  // The full-attn KV cache grew: the prefill wrote non-zero K/V into block 0.
  const PagedKvCache& kv = runner.attn_kv()[0];
  const auto* kvp = static_cast<const float*>(kv.data);
  bool kv_nonzero = false;
  for (int64_t i = 0; i < 2 * kBlockSize * c.num_key_value_heads * c.head_dim;
       ++i)
    if (kvp[i] != 0.0f) kv_nonzero = true;
  CHECK(kv_nonzero);

  // Steps 2..N: decode. Each reads the previous sampled token + the grown cache.
  int computed = P;
  int outputs = 1;
  int32_t prev = tok1;
  for (int stepn = 0; stepn < 4; ++stepn) {
    SchedulerOutput sd = DecodeStep("A", computed, outputs);
    CHECK_FALSE(runner.execute_model(sd).has_value());
    // prepare_inputs must have read the previously sampled token as the input.
    CHECK(runner.last_step().input_token_ids == std::vector<int32_t>{prev});
    CHECK(runner.last_step().positions == std::vector<int64_t>{computed});
    ModelRunnerOutput md = runner.sample_tokens(std::nullopt);
    REQUIRE(md.sampled_token_ids[0].size() == 1);
    prev = md.sampled_token_ids[0][0];
    computed += 1;
    outputs += 1;
    // The new token appended at the next column.
    CHECK(runner.input_batch().num_tokens_no_spec[0] == computed + 1);
    CHECK(runner.input_batch().token_id(0, computed) == prev);
  }
  CHECK(outputs == 5);  // 1 prefill sample + 4 decodes
}

// ─── 4. Two-request greedy batch step (per-request logits rows) ───────────────
TEST_CASE("runner: 2-request greedy batch samples each from its own logits row") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);

  const std::vector<int32_t> a_prompt = {5, 9, 2, 31};
  const std::vector<int32_t> b_prompt = {7, 1, 22};

  // Both fresh prefills, greedy; A at fa blocks {0,1}/gdn 0, B at {2,3}/gdn 1.
  SchedulerOutput so = NewStep(
      {MakeNewReq("A", a_prompt, {}, 0, {0, 1}, 0, Greedy()),
       MakeNewReq("B", b_prompt, {}, 0, {2, 3}, 1, Greedy())},
      {{"A", static_cast<int>(a_prompt.size())},
       {"B", static_cast<int>(b_prompt.size())}});

  CHECK_FALSE(runner.execute_model(so).has_value());
  ModelRunnerOutput mro = runner.sample_tokens(std::nullopt);

  REQUIRE(mro.req_ids.size() == 2);
  // Both prefills -> no reorder; dense order == admission order [A, B].
  CHECK(mro.req_ids[0] == "A");
  CHECK(mro.req_ids[1] == "B");
  REQUIRE(mro.sampled_token_ids[0].size() == 1);
  REQUIRE(mro.sampled_token_ids[1].size() == 1);

  // Each request's greedy token == the argmax of its OWN last-token logits from
  // a standalone dense forward (the paged batch row is per-request independent).
  vt::Queue q = Q();
  std::vector<int32_t> a_pos(a_prompt.size());
  for (size_t i = 0; i < a_pos.size(); ++i) a_pos[i] = static_cast<int32_t>(i);
  std::vector<int32_t> b_pos(b_prompt.size());
  for (size_t i = 0; i < b_pos.size(); ++i) b_pos[i] = static_cast<int32_t>(i);
  const std::vector<float> a_dense =
      Qwen3_5Model::ForwardDense(a_prompt, a_pos, w, c, q);
  const std::vector<float> b_dense =
      Qwen3_5Model::ForwardDense(b_prompt, b_pos, w, c, q);
  const int a_expect = GreedyArgmax(
      a_dense, static_cast<int64_t>(a_prompt.size()) - 1, c.vocab_size);
  const int b_expect = GreedyArgmax(
      b_dense, static_cast<int64_t>(b_prompt.size()) - 1, c.vocab_size);

  CHECK(mro.sampled_token_ids[0][0] == a_expect);
  CHECK(mro.sampled_token_ids[1][0] == b_expect);
}
