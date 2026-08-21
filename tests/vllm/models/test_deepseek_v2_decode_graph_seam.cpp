// THE G2 REACHABILITY GATE for ENG-CUDAGRAPH-BREAK W3 (#1291, parent #1163),
// for `DeepseekV2DecodeGraph` — per `.agents/reachability.md` and the AGENTS.md
// "Nothing lands dead" rule.
//
// WHAT IT MEASURES, and why the sibling gates do not cover it. W2 (#1261)
// migrated `Qwen3DenseDecodeGraph`; this stage migrates the three remaining
// plain batched drivers, and each owes its own gate because a driver that kept
// its hand-rolled `BeginCapture`/`EndCaptureGraph`/`ReplayGraph`/`DestroyGraph`
// sequence produces IDENTICAL logits, an IDENTICAL backend log and an identical
// `replay_count()`. `vt::GraphBreakStats::segments_captured` moves only when a
// `vt::GraphCaptureScope` closes a segment and `replays` only inside
// `vt::BreakableGraph::Replay`, so those two are the only observables that
// separate "this driver captured a graph" from "this driver captured a graph
// THROUGH THE SEAM".
//
// THE MUTATION THIS FILE ANSWERS TO: in a scratch copy, replace this driver's
// `vt::GraphCaptureScope` and `BreakableGraph::Replay` with the raw backend pair
// they replaced. Every other gate stays green and this file goes RED.
//
// DEEPSEEK'S SHAPE, stated because it is the one W3 driver that is not dense
// attention. Its capturable region is MLA decode (`vt::MlaDecodeAttention`), so
// its KV cache is one latent cache per layer with `num_kv_heads == 1` and no
// separate V, and its replay branch re-records the W8 MLA split-shape counters
// from the padded metadata because `BuildMlaStep` does not run on a replay. The
// capture machinery it re-derived is nevertheless the same one, which is the
// point: this gate holds it to the shared seam like its siblings.
//
// kFULL, AND THE BREAK COUNT IS ZERO ON PURPOSE. vLLM's v1 default
// `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:63` @ pin `5559679229`) is a
// FULL graph for DECODE batches (`:630-632`, `decode_mode()` `:65-66`), so
// `breaks_registered` must stay 0: a migration that opened its scope
// `kPiecewise` would split this step into one eager attention call per layer
// between graph replays, which is not vLLM's decode behaviour.
//
// The harness and what it cannot see are stated once, in
// `decode_graph_seam_harness.h`. In one line: a CPU "replay" recomputes nothing,
// so this file gates the ROUTING and the capture step's numerics and NOT that a
// replayed segment reproduces the eager forward. That is G1; the spec's
// `## Owed` records it.
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
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "decode_graph_seam_harness.h"
#include "vllm/model_executor/models/deepseek_v2.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace fs = std::filesystem;

namespace {

using vllm::DeepseekV2DenseMlp;
using vllm::DeepseekV2LayerWeights;
using vllm::DeepseekV2Params;
using vllm::DeepseekV2Weights;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;
using vllm_test::StaticGraphCpu;
using vt::DType;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

std::string ScratchDir() {
  const char* env = std::getenv("TMPDIR");
  const fs::path base = env != nullptr ? fs::path(env) : fs::temp_directory_path();
  const fs::path dir = base / "vllm_cpp_deepseek_v2_seam_test";
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


// One single-request pure-decode step at `pos`.
CommonAttentionMetadata DecodeMeta(int32_t pos) {
  CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = 1;
  am.query_start_loc = {0, 1};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {pos + 1};
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = 1;
  am.max_seq_len = pos + 1;
  am.block_table_num_cols = 1;
  am.block_table_tensor = {0};
  am.slot_mapping = {pos};
  am.causal = true;
  return am;
}

DeepseekV2Params TinyParams() {
  return vllm::ParseDeepseekV2Params(
      vllm::LoadHfConfig(WriteTinyConfig("seam_tiny", 1, 2)));
}

}  // namespace

TEST_CASE("G2: DeepseekV2DecodeGraph captures and replays THROUGH the vt seam") {
  const DeepseekV2Params p = TinyParams();
  const DeepseekV2Weights w = TinyWeights(p);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  MlaCachePool pool(p, /*num_blocks=*/2, /*block_size=*/8);

  vt::ResetGraphBreakStats();
  vllm::DeepseekV2DecodeGraph graph(w, q, /*max_num_reqs=*/8);

  // Step 1, COLD: an eager pre-warm step. Nothing is captured yet.
  graph.Step({11}, {0}, DecodeMeta(0), pool.attn_kv);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 0);
    CHECK(s.replays == 0);
    CHECK_FALSE(graph.captured());
  }

  // Step 2, WARM: the driver captures. This is the assertion that fails when the
  // driver keeps its own BeginCapture/EndCaptureGraph pair.
  graph.Step({12}, {1}, DecodeMeta(1), pool.attn_kv);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 1);
    CHECK(graph.captured());
    // kFULL, ASSERTED ON THE MODE ITSELF and not inferred from a side effect.
    // `breaks_registered == 0` is TRUE IN BOTH MODES for this model, because
    // nothing in its forward registers a `vt::GraphBreak` — the one production
    // break point in the tree is W1's, in `qwen3.cpp`. Measured, not reasoned:
    // flipping this driver's `kFull` to `kPiecewise` compiled clean and left
    // this whole file green. `full_scopes` / `piecewise_scopes` (W3, #1291) are
    // the observable that actually moves, so that flip now reds this case.
    CHECK(s.full_scopes == 1);
    CHECK(s.piecewise_scopes == 0);
    CHECK(s.breaks_registered == 0);
  }

  // Step 3, CAPTURED: the driver replays. `replays` moves ONLY inside
  // vt::BreakableGraph::Replay.
  graph.Step({13}, {2}, DecodeMeta(2), pool.attn_kv);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.replays >= 1);
    CHECK(graph.replay_count() >= 1);
  }

  // Step 4, a SECOND replay: the REPLAY branch is re-entrant. A branch that
  // re-captured, reset the container or left `warm` set would still look correct
  // after one step and diverge only on the next.
  graph.Step({14}, {3}, DecodeMeta(3), pool.attn_kv);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.replays >= 2);
    CHECK(s.segments_captured == 1);  // a replay must never re-enter capture
  }

  CHECK(harness.backend().Count("Begin") == 1);
  CHECK(harness.backend().Count("EndCaptureGraph") == 1);
  CHECK(harness.backend().Count("ReplayGraph") >= 2);
}

// G4 for this driver: the capture step's logits are bit-identical to the SAME
// step taken through the driver's own EAGER arm — G4's own words, "with capture
// disabled, output bit-identical to the pre-migration eager output". The stock
// CPU platform never answers `support_static_graph_mode()` true, so the driver
// declines itself and runs `ForwardBody`.
TEST_CASE("G4: the seam changed no numerics on DeepseekV2DecodeGraph's capture step") {
  const DeepseekV2Params p = TinyParams();
  const DeepseekV2Weights w = TinyWeights(p);
  vt::Queue q = Q();

  const std::vector<int32_t> tok0 = {11}, pos0 = {0};
  const std::vector<int32_t> tok1 = {12}, pos1 = {1};

  std::vector<float> eager;
  {
    MlaCachePool ref_pool(p, 2, 8);
    vllm::DeepseekV2DecodeGraph plain(w, q, /*max_num_reqs=*/8);
    plain.Step(tok0, pos0, DecodeMeta(0), ref_pool.attn_kv);
    const vllm::ForwardLogits fl = plain.Step(tok1, pos1, DecodeMeta(1), ref_pool.attn_kv);
    REQUIRE_FALSE(plain.captured());
    REQUIRE(fl.on_device());
    const auto* ep = static_cast<const float*>(fl.device_tensor.data);
    REQUIRE(ep != nullptr);
    eager.assign(ep, ep + p.vocab_size);
  }

  StaticGraphCpu harness;
  MlaCachePool pool(p, 2, 8);
  vllm::DeepseekV2DecodeGraph graph(w, q, /*max_num_reqs=*/8);
  graph.Step(tok0, pos0, DecodeMeta(0), pool.attn_kv);
  const vllm::ForwardLogits captured = graph.Step(tok1, pos1, DecodeMeta(1), pool.attn_kv);
  REQUIRE(graph.captured());
  REQUIRE(captured.on_device());
  const auto* cp = static_cast<const float*>(captured.device_tensor.data);
  REQUIRE(cp != nullptr);

  size_t differing = 0;
  for (size_t i = 0; i < eager.size(); ++i)
    if (std::memcmp(&eager[i], &cp[i], sizeof(float)) != 0) ++differing;
  CHECK(differing == 0);
  MESSAGE("driver capture step vs its own eager arm, bit for bit: "
          << eager.size() << " values, " << differing << " differing");
  for (size_t i = 0; i < eager.size(); ++i) REQUIRE(std::isfinite(cp[i]));
}

// THE CAPTURE-FAILURE GATE. `~GraphCaptureScope` must swallow a throwing
// `EndCaptureGraph` — a destructor that propagates terminates — so after a
// FAILED capture the container reports exactly what an INERT scope reports:
// `captured() == false`. A driver that reads only that bit returns the value its
// forward "produced" under capture; on CUDA nothing between `BeginCapture` and
// the throw executed, so that buffer holds whatever the `DevicePool` last left
// there and the step hands back pool-recycled memory as its logits — silently
// wrong tokens, no fault, and a token gate cannot see it.
// `vt::BreakableGraph::capture_failed()` separates the two states and this driver
// RETHROWS, which is what its PRE-W3 code did (`s.graph = b.EndCaptureGraph(...)`
// was unguarded).
TEST_CASE("DeepseekV2DecodeGraph: a capture that FAILS reaches the caller") {
  const DeepseekV2Params p = TinyParams();
  const DeepseekV2Weights w = TinyWeights(p);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  MlaCachePool pool(p, /*num_blocks=*/2, /*block_size=*/8);
  vllm::DeepseekV2DecodeGraph graph(w, q, /*max_num_reqs=*/8);

  graph.Step({11}, {0}, DecodeMeta(0), pool.attn_kv);
  REQUIRE_FALSE(graph.captured());

  harness.backend().FailNextEndCapture();
  CHECK_THROWS_AS(graph.Step({12}, {1}, DecodeMeta(1), pool.attn_kv), std::runtime_error);
  CHECK_FALSE(graph.captured());

  // AND THE DRIVER RECOVERS: the failed capture is not sticky. Without this arm
  // the fix could be "throw forever" and still look gated.
  graph.Step({13}, {2}, DecodeMeta(2), pool.attn_kv);
  const vllm::ForwardLogits after = graph.Step({14}, {3}, DecodeMeta(3), pool.attn_kv);
  CHECK(graph.captured());
  REQUIRE(after.on_device());
  const auto* pp = static_cast<const float*>(after.device_tensor.data);
  REQUIRE(pp != nullptr);
  for (int64_t i = 0; i < p.vocab_size; ++i) REQUIRE(std::isfinite(pp[i]));
}
