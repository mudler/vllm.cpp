// THE W4 G2 REACHABILITY GATE for `Qwen3_5DecodeGraph`, the GDN-hybrid MoE
// decode driver — the RICHEST of the nine and the one that HAS the persistent
// device input path (`StepDevInputs`, 41 lines of `qwen3_5.cpp`).
//
// Row ENG-CUDAGRAPH-BREAK W4, spec `.agents/specs/eng-cudagraph-break.md`,
// issue #1307, parent #1163.
//
// WHY THIS FILE EXISTS AND A TOKEN GATE DOES NOT SUFFICE. A driver that kept its
// hand-rolled `BeginCapture`/`EndCaptureGraph` pair produces IDENTICAL logits, an
// IDENTICAL backend call log and an identical `replay_count()`.
// `vt::GraphBreakStats::segments_captured` moves only when a
// `vt::GraphCaptureScope` closes a segment and `replays` only inside
// `vt::BreakableGraph::Replay`, so those two are the only observables in the
// tree that separate "captured a graph" from "captured a graph THROUGH THE
// SEAM". W2 (#1261) and W3 (#1291) established this shape; this file applies it
// to the two drivers W4 owns.
//
// kFULL, ASSERTED ON THE MODE ITSELF. vLLM's v1 default `FULL_AND_PIECEWISE`
// (`vllm/config/compilation.py:63` @ pin `5559679229`) is documented at
// `:630-632` as a FULL graph for DECODE batches and a piecewise one for prefill
// and mixed batches, and `decode_mode()` (`:65-66`) returns the full half. W3
// measured what happens when the mode is inferred from a side effect instead of
// counted: flipping one token from `kFull` to `kPiecewise` compiled clean and
// left a whole driver gate GREEN at 226/226, because `breaks_registered == 0` is
// true in BOTH modes for a model that registers no break point. `full_scopes`
// and `piecewise_scopes` are what actually move.
//
// The harness and what it cannot see are stated once, in
// `decode_graph_seam_harness.h`. In one line: a CPU "replay" recomputes nothing,
// so this file gates the ROUTING and the capture step's numerics and NOT that a
// replayed segment reproduces the eager forward. That is G1, it needs a real
// device, and the spec's `## Gates` G1 records where it ran.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "decode_graph_seam_harness.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"  // W6 (#1374) dispatch counters
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/dtype.h"
#include "vt/persistent_step_input.h"
#include "vt/tensor.h"

namespace {

using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::Qwen3_5MoeWeights;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vllm_test::StaticGraphCpu;
using vt::DType;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// The tiny model is the one `tests/vllm/models/test_qwen35_paged_forward.cpp`
// already uses, so this driver runs the SAME arithmetic that file gates.
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
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

HfConfig TinyConfig() {
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

// KV plus recurrent-state caches for the GDN hybrid. Each arm of a comparison
// gets its OWN pool, so neither can read the other's writes and call the
// agreement a result.
struct CachePool {
  const HfConfig& c;
  int64_t num_blocks;
  int64_t block_size;
  std::vector<std::vector<float>> full_attn_buf;
  std::vector<std::vector<float>> gdn_ssm_buf;
  std::vector<std::vector<float>> gdn_conv_buf;
  std::vector<PagedKvCache> attn_kv;
  std::vector<GdnStateCache> gdn_state;
  int64_t spec_max_query_len = 1;

  // `spec_mql` is the longest speculative query length this pool must serve.
  // `causal_conv1d_spec_update` derives it from the conv state WIDTH --
  // `state_len = (K - 1) + (mql - 1)` (`src/vt/ops.cpp:1903`) -- so a pool built
  // at the plain decode width refuses every q above 1. Default 1 keeps every
  // existing case byte-identical.
  CachePool(const HfConfig& cfg, int64_t nb, int64_t bs, int64_t spec_mql = 1)
      : c(cfg), num_blocks(nb), block_size(bs), spec_max_query_len(spec_mql) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    const int64_t Hv = c.linear_num_value_heads, Dv = c.linear_value_head_dim,
                  Dk = c.linear_key_head_dim, Kw = c.linear_conv_kernel_dim;
    const int64_t key_dim = c.linear_num_key_heads * Dk, value_dim = Hv * Dv;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      if (c.layer_types[static_cast<size_t>(l)] == "linear_attention") {
        gdn_ssm_buf.emplace_back(static_cast<size_t>(nb * Hv * Dv * Dk), 0.0f);
        gdn_conv_buf.emplace_back(
            static_cast<size_t>(nb * conv_dim * (Kw - 1 + spec_max_query_len - 1)),
            0.0f);
      } else {
        full_attn_buf.emplace_back(static_cast<size_t>(nb * 2 * bs * Hkv * Dh), 0.0f);
      }
    }
    const vt::Device dev{vt::DeviceType::kCPU, 0};
    for (auto& b : full_attn_buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = nb;
      kv.block_size = bs;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
    for (size_t g = 0; g < gdn_ssm_buf.size(); ++g) {
      GdnStateCache gs;
      gs.ssm_state = vt::Tensor::Contiguous(gdn_ssm_buf[g].data(), DType::kF32, dev,
                                            {nb, Hv, Dv, Dk});
      gs.conv_state = vt::Tensor::Contiguous(
          gdn_conv_buf[g].data(), DType::kF32, dev,
          {nb, conv_dim, Kw - 1 + spec_max_query_len - 1});
      gdn_state.push_back(gs);
    }
  }
};

// One single-request PURE-DECODE step at `pos`.
CommonAttentionMetadata DecodeAttnMeta(int32_t pos) {
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

GDNAttentionMetadata DecodeGdnMeta() {
  GDNAttentionMetadata gm;
  gm.num_prefills = 0;
  gm.num_prefill_tokens = 0;
  gm.num_decodes = 1;
  gm.num_decode_tokens = 1;
  gm.num_actual_tokens = 1;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1};
  return gm;
}

}  // namespace

// THE GATE. Four decode steps at one shape drive the driver's whole state
// machine: cold (eager pre-warm), warm (CAPTURE), captured (REPLAY), and a
// second REPLAY.
TEST_CASE("G2: Qwen3_5DecodeGraph captures and replays THROUGH the vt seam") {
  const HfConfig c = TinyConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/16);

  vt::ResetGraphBreakStats();
  vllm::Qwen3_5DecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  // Step 1, COLD: an eager pre-warm step. Nothing is captured yet.
  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 0);
    CHECK(s.replays == 0);
    CHECK_FALSE(graph.captured());
  }

  // Step 2, WARM: the driver captures. This is the assertion that fails while
  // the driver keeps its own BeginCapture/EndCaptureGraph pair.
  graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 1);
    CHECK(graph.captured());
    CHECK(s.full_scopes == 1);
    CHECK(s.piecewise_scopes == 0);
    CHECK(s.breaks_registered == 0);
  }

  // Step 3, CAPTURED: the driver replays. `replays` moves ONLY inside
  // vt::BreakableGraph::Replay.
  graph.Step({13}, {2}, DecodeAttnMeta(2), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.replays >= 1);
    CHECK(graph.replay_count() >= 1);
  }

  // Step 4, a SECOND replay: the REPLAY branch is re-entrant. A branch that
  // re-captured, reset the container or left `warm` set would still look correct
  // after one step and diverge only on the next.
  graph.Step({14}, {3}, DecodeAttnMeta(3), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.replays >= 2);
    CHECK(s.segments_captured == 1);  // a replay must never re-enter capture
  }

  // The backend saw exactly one capture pair, and the seam owns the release:
  // `BreakableGraph`'s destructor routes every handle through
  // Backend::DestroyGraph, which is what lets #1162 interpose later (spec D4).
  CHECK(harness.backend().Count("Begin") == 1);
  CHECK(harness.backend().Count("EndCaptureGraph") == 1);
  CHECK(harness.backend().Count("ReplayGraph") >= 2);
}

// G4 for this driver: with the seam in place, the CAPTURE step's logits are
// bit-identical to the same step run through the plain eager forward. This is
// what makes the migration reversible — and it is NOT G1, which needs a real
// device because a CPU "replay" recomputes nothing.
TEST_CASE("G4: the seam changed no numerics on Qwen3_5DecodeGraph's capture step") {
  const HfConfig c = TinyConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();

  // EAGER reference, on the stock CPU platform where the driver does not admit
  // itself, with its own caches.
  CachePool ref(c, 4, 16);
  vllm::Qwen3_5Model::Forward({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(),
                              ref.attn_kv, ref.gdn_state, w, c, q);
  const std::vector<float> eager = vllm::Qwen3_5Model::Forward(
      {12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), ref.attn_kv, ref.gdn_state, w,
      c, q);
  REQUIRE(eager.size() == static_cast<size_t>(c.vocab_size));

  // The DRIVER's cold step then its capture step, on its own caches.
  StaticGraphCpu harness;
  CachePool pool(c, 4, 16);
  vllm::Qwen3_5DecodeGraph graph(w, c, q, /*max_num_reqs=*/4);
  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  const vllm::ForwardLogits captured =
      graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
                 pool.gdn_state);
  REQUIRE(graph.captured());

  const auto* cp = static_cast<const float*>(captured.device_tensor.data);
  REQUIRE(cp != nullptr);
  size_t differing = 0;
  for (size_t i = 0; i < eager.size(); ++i)
    if (cp[i] != eager[i]) ++differing;
  MESSAGE("driver capture step vs eager, bit for bit: " << eager.size() << " values, "
                                                        << differing << " differing");
  CHECK(differing == 0);
  for (size_t i = 0; i < eager.size(); ++i) REQUIRE(std::isfinite(cp[i]));
}

// A capture that FAILS must reach the caller, and the step must never return the
// buffer as logits. Under stream capture NOTHING between `BeginCapture` and a
// throwing `EndCaptureGraph` executed — every kernel was RECORDED — so that
// buffer holds whatever the `DevicePool` last left there. Returning it is
// silently wrong tokens with no fault, and a token gate cannot see it. This was
// a live HIGH on W2's first head; the pre-W4 driver propagated because its
// `s.graph = b.EndCaptureGraph(...)` was unguarded, and this case is what stops
// the migration from losing that.
TEST_CASE("A capture that FAILS reaches the caller; Qwen3_5DecodeGraph never returns it") {
  const HfConfig c = TinyConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();

  StaticGraphCpu harness;
  CachePool pool(c, 4, 16);
  vllm::Qwen3_5DecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // cold pre-warm
  harness.backend().FailNextEndCapture();
  CHECK_THROWS_AS(graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(),
                             pool.attn_kv, pool.gdn_state),
                  std::exception);
  CHECK_FALSE(graph.captured());
}

// ───────────────────────────────────────────────────────────────────────────
// THE DENSE SIBLING, `Qwen3_5DenseDecodeGraph`.
// ───────────────────────────────────────────────────────────────────────────
//
// IT OWES ITS OWN GATE and shares this file's harness, which is the split W3
// arrived at the hard way. Its own gate, because nothing else can see the
// difference: a driver that kept its raw `BeginCapture`/`EndCaptureGraph` pair
// produces identical logits, an identical backend log and an identical
// `replay_count()`, and the two drivers migrate through separate code even
// though they migrate identically. This file rather than a fifth one, because
// copying the harness per gate would reproduce inside `tests/` the exact
// duplication this row removes from `src/`, and two copies of a harness diverge
// invisibly — both stay green while measuring different things.
namespace dense {

using vllm::DenseMlpWeights;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5DenseWeights;

// The tiny model `tests/vllm/models/test_qwen27_paged_forward.cpp` already uses:
// 27B-shaped, layer_types [LA, LA, LA, FA], no experts, GQA ratio 3.
HfConfig TinyConfig() {
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
  c.linear_num_value_heads = 6;
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
    Qwen3_5DenseLayerWeights lw;
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
    lw.mlp = MakeMlp(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

}  // namespace dense

TEST_CASE("G2: Qwen3_5DenseDecodeGraph captures and replays THROUGH the vt seam") {
  const HfConfig c = dense::TinyConfig();
  const dense::Qwen3_5DenseWeights w = dense::MakeWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/16);

  vt::ResetGraphBreakStats();
  vllm::Qwen3_5DenseDecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // COLD
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 0);
    CHECK(s.replays == 0);
    CHECK_FALSE(graph.captured());
  }

  graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // WARM: capture
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 1);
    CHECK(graph.captured());
    CHECK(s.full_scopes == 1);
    CHECK(s.piecewise_scopes == 0);
    CHECK(s.breaks_registered == 0);
  }

  graph.Step({13}, {2}, DecodeAttnMeta(2), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // REPLAY
  CHECK(vt::GetGraphBreakStats().replays >= 1);
  CHECK(graph.replay_count() >= 1);

  graph.Step({14}, {3}, DecodeAttnMeta(3), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // REPLAY again
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.replays >= 2);
    CHECK(s.segments_captured == 1);  // a replay must never re-enter capture
  }

  CHECK(harness.backend().Count("Begin") == 1);
  CHECK(harness.backend().Count("EndCaptureGraph") == 1);
  CHECK(harness.backend().Count("ReplayGraph") >= 2);
}

TEST_CASE("G4: the seam changed no numerics on Qwen3_5DenseDecodeGraph's capture step") {
  const HfConfig c = dense::TinyConfig();
  const dense::Qwen3_5DenseWeights w = dense::MakeWeights(c);
  vt::Queue q = Q();

  CachePool ref(c, 4, 16);
  vllm::Qwen3_5DenseModel::Forward({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(),
                                   ref.attn_kv, ref.gdn_state, w, c, q);
  const std::vector<float> eager = vllm::Qwen3_5DenseModel::Forward(
      {12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), ref.attn_kv, ref.gdn_state, w,
      c, q);
  REQUIRE(eager.size() == static_cast<size_t>(c.vocab_size));

  StaticGraphCpu harness;
  CachePool pool(c, 4, 16);
  vllm::Qwen3_5DenseDecodeGraph graph(w, c, q, /*max_num_reqs=*/4);
  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  const vllm::ForwardLogits captured =
      graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
                 pool.gdn_state);
  REQUIRE(graph.captured());

  const auto* cp = static_cast<const float*>(captured.device_tensor.data);
  REQUIRE(cp != nullptr);
  size_t differing = 0;
  for (size_t i = 0; i < eager.size(); ++i)
    if (cp[i] != eager[i]) ++differing;
  MESSAGE("dense driver capture step vs eager, bit for bit: "
          << eager.size() << " values, " << differing << " differing");
  CHECK(differing == 0);
  for (size_t i = 0; i < eager.size(); ++i) REQUIRE(std::isfinite(cp[i]));
}

TEST_CASE("A capture that FAILS reaches the caller; Qwen3_5DenseDecodeGraph never returns it") {
  const HfConfig c = dense::TinyConfig();
  const dense::Qwen3_5DenseWeights w = dense::MakeWeights(c);
  vt::Queue q = Q();

  StaticGraphCpu harness;
  CachePool pool(c, 4, 16);
  vllm::Qwen3_5DenseDecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  harness.backend().FailNextEndCapture();
  CHECK_THROWS_AS(graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(),
                             pool.attn_kv, pool.gdn_state),
                  std::exception);
  CHECK_FALSE(graph.captured());
}

// ───────────────────────────────────────────────────────────────────────────
// THE CAPABILITY IS REACHED FROM A PRODUCTION STEP.
// ───────────────────────────────────────────────────────────────────────────
//
// `.agents/reachability.md` and the AGENTS.md "Nothing lands dead" rule: a seam
// capability that lands unreachable by the drivers that need it is the exact
// failure they exist for, and a unit test that constructs the type by hand
// proves the class works and never that anything reaches it. So this case enters
// through `Qwen3_5DecodeGraph::Step` — the same entry the registered forward
// calls — and asserts the PROCESS-WIDE counters, which move only inside
// `vt::PersistentStepInput`.
//
// IT SETS `VT_ASYNC_EXECUTOR=1`, and that is a limit stated rather than hidden.
// The persistent device input path is behind that lever (default OFF) plus the
// speculative-decode arm, so on a default text-decode step the driver holds no
// `StepDevInputs` at all and stages nothing. What this case therefore proves is
// that the production `Step` REACHES the capability on the configuration that
// has persistent device inputs — not that every step does. The lever is read
// once into `Impl::dbuf` at construction, so it is set before the driver is
// built and restored immediately after.
namespace {
struct ScopedEnv {
  const char* name;
  std::string prev;
  bool had;
  ScopedEnv(const char* n, const char* v) : name(n) {
    const char* p = std::getenv(n);
    had = p != nullptr;
    if (had) prev = p;
    ::setenv(n, v, 1);
  }
  ~ScopedEnv() {
    if (had)
      ::setenv(name, prev.c_str(), 1);
    else
      ::unsetenv(name);
  }
};
}  // namespace

TEST_CASE("Qwen3_5DecodeGraph::Step REACHES vt::PersistentStepInput") {
  const HfConfig c = TinyConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();

  StaticGraphCpu harness;
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/16);
  vt::ResetStepInputStats();
  REQUIRE(vt::GetStepInputStats().binds == 0);

  const ScopedEnv async_on("VT_ASYNC_EXECUTOR", "1");
  vllm::Qwen3_5DecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  // THE LEVER ALSO TURNS ON THE 2-SLOT PARITY RING, so each padded size holds
  // two slots and consecutive steps alternate between them. Slot 0 is therefore
  // cold on step 1 and WARM on step 3, which is where it captures — a three-step
  // walk here rather than the two-step one the single-slot cases use. That ring
  // is the reason the lever exists: at depth 2 the engine enqueues sample(i-1)
  // AFTER forward(i), so a single persistent logits buffer would be overwritten
  // before it is read.
  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // slot 0, COLD
  graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // slot 1, COLD
  graph.Step({13}, {2}, DecodeAttnMeta(2), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // slot 0, WARM: binds the persistent inputs, captures
  REQUIRE(graph.captured());
  const vt::StepInputStats after_capture = vt::GetStepInputStats();
  MESSAGE("after capture: binds=" << after_capture.binds << " host_refreshes="
                                  << after_capture.host_refreshes);
  // Five inputs are bound unconditionally (positions, slot_mapping, block_table,
  // seq_lens, query_start_loc) and the GDN state index only when the step
  // carries one, so the floor is five rather than an exact count.
  CHECK(after_capture.binds >= 5);

  graph.Step({14}, {3}, DecodeAttnMeta(3), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // slot 1, WARM
  graph.Step({15}, {4}, DecodeAttnMeta(4), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // slot 0, REPLAY: stages through the seam
  const vt::StepInputStats after_replay = vt::GetStepInputStats();
  CHECK(after_replay.host_refreshes >= 5);

  // The DEVICE arm is NOT exercised here, and saying so is the point of this
  // line. No decode driver refreshes an input from a device source yet: the one
  // input an asynchronous mirror patches is the token ids, and no driver has a
  // device destination for them (see PinnedStepInputs in `qwen3_5.cpp`). The
  // capability is landed and reached; adopting its device arm is what the
  // `qwen3.cpp` decline needs, and the spec's `## Owed` names it.
  CHECK(after_replay.device_refreshes == 0);
}

// ───────────────────────────────────────────────────────────────────────────
// ENG-CUDAGRAPH-BREAK W6 (#1374): THE SLOT KEY, closing the half of
// [#1020](https://github.com/mudler/vllm.cpp/issues/1020) that is not the
// predicate.
//
// WHY THIS IS NOT A UNIT TEST OF A COMPARATOR. `DecodeGraphSlotKey`'s ordering
// is four lines and a test that constructed one by hand would prove the
// comparator sorts. The claim here is different: that two SPEC steps whose
// padded token counts are EQUAL and whose uniform query lengths DIFFER get two
// graphs rather than one. Only `Qwen3_5DecodeGraph::Step` can answer that, and
// only through the driver's own capture counters -- a token gate cannot see
// which graph a replay came from, which is precisely how the collision stayed
// invisible.
//
// THE COLLISION IS REACHABLE TODAY AND NOT ONLY AFTER THE WIDENING.
// `S = spec_step ? B : PadToCaptureSize(B)`, so 2 requests x 2 tokens and
// 1 request x 4 tokens are both S == 4. Before W6 both indexed `slots[4]`, and
// the second replayed a graph captured against the first's metadata --
// different `spec_query_start_loc`, different `num_accepted_tokens`, different
// row count. `SizeSlot::Refresh` copies IN PLACE only while the sizes match and
// REASSIGNS the vector when they do not, which also moves the host addresses a
// capture baked (spec `## Risks/decisions` D2). Silently wrong logits.
namespace {

// One PURE-SPEC batch: `reqs` requests, each verifying `q` tokens.
// Mirrors the gdn_attn.cpp spec contract the driver's
// `ValidateGdnDecodeGraphState` enforces (`qwen3_5.cpp:367-421`): no non-spec
// rows, so `num_decodes == num_prefills == 0` and the non-spec segmentation is
// nullopt by construction.
GDNAttentionMetadata SpecGdnMeta(int32_t reqs, int32_t q) {
  GDNAttentionMetadata gm;
  gm.num_prefills = 0;
  gm.num_prefill_tokens = 0;
  gm.num_decodes = 0;
  gm.num_decode_tokens = 0;
  gm.num_spec_decodes = reqs;
  gm.num_spec_decode_tokens = reqs * q;
  gm.num_actual_tokens = reqs * q;
  gm.spec_state_indices_num_cols = q;
  std::vector<int32_t> ssi(static_cast<size_t>(reqs) * static_cast<size_t>(q));
  for (int32_t r = 0; r < reqs; ++r)
    for (int32_t j = 0; j < q; ++j)
      ssi[static_cast<size_t>(r * q + j)] = r;
  gm.spec_state_indices_tensor = std::move(ssi);
  std::vector<int32_t> sqsl(static_cast<size_t>(reqs) + 1);
  for (int32_t r = 0; r <= reqs; ++r) sqsl[static_cast<size_t>(r)] = r * q;
  gm.spec_query_start_loc = std::move(sqsl);
  gm.spec_sequence_masks = std::vector<uint8_t>(static_cast<size_t>(reqs), 1);
  std::vector<int32_t> stx(static_cast<size_t>(reqs) * static_cast<size_t>(q));
  for (size_t i = 0; i < stx.size(); ++i) stx[i] = static_cast<int32_t>(i);
  gm.spec_token_indx = std::move(stx);
  // Accept ONE token per request, the shape a verify step arrives in before the
  // rejection sampler has run. Column `acc - 1 == 0` is the live initial slot.
  gm.num_accepted_tokens = std::vector<int32_t>(static_cast<size_t>(reqs), 1);
  return gm;
}

CommonAttentionMetadata SpecAttnMeta(int32_t reqs, int32_t q, int32_t pos) {
  CommonAttentionMetadata am;
  am.num_reqs = reqs;
  am.num_actual_tokens = reqs * q;
  am.query_start_loc.resize(static_cast<size_t>(reqs) + 1);
  for (int32_t r = 0; r <= reqs; ++r)
    am.query_start_loc[static_cast<size_t>(r)] = r * q;
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens.assign(static_cast<size_t>(reqs), pos + q);
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = q;
  am.max_seq_len = pos + q;
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(static_cast<size_t>(reqs), 0);
  am.slot_mapping.resize(static_cast<size_t>(reqs) * static_cast<size_t>(q));
  for (size_t i = 0; i < am.slot_mapping.size(); ++i)
    am.slot_mapping[i] = pos + static_cast<int32_t>(i);
  am.causal = true;
  return am;
}

std::vector<int32_t> Iota(int32_t n, int32_t base) {
  std::vector<int32_t> v(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = base + i;
  return v;
}

}  // namespace

TEST_CASE("W6: two spec shapes of EQUAL S and different q get two graphs") {
  // The conv window bounds the verify length this model can run:
  // `causal_conv1d_spec_update` refuses a query length above
  // `linear_conv_kernel_dim - 1` (`src/vt/ops.cpp:1913`). The shared
  // `TinyConfig` uses 4, which caps q at 3 and leaves no room for the third
  // distinct length the capture bound needs. Widen the window in a LOCAL copy
  // and build this case's weights and caches from that same copy, so nothing
  // else in the file observes it.
  HfConfig c = TinyConfig();
  c.linear_conv_kernel_dim = 6;
  const Qwen3_5MoeWeights w = MakeWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/16, /*spec_mql=*/4);

  vt::ResetGraphBreakStats();
  vllm::v1::ResetGraphDispatchStats();
  vllm::Qwen3_5DecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  // SHAPE A: 3 requests x 2 tokens == 6 tokens. THREE steps, because a spec step
  // always takes the two-slot parity ring (`dbuf = impl_->dbuf || spec_step`):
  // slot 0 runs cold, slot 1 runs cold, and slot 0 is warm and captures on the
  // THIRD. Two steps would leave nothing captured and the collision below
  // unexercised, which is the "assertions: 0 wearing a pass" shape.
  for (int step = 0; step < 3; ++step) {
    graph.Step(Iota(6, 10), Iota(6, 0), SpecAttnMeta(3, 2, 0), SpecGdnMeta(3, 2),
               pool.attn_kv, pool.gdn_state);
  }
  CHECK(vllm::v1::GetGraphDispatchStats().capture_shapes == 1);
  CHECK(vt::GetGraphBreakStats().segments_captured == 1);

  // SHAPE B: 2 requests x 3 tokens == 6 tokens. The SAME S, a DIFFERENT q.
  // Before W6 this indexed the ring shape A already captured, found
  // `graph.captured()` true on the FIRST step, and replayed shape A's graph
  // against shape B's metadata. The key separates them, so this opens a ring of
  // its own and takes the cold path again.
  for (int step = 0; step < 3; ++step) {
    graph.Step(Iota(6, 20), Iota(6, 0), SpecAttnMeta(2, 3, 8), SpecGdnMeta(2, 3),
               pool.attn_kv, pool.gdn_state);
  }
  const vllm::v1::GraphDispatchStats st = vllm::v1::GetGraphDispatchStats();
  CAPTURE(st.capture_shapes);
  CAPTURE(st.qlen_cap_declines);
  // THE ASSERTION THE MUTATION MOVES. Two rings, not one.
  CHECK(st.capture_shapes == 2);
  // And two captures, which is what says the second ring was actually USED
  // rather than merely created. `segments_captured` moves only when a
  // `vt::GraphCaptureScope` closes a segment.
  CHECK(vt::GetGraphBreakStats().segments_captured == 2);
  CHECK(st.qlen_cap_declines == 0);

  // SHAPE C: 1 request x 4 tokens, a THIRD distinct speculative query length.
  // `VT_SPEC_GRAPH_MAX_QLENS` defaults to 2, so this one is refused and runs
  // eager -- which is what every clamped shape did before W6. The difference is
  // that it now moves a counter instead of nothing, which is the half of #1020
  // its title is about.
  for (int step = 0; step < 3; ++step) {
    graph.Step(Iota(4, 30), Iota(4, 0), SpecAttnMeta(1, 4, 20), SpecGdnMeta(1, 4),
               pool.attn_kv, pool.gdn_state);
  }
  const vllm::v1::GraphDispatchStats st2 = vllm::v1::GetGraphDispatchStats();
  CAPTURE(st2.capture_shapes);
  CAPTURE(st2.qlen_cap_declines);
  CHECK(st2.qlen_cap_declines == 3);
  CHECK(st2.capture_shapes == 2);  // no third ring was opened
  CHECK(vt::GetGraphBreakStats().segments_captured == 2);
}
