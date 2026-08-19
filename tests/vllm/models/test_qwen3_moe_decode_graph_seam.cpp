// THE G2 REACHABILITY GATE for ENG-CUDAGRAPH-BREAK W3 (#1291, parent #1163),
// for `Qwen3MoeDecodeGraph` — per `.agents/reachability.md` and the AGENTS.md
// "Nothing lands dead" rule.
//
// WHAT IT MEASURES. W2 (#1261) migrated `Qwen3DenseDecodeGraph` onto the shared
// break-point capture seam and gated it at
// `tests/vllm/models/test_qwen3_decode_graph_seam.cpp`. W3 migrates the three
// remaining PLAIN BATCHED drivers, of which this is one, and each repeat owes
// its own gate rather than inheriting W2's: a driver that kept its hand-rolled
// `BeginCapture`/`EndCaptureGraph`/`ReplayGraph`/`DestroyGraph` sequence
// produces IDENTICAL logits, an IDENTICAL backend log and an identical
// `replay_count()`, so no gate on any of those three can tell the two apart.
//
// WHY THE SEAM COUNTERS AND NOT THE BACKEND LOG. The seam bottoms out in the
// same `Backend::` calls the raw driver made, so the log is not an observable
// that separates them. `vt::GraphBreakStats::segments_captured` moves only when
// a `vt::GraphCaptureScope` closes a segment, and `replays` moves only inside
// `vt::BreakableGraph::Replay`. Those two numbers are what separate "this driver
// captured a graph" from "this driver captured a graph THROUGH THE SEAM", which
// is the whole distinction G2 exists to make.
//
// THE MUTATION THIS FILE ANSWERS TO: in a scratch copy, replace this driver's
// `vt::GraphCaptureScope` and `BreakableGraph::Replay` with the raw backend pair
// they replaced. Every other gate in this tree stays green — the model's
// arithmetic is untouched — and the counter assertions below go to zero and turn
// this file RED.
//
// kFULL, AND THE BREAK COUNT IS ZERO ON PURPOSE. vLLM's v1 default
// `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:63` @ pin `5559679229`) is
// documented at `:630-632` as a FULL graph for DECODE batches and a piecewise
// one for prefill and mixed batches, and `decode_mode()` (`:65-66`) returns the
// full half. This is a decode driver, so `breaks_registered` must stay 0: a
// migration that opened its scope `kPiecewise` would split this step into one
// eager attention call per layer between graph replays, which is not vLLM's
// decode behaviour. The assertion below is what stops that from landing quietly.
//
// The harness and what it cannot see are stated once, in
// `decode_graph_seam_harness.h`. In one line: a CPU "replay" recomputes nothing,
// so this file gates the ROUTING and the capture step's numerics and NOT that a
// replayed segment reproduces the eager forward. That is G1; the spec's
// `## Owed` records it.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "decode_graph_seam_harness.h"
#include "vllm/model_executor/models/qwen3_moe.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::HfConfig;
using vllm::PagedKvCache;
using vllm::Qwen3MoeWeights;
using vllm::v1::CommonAttentionMetadata;
using vllm_test::StaticGraphCpu;
using vt::DType;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// The tiny model is the one `tests/vllm/models/test_qwen3_moe_forward.cpp`
// already uses, so the driver here runs the SAME arithmetic that file gates.
vllm::OwnedTensor MakeBf16(const std::vector<int64_t>& shape, bool nk, uint32_t seed,
                           float scale = 0.08f) {
  vllm::OwnedTensor o;
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

HfConfig TinyConfig() {
  HfConfig c;
  c.num_hidden_layers = 2;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.rotary_dim = 16;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 10000000.0;
  c.vocab_size = 100;
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 32;
  c.shared_expert_intermediate_size = 0;  // NO shared expert
  return c;
}

Qwen3MoeWeights TinyWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads, Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim, V = c.vocab_size;
  const int64_t E = c.num_experts, I = c.moe_intermediate_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  Qwen3MoeWeights w;
  w.tie_word_embeddings = false;  // UNTIED
  w.attention_bias = false;
  w.embed_tokens = MakeBf16({V, H}, /*nk=*/false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.5f);
  w.lm_head = MakeBf16({H, V}, /*nk=*/false, 3);
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    vllm::Qwen3MoeLayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, /*nk=*/true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, /*nk=*/true, seed++);
    lw.attn.q_norm = MakeBf16({Dh}, false, seed++, 0.5f);
    lw.attn.k_norm = MakeBf16({Dh}, false, seed++, 0.5f);
    lw.moe.router_gate = MakeBf16({H, E}, /*nk=*/false, seed++);
    for (int64_t e = 0; e < E; ++e) {
      lw.moe.expert_gate.push_back(MakeBf16({H, I}, /*nk=*/false, seed++));
      lw.moe.expert_up.push_back(MakeBf16({H, I}, /*nk=*/false, seed++));
      lw.moe.expert_down.push_back(MakeBf16({I, H}, /*nk=*/false, seed++));
    }
    w.layers.push_back(std::move(lw));
  }
  return w;
}

struct CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<PagedKvCache> attn_kv;
  CachePool(const HfConfig& c, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l)
      buf.emplace_back(static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0.0f);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
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

}  // namespace

// The gate. Four decode steps at one shape drive the driver's whole state
// machine: cold (eager pre-warm), warm (CAPTURE), captured (REPLAY), and a
// second REPLAY.
TEST_CASE("G2: Qwen3MoeDecodeGraph captures and replays THROUGH the vt seam") {
  const HfConfig c = TinyConfig();
  const Qwen3MoeWeights w = TinyWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);

  vt::ResetGraphBreakStats();
  vllm::Qwen3MoeDecodeGraph graph(w, c, q, /*max_num_reqs=*/8);

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

  // Step 4, a SECOND replay: the driver's REPLAY branch is re-entrant. A branch
  // that re-captured, reset the container or left `warm` set would still look
  // correct after one step and diverge only on the next.
  graph.Step({14}, {3}, DecodeMeta(3), pool.attn_kv);
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
TEST_CASE("G4: the seam changed no numerics on Qwen3MoeDecodeGraph's capture step") {
  const HfConfig c = TinyConfig();
  const Qwen3MoeWeights w = TinyWeights(c);
  vt::Queue q = Q();

  const std::vector<int32_t> tok0 = {11}, pos0 = {0};
  const std::vector<int32_t> tok1 = {12}, pos1 = {1};

  // Eager reference: the same two steps through Qwen3MoeModel::Forward, on the
  // stock CPU platform where the driver does not admit itself.
  CachePool ref_pool(c, 2, 8);
  const std::vector<int32_t> no_gather;
  vllm::Qwen3MoeModel::Forward(tok0, pos0, DecodeMeta(0), ref_pool.attn_kv, w, c, q,
                               no_gather);
  const std::vector<float> eager = vllm::Qwen3MoeModel::Forward(
      tok1, pos1, DecodeMeta(1), ref_pool.attn_kv, w, c, q, no_gather);
  REQUIRE(eager.size() == static_cast<size_t>(c.vocab_size));

  StaticGraphCpu harness;
  CachePool pool(c, 2, 8);
  vllm::Qwen3MoeDecodeGraph graph(w, c, q, /*max_num_reqs=*/8);
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
  MESSAGE("driver capture step vs eager, bit for bit: " << eager.size() << " values, "
                                                        << differing << " differing");
  for (size_t i = 0; i < eager.size(); ++i) REQUIRE(std::isfinite(cp[i]));
}

// THE CAPTURE-FAILURE GATE. `~GraphCaptureScope` must swallow a throwing
// `EndCaptureGraph` — a destructor that propagates terminates — so after a
// FAILED capture the container reports exactly what an INERT scope reports:
// `captured() == false`. A driver that reads only that one bit returns the value
// its forward "produced" under capture. On CUDA nothing between `BeginCapture`
// and the throw executed: every kernel was RECORDED, not run, so that buffer
// holds whatever the `DevicePool` last left there and the step returns
// pool-recycled memory as its logits — silently wrong tokens, no fault, and a
// token gate cannot see it. `vt::BreakableGraph::capture_failed()` separates the
// two states and this driver RETHROWS, which is exactly what its PRE-W3 code did
// (`s.graph = b.EndCaptureGraph(...)` was unguarded).
//
// WHAT THIS HARNESS CANNOT SEE, named rather than claimed away: a CPU "capture"
// EXECUTES, so the logits here hold real numbers and this file cannot exhibit
// the uncomputed memory itself. It gates the observable that separates the two
// behaviours — whether the failure REACHES the caller.
TEST_CASE("Qwen3MoeDecodeGraph: a capture that FAILS reaches the caller") {
  const HfConfig c = TinyConfig();
  const Qwen3MoeWeights w = TinyWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);
  vllm::Qwen3MoeDecodeGraph graph(w, c, q, /*max_num_reqs=*/8);

  // Step 1, COLD: the eager pre-warm. No capture is attempted yet.
  graph.Step({11}, {0}, DecodeMeta(0), pool.attn_kv);
  REQUIRE_FALSE(graph.captured());

  // Step 2, WARM: the driver captures and the close REFUSES. The failure must
  // leave `Step`. A step that returns normally here is the defect.
  harness.backend().FailNextEndCapture();
  CHECK_THROWS_AS(graph.Step({12}, {1}, DecodeMeta(1), pool.attn_kv), std::runtime_error);
  CHECK_FALSE(graph.captured());

  // AND THE DRIVER RECOVERS. Without this arm the fix could be "throw forever"
  // and still look gated.
  graph.Step({13}, {2}, DecodeMeta(2), pool.attn_kv);
  const vllm::ForwardLogits after = graph.Step({14}, {3}, DecodeMeta(3), pool.attn_kv);
  CHECK(graph.captured());
  REQUIRE(after.on_device());
  const auto* p = static_cast<const float*>(after.device_tensor.data);
  REQUIRE(p != nullptr);
  for (int64_t i = 0; i < c.vocab_size; ++i) REQUIRE(std::isfinite(p[i]));
}
