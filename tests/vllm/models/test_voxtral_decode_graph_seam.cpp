// THE G2 REACHABILITY GATE for ENG-CUDAGRAPH-BREAK W3 (#1291, parent #1163),
// for `VoxtralDecodeGraph` — per `.agents/reachability.md` and the AGENTS.md
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
// VOXTRAL'S SHAPE, stated because it differs from its siblings. The single-
// sequence multimodal greedy driver only ever pads to S == 1
// (`max_num_reqs == 1`), so the captured region is the B == S == 1 bit-identical
// rebuild case, and the capturable region is `ForwardLastLogits` — the LAST row
// gathered before an untied lm_head, returning [1, vocab] — rather than the full
// [S, vocab] its siblings capture. The state machine it drives is the same one.
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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "decode_graph_seam_harness.h"
#include "vllm/model_executor/models/voxtral.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::HfConfig;
using vllm::PagedKvCache;
using vllm::Qwen3DenseWeights;
using vllm::v1::CommonAttentionMetadata;
using vllm_test::StaticGraphCpu;
using vt::DType;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

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

// The Voxtral TEXT backbone is Mistral/Llama: the shared dense attention with
// q_norm/k_norm left EMPTY, a SwiGLU MLP and an UNTIED lm_head.
HfConfig TinyConfig() {
  HfConfig c;
  c.num_hidden_layers = 2;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.rotary_dim = 16;
  c.intermediate_size = 128;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 1000000.0;
  c.vocab_size = 100;
  return c;
}

Qwen3DenseWeights TinyWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads, Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim, I = c.intermediate_size, V = c.vocab_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  Qwen3DenseWeights w;
  w.tie_word_embeddings = false;  // Voxtral's text head is UNTIED
  w.attention_bias = false;
  w.embed_tokens = MakeBf16({V, H}, /*nk=*/false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.5f);
  w.lm_head = MakeBf16({H, V}, /*nk=*/false, 3);  // Matmul-B [H,V]
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    vllm::Qwen3DenseLayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, /*nk=*/true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, /*nk=*/true, seed++);
    // q_norm / k_norm stay EMPTY: Voxtral text has no qk-norm branch.
    lw.mlp.gate_up_proj = MakeBf16({2 * I, H}, /*nk=*/true, seed++);
    lw.mlp.down_proj = MakeBf16({H, I}, /*nk=*/true, seed++);
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

TEST_CASE("G2: VoxtralDecodeGraph captures and replays THROUGH the vt seam") {
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);

  vt::ResetGraphBreakStats();
  // max_num_reqs == 1: the single-sequence multimodal greedy driver's value.
  vllm::VoxtralDecodeGraph graph(w, c, q, /*max_num_reqs=*/1);

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
// step taken through the driver's own EAGER arm.
//
// WHY THE DRIVER'S OWN EAGER ARM AND NOT A MODEL-LEVEL FORWARD. Voxtral's text
// forward (`ForwardLastLogits`) lives in `voxtral.cpp`'s anonymous namespace and
// is reachable only through the driver or through the multimodal greedy driver
// that owns the audio tower. `VoxtralDecodeGraph::Step` on a platform that does
// not admit capture runs exactly that forward and returns owning logits, which
// is the pre-migration behaviour this gate has to hold — G4's own words are
// "with capture disabled, the model produces output bit-identical to its
// pre-migration eager output".
TEST_CASE("G4: the seam changed no numerics on VoxtralDecodeGraph's capture step") {
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  vt::Queue q = Q();

  const std::vector<int32_t> tok0 = {11}, pos0 = {0};
  const std::vector<int32_t> tok1 = {12}, pos1 = {1};

  // EAGER arm: the stock CPU platform never answers `support_static_graph_mode()`
  // true, so the driver declines itself and runs the plain forward.
  std::vector<float> eager;
  {
    CachePool ref_pool(c, 2, 8);
    vllm::VoxtralDecodeGraph plain(w, c, q, /*max_num_reqs=*/1);
    plain.Step(tok0, pos0, DecodeMeta(0), ref_pool.attn_kv);
    const vllm::ForwardLogits fl = plain.Step(tok1, pos1, DecodeMeta(1), ref_pool.attn_kv);
    REQUIRE_FALSE(plain.captured());
    REQUIRE(fl.on_device());
    const auto* p = static_cast<const float*>(fl.device_tensor.data);
    REQUIRE(p != nullptr);
    eager.assign(p, p + c.vocab_size);
  }

  // CAPTURE arm.
  StaticGraphCpu harness;
  CachePool pool(c, 2, 8);
  vllm::VoxtralDecodeGraph graph(w, c, q, /*max_num_reqs=*/1);
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
TEST_CASE("VoxtralDecodeGraph: a capture that FAILS reaches the caller") {
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);
  vllm::VoxtralDecodeGraph graph(w, c, q, /*max_num_reqs=*/1);

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
  const auto* p = static_cast<const float*>(after.device_tensor.data);
  REQUIRE(p != nullptr);
  for (int64_t i = 0; i < c.vocab_size; ++i) REQUIRE(std::isfinite(p[i]));
}
