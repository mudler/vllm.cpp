// THE G2 REACHABILITY GATE for ENG-CUDAGRAPH-BREAK W1 (#1192, parent #1163),
// per `.agents/reachability.md` and the AGENTS.md "Nothing lands dead" rule.
//
// WHAT IT MEASURES, and why a unit test could not. `tests/vt/test_breakable_graph.cpp`
// proves the seam WORKS by constructing it by hand; that measures a class. This
// file proves something else entirely: that a REAL MODEL'S FORWARD reaches it.
// It drives `Qwen3DenseModel::Forward` — the same function the registered
// forward calls under `ModelRegistry::Forward`
// (`src/vllm/v1/worker/gpu/runner.cpp:1465`) — with a capture scope open, and
// counts the segments that come out. Delete the one `vt::GraphBreak` line at the
// dense attention entry in `src/vllm/model_executor/models/qwen3.cpp` and this
// file goes RED, because the forward then produces one segment instead of
// `num_hidden_layers + 1`.
//
// The recording backend is the harness adaptation stated once in
// `tests/vt/recording_capture_backend.h`: the model still computes on the CPU
// backend, and the scope's backend only records the capture call sequence, so
// this gate needs no GPU and runs in continuous integration.
//
// It also carries the G4 half of the spec's `## Gates`: the seam changes NO
// numerics. The logits from the scoped run are compared BIT FOR BIT against the
// unscoped one, which is what makes this stage reversible.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_graph_break.h"  // the writeback overload
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/dtype.h"
#include "vt/recording_capture_backend.h"
#include "vt/tensor.h"

// THE WRITEBACK BRANCH IS SELECTED, asserted at COMPILE TIME rather than
// believed. The destination form's `static_assert` already refuses a type with
// no `CopyOutput`, so a renamed, moved or shadowed overload is now a build
// failure at the model — but only this line says WHICH branch the production
// destination type takes, and it is the branch the D9 wrong-numerics path hides
// behind. Renaming `vllm::dense_attn::CopyOutput` reds here and at the model,
// where it used to leave both suites green.
static_assert(vt::detail::HasCopyOutput<std::optional<vllm::dense_attn::DBuf>>::value,
              "the qwen3 break point's destination must have a CopyOutput overload; without "
              "one every replay leaves the following segment reading capture-time data");

namespace {

using vllm::HfConfig;
using vllm::PagedKvCache;
using vllm::Qwen3DenseWeights;
using vllm::v1::CommonAttentionMetadata;
using vt::DType;
using vt_test::RecordingCaptureBackend;

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

HfConfig TinyConfig() {
  HfConfig c;
  c.num_hidden_layers = 2;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.rotary_dim = 16;  // partial_rotary_factor 1.0
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
  w.tie_word_embeddings = true;
  w.attention_bias = false;
  w.embed_tokens = MakeBf16({V, H}, /*nk=*/false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.5f);
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    vllm::Qwen3DenseLayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, /*nk=*/true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, /*nk=*/true, seed++);
    lw.attn.q_norm = MakeBf16({Dh}, false, seed++, 0.5f);
    lw.attn.k_norm = MakeBf16({Dh}, false, seed++, 0.5f);
    lw.mlp.gate_up_proj = MakeBf16({2 * I, H}, /*nk=*/true, seed++);
    lw.mlp.down_proj = MakeBf16({H, I}, /*nk=*/true, seed++);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// Single-sequence prefill paged KV cache (all layers full-attention).
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

// One prefill forward through the production model function, optionally with a
// capture scope open over `rec`.
std::vector<float> RunForward(const HfConfig& c, const Qwen3DenseWeights& w,
                              RecordingCaptureBackend* rec, vt::BreakableGraph* g) {
  const int64_t T = 5;
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillMeta(T, 8);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Q();
  if (rec == nullptr)
    return vllm::Qwen3DenseModel::Forward(tokens, positions, am, pool.attn_kv, w, c, q);
  vt::Queue rq = rec->CreateQueue();
  // The CAPTURING lane is this gate's precondition: with VLLM_CPP_CUDAGRAPH=0
  // exported the scope is inert by design, and an environment red must not read
  // as a code red.
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");
  vt::GraphCaptureScope scope(*rec, rq, *g);
  REQUIRE(scope.active());
  return vllm::Qwen3DenseModel::Forward(tokens, positions, am, pool.attn_kv, w, c, q);
}

}  // namespace

TEST_CASE("G2: the Qwen3 dense forward REACHES the break seam, once per layer") {
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  REQUIRE(c.num_hidden_layers == 2);

  RecordingCaptureBackend rec;
  vt::BreakableGraph g;
  const std::vector<float> scoped = RunForward(c, w, &rec, &g);

  // One break per decoder layer, so N layers yield N breaks and N+1 segments.
  // This is the number that goes to 0 and 1 when the call site is deleted.
  CHECK(g.break_count() == static_cast<size_t>(c.num_hidden_layers));
  CHECK(g.segment_count() == static_cast<size_t>(c.num_hidden_layers) + 1);
  CHECK(rec.Count("Begin") == c.num_hidden_layers + 1);
  CHECK(rec.Count("EndCaptureGraph") == c.num_hidden_layers + 1);
  // The invariant the seam ASSERTS — `GraphCaptureScope` refuses a container that
  // already holds a capture, and `Replay` refuses counts that disagree —
  // restated here where a real model produced them.
  CHECK(g.segment_count() == g.break_count() + 1);

  // G4: the seam changed no numerics. Bit for bit, not approximately.
  const std::vector<float> eager = RunForward(c, w, nullptr, nullptr);
  REQUIRE(scoped.size() == eager.size());
  REQUIRE(!scoped.empty());
  size_t differing = 0;
  for (size_t i = 0; i < scoped.size(); ++i)
    if (std::memcmp(&scoped[i], &eager[i], sizeof(float)) != 0) ++differing;
  CHECK(differing == 0);
  MESSAGE("logits compared bit for bit: " << scoped.size() << " values, " << differing
                                          << " differing");
  for (float v : scoped) REQUIRE(std::isfinite(v));
}

// Outside a scope — which is every production step today, until W2 migrates this
// model's decode driver onto the seam — the registered break point is a pure
// pass-through: the forward runs eager and makes ZERO backend capture calls.
TEST_CASE("G2: with no scope open the same forward captures nothing at all") {
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  RecordingCaptureBackend rec;

  const vt::GraphBreakStats before = vt::GetGraphBreakStats();
  const std::vector<float> out = RunForward(c, w, nullptr, nullptr);
  const vt::GraphBreakStats after = vt::GetGraphBreakStats();

  REQUIRE(!out.empty());
  CHECK(rec.Trace().empty());
  CHECK(after.segments_captured == before.segments_captured);
  CHECK(after.breaks_registered == before.breaks_registered);
  // The break point was still REACHED, once per layer — which is the counter
  // that distinguishes "the model routes into the seam" from "the seam exists".
  CHECK(after.break_points_reached - before.break_points_reached ==
        c.num_hidden_layers);
}
