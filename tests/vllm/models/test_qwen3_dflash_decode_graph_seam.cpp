// THE G2 REACHABILITY GATE for the DFlash DRAFT graph on the break-point capture
// seam. Row `ENG-CUDAGRAPH-BREAK` W5, issue #1335, parent #1163; spec
// `.agents/specs/eng-cudagraph-break.md` `## Gates` G2 and `## Work breakdown` W5.
//
// WHY THIS FILE HAS TO EXIST, and it is the same argument W3 made for each of its
// three drivers. A driver that kept its hand-rolled `BeginCapture` /
// `EndCaptureGraph` pair produces IDENTICAL logits, an IDENTICAL backend call log
// and an identical private replay count. `vt::GraphBreakStats::segments_captured`
// moves only when a `vt::GraphCaptureScope` closes a segment, and `replays` only
// inside `vt::BreakableGraph::Replay`, so those two counters are the ONLY
// observables that separate "captured a graph" from "captured a graph THROUGH THE
// SEAM". Nothing else in this tree can see the difference, which is exactly why a
// migration without its own gate is a refactor nobody can prove happened.
//
// WHY THE DFLASH DRIVER IS THE ONE SINGLE-SHAPE DRIVER THAT CAN BE GATED HERE.
// The other two W5 drivers refuse a CPU queue before they reach their capture:
// DeepSeek V4's `CanRunResidentDecode` returns false for
// `device.type == kCPU` and again unless the four CUDA-registered V4 kernel
// families are present, and Laguna's whole capture class is compiled only under
// `VT_MARLIN_NVFP4`, which needs a CUDA build on a marlin-nvfp4 architecture.
// The DFlash draft graph's admission predicate names neither a device type nor a
// kernel registry — it is `VT_DFLASH_GRAPH` plus
// `Backend::SupportsGraphCapture()` plus `Platform::support_static_graph_mode()`
// — so the shared harness's two swapped registries are enough to reach it. The
// spec's `## Owed` records the other two as owed on hardware rather than
// implying this file covers them.
//
// WHAT THIS HARNESS CANNOT SEE, named rather than claimed away: CPU kernels are
// direct calls, so a "captured" region EXECUTES eagerly here and a "replay"
// recomputes nothing. This gate holds the ROUTING and the capture step's
// numerics; that a replayed segment reproduces the eager forward is G1, it needs
// a real device, and `## Owed` records it.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/dtype.h"

#include "decode_graph_seam_harness.h"

using namespace vllm;

namespace {

vt::Queue Cpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// The same deterministic synthetic weights `tests/vllm/v1/spec_decode/
// test_dflash_propose.cpp` builds, so this gate's model is the one the existing
// device-KV bit-exactness case already pins rather than a second invention.
OwnedTensor MkBf16(const std::vector<int64_t>& shape, double seed, double amp, bool nk) {
  OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.rank = static_cast<int>(shape.size());
  t.nk = nk;
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= t.shape[i];
  }
  t.bytes.resize(static_cast<size_t>(n) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (int64_t i = 0; i < n; ++i)
    p[i] = vt::F32ToBF16(static_cast<float>(amp * std::sin(seed + 0.7 * static_cast<double>(i))));
  return t;
}

struct Dims {
  int64_t H = 4, Hq = 2, Hkv = 1, Dh = 2, I = 6, vocab = 8, layers = 2, taps = 2;
};

HfConfig MakeConfig(const Dims& dm) {
  HfConfig c;
  c.hidden_size = dm.H;
  c.num_attention_heads = dm.Hq;
  c.num_key_value_heads = dm.Hkv;
  c.head_dim = dm.Dh;
  c.rotary_dim = dm.Dh;
  c.rope_theta = 10000.0;
  c.intermediate_size = dm.I;
  c.vocab_size = dm.vocab;
  c.num_hidden_layers = dm.layers;
  c.rms_norm_eps = 1e-6;
  c.sliding_window = 64;
  c.layer_types = {"sliding_attention", "full_attention"};
  c.raw = nlohmann::json::object();
  c.raw["dflash_config"] = {{"mask_token_id", 7}};
  return c;
}

Qwen3DFlashWeights MakeWeights(const Dims& dm) {
  Qwen3DFlashWeights w;
  w.num_taps = dm.taps;
  w.mask_token_id = 7;
  w.draft_vocab_size = dm.vocab;
  const int64_t qdim = dm.Hq * dm.Dh, kdim = dm.Hkv * dm.Dh;
  w.embed_tokens = MkBf16({dm.vocab, dm.H}, 0.1, 0.3, false);
  w.fc = MkBf16({dm.H, dm.H * dm.taps}, 0.2, 0.2, true);
  w.hidden_norm = MkBf16({dm.H}, 0.3, 0.5, false);
  w.final_norm = MkBf16({dm.H}, 0.4, 0.5, false);
  w.lm_head = MkBf16({dm.vocab, dm.H}, 0.5, 0.3, true);
  const std::vector<Qwen3DFlashLayerAttnMode> modes = {{true, 64}, {false, 0}};
  for (int64_t l = 0; l < dm.layers; ++l) {
    Qwen3DFlashLayerWeights lw;
    const double s = 1.0 + static_cast<double>(l);
    lw.input_layernorm = MkBf16({dm.H}, s + 0.01, 0.5, false);
    lw.post_attention_layernorm = MkBf16({dm.H}, s + 0.02, 0.5, false);
    lw.qkv_proj = MkBf16({qdim + 2 * kdim, dm.H}, s + 0.03, 0.25, true);
    lw.o_proj = MkBf16({dm.H, qdim}, s + 0.04, 0.25, true);
    lw.q_norm = MkBf16({dm.Dh}, s + 0.05, 0.5, false);
    lw.k_norm = MkBf16({dm.Dh}, s + 0.06, 0.5, false);
    lw.gate_up_proj = MkBf16({2 * dm.I, dm.H}, s + 0.07, 0.2, true);
    lw.down_proj = MkBf16({dm.H, dm.I}, s + 0.08, 0.2, true);
    lw.attn_mode = modes[static_cast<size_t>(l)];
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// One context feature row per position, deterministic in the position.
std::vector<float> Ctx(int64_t rows, int64_t H) {
  std::vector<float> c(static_cast<size_t>(rows * H));
  for (size_t i = 0; i < c.size(); ++i)
    c[i] = 0.2f * static_cast<float>(std::sin(0.13 * static_cast<double>(i) + 0.4));
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// G2 — the DFlash draft graph is captured and replayed THROUGH THE SEAM.
// ---------------------------------------------------------------------------
//
// The step ladder mirrors the production propose loop: the store is appended to
// once per accepted token and `ForwardBlockLogitsWithDeviceKV` runs the (1+k)
// block per propose. The driver's own state machine is cold-with-capture on the
// FIRST propose (its warm pass and its capture happen in one step, because a
// full target verify runs between two draft steps and perturbs the shared pool),
// then replay on every later one. So after N proposes the seam must report ONE
// closed segment and N-1 replays.
TEST_CASE("dflash draft graph: the capture and every replay go through vt::BreakableGraph") {
  vllm_test::StaticGraphCpu graph_cpu;  // capture-capable CPU backend + static-graph platform

  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  const int64_t H = dm.H;

  auto store = Qwen3DFlashModel::MakeDeviceKVStore(cfg, q);
  const std::vector<float> ctx = Ctx(3, H);
  Qwen3DFlashModel::AppendContextKVDevice(*store, ctx, {0, 1, 2}, w, cfg, q);
  REQUIRE(Qwen3DFlashModel::DeviceKVNumCtx(*store) == 3);

  std::vector<DflashDeviceKVStore*> stores = {store.get()};
  const std::vector<int32_t> ids = {2, 7, 7};   // anchor + two MASK rows: Tq = 1 + k
  const std::vector<int32_t> pos = {3, 4, 5};
  const std::vector<int32_t> block_cu = {0, 3};
  const std::vector<int32_t> ctx_cu = {0, 3};

  vt::ResetGraphBreakStats();

  // PROPOSE 1 — the driver's cold step: it runs the eager warm pass, then opens
  // the seam's scope over the identical forward and files the segment.
  const std::vector<float> first = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
      stores, ctx_cu, ids, pos, block_cu, w, cfg, q);
  REQUIRE(first.size() == static_cast<size_t>(3) * dm.vocab);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    // ONE segment closed by a scope, and the mode is FULL. Both are asserted,
    // and `full_scopes` is not decoration: W3 measured that a `kFull` ->
    // `kPiecewise` flip in a migrated driver compiled clean and left the whole
    // gate green, because the mode was unobservable from outside the driver.
    CHECK(s.segments_captured == 1);
    CHECK(s.full_scopes == 1);
    CHECK(s.piecewise_scopes == 0);
    // NO BREAK POINT IS REGISTERED, and that is a decision rather than an
    // omission. Under `kFull` a `vt::GraphBreak` takes the same pass-through arm
    // it takes outside a scope, so registering one here would land machinery no
    // gate could exercise. The DFlash draft forward reaches no `GraphBreak` site
    // at all today, which is what this reads.
    CHECK(s.breaks_registered == 0);
    CHECK(s.replays == 0);  // the cold step's own output is its eager warm pass
  }

  // PROPOSE 2 and 3 — replays, and they must go through the CONTAINER. A driver
  // that called `Backend::ReplayGraph` on a raw handle would leave the backend
  // log identical and this counter at zero.
  for (int i = 0; i < 2; ++i) {
    const std::vector<float> again = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
        stores, ctx_cu, ids, pos, block_cu, w, cfg, q);
    REQUIRE(again.size() == first.size());
  }
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 1);  // captured ONCE for this block width
    CHECK(s.replays == 2);
    CHECK(s.full_scopes == 1);
  }
}

// ---------------------------------------------------------------------------
// G4 — the capture step's logits are bit-identical to the eager arm.
// ---------------------------------------------------------------------------
//
// This is what makes the migration reversible. `VT_DFLASH_GRAPH=0` selects the
// driver's OWN eager paged path, over the same store and the same weights, and
// the two must agree to the bit. Exact equality, because the graph lane is not
// an approximation of the eager one: it is the same kernels over the same
// persistent buffers.
TEST_CASE("dflash draft graph: the seam capture step is bit-identical to the eager arm") {
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  const int64_t H = dm.H;
  const std::vector<float> ctx = Ctx(3, H);
  const std::vector<int32_t> ids = {2, 7, 7};
  const std::vector<int32_t> pos = {3, 4, 5};
  const std::vector<int32_t> block_cu = {0, 3};
  const std::vector<int32_t> ctx_cu = {0, 3};

  // EAGER: no capture-capable backend, so `graph_ok` is false and the driver
  // takes its own eager paged lane.
  auto eager_store = Qwen3DFlashModel::MakeDeviceKVStore(cfg, q);
  Qwen3DFlashModel::AppendContextKVDevice(*eager_store, ctx, {0, 1, 2}, w, cfg, q);
  std::vector<DflashDeviceKVStore*> eager_stores = {eager_store.get()};
  const std::vector<float> eager = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
      eager_stores, ctx_cu, ids, pos, block_cu, w, cfg, q);

  // SEAM: the identical call under the capture-capable registries.
  std::vector<float> captured;
  {
    vllm_test::StaticGraphCpu graph_cpu;
    vt::ResetGraphBreakStats();
    auto store = Qwen3DFlashModel::MakeDeviceKVStore(cfg, q);
    Qwen3DFlashModel::AppendContextKVDevice(*store, ctx, {0, 1, 2}, w, cfg, q);
    std::vector<DflashDeviceKVStore*> stores = {store.get()};
    captured = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(stores, ctx_cu, ids, pos,
                                                               block_cu, w, cfg, q);
    REQUIRE(vt::GetGraphBreakStats().segments_captured == 1);
  }

  REQUIRE(captured.size() == eager.size());
  size_t differing = 0;
  for (size_t i = 0; i < captured.size(); ++i)
    if (captured[i] != eager[i]) ++differing;
  CHECK(differing == 0);
}

// ---------------------------------------------------------------------------
// THE FAILED-CAPTURE ARM — a capture that was abandoned must NOT return its
// buffer.
// ---------------------------------------------------------------------------
//
// Under stream capture nothing between `BeginCapture` and a throwing
// `EndCaptureGraph` executed: the kernels were RECORDED, not run. So the logits
// buffer the abandoned region produced holds whatever the pool last left there,
// and downloading it would hand the speculator uncomputed device memory as its
// draft. No fault, and a token gate cannot see it, because a draft the target
// rejects looks exactly like a bad draft. The pre-W5 driver rethrew after
// draining the stream and so must this one.
TEST_CASE("dflash draft graph: an abandoned capture propagates instead of returning garbage") {
  vllm_test::StaticGraphCpu graph_cpu;

  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  const std::vector<float> ctx = Ctx(3, dm.H);
  auto store = Qwen3DFlashModel::MakeDeviceKVStore(cfg, q);
  Qwen3DFlashModel::AppendContextKVDevice(*store, ctx, {0, 1, 2}, w, cfg, q);
  std::vector<DflashDeviceKVStore*> stores = {store.get()};
  const std::vector<int32_t> ids = {2, 7, 7};
  const std::vector<int32_t> pos = {3, 4, 5};
  const std::vector<int32_t> block_cu = {0, 3};
  const std::vector<int32_t> ctx_cu = {0, 3};

  graph_cpu.backend().FailNextEndCapture();
  CHECK_THROWS(Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(stores, ctx_cu, ids, pos,
                                                                block_cu, w, cfg, q));

  // AND THE STORE RECOVERS. The drain reset the container, so the very next
  // propose captures cleanly rather than inheriting a poisoned state — which is
  // the half a `throw` alone would not give.
  vt::ResetGraphBreakStats();
  const std::vector<float> after = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
      stores, ctx_cu, ids, pos, block_cu, w, cfg, q);
  CHECK(after.size() == static_cast<size_t>(3) * dm.vocab);
  CHECK(vt::GetGraphBreakStats().segments_captured == 1);
}
