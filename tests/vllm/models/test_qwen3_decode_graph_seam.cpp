// THE G2 REACHABILITY GATE for ENG-CUDAGRAPH-BREAK W2 (#1261, parent #1163),
// per `.agents/reachability.md` and the AGENTS.md "Nothing lands dead" rule.
//
// WHAT W1 LEFT OPEN, and what this file closes. W1 (#1192) landed the seam and
// registered ONE break point on the dense attention entry, and its own record
// named the staged slice: `GraphCaptureScope` and `BreakableGraph` were
// constructed by the GATES and by nothing else, because no driver opened a
// scope. `tests/vllm/models/test_qwen3_break_point.cpp` proves the model REACHES
// the break point; it opens the scope itself. This file proves the other half —
// that the production DRIVER, `Qwen3DenseDecodeGraph::Step`, captures and
// replays THROUGH the seam rather than through its own hand-rolled
// `BeginCapture`/`EndCaptureGraph`/`ReplayGraph`/`DestroyGraph` sequence.
//
// WHY THE SEAM COUNTERS AND NOT THE BACKEND LOG. A driver that called
// `Backend::ReplayGraph` directly would leave a backend log IDENTICAL to one the
// seam produces, because the seam bottoms out in the same call. `segments_captured`
// moves only when a `vt::GraphCaptureScope` closes a segment, and `replays` moves
// only inside `vt::BreakableGraph::Replay`, so those two numbers separate "the
// driver captured a graph" from "the driver captured a graph THROUGH THE SEAM".
// That is exactly the distinction G2 exists to make, and it is why the spec's G3
// observability counters are load-bearing rather than decoration.
//
// THE MUTATION THIS FILE ANSWERS TO: in a scratch copy, replace the driver's
// `vt::GraphCaptureScope` + `BreakableGraph::Replay` with the raw backend calls
// they replaced. The model still produces identical logits and the backend log
// is unchanged, so every OTHER gate in this tree stays green — and the three
// counter assertions below go to zero and turn this file RED.
//
// THE HARNESS, stated once. This box and continuous integration have no NVIDIA
// GPU, and the driver admits itself only where `support_static_graph_mode()` and
// `Backend::SupportsGraphCapture()` are both true — CUDA and Tenstorrent, never
// CPU. So the two seams the driver ASKS are swapped for the duration of the
// case: a backend that delegates every memory and compute-support call to the
// real CPU backend and implements the capture vocabulary by logging, and a
// platform that delegates everything to the real CPU platform and answers
// `support_static_graph_mode()` true. Both are restored in a destructor.
//
// WHAT THAT HARNESS CANNOT SEE, named rather than claimed away. CPU kernels are
// direct function calls, not backend submissions, so a "captured" region here
// EXECUTES eagerly and a "replay" recomputes nothing. This file therefore gates
// the ROUTING and the capture step's numerics; it does NOT gate that a replayed
// segment reproduces the eager forward. That is G1, it needs a real device, and
// `.agents/specs/eng-cudagraph-break.md` `## Owed` records it as owed rather
// than implying it ran here.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/model_executor/models/qwen3.h"
#include "vllm/platforms/interface.h"
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
using vt::DType;

// --- The two swapped seams -------------------------------------------------

// Delegates every real operation to the CPU backend and implements the capture
// vocabulary (`include/vt/backend.h:208-222`) by logging. It must delegate and
// not simulate, because the model's arithmetic has to be the REAL CPU
// arithmetic for the capture-step bit-exactness assertion below to mean
// anything.
class CaptureCapableCpuBackend final : public vt::Backend {
 public:
  explicit CaptureCapableCpuBackend(vt::Backend& inner) : inner_(inner) {}

  void* Alloc(size_t bytes) override { return inner_.Alloc(bytes); }
  void Free(void* p) override { inner_.Free(p); }
  void Memset(vt::Queue& q, void* p, int v, size_t bytes) override {
    inner_.Memset(q, p, v, bytes);
  }
  void Copy(vt::Queue& q, void* dst, const void* src, size_t bytes) override {
    inner_.Copy(q, dst, src, bytes);
  }
  vt::Queue CreateQueue() override { return inner_.CreateQueue(); }
  void DestroyQueue(vt::Queue& q) override { inner_.DestroyQueue(q); }
  void Synchronize(vt::Queue& q) override { inner_.Synchronize(q); }
  void FlushPending() override { inner_.FlushPending(); }
  bool UnifiedMemory() const override { return inner_.UnifiedMemory(); }
  bool DeviceMemoryIsHostAddressable() const override {
    return inner_.DeviceMemoryIsHostAddressable();
  }
  bool SupportsAsyncSampledTokenReadback() const override {
    return inner_.SupportsAsyncSampledTokenReadback();
  }
  bool SupportsCompressedConvState() const override {
    return inner_.SupportsCompressedConvState();
  }
  bool SupportsCompressedGdnState() const override {
    return inner_.SupportsCompressedGdnState();
  }
  bool SupportsAuxStream() const override { return inner_.SupportsAuxStream(); }

  bool SupportsGraphCapture() const override { return true; }
  void BeginCapture(vt::Queue&) override { log_.push_back("Begin"); }
  void* EndCaptureGraph(vt::Queue&) override {
    // Arm-once refusal, the shape of a real `cudaStreamEndCapture` returning
    // `cudaErrorStreamCaptureInvalidated` / `WrongThread`, or a failing
    // `cudaGraphInstantiate` — all three of which `Check()`
    // (`src/vt/cuda/cuda_backend.cu:50,229`) turns into a throw.
    if (fail_next_end_) {
      fail_next_end_ = false;
      throw std::runtime_error("capture-capable CPU backend: EndCaptureGraph refused");
    }
    log_.push_back("EndCaptureGraph");
    void* tag = std::malloc(1);
    tags_.push_back(tag);
    return tag;
  }
  void ReplayGraph(vt::Queue&, void*) override { log_.push_back("ReplayGraph"); }
  void DestroyGraph(void* g) override {
    if (g == nullptr) return;
    log_.push_back("DestroyGraph");
  }

  ~CaptureCapableCpuBackend() override {
    for (void* p : tags_) std::free(p);
  }

  void FailNextEndCapture() { fail_next_end_ = true; }

  size_t Count(std::string_view what) const {
    size_t n = 0;
    for (const auto& e : log_)
      if (e == what) ++n;
    return n;
  }

 private:
  vt::Backend& inner_;
  bool fail_next_end_ = false;
  std::vector<std::string> log_;
  std::vector<void*> tags_;
};

// Answers `support_static_graph_mode()` true and delegates the rest, so the
// driver's OWN admission predicate is the thing under test rather than a
// predicate this file reimplements.
class StaticGraphCpuPlatform final : public vllm::platforms::Platform {
 public:
  StaticGraphCpuPlatform(vllm::platforms::Platform& inner, vt::Backend& backend)
      : inner_(inner), backend_(backend) {}

  vt::DeviceType device_type() const override { return inner_.device_type(); }
  vt::Backend& backend() const override { return backend_; }
  vllm::platforms::DeviceCapability get_device_capability() const override {
    return inner_.get_device_capability();
  }
  bool supports_fp8() const override { return inner_.supports_fp8(); }
  bool cutlass_fp4_supported() const override { return inner_.cutlass_fp4_supported(); }
  bool opaque_attention_op() const override { return inner_.opaque_attention_op(); }
  bool is_integrated_gpu() const override { return inner_.is_integrated_gpu(); }
  bool needs_weight_staging() const override { return inner_.needs_weight_staging(); }
  bool supports_fa2_attention() const override { return inner_.supports_fa2_attention(); }
  std::vector<DType> supported_dtypes() const override { return inner_.supported_dtypes(); }
  vllm::platforms::ResidencyPolicy residency_policy() const override {
    return inner_.residency_policy();
  }
  bool supports_model_architecture(std::string_view a) const override {
    return inner_.supports_model_architecture(a);
  }
  std::vector<std::string> get_attn_backend_priority(
      const vllm::platforms::AttnSelectorConfig& cfg) const override {
    return inner_.get_attn_backend_priority(cfg);
  }
  std::vector<std::string> get_mla_prefill_backend_priority() const override {
    return inner_.get_mla_prefill_backend_priority();
  }

  // THE ONE ANSWER THAT DIFFERS.
  bool support_static_graph_mode() const override { return true; }

 private:
  vllm::platforms::Platform& inner_;
  vt::Backend& backend_;
};

// Swaps both registries for the life of the object and puts back exactly what
// was there. A doctest binary runs every case in one process, so a leaked
// override would make an unrelated case observe a CPU device that captures.
class StaticGraphCpu {
 public:
  StaticGraphCpu()
      : prev_backend_(&vt::GetBackend(vt::DeviceType::kCPU)),
        prev_platform_(&vllm::platforms::GetPlatform(vt::DeviceType::kCPU)),
        backend_(*prev_backend_),
        platform_(*prev_platform_, backend_) {
    vt::RegisterBackend(vt::DeviceType::kCPU, &backend_);
    vllm::platforms::RegisterPlatform(vt::DeviceType::kCPU, &platform_);
  }
  ~StaticGraphCpu() {
    vt::RegisterBackend(vt::DeviceType::kCPU, prev_backend_);
    vllm::platforms::RegisterPlatform(vt::DeviceType::kCPU, prev_platform_);
  }
  CaptureCapableCpuBackend& backend() { return backend_; }

 private:
  vt::Backend* prev_backend_;
  vllm::platforms::Platform* prev_platform_;
  CaptureCapableCpuBackend backend_;
  StaticGraphCpuPlatform platform_;
};

// --- The tiny model, same shape as tests/vllm/models/test_qwen3_forward.cpp --

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

// The gate. Three decode steps at one shape drive the driver's whole state
// machine: cold (eager pre-warm), warm (CAPTURE), captured (REPLAY).
TEST_CASE("G2: Qwen3DenseDecodeGraph captures and replays THROUGH the vt seam") {
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);

  vt::ResetGraphBreakStats();
  vllm::Qwen3DenseDecodeGraph graph(w, c, q, /*max_num_reqs=*/8);

  // Step 1, COLD: an eager pre-warm step. Nothing is captured yet.
  graph.Step({11}, {0}, DecodeMeta(0), pool.attn_kv);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 0);
    CHECK(s.replays == 0);
    CHECK_FALSE(graph.captured());
  }

  // Step 2, WARM: the driver captures. This is the assertion that fails when the
  // driver keeps its own BeginCapture/EndCaptureGraph pair: `segments_captured`
  // moves ONLY when a vt::GraphCaptureScope closes a segment.
  graph.Step({12}, {1}, DecodeMeta(1), pool.attn_kv);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 1);
    CHECK(graph.captured());
    // FULL mode, mirroring vLLM's `FULL_AND_PIECEWISE` decode arm
    // (`vllm/config/compilation.py:63,65-66,630-632` @ pin `5559679229`): a
    // decode batch gets ONE graph and the attention break points do not split
    // it. `break_points_reached` still moves, which is what separates a
    // full-mode capture from a DELETED break point.
    CHECK(s.breaks_registered == 0);
    CHECK(s.break_points_reached > 0);
  }

  // Step 3, CAPTURED: the driver replays. `replays` moves ONLY inside
  // vt::BreakableGraph::Replay.
  graph.Step({13}, {2}, DecodeMeta(2), pool.attn_kv);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.replays >= 1);
    CHECK(graph.replay_count() >= 1);
  }

  // Step 4: a SECOND replay of the same captured graph. NOT because one replay
  // could have dropped a tail — this container is `kFull`, so `break_count()`
  // is 0 and there is no tail to drop. What a second replay pins is that the
  // driver's REPLAY branch is re-entrant: the third step is the first that takes
  // it, so a branch that re-captured, reset the container, or left `warm` set
  // would still look correct after one step and only diverge on the next.
  graph.Step({14}, {3}, DecodeMeta(3), pool.attn_kv);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.replays >= 2);
    // Still ONE captured segment: a replay must never re-enter capture.
    CHECK(s.segments_captured == 1);
  }

  // The backend saw exactly one capture pair, and the seam owns its release:
  // `BreakableGraph`'s destructor routes every handle through
  // Backend::DestroyGraph, which is what lets #1162 interpose later (spec D4).
  CHECK(harness.backend().Count("Begin") == 1);
  CHECK(harness.backend().Count("EndCaptureGraph") == 1);
  CHECK(harness.backend().Count("ReplayGraph") >= 2);
}

// G4 for the driver: with the seam in place, the CAPTURE step's logits are
// bit-identical to the same step run through the plain eager forward. This is
// what makes the migration reversible — and it is NOT G1, which needs a real
// device because a CPU "replay" recomputes nothing.
TEST_CASE("G4: the seam changed no numerics on the driver's capture step") {
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  vt::Queue q = Q();

  const std::vector<int32_t> tok0 = {11}, pos0 = {0};
  const std::vector<int32_t> tok1 = {12}, pos1 = {1};

  // Eager reference: the same two steps through Qwen3DenseModel::Forward, on the
  // stock CPU platform where the driver does not exist.
  CachePool ref_pool(c, 2, 8);
  vllm::Qwen3DenseModel::Forward(tok0, pos0, DecodeMeta(0), ref_pool.attn_kv, w, c, q);
  const std::vector<float> eager =
      vllm::Qwen3DenseModel::Forward(tok1, pos1, DecodeMeta(1), ref_pool.attn_kv, w, c, q);
  REQUIRE(eager.size() == static_cast<size_t>(c.vocab_size));

  StaticGraphCpu harness;
  CachePool pool(c, 2, 8);
  vllm::Qwen3DenseDecodeGraph graph(w, c, q, /*max_num_reqs=*/8);
  graph.Step(tok0, pos0, DecodeMeta(0), pool.attn_kv);
  const vllm::ForwardLogits captured =
      graph.Step(tok1, pos1, DecodeMeta(1), pool.attn_kv);
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

// THE ASYNC DECLINE, GATED — W2 KEEPS IT, and this case is what stops the
// migration from dropping it silently in either direction.
//
// `DenseDecodeGraphForward` (`src/vllm/model_executor/models/qwen3.cpp`) returns
// `std::nullopt` whenever `input.device_token_ids` is live, on its own measured
// battery: depth-1 graph ON PASS 78/78, depth-2 graph OFF PASS 82/82, depth-2
// graph ON FAIL with slots 1-3 degenerate (#323, #1179). The cause is that
// `Step()` replays against the HOST `token_ids` while the async combine has
// patched the DEVICE ids, and the named fix is to read the identifiers at REPLAY
// time from a stable device buffer — which exists in exactly one sibling driver,
// as `StepDevInputs` (`qwen3_5.cpp:3894`, 41 lines there and 0 here).
//
// W2 DOES NOT REMOVE THE DECLINE, and the reason is the spec's own work
// breakdown rather than appetite: `## Work breakdown` W4 is where the persistent
// device input path becomes a seam capability instead of one driver's private
// code, and that is what makes it available to the four drivers that lack it.
// Reproducing `StepDevInputs` inside this driver at W2 would be the tenth
// hand-rolled copy of the capability this row exists to stop copying. The spec's
// `## Owed` names W4 as the owner.
//
// So the mitigation stands, and it is now EXECUTABLE rather than a comment. Both
// arms are asserted, because a gate that only checks the decline would stay
// green if the routing broke entirely.
TEST_CASE("The async device-token decline still holds, and only for that condition") {
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);

  const std::vector<int32_t> tokens = {11};
  const std::vector<int32_t> positions = {0};
  const CommonAttentionMetadata am = DecodeMeta(0);
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::GdnStateCache> gdn_state;
  const std::vector<int32_t> no_gather;

  StaticGraphCpu harness;
  std::unique_ptr<vllm::Qwen3DenseDecodeGraph> graph;

  vllm::ModelForwardInput in{tokens,   positions, am,       gdn_meta, pool.attn_kv,
                             gdn_state, c,        q,        no_gather};
  in.num_reqs = 1;
  in.pure_decode = true;
  in.gdn_state_slots = 8;

  // ARM 1, the ROUTING: with no device mirror the step reaches the driver.
  REQUIRE(in.device_token_ids == nullptr);
  CHECK(DenseDecodeGraphForward(graph, w, in).has_value());
  REQUIRE(graph != nullptr);

  // ARM 2, the DECLINE: with the mirror live the step is refused and the caller
  // falls back to its proven-correct eager forward.
  const int32_t dev_ids[1] = {11};
  in.device_token_ids = dev_ids;
  CHECK_FALSE(DenseDecodeGraphForward(graph, w, in).has_value());
}

// THE CAPTURE-FAILURE GATE, and it is the one W2's first head did not have.
//
// WHAT GOES WRONG WITHOUT IT. `~GraphCaptureScope` must swallow a throwing
// `EndCaptureGraph` — a destructor that propagates terminates — so after a
// FAILED capture the container reports exactly what an INERT scope reports:
// `captured() == false`. The first W2 head read that one bit and returned the
// value `ForwardLayers` had produced. On a non-capturing backend that value is a
// real eager result. On CUDA it is not: `Backend::EndCaptureGraph`
// (`src/vt/cuda/cuda_backend.cu:229`, `Check()` at `:50`) throws when
// `cudaStreamEndCapture` returns `cudaErrorStreamCaptureInvalidated` or
// `WrongThread`, or when `cudaGraphInstantiate` fails — and under stream capture
// NOTHING between `BeginCapture` and the throw executed. Every kernel was
// RECORDED. So the buffer held whatever the `DevicePool` last left there, and
// the step returned pool-recycled memory as its logits: silently wrong tokens,
// no fault, and a token gate cannot see it. The PRE-W2 driver propagated,
// because its `s.graph = b.EndCaptureGraph(...)` was unguarded.
//
// WHAT THE DRIVER DOES NOW. `vt::BreakableGraph::capture_error()` separates the
// two states the drain had merged, and the driver RETHROWS the runtime's own
// exception rather than inventing a recovery for a stream whose state the
// failure has not described. That restores pre-W2 behaviour exactly.
//
// WHAT THIS HARNESS CANNOT SEE, named rather than claimed away: a CPU "capture"
// EXECUTES, so `lg` here holds real numbers and this file cannot exhibit the
// uncomputed memory itself. It gates the observable that separates the two
// behaviours — whether the failure REACHES the caller — which is exactly the
// mutation shape the fresh review used.
TEST_CASE("A capture that FAILS reaches the caller; the step never returns it as logits") {
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);
  vllm::Qwen3DenseDecodeGraph graph(w, c, q, /*max_num_reqs=*/8);

  // Step 1, COLD: the eager pre-warm. No capture is attempted yet.
  graph.Step({11}, {0}, DecodeMeta(0), pool.attn_kv);
  REQUIRE_FALSE(graph.captured());

  // Step 2, WARM: the driver captures, and the close REFUSES. The failure must
  // leave `Step`. A step that returns normally here is the defect: its result is
  // whatever the pool held.
  harness.backend().FailNextEndCapture();
  CHECK_THROWS_AS(graph.Step({12}, {1}, DecodeMeta(1), pool.attn_kv), std::runtime_error);
  CHECK_FALSE(graph.captured());

  // AND THE DRIVER RECOVERS. The failed capture is not sticky: the slot went
  // back to cold, so the next step is eager and the one after it captures
  // cleanly. Without this arm the fix could be "throw forever" and look gated.
  graph.Step({13}, {2}, DecodeMeta(2), pool.attn_kv);
  const vllm::ForwardLogits after =
      graph.Step({14}, {3}, DecodeMeta(3), pool.attn_kv);
  CHECK(graph.captured());
  REQUIRE(after.on_device());
  const auto* p = static_cast<const float*>(after.device_tensor.data);
  REQUIRE(p != nullptr);
  for (int64_t i = 0; i < c.vocab_size; ++i) REQUIRE(std::isfinite(p[i]));
}
