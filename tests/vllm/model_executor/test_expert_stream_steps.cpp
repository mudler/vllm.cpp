// ENG-EXPERT-STREAM (#912, repairs #1091): the step clock, at every MoE entry
// point, and the one statistics line that always prints.
//
// WHY A SECOND WIRING BINARY. `test_expert_stream_wiring` asks whether the paged
// forward reaches the lane at all. It cannot ask these two questions, because
// both need a process whose step clock starts at zero and stays there: the
// singleton store and the once-read `VT_MOE_EXPERT_STREAM` are process-scoped,
// so a case that must observe `steps == 0` has to run before anything ends a
// step, and the reachability case ends three.
//
// WHAT IS UNDER TEST.
//
//   1. `ForwardLayers` is not the only MoE entry point, and its comment used to
//      say it was. `Qwen3_5Model::ForwardDense`, both MTP forwards and
//      `Qwen3_5ReplayLayer` all reach `ExpertMlpKq -> KqExpertSlice` and none of
//      them marked a step. A forward that takes slices and never ends its step
//      leaves every entry it acquired `protected_this_step` forever, which is
//      defect F1 with a smaller blast radius: on a draft+target pair the draft's
//      slots stay pinned across the target's forward and shrink the evictable
//      set for the whole run.
//
//      WHICH OF THE FOUR PRODUCTION ACTUALLY RUNS: one. Only
//      `Qwen3_5MTPModel::ForwardPaged` has a production caller (`runner.cpp:2183`
//      -> `spec_decode/mtp/speculator.cpp:107,262`), and that caller runs only
//      when a speculator is configured (`runner.cpp:2120`), so no
//      default-configuration run reaches any of the four. `Qwen3_5MTPModel::Forward`
//      is reached only through `ForwardLogitsHost`, which `qwen3_5_mtp.h:135`
//      calls a "standalone parity convenience" and which has no caller outside
//      `tests/`; `ForwardDense` and `Qwen3_5ReplayLayer` are parity references
//      the same way. An earlier revision of this comment called both MTP
//      forwards the production draft path (#1106 finding 2). So for three of the
//      four guards this binary is the ONLY driver there is: the cases below pin
//      the boundary, they do not demonstrate reach, and #1108 plus the spec's
//      `## Owed` carry that debt.
//
//      ONE FORWARD IS ONE STEP. That is the definition the cache is built
//      against, and it is why the draft gets its own step rather than sharing
//      the target's: the draft is a complete forward whose slices are finished
//      with when it returns, and folding it into the target's step would pin
//      them across a second forward for no benefit. Each case below asserts a
//      DELTA of exactly one, so ordering between cases cannot flatter it, and
//      an entry point that marked its step twice would fail just as loudly as
//      one that never marked it.
//
//   2. The statistics line has to print even when the run did nothing. It used
//      to be emitted only from `EndStep`, and only on a step that was a
//      multiple of `VT_MOE_EXPERT_STREAM_STATS_EVERY` — so the one run that
//      most needed it, the one where the step boundary is never reached, was
//      exactly the run that printed nothing at all. Both docs told an operator
//      to read `steps == 0` off a line that could not exist.
#include <stdlib.h>
#if !defined(_WIN32)
// The two questions about the statistics LINE need POSIX: one redirects stderr
// across the flush, the other runs this binary again as a child. The step-clock
// questions below need neither and are built everywhere — which they were NOT
// when this comment was first written. `::setenv` sat at namespace scope with
// no guard, and it is POSIX: MSVC's CRT has only `_putenv_s`, so the whole
// translation unit failed to compile there and none of the cases existed on
// Windows at all. `tests/CMakeLists.txt` adds this target unconditionally and
// `scripts/build-windows-release.ps1` configures `VLLM_CPP_BUILD_TESTS=ON`, so
// the only reason CI stayed quiet is that the Windows lanes already fail
// earlier, inside the product library, on #1068 and never reach a test
// translation unit. The repair is `vllm_test::SetEnv` from
// `support/test_env.h`, which is where the `_putenv_s` branch already lived.
//
// Streaming itself is a POSIX lane — `EnsureFile` refuses on _WIN32 by name —
// but the STEP CLOCK is not: it advances on the mapping-copy fallback too, so
// these cases have something to measure there.
#include <unistd.h>
#endif

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/expert_stream_model.h"
#include "support/test_env.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_moe_block.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"

using expert_stream_test::CachePool;
using expert_stream_test::MakeConfig;
using expert_stream_test::MakeKqMoe;
using expert_stream_test::MakeOwned;
using expert_stream_test::MakeWeights;
using expert_stream_test::PrefillAttnMeta;
using expert_stream_test::Q;
using vllm::HfConfig;
using vllm::Qwen3_5MoeWeights;
using vllm::Qwen3_5Model;
using vllm::Qwen3_5MTPKind;
using vllm::Qwen3_5MTPModel;
using vllm::Qwen3_5MTPWeights;
using vt::DType;

namespace {

// Same knobs as the reachability binary, and for the same reasons. The one that
// matters here is `STATS_EVERY=0`: it SILENCES the periodic line, so any
// statistics line this process emits can only have come from the final flush.
//
// Through `vllm_test::SetEnv` and not `::setenv`, which is what this file did
// and is the whole of the Windows defect: the shim in `support/test_env.h` is
// the one place the `_putenv_s` branch lives (#603), and a new env-flipping
// test is exactly what it says to use.
struct EnableExpertStreaming {
  EnableExpertStreaming() {
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM", "1");
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_SLOTS", "64");
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_SLOT_BYTES", "8192");
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_STATS_EVERY", "0");
    vllm_test::SetEnv("VT_QWEN35_GROUPED_MOE", "0");
  }
};
const EnableExpertStreaming kEnableExpertStreaming;

// This process was spawned by the teardown case below and must do the one thing
// that case measures — build a store and exit — and nothing else.
bool IsFlushChild() { return ::getenv("VT_ES_FLUSH_CHILD") != nullptr; }

int64_t Steps() { return vllm::detail::ExpertStreamSnapshot().steps; }

// A full-attention MoE layer, which is the only layer type an MTP head has
// (qwen3_5.cpp: "The MTP layer is always layer_type=full_attention").
vllm::Qwen3_5MoeLayerWeights MakeFullAttnMoeLayer(const HfConfig& c, uint64_t s) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads,
                Hkv = c.num_key_value_heads, Dh = c.head_dim;
  vllm::Qwen3_5MoeLayerWeights lw;
  lw.is_linear_attention = false;
  lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
  lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
  lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
  lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
  lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
  lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
  lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
  lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
  lw.moe = MakeKqMoe(c, s + 500);
  return lw;
}

Qwen3_5MTPWeights MakeMtpWeights(const HfConfig& c, uint64_t s) {
  const int64_t H = c.hidden_size;
  Qwen3_5MTPWeights w;
  w.kind = Qwen3_5MTPKind::kMoe;
  w.fc = MakeOwned(DType::kBF16, {H, 2 * H}, s + 1);
  w.fc.nk = true;  // raw torch Linear [H,2H], as the forward's precondition says
  w.pre_fc_norm_embedding = MakeOwned(DType::kBF16, {H}, s + 2);
  w.pre_fc_norm_hidden = MakeOwned(DType::kBF16, {H}, s + 3);
  w.final_norm = MakeOwned(DType::kBF16, {H}, s + 4);
  w.moe_layers.push_back(MakeFullAttnMoeLayer(c, s + 1000));
  return w;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// This case runs FIRST on purpose: it is the only place in this binary where
// `steps` is still 0, which is the state the whole finding is about.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("the final statistics line prints on a run whose step clock never advanced") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();

  // `RunMoeBlock` is the seam qwen3_moe.cpp composes the same MoE block
  // through, and it deliberately carries NO step guard: that model's own layer
  // driver owns the boundary (qwen3_moe.cpp:150). Driving it directly therefore
  // reproduces the exact production shape this line exists to report — expert
  // slices taken, no step ended — without having to break anything.
  const int64_t T = 2, H = c.hidden_size;
  std::vector<uint16_t> hidden(static_cast<size_t>(T * H));
  for (size_t i = 0; i < hidden.size(); ++i)
    hidden[i] = vt::F32ToBF16(expert_stream_test::RandV(7 + i));
  const vt::Tensor dh = vt::Tensor::Contiguous(
      hidden.data(), DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0}, {T, H});
  const vllm::MoeBlockOutput out =
      vllm::RunMoeBlock(q, w.layers[0].moe, c, dh, T);
  REQUIRE(out.storage != nullptr);

  const vllm::detail::ExpertStreamStats s = vllm::detail::ExpertStreamSnapshot();
  REQUIRE(s.active);      // a store was built, so there ARE numbers to print
  REQUIRE(s.fills > 0);   // and slices really were taken
  REQUIRE(s.steps == 0);  // and no step ended: the F1 signature, reproduced

  // The child's job ends here. Its remaining line has to come from teardown, so
  // it must not call the flush itself.
  if (IsFlushChild()) return;

#if !defined(_WIN32)
  // Capture stderr across the flush. Everything this process printed before now
  // (the one-off `[expert-stream] ON ...` banner) is outside the redirect, so a
  // statistics line inside it can only be the one under test.
  std::FILE* cap = std::tmpfile();
  REQUIRE(cap != nullptr);
  std::fflush(stderr);
  const int saved = ::dup(STDERR_FILENO);
  REQUIRE(saved >= 0);
  REQUIRE(::dup2(::fileno(cap), STDERR_FILENO) >= 0);

  vllm::detail::ExpertStreamFlushStats();

  std::fflush(stderr);
  REQUIRE(::dup2(saved, STDERR_FILENO) >= 0);
  ::close(saved);

  std::rewind(cap);
  std::string captured;
  char buf[512];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), cap)) > 0) captured.append(buf, n);
  std::fclose(cap);

  // EXACTLY ONE line, and it carries the zero. `stats_every_` is 0 here, which
  // silences the periodic report entirely, and `steps` is 0, which the periodic
  // report skips as well — so both of the early returns that made this line
  // unreachable are being crossed at once.
  size_t lines = 0;
  for (size_t at = captured.find("[expert-stream] steps=");
       at != std::string::npos;
       at = captured.find("[expert-stream] steps=", at + 1))
    ++lines;
  INFO("captured stderr: ", captured);
  CHECK(lines == 1);
  CHECK(captured.find("[expert-stream] steps=0 ") != std::string::npos);
  CHECK(captured.find(" fills=") != std::string::npos);
#endif  // !_WIN32
}

TEST_CASE("Qwen3_5Model::ForwardDense marks exactly one step") {
  if (IsFlushChild()) return;
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const std::vector<int32_t> ids = {5, 9, 2};
  const std::vector<int32_t> pos = {0, 1, 2};

  const int64_t before = Steps();
  const std::vector<float> logits =
      Qwen3_5Model::ForwardDense(ids, pos, w, c, q);
  REQUIRE(logits.size() ==
          static_cast<size_t>(ids.size()) * static_cast<size_t>(c.vocab_size));
  CHECK(Steps() - before == 1);
}

TEST_CASE("Qwen3_5MTPModel::Forward marks exactly one step") {
  if (IsFlushChild()) return;
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights target = MakeWeights(c);
  const Qwen3_5MTPWeights mtp = MakeMtpWeights(c, 4242);
  const Qwen3_5MTPModel model(mtp, target, c);
  vt::Queue q = Q();

  const int64_t T = 3, H = c.hidden_size;
  std::vector<uint16_t> th(static_cast<size_t>(T * H));
  for (size_t i = 0; i < th.size(); ++i)
    th[i] = vt::F32ToBF16(expert_stream_test::RandV(31 + i));
  const vt::Tensor target_hidden = vt::Tensor::Contiguous(
      th.data(), DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0}, {T, H});
  const std::vector<int32_t> ids = {1, 2, 3};
  const std::vector<int32_t> pos = {0, 1, 2};

  const int64_t before = Steps();
  const vllm::Qwen3_5MTPHiddenStates h =
      model.Forward(ids, pos, target_hidden, q);
  REQUIRE(h.storage != nullptr);
  CHECK(Steps() - before == 1);
}

TEST_CASE("Qwen3_5MTPModel::ForwardPaged marks exactly one step") {
  if (IsFlushChild()) return;
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights target = MakeWeights(c);
  const Qwen3_5MTPWeights mtp = MakeMtpWeights(c, 909);
  const Qwen3_5MTPModel model(mtp, target, c);
  vt::Queue q = Q();

  const int64_t T = 3, H = c.hidden_size;
  std::vector<uint16_t> th(static_cast<size_t>(T * H));
  for (size_t i = 0; i < th.size(); ++i)
    th[i] = vt::F32ToBF16(expert_stream_test::RandV(57 + i));
  const vt::Tensor target_hidden = vt::Tensor::Contiguous(
      th.data(), DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0}, {T, H});
  const std::vector<int32_t> ids = {4, 5, 6};
  const std::vector<int32_t> pos = {0, 1, 2};

  // The draft KV cache: one full-attention layer's worth, which is all an MTP
  // head has.
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/8);
  REQUIRE(!pool.attn_kv.empty());
  const std::vector<int32_t> blocks = {0};

  const int64_t before = Steps();
  const vllm::Qwen3_5MTPHiddenStates h =
      model.ForwardPaged(ids, pos, target_hidden,
                         PrefillAttnMeta(T, blocks, 8, 0), pool.attn_kv[0], q);
  REQUIRE(h.storage != nullptr);
  CHECK(Steps() - before == 1);
}

#if defined(__linux__)
TEST_CASE("the final statistics line is wired to process TEARDOWN") {
  // The case above proves the flush prints what it should when something calls
  // it. This one proves something calls it, which is the part an in-process
  // assertion cannot reach: the flush runs from a static destructor, after
  // doctest's main has returned.
  //
  // So it runs THIS BINARY again as a child with `VT_ES_FLUSH_CHILD` set. In
  // that mode every case returns early except the first, which builds a store,
  // takes slices, ends no step and — crucially — does NOT call the flush. Any
  // statistics line in the child's output therefore came from teardown, and
  // `VT_MOE_EXPERT_STREAM_STATS_EVERY=0` rules out the periodic report.
  //
  // /proc/self/exe rather than argv[0], because doctest's main owns argv and a
  // relative argv[0] would depend on the working directory ctest chose. It is
  // RESOLVED here rather than handed to the shell: `popen` runs `/bin/sh`, so a
  // literal /proc/self/exe in the command line names the SHELL and the child
  // would print nothing at all — which looks exactly like the defect.
  if (IsFlushChild()) return;

  char exe[4096];
  const ssize_t len = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(len > 0);
  exe[len] = '\0';
  const std::string cmd =
      std::string("VT_ES_FLUSH_CHILD=1 '") + exe + "' 2>&1";

  std::FILE* child = ::popen(cmd.c_str(), "r");
  REQUIRE(child != nullptr);
  std::string out;
  char buf[512];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), child)) > 0) out.append(buf, n);
  const int status = ::pclose(child);

  INFO("child output: ", out);
  CHECK(status == 0);

  // The child really ran, and really built a store. Without this a filter or a
  // crash that produced no output at all would read as "one line, absent",
  // which is the same shape as the defect.
  CHECK(out.find("[expert-stream] ON slots=") != std::string::npos);
  CHECK(out.find("0 failed") != std::string::npos);

  size_t lines = 0;
  for (size_t at = out.find("[expert-stream] steps=");
       at != std::string::npos;
       at = out.find("[expert-stream] steps=", at + 1))
    ++lines;
  CHECK(lines == 1);
  CHECK(out.find("[expert-stream] steps=0 ") != std::string::npos);

  // And it carries the STORE's numbers. The child filled slots, so a line
  // reporting `fills=0` would mean something other than the store printed it —
  // exactly what a well-meaning second teardown hook would produce, and it would
  // otherwise satisfy every assertion above.
  CHECK(out.find(" fills=0 ") == std::string::npos);
}
#endif  // __linux__

TEST_CASE("Qwen3_5ReplayLayer marks exactly one step") {
  if (IsFlushChild()) return;
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();

  const int64_t T = 2, H = c.hidden_size;
  std::vector<float> hidden_in(static_cast<size_t>(T * H));
  for (size_t i = 0; i < hidden_in.size(); ++i)
    hidden_in[i] = expert_stream_test::RandV(99 + i);
  const std::vector<int32_t> pos = {0, 1};

  const int64_t before = Steps();
  const std::vector<float> out =
      vllm::Qwen3_5ReplayLayer(w.layers[3], c, hidden_in, pos, T, q);
  REQUIRE(out.size() == hidden_in.size());
  CHECK(Steps() - before == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// The nesting refusal, which every case above depends on and none of them can
// reach.
//
// WHY IT NEEDED ITS OWN CASE. "One forward is one step, and the guard REFUSES
// to nest" was stated in the source, in the spec and in the pull request body,
// and deleting the `VT_CHECK` that implements it left BOTH focused binaries
// fully green — 6/6 and 4/4. That is the same shape as every other finding this
// row has carried: an asserted guarantee no gate could see. It is unreachable
// through production code by construction, because every forward that takes
// expert slices is a complete forward that no other one contains, so there is
// no legitimate call graph that nests one. `detail::ExpertStreamStepScope`
// exists for exactly this, and it forwards to the guard's own `Begin`/`End`
// rather than re-stating the flag, so what is measured here is the production
// boundary and not a copy of it.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("the step guard REFUSES to nest, and the refusal reaches a real forward") {
  if (IsFlushChild()) return;
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const std::vector<int32_t> ids = {5, 9, 2};
  const std::vector<int32_t> pos = {0, 1, 2};

  const int64_t before = Steps();
  {
    const vllm::detail::ExpertStreamStepScope outer;

    // A SECOND scope on this thread is refused by name. Not `CHECK_THROWS_WITH_AS`:
    // VT_CHECK appends " at <file>:<line>", so an exact-message match would
    // break on any edit above it and say nothing about the guarantee.
    bool nested_threw = false;
    std::string nested_what;
    try {
      const vllm::detail::ExpertStreamStepScope inner;
      (void)inner;
    } catch (const std::runtime_error& e) {
      nested_threw = true;
      nested_what = e.what();
    }
    CHECK(nested_threw);
    CHECK(nested_what.find("must not nest") != std::string::npos);

    // AND THE SCOPE IS THE PRODUCTION GUARD, not a parallel flag. A real
    // forward entered while the scope is held is refused too — which is the
    // only way to show that the two share a boundary, and which also measures
    // the breadth of the refusal: it is armed on the DEFAULT path, so a nest
    // reds every Qwen3.5 forward and not merely the streamed ones. That is the
    // intended polarity (the note on `Qwen35ExpertStreamStep` argues it): a
    // nest is a defect in the call graph whether or not a store exists, and
    // arming it only on the rare configuration would let the default path
    // establish one that nobody sees until streaming is switched on.
    bool forward_threw = false;
    std::string forward_what;
    try {
      (void)Qwen3_5Model::ForwardDense(ids, pos, w, c, q);
    } catch (const std::runtime_error& e) {
      forward_threw = true;
      forward_what = e.what();
    }
    CHECK(forward_threw);
    CHECK(forward_what.find("must not nest") != std::string::npos);
  }

  // The refusal is NOT sticky. A constructor that throws leaves no object, so
  // no destructor runs and no step is charged for it; the outer scope closed
  // exactly one. Two refused opens plus one closed scope must therefore be one
  // step, and the same forward must now succeed — a guard that leaked its flag
  // on the throw would fail here rather than at some unrelated later case.
  CHECK(Steps() - before == 1);
  const std::vector<float> logits = Qwen3_5Model::ForwardDense(ids, pos, w, c, q);
  CHECK(logits.size() ==
        static_cast<size_t>(ids.size()) * static_cast<size_t>(c.vocab_size));
  CHECK(Steps() - before == 2);
}
