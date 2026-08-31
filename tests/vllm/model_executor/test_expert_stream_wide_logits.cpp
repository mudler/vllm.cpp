// MODEL-TEXT-GLM-MOE-DSA W3 (#2214, spec .agents/specs/glm-dsa-latest-deepseek.md
// §3.7) — G3 on a model the existing fixture cannot discriminate on.
//
// THE GATE. A streamed slice and the resident tower produce IDENTICAL logits.
// That is the row's actual novelty and it needs no oracle at all: vLLM at the
// pin implements GLM-5.3's architecture and cannot run it on any device this
// project can reach (§3.6, `## Owed` O1), so no wave may promise a token-exact
// number against it. What every wave from W3 on CAN prove is that turning the
// lane on does not change a byte.
//
// WHY THIS IS A SECOND BINARY AND NOT A CASE IN `test_expert_stream_wiring`.
// `test_expert_stream_wiring` already asserts this on the 4-layer/4-expert
// model, and spec §3.2 gap 7 records why that is not enough: quoting
// `expert-streaming.md` `## Owed`, "no test model has more than 4 experts or 4
// layers". The wider model needs a LARGER slot budget than that binary sets, and
// `VT_MOE_EXPERT_STREAM_SLOTS` is read once into a function-local static behind
// a process-lifetime singleton — the reason `expert_stream_model.h` gives for
// there being separate binaries at all. Changing the existing binary's budget to
// fit this model would silently move the conditions of the case already there.
//
// WHAT WIDENING BUYS, stated so a reader does not read 6 and 8 as arbitrary. At
// 4 experts with top-2 the routed set is half the tower, so an expert index that
// is off by a constant, folded modulo the expert count, or clamped still names a
// real expert and the logits stay finite and plausible. At 8 experts with top-3
// the routed set is a minority of the tower and those cases point at an expert
// the router did not choose. At 6 layers the largest row offset is past every
// smaller stride, so a slice scaled by the wrong one of the tower's two
// dimensions leaves the slice it was meant to name. The unstreamed arm is the
// control: both arms take the same router, so a routing defect moves both
// together and this gate would stay green — which is exactly why the assertion
// is bit-exact equality between the two arms AND a separate check that the
// streamed arm really streamed.
#include <stdlib.h>

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "support/expert_stream_model.h"
#include "support/test_env.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"

using expert_stream_test::CachePool;
using expert_stream_test::MakeWeights;
using expert_stream_test::MakeWideConfig;
using expert_stream_test::PrefillAttnMeta;
using expert_stream_test::PrefillGdnMeta;
using expert_stream_test::Q;
using vllm::HfConfig;
using vllm::Qwen3_5MoeWeights;
using vllm::Qwen3_5Model;

namespace {

struct EnableExpertStreaming {
  EnableExpertStreaming() {
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM", "1");
    // The whole tower set, not one step's routed subset: 6 layers x 3 towers x 8
    // experts = 144 distinct slices are REACHABLE across the tokens of one
    // prefill, because different tokens route to different experts and every
    // entry stays protected until the step ends. A budget below that would make
    // `exhausted` nonzero for an honest reason and turn this gate's control arm
    // into the thing it is measuring against. 256 is comfortably above it.
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_SLOTS", "256");
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_SLOT_BYTES", "8192");
    if (std::getenv("VT_MOE_EXPERT_STREAM_STATS_EVERY") == nullptr)
      vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_STATS_EVERY", "0");
    // The grouped keep-quant path stages the whole tower and cannot stream; the
    // production code already disables it when streaming is requested. Being
    // explicit here keeps the test honest about which path it is measuring.
    vllm_test::SetEnv("VT_QWEN35_GROUPED_MOE", "0");
  }
};
const EnableExpertStreaming kEnableExpertStreaming;

// One full paged forward through the PRODUCTION entry point, over a fresh cache
// so every call is independent.
std::vector<float> OneForward(const HfConfig& c, const Qwen3_5MoeWeights& w,
                              const std::vector<int32_t>& ids) {
  vt::Queue q = Q();
  const int64_t T = static_cast<int64_t>(ids.size());
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/8);
  const std::vector<int32_t> blocks = {0};
  return Qwen3_5Model::Forward(ids, pos, PrefillAttnMeta(T, blocks, 8, 0),
                               PrefillGdnMeta(T, 0), pool.attn_kv,
                               pool.gdn_state, w, c, q, {});
}

}  // namespace

TEST_CASE("G3: streamed and resident logits are IDENTICAL past 4 experts and 4 layers") {
  const HfConfig c = MakeWideConfig();
  // The fixture really is wider than the one the existing gate uses. Asserted
  // rather than assumed, because a later edit to `MakeConfig` that this config
  // derives from could shrink either bound and leave every assertion below
  // green while measuring the 4x4 model again.
  REQUIRE(c.num_hidden_layers > 4);
  REQUIRE(c.num_experts > 4);
  REQUIRE(c.num_experts_per_tok < c.num_experts);

  const Qwen3_5MoeWeights w = MakeWeights(c);
  const std::vector<int32_t> ids = {7, 1, 22, 4, 13, 30};

  vllm::detail::ExpertStreamSetForceFallback(false);
  const std::vector<float> streamed = OneForward(c, w, ids);
  const vllm::detail::ExpertStreamStats on =
      vllm::detail::ExpertStreamSnapshot();

  // THE INSTRUMENT RAN. Without this the case below is satisfied by a process
  // in which streaming never engaged at all: both arms would then be the
  // resident tower and "identical" would be a tautology.
  REQUIRE(on.active);
  const int64_t fills_after_streamed = on.fills;
  REQUIRE(fills_after_streamed > 0);
  // And the budget was not the thing under test: nothing was refused, so the
  // streamed arm streamed every slice it asked for.
  CHECK(on.exhausted == 0);
  // It really covered the wider tower set. `misses` is one per distinct
  // (tower, expert) first seen, and top-3 over 6 tokens must reach more than
  // the 4x4 model's entire routed set could.
  CHECK(on.fills > 48);

  vllm::detail::ExpertStreamSetForceFallback(true);
  const std::vector<float> tower = OneForward(c, w, ids);
  const vllm::detail::ExpertStreamStats off =
      vllm::detail::ExpertStreamSnapshot();
  vllm::detail::ExpertStreamSetForceFallback(false);

  // The unstreamed arm really did NOT stream: no new bytes moved, and every
  // slice it asked for was refused into the resident-tower fallback.
  CHECK(off.fills == fills_after_streamed);
  CHECK(off.forced > on.forced);
  // AND IT WAS NOT COUNTED AS A BUDGET REFUSAL (#1091 finding 6). `exhausted`
  // is the operator-facing number; the forced-fallback switch has no production
  // caller and must not inflate it.
  CHECK(off.exhausted == 0);

  // BIT-EXACT, not close. The slot holds a byte copy of the same tower bytes, so
  // the two arms feed the kernel identical inputs; anything but equality means
  // the copy is wrong, which a tolerance-based check would hide.
  REQUIRE(streamed.size() == tower.size());
  REQUIRE(streamed.size() ==
          ids.size() * static_cast<size_t>(c.vocab_size));
  size_t differing = 0;
  for (size_t i = 0; i < streamed.size(); ++i)
    if (!(streamed[i] == tower[i])) ++differing;
  CHECK(differing == 0);

  // NOT ALL ZERO, and not degenerate. A seam that returned an empty or constant
  // slice on both arms would satisfy every equality above.
  size_t finite = 0, nonzero = 0;
  for (const float v : streamed) {
    if (std::isfinite(v)) ++finite;
    if (v != 0.0f) ++nonzero;
  }
  CHECK(finite == streamed.size());
  CHECK(nonzero > 0);
}
