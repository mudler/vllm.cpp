// ENG-WEIGHT-OFFLOAD W2a (#797) — the offload decision and its byte budget.
//
// Ported from (test-porting protocol .agents/porting.md), all @ 555967922:
//   * vllm/model_executor/offloader/uva.py:74-75 — a spent budget early-returns
//     a whole module.
//   * vllm/model_executor/offloader/uva.py:80-84 — the per-parameter budget
//     check BREAKS, so one module can end up partly offloaded.
//   * vllm/model_executor/offloader/uva.py:86-95 — targeting CONTINUES, so an
//     untargeted parameter consumes no budget and does not stop the walk.
//   * vllm/model_executor/offloader/uva.py:107 — the running total advances
//     after a parameter is offloaded, and only then.
//   * vllm/config/offload.py:36-38 — an empty target set matches everything.
//
// Upstream has NO unit test for this logic. Its coverage is
// tests/basic_correctness/test_cpu_offload.py, which needs a GPU and a
// checkpoint. These cases are AUTHORED against the upstream source.
//
// The two properties most likely to be got wrong, and the reason each has a
// dedicated case: BREAK and CONTINUE are different, and the budget must not
// advance for a weight that was never offloaded.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/config/offload.h"
#include "vllm/model_executor/weight_offload_policy.h"

using vllm::OffloadBackend;
using vllm::OffloadConfig;
using vllm::WeightOffloadDecision;
using vllm::WeightOffloadPolicy;

namespace {
constexpr int64_t kGiB = 1024LL * 1024LL * 1024LL;
}  // namespace

TEST_CASE("an empty target set offloads every weight until the budget is spent") {
  // offload.py:36-38 — empty means "offload non-selectively until the limit".
  WeightOffloadPolicy p(100, {});
  CHECK(p.active());
  CHECK_FALSE(p.exhausted());
  CHECK(p.Consider("mlp.experts.w2_weight", 40) == WeightOffloadDecision::kOffload);
  CHECK(p.offloaded_bytes() == 40);
  CHECK(p.Consider("self_attn.q_proj.weight", 40) == WeightOffloadDecision::kOffload);
  CHECK(p.offloaded_bytes() == 80);
  // 80 < 100, so this one is still admitted IN FULL: upstream checks the budget
  // before the parameter, never against the parameter's size.
  CHECK(p.Consider("self_attn.k_proj.weight", 999) == WeightOffloadDecision::kOffload);
  CHECK(p.offloaded_bytes() == 1079);
  CHECK(p.exhausted());
  // Now everything after is refused.
  CHECK(p.Consider("anything", 1) == WeightOffloadDecision::kBudgetExhausted);
  CHECK(p.offloaded_bytes() == 1079);
}

TEST_CASE("targeting CONTINUES: an untargeted weight consumes no budget") {
  // uva.py:94-95 is a `continue`, not a `break`. A policy that broke here would
  // stop at the first untargeted weight and offload almost nothing, and a
  // whole-model byte total would still look plausible.
  WeightOffloadPolicy p(1000, {"experts"});
  CHECK(p.Consider("self_attn.q_proj.weight", 500) ==
        WeightOffloadDecision::kNotTargeted);
  CHECK(p.offloaded_bytes() == 0);
  CHECK_FALSE(p.exhausted());
  // The walk continues, and a later targeted weight is still offloaded.
  CHECK(p.Consider("mlp.experts.w2_weight", 500) == WeightOffloadDecision::kOffload);
  CHECK(p.offloaded_bytes() == 500);
}

TEST_CASE("the budget check precedes targeting, so exhaustion never reads as untargeted") {
  // uva.py:80-84 runs before uva.py:86. The order matters for the caller's
  // report: a weight skipped for budget is a capacity fact, and a weight
  // skipped for targeting is a configuration fact.
  WeightOffloadPolicy p(10, {"experts"});
  CHECK(p.Consider("mlp.experts.w2_weight", 10) == WeightOffloadDecision::kOffload);
  CHECK(p.exhausted());
  // An UNTARGETED name now reports exhaustion, not kNotTargeted.
  CHECK(p.Consider("self_attn.q_proj.weight", 1) ==
        WeightOffloadDecision::kBudgetExhausted);
}

TEST_CASE("a module ends up PARTLY offloaded, per uva.py:82-84") {
  // The comment upstream is explicit: "we use per-parameter offloading / one
  // module might have some parameters offloaded and some not". A per-MODULE
  // implementation matches on a whole-model byte total and diverges here, which
  // is why this case counts the split instead of checking a total.
  WeightOffloadPolicy p(25, {});
  const std::vector<std::string> module_params = {
      "layer.0.w1", "layer.0.w2", "layer.0.w3", "layer.0.w4"};
  int offloaded = 0;
  int refused = 0;
  for (const std::string& name : module_params) {
    if (p.Consider(name, 10) == WeightOffloadDecision::kOffload) {
      ++offloaded;
    } else {
      ++refused;
    }
  }
  // The budget is tested BEFORE each weight, never against its size, so the
  // total is allowed to overshoot by one weight: 0, 10 and 20 are all under 25,
  // so three are offloaded and the total reaches 30. The fourth sees 30 >= 25
  // and is refused. Three of four is the split a per-module implementation
  // cannot produce.
  CHECK(offloaded == 3);
  CHECK(refused == 1);
  CHECK(p.offloaded_bytes() == 30);
}

TEST_CASE("dotted-segment targeting matches upstream's documented examples") {
  // offload.py:39-43 through the policy rather than the matcher, so the wiring
  // is covered too.
  WeightOffloadPolicy exact(kGiB, {"experts"});
  CHECK(exact.Consider("mlp.experts.w2_weight", 1) == WeightOffloadDecision::kOffload);

  WeightOffloadPolicy partial(kGiB, {"expert"});
  CHECK(partial.Consider("mlp.experts.w2_weight", 1) ==
        WeightOffloadDecision::kNotTargeted);

  WeightOffloadPolicy suffix(kGiB, {"w2"});
  CHECK(suffix.Consider("mlp.experts.w2_weight", 1) ==
        WeightOffloadDecision::kNotTargeted);

  // The distinction the docstring exists for.
  WeightOffloadPolicy scale(kGiB, {"w2_weight"});
  CHECK(scale.Consider("mlp.experts.w2_weight_scale", 1) ==
        WeightOffloadDecision::kNotTargeted);
  CHECK(scale.Consider("mlp.experts.w2_weight", 1) == WeightOffloadDecision::kOffload);
}

TEST_CASE("a zero budget is inert, so a caller keeps its existing path") {
  WeightOffloadPolicy p(0, {});
  CHECK_FALSE(p.active());
  CHECK(p.exhausted());
  CHECK(p.Consider("anything", 1) == WeightOffloadDecision::kBudgetExhausted);
  CHECK(p.offloaded_bytes() == 0);
  // A negative budget is clamped rather than trusted, so a caller cannot make
  // the policy offload without limit by passing a bad number.
  WeightOffloadPolicy negative(-1, {});
  CHECK_FALSE(negative.active());
  CHECK(negative.Consider("anything", 1) == WeightOffloadDecision::kBudgetExhausted);
}

TEST_CASE("a non-positive size cannot advance the budget") {
  // uva.py:107 counts real bytes. A caller that reports 0 or a negative size
  // must not be able to spend the budget without moving anything, because the
  // budget is the only thing standing between the feature and an OOM.
  WeightOffloadPolicy p(100, {});
  CHECK(p.Consider("a", 0) == WeightOffloadDecision::kOffload);
  CHECK(p.offloaded_bytes() == 0);
  CHECK(p.Consider("b", -5) == WeightOffloadDecision::kOffload);
  CHECK(p.offloaded_bytes() == 0);
  CHECK_FALSE(p.exhausted());
}

TEST_CASE("FromConfig builds the UVA budget and refuses the other backends") {
  // The UVA arm is the only one with a byte budget (offload.py:23). Prefetch
  // selects layers by position, so it is W5's shape and not a budget at all.
  OffloadConfig uva;
  uva.uva.cpu_offload_gb = 10.0;
  uva.uva.cpu_offload_params = {"experts"};
  uva.Validate();
  WeightOffloadPolicy p = WeightOffloadPolicy::FromConfig(uva);
  CHECK(p.active());
  CHECK(p.max_bytes() == 10LL * kGiB);
  CHECK(p.Consider("mlp.experts.w2_weight", 1) == WeightOffloadDecision::kOffload);
  CHECK(p.Consider("self_attn.q_proj.weight", 1) ==
        WeightOffloadDecision::kNotTargeted);

  // Prefetch resolves to a backend, and yields NO budget here.
  OffloadConfig prefetch;
  prefetch.prefetch.offload_group_size = 8;
  prefetch.prefetch.offload_num_in_group = 2;
  prefetch.Validate();
  CHECK(prefetch.ResolvedBackend() == OffloadBackend::kPrefetch);
  CHECK_FALSE(WeightOffloadPolicy::FromConfig(prefetch).active());

  // An unconfigured config is inert.
  CHECK_FALSE(WeightOffloadPolicy::FromConfig(OffloadConfig{}).active());

  // An explicit uva backend at a ZERO budget is inert too: it selects a backend
  // but would move nothing.
  OffloadConfig zero;
  zero.offload_backend = OffloadBackend::kUva;
  zero.Validate();
  CHECK(zero.ResolvedBackend() == OffloadBackend::kUva);
  CHECK_FALSE(WeightOffloadPolicy::FromConfig(zero).active());
}
