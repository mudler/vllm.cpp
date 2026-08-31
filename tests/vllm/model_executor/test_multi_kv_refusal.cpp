// `KV-DSV4-MULTICACHE` W5 (#2323) — the refusal predicate, both polarities.
//
// W3 made the runner allocate EVERY published cache and hand them to the forward
// keyed by the name each was published under. `ModelRegistry::Forward` refused
// any such set unconditionally, because no forward consumed one. W5 turns that
// into a DISPATCH gated on `ModelFactory::consumes_multi_kv`.
//
// WHY THIS FILE EXISTS RATHER THAN ONLY THE RUNNER CASE. The runner test
// ("a multi-cache forward is REFUSED, naming the channel") drives a real engine,
// so it can only exercise the polarity its architecture happens to have. The
// rule W5 adds is a two-sided one, and the side that MATTERS most is the one a
// full engine cannot easily reach today: a model that HAS claimed the capability
// must NOT be refused. A pure predicate is gateable without a model, a runner or
// a registry, which is the same reason W1 made the staging budget a pure
// function "so it is gateable without a device".
//
// WHAT IT PROTECTS. The failure this rule prevents is invisible to a token gate:
// a forward that discards an allocated topology still emits the right tokens,
// while decoding by recomputing the whole prefix. So the guard cannot be
// validated by output comparison, and has to be pinned directly.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"

namespace {

vllm::MultiKvCacheIndex MakeIndex(const std::vector<std::string>* names) {
  vllm::MultiKvCacheIndex mk;
  mk.layer_names = names;
  return mk;
}

}  // namespace

TEST_CASE("W5: a name-keyed cache set is refused ONLY when unclaimed") {
  const std::vector<std::string> names{"model.layers.0.attn", "model.layers.0.swa"};
  const vllm::MultiKvCacheIndex mk = MakeIndex(&names);

  // THE PRE-W5 BEHAVIOUR, preserved: a set arrives at a model that has not
  // claimed it, and it is refused. Every model in the tree is this case today.
  CHECK(vllm::MultiKvRefusalApplies(&mk, /*consumes_multi_kv=*/false));

  // THE SIDE W5 ADDS, and the reason this file is not redundant with the runner
  // case: a model that HAS wired its forward proceeds. A predicate that refused
  // here would make the capability unreachable while every existing gate stayed
  // green, because nothing else asserts this direction.
  CHECK_FALSE(vllm::MultiKvRefusalApplies(&mk, /*consumes_multi_kv=*/true));

  // NO SET, EITHER WAY. `multi_kv` is null for every topology the positional
  // `attn_kv` convention expresses -- which is every model but DeepSeek-V4 -- so
  // a guard that fired here would break every ordinary forward in the tree. That
  // is the opposite failure and a worse one, since it is loud and universal.
  CHECK_FALSE(vllm::MultiKvRefusalApplies(nullptr, /*consumes_multi_kv=*/false));
  CHECK_FALSE(vllm::MultiKvRefusalApplies(nullptr, /*consumes_multi_kv=*/true));
}

TEST_CASE("W5: ARRIVAL is nullness, not size — an empty set that arrived still refuses") {
  // The channel can arrive carrying nothing, and that is NOT the same as not
  // arriving. `ModelRegistry::Forward`'s message reads the payload -- the count,
  // the distinct group count and the first name -- precisely so an empty arrival
  // reports differently from a populated one, which W3 wrote it that way for.
  //
  // A predicate that tested `mk->size() > 0` instead of nullness would let an
  // empty-but-present set through to a forward that cannot consume it, and the
  // resulting silent discard is the failure this whole guard exists to stop.
  const std::vector<std::string> empty;
  const vllm::MultiKvCacheIndex mk = MakeIndex(&empty);
  REQUIRE(mk.size() == 0);

  CHECK(vllm::MultiKvRefusalApplies(&mk, /*consumes_multi_kv=*/false));
  CHECK_FALSE(vllm::MultiKvRefusalApplies(&mk, /*consumes_multi_kv=*/true));
}
