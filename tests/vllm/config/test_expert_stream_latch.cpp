// `ENG-RESIDENCY-CONFIG` (issues #1110, #1122) — does the HEADLINE knob's config
// field reach the streaming decision?
//
// WHY THIS IS ITS OWN BINARY. `ResolveExpertStreamRequested()` caches its answer
// in a function-local static, so a process gets exactly ONE chance to observe what
// it resolved. Every other case for this row therefore has to avoid calling it, or
// avoid asserting its value, and that left the single most important wiring in the
// change unwatched: rewiring the resolver to read `ActiveWeightResidencyConfig()
// .mmap` instead of `.expert_stream` — the same type, the adjacent field — left all
// four suites GREEN on both heads of PR #1119. The pure decision
// (`ExpertStreamRequestedFrom`) was covered and the environment NAME was covered;
// the field the wrapper reads was not.
//
// This binary spends its one observation on that. Nothing else here may call
// `ResolveExpertStreamRequested` before the case below does, which is the whole
// reason it is not a case in `test_weight_residency_config.cpp`.
//
// The `mmap` half of the same question is pinned by
// `test_weight_residency_reach.cpp`'s "the GGUF load POLICY consults the installed
// config", which can be asserted repeatedly because that knob does not latch.
#include <doctest/doctest.h>

#include <cstdlib>
#include <stdexcept>

#include "vllm/config/weight_residency.h"

TEST_CASE("expert stream: the CONFIG field alone turns streaming on") {
  vllm::ResetWeightResidencyConfigForTesting();
  // No environment variable anywhere in this case: the config is the ONLY input, so
  // a resolver that ignored the config or read a neighbouring field cannot pass.
  ::unsetenv("VT_MOE_EXPERT_STREAM");

  const vllm::WeightResidencyConfig cfg =
      vllm::parse_weight_residency_extension_json(
          R"({"vllm_cpp":{"expert_stream":{"enabled":true}}})");
  REQUIRE(cfg.expert_stream.has_value());
  REQUIRE(*cfg.expert_stream == true);
  // Deliberately NOT set. `mmap` is the field the surviving mutation read instead,
  // and leaving it unset makes that mutation resolve to the built-in OFF.
  REQUIRE_FALSE(cfg.mmap.has_value());
  vllm::SetWeightResidencyConfig(cfg);

  CHECK_FALSE(vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStream));
  CHECK(vllm::ResolveExpertStreamRequested() == true);

  // And the read LATCHED, which is the fact `SetWeightResidencyConfig` refuses a
  // late change against. Marking it is the production resolver's own work, not a
  // test hook, so deleting the mark turns this red.
  CHECK(vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStream));
  CHECK(vllm::WeightResidencyLatched());

  // The cached answer does not change afterwards, whatever arrives — which is why
  // a config that would change it is refused rather than recorded.
  vllm::WeightResidencyConfig off;
  off.expert_stream = false;
  CHECK_THROWS_AS(vllm::SetWeightResidencyConfig(off), std::logic_error);
  CHECK(vllm::ResolveExpertStreamRequested() == true);
}
