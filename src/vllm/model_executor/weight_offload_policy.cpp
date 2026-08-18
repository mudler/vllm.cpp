// Ported from: vllm/model_executor/offloader/uva.py:74-107 @ 555967922.
#include "vllm/model_executor/weight_offload_policy.h"

#include <utility>

namespace vllm {

WeightOffloadPolicy::WeightOffloadPolicy(int64_t max_bytes,
                                         std::set<std::string> target_segments)
    : max_bytes_(max_bytes > 0 ? max_bytes : 0),
      target_segments_(std::move(target_segments)) {}

WeightOffloadPolicy WeightOffloadPolicy::FromConfig(const OffloadConfig& config) {
  // Only the UVA arm carries a byte budget upstream (offload.py:23). The
  // prefetch arm selects layers by position instead, so it is not a budget and
  // is not modelled here; W5 owns it.
  const std::optional<OffloadBackend> backend = config.ResolvedBackend();
  if (!config.is_offloading_enabled() || !backend.has_value() ||
      *backend != OffloadBackend::kUva) {
    return WeightOffloadPolicy(0, {});
  }
  return WeightOffloadPolicy(config.uva.cpu_offload_max_bytes(),
                             config.uva.cpu_offload_params);
}

WeightOffloadDecision WeightOffloadPolicy::Consider(
    const std::string& canonical_name, int64_t bytes) {
  // uva.py:80-84. The budget is checked BEFORE the targeting test, so an
  // exhausted budget stops everything and never reports a name as untargeted.
  if (exhausted()) return WeightOffloadDecision::kBudgetExhausted;

  // uva.py:86-95. An EMPTY set matches everything (offload.py:36-38). A
  // non-matching name CONTINUES: it consumes no budget and does not stop the
  // walk. Reusing the config's matcher keeps one implementation of the
  // dot-anchored rule.
  if (!target_segments_.empty()) {
    bool targeted = false;
    for (const std::string& segment : target_segments_) {
      if (UVAOffloadConfig::MatchesSegment(canonical_name, segment)) {
        targeted = true;
        break;
      }
    }
    if (!targeted) return WeightOffloadDecision::kNotTargeted;
  }

  // uva.py:107. The total advances only for a weight that is actually
  // offloaded. A negative or zero size cannot advance it either, so a caller
  // that reports a bad size cannot spend the budget without moving anything.
  if (bytes > 0) offloaded_bytes_ += bytes;
  return WeightOffloadDecision::kOffload;
}

}  // namespace vllm
