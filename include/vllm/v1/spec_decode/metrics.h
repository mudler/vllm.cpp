// Ported from: vllm/v1/spec_decode/metrics.py @ e126687a9.
//
// SCOPE (SERVE-METRICS, #2770): the SPECULATIVE-DECODING half of vLLM's
// Prometheus catalog — the per-step aggregate the scheduler builds
// (`SpecDecodingStats`, metrics.py:17-49) and the four counter families the
// stat logger registers for it (`SpecDecodingProm`, metrics.py:177-281).
// Config-gated upstream and config-gated here: with no speculative config,
// nothing is registered and nothing is exposed.
//
// WHY THE NAMES MATTER MORE THAN THE VALUES. These four are the mechanism
// metric for speculative decoding, and every dashboard, PromQL recipe and
// cross-engine comparison names them literally. metrics.py:180-196 records the
// queries the names are chosen to serve:
//
//   acceptance rate     rate(vllm:spec_decode_num_accepted_tokens_total[$i]) /
//                       rate(vllm:spec_decode_num_draft_tokens_total[$i])
//   mean accept length  1 + rate(..._num_accepted_tokens_total[$i]) /
//                           rate(vllm:spec_decode_num_drafts[$i])
//
// so a renamed family is a broken query, not a cosmetic difference.
//
// DEVIATIONS from upstream, both structural:
//   * `SpecDecodingProm` writes into a caller-owned `PromRegistry` instead of
//     owning `prometheus_client.Counter` objects, which is what this port
//     already does for every other family (see metrics/prometheus.h).
//   * The `is_diffusion` arm (metrics.py:215-226, `vllm:diffusion_num_*`) is
//     NOT ported. It renames these same counters for dLLM models, and this
//     port registers no diffusion text engine, so there is nothing to gate on.
#ifndef VLLM_V1_SPEC_DECODE_METRICS_H_
#define VLLM_V1_SPEC_DECODE_METRICS_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/v1/metrics/prometheus.h"

namespace vllm {
namespace v1 {
namespace spec_decode {

// The four registered family names (metrics.py:229-231,255). The Prometheus
// text format appends `_total` to a counter, so a scrape shows
// `vllm:spec_decode_num_accepted_tokens_total`; the REGISTERED name is the one
// below. Both spellings are asserted by the gate so neither can drift.
inline constexpr const char* kNumDraftsMetric = "vllm:spec_decode_num_drafts";
inline constexpr const char* kNumDraftTokensMetric =
    "vllm:spec_decode_num_draft_tokens";
inline constexpr const char* kNumAcceptedTokensMetric =
    "vllm:spec_decode_num_accepted_tokens";
inline constexpr const char* kNumAcceptedTokensPerPosMetric =
    "vllm:spec_decode_num_accepted_tokens_per_pos";

// Upstream: @dataclass SpecDecodingStats (metrics.py:17-49).
//
// ONE SCHEDULER STEP's speculative aggregate, across every request that
// verified drafts in it. The scheduler builds it, `SchedulerStats` carries it
// to the frontend, and `SpecDecodingProm` folds it into the registry — exactly
// the channel every other family already travels.
struct SpecDecodingStats {
  // speculative_config.num_speculative_tokens: the per-position vectors' fixed
  // length, and the upper bound observe_draft asserts against.
  int num_spec_tokens = 0;
  // Number of (request, step) pairs that carried a draft. NOT forward passes:
  // this is upstream's `spec_decode_num_drafts`, so `1 + accepted / drafts` is
  // the mean acceptance length including the bonus token.
  int64_t num_drafts = 0;
  // Draft tokens VERIFIED this step, after the invalid-draft subtraction.
  int64_t num_draft_tokens = 0;
  // Draft tokens ACCEPTED this step. Excludes the bonus/replacement token the
  // target always emits, which is why it can be 0 on a step that made progress.
  int64_t num_accepted_tokens = 0;
  // Both vectors are `num_spec_tokens` long. Index d is draft depth d.
  std::vector<int64_t> num_accepted_tokens_per_pos;
  std::vector<int64_t> num_draft_tokens_per_pos;

  // Upstream: SpecDecodingStats.new (metrics.py:33-39).
  static SpecDecodingStats New(int num_spec_tokens) {
    SpecDecodingStats stats;
    stats.num_spec_tokens = num_spec_tokens;
    const std::size_t n =
        num_spec_tokens > 0 ? static_cast<std::size_t>(num_spec_tokens) : 0;
    stats.num_accepted_tokens_per_pos.assign(n, 0);
    stats.num_draft_tokens_per_pos.assign(n, 0);
    return stats;
  }

  // Upstream: observe_draft (metrics.py:41-49). Folds ONE request's verify
  // result into this step's aggregate. The rejection sampler accepts a PREFIX
  // of the draft, so `num_accepted_tokens` accepted drafts means depths
  // 0..num_accepted_tokens-1 were accepted — which is what makes the
  // per-position vector a plain prefix increment rather than a mask.
  void ObserveDraft(int64_t drafted, int64_t accepted) {
    num_drafts += 1;
    num_draft_tokens += drafted;
    num_accepted_tokens += accepted;
    // metrics.py:45 `assert num_accepted_tokens <= self.num_spec_tokens`, and
    // metrics.py:48-49 would raise IndexError past the same bound. Kept as a
    // THROW rather than a clamp: a step that accepted more than it drafted is
    // a scheduler defect, and clamping would hide it inside a metric.
    if (accepted < 0 || drafted < 0 ||
        accepted > static_cast<int64_t>(num_accepted_tokens_per_pos.size()) ||
        drafted > static_cast<int64_t>(num_draft_tokens_per_pos.size())) {
      throw std::out_of_range(
          "SpecDecodingStats::ObserveDraft: drafted=" + std::to_string(drafted) +
          " accepted=" + std::to_string(accepted) + " outside num_spec_tokens=" +
          std::to_string(num_spec_tokens));
    }
    for (int64_t i = 0; i < accepted; ++i) {
      num_accepted_tokens_per_pos[static_cast<std::size_t>(i)] += 1;
    }
    for (int64_t i = 0; i < drafted; ++i) {
      num_draft_tokens_per_pos[static_cast<std::size_t>(i)] += 1;
    }
  }
};

// Upstream: class SpecDecodingProm (metrics.py:177-281).
//
// Registers the four families into a caller-owned registry and folds one
// step's `SpecDecodingStats` into them. Default-constructed it is DISABLED and
// every method is a no-op, which is the `speculative_config is None` arm.
class SpecDecodingProm {
 public:
  SpecDecodingProm() = default;

  // metrics.py:200-264. `num_speculative_tokens` is
  // `speculative_config.num_speculative_tokens`; 0 means "no speculative
  // config" and leaves the registry untouched, so a non-speculative server's
  // /metrics is byte-identical to what it exposed before this port.
  //
  // `registry` must outlive this object. `labelnames`/`labelvalues` are the
  // logger's own {model_name, engine} schema; the per-position family appends
  // a `position` label to both (metrics.py:253).
  void Register(metrics::PromRegistry* registry, int num_speculative_tokens,
                const std::vector<std::string>& labelnames,
                const std::vector<std::string>& labelvalues);

  // metrics.py:211 spec_decoding_enabled.
  bool enabled() const { return enabled_; }

  // metrics.py:266-281. No-op while disabled.
  void Observe(const SpecDecodingStats& stats);

 private:
  metrics::PromRegistry* registry_ = nullptr;
  bool enabled_ = false;
  int num_speculative_tokens_ = 0;
  std::vector<std::string> labelvalues_;
};

}  // namespace spec_decode
}  // namespace v1
}  // namespace vllm

#endif  // VLLM_V1_SPEC_DECODE_METRICS_H_
