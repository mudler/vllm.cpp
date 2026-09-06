// Ported from: vllm/v1/spec_decode/metrics.py (SpecDecodingProm) @ e126687a9.
#include "vllm/v1/spec_decode/metrics.h"

#include <algorithm>
#include <string>
#include <vector>

namespace vllm {
namespace v1 {
namespace spec_decode {

void SpecDecodingProm::Register(metrics::PromRegistry* registry,
                                int num_speculative_tokens,
                                const std::vector<std::string>& labelnames,
                                const std::vector<std::string>& labelvalues) {
  // metrics.py:211-213: `spec_decoding_enabled = speculative_config is not
  // None`, and an early return that registers NOTHING when it is off. A server
  // started without --speculative-config therefore exposes no spec_decode
  // family at all, rather than four counters frozen at zero.
  if (registry == nullptr || num_speculative_tokens <= 0) {
    return;
  }
  registry_ = registry;
  enabled_ = true;
  num_speculative_tokens_ = num_speculative_tokens;
  labelvalues_ = labelvalues;

  // metrics.py:228-232 counter_specs, verbatim in name and help string.
  registry_->RegisterCounter(kNumDraftsMetric, "Number of spec decoding drafts.",
                             labelnames);
  registry_->RegisterCounter(kNumDraftTokensMetric, "Number of draft tokens.",
                             labelnames);
  registry_->RegisterCounter(kNumAcceptedTokensMetric,
                             "Number of accepted tokens.", labelnames);
  // metrics.py:250-264: one extra `position` label, and exactly
  // num_speculative_tokens child series per engine — the cardinality is fixed
  // at construction, never grown by an observation.
  std::vector<std::string> pos_labelnames = labelnames;
  pos_labelnames.push_back("position");
  registry_->RegisterCounter(kNumAcceptedTokensPerPosMetric,
                             "Accepted tokens per draft position.",
                             pos_labelnames);

  // prometheus_client child metrics exist from construction, so every series
  // is present at zero before the first draft is verified. Without this the
  // families would be absent from a scrape taken during prefill, and a
  // dashboard cannot tell an absent series from a broken exporter.
  registry_->Prime(kNumDraftsMetric, labelvalues_);
  registry_->Prime(kNumDraftTokensMetric, labelvalues_);
  registry_->Prime(kNumAcceptedTokensMetric, labelvalues_);
  for (int pos = 0; pos < num_speculative_tokens_; ++pos) {
    std::vector<std::string> pos_labelvalues = labelvalues_;
    pos_labelvalues.push_back(std::to_string(pos));
    registry_->Prime(kNumAcceptedTokensPerPosMetric, pos_labelvalues);
  }
}

void SpecDecodingProm::Observe(const SpecDecodingStats& stats) {
  // metrics.py:267-268: the disabled arm returns before touching anything.
  if (!enabled_) {
    return;
  }
  registry_->IncCounter(kNumDraftsMetric, labelvalues_,
                        static_cast<double>(stats.num_drafts));
  registry_->IncCounter(kNumDraftTokensMetric, labelvalues_,
                        static_cast<double>(stats.num_draft_tokens));
  registry_->IncCounter(kNumAcceptedTokensMetric, labelvalues_,
                        static_cast<double>(stats.num_accepted_tokens));
  // metrics.py:278-281 iterates the REGISTERED counters, so the registered
  // cardinality bounds the loop. The min() guards the one way the two can
  // disagree: the scheduler and the logger each resolve num_speculative_tokens
  // from the speculative config, and a caller that constructed the logger with
  // a different k would otherwise index past the stats vector.
  const std::size_t n =
      std::min(static_cast<std::size_t>(num_speculative_tokens_),
               stats.num_accepted_tokens_per_pos.size());
  for (std::size_t pos = 0; pos < n; ++pos) {
    std::vector<std::string> pos_labelvalues = labelvalues_;
    pos_labelvalues.push_back(std::to_string(pos));
    registry_->IncCounter(
        kNumAcceptedTokensPerPosMetric, pos_labelvalues,
        static_cast<double>(stats.num_accepted_tokens_per_pos[pos]));
  }
}

}  // namespace spec_decode
}  // namespace v1
}  // namespace vllm
