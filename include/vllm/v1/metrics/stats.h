// Ported from: vllm/v1/metrics/stats.py @ e24d1b24
//
// Scope: the CACHE-HIT statistics surface. Two ported types plus their shared
// base, mirroring upstream 1:1:
//   * BaseCacheStats            (stats.py:18-32)  — reset/requests/queries/hits
//   * PrefixCacheStats          (stats.py:115-142) — adds the three
//     mutually-exclusive preempted_* counters and record()
//   * CachingMetrics            (stats.py:35-111) — the hit-rate aggregator over
//     a sliding window of the most recent N REQUESTS (not steps, not tokens)
//
// WHY THIS EXISTS: before this port there were NO prefix-cache statistics at
// any level in the tree (`src/vllm/v1/core/kv_cache_manager.cpp:139` was a
// deferral comment). The benchmark protocol makes "prove cache hits in every
// arm" mandatory, so an arm without a hit counter is VOID — which made every
// caching / offloading benchmark unrunnable. These counters are the evidence
// surface for `KV-PREFIX-CACHE`, `KV-OFFLOAD` and
// `BACKEND-GATE-CUDA-SGLANG-PREFIX`.
//
// SEMANTICS THAT ARE EASY TO GET WRONG (all mirrored exactly):
//   - `queries` and `hits` count TOKENS, not requests or blocks. `queries` is
//     request.num_tokens (the whole prompt), `hits` is the number of tokens
//     served from cache (stats.py:236-239).
//   - The preempted_* counters are MUTUALLY EXCLUSIVE with requests/queries/
//     hits: a request that has been preempted at least once contributes to the
//     preempted_* triple ONLY (stats.py:134-142). This keeps a preempted
//     request's guaranteed second-pass hit from inflating the headline rate.
//   - CachingMetrics' window is measured in REQUESTS (default 1000), and the
//     eviction loop always keeps at least one entry, so the most recent
//     observation is never dropped (stats.py:96-104).
//   - An observation with requests == 0 is NOT appended, so idle steps cannot
//     flush useful history out of the window (stats.py:70-72).
//   - `stats.reset` (set by reset_prefix_cache) resets the aggregate BEFORE the
//     current observation is folded in (stats.py:65-68).
//
// DEVIATIONS: none in behaviour. Upstream's dataclass field defaults and the
// deque of (requests, queries, hits) triples are a plain struct and a
// std::deque here.
#ifndef VLLM_V1_METRICS_STATS_H_
#define VLLM_V1_METRICS_STATS_H_

#include <cstdint>
#include <deque>
#include <tuple>

namespace vllm::v1 {

// Upstream: @dataclass BaseCacheStats (vllm/v1/metrics/stats.py:18-32).
struct BaseCacheStats {
  // Whether the cache was reset since the last observation.
  bool reset = false;
  // The number of requests in this update.
  int64_t requests = 0;
  // The number of queried TOKENS in these requests.
  int64_t queries = 0;
  // The number of TOKENS served from cache in these requests.
  int64_t hits = 0;
};

// Upstream: @dataclass PrefixCacheStats (vllm/v1/metrics/stats.py:115-142).
struct PrefixCacheStats : BaseCacheStats {
  // The number of previously-preempted requests in this update.
  int64_t preempted_requests = 0;
  // The `queries` number for preempted requests.
  int64_t preempted_queries = 0;
  // The `hits` number for preempted requests.
  int64_t preempted_hits = 0;

  // Aggregate one request's lookup. `preempted` is
  // `request.num_preemptions > 0` at the call site. Upstream: record()
  // (stats.py:130-142).
  void record(int64_t num_tokens, int64_t num_hits, bool preempted) {
    if (preempted) {
      preempted_requests += 1;
      preempted_queries += num_tokens;
      preempted_hits += num_hits;
    } else {
      requests += 1;
      queries += num_tokens;
      hits += num_hits;
    }
  }
};

// Upstream: class CachingMetrics (vllm/v1/metrics/stats.py:35-111).
// Hit rate over the most recent `max_recent_requests` requests.
class CachingMetrics {
 public:
  explicit CachingMetrics(int64_t max_recent_requests = 1000)
      : max_recent_requests_(max_recent_requests) {}

  // Fold one take-and-swap observation into the window.
  // Upstream: observe() (stats.py:58-104).
  void observe(const BaseCacheStats& stats) {
    // reset_prefix_cache was invoked before the current update: reset the
    // metrics BEFORE aggregating the current stats.
    if (stats.reset) {
      reset();
    }
    // Do NOT append empty stats: an idle step would otherwise slide useful
    // history out of the window.
    if (stats.requests == 0) {
      return;
    }
    queue_.emplace_back(stats.requests, stats.queries, stats.hits);
    aggregated_requests_ += stats.requests;
    aggregated_query_total_ += stats.queries;
    aggregated_query_hit_ += stats.hits;

    // Drop the oldest observations until the window fits. The `size() > 1`
    // guard preserves the latest observation regardless of its size.
    while (queue_.size() > 1 && aggregated_requests_ > max_recent_requests_) {
      const auto& front = queue_.front();
      aggregated_requests_ -= std::get<0>(front);
      aggregated_query_total_ -= std::get<1>(front);
      aggregated_query_hit_ -= std::get<2>(front);
      queue_.pop_front();
    }
  }

  // Upstream: reset() (stats.py:106-111).
  void reset() {
    aggregated_requests_ = 0;
    aggregated_query_total_ = 0;
    aggregated_query_hit_ = 0;
    queue_.clear();
  }

  // Upstream: @property empty.
  bool empty() const { return aggregated_requests_ == 0; }

  // Upstream: @property hit_rate. 0.0 when nothing has been queried.
  double hit_rate() const {
    if (aggregated_query_total_ == 0) {
      return 0.0;
    }
    return static_cast<double>(aggregated_query_hit_) /
           static_cast<double>(aggregated_query_total_);
  }

  int64_t aggregated_requests() const { return aggregated_requests_; }
  int64_t aggregated_query_total() const { return aggregated_query_total_; }
  int64_t aggregated_query_hit() const { return aggregated_query_hit_; }

 private:
  int64_t max_recent_requests_;
  int64_t aggregated_requests_ = 0;
  int64_t aggregated_query_total_ = 0;
  int64_t aggregated_query_hit_ = 0;
  // (requests, queries, hits) for the most recent observations.
  std::deque<std::tuple<int64_t, int64_t, int64_t>> queue_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_METRICS_STATS_H_
