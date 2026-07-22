// Ported from: tests/v1/core/test_kv_cache_metrics.py and
// tests/v1/core/test_scheduler_e2e.py::test_prefix_cache_stats_is_recorded
// @ e24d1b24; the aggregator semantics come from
// vllm/v1/metrics/stats.py:35-142.
//
// WHY THIS SUITE EXISTS: before this change there were NO prefix-cache
// statistics at any level of the tree, so no benchmark arm could satisfy the
// protocol's "prove cache hits in every arm" requirement and every caching /
// offloading benchmark was VOID. The last case here is the first end-to-end
// demonstration that our APC actually serves cached tokens.
#include <doctest/doctest.h>

#include <memory>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/metrics/stats.h"
#include "vllm/v1/request.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::v1::CachingMetrics;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::PrefixCacheStats;
using vllm::v1::Request;
using vllm::v1::Scheduler;
using vllm::v1::sha256_cbor;
using vt::DType;

namespace {

constexpr int kBlockSize = 16;

std::unique_ptr<Scheduler> CreateScheduler(bool enable_caching = true,
                                           int num_blocks = 4096) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = 16;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(kBlockSize, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return std::make_unique<Scheduler>(cfg, kv_cfg, kBlockSize, enable_caching);
}

std::unique_ptr<Request> MakeRequest(const std::string& id,
                                     const std::vector<int32_t>& prompt) {
  auto hasher = get_request_block_hasher(kBlockSize, sha256_cbor);
  SamplingParams params;
  params.max_tokens = 1;
  return std::make_unique<Request>(id, prompt, params, /*arrival_time=*/0.0,
                                   hasher);
}

// A prompt made of `shared_blocks` blocks of a common system prompt followed by
// `unique_blocks` blocks unique to `seed`. This is the shape every real
// shared-prefix workload has.
std::vector<int32_t> SharedPrefixPrompt(int shared_blocks, int unique_blocks,
                                        int seed) {
  std::vector<int32_t> prompt;
  for (int b = 0; b < shared_blocks; ++b) {
    for (int i = 0; i < kBlockSize; ++i) {
      prompt.push_back(1000 + b * kBlockSize + i);
    }
  }
  for (int b = 0; b < unique_blocks; ++b) {
    for (int i = 0; i < kBlockSize; ++i) {
      prompt.push_back(500000 + seed * 10000 + b * kBlockSize + i);
    }
  }
  return prompt;
}

}  // namespace

// --- PrefixCacheStats.record (stats.py:130-142) -------------------------------

TEST_CASE("PrefixCacheStats.record counts TOKENS, not requests") {
  PrefixCacheStats stats;
  stats.record(/*num_tokens=*/100, /*num_hits=*/32, /*preempted=*/false);
  stats.record(/*num_tokens=*/50, /*num_hits=*/0, /*preempted=*/false);
  CHECK(stats.requests == 2);
  CHECK(stats.queries == 150);
  CHECK(stats.hits == 32);
}

TEST_CASE("preempted requests go ONLY into the preempted_* counters") {
  // The mutual exclusion is load-bearing: a preempted request is guaranteed to
  // re-hit its own blocks, so folding it into the headline counters would
  // inflate the reported hit rate with work we already did.
  PrefixCacheStats stats;
  stats.record(/*num_tokens=*/100, /*num_hits=*/0, /*preempted=*/false);
  stats.record(/*num_tokens=*/100, /*num_hits=*/96, /*preempted=*/true);
  CHECK(stats.requests == 1);
  CHECK(stats.queries == 100);
  CHECK(stats.hits == 0);
  CHECK(stats.preempted_requests == 1);
  CHECK(stats.preempted_queries == 100);
  CHECK(stats.preempted_hits == 96);
}

// --- CachingMetrics (stats.py:35-111) ----------------------------------------

TEST_CASE("CachingMetrics reports 0.0 before anything is observed") {
  CachingMetrics metrics;
  CHECK(metrics.empty());
  CHECK(metrics.hit_rate() == doctest::Approx(0.0));
}

TEST_CASE("CachingMetrics aggregates a hit rate over its window") {
  CachingMetrics metrics;
  PrefixCacheStats s;
  s.record(100, 25, false);
  metrics.observe(s);
  CHECK_FALSE(metrics.empty());
  CHECK(metrics.hit_rate() == doctest::Approx(0.25));

  PrefixCacheStats s2;
  s2.record(100, 75, false);
  metrics.observe(s2);
  CHECK(metrics.hit_rate() == doctest::Approx(0.5));
}

TEST_CASE("CachingMetrics ignores EMPTY observations") {
  // stats.py:70-72 — an idle step must not slide useful history out of the
  // window.
  CachingMetrics metrics(/*max_recent_requests=*/2);
  PrefixCacheStats s;
  s.record(100, 50, false);
  metrics.observe(s);
  for (int i = 0; i < 100; ++i) {
    metrics.observe(PrefixCacheStats());
  }
  CHECK(metrics.aggregated_requests() == 1);
  CHECK(metrics.hit_rate() == doctest::Approx(0.5));
}

TEST_CASE("CachingMetrics slides its window but ALWAYS keeps the newest entry") {
  // stats.py:96-104 — the `len(queue) > 1` guard means even an observation
  // larger than the whole window survives.
  CachingMetrics metrics(/*max_recent_requests=*/2);
  for (int i = 0; i < 5; ++i) {
    PrefixCacheStats s;
    s.record(10, 0, false);
    metrics.observe(s);
  }
  CHECK(metrics.aggregated_requests() <= 2);

  CachingMetrics big(/*max_recent_requests=*/2);
  PrefixCacheStats huge;
  for (int i = 0; i < 10; ++i) {
    huge.record(10, 10, false);
  }
  big.observe(huge);
  CHECK(big.aggregated_requests() == 10);
  CHECK(big.hit_rate() == doctest::Approx(1.0));
}

TEST_CASE("a reset flag clears the aggregate BEFORE folding in the new stats") {
  CachingMetrics metrics;
  PrefixCacheStats s;
  s.record(100, 100, false);
  metrics.observe(s);
  CHECK(metrics.hit_rate() == doctest::Approx(1.0));

  PrefixCacheStats after_reset;
  after_reset.reset = true;
  after_reset.record(100, 0, false);
  metrics.observe(after_reset);
  // Pre-reset history must not blend into the post-reset rate.
  CHECK(metrics.aggregated_requests() == 1);
  CHECK(metrics.hit_rate() == doctest::Approx(0.0));
}

// --- the manager / scheduler wiring ------------------------------------------

TEST_CASE("make_prefix_cache_stats is take-and-swap") {
  auto sched = CreateScheduler();
  auto& mgr = *sched->kv_cache_manager;
  REQUIRE(mgr.log_stats);
  mgr.prefix_cache_stats.record(10, 5, false);
  auto first = mgr.make_prefix_cache_stats();
  REQUIRE(first.has_value());
  CHECK(first->queries == 10);
  // A second read must be EMPTY — the consumer aggregates deltas, so a
  // non-destructive read would double count.
  auto second = mgr.make_prefix_cache_stats();
  REQUIRE(second.has_value());
  CHECK(second->queries == 0);
}

TEST_CASE("reset_prefix_cache flags the stats so the window clears") {
  auto sched = CreateScheduler();
  auto& mgr = *sched->kv_cache_manager;
  CHECK(mgr.reset_prefix_cache());
  auto stats = mgr.make_prefix_cache_stats();
  REQUIRE(stats.has_value());
  CHECK(stats->reset);
}

TEST_CASE("no stats are produced when log_stats is off") {
  // stats.py's counterpart: make_prefix_cache_stats returns None.
  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = 256;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(kBlockSize, 1, 1, DType::kF32));
  vllm::v1::KVCacheManager mgr(kv_cfg, /*max_model_len=*/1024,
                               /*scheduler_block_size=*/kBlockSize,
                               /*hash_block_size=*/kBlockSize,
                               /*max_num_batched_tokens=*/std::nullopt,
                               /*enable_caching=*/true, /*use_eagle=*/false,
                               /*log_stats=*/false);
  CHECK_FALSE(mgr.make_prefix_cache_stats().has_value());
}

// --- THE MEASUREMENT ---------------------------------------------------------

TEST_CASE("MEASURED cache-hit rate on a repeated-prefix workload") {
  // The first end-to-end demonstration in this project that automatic prefix
  // caching actually serves cached tokens. Sixteen requests share an 8-block
  // (128-token) system prompt and carry 2 unique blocks each.
  //
  // Expected arithmetic, so the number is a PREDICTION and not just an
  // observation:
  //   * request 0 populates the cache and hits nothing;
  //   * requests 1..15 hit the full 8 shared blocks = 128 tokens each;
  //   * queries counts request.num_tokens = 160 tokens per request.
  //   => hits  = 15 * 128 = 1920
  //      queries = 16 * 160 = 2560
  //      hit rate = 0.75
  init_none_hash(sha256_cbor);
  const int kSharedBlocks = 8;
  const int kUniqueBlocks = 2;
  const int kRequests = 16;

  auto sched = CreateScheduler(/*enable_caching=*/true);

  for (int i = 0; i < kRequests; ++i) {
    auto req = MakeRequest(
        "r" + std::to_string(i),
        SharedPrefixPrompt(kSharedBlocks, kUniqueBlocks, /*seed=*/i));
    sched->add_request(std::move(req));
    // Schedule one request at a time so each fully caches its blocks before the
    // next looks them up.
    sched->schedule();
  }

  const CachingMetrics& metrics = sched->prefix_cache_metrics();
  MESSAGE("MEASURED prefix-cache hit rate: "
          << metrics.hit_rate() << " (" << metrics.aggregated_query_hit()
          << " hit tokens / " << metrics.aggregated_query_total()
          << " queried tokens over " << metrics.aggregated_requests()
          << " requests)");

  CHECK(metrics.aggregated_requests() == kRequests);
  CHECK(metrics.aggregated_query_total() ==
        kRequests * (kSharedBlocks + kUniqueBlocks) * kBlockSize);
  CHECK(metrics.aggregated_query_hit() ==
        (kRequests - 1) * kSharedBlocks * kBlockSize);
  CHECK(metrics.hit_rate() == doctest::Approx(0.75));
}

TEST_CASE("the same workload with caching OFF reports a ZERO hit rate") {
  // The negative control: without it, a non-zero number above could be an
  // artefact of the counter rather than evidence of a working cache.
  init_none_hash(sha256_cbor);
  auto sched = CreateScheduler(/*enable_caching=*/false);
  for (int i = 0; i < 16; ++i) {
    sched->add_request(MakeRequest("r" + std::to_string(i),
                                   SharedPrefixPrompt(8, 2, /*seed=*/i)));
    sched->schedule();
  }
  const CachingMetrics& metrics = sched->prefix_cache_metrics();
  // get_computed_blocks early-outs before recording when caching is disabled,
  // exactly as upstream does, so nothing is observed at all.
  CHECK(metrics.hit_rate() == doctest::Approx(0.0));
}
