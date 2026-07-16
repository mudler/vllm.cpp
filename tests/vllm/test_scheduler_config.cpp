// Tests for the SchedulerConfig T0-subset port
// (vllm/config/scheduler.py @ e24d1b24).
#include <doctest/doctest.h>

#include <cstdlib>

#include "vllm/config/scheduler.h"

using vllm::AsyncSchedulingEnabled;
using vllm::SchedulerConfig;
using vllm::SchedulerPolicy;

TEST_CASE("SchedulerConfig defaults mirror upstream") {
  SchedulerConfig config;
  CHECK(config.max_num_batched_tokens == 2048);  // DEFAULT_MAX_NUM_BATCHED_TOKENS
  CHECK(config.max_num_seqs == 128);             // DEFAULT_MAX_NUM_SEQS
  CHECK_FALSE(config.max_num_scheduled_tokens.has_value());
  CHECK(config.long_prefill_token_threshold == 0);
  CHECK(config.enable_chunked_prefill == true);
  CHECK(config.policy == SchedulerPolicy::kFCFS);
}

TEST_CASE("ResolvedMaxNumScheduledTokens falls back to max_num_batched_tokens") {
  // scheduler.py: max_num_scheduled_tokens if set else max_num_batched_tokens.
  SchedulerConfig config;
  CHECK(config.ResolvedMaxNumScheduledTokens() == config.max_num_batched_tokens);

  config.max_num_scheduled_tokens = 512;
  CHECK(config.ResolvedMaxNumScheduledTokens() == 512);
}

TEST_CASE("PostInit derives the encoder budgets from max_num_batched_tokens") {
  SchedulerConfig config;
  config.max_num_batched_tokens = 4096;
  config.PostInit(/*max_model_len=*/2048);

  CHECK(config.max_model_len == 2048);
  CHECK(config.max_num_encoder_input_tokens == 4096);
  CHECK(config.encoder_cache_size == 4096);
}

TEST_CASE("PostInit for encoder-decoder disables chunked prefill") {
  SchedulerConfig config;
  config.long_prefill_token_threshold = 32;
  config.PostInit(/*max_model_len=*/2048, /*is_encoder_decoder=*/true);

  CHECK_FALSE(config.enable_chunked_prefill);
  CHECK(config.long_prefill_token_threshold == 0);
}

TEST_CASE("VerifyMaxModelLen: short budget without chunked prefill is rejected") {
  SchedulerConfig config;
  config.max_num_batched_tokens = 1024;
  config.enable_chunked_prefill = false;
  // max_num_batched_tokens (1024) < max_model_len (8192) && !chunked -> error.
  CHECK_THROWS(config.PostInit(/*max_model_len=*/8192));
}

TEST_CASE("VerifyMaxModelLen: short budget WITH chunked prefill is allowed") {
  SchedulerConfig config;
  config.max_num_batched_tokens = 1024;
  config.enable_chunked_prefill = true;
  CHECK_NOTHROW(config.PostInit(/*max_model_len=*/8192));
}

TEST_CASE("VerifyMaxModelLen: budget below max_num_seqs is rejected") {
  SchedulerConfig config;
  config.max_num_batched_tokens = 64;  // < max_num_seqs (128)
  config.max_num_seqs = 128;
  config.enable_chunked_prefill = true;  // isolate the max_num_seqs check
  CHECK_THROWS(config.PostInit(/*max_model_len=*/64));
}

// --- async_scheduling resolution (ENG-ASYNC-SCHED, spec W3) -----------------
TEST_CASE("async_scheduling default (nullopt) resolves ON when compatible") {
  // scheduler.py:158-162 default None; vllm/config/vllm.py:990-1038 resolves it
  // to True on a single-GPU MRV2 setup.
  SchedulerConfig config;
  CHECK_FALSE(config.async_scheduling.has_value());  // tri-state default None
  CHECK(config.ResolveAsyncScheduling(/*runner_supports_async=*/true) == true);
}

TEST_CASE("async_scheduling resolves OFF on incompatible combos") {
  SchedulerConfig config;  // default nullopt
  // Runner without the async device-input path (our GPUModelRunner today).
  CHECK(config.ResolveAsyncScheduling(/*runner_supports_async=*/false) == false);
  // Pooling model (vllm/config/vllm.py:959-1021).
  CHECK(config.ResolveAsyncScheduling(/*runner_supports_async=*/true,
                                      /*is_pooling_model=*/true) == false);
  // Incompatible spec-decode method.
  CHECK(config.ResolveAsyncScheduling(/*runner_supports_async=*/true,
                                      /*is_pooling_model=*/false,
                                      /*spec_decode_incompatible=*/true) == false);
}

TEST_CASE("async_scheduling explicit false forces OFF even when compatible") {
  SchedulerConfig config;
  config.async_scheduling = false;
  CHECK(config.ResolveAsyncScheduling(/*runner_supports_async=*/true) == false);
}

TEST_CASE("async_scheduling explicit true stays ON when compatible") {
  SchedulerConfig config;
  config.async_scheduling = true;
  CHECK(config.ResolveAsyncScheduling(/*runner_supports_async=*/true) == true);
  // But a hard-incompatible combo still disables it (conservative; upstream
  // raises for a forced-true incompatible combo).
  CHECK(config.ResolveAsyncScheduling(/*runner_supports_async=*/false) == false);
}

TEST_CASE("MaxConcurrentBatches: 2 under async scheduling, 1 otherwise") {
  // vllm/config/vllm.py:490-501: pp_size + 1 == 2 on a single GPU under async.
  SchedulerConfig config;
  CHECK(config.MaxConcurrentBatches(/*async_scheduling_effective=*/true) == 2);
  CHECK(config.MaxConcurrentBatches(/*async_scheduling_effective=*/false) == 1);
}

TEST_CASE("VT_ASYNC_SCHED env override (house rollback convention)") {
  // Unset: the config resolution is used as-is.
  ::unsetenv("VT_ASYNC_SCHED");
  CHECK(AsyncSchedulingEnabled(/*resolved=*/true) == true);
  CHECK(AsyncSchedulingEnabled(/*resolved=*/false) == false);

  // "0": force OFF (the rollback arm) regardless of the resolution.
  ::setenv("VT_ASYNC_SCHED", "0", /*overwrite=*/1);
  CHECK(AsyncSchedulingEnabled(/*resolved=*/true) == false);
  CHECK(AsyncSchedulingEnabled(/*resolved=*/false) == false);

  // "1": force ON regardless of the resolution.
  ::setenv("VT_ASYNC_SCHED", "1", /*overwrite=*/1);
  CHECK(AsyncSchedulingEnabled(/*resolved=*/false) == true);
  CHECK(AsyncSchedulingEnabled(/*resolved=*/true) == true);

  ::unsetenv("VT_ASYNC_SCHED");
}
