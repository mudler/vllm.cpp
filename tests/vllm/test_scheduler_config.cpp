// Tests for the SchedulerConfig T0-subset port
// (vllm/config/scheduler.py @ e24d1b24).
#include <doctest/doctest.h>

#include "vllm/config/scheduler.h"

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
