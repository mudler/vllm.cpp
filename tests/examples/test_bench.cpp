// Smoke test for the M2.1 benchmark harness (examples/bench/bench_core.h): drive
// the SYNTHETIC CPU engine through the full admission + step() measurement loop
// and assert it produces sane metrics. The NUMBERS are meaningless (toy weights)
// — this asserts the HARNESS: all N requests finish, throughput > 0, TTFT > 0,
// and the token accounting is coherent. The real parity numbers come from a GB10
// run with --model (dgx-pending), which this same code path drives.
#include "bench_core.h"

#include <doctest/doctest.h>

using vllm::bench::BenchConfig;
using vllm::bench::BenchResult;
using vllm::bench::RunBench;

TEST_CASE("bench: synthetic engine completes all requests with sane metrics") {
  BenchConfig cfg;
  cfg.num_prompts = 8;
  cfg.input_len = 16;
  cfg.output_len = 16;
  cfg.concurrency = 4;
  cfg.seed = 123;
  cfg.temperature = 0.0;  // greedy => deterministic, exactly output_len tokens.

  const BenchResult r = RunBench(cfg);

  // All N requests finished through the engine loop.
  CHECK(r.completed == cfg.num_prompts);
  // Wall time advanced and throughput is positive.
  CHECK(r.duration_s > 0.0);
  CHECK(r.request_throughput > 0.0);
  CHECK(r.output_throughput > 0.0);
  CHECK(r.total_token_throughput > 0.0);
  CHECK(r.input_throughput > 0.0);
  // Token accounting: greedy w/ no eos => exactly output_len tokens per request.
  CHECK(r.total_output == static_cast<int64_t>(cfg.num_prompts) * cfg.output_len);
  CHECK(r.total_input > 0);
  CHECK(r.total_token_throughput ==
        doctest::Approx(r.input_throughput + r.output_throughput));
  // Latency metrics are engaged (first token observed => TTFT > 0; multi-token
  // decode => TPOT/ITL > 0).
  CHECK(r.mean_ttft_ms > 0.0);
  CHECK(r.mean_tpot_ms > 0.0);
  CHECK(r.mean_itl_ms > 0.0);
  CHECK(r.mean_e2el_ms >= r.mean_ttft_ms);
  CHECK(r.mean_per_stream_decode > 0.0);
}

TEST_CASE("bench: concurrency=1 (serial) also completes and is coherent") {
  BenchConfig cfg;
  cfg.num_prompts = 4;
  cfg.input_len = 8;
  cfg.output_len = 8;
  cfg.concurrency = 1;
  cfg.seed = 7;

  const BenchResult r = RunBench(cfg);

  CHECK(r.completed == 4);
  CHECK(r.total_output == 4 * 8);
  CHECK(r.request_throughput > 0.0);
  CHECK(r.mean_ttft_ms > 0.0);
}
