// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// CPU-tier contract for the env-gated cuBLASLt GEMM algo-selection diagnostic
// helpers (src/vt/cuda/gemm_algo_log.h): the VT_GEMM_ALGO_LOG flag predicate and
// the LogOncePerKey "one line per unique key" bookkeeping. The cuBLASLt emit
// itself is CUDA-only and lives in cuda_matmul.cu; this suite pins the portable
// plumbing that gates it (default OFF, one line per unique key) so the flag/
// uniqueness logic is regression-covered on every platform, not just DGX.
#include <doctest/doctest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "vt/cuda/gemm_algo_log.h"

using vt::cuda::GemmAlgoLogFlagIsOn;
using vt::cuda::LogOncePerKey;

TEST_CASE("VT_GEMM_ALGO_LOG is on ONLY for exactly \"1\"; default (unset) is OFF") {
  CHECK(GemmAlgoLogFlagIsOn("1"));
  CHECK_FALSE(GemmAlgoLogFlagIsOn(nullptr));  // unset -> OFF: the zero-cost default
  CHECK_FALSE(GemmAlgoLogFlagIsOn("0"));
  CHECK_FALSE(GemmAlgoLogFlagIsOn(""));
  CHECK_FALSE(GemmAlgoLogFlagIsOn("2"));
  CHECK_FALSE(GemmAlgoLogFlagIsOn("10"));   // prefix "1" must not enable
  CHECK_FALSE(GemmAlgoLogFlagIsOn("1 "));   // trailing space must not enable
  CHECK_FALSE(GemmAlgoLogFlagIsOn(" 1"));   // leading space must not enable
  CHECK_FALSE(GemmAlgoLogFlagIsOn("true"));
  CHECK_FALSE(GemmAlgoLogFlagIsOn("on"));
}

TEST_CASE("LogOncePerKey emits once per DISTINCT key and suppresses repeats") {
  LogOncePerKey once;
  // Two keys that differ ONLY in the output (c) dtype: the packed-arm question is
  // whether a BF16-C vs F32-C selection latches a different algo, so these must
  // be treated as distinct selections (each logged once).
  const std::string bf16_c = "cublasLt|m=8,n=256,k=2048|a=bf16,b=bf16,c=bf16|TN";
  const std::string f32_c = "cublasLt|m=8,n=256,k=2048|a=bf16,b=bf16,c=f32|TN";
  CHECK(once.ShouldLog(bf16_c));        // first sighting -> log
  CHECK_FALSE(once.ShouldLog(bf16_c));  // repeat -> suppressed (hot call site)
  CHECK_FALSE(once.ShouldLog(bf16_c));
  CHECK(once.ShouldLog(f32_c));         // distinct dtype-combo key -> log
  CHECK_FALSE(once.ShouldLog(f32_c));
  CHECK(once.size() == 2);
}

TEST_CASE("LogOncePerKey is thread-safe: exactly one winner per key under contention") {
  LogOncePerKey once;
  const std::string key = "cublasLt|m=8,n=6144,k=2048|a=bf16,b=bf16,c=bf16|TN";
  constexpr int kThreads = 16;
  std::atomic<int> winners{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&] {
      if (once.ShouldLog(key)) winners.fetch_add(1, std::memory_order_relaxed);
    });
  }
  for (auto& t : threads) t.join();
  CHECK(winners.load() == 1);  // exactly one thread logs; every other is suppressed
  CHECK(once.size() == 1);
}
