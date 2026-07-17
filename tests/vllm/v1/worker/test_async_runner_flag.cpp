// vllm.cpp original (vt runtime env-flag plumbing). CPU-tier contract for the
// VT_ASYNC_RUNNER flag predicate (include/vllm/v1/worker/gpu/async_runner_flag.h)
// that gates the runner's async device path — the piece whose default was FLIPPED
// ON on 2026-07-17 to mirror vLLM's async-scheduling default. The device kernels
// and the depth-2 engine loop the flag ultimately engages are exercised by the
// DGX model gates + the CPU construction-matrix (test_loaded_engine_dense.cpp) and
// runner token-exactness (test_runner.cpp) suites; this suite pins the portable
// default-ON / '0'-rollback parse so the rollback contract is regression-covered on
// every platform, not just DGX (house convention: test_rmsnorm_decode_fast.cpp).
#include <doctest/doctest.h>

#include "vllm/v1/worker/gpu/async_runner_flag.h"

using vllm::v1::AsyncRunnerFlagIsOn;

TEST_CASE(
    "VT_ASYNC_RUNNER defaults ON; only a '0'-leading value rolls back") {
  // Default (unset) is ON: the async device path ships, mirroring vLLM's
  // async-scheduling default-on-when-compatible (DGX-proven token-exact).
  CHECK(AsyncRunnerFlagIsOn(nullptr));
  // Non-'0'-leading values stay ON (including the explicit opt-in and junk).
  CHECK(AsyncRunnerFlagIsOn("1"));
  CHECK(AsyncRunnerFlagIsOn(""));
  CHECK(AsyncRunnerFlagIsOn("on"));
  CHECK(AsyncRunnerFlagIsOn("true"));
  CHECK(AsyncRunnerFlagIsOn("2"));
  CHECK(AsyncRunnerFlagIsOn(" 0"));  // leading space, not a '0' first char
  CHECK(AsyncRunnerFlagIsOn("10"));
  // Runner-level rollback: FIRST character '0' (the same-binary sync path).
  CHECK_FALSE(AsyncRunnerFlagIsOn("0"));
  CHECK_FALSE(AsyncRunnerFlagIsOn("0abc"));
  CHECK_FALSE(AsyncRunnerFlagIsOn("00"));
}
