// vllm.cpp original — CPU-tier contract for the CUDA-graph-safe grow-only device
// scratch bookkeeping (src/vt/cuda/graph_safe_scratch.h).
//
// The production hazard is CUDA-only: a grow-only per-stream device scratch buffer
// (MoE grouped-GEMM index scratch, cutlass NVFP4 / FP8 GEMM workspaces) whose old
// block was FREED on growth would dangle a pointer baked into a captured pure-decode
// CUDA graph, causing an illegal memory access on the next replay at concurrency > 1.
// The fix RETIRES the old block (keeps it resident) instead of freeing it. This suite
// pins the portable retire bookkeeping so the "never free a retired block" guarantee
// is regression-covered on every platform, not just DGX. The device-side proof (that
// the 35B c2+ online-serving IMA is gone) is the DGX serving harness.
#include <doctest/doctest.h>

#include "vt/cuda/graph_safe_scratch.h"

using vt::cuda::RetireGraphScratch;
using vt::cuda::RetiredGraphScratchCount;

TEST_CASE("RetireGraphScratch keeps grow-only scratch blocks resident (never frees)") {
  // The retired list is process-global and monotonic; measure relative growth so the
  // suite is order-independent.
  const std::size_t before = RetiredGraphScratchCount();

  // A null old block (nothing allocated yet — the very first Ensure*) retires nothing.
  RetireGraphScratch(nullptr);
  CHECK(RetiredGraphScratchCount() == before);

  // Each real growth retires exactly one old block. There is deliberately NO API to
  // free a retired block: the structural guarantee is that a captured-graph pointer,
  // once retired, is valid for the whole process lifetime.
  int sentinel_a = 0, sentinel_b = 0, sentinel_c = 0;
  RetireGraphScratch(&sentinel_a);
  CHECK(RetiredGraphScratchCount() == before + 1);
  RetireGraphScratch(&sentinel_b);
  RetireGraphScratch(&sentinel_c);
  CHECK(RetiredGraphScratchCount() == before + 3);

  // Null between real retires is still ignored (no phantom entries).
  RetireGraphScratch(nullptr);
  CHECK(RetiredGraphScratchCount() == before + 3);
}
