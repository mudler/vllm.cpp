// Row-wise dispatch for the HOST-REFERENCE model kernels — the scalar loops
// that mirror an upstream module directly rather than routing through a `vt`
// op, because no `vt` op expresses them (`vocoder1d::ConvTranspose1d`) or
// because their reduction order is itself the thing a gate pinned
// (`music3::LinearNoBias`, see minimax_music3_ar.h on `ArCompute`).
//
// WHY THIS IS NOT A SECOND THREADPOOL. It is a two-line adapter over the ONE
// pool `vt::cpu` already owns (`src/vt/cpu/cpu_threadpool.h`, the 1:1 ggml
// port), reached the same way `src/vt/tenstorrent/tenstorrent_ops.cpp:21` and
// the `vt` determinism suites reach it. Nothing here creates threads, chunks
// work, or decides a thread count; `ParallelForRows` does all three.
//
// THE DETERMINISM CONTRACT IS INHERITED, NOT RESTATED (cpu_threadpool.h:39-43):
// parallelism partitions OUTPUT elements only. Every output element is produced
// by exactly one worker running the same instruction sequence, in the same
// order, as the single-thread code. No atomic accumulation into a shared
// output, no split reductions, no reassociation — so a result is bit-identical
// to the serial loop BY CONSTRUCTION rather than within a tolerance. A caller
// that cannot honour that must not use this header.
//
// The one thing this adds is the SIZE GUARD. `ParallelForRows` kicks the pool
// for any `nr > 1`, and the MiniMax-Music3 autoregressive half already spends
// ~25 % of its wall clock inside `Threadpool::Barrier` (spec
// `.agents/specs/minimax-music3.md` §11.4): handing it more sub-microsecond
// dispatches makes it slower, not faster. Below the threshold the body runs
// inline on the caller — the SAME body over the SAME range, so the guard is a
// scheduling decision and never a numeric one.
#pragma once

#include <cstdint>
#include <functional>

#include "vt/cpu/cpu_threadpool.h"  // via -I src (CMakeLists.txt:1317)

namespace vllm {
namespace host_parallel {

// Multiply-accumulate-ish operations below which a pool kick costs more than
// the work it distributes. Measured order of magnitude, not a tuned constant:
// a kick plus two barriers is ~1-10 us on the boxes this runs on, and 64 Ki
// scalar FMAs is the same order. It only ever moves WHERE a body runs.
inline constexpr int64_t kMinParallelWork = 1 << 16;

// `body(r0, r1)` produces output rows [r0, r1). `work_per_row` is the caller's
// own estimate of the inner-loop trip count for one row; it selects inline vs
// pooled execution and nothing else.
inline void ForOutputRows(int64_t rows, int64_t work_per_row,
                          const std::function<void(int64_t r0, int64_t r1)>& body) {
  if (rows <= 0) return;
  if (rows == 1 || rows * work_per_row < kMinParallelWork) {
    body(0, rows);
    return;
  }
  vt::cpu::ParallelForRows(vt::cpu::CurrentThreadpool(), rows, body);
}

}  // namespace host_parallel
}  // namespace vllm
