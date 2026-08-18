// vllm.cpp original — the error-latch discipline for the graph dedup ops table.
//
// Row ENG-CUDAGRAPH-DEDUP, issues #1162 and #1184, spec
// .agents/specs/eng-cudagraph-dedup.md.
//
// WHAT THIS EXISTS FOR. `src/vt/graph_dedup_runtime.h` calls the CUDA / HIP runtime and
// is ALLOWED to see several of those calls fail. That is not an accident of the port,
// it is the feature: the whole safety argument for dedup is that the signature is only
// a lookup key and `cudaGraphExecUpdate` is the authority, so a probe the driver
// REFUSES is the normal, expected outcome that means "do not fold". The topology walk
// has five more escapes of the same kind, each of which degrades the key rather than
// aborting inside a capture.
//
// A CUDA runtime call that fails LATCHES its code in the runtime's per-thread sticky
// slot, and nothing in that file consumed the latch. The rest of this tree launches
// kernels with the ordinary `kernel<<<>>>(...); Check(cudaGetLastError())` pattern, so
// the NEXT unrelated kernel read our routine refusal and reported it as its own
// failure. That is #1184, observed 6/6 on GB10 as
//
//     vt graph dedup: captured 1 graphs, deduped to 1 execs
//     engine-fatal: EngineCore busy loop threw:
//       vt cuda: greedy_argmax launch: invalid device function
//
// from a `greedy_argmax` launch that had SUCCEEDED. Every symptom follows: it needs a
// real capture to exist (no capture, no probe, no latch), `CUDA_LAUNCH_BLOCKING=1` does
// not move it (the latch is host-side and synchronous), and `cudaGraphLaunch` itself
// returns success (a return value does not consume the latch).
//
// WHY THIS IS A TYPE AND NOT TWELVE `cudaGetLastError()` CALLS. Twelve hand-placed
// clears are a fix the thirteenth fallible call silently misses, and a file whose whole
// design is "these calls are allowed to fail" will grow a thirteenth. The clear
// therefore lives in a destructor at the file's ENTRY POINTS, and the entry points are
// exactly the six members of `GraphDedupOps`, which the backends reach only through
// `graph_dedup_rt::Ops()`. Every path out of the file — a plain return, a degradation
// escape, or a `VT_CHECK` unwinding — runs it.
//
// `MakeLatchGuardedOps` below is the only constructor of that table, and it takes the
// raw functions as template arguments, so their addresses never reach a `GraphDedupOps`
// field. A seventh operation added to `GraphDedupOps` and wired anywhere else leaves the
// field null, and `GraphDedupRegistry`'s constructor refuses an incomplete table. So the
// bypass is not merely discouraged, it does not construct.
//
// This header is device-free on purpose, the same way `vt/graph_dedup.h` is: the guard
// discipline is then gated by a CPU test on every platform instead of only where a
// driver exists. See the honesty note in `tests/vt/test_graph_dedup_runtime.cpp` about
// what such a test can and cannot prove.
#ifndef VT_GRAPH_DEDUP_LATCH_H_
#define VT_GRAPH_DEDUP_LATCH_H_

#include <string>

#include "vt/graph_dedup.h"

namespace vt::graph_dedup_latch {

// Runs `Rt::ClearLatchedError()` on every exit from the scope it is declared in,
// including an exception unwinding through it. `Rt` is the runtime policy —
// `graph_dedup_rt::CudaRuntime` in production, a counting fake in the suite.
template <class Rt>
struct ScopedLatchClear {
  ScopedLatchClear() = default;
  // Destructors are implicitly noexcept, and cudaGetLastError / hipGetLastError cannot
  // throw, so this is safe on the unwinding path that `Launch`'s VT_CHECK takes.
  ~ScopedLatchClear() { Rt::ClearLatchedError(); }
  ScopedLatchClear(const ScopedLatchClear&) = delete;
  ScopedLatchClear& operator=(const ScopedLatchClear&) = delete;
};

// One wrapper per `GraphDedupOps` member. The signatures are spelled out rather than
// deduced through an `auto` non-type parameter partial specialisation, because this
// header compiles under MSVC as well and the explicit form needs nothing past C++14.
template <class Rt, std::string (*Fn)(void*)>
inline std::string GuardedSignature(void* raw_graph) {
  ScopedLatchClear<Rt> clear;
  return Fn(raw_graph);
}

template <class Rt, void* (*Fn)(void*)>
inline void* GuardedInstantiate(void* raw_graph) {
  ScopedLatchClear<Rt> clear;
  return Fn(raw_graph);
}

template <class Rt, bool (*Fn)(void*, void*, std::string*)>
inline bool GuardedUpdate(void* exec, void* raw_graph, std::string* detail) {
  ScopedLatchClear<Rt> clear;
  return Fn(exec, raw_graph, detail);
}

template <class Rt, void (*Fn)(void*)>
inline void GuardedUnary(void* handle) {
  ScopedLatchClear<Rt> clear;
  Fn(handle);
}

template <class Rt, void (*Fn)(void*, void*)>
inline void GuardedLaunch(void* exec, void* stream) {
  ScopedLatchClear<Rt> clear;
  Fn(exec, stream);
}

// The ONLY way a GraphDedupOps table is built for a real runtime. Every field is a
// wrapper address; none is a raw function address.
template <class Rt, std::string (*Signature)(void*), void* (*Instantiate)(void*),
          bool (*Update)(void*, void*, std::string*), void (*DestroyExec)(void*),
          void (*DestroyGraph)(void*), void (*Launch)(void*, void*)>
inline GraphDedupOps MakeLatchGuardedOps() {
  GraphDedupOps table;
  table.signature = &GuardedSignature<Rt, Signature>;
  table.instantiate = &GuardedInstantiate<Rt, Instantiate>;
  table.update = &GuardedUpdate<Rt, Update>;
  table.destroy_exec = &GuardedUnary<Rt, DestroyExec>;
  table.destroy_graph = &GuardedUnary<Rt, DestroyGraph>;
  table.launch = &GuardedLaunch<Rt, Launch>;
  return table;
}

}  // namespace vt::graph_dedup_latch

#endif  // VT_GRAPH_DEDUP_LATCH_H_
