// PORT of SGLang's breakable CUDA graph (BCG) at pin `f63458b5be`,
// `python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py`
// (374 lines; every bare `:N` below is a line in that file).
//
// Row ENG-CUDAGRAPH-BREAK, spec `.agents/specs/eng-cudagraph-break.md`,
// issue #1192 (W1), parent #1163.
//
// WHAT THIS IS. A forward is captured as a SEQUENCE of graph segments split at
// break points, so a forward containing a host-dependent operation is still
// graphed except at that operation, instead of running eager for the whole step.
//
// WHERE THE BOUNDARY COMES FROM. vLLM, the primary oracle, at pin `5559679229`.
// `CUDAGraphMode.PIECEWISE` (`vllm/config/compilation.py:60-63`) keeps "the
// cudagraph incompatible ops (i.e. some attention ops) outside the cudagraph"
// (`:608-635`), and the split points are `splitting_ops` (`:517`), defaulted at
// `:1145` to the attention family listed at `:764-772`. That boundary transfers.
// Its MECHANISM does not: vLLM gets the split from Dynamo and FX, and we have no
// compiler. SGLang reaches the same coverage with no compiler, so SGLang is the
// secondary oracle for the CONSTRUCTION only, per `.agents/oracles/sglang.md`.
//
// WHAT THIS IS NOT. This is a COVERAGE and CORRECTNESS seam, not a speed lever,
// and no throughput claim is attached to it anywhere. GB10 measured prefill idle
// between launches at 3.8% with GPU-busy above 96%, and the 27B prefill gap at
// 92.5% non-GEMM glue GPU work; decode is already captured and already banked
// its launch-overhead win. A speed claim from this seam is admissible only after
// naming a path that is BOTH currently eager AND currently host-bound.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "vt/backend.h"
#include "vt/tensor.h"

namespace vt {

// ---------------------------------------------------------------------------
// The writeback contract. Port of `_copy_output` (`:172-201`).
// ---------------------------------------------------------------------------
//
// THIS IS LOAD-BEARING, NOT DECORATOR SUGAR (spec `## Risks/decisions` D9). On
// replay N the eager break operation returns a FRESH allocation, whose address
// is NOT the one the following segment baked at capture time. A container that
// replayed the caller's raw function and discarded its result would leave
// segment `i+1` reading capture-time data forever while the break function wrote
// somewhere else. The failure is WRONG NUMERICS, not a fault, which is the same
// detection class a clean `compute-sanitizer` run cannot see.
//
// Overload `CopyOutput` for a break function's output type; it is found by
// argument-dependent lookup. The tensor overload copies IN PLACE and preserves
// the destination's address, mirroring upstream's `assertIs(result, dst)`
// (`:190`). A type with NO overload is the documented non-copyable fallback
// (`:201`): the seam fabricates no copy, and such a break must use the in-place
// form of `GraphBreak` instead.
void CopyOutput(Backend& b, Queue& q, Tensor& dst, const Tensor& src);
void CopyOutput(Backend& b, Queue& q, std::vector<Tensor>& dst,
                const std::vector<Tensor>& src);
void CopyOutput(Backend& b, Queue& q, std::map<std::string, Tensor>& dst,
                const std::map<std::string, Tensor>& src);

namespace detail {
template <class T, class = void>
struct HasCopyOutput : std::false_type {};
template <class T>
struct HasCopyOutput<
    T, std::void_t<decltype(CopyOutput(std::declval<Backend&>(), std::declval<Queue&>(),
                                       std::declval<T&>(), std::declval<const T&>()))>>
    : std::true_type {};

// Bumps GraphBreakStats::break_points_reached. Declared here because the
// GraphBreak templates are header-defined and the counter is not.
void CountBreakPoint();
}  // namespace detail

// ---------------------------------------------------------------------------
// G3 observability (spec `## Gates` G3).
// ---------------------------------------------------------------------------
// Without a count there is no way to tell a two-segment capture from a fully
// eager step, and "the graph ran" is exactly the claim a broken instrument
// fabricates. Each existing driver carries a private `replay_count()` and no
// segment count exists at all.
struct GraphBreakStats {
  int64_t break_points_reached = 0;  // GraphBreak call sites executed, capturing or not
  int64_t segments_captured = 0;
  int64_t breaks_registered = 0;
  int64_t replays = 0;  // BreakableGraph::Replay calls
};
GraphBreakStats GetGraphBreakStats();
void ResetGraphBreakStats();

// The one kill switch, read ONCE per process into a function-local static.
// Today six drivers each read `VLLM_CPP_CUDAGRAPH` for themselves and the three
// single-shape drivers invented their own switch instead, so there is no one
// switch that turns capture off; this is that switch for the seam.
bool GraphCaptureEnabled();

class GraphCaptureScope;

// ---------------------------------------------------------------------------
// vt::BreakableGraph — the segment container. Port of `BreakableCUDAGraph`
// (`:246-274`); fields `:251-252`, interleaved replay `:255-264`.
// ---------------------------------------------------------------------------
//
// Holds one opaque handle per segment (each from `Backend::EndCaptureGraph`) and
// one seam-built replay closure per break. The invariant is
// `segment_count() == break_count() + 1` for any capture that completed.
//
// A segment handle is treated as OPAQUE and every acquisition and release goes
// through `Backend::EndCaptureGraph` and `Backend::DestroyGraph`, so
// ENG-CUDAGRAPH-DEDUP (#1162) can interpose at the backend without editing this
// container (spec `## Risks/decisions` D4).
class BreakableGraph {
 public:
  BreakableGraph() = default;
  ~BreakableGraph();
  BreakableGraph(const BreakableGraph&) = delete;
  BreakableGraph& operator=(const BreakableGraph&) = delete;

  // segment 0, break 0, segment 1, break 1, ..., segment N. Interleaved, never
  // batched: the break between two segments runs BETWEEN their replays.
  void Replay(Queue& q);

  size_t segment_count() const { return segments_.size(); }
  size_t break_count() const { return break_fns_.size(); }
  bool captured() const { return !segments_.empty(); }
  int64_t replay_count() const { return replays_; }

  // Releases every segment through Backend::DestroyGraph and clears the breaks.
  void Reset();

 private:
  friend class GraphCaptureScope;
  Backend* backend_ = nullptr;
  std::vector<void*> segments_;
  std::vector<std::function<void()>> break_fns_;
  int64_t replays_ = 0;
};

// ---------------------------------------------------------------------------
// vt::GraphCaptureScope — the RAII capture scope. Port of
// `BreakableCUDAGraphCapture` (`:277-367`); `__enter__` `:309-320`, `__exit__`
// `:322-333`, segment open and close `:335-350` and `:352-367`.
// ---------------------------------------------------------------------------
//
// The scope pointer is THREAD-LOCAL, not global, for the same reason upstream
// chose a `ContextVar` (`:63`) and our CUDA backend chose
// `cudaStreamCaptureModeThreadLocal` (`src/vt/cuda/cuda_backend.cu:204-206`):
// capture is a property of one thread's stream, and a process-wide flag would
// make an unrelated thread's forward observe a capture it is not part of.
//
// INERT when the backend cannot capture (`SupportsGraphCapture()` false, which
// is Vulkan `vulkan_backend.cpp:16` and Metal `metal_backend.mm:13`) or when
// `VLLM_CPP_CUDAGRAPH=0`. An inert scope captures nothing, every `GraphBreak`
// inside it is a pass-through, and the forward runs eager exactly as today.
class GraphCaptureScope {
 public:
  GraphCaptureScope(Backend& b, Queue& q, BreakableGraph& out);
  ~GraphCaptureScope();
  GraphCaptureScope(const GraphCaptureScope&) = delete;
  GraphCaptureScope& operator=(const GraphCaptureScope&) = delete;

  bool active() const { return active_; }
  static GraphCaptureScope* Current();

  Backend& backend() const { return *b_; }
  Queue& queue() const { return *q_; }

  // Called by GraphBreak. Closes the segment that captured up to the break
  // point, and opens a fresh one for the remainder of the forward.
  void EndSegment();
  void BeginSegment();
  void AppendBreak(std::function<void()> fn);

 private:
  Backend* b_;
  Queue* q_;
  BreakableGraph* g_;
  bool active_ = false;
  bool segment_open_ = false;
  GraphCaptureScope* prev_ = nullptr;
};

// ---------------------------------------------------------------------------
// vt::GraphBreak — the break point. Port of `eager_on_graph` (`:204-243`),
// whose wrapper body is `:209-241`.
// ---------------------------------------------------------------------------
//
// The SITE IS THE REGISTRATION. vLLM registers `splitting_ops` by operation NAME
// because it has an FX graph to match names against; we have neither, so our
// equivalent is one line at the break site, exactly as SGLang's is
// (`layers/radix_attention.py:256`, `layers/radix_linear_attention.py:159`).
//
// OUTSIDE a capture scope every form calls the function and returns, so a
// non-capturing forward is byte-identical to today and makes ZERO backend calls.
//
// THREE LIFETIME RULES, stated here because they are the rules a new model
// author is least likely to know, and because two of the three produce wrong
// numbers rather than a fault:
//
//   1. The DESTINATION must outlive the `BreakableGraph`, and no replay may
//      reallocate it. The following segment bakes its address at capture time.
//   2. A closure must capture DEVICE POINTERS or `Tensor` views that are stable
//      across replays, never a reference to a host temporary that dies with the
//      capturing call.
//   3. Capture bakes the HOST SOURCE ADDRESS of an upload as well as the device
//      destination (spec D2, `.agents/specs/decode-graph-scratch-uaf-2026-07-18.md`).
//      A break function that reads a host vector which is reallocated between
//      capture and replay reads freed host memory on replay. A clean
//      `compute-sanitizer` run is NOT evidence against any of these: the
//      recorded incident reproduced 5 of 5 times in normal operation and 0 of 2
//      under memcheck.

// The BARE marker. Splits the segment and runs nothing — the form a model uses
// when the host-dependent work already ran outside the forward. Port of
// `break_graph` (`:370-374`), an empty body under the decorator.
void GraphBreak();

// Form ONE, in place. `fn` writes into a PERSISTENT buffer the model owns and
// that no replay reallocates, and returns nothing. The writeback contract is
// vacuous for it, because there is no fresh allocation to write back.
template <class Fn, class = std::enable_if_t<std::is_void_v<decltype(std::declval<Fn&>()())>>>
void GraphBreak(Fn&& fn) {
  GraphCaptureScope* s = GraphCaptureScope::Current();
  if (s == nullptr || !s->active()) {
    fn();  // pass-through: byte-identical to today, zero backend calls
    detail::CountBreakPoint();
    return;
  }
  s->EndSegment();
  fn();  // run once eagerly, so the outputs hold real data (`:222-223`)
  s->AppendBreak([f = std::forward<Fn>(fn)]() mutable { f(); });
  s->BeginSegment();
  detail::CountBreakPoint();
}

// Form TWO, destination-carrying. `fn` returns a fresh result and the seam
// copies it into `out` on every replay. Direct port of `replay_fn` plus
// `_copy_output` (`:231-235`, `:172-201`), with `out` playing the part
// `_weak_ref_if_tensor` (`:156-169`) plays upstream: the capture-time
// destination that outlives the eager call.
template <class Fn, class Out>
void GraphBreak(Fn&& fn, Out& out) {
  GraphCaptureScope* s = GraphCaptureScope::Current();
  if (s == nullptr || !s->active()) {
    out = fn();  // pass-through: a move, never a copy
    detail::CountBreakPoint();
    return;
  }
  s->EndSegment();
  out = fn();  // the capture-time destination IS this first result (`:222-227`)
  if constexpr (detail::HasCopyOutput<Out>::value) {
    Backend* b = &s->backend();
    Queue* q = &s->queue();
    Out* dst = &out;  // lifetime rule 1 above
    s->AppendBreak([b, q, dst, f = std::forward<Fn>(fn)]() mutable {
      CopyOutput(*b, *q, *dst, f());
    });
  } else {
    // The non-copyable fallback (`:201`). Upstream returns `src` and leaves the
    // destination untouched; so does this. Such a break must use form ONE.
    s->AppendBreak([f = std::forward<Fn>(fn)]() mutable { (void)f(); });
  }
  s->BeginSegment();
  detail::CountBreakPoint();
}

}  // namespace vt
