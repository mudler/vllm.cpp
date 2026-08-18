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
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"  // VT_CHECK
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
// (`:190`). A type with NO overload cannot use the destination form at all: the
// static assertion in `GraphBreak` below refuses it by name, and the documented
// non-copyable fallback (`:201`) has to be asked for with `vt::NoWriteback{}`.
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

// The EXPLICIT opt-in to upstream's non-copyable fallback (`:201`), where
// `_copy_output` returns `src` and the destination is left untouched. Passing it
// says "this break has no device output and I know the seam writes nothing
// back". Without it a destination type with no `CopyOutput` overload is a
// COMPILE error rather than a silent downgrade to the wrong-numerics path, which
// is the whole reason the tag exists: `HasCopyOutput` as a plain `if constexpr`
// default made a misspelled, moved or shadowed overload compile clean and be
// wrong only on replay N.
struct NoWriteback {};

// ---------------------------------------------------------------------------
// vt::BreakSlot<T> — the destination of a destination-carrying break point.
// Port of `captured_output = _weak_ref_if_tensor(output)` (`:156-169`, `:225`).
// ---------------------------------------------------------------------------
//
// WHY THIS TYPE EXISTS AND A REFERENCE DOES NOT. The destination must outlive
// the `BreakableGraph`, because the following segment bakes its address at
// capture time and every replay writes through that address. Upstream gets that
// for free: `captured_output` is held BY VALUE in the replay closure and its
// storage is pinned by the segment graphs' mempool. A C++ seam taking a bare
// `Out&` gets the opposite — the natural call site declares the destination as a
// local of the function containing the break, which dies on the very next
// `return`, and the first such site did exactly that. A comment at the
// declaration is not a mechanism; every caller re-derives it and one of them is
// wrong.
//
// So the seam takes a slot whose storage it can OWN. On the capturing path the
// value is moved into a heap cell before the eager call produces it, and the
// replay closure holds a `shared_ptr` to that cell, so the destination lives
// exactly as long as the break function that writes into it — by construction,
// not by convention.
//
// On the NON-capturing path (every production step until a driver opens a
// scope) the cell is never allocated and the value lives inline in the slot, so
// a pass-through forward allocates nothing extra and is byte-identical to the
// plain local it replaces.
//
// Read the value through `*slot` or `slot->`, never through a pointer cached
// before the break: pinning moves the value, exactly as the destination form's
// `out = fn()` rebinds it (`:222-227`).
template <class T>
class BreakSlot {
 public:
  BreakSlot() = default;
  explicit BreakSlot(T v) : inline_(std::move(v)) {}

  T& operator*() { return cell_ ? *cell_ : inline_; }
  const T& operator*() const { return cell_ ? *cell_ : inline_; }
  T* operator->() { return cell_ ? cell_.get() : &inline_; }
  const T* operator->() const { return cell_ ? cell_.get() : &inline_; }

  // True once the slot's storage is owned by the seam rather than by this
  // object, which is the state the writeback contract requires.
  bool pinned() const { return static_cast<bool>(cell_); }

  // Called by GraphBreak ONLY, and only on the capturing path, BEFORE the eager
  // result is produced. Moving an as-yet-unwritten value is free, and from this
  // point the address is stable for the life of the closure that holds the cell.
  const std::shared_ptr<T>& PinForCapture() {
    if (!cell_) cell_ = std::make_shared<T>(std::move(inline_));
    return cell_;
  }

 private:
  T inline_{};
  std::shared_ptr<T> cell_;
};

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
// `segment_count() == break_count() + 1` for any capture that completed, and it
// is ASSERTED in two places rather than documented: `GraphCaptureScope` refuses
// to open on a container that already holds a capture, and `Replay` refuses a
// container whose counts disagree. A graph with `break_count() == segment_count()`
// would otherwise replay with its last break silently dropped.
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

  // Releases every segment through Backend::DestroyGraph, clears the breaks, and
  // returns the container to its as-constructed state — replay count included,
  // because a stale count on a reset container is a number that describes a
  // graph that no longer exists.
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
//
// THREE THINGS IT REFUSES, all `VT_CHECK`. The first two are refused at
// construction, before any backend call:
//
//   * A NESTED scope on the same thread. `BeginCapture` on a stream that is
//     already capturing is `CUDA_ERROR_ILLEGAL_STATE` on a real backend, and the
//     sequence a nested pair traces is not a legal capture on any of them.
//   * A container that already holds a capture. Re-entering appends to it, and
//     the result has `break_count() == segment_count()`, whose last break
//     `Replay` would silently drop. Call `Reset()` first.
//   * TWO BREAK POINTS SHARING ONE DESTINATION, refused at registration. Both
//     replay closures would write through the same address, so the earlier
//     writeback is overwritten and any segment that baked the earlier
//     destination reads the later break's data. See lifetime rule 1 below;
//     this one throws from `AppendBreak`, mid-capture, and the drain below is
//     what turns it into no capture at all.
//
// AND ONE THING IT DRAINS. If the scope exits by EXCEPTION — a break function
// that throws, or ordinary model code between two break points — the stream is
// taken out of capture and the half-built container is `Reset()`. A partial
// capture must never be reported as `captured()`, because a partial capture is
// replayable, and replaying half a forward is silently wrong numerics.
//
// THE DRAIN'S LIMIT, stated because it is not obvious. It compares
// `std::uncaught_exceptions()` against the depth at entry, so it sees an
// exception that is PROPAGATING at scope exit. An exception CAUGHT INSIDE the
// scope leaves `segment_open_ == false` with the rest of the forward
// uncaptured, and nothing is unwinding at scope exit, so `captured()` stays
// true over a forward missing its tail. Do not `try` around a break point
// inside a capture scope. Owed to W2 in the spec's `## Owed`.
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
  // `destination` is the CELL this break's replay closure writes through, or
  // `nullptr` for a form whose replay writes back nowhere. It is a REQUIRED
  // parameter, not an optional one, so a break cannot be registered without
  // stating where it lands: two breaks naming one destination are refused here
  // (see the aliasing paragraph at lifetime rule 1).
  //
  // The CELL and not the SLOT, because a slot's ADDRESS is not its identity.
  // The production site declares its slot as a local of `RunLayer`, so every
  // layer's slot occupies the SAME stack address and slot-address identity
  // refuses the correct program on layer 2. The cell is heap storage the
  // replay closures keep alive for the whole capture, so two live cells cannot
  // share an address — and the cell is what actually aliases.
  void AppendBreak(std::function<void()> fn, const void* destination);

 private:
  Backend* b_;
  Queue* q_;
  BreakableGraph* g_;
  bool active_ = false;
  bool segment_open_ = false;
  int uncaught_on_entry_ = 0;
  // One entry per registered break that names a destination, for the life of
  // this capture. A capture has as many breaks as the model has split points,
  // so a linear scan here is cheaper than the map that would replace it.
  std::vector<const void*> destinations_;
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
// THREE LIFETIME RULES. Rule 1 used to be a comment; it is now the type system,
// because the one site that had to obey it did not. Rules 2 and 3 remain rules a
// new model author is least likely to know, and both produce wrong numbers
// rather than a fault:
//
//   1. The DESTINATION must outlive the `BreakableGraph`, no replay may
//      reallocate it, and no two break points may share it. ENFORCED in two
//      different ways, because the rule has two halves and only one of them is
//      a lifetime:
//        * LIFETIME, by the TYPE. The destination form takes a
//          `vt::BreakSlot<T>` and the seam owns its storage for the life of the
//          replay closure. There is no overload taking a bare reference, so
//          handing the break a destination that dies first is not writable.
//        * ALIASING, by a REFUSAL. One slot reused for two break points in one
//          capture is still writable — `PinForCapture` hands back the cell it
//          already made — and it binds BOTH replay closures to the SAME
//          address, so the earlier writeback is overwritten and any segment
//          that baked the earlier destination reads the later break's data.
//          `AppendBreak` throws on the second registration naming a cell
//          already registered in this capture. The constraint is per CAPTURE,
//          not per slot for life: a later capture bakes its own addresses and
//          may reuse the slot. The NON-COPYABLE fallback is outside this
//          refusal and says so at its own declaration.
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

// The IN-PLACE form. `fn` writes into a PERSISTENT buffer the model owns and
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
  s->AppendBreak([f = std::forward<Fn>(fn)]() mutable { f(); }, nullptr);
  s->BeginSegment();
  detail::CountBreakPoint();
}

// The DESTINATION form. `fn` returns a fresh result and the seam copies it into
// the slot on every replay. Direct port of `replay_fn` plus `_copy_output`
// (`:231-235`, `:172-201`), with `BreakSlot` playing the part
// `_weak_ref_if_tensor` (`:156-169`) plays upstream: the capture-time
// destination that outlives the eager call.
//
// The static assertion is the one MEDIUM the fallback branch used to hide. When
// selecting the fallback was a silent `if constexpr` default, renaming or moving
// a `CopyOutput` overload left both suites green while the break dropped into
// the branch whose own comment says such a break must use the in-place form —
// the D9 wrong-numerics path, reached by a typo.
template <class Fn, class Out>
void GraphBreak(Fn&& fn, BreakSlot<Out>& out) {
  static_assert(detail::HasCopyOutput<Out>::value,
                "vt::GraphBreak destination form: no CopyOutput(Backend&, Queue&, Out&, "
                "const Out&) is visible for this destination type. Declare one (found by "
                "argument-dependent lookup) so every replay writes back into the address "
                "the following segment baked, or use the in-place form GraphBreak(fn), or "
                "ask for the non-copyable fallback explicitly with vt::NoWriteback{}.");
  GraphCaptureScope* s = GraphCaptureScope::Current();
  if (s == nullptr || !s->active()) {
    *out = fn();  // pass-through: a move, never a copy, and the cell stays unpinned
    detail::CountBreakPoint();
    return;
  }
  s->EndSegment();
  // Pin BEFORE the eager call, so the address the following segment bakes is the
  // one the replay closure owns. `:222-227`: the capture-time destination IS the
  // first eager result.
  std::shared_ptr<Out> cell = out.PinForCapture();
  *cell = fn();
  Backend* b = &s->backend();
  Queue* q = &s->queue();
  s->AppendBreak(
      [b, q, cell, f = std::forward<Fn>(fn)]() mutable { CopyOutput(*b, *q, *cell, f()); },
      cell.get());
  s->BeginSegment();
  detail::CountBreakPoint();
}

// The DESTINATION form's non-copyable fallback (`:201`), asked for by name.
// Upstream returns `src` and leaves the destination untouched; so does this. The
// slot then holds the CAPTURE-TIME value forever, which is correct only when
// nothing downstream of the break reads it on replay.
//
// It registers NO destination, and that is not an oversight: this form pins no
// cell and its replay closure writes back nowhere, so the aliasing refusal has
// nothing to compare. Two of these sharing one slot still overwrite each other
// at CAPTURE time, which is a narrower hazard than the writeback one and is not
// gated here.
template <class Fn, class Out>
void GraphBreak(Fn&& fn, BreakSlot<Out>& out, NoWriteback) {
  GraphCaptureScope* s = GraphCaptureScope::Current();
  if (s == nullptr || !s->active()) {
    *out = fn();
    detail::CountBreakPoint();
    return;
  }
  s->EndSegment();
  *out = fn();
  s->AppendBreak([f = std::forward<Fn>(fn)]() mutable { (void)f(); }, nullptr);
  s->BeginSegment();
  detail::CountBreakPoint();
}

}  // namespace vt
