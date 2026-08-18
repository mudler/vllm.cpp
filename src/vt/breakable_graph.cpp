// Implementation of the break-point capture seam. See
// `include/vt/breakable_graph.h` for the port map and the lifetime rules, and
// `.agents/specs/eng-cudagraph-break.md` for the design. Row
// ENG-CUDAGRAPH-BREAK W1, issue #1192, parent #1163.
//
// Upstream: SGLang at pin `f63458b5be`,
// `python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py`;
// every bare `:N` is a line in that file.
#include "vt/breakable_graph.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace vt {
namespace {

// The capture scope of the CURRENT THREAD. Replaces upstream's `ContextVar`
// (`:63`); see the header for why this is thread-local and not global.
thread_local GraphCaptureScope* t_current_scope = nullptr;

std::atomic<int64_t> g_break_points{0};
std::atomic<int64_t> g_segments{0};
std::atomic<int64_t> g_breaks{0};
std::atomic<int64_t> g_replays{0};

}  // namespace

namespace detail {
void CountBreakPoint() { g_break_points.fetch_add(1, std::memory_order_relaxed); }
}  // namespace detail

GraphBreakStats GetGraphBreakStats() {
  GraphBreakStats s;
  s.break_points_reached = g_break_points.load(std::memory_order_relaxed);
  s.segments_captured = g_segments.load(std::memory_order_relaxed);
  s.breaks_registered = g_breaks.load(std::memory_order_relaxed);
  s.replays = g_replays.load(std::memory_order_relaxed);
  return s;
}

void ResetGraphBreakStats() {
  g_break_points.store(0, std::memory_order_relaxed);
  g_segments.store(0, std::memory_order_relaxed);
  g_breaks.store(0, std::memory_order_relaxed);
  g_replays.store(0, std::memory_order_relaxed);
}

// Read ONCE into a function-local static, so a process is in exactly one lane
// for its whole life and nothing can toggle it mid-run.
bool GraphCaptureEnabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("VLLM_CPP_CUDAGRAPH");
    return e == nullptr || std::string(e) != "0";
  }();
  return enabled;
}

// ---------------------------------------------------------------------------
// CopyOutput — port of `_copy_output` (`:172-201`).
// ---------------------------------------------------------------------------

void CopyOutput(Backend& b, Queue& q, Tensor& dst, const Tensor& src) {
  if (dst.data == nullptr || src.data == nullptr) return;
  if (dst.data == src.data) return;  // already the same storage: nothing to do
  size_t numel = 1;
  for (int i = 0; i < dst.rank; ++i) numel *= static_cast<size_t>(dst.shape[i]);
  const size_t bytes = numel * SizeOf(dst.dtype);
  // In place, into the address the FOLLOWING SEGMENT baked. `dst` is not
  // rebound, which is upstream's `assertIs(result, dst)` (`:190`).
  b.Copy(q, dst.data, src.data, bytes);
}

void CopyOutput(Backend& b, Queue& q, std::vector<Tensor>& dst,
                const std::vector<Tensor>& src) {
  const size_t n = dst.size() < src.size() ? dst.size() : src.size();
  for (size_t i = 0; i < n; ++i) CopyOutput(b, q, dst[i], src[i]);
}

void CopyOutput(Backend& b, Queue& q, std::map<std::string, Tensor>& dst,
                const std::map<std::string, Tensor>& src) {
  for (const auto& kv : src) {
    auto it = dst.find(kv.first);
    // A key the destination holds is copied IN PLACE; a key it does not hold is
    // ASSIGNED, which is upstream's `dst[key] = src_val` branch (`:200`).
    if (it != dst.end())
      CopyOutput(b, q, it->second, kv.second);
    else
      dst[kv.first] = kv.second;
  }
}

// ---------------------------------------------------------------------------
// BreakableGraph — port of `BreakableCUDAGraph` (`:246-274`).
// ---------------------------------------------------------------------------

BreakableGraph::~BreakableGraph() { Reset(); }

void BreakableGraph::Reset() {
  if (backend_ != nullptr) {
    for (void* g : segments_) backend_->DestroyGraph(g);
  }
  segments_.clear();
  break_fns_.clear();
}

void BreakableGraph::Replay(Queue& q) {
  if (backend_ == nullptr || segments_.empty()) return;
  // Interleaved, mirroring `:255-263`: segment i, then break i.
  for (size_t i = 0; i < segments_.size(); ++i) {
    backend_->ReplayGraph(q, segments_[i]);
    if (i < break_fns_.size()) break_fns_[i]();
  }
  ++replays_;
  g_replays.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// GraphCaptureScope — port of `BreakableCUDAGraphCapture` (`:277-367`).
// ---------------------------------------------------------------------------

GraphCaptureScope* GraphCaptureScope::Current() { return t_current_scope; }

GraphCaptureScope::GraphCaptureScope(Backend& b, Queue& q, BreakableGraph& out)
    : b_(&b), q_(&q), g_(&out) {
  active_ = GraphCaptureEnabled() && b.SupportsGraphCapture();
  if (!active_) return;  // inert: the forward runs eager, exactly as today
  g_->backend_ = b_;
  prev_ = t_current_scope;
  t_current_scope = this;
  BeginSegment();
}

GraphCaptureScope::~GraphCaptureScope() {
  if (!active_) return;
  // `__exit__` closes the final segment inside a `finally` (`:322-333`), so the
  // scope pointer is restored even when the close throws. A destructor that
  // propagates would terminate, and a skipped close poisons the stream
  // permanently — which is why three drivers hand-rolled this same drain
  // (`qwen3_5.cpp:9913`, `qwen3_5.cpp:10335`, `qwen3_dflash.cpp:1106`).
  try {
    EndSegment();
  } catch (...) {
    g_->Reset();
  }
  t_current_scope = prev_;
}

void GraphCaptureScope::BeginSegment() {
  if (!active_ || segment_open_) return;
  b_->BeginCapture(*q_);
  segment_open_ = true;
}

void GraphCaptureScope::EndSegment() {
  if (!active_ || !segment_open_) return;
  segment_open_ = false;  // cleared FIRST: a throwing end must not be retried
  void* seg = b_->EndCaptureGraph(*q_);
  g_->segments_.push_back(seg);
  g_segments.fetch_add(1, std::memory_order_relaxed);
}

void GraphCaptureScope::AppendBreak(std::function<void()> fn) {
  if (!active_) return;
  g_->break_fns_.push_back(std::move(fn));
  g_breaks.fetch_add(1, std::memory_order_relaxed);
}

// The bare marker (`:370-374`).
void GraphBreak() {
  GraphCaptureScope* s = GraphCaptureScope::Current();
  if (s == nullptr || !s->active()) {
    detail::CountBreakPoint();
    return;
  }
  s->EndSegment();
  s->AppendBreak([] {});
  s->BeginSegment();
  detail::CountBreakPoint();
}

}  // namespace vt
