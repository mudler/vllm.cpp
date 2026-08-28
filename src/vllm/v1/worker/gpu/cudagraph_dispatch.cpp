// The dispatch counters declared in cudagraph_dispatch.h.
//
// ENG-CUDAGRAPH-BREAK W6 (#1374), spec `.agents/specs/eng-cudagraph-break.md`,
// parent #1163, closing [#1020](https://github.com/mudler/vllm.cpp/issues/1020).
//
// WHY A TRANSLATION UNIT FOR THREE INTEGERS. The predicate itself stays inline
// shape arithmetic, because it is on the per-step path and has no state. The
// counters cannot be inline: a header-defined mutable global gets one copy per
// translation unit under the one-definition rule's vague-linkage rules only if
// it is declared `inline`, and a gate that reads one copy while the runner
// writes another is the "broken instrument fails toward a code verdict" shape.
// One definition, here.
//
// NOT THREAD-SAFE BY DESIGN, and the choice mirrors `vt::GraphBreakStats`
// (`src/vt/breakable_graph.cpp`): these are single-process diagnostics read by a
// gate that drives one runner on one thread. An atomic would suggest a
// cross-thread guarantee nothing here provides.
#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"

#include <cstdio>
#include <cstdlib>

namespace vllm {
namespace v1 {
namespace {
GraphDispatchStats& Stats() {
  static GraphDispatchStats s;
  return s;
}
}  // namespace

GraphDispatchStats GetGraphDispatchStats() { return Stats(); }
void ResetGraphDispatchStats() { Stats() = GraphDispatchStats{}; }

void NoteGraphDispatch(int64_t query_len, int64_t configured_query_len,
                       const RaggedStepShape& ragged_shape) {
  GraphDispatchStats& s = Stats();
  if (query_len <= 0) {
    ++s.ragged_steps;
    // W13 (#2117): the three buckets PARTITION `ragged_steps`, so the branch is
    // an if/else-if/else and never three independent tests. A step with neither
    // bit set has no rows at all and lands in the spec-only bucket, which is the
    // only bucket that does not assert a prefill row was present.
    if (ragged_shape.has_prefill_row && ragged_shape.has_decode_row) {
      ++s.ragged_mixed_steps;
    } else if (ragged_shape.has_prefill_row) {
      ++s.ragged_prefill_only_steps;
    } else {
      ++s.ragged_spec_only_steps;
    }
    return;
  }
  ++s.uniform_steps;
  if (query_len > 1) {
    ++s.uniform_spec_steps;
    if (query_len < configured_query_len) ++s.clamped_spec_steps;
  }
}

int64_t GraphStatsDumpPeriod() {
  const char* v = std::getenv("VT_GRAPH_STATS");
  if (v == nullptr || v[0] == '\0') return 0;
  const long long n = std::strtoll(v, nullptr, 10);
  return n > 0 ? static_cast<int64_t>(n) : 0;
}

std::string FormatGraphDispatchStats(const GraphDispatchStats& s) {
  const int64_t total = GraphDispatchTotalSteps(s);
  // The share #2117 asks for, computed here rather than by the reader, because
  // a reader who has to divide is a reader who will divide by the wrong
  // denominator once.
  const double ragged_pct =
      total > 0 ? 100.0 * static_cast<double>(s.ragged_steps) /
                      static_cast<double>(total)
                : 0.0;
  char buf[512];
  const int n = std::snprintf(
      buf, sizeof(buf),
      "[graph-dispatch] steps=%lld uniform=%lld uniform_spec=%lld "
      "clamped_spec=%lld ragged=%lld ragged_mixed=%lld ragged_prefill=%lld "
      "ragged_spec=%lld ragged_pct=%.1f spec_as_decode=%lld capture_shapes=%lld "
      "qlen_cap_declines=%lld",
      static_cast<long long>(total), static_cast<long long>(s.uniform_steps),
      static_cast<long long>(s.uniform_spec_steps),
      static_cast<long long>(s.clamped_spec_steps),
      static_cast<long long>(s.ragged_steps),
      static_cast<long long>(s.ragged_mixed_steps),
      static_cast<long long>(s.ragged_prefill_only_steps),
      static_cast<long long>(s.ragged_spec_only_steps), ragged_pct,
      static_cast<long long>(s.spec_as_decode_steps),
      static_cast<long long>(s.capture_shapes),
      static_cast<long long>(s.qlen_cap_declines));
  return n > 0 ? std::string(buf, static_cast<size_t>(n) < sizeof(buf)
                                      ? static_cast<size_t>(n)
                                      : sizeof(buf) - 1)
               : std::string();
}

void NoteDecodeGraphShape() { ++Stats().capture_shapes; }
void NoteSpecAsDecode() { ++Stats().spec_as_decode_steps; }
void NoteDecodeGraphQueryLenDecline() { ++Stats().qlen_cap_declines; }

}  // namespace v1
}  // namespace vllm
