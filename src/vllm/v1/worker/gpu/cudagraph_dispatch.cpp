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

void NoteGraphDispatch(int64_t query_len, int64_t configured_query_len) {
  GraphDispatchStats& s = Stats();
  if (query_len <= 0) {
    ++s.ragged_steps;
    return;
  }
  ++s.uniform_steps;
  if (query_len > 1) {
    ++s.uniform_spec_steps;
    if (query_len < configured_query_len) ++s.clamped_spec_steps;
  }
}

void NoteDecodeGraphShape() { ++Stats().capture_shapes; }
void NoteDecodeGraphQueryLenDecline() { ++Stats().qlen_cap_declines; }

}  // namespace v1
}  // namespace vllm
