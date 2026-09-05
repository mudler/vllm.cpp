// spec_acceptance.h — the ONE mapping from the runner's speculative counters
// onto the `vllm_spec_acceptance` struct the C ABI publishes (ABI v25, row
// `SPEC-DFLASH2`, issue #2832).
//
// WHY THIS IS NOT THREE LINES INSIDE `vllm_c.cpp`. It was, and nothing pinned
// it. A fresh review swapped `drafts_proposed` with `drafts_accepted` AND
// changed `front()` to `back()` in the accessor's body and got a FULL GREEN:
// the focused gate, the release-binary contract and the surface-coverage
// checker all passed, and the capi suite could not have caught it either,
// because its cases reach the accessor only with a NULL argument and return at
// the null guard before touching the mapping at all. The only thing that CAN
// reach the mapping over there is a case that builds an engine, and building an
// engine means linking the whole 559-translation-unit library — which is
// exactly the link the authoring host could not afford and which left the capi
// suite unrun in the first place.
//
// So the mapping lives here, as a template over anything that carries the three
// runner accessors. `tests/tools/test_dflash2_speed_harness.py` compiles this
// header into a small probe with a stand-in runner and RUNS it, on any box, with
// no library and no GPU — and both halves of that mutation go red there.
//
// WHAT THAT PIN DOES AND DOES NOT COVER, stated because the bound matters. It
// covers WHICH counter becomes WHICH field and WHICH element of the per-depth
// vector is read, which is the whole of what the mutation moved. It does NOT
// cover that the object passed in is the engine's own runner: that is the one
// expression left at the call site, it has no operand order to get wrong, and
// `SpecAcceptanceSourceContractTest` holds `vllm_c.cpp` to delegating here
// rather than growing a second copy of the mapping.
#pragma once

#include <cstdint>

#include "vllm.h"

namespace vllm::capi {

// Copy `runner`'s three speculative counters into `*out`. COMPUTES NOTHING:
// every value is a counter the verify write-back already incremented.
//
// `drafted_request_steps` comes from `spec_drafts_proposed_by_depth()[0]`
// rather than from a counter of its own, and that is the SAME already-computed
// value: the runner increments index 0 exactly once for each (request, step)
// whose draft list was non-empty. Index 0 and NOT `back()` — `by_depth[d]`
// counts the request-steps whose draft list was longer than `d`, so on a ragged
// or mixed batch the last element counts only the rows that drafted to the
// deepest k ever seen, which would silently inflate every per-step figure
// computed from it. The vector is EMPTY on an engine that never speculated, and
// 0 is then the true answer rather than an error.
//
// `out` must be non-null; the C ABI entry point owns that guard, because it also
// owns the error string that goes with it.
template <typename Runner>
void FillSpecAcceptance(const Runner& runner, vllm_spec_acceptance* out) {
  out->drafts_proposed = runner.spec_drafts_proposed();
  out->drafts_accepted = runner.spec_drafts_accepted();
  const auto& by_depth = runner.spec_drafts_proposed_by_depth();
  out->drafted_request_steps =
      by_depth.empty() ? static_cast<int64_t>(0)
                       : static_cast<int64_t>(by_depth.front());
}

}  // namespace vllm::capi
