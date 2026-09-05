// Multimodal placeholder-window helpers — C++ mirror of vllm/multimodal/utils.py.
//
// Ported from: vllm/multimodal/utils.py::get_mm_features_in_window (:116) @ vLLM
// 5559679229. `grep -rn 'def get_mm_features_in_window' vllm/` == 1, so the
// anchor is unique.
//
// Both the SCHEDULER (_try_schedule_encoder_inputs) and the RUNNER
// (_gather_mm_embeddings) need the SAME answer to "which multimodal items
// overlap the token range this step covers", and upstream shares one function
// between them. Sharing it here is what keeps the two from drifting: a
// scheduler that admits an encoder input the runner does not gather (or the
// reverse) is an encoder-cache miss, which is a throw on the serving path.
#ifndef VLLM_MULTIMODAL_UTILS_H_
#define VLLM_MULTIMODAL_UTILS_H_

#include <utility>
#include <vector>

#include "vllm/multimodal/inputs.h"

namespace vllm::multimodal {

// Return the [lo, hi) index range of `mm_features` overlapping [start, end).
//
// Assumes the features are sorted by offset and non-overlapping, so offset +
// length is sorted too — exactly the precondition upstream's docstring states.
// Upstream uses bisect with a key; std::lower_bound over the index range is the
// same predicate:
//   lo = first i with offset[i] + length[i] >= start + 1
//   hi = first i with offset[i]            >= end
// An empty feature list, or a range that touches none of them, returns lo == hi
// and every caller's loop body runs zero times — which is the text path. A
// DEGENERATE window (start >= end) can return lo > hi; the pair is returned raw,
// exactly as upstream returns it, and `for (i = lo; i < hi; ++i)` is empty for
// it just as the Python slice is.
std::pair<int, int> GetMmFeaturesInWindow(
    const std::vector<MultiModalFeatureSpec>& mm_features, int start, int end);

}  // namespace vllm::multimodal

#endif  // VLLM_MULTIMODAL_UTILS_H_
