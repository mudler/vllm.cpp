// Ported from: vllm/multimodal/utils.py::get_mm_features_in_window (:116) @ vLLM
// 5559679229. See include/vllm/multimodal/utils.h.
#include "vllm/multimodal/utils.h"

#include <algorithm>

namespace vllm::multimodal {

std::pair<int, int> GetMmFeaturesInWindow(
    const std::vector<MultiModalFeatureSpec>& mm_features, int start, int end) {
  const int n = static_cast<int>(mm_features.size());
  // bisect_left(mm_features, start + 1, key=offset + length): the first item
  // whose END is at or past start + 1, i.e. the first that is not entirely
  // before `start`.
  int lo = 0;
  {
    int a = 0, b = n;
    while (a < b) {
      const int mid = a + (b - a) / 2;
      const MultiModalFeatureSpec& f = mm_features[static_cast<size_t>(mid)];
      if (f.offset + f.length < start + 1) {
        a = mid + 1;
      } else {
        b = mid;
      }
    }
    lo = a;
  }
  // bisect_left(mm_features, end, key=offset): the first item that STARTS at or
  // past `end`, i.e. one past the last that overlaps.
  int hi = 0;
  {
    int a = 0, b = n;
    while (a < b) {
      const int mid = a + (b - a) / 2;
      if (mm_features[static_cast<size_t>(mid)].offset < end) {
        a = mid + 1;
      } else {
        b = mid;
      }
    }
    hi = a;
  }
  // Upstream returns the RAW pair, and so does this. A degenerate window
  // (start >= end) can make lo > hi, and every caller's `for (i = lo; i < hi)`
  // then runs zero times — exactly what upstream's `mm_features[lo:hi]` slice
  // does. Clamping `hi` up to `lo` would read as harmless and would silently
  // stop reproducing the oracle's return value for that case, which is what a
  // ported unit test compares.
  return {lo, hi};
}

}  // namespace vllm::multimodal
