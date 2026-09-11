#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vt::rocm::detail {

enum class MmvqRoute : uint8_t { kBaseline = 0, kGemv = 1, kFused = 2 };

inline constexpr int64_t kMmvqFoldMaxRowsDefault = 512;
inline constexpr size_t kMmvqFoldLdsBytes = 32 * 1024;

inline int64_t ParseMmvqFoldMaxRows(const char* value) {
  if (value == nullptr || value[0] == '\0') return kMmvqFoldMaxRowsDefault;
  int64_t parsed = 0;
  const char* end = value + std::strlen(value);
  const auto result = std::from_chars(value, end, parsed, 10);
  if (result.ec != std::errc{} || result.ptr != end || parsed <= 0) {
    return kMmvqFoldMaxRowsDefault;
  }
  return parsed;
}

inline MmvqRoute SelectMmvqRoute(const char* enabled, const char* fold_max,
                                 int64_t m, int64_t n,
                                 size_t activation_scratch_bytes) {
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0 || m != 1) {
    return MmvqRoute::kBaseline;
  }
  if (n <= ParseMmvqFoldMaxRows(fold_max) &&
      activation_scratch_bytes <= kMmvqFoldLdsBytes) {
    return MmvqRoute::kFused;
  }
  return MmvqRoute::kGemv;
}

}  // namespace vt::rocm::detail

namespace vt::rocm {

struct MmvqRouteCounts {
  uint64_t baseline;
  uint64_t gemv;
  uint64_t fused;
};

void MmvqResetRouteCountsForTesting();
MmvqRouteCounts MmvqRouteCountsForTesting();

}  // namespace vt::rocm
