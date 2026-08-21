// ROCm wvSplitK architecture eligibility. This header stays free of HIP
// headers so a controlled resolver can gate device-hop behavior on any host.
#pragma once

#include <string>
#include <vector>

#include "vt/rocm/rocm_arch.h"

namespace vt::rocm {

using SkinnyGemmArchResolver = std::string (*)(int) noexcept;

// The wvSplitK port carries only the wave32 reduction arm. Upstream uses
// ROW_BCAST15/31 on gfx9 wave64 devices, and that arm is not ported. The
// predicate accepts gfx11 and gfx12 only. An unknown architecture refuses.
namespace detail {

class SkinnyGemmArchCache {
 public:
  bool Eligible(int device_index, SkinnyGemmArchResolver resolve) {
    for (const Entry& entry : entries_) {
      if (entry.device_index == device_index && entry.resolve == resolve) {
        return entry.eligible;
      }
    }

    const auto cap = CapabilityFromGcnArch(resolve(device_index));
    const bool eligible = cap.has_value() && (cap->first == 11 || cap->first == 12);
    entries_.push_back(Entry{device_index, resolve, eligible});
    return eligible;
  }

 private:
  struct Entry {
    int device_index;
    SkinnyGemmArchResolver resolve;
    bool eligible;
  };

  std::vector<Entry> entries_;
};

}  // namespace detail

inline bool SkinnyGemmArchOk(int device_index, SkinnyGemmArchResolver resolve) {
  // Each worker reads a device property once per device. Per-thread storage
  // keeps the decode path free of a process-wide lock.
  static thread_local detail::SkinnyGemmArchCache cache;
  return cache.Eligible(device_index, resolve);
}

}  // namespace vt::rocm
