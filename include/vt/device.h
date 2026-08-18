// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <cstddef>
#include <cstdint>

namespace vt {

// Queue identities are process-unique and monotonic. A native stream handle is
// not an identity by itself: CUDA's per-device default streams are all null and
// destroyed streams may later reuse the same handle value.
uint64_t NextQueueId() noexcept;

// Open device enum (.agents/backends.md): reserved entries for platforms we
// have not implemented yet keep engine-visible types backend-agnostic.
enum class DeviceType : uint8_t {
  kCPU = 0,
  kCUDA = 1,
  kMETAL = 2,
  kVULKAN = 3,
  kXPU = 4,
  kROCM = 5,
  // Tenstorrent Blackhole (P100/P150 PCIe cards). Named after the vendor/stack
  // (kROCM precedent), not the chip family: this codebase's CUDA layer already
  // uses "Blackwell" (sm_120/121, GB10) pervasively, and kBLACKHOLE next to
  // those would be a near-miss for both humans and grep.
  // (.agents/specs/tenstorrent-backend.md, BACKEND-TENSTORRENT)
  kTENSTORRENT = 6
};
constexpr size_t kNumDeviceTypes = 7;

// The canonical lowercase spelling of a device, for user-facing messages (and
// the docs that quote them). Lives here, beside the enum, rather than in the
// shared vllm layer: it names every platform EQUALLY, so it is a data list like
// the platform priority walk, not a device-specific branch — and keeping it in
// vt means adding a platform touches one enum and one switch, both in this file.
constexpr const char* DeviceTypeName(DeviceType device) {
  switch (device) {
    case DeviceType::kCPU:
      return "cpu";
    case DeviceType::kCUDA:
      return "cuda";
    case DeviceType::kMETAL:
      return "metal";
    case DeviceType::kVULKAN:
      return "vulkan";
    case DeviceType::kXPU:
      return "xpu";
    case DeviceType::kROCM:
      // Upstream `vllm/platforms/rocm.py` sets `device_name = "rocm"` while its
      // torch-facing `device_type` stays "cuda" (rocm.py:447-449) because ROCm
      // reuses the CUDA dispatch key. We have no torch, so only the honest name
      // survives here; the HIP-reuses-CUDA-spelling question does not arise.
      return "rocm";
    case DeviceType::kTENSTORRENT:
      return "tenstorrent";
  }
  return "unknown";
}

// The inverse of `DeviceTypeName`: resolves a lowercase spelling back to its
// enum, returning false when nothing matches. It lives here, beside the forward
// direction, for the same reason that one does — it names every platform
// EQUALLY, so it is a data list rather than a device-specific branch, and adding
// a platform still touches one enum and this one file.
//
// It exists because the shared `vllm` layer has to honour a device NAME an
// operator typed (`VLLM_CPP_VOCODER_DEVICE`, #672) WITHOUT either spelling a
// device enumerator or casting an integer into one. Both are exactly what
// `scripts/check-device-leakage.py` counts, and it is right to: a cast hardcodes
// a device by ENUM VALUE, never writes the token, and silently re-points if the
// enum is ever reordered. Enumerating the list HERE keeps that hazard inside the
// seam that owns the enum, and the static_assert makes adding a platform without
// listing it a build error rather than a silent gap.
inline bool DeviceTypeFromName(const char* name, DeviceType* out) {
  if (name == nullptr || out == nullptr) return false;
  constexpr DeviceType kAll[] = {DeviceType::kCPU,    DeviceType::kCUDA, DeviceType::kMETAL,
                                 DeviceType::kVULKAN, DeviceType::kXPU,  DeviceType::kROCM,
                                 DeviceType::kTENSTORRENT};
  static_assert(sizeof(kAll) / sizeof(kAll[0]) == kNumDeviceTypes,
                "DeviceTypeFromName must list every DeviceType");
  for (const DeviceType device : kAll) {
    const char* a = name;
    const char* b = DeviceTypeName(device);
    while (*a != '\0' && *a == *b) {
      ++a;
      ++b;
    }
    if (*a == '\0' && *b == '\0') {
      *out = device;
      return true;
    }
  }
  return false;
}

struct Device {
  DeviceType type = DeviceType::kCPU;
  int32_t index = 0;
  friend bool operator==(const Device& a, const Device& b) {
    return a.type == b.type && a.index == b.index;
  }
};

// Per-device execution queue (CUDA stream / Metal command queue / SYCL queue).
// CPU uses handle == nullptr. Ops never assume a global stream.
struct Queue {
  Device device;
  void* handle = nullptr;
  uint64_t id = NextQueueId();
};

}  // namespace vt
