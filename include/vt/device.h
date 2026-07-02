// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <cstddef>
#include <cstdint>

namespace vt {

// Open device enum (.agents/backends.md): reserved entries for platforms we
// have not implemented yet keep engine-visible types backend-agnostic.
enum class DeviceType : uint8_t { kCPU = 0, kCUDA = 1, kMETAL = 2, kVULKAN = 3, kXPU = 4 };
constexpr size_t kNumDeviceTypes = 5;

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
};

}  // namespace vt
