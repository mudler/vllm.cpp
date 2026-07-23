// Faithful 1:1 port of vllm/platforms/interface.py:134-229 (class Platform) @
// pin e24d1b24 — the platform registry + shared capability logic. Copies the
// RegisterBackend/GetBackend static-init idiom (src/vt/backend.cpp).
#include "vllm/platforms/interface.h"

#include <array>
#include <string>

#include "vt/dtype.h"  // VT_CHECK

namespace vllm::platforms {

// interface.py:417-439 has_device_capability — is this platform's capability >=
// the required (major, minor)? Lexicographic on (major, minor), mirroring the
// DeviceCapability tuple comparison; false when there is no queryable
// capability (get_device_capability() -> None).
bool Platform::has_device_capability(int major, int minor) const {
  const DeviceCapability cap = get_device_capability();
  if (!cap.present()) return false;
  if (cap.major != major) return cap.major > major;
  return cap.minor >= minor;
}

// interface.py:441-476 is_device_capability_family — is the device capability any
// <major>.x? Mirrors upstream exactly: `(to_int() // 10) == (capability // 10)`,
// so sm_120 and sm_121 both map to the 12.x family. False when there is no
// queryable capability (CPU / get_device_capability() -> None).
bool Platform::is_device_capability_family(int capability) const {
  const DeviceCapability cap = get_device_capability();
  if (!cap.present()) return false;
  return (cap.to_int() / 10) == (capability / 10);
}

namespace {
std::array<Platform*, vt::kNumDeviceTypes>& Registry() {
  static std::array<Platform*, vt::kNumDeviceTypes> registry{};
  return registry;
}

size_t Index(DeviceType type) {
  const size_t index = static_cast<size_t>(type);
  VT_CHECK(index < vt::kNumDeviceTypes, "invalid device type");
  return index;
}

// Accelerator-first, CPU last — mirrors vLLM resolving `current_platform` by
// probing accelerators before falling back to CPU (platforms/__init__.py).
constexpr DeviceType kCurrentPriority[] = {
    DeviceType::kCUDA, DeviceType::kXPU, DeviceType::kVULKAN,
    DeviceType::kMETAL, DeviceType::kCPU};
}  // namespace

void RegisterPlatform(DeviceType type, Platform* platform) {
  VT_CHECK(platform != nullptr, "cannot register a null platform");
  Registry()[Index(type)] = platform;
}

Platform& GetPlatform(DeviceType type) {
  Platform* p = Registry()[Index(type)];
  VT_CHECK(p != nullptr, std::string("no platform registered for device type ") +
                             std::to_string(static_cast<int>(type)));
  return *p;
}

bool HasPlatform(DeviceType type) { return Registry()[Index(type)] != nullptr; }

Platform& CurrentPlatform() {
  for (DeviceType type : kCurrentPriority) {
    Platform* p = Registry()[static_cast<size_t>(type)];
    if (p != nullptr) return *p;
  }
  VT_CHECK(false, "no platform registered (not even CPU)");
  return GetPlatform(DeviceType::kCPU);  // unreachable; VT_CHECK throws
}

}  // namespace vllm::platforms
