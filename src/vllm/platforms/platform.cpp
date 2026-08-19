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
// kROCM sits directly after kCUDA because that is upstream's own probe ORDER:
// builtin_platform_plugins is {tpu, cuda, rocm, xpu, cpu}
// (platforms/__init__.py:202-208).
//
// THIS ARRAY IS THE ONE PLACE A NEW PLATFORM IS NOT ADDITIVE. The compiler
// cannot catch an omission here the way -Werror=switch catches a missing enum
// case: a platform left out of this walk registers fine, answers every query
// correctly, and is simply never SELECTED. tests/vllm/platforms/test_platform.cpp
// gates the membership so the next backend does not rediscover this.
// kTENSTORRENT sits after kMETAL: an extension platform with no upstream probe
// order to mirror (same as kVULKAN/kMETAL), placed last among accelerators.
// W2: OPT-125m e2e STRICT token-exact on real Blackhole; still least proven
// among model-running backends.
constexpr DeviceType kCurrentPriority[] = {
    DeviceType::kCUDA,   DeviceType::kROCM,        DeviceType::kXPU,
    DeviceType::kVULKAN, DeviceType::kMETAL, DeviceType::kTENSTORRENT,
    DeviceType::kCPU};
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

Platform* FindPlatformByName(std::string_view name) {
  for (size_t i = 0; i < vt::kNumDeviceTypes; ++i) {
    Platform* platform = Registry()[i];
    if (platform == nullptr) continue;
    const DeviceType type = static_cast<DeviceType>(i);
    if (name == vt::DeviceTypeName(type)) return platform;
  }
  return nullptr;
}

Platform& CurrentPlatform() {
  for (DeviceType type : kCurrentPriority) {
    Platform* p = Registry()[static_cast<size_t>(type)];
    if (p != nullptr) return *p;
  }
  VT_CHECK(false, "no platform registered (not even CPU)");
  return GetPlatform(DeviceType::kCPU);  // unreachable; VT_CHECK throws
}

const DeviceType* CurrentPlatformPriority(size_t& count) {
  count = sizeof(kCurrentPriority) / sizeof(kCurrentPriority[0]);
  return kCurrentPriority;
}

bool HostMemoryIsDeviceAddressableFromAttrs(int pageable_memory_access,
                                            int integrated) {
  // BOTH, and the header says which device class each term excludes. It lives in
  // this always-compiled translation unit rather than in `cuda.cpp` so that the
  // CPU tier can run it: `cuda.cpp` compiles only in a CUDA build, which is the
  // reason the conjunction went ungated in the first place.
  return pageable_memory_access != 0 && integrated != 0;
}

}  // namespace vllm::platforms
