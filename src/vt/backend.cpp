// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/backend.h"

#include <array>

namespace vt {

void Backend::BeginCapture(Queue&) { VT_CHECK(false, "graph capture unsupported on this backend"); }
void Backend::EndCapture(Queue&) { VT_CHECK(false, "graph capture unsupported on this backend"); }
void Backend::Replay(Queue&) { VT_CHECK(false, "graph capture unsupported on this backend"); }

namespace {
std::array<Backend*, kNumDeviceTypes>& Registry() {
  static std::array<Backend*, kNumDeviceTypes> registry{};
  return registry;
}
}  // namespace

Backend& GetBackend(DeviceType type) {
  VT_CHECK(static_cast<size_t>(type) < kNumDeviceTypes, "invalid device type");
  Backend* b = Registry()[static_cast<size_t>(type)];
  VT_CHECK(b != nullptr, std::string("no backend registered for device type ") +
                             std::to_string(static_cast<int>(type)));
  return *b;
}

void RegisterBackend(DeviceType type, Backend* backend) {
  Registry()[static_cast<size_t>(type)] = backend;
}

}  // namespace vt
