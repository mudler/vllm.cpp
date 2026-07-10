// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include "vt/backend.h"

#include <array>
#include <atomic>

namespace vt {

uint64_t NextQueueId() noexcept {
  static std::atomic<uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

void Backend::BeginCapture(Queue&) { VT_CHECK(false, "graph capture unsupported on this backend"); }
void Backend::EndCapture(Queue&) { VT_CHECK(false, "graph capture unsupported on this backend"); }
void Backend::Replay(Queue&) { VT_CHECK(false, "graph capture unsupported on this backend"); }
void* Backend::EndCaptureGraph(Queue&) { VT_CHECK(false, "graph capture unsupported on this backend"); return nullptr; }
void Backend::ReplayGraph(Queue&, void*) { VT_CHECK(false, "graph capture unsupported on this backend"); }
void Backend::DestroyGraph(void*) {}

namespace {
struct RegistryEntry {
  Backend* backend = nullptr;
  const DeviceResourceOps* resources = nullptr;
};

std::array<RegistryEntry, kNumDeviceTypes>& Registry() {
  static std::array<RegistryEntry, kNumDeviceTypes> registry{};
  return registry;
}

size_t DeviceIndex(DeviceType type) {
  const size_t index = static_cast<size_t>(type);
  VT_CHECK(index < kNumDeviceTypes, "invalid device type");
  return index;
}
}  // namespace

Backend& GetBackend(DeviceType type) {
  Backend* b = Registry()[DeviceIndex(type)].backend;
  VT_CHECK(b != nullptr, std::string("no backend registered for device type ") +
                             std::to_string(static_cast<int>(type)));
  return *b;
}

void RegisterBackend(DeviceType type, Backend* backend) {
  VT_CHECK(backend != nullptr, "cannot register a null backend");
  Registry()[DeviceIndex(type)].backend = backend;
}

void RegisterDeviceResourceOps(DeviceType type, const DeviceResourceOps* ops) {
  VT_CHECK(ops != nullptr && ops->alloc != nullptr && ops->free != nullptr &&
               ops->create_queue != nullptr && ops->destroy_queue != nullptr,
           "device resource table is incomplete");
  Registry()[DeviceIndex(type)].resources = ops;
}

void* Alloc(Device device, size_t bytes) {
  VT_CHECK(bytes > 0, "device allocation size must be positive");
  RegistryEntry& entry = Registry()[DeviceIndex(device.type)];
  if (entry.resources != nullptr) return entry.resources->alloc(device, bytes);
  VT_CHECK(device.index == 0,
           "legacy backend allocation shim only supports device index 0");
  return GetBackend(device.type).Alloc(bytes);
}

void Free(Device device, void* p) {
  RegistryEntry& entry = Registry()[DeviceIndex(device.type)];
  if (entry.resources != nullptr) {
    entry.resources->free(device, p);
    return;
  }
  VT_CHECK(device.index == 0, "legacy backend free shim only supports device index 0");
  GetBackend(device.type).Free(p);
}

Queue CreateQueue(Device device) {
  RegistryEntry& entry = Registry()[DeviceIndex(device.type)];
  if (entry.resources != nullptr) return entry.resources->create_queue(device);
  VT_CHECK(device.index == 0,
           "legacy backend queue-creation shim only supports device index 0");
  Queue q = GetBackend(device.type).CreateQueue();
  VT_CHECK(q.device == device, "legacy backend returned a queue for the wrong device");
  return q;
}

void DestroyQueue(Queue& q) {
  if (q.id == 0) return;
  RegistryEntry& entry = Registry()[DeviceIndex(q.device.type)];
  if (entry.resources != nullptr) {
    entry.resources->destroy_queue(q);
  } else {
    VT_CHECK(q.device.index == 0,
             "legacy backend queue-destroy shim only supports device index 0");
    Backend& backend = GetBackend(q.device.type);
    backend.Synchronize(q);
    backend.DestroyQueue(q);
  }
  q.handle = nullptr;
  q.id = 0;
}

}  // namespace vt
