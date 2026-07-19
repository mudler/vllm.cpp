// Ported from: vllm/v1/attention/backends/registry.py + vllm/platforms/cuda.py
// :361-470 (get_valid_backends / get_attn_backend_cls) @ pin e24d1b24 — the
// attention-backend registry storage + platform-priority selection. See
// registry.h for the design; this file is the registry table and the selector.
#include "vllm/v1/attention/registry.h"

#include <array>
#include <map>
#include <stdexcept>

#include "vt/dtype.h"  // VT_CHECK

namespace vllm::v1 {

namespace {

// [DeviceType] -> {name -> factory}. Function-local static (Meyers) so the table
// is constructed before any static AttentionBackendRegistrar runs, mirroring the
// vt op-table Table() / platform Registry() idiom.
using DeviceTable = std::map<std::string, AttentionBackendFactory>;
std::array<DeviceTable, vt::kNumDeviceTypes>& Registry() {
  static std::array<DeviceTable, vt::kNumDeviceTypes> registry{};
  return registry;
}

size_t Index(vt::DeviceType device) {
  const size_t index = static_cast<size_t>(device);
  VT_CHECK(index < vt::kNumDeviceTypes, "invalid device type");
  return index;
}

}  // namespace

void RegisterAttentionBackend(vt::DeviceType device, const std::string& name,
                              AttentionBackendFactory factory) {
  VT_CHECK(factory != nullptr, "cannot register a null attention-backend factory");
  Registry()[Index(device)][name] = factory;
}

bool HasAttentionBackend(vt::DeviceType device, const std::string& name) {
  const DeviceTable& table = Registry()[Index(device)];
  return table.find(name) != table.end();
}

std::unique_ptr<AttentionBackend> MakeAttentionBackend(vt::DeviceType device,
                                                       const std::string& name) {
  const DeviceTable& table = Registry()[Index(device)];
  const auto it = table.find(name);
  VT_CHECK(it != table.end(),
           std::string("no attention backend '") + name +
               "' registered for device type " +
               std::to_string(static_cast<int>(device)));
  return it->second();
}

std::string SelectAttentionBackendName(const platforms::Platform& platform,
                                       const std::string& selected) {
  const vt::DeviceType device = platform.device_type();

  // Explicit override (upstream get_attn_backend_cls's selected_backend arg /
  // VLLM_ATTENTION_BACKEND). It must be registered for this device.
  if (!selected.empty()) {
    if (HasAttentionBackend(device, selected)) return selected;
    throw std::invalid_argument(
        std::string("selected attention backend '") + selected +
        "' is not registered for device type " +
        std::to_string(static_cast<int>(device)));
  }

  // Walk the platform's capability-ordered priority list; the first name that is
  // registered wins (upstream's min-priority valid backend). An unregistered name
  // is skipped, exactly as an ImportError-ing backend is in get_valid_backends.
  for (const std::string& name : platform.get_attn_backend_priority()) {
    if (HasAttentionBackend(device, name)) return name;
  }

  throw std::runtime_error(
      std::string("no valid attention backend registered for device type ") +
      std::to_string(static_cast<int>(device)) +
      " (priority list yielded no registered backend)");
}

std::unique_ptr<AttentionBackend> SelectAttentionBackend(
    const platforms::Platform& platform, const std::string& selected) {
  return MakeAttentionBackend(platform.device_type(),
                              SelectAttentionBackendName(platform, selected));
}

}  // namespace vllm::v1
