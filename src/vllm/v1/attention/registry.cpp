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

namespace {

// The capability half of vllm/v1/attention/backend.py:307-360
// validate_configuration: a candidate must agree with the request on is_mla()
// and is_sparse(). Backends are stateless descriptors, so constructing one to
// ask is cheap and mirrors upstream querying the CLASS. This is the ONLY place
// sparse/DSA needs to be understood — see registry.h "the DSA seam".
bool CandidateMatchesConfig(vt::DeviceType device, const std::string& name,
                            const platforms::AttnSelectorConfig& cfg) {
  const std::unique_ptr<AttentionBackend> backend = MakeAttentionBackend(device, name);
  return backend->is_mla() == cfg.use_mla && backend->is_sparse() == cfg.use_sparse;
}

}  // namespace

std::string SelectAttentionBackendName(const platforms::Platform& platform,
                                       const std::string& selected,
                                       const platforms::AttnSelectorConfig& cfg) {
  const vt::DeviceType device = platform.device_type();

  // Explicit override (upstream get_attn_backend_cls's selected_backend arg /
  // VLLM_ATTENTION_BACKEND). It must be registered for this device AND still
  // satisfy validate_configuration — upstream raises on an override whose
  // capabilities do not match the request, it does not silently honor it.
  if (!selected.empty()) {
    if (!HasAttentionBackend(device, selected)) {
      throw std::invalid_argument(
          std::string("selected attention backend '") + selected +
          "' is not registered for device type " +
          std::to_string(static_cast<int>(device)));
    }
    if (!CandidateMatchesConfig(device, selected, cfg)) {
      throw std::invalid_argument(
          std::string("selected attention backend '") + selected +
          "' does not satisfy the request (use_mla=" +
          (cfg.use_mla ? "true" : "false") +
          ", use_sparse=" + (cfg.use_sparse ? "true" : "false") + ")");
    }
    return selected;
  }

  // Walk the platform's capability-ordered priority list; the first name that is
  // registered AND valid for this request wins (upstream's min-priority valid
  // backend). An unregistered name is skipped exactly as an ImportError-ing
  // backend is in get_valid_backends; a registered-but-mismatched one is skipped
  // exactly as validate_configuration's "invalid reasons" reject it.
  for (const std::string& name : platform.get_attn_backend_priority(cfg)) {
    if (!HasAttentionBackend(device, name)) continue;
    if (!CandidateMatchesConfig(device, name, cfg)) continue;
    return name;
  }

  throw std::runtime_error(
      std::string("no valid attention backend registered for device type ") +
      std::to_string(static_cast<int>(device)) +
      " (priority list yielded no registered backend; use_mla=" +
      (cfg.use_mla ? "true" : "false") +
      ", use_sparse=" + (cfg.use_sparse ? "true" : "false") + ")");
}

std::unique_ptr<AttentionBackend> SelectAttentionBackend(
    const platforms::Platform& platform, const std::string& selected,
    const platforms::AttnSelectorConfig& cfg) {
  return MakeAttentionBackend(platform.device_type(),
                              SelectAttentionBackendName(platform, selected, cfg));
}

}  // namespace vllm::v1
