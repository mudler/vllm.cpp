// Faithful port of vllm/platforms/cpu.py (CpuPlatform) @ pin e24d1b24 — the CPU
// leg of the Platform seam. Self-registers kCPU via a static Registrar, copying
// the cpu_backend.cpp registrar idiom.
#include "vllm/platforms/interface.h"

#include "vt/backend.h"

namespace vllm::platforms {
namespace {

class CpuPlatform final : public Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kCPU; }
  Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }

  // cpu.py get_device_capability -> None: a CPU has no queryable compute
  // capability, so has_device_capability(...) is always false.
  DeviceCapability get_device_capability() const override { return {}; }

  // interface.py:181-187 supported_dtypes order (bf16 default fallback).
  std::vector<DType> supported_dtypes() const override {
    return {DType::kBF16, DType::kF16, DType::kF32};
  }

  // Unified host memory: no host-weight release, no device pool.
  ResidencyPolicy residency_policy() const override { return {}; }
};

struct Registrar {
  Registrar() {
    static CpuPlatform platform;
    RegisterPlatform(DeviceType::kCPU, &platform);
  }
} registrar;

}  // namespace
}  // namespace vllm::platforms
