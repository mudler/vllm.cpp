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

  // Attention-backend priority. Upstream cpu.py::get_attn_backend_cls:75-87 @ pin
  // e24d1b24 pins CPU to a single backend (AttentionBackendEnum.CPU_ATTN). Our
  // CPU paged-attention kernel does NOT implement CPU_ATTN's [N,H,block,head]
  // layout; it reuses the FlashAttention NHD layout (num_blocks,2,block,H,D) —
  // the recorded deviation in src/vt/cpu/cpu_paged_attn.cpp:6. So CPU_ATTN is
  // listed first for upstream fidelity but is unregistered; the walk falls through
  // to FLASH_ATTN, the backend our CPU KV layout actually matches. This both
  // mirrors upstream's CPU preference AND is behavior-preserving (selection
  // returns "FLASH_ATTN", the layout used today), and demonstrates the
  // first-registered-in-priority fallthrough that IS the selection mechanism.
  std::vector<std::string> get_attn_backend_priority() const override {
    return {"CPU_ATTN", "FLASH_ATTN"};
  }
};

struct Registrar {
  Registrar() {
    static CpuPlatform platform;
    RegisterPlatform(DeviceType::kCPU, &platform);
  }
} registrar;

}  // namespace
}  // namespace vllm::platforms
