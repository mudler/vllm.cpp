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
  // e24d1b24 pins CPU to a single backend (AttentionBackendEnum.CPU_ATTN), and
  // since issue #1371 so do we: CPU_ATTN is REGISTERED
  // (src/vllm/v1/attention/backends/cpu_attn.cpp) and the walk stops on it.
  //
  // It used to be a name alone. Our CPU paged-attention kernel does not use
  // upstream CPU_ATTN's HND [N,H,block,2*D] layout; it reads the FlashAttention
  // NHD layout (num_blocks,2,block,H,D) — the recorded deviation in
  // src/vt/cpu/cpu_paged_attn.cpp:5-8 — so CPU_ATTN was listed first for
  // upstream fidelity while being unregistered, and the walk fell through to
  // FLASH_ATTN. That was behavior-preserving only for as long as FLASH_ATTN
  // accepted everything CPU asked of it. #1332 M1 gave FLASH_ATTN flash_attn.py's
  // `head_size % 8 == 0` rule, and a CPU request with head_size 6 then matched no
  // registered backend at all. FLASH_ATTN stays second, and stays registered for
  // kCPU, because that rule is right about the FA2 kernel and wrong about ours.
  // MLA is not reachable on the CPU platform (no CPU MLA backend upstream at the
  // pin), so the list is the same for both cfg values — an MLA request simply
  // finds no is_mla() backend registered and the selector throws, rather than
  // silently returning a dense backend.
  std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg) const override {
    (void)cfg;
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
