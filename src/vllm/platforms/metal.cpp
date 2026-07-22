// Metal leg of the Platform seam (BACKEND-METAL-MLX, W0 skeleton). Self-registers
// kMETAL via a static Registrar, copying the `src/vllm/platforms/cpu.cpp` /
// `cuda.cpp` registrar idiom. Compiled only in Metal builds (CMake
// target_sources gate).
//
// NO UPSTREAM MIRROR. vLLM has no `vllm/platforms/metal.py`; this is a recorded
// extension of the `vllm/platforms/interface.py:134-229 class Platform` seam
// (.agents/porting-inventory.md §9). Where a value has an upstream analogue the
// analogue is cited; where it does not, that is said outright.
//
// Deliberately plain C++, not ObjC++ — everything Apple-specific is reached
// through the vt::Backend virtuals (DeviceCapabilityMajor/Minor, UnifiedMemory),
// so the engine-side platform tree stays free of Metal headers.
#include "vllm/platforms/interface.h"

#include <vector>

#include "vt/backend.h"
#include "vt/metal/metal_context.h"

namespace vllm::platforms {
namespace {

class MetalPlatform final : public Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kMETAL; }
  Backend& backend() const override { return vt::GetBackend(DeviceType::kMETAL); }

  // interface.py:409-415 get_device_capability. CUDA answers with (sm_major,
  // sm_minor); the Apple-silicon analogue is the MTLGPUFamilyApple GENERATION,
  // which is what src/vt/metal/metal_context.mm probes and the backend exposes.
  // {9, 0} on the M4 gate box. This makes has_device_capability(N, 0) mean
  // "Apple family >= N", the same shape of question CUDA code already asks.
  DeviceCapability get_device_capability() const override {
    Backend& b = backend();
    return DeviceCapability{b.DeviceCapabilityMajor(), b.DeviceCapabilityMinor()};
  }

  // interface.py:181-187 supported_dtypes order (bf16 default fallback). Metal 3
  // on Apple family 9 handles all three; the kernels in src/vt/metal/metal_msl.h
  // implement f32/f16/bf16 storage explicitly.
  std::vector<DType> supported_dtypes() const override {
    return {DType::kBF16, DType::kF16, DType::kF32};
  }

  // Apple silicon is UNIFIED memory: there is one copy of the bytes, so freeing a
  // "host mirror" after "upload" would free the only copy, and there is no
  // discrete device pool to serve scratch from. Same answer, same reason, as the
  // CPU platform (src/vllm/platforms/cpu.cpp) — NOT the CUDA/GB10 answer, which
  // pools and host-frees for reasons specific to the CUDA allocator.
  ResidencyPolicy residency_policy() const override { return {}; }

  // Attention-backend priority. There is NO Metal attention kernel in this
  // skeleton — kPagedAttention is not registered for kMETAL — so returning a
  // name would be a claim we cannot honour. The EMPTY list is the honest and
  // mechanically correct answer: SelectAttentionBackendName walks the list and
  // takes the first REGISTERED name (include/vllm/v1/attention/registry.h:59-78),
  // so an empty list makes selection throw loudly instead of silently handing
  // back a backend whose kernels do not exist on this device. Work row M3 adds
  // the kernel and the one-line name here in the same change.
  std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg) const override {
    (void)cfg;
    return {};
  }
};

// Registers kMETAL during static init (registration completes before main() per
// the interface.h contract). Stays silent on a Metal-enabled build running where
// no Metal device exists — the exact shape of cuda.cpp's registrar, which
// likewise probes the device rather than trusting another TU's initializer.
// Registering a platform whose backend() would throw is worse than not
// registering: CurrentPlatform() (src/vllm/platforms/platform.cpp:38-40) walks
// {kCUDA, kXPU, kVULKAN, kMETAL, kCPU} and must be able to fall through to CPU.
struct Registrar {
  Registrar() noexcept {
    if (!vt::metal::MetalDeviceAvailable()) return;
    static MetalPlatform platform;
    RegisterPlatform(DeviceType::kMETAL, &platform);
  }
} registrar;

}  // namespace
}  // namespace vllm::platforms
