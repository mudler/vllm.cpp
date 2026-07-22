// Vulkan leg of the Platform seam (BACKEND-VULKAN, W0 skeleton). Self-registers
// kVULKAN via a static Registrar, copying the `src/vllm/platforms/cpu.cpp` /
// `cuda.cpp` / `metal.cpp` registrar idiom. Compiled only in Vulkan builds
// (CMake target_sources gate).
//
// NO UPSTREAM MIRROR. vLLM has no `vllm/platforms/vulkan.py` and no Vulkan path
// anywhere in its tree or its dependency chain; this is a recorded extension of
// the `vllm/platforms/interface.py:134-229 class Platform` seam
// (.agents/porting-inventory.md §9). Where a value has an upstream analogue the
// analogue is cited; where it does not, that is said outright.
//
// Deliberately plain C++, not a Vulkan TU — everything Vulkan-specific is
// reached through the vt::Backend virtuals (DeviceCapabilityMajor/Minor,
// UnifiedMemory), so the engine-side platform tree stays free of Vulkan headers.
#include "vllm/platforms/interface.h"

#include <vector>

#include "vt/backend.h"
#include "vt/vulkan/vulkan_context.h"

namespace vllm::platforms {
namespace {

class VulkanPlatform final : public Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kVULKAN; }
  Backend& backend() const override { return vt::GetBackend(DeviceType::kVULKAN); }

  // interface.py:409-415 get_device_capability. CUDA answers with (sm_major,
  // sm_minor) and the Metal skeleton with the Apple GPU family; the Vulkan
  // analogue is the API VERSION the physical device reports — {1, 4} on GB10
  // (Vulkan 1.4.312). That makes has_device_capability(1, 1) mean "Vulkan >= 1.1",
  // the same shape of question the CUDA code already asks, and it is the version
  // the feature gates that matter here (16-bit storage, cooperative matrix,
  // subgroup ops) are actually keyed to.
  DeviceCapability get_device_capability() const override {
    Backend& b = backend();
    return DeviceCapability{b.DeviceCapabilityMajor(), b.DeviceCapabilityMinor()};
  }

  // interface.py:181-187 supported_dtypes order (bf16 default fallback). All
  // three are implemented as STORAGE dtypes by the shaders in
  // src/vt/vulkan/shaders/, whose f16 and bf16 codecs are transcriptions of
  // src/vt/dtype.cpp rather than driver intrinsics — so support does not depend
  // on any optional device float16 feature.
  std::vector<DType> supported_dtypes() const override {
    return {DType::kBF16, DType::kF16, DType::kF32};
  }

  // GB10 is UNIFIED memory (one 89.72 GiB DEVICE_LOCAL|HOST_VISIBLE heap), and
  // this skeleton allocates every buffer host-visible and persistently mapped on
  // every device, so there is exactly one copy of the bytes: freeing a "host
  // mirror" after "upload" would free the only copy, and there is no separate
  // device pool to serve scratch from. Same answer, same reason, as the CPU and
  // Metal platforms — NOT the CUDA answer, which pools and host-frees for reasons
  // specific to the CUDA allocator. A discrete-GPU staging path (which WOULD
  // want a pool) is not implemented in W0.
  ResidencyPolicy residency_policy() const override { return {}; }

  // Attention-backend priority. There is NO Vulkan attention kernel in this
  // skeleton — kPagedAttention is not registered for kVULKAN — so returning a
  // name would be a claim we cannot honour. The EMPTY list is the honest and
  // mechanically correct answer: SelectAttentionBackendName walks the list and
  // takes the first REGISTERED name (include/vllm/v1/attention/registry.h:59-78),
  // so an empty list makes selection throw loudly instead of silently handing
  // back a backend whose kernels do not exist on this device. Work row V4 (the
  // port of llama.cpp's flash_attn.comp family) adds the kernel and the one-line
  // name here in the same change.
  std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg) const override {
    (void)cfg;
    return {};
  }
};

// Registers kVULKAN during static init (registration completes before main() per
// the interface.h contract). Stays silent on a Vulkan-enabled build running
// where there is no loader or no conformant device — the exact shape of
// cuda.cpp's and metal.cpp's registrars, which likewise probe the DEVICE rather
// than trusting another TU's initializer (static-init order across TUs is
// unspecified, so asking "did the backend registrar run?" would intermittently
// skip platform registration). Registering a platform whose backend() would
// throw is worse than not registering: CurrentPlatform()
// (src/vllm/platforms/platform.cpp:38-40) walks {kCUDA, kXPU, kVULKAN, kMETAL,
// kCPU} and must be able to fall through to CPU.
struct Registrar {
  Registrar() noexcept {
    if (!vt::vulkan::VulkanDeviceAvailable()) return;
    static VulkanPlatform platform;
    RegisterPlatform(DeviceType::kVULKAN, &platform);
  }
} registrar;

}  // namespace
}  // namespace vllm::platforms
