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

  // interface.py:420-431 get_device_capability, whose docstring defines the UNIT:
  // "Stateless version of torch.cuda.get_device_capability". It is an NVIDIA SM
  // version. A Vulkan device has no SM version, so the honest answer is ABSENT —
  // upstream's own answer for the same situation, xpu.py:228-234: "capacity
  // format differs from cuda's and will cause unexpected failure, so use None
  // directly".
  //
  // #1823, the Vulkan half. This used to report the Vulkan API VERSION ({1, 4} on
  // GB10, and 1.x on every Vulkan device that will ever exist), which
  // FlashAttentionBackend::supports_compute_capability (flash_attn.py:200-202)
  // then compared against `>= (8, 0)`. FLASH_ATTN is the ONLY entry in
  // get_attn_backend_priority(), so SelectAttentionBackendName threw on kVULKAN
  // unconditionally. Metal at least had a coincidence; this had none, and it was
  // invisible because the lane's test asserted that FLASH_ATTN is NAMED in the
  // priority list rather than that the selector REACHES it.
  //
  // The API version is not lost and was never this seam's to report: it stays on
  // vt::Backend::DeviceCapabilityMajor/Minor
  // (src/vt/vulkan/vulkan_backend.cpp:144-145) and VulkanContext::api_major/minor,
  // which is where the 16-bit-storage / cooperative-matrix / subgroup feature
  // gates actually read it.
  DeviceCapability get_device_capability() const override { return DeviceCapability{}; }

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
  // interface.py get_attn_backend_cls. EMPTY until VK-B: the selector must not be
  // able to hand out a backend whose kernels do not exist, and until Vulkan had
  // native kPagedAttention + kReshapeAndCache the honest answer was "none".
  //
  // Both now exist and read/write the same NHD layout FlashAttentionBackend's
  // get_kv_cache_shape allocates, so FLASH_ATTN is reachable on exactly the
  // footing Metal reached it (metal.cpp:88-92).
  //
  // MLA still returns EMPTY, and that is not a stub: MLA needs
  // kMlaDecodeAttention / kMlaPrefillAttention / kConcatAndCacheMla, none of
  // which has a Vulkan kernel. Answering FLASH_ATTN there would route an MLA
  // model into a backend that cannot serve it.
  std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg) const override {
    if (cfg.use_mla) return {};
    return {"FLASH_ATTN"};
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
