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

#include <string_view>
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

  // The architectures this backend has actually registered the ops for (work
  // row M3a). Metal is a PARTIAL backend — 15 of 75 ops — so the honest answer
  // is an explicit allow-list, not `true`.
  //
  // OPT-125m (`OPTForCausalLM`) is the whole list today, and that is a MEASURED
  // choice rather than an arbitrary one: the reuse study found all four OPT TUs
  // contain zero CUDA references and that OPT needs the fewest new kernels of
  // any model in the tree. Anything else pointed at Metal falls back to the CPU
  // reference in model_loader.cpp::SelectQueue and runs correctly, just slowly —
  // which is strictly better than dying inside a kernel bind.
  //
  // Qwen3-dense (`Qwen3ForCausalLM`) is the NEXT entry and is deliberately NOT
  // here yet: it additionally needs kRopeCosSinCache + kRopeFromCache, which are
  // unregistered (work row M3b). Adding an architecture to this list without its
  // kernels would recreate exactly the silent-failure mode the seam removes.
  bool supports_model_architecture(std::string_view architecture) const override {
    return architecture == "OPTForCausalLM";
  }

  // Attention-backend priority (W0b-1 item 4, closed by work row M3a). The W0
  // skeleton returned {} because kPagedAttention was not registered for kMETAL
  // and naming a backend would have been a claim we could not honour. It IS
  // registered now (src/vt/metal/metal_ops.mm), against the same NHD
  // (num_blocks, 2, block_size, num_kv_heads, head_size) cache layout
  // FlashAttentionBackend::get_kv_cache_shape allocates and our CPU reference
  // reads — so FLASH_ATTN is the correct and only name, exactly as on CPU
  // (src/vllm/platforms/cpu.cpp) and CUDA. The backend itself is device-agnostic
  // host metadata, so it self-registers for kMETAL in
  // src/vllm/v1/attention/backend.cpp alongside kCUDA/kCPU.
  //
  // MLA is NOT offered: no Metal MLA kernel exists, so a use_mla request must
  // keep finding nothing and throwing rather than selecting a backend whose
  // kernels are unregistered.
  std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg) const override {
    if (cfg.use_mla) return {};
    return {"FLASH_ATTN"};
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
