// Tenstorrent leg of the Platform seam (BACKEND-TENSTORRENT).
// Self-registers kTENSTORRENT via a static Registrar, copying the
// `src/vllm/platforms/cpu.cpp` / `cuda.cpp` / `metal.cpp` registrar idiom.
// Compiled only in Tenstorrent builds (CMake target_sources gate).
//
// NO UPSTREAM MIRROR. vLLM has no `vllm/platforms/tenstorrent.py` and no
// Tenstorrent path anywhere in its tree; this is a recorded extension of the
// `vllm/platforms/interface.py:134-229 class Platform` seam
// (.agents/porting-inventory.md §9, item 15).
//
// Deliberately plain C++, not a ttnn TU — everything Tenstorrent-specific is
// reached through the vt::Backend virtuals, so the engine-side platform tree
// stays free of ttnn headers.
#include "vllm/platforms/interface.h"

#include <string_view>
#include <vector>

#include "vt/backend.h"
#include "vt/tenstorrent/tenstorrent_device.h"

namespace vllm::platforms {
namespace {

class TenstorrentPlatform final : public Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kTENSTORRENT; }
  Backend& backend() const override { return vt::GetBackend(DeviceType::kTENSTORRENT); }

  // interface.py:409-415 get_device_capability. Tenstorrent's Tensix cores
  // have no CUDA-SM-shaped "compute capability" to report; the base {0, 0}
  // ("no meaningful compute capability", backend.h) is the honest answer,
  // same as CPU.
  DeviceCapability get_device_capability() const override { return DeviceCapability{}; }

  // OPT-125m runs BF16 weights/activations with F32 logits. The adapter
  // host-converts both float dtypes into ttnn BFLOAT16 tiles (and back).
  std::vector<DType> supported_dtypes() const override {
    return {DType::kBF16, DType::kF32};
  }

  // Discrete PCIe device (tenstorrent_backend.cpp UnifiedMemory()==false); no
  // host-weight-release/pool-cap policy has been worked out for it yet, so the
  // default (empty) ResidencyPolicy is the honest answer — same non-decision
  // Vulkan's early skeleton made for the same reason.
  ResidencyPolicy residency_policy() const override { return {}; }

  // Explicit allow-list of architectures whose full op set is registered for
  // kTENSTORRENT (mirrors MetalPlatform::supports_model_architecture). OPT-125m
  // was the first bring-up; Qwen3-dense is the second (same OPT→Qwen3 sequence
  // Metal used for M3a/M3b). Mistral-7B-v0.3 is the third: it reuses the
  // Qwen3-dense forward verbatim (qk-norm skipped, plain rope, untied lm_head),
  // so every op is already registered — no new kernel. Anything else falls back
  // to CPU via SelectQueue.
  bool supports_model_architecture(std::string_view architecture) const override {
    return architecture == "OPTForCausalLM" ||
           architecture == "Qwen3ForCausalLM" ||
           architecture == "MistralForCausalLM";
  }

  // kPagedAttention + kReshapeAndCache are registered against the NHD
  // FlashAttentionBackend layout, so FLASH_ATTN is the correct name — same as
  // CPU/CUDA/Metal/Vulkan. MLA is not offered.
  std::vector<std::string> get_attn_backend_priority(const AttnSelectorConfig& cfg) const override {
    if (cfg.use_mla) return {};
    return {"FLASH_ATTN"};
  }
};

// Registers kTENSTORRENT during static init. Stays silent when no Blackhole
// card is present (same shape as metal.cpp / vulkan.cpp).
struct Registrar {
  Registrar() noexcept {
    if (!vt::tenstorrent::DeviceAvailable()) return;
    static TenstorrentPlatform platform;
    RegisterPlatform(DeviceType::kTENSTORRENT, &platform);
  }
} registrar;

}  // namespace
}  // namespace vllm::platforms
