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

  // interface.py:420-431 get_device_capability, whose unit is an NVIDIA SM
  // version. Tenstorrent's Tensix cores have no SM version to report, so ABSENT
  // is the honest answer — same as CPU, and the same answer Metal and Vulkan
  // give since #1823.
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
  // so every op was already registered — no new kernel. Qwen3.5 (GDN hybrid)
  // is the fourth: its op delta (kGdnPostConv, kSigmoidGateBf16,
  // kAttnQkNormRopeGate, the GDN decode set) landed in the GDN/Qwen35 rows,
  // with the bf16 mamba-cache arms enabled by the backend's compressed-state
  // capabilities. Anything else falls back to CPU via SelectQueue.
  bool supports_model_architecture(std::string_view architecture) const override {
    return architecture == "OPTForCausalLM" ||
           architecture == "Qwen3ForCausalLM" ||
           architecture == "MistralForCausalLM" ||
           architecture == "Qwen3_5ForConditionalGeneration";
  }

  // kPagedAttention + kReshapeAndCache are registered against the NHD
  // FlashAttentionBackend layout, so FLASH_ATTN is the correct name — same as
  // CPU/CUDA/Metal/Vulkan. MLA is not offered.
  std::vector<std::string> get_attn_backend_priority(const AttnSelectorConfig& cfg) const override {
    if (cfg.use_mla) return {};
    return {"FLASH_ATTN"};
  }

  // HOST-FREE-FORWARD R5 flip (#1604): host-free decode is now the DEFAULT
  // (VT_TT_HOST_FREE_DECODE unset or any value except "0"). Set
  // VT_TT_HOST_FREE_DECODE=0 to opt out and restore the pre-flip default path.
  //
  // CAPTURE FLIP (#1625): the captured arm is now the DEFAULT too — same
  // polarity, VT_TT_DECODE_CAPTURE=0 opts out and restores the eager
  // host-free arm. Capture stayed opt-in while this comment said: the
  // captured arm "hangs deterministically on the FIRST multi-request run
  // (16-prompt test_qwen3_paged_engine, plain and VT_TT_RECAPTURE_EVERY=8
  // alike; one tt-metal worker spins at 100%, main thread blocked), while
  // single-request captured legs and the whole host-free EAGER path never
  // hit it." The hang was the short-chunk device KV push clobber (#2669):
  // prefill chunks shorter than kPagedFillMinTokens ran the batched device
  // update per-token over one shared physical block, and the last writer
  // won, leaving the captured decode to attend a dead request's rows. With
  // that repaired, the captured multi-request battery is 16/16 green and
  // hang-free, and the captured arm is the fastest measured arm (28.61 tok/s
  // warm median against 12.21 eager host-free, same-binary A/B, #2566
  // recipe), so the default ships it.
  bool support_static_graph_mode() const override {
    return vt::tenstorrent::HostFreeDecodeEnabled() &&
           vt::tenstorrent::DecodeCaptureEnabled();
  }

  // #1625/#2812: the default-on above is EVIDENCE-SCOPED to the families with
  // committed TT captured-arm gate evidence (Qwen3-dense, Mistral-7B,
  // Qwen3.5-GDN). Every other decode driver conjuncts this query and keeps
  // the pre-flip explicit opt-in until its own captured arm is brought up
  // and gated.
  bool static_graph_requires_opt_in() const override {
    return !vt::tenstorrent::DecodeCaptureRequested();
  }

  // Architecture-scoped carve-out for the evidence families (Qwen3-dense,
  // Mistral-7B, Qwen3.5-GDN): they capture by default — each gates its
  // captured arm against its own committed teacher-forced pair (#2812);
  // Llama and InternLM2, which construct the same class, keep the explicit
  // opt-in until their captured arms are brought up.
  bool static_graph_requires_opt_in(
      const std::vector<std::string>& architectures) const override {
    if (vt::tenstorrent::DecodeCaptureRequested()) return false;
    return !vt::tenstorrent::DecodeCaptureDefaultArch(architectures);
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
