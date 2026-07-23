// Faithful port of vllm/platforms/cuda.py (CudaPlatform) @ pin e24d1b24 — the
// CUDA leg of the Platform seam. Self-registers kCUDA via a static Registrar
// that probes device presence + compute capability, copying the
// cuda_backend.cu registrar idiom (silent on a machine with the toolkit but no
// usable GPU). Compiled only in CUDA builds (CMake target_sources gate).
#include <cuda_runtime.h>

#include <vector>

#include "vllm/platforms/cuda_attn_priority.h"
#include "vllm/platforms/interface.h"

#include "vt/backend.h"

namespace vllm::platforms {
namespace {

class CudaPlatform final : public Platform {
 public:
  CudaPlatform(int cc_major, int cc_minor, bool integrated)
      : cap_{cc_major, cc_minor}, integrated_{integrated} {}

  DeviceType device_type() const override { return DeviceType::kCUDA; }
  Backend& backend() const override { return vt::GetBackend(DeviceType::kCUDA); }

  // cuda.py get_device_capability: torch.cuda.get_device_capability probed once
  // at registration (device 0 only for now, matching the backend registrar).
  DeviceCapability get_device_capability() const override { return cap_; }

  // --- Portable capability predicates (work row S3) --------------------------
  // Faithful ports of the CudaPlatform overrides in vllm/platforms/cuda.py @ pin
  // e24d1b24. Each returns exactly what a raw `device.type == kCUDA` returned at
  // the shared-layer gates S3 converts to it (true on this GB10/sm_121 CUDA leg,
  // and the base false on every other platform) — so the conversion is
  // byte-identical today, while a future accelerator answers for itself.

  // cuda.py:562 supports_fp8 -> has_device_capability(89). GB10 (sm_121) is >= 8.9
  // -> true; this is the fp8-fused-path gate the S4-deferred sites (§9.3) use.
  bool supports_fp8() const override { return has_device_capability(8, 9); }

  // nvfp4_utils.py:56 cutlass_fp4_supported -> is_cuda() && the csrc CC check
  // (nvfp4_scaled_mm_entry.cu:71: CC in [100,130) with the SM100/SM120 NVFP4
  // kernels compiled in). GB10 (cap 12.1 -> 121) qualifies -> true; the true-W4A4
  // fp4-activation gate the 27B razor takes on this device.
  bool cutlass_fp4_supported() const override {
    const int cc = cap_.to_int();
    return cc >= 100 && cc < 130;
  }

  // cuda.py:570 opaque_attention_op -> True.
  bool opaque_attention_op() const override { return true; }

  // cuda.py:675 is_integrated_gpu -> torch is_integrated; the C++ analogue is
  // cudaDevAttrIntegrated, probed once at registration. GB10 (Grace-Blackwell UMA)
  // reports integrated. Surface parity for the ROCm/memory-reporting port.
  bool is_integrated_gpu() const override { return integrated_; }

  // cuda.py:662 support_static_graph_mode -> True (CUDA graph capture mode).
  bool support_static_graph_mode() const override { return true; }

  // interface.py:181-187 supported_dtypes order (bf16 default fallback).
  std::vector<DType> supported_dtypes() const override {
    return {DType::kBF16, DType::kF16, DType::kF32};
  }

  // The residency/memory-model policy (the PR #4 debt as data) — CONSUMED by the
  // model residency path since item 2 (BACKEND-PLATFORM), no longer inline. These
  // values REPRODUCE today's GB10 behavior EXACTLY:
  //   * release_host_weights_after_upload = true: the routed MoE experts' ~16.9 GiB
  //     host fp4 mirror is freed after the per-layer device Marlin build
  //     (ENG-MOE-HOSTFREE ac77bec + ENG-MOE-LOADSTREAM ce7e1a0). qwen3_5.cpp's
  //     host-free + load-stream sites read this via ShouldReleaseHostWeights /
  //     ShouldInterleaveLoadStream; the wmma-fallback SAFETY gate stays
  //     MarlinMoeEnabled() (orthogonal kernel-path question). VT_MOE_HOST_FREE=0
  //     still overrides (house A/B convention).
  //   * uses_device_memory_pool = true + device_pool_cap_bytes = 0: the DevicePool
  //     scratch reuse, uncapped, exactly as today.
  // A discrete GPU sets different values (e.g. a pool cap) and NO model code is
  // touched — that is the item-2 additive win.
  ResidencyPolicy residency_policy() const override {
    ResidencyPolicy p;
    p.release_host_weights_after_upload = true;   // freed after Marlin build (today)
    p.uses_device_memory_pool = true;             // qwen3_5.cpp DevicePool
    p.device_pool_cap_bytes = 0;                  // uncapped
    return p;
  }

  // Capability-ordered attention-backend priority — a faithful port of
  // vllm/platforms/cuda.py::_get_backend_priorities:84-176 @ pin e24d1b24, BOTH
  // branches, expressed as the data table above (W2 completes the MLA branch the
  // pre-W2 comment here deferred).
  //   * cfg.use_mla == false (our Qwen3 dense + GDN gate models): unchanged —
  //       major 10 → FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT
  //       else (incl. GB10 sm_121 == major 12) → FLASH_ATTN first.
  //     Behavior-preserving: FLASH_ATTN is the only registered CUDA backend, so
  //     SelectAttentionBackendName still returns "FLASH_ATTN".
  //   * cfg.use_mla == true, major 12 (GB10) → [TRITON_MLA,
  //     FLASHINFER_MLA_SPARSE_SM120]; the sparse entry is filtered by the
  //     selector for a dense request, so TRITON_MLA is the answer — matching the
  //     W0 runtime OBSERVATION from the vLLM 0.25.0 oracle on sm_121.
  std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg) const override {
    return LookupAttnPriority(cap_.major, cfg);
  }

  // MLA prefill selector (mla/prefill/selector.py:47-76). GB10 → [FLASH_ATTN].
  std::vector<std::string> get_mla_prefill_backend_priority() const override {
    return LookupMlaPrefillPriority(cap_.major);
  }

 private:
  DeviceCapability cap_;
  bool integrated_ = false;
};

// Registers kCUDA during static init (registration must complete before main()
// per the interface.h contract). Stays silent on a machine that has the CUDA
// toolkit but no usable GPU: no throw, no print — it just leaves kCUDA
// unregistered and CurrentPlatform() falls back to CPU, consistent with the
// backend registrar (cuda_backend.cu:255-266).
struct Registrar {
  Registrar() noexcept {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return;
    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess) {
      return;
    }
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess) {
      return;
    }
    // is_integrated_gpu (cuda.py:675 torch is_integrated) — probe once here, the
    // same place the compute capability is probed. A query failure defaults to
    // false (non-integrated), the conservative answer.
    int integrated = 0;
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0) != cudaSuccess) {
      integrated = 0;
    }
    // GCC 13 false-positive: -Wdangling-pointer mis-flags a static local with a
    // vtable constructed from automatic ints, though CudaPlatform copies both
    // into cap_ by value (no pointer/reference to major/minor is retained). The
    // static outlives the registrar as RegisterPlatform requires.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-pointer"
    static CudaPlatform platform(major, minor, integrated != 0);  // device 0 only
#pragma GCC diagnostic pop
    RegisterPlatform(DeviceType::kCUDA, &platform);
  }
} registrar;

}  // namespace
}  // namespace vllm::platforms
