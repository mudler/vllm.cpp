// Faithful port of vllm/platforms/cuda.py (CudaPlatform) @ pin e24d1b24 — the
// CUDA leg of the Platform seam. Self-registers kCUDA via a static Registrar
// that probes device presence + compute capability, copying the
// cuda_backend.cu registrar idiom (silent on a machine with the toolkit but no
// usable GPU). Compiled only in CUDA builds (CMake target_sources gate).
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "vllm/platforms/cuda_attn_priority.h"
#include "vllm/platforms/interface.h"

#include "vt/backend.h"

namespace vllm::platforms {
namespace {

class CudaPlatform final : public Platform {
 public:
  CudaPlatform(int cc_major, int cc_minor, bool integrated,
               size_t device_memory_total_bytes)
      : cap_{cc_major, cc_minor},
        integrated_{integrated},
        device_memory_total_bytes_{device_memory_total_bytes} {}

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

  // S7 residency POLICY -> True: the CUDA path stages host tensors into distinct
  // device-resident buffers (ResidentWeight, device-resident GDN state I/O, the
  // merged/packed device-resident projections, direct-device-load), regardless of
  // GB10 being physically unified. This is the memory-model residency the model's
  // device-resident forward branches on; base false answers the host-resident
  // direct-view reference path. Exactly what `device==kCUDA` returned.
  bool needs_weight_staging() const override { return true; }

  // S7 attention fast-path POLICY -> True: this device carries the vendored
  // flash-attention-2 native-bf16 split-KV kernel the FA2 dispatch selects
  // (cuda_paged_attn.cu). Base false answers the f32 graph-captured fallback.
  // Exactly what `device==kCUDA` returned at the FA2 dispatch gate.
  bool supports_fa2_attention() const override { return true; }

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
  //   * device_memory_total_bytes = cudaMemGetInfo's `total`, probed at
  //     registration (issue #1123). NEW data, consumed only by the load-time
  //     GGUF fit refusal; nothing that read this struct before sees a change.
  // A discrete GPU sets different values (e.g. a pool cap) and NO model code is
  // touched — that is the item-2 additive win.
  //
  // The four assignments themselves live in `CudaResidencyPolicy`
  // (`vllm/platforms/interface.h`), not here, because this translation unit
  // compiles only in a CUDA build: while they were inline, nothing on a host
  // without a CUDA toolkit could reach them, which is why #1123 had to record
  // "delete the device_memory_total_bytes assignment" as an unproven mutation.
  // test_platform.cpp now pins the assembly on every host (#1136). What stays
  // CUDA-only here is the `cudaMemGetInfo` probe below and the value it threads.
  ResidencyPolicy residency_policy() const override {
    return CudaResidencyPolicy(device_memory_total_bytes_);
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
  // cudaMemGetInfo's `total`, probed once at registration; 0 == UNKNOWN.
  size_t device_memory_total_bytes_ = 0;
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
    // ResidencyPolicy::device_memory_total_bytes (issue #1123) — probe once here,
    // beside the other device probes. `nvidia-smi` is the WRONG instrument for
    // this on a GB10: `--query-gpu=memory.total,memory.free,memory.used` answers
    // `[N/A], [N/A], [N/A]` because host and device share one pool, and the `rc`
    // fleet label records `vram=[N/A]M` for the same reason. `cudaMemGetInfo`
    // answers honestly. Measured on dgx:gpu0 through libcudart.so.13:
    // total = 128452956160 (119.631 GiB), free = 122059919360 (113.677 GiB),
    // and `total` is EXACTLY `/proc/meminfo MemTotal` (125442340 kB) times 1024.
    //
    // A query failure leaves 0 = UNKNOWN, which the consumer treats as "do not
    // decide" rather than as "nothing fits". `free_bytes` is read and discarded:
    // this is a load-time budget, and `free` makes it a function of contention.
    size_t total_bytes = 0;
    size_t free_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
      total_bytes = 0;
    }
    // GCC 13 false-positive: -Wdangling-pointer mis-flags a static local with a
    // vtable constructed from automatic ints, though CudaPlatform copies both
    // into cap_ by value (no pointer/reference to major/minor is retained). The
    // static outlives the registrar as RegisterPlatform requires.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-pointer"
    static CudaPlatform platform(major, minor, integrated != 0,
                                 total_bytes);  // device 0 only
#pragma GCC diagnostic pop
    RegisterPlatform(DeviceType::kCUDA, &platform);
  }
} registrar;

}  // namespace
}  // namespace vllm::platforms
