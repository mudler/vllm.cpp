// Faithful port of vllm/platforms/cuda.py (CudaPlatform) @ pin e24d1b24 — the
// CUDA leg of the Platform seam. Self-registers kCUDA via a static Registrar
// that probes device presence + compute capability, copying the
// cuda_backend.cu registrar idiom (silent on a machine with the toolkit but no
// usable GPU). Compiled only in CUDA builds (CMake target_sources gate).
#include <cuda_runtime.h>

#include "vllm/platforms/interface.h"

#include "vt/backend.h"

namespace vllm::platforms {
namespace {

class CudaPlatform final : public Platform {
 public:
  CudaPlatform(int cc_major, int cc_minor) : cap_{cc_major, cc_minor} {}

  DeviceType device_type() const override { return DeviceType::kCUDA; }
  Backend& backend() const override { return vt::GetBackend(DeviceType::kCUDA); }

  // cuda.py get_device_capability: torch.cuda.get_device_capability probed once
  // at registration (device 0 only for now, matching the backend registrar).
  DeviceCapability get_device_capability() const override { return cap_; }

  // interface.py:181-187 supported_dtypes order (bf16 default fallback).
  std::vector<DType> supported_dtypes() const override {
    return {DType::kBF16, DType::kF16, DType::kF32};
  }

  // The residency/memory-model advertisement (the PR #4 debt as data). Current
  // behavior: host weights RETAINED (qwen3_5.cpp:810 diagnostic parity) and a
  // device reuse pool (qwen3_5.cpp DevicePool). A discrete-GPU platform flips
  // release_host_weights_after_upload without touching model code (item 2).
  ResidencyPolicy residency_policy() const override {
    ResidencyPolicy p;
    p.release_host_weights_after_upload = false;  // retained today
    p.uses_device_memory_pool = true;             // qwen3_5.cpp DevicePool
    p.device_pool_cap_bytes = 0;                  // uncapped
    return p;
  }

  // Capability-ordered attention-backend priority — a faithful port of the
  // non-MLA branch of vllm/platforms/cuda.py::_get_backend_priorities:154-166 @
  // pin e24d1b24. Our gate models (Qwen3 dense + GDN) are non-MLA; the MLA
  // branch (sparse/dense MLA orderings) is out of scope until an MLA model ports.
  //   * device_capability.major == 10 (Blackwell datacenter, sm_100):
  //         FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT
  //   * else (incl. GB10 sm_121 == major 12, and Ampere/Hopper):
  //         FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT
  // On GB10 (major 12 → else branch) FLASH_ATTN is preferred; since it is the
  // only registered CUDA backend today, selection is behavior-preserving
  // (SelectAttentionBackendName walks this list and returns "FLASH_ATTN").
  std::vector<std::string> get_attn_backend_priority() const override {
    if (cap_.major == 10) {
      return {"FLASHINFER", "FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION",
              "TURBOQUANT"};
    }
    return {"FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION",
            "TURBOQUANT"};
  }

 private:
  DeviceCapability cap_;
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
    // GCC 13 false-positive: -Wdangling-pointer mis-flags a static local with a
    // vtable constructed from automatic ints, though CudaPlatform copies both
    // into cap_ by value (no pointer/reference to major/minor is retained). The
    // static outlives the registrar as RegisterPlatform requires.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-pointer"
    static CudaPlatform platform(major, minor);  // device 0 only for now
#pragma GCC diagnostic pop
    RegisterPlatform(DeviceType::kCUDA, &platform);
  }
} registrar;

}  // namespace
}  // namespace vllm::platforms
