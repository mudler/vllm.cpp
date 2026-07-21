// vllm.cpp original (vt runtime, inventory deviation §9.1) — the CUDA device
// CAPABILITY seam (BACKEND-CUDA-ARCH-ADDITIVITY, seam-gap #4 of
// .agents/specs/breadth-sweep-plan.md §A.2).
//
// THE GAP THIS CLOSES. The authoritative compute capability `(major, minor)`
// lived only on `vllm::platforms::CudaPlatform` (src/vllm/platforms/cuda.cpp,
// probed once at static registration). The kernel layer under src/vt/cuda/ could
// not see it: cuda_backend.cu probed `cudaDevAttrPageableMemoryAccess` /
// `cudaDevAttrIntegrated` and carried NO SM state at all, so no host launcher
// could ever ask "which architecture am I actually running on?". Any runtime
// per-arch kernel selector (seam-gap #2, src/vt/cuda/cuda_arch_tactics.h) needs
// that answer first. This header is that answer: ONE cached probe of the device
// attributes every arch-dependent decision in the kernel layer is allowed to
// consult.
//
// UPSTREAM. Mirrors what vLLM's platform layer exposes to its kernels —
// `vllm/platforms/cuda.py::CudaPlatform.get_device_capability` (@ pin e24d1b24)
// backed by `torch.cuda.get_device_properties`, which likewise caches
// major/minor/shared-memory-per-block/multiprocessor-count per device rather
// than re-querying per launch. Our `DeviceCapability` type in
// include/vllm/platforms/interface.h:82-125 is the engine-side mirror; this is
// the kernel-side one, and both read the same CUDA attributes.
//
// DISCIPLINE. Every field is a `cudaDeviceGetAttribute` result, cached on first
// use and NEVER re-queried per launch (a per-launch attribute query costs a
// driver round-trip on the hot path). Nothing here is compile-time: the whole
// point is that a fat binary built for several architectures resolves its
// behavior at RUNTIME.
#pragma once

namespace vt::cuda {

// Cached CUDA device capability. `valid == false` means the probe failed (no
// GPU / no driver); callers must treat that as "no arch-specific capability"
// and fall back to the portable path, exactly like the silent-registration
// contract in cuda_backend.cu.
struct DeviceCaps {
  bool valid = false;
  int device = 0;
  // cudaDevAttrComputeCapabilityMajor / ...Minor — the authoritative SM version.
  int sm_major = 0;
  int sm_minor = 0;
  // cudaDevAttrMaxSharedMemoryPerBlockOptin — the opt-in dynamic shared-memory
  // ceiling. Replaces the hardcoded "GB10 caps opt-in shared at ~99 KiB"
  // comment-assumption in cuda_paged_attn.cu (seam-gap #3). GB10/sm_121 reports
  // 101376; other architectures differ and MUST be asked, not assumed.
  int max_shared_memory_per_block_optin = 0;
  // cudaDevAttrMultiProcessorCount — SM count, for occupancy-derived grid sizing.
  int multiprocessor_count = 0;
  // cudaDevAttrPageableMemoryAccess / cudaDevAttrIntegrated — the zero-copy
  // residency probe cuda_backend.cu's registrar already performed; hoisted here
  // so there is exactly ONE device probe in the kernel layer.
  bool pageable_memory_access = false;
  bool integrated = false;

  // Compact SM identity in the familiar `sm_XXX` numbering: sm_121 -> 121,
  // sm_90 -> 90, sm_80 -> 80. Use `sm_major` for family tests (a tactic that
  // works on all of sm_12x asks `sm_major == 12`), and `sm_arch()` only when a
  // specific board is meant.
  int sm_arch() const { return sm_major * 10 + sm_minor; }
};

// Cached capability of the current CUDA device (cudaGetDevice). Safe to call on
// any thread and from any launcher; the probe happens at most once per device.
// Never call before main() — the tactic registrars are table-fill only for
// exactly this reason (see cuda_ops.cu's pre-main discipline note).
const DeviceCaps& GetDeviceCaps();

// Cached capability of an explicit device ordinal.
const DeviceCaps& GetDeviceCaps(int device);

// True when `bytes` of dynamic shared memory can be opted into on this device.
// The hardcoded 100 KiB GB10 assumption becomes a query through this helper.
bool DynamicSmemFits(long long bytes);

}  // namespace vt::cuda
