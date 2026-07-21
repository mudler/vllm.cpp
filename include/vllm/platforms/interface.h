// Faithful 1:1 port of vllm/platforms/interface.py:134-229 (class Platform) @
// pin e24d1b24 — the device-capability / memory-model seam. See
// .agents/porting-inventory.md §9 note 8 (the platforms/ tree is a faithful
// mirror of the upstream seam, NOT a deviation) and
// .agents/specs/extensibility-platform-seam-2026-07-18.md.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

namespace vllm::platforms {

using vt::Backend;
using vt::DeviceType;
using vt::DType;

// Mirrors vllm/platforms/interface.py:69-131 (class DeviceCapability): a device's
// compute capability as (major, minor). `present() == false` models a platform
// with no queryable capability (CPU) — upstream get_device_capability() -> None.
struct DeviceCapability {
  int major = -1;
  int minor = -1;
  bool present() const { return major >= 0; }
  // interface.py:124 to_int — <major><minor>, assuming a single-digit minor.
  int to_int() const { return major * 10 + minor; }
};

// Discrete-vs-unified memory residency policy — the PR #4 discrete-memory debt
// made DATA and, since item 2 (BACKEND-PLATFORM), CONSUMED by the model residency
// path (qwen3_5.cpp) rather than decided inline. The host-free / load-stream /
// device-scratch-pool decisions read these fields from
// `GetPlatform(<obj>.device.type).residency_policy()`, so a new (e.g. discrete)
// GPU changes ONLY its platform's policy values and the model code is UNCHANGED.
// On GB10/unified today CUDA advertises release=true (the routed MoE experts'
// host mirror IS freed after the device Marlin build — ENG-MOE-HOSTFREE /
// -LOADSTREAM) + pool=true/uncapped (the DevicePool); CPU advertises the
// unified-host retain/no-pool defaults below.
struct ResidencyPolicy {
  // May host-side weight bytes be freed once a device-resident copy exists?
  // A discrete GPU reclaims host RAM; unified/host platforms keep the one copy.
  // Consumed by ShouldReleaseHostWeights / ShouldInterleaveLoadStream below.
  bool release_host_weights_after_upload = false;
  // Is device scratch served from a reuse pool (qwen3_5.cpp DevicePool) rather
  // than freed straight to the driver? cudaMalloc/cudaFree are expensive and
  // illegal under graph capture, so CUDA pools; host platforms do not need it.
  bool uses_device_memory_pool = false;
  // Optional soft cap (bytes) on pooled device scratch; 0 == uncapped (today).
  // Consumed by the DevicePool: a discrete GPU sets a bound; GB10 leaves it 0.
  size_t device_pool_cap_bytes = 0;
};

// Residency DECISIONS derived from a ResidencyPolicy — the single, testable place
// the host-weight-release and per-layer load-stream-interleave questions are
// answered from platform data (BACKEND-PLATFORM item 2). Consumed by
// qwen3_5.cpp; unit-tested in tests/vllm/platforms/test_platform.cpp so the
// wiring the model uses is exercised on the CPU tier. The kernel-path predicate
// `marlin_committed` (is the Marlin resident the committed compute path, so the
// host-reading wmma fallback can never re-read the freed bytes) is ORTHOGONAL to
// the platform POLICY and is supplied by the caller; `host_free_env` is the
// VT_MOE_HOST_FREE same-binary A/B override (house convention). A discrete GPU
// flips ResidencyPolicy and both this logic and the model are unchanged.
inline bool ShouldReleaseHostWeights(const ResidencyPolicy& policy,
                                     bool marlin_committed, bool host_free_env) {
  return marlin_committed && policy.release_host_weights_after_upload &&
         host_free_env;
}
inline bool ShouldInterleaveLoadStream(const ResidencyPolicy& policy,
                                       bool marlin_committed) {
  return marlin_committed && policy.release_host_weights_after_upload;
}

// The selection inputs of vllm/platforms/cuda.py::_get_backend_priorities @ pin
// e24d1b24 (`use_mla`, `device_capability`, `num_heads`, `kv_cache_dtype`) plus
// the sparse flag that `AttentionBackend.is_sparse()` /
// `vllm/v1/attention/backend.py:307-360 validate_configuration` keys on. The
// capability itself is NOT a field — it is the platform's own
// `get_device_capability()`, exactly as upstream passes it in.
//
// Defaults reproduce today's non-MLA dense selection EXACTLY, so every existing
// caller (`get_attn_backend_priority()` with no argument) is unchanged.
struct AttnSelectorConfig {
  // model_config.use_mla — the MLA branch of _get_backend_priorities:93-142.
  // On our gate models (Qwen3 dense / GDN) this is false.
  bool use_mla = false;
  // Sparse (DSA / V3.2 indexer) attention. NOT a _get_backend_priorities input
  // upstream — it is the `is_sparse()` check in validate_configuration that
  // REJECTS a sparse backend for a dense request and vice versa. Carried here so
  // the selector can apply that filter (see SelectAttentionBackendName): this is
  // the seam a future DSA/sparse-MLA backend slots into with ZERO selector edit.
  bool use_sparse = false;
  // cuda.py:105 `num_heads is not None and num_heads <= 16` (sm_100 sparse-MLA
  // ordering only). 0 == unknown, upstream's `None`.
  int num_heads = 0;
  // cuda.py:96 `is_quantized_kv_cache(kv_cache_dtype)` (sm_100 sparse-MLA
  // ordering only). Our KV cache is bf16/f32 today, so false.
  bool quantized_kv_cache = false;
};

// Faithful port of vllm/platforms/interface.py:134-229 `class Platform`. Exposes
// the capability queries the engine/model code branches on. A Platform COMPOSES
// a vt::Backend (include/vt/backend.h) for the concrete device ops — it does NOT
// replace it: the memory-model queries delegate to the backend, while
// capability/dtype/residency queries are platform metadata.
class Platform {
 public:
  virtual ~Platform() = default;

  virtual DeviceType device_type() const = 0;

  // interface.py:189-202 is_cuda / is_cpu.
  bool is_cuda() const { return device_type() == DeviceType::kCUDA; }
  bool is_cpu() const { return device_type() == DeviceType::kCPU; }

  // The composed vt::Backend (index 0). Memory-model queries delegate here.
  virtual Backend& backend() const = 0;

  // True when host and device share one address space (backend.h:45). CPU and
  // GB10/Apple are unified; a discrete GPU is not.
  bool is_unified_memory() const { return backend().UnifiedMemory(); }

  // Graph/command capture capability (backend.h:80 SupportsGraphCapture).
  bool supports_graph_capture() const { return backend().SupportsGraphCapture(); }

  // interface.py:409-415 get_device_capability. `present() == false` on CPU.
  virtual DeviceCapability get_device_capability() const = 0;

  // interface.py:417-439 has_device_capability — is this platform >= a required
  // (major, minor)? False when the platform has no queryable capability (CPU).
  bool has_device_capability(int major, int minor) const;

  // interface.py:181-187 supported_dtypes. The FIRST entry is the default
  // fallback for the platform ("auto" dtype resolution).
  virtual std::vector<DType> supported_dtypes() const = 0;

  // Discrete-vs-unified residency/memory-model policy (folds the PR #4 debt).
  virtual ResidencyPolicy residency_policy() const = 0;

  // Capability-ordered attention-backend priority: the ordered list of backend
  // NAMES this platform prefers, highest-priority first. Mirrors
  // vllm/platforms/cuda.py::_get_backend_priorities (the ordered
  // AttentionBackendEnum list) and is consumed by
  // vllm::v1::SelectAttentionBackendName (mirror of get_attn_backend_cls), which
  // returns the first REGISTERED name. The base default is empty (no preference);
  // CudaPlatform/CpuPlatform override with the vLLM-mirrored lists. (Item 4 —
  // attention-backend registry seam; formerly a stub returning 0.)
  //
  // `cfg` carries upstream's `use_mla` (+ the sparse/num_heads/kv-dtype inputs).
  // The DEFAULT argument lives here only: overrides declare the parameter
  // without a default, and every call through a `Platform&` picks this one up,
  // so the pre-MLA zero-argument call sites are unchanged.
  virtual std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg = AttnSelectorConfig{}) const {
    (void)cfg;
    return {};
  }

  // Capability-ordered MLA *prefill* backend priority — a separate selector
  // upstream: vllm/v1/attention/backends/mla/prefill/selector.py:47-76
  // `_get_mla_prefill_backend_priorities`. Distinct from the decode/dense list
  // above because MLA runs a materialized-MHA prefill against a different
  // backend family. Base default empty; CudaPlatform mirrors the upstream lists.
  virtual std::vector<std::string> get_mla_prefill_backend_priority() const {
    return {};
  }
};

// Self-registered per DeviceType, copying the RegisterBackend/GetBackend
// static-init idiom (src/vt/backend.cpp). All registration completes before
// main() runs (platforms register via static initializers); afterwards these
// are lock-free reads with no synchronization.
void RegisterPlatform(DeviceType type, Platform* platform);
// The platform for a specific DeviceType (throws if none registered).
Platform& GetPlatform(DeviceType type);
bool HasPlatform(DeviceType type);

// The process's active compute platform: the highest-priority registered
// accelerator, else CPU — mirrors vLLM resolving `current_platform` by probing
// for an accelerator and falling back to CPU. This answers ONLY the process-level
// "what accelerator is this process on" question. It is NOT a per-tensor device
// test: on a GPU box it returns the CUDA platform regardless of the object being
// operated on, so a CPU queue/tensor on a GPU box would wrongly read as CUDA. For
// per-object device dispatch use `GetPlatform(<obj>.device.type)` (BACKEND-PLATFORM).
Platform& CurrentPlatform();

}  // namespace vllm::platforms
