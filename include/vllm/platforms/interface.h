// Faithful 1:1 port of vllm/platforms/interface.py:134-229 (class Platform) @
// pin e24d1b24 — the device-capability / memory-model seam. See
// .agents/porting-inventory.md §9 note 8 (the platforms/ tree is a faithful
// mirror of the upstream seam, NOT a deviation) and
// .agents/specs/extensibility-platform-seam-2026-07-18.md.
#pragma once

#include <cstddef>
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
// made DATA. Nothing branches on it yet: it is the advertisement item 2 wires
// into the residency path. Current platforms RETAIN host weights (diagnostic
// parity, qwen3_5.cpp:810), so `release_host_weights_after_upload` is false
// everywhere today; item 2 flips it for a discrete GPU without touching model
// code.
struct ResidencyPolicy {
  // May host-side weight bytes be freed once a device-resident copy exists?
  // A discrete GPU reclaims host RAM; unified/host platforms keep the one copy.
  bool release_host_weights_after_upload = false;
  // Is device scratch served from a reuse pool (qwen3_5.cpp DevicePool) rather
  // than freed straight to the driver? cudaMalloc/cudaFree are expensive and
  // illegal under graph capture, so CUDA pools; host platforms do not need it.
  bool uses_device_memory_pool = false;
  // Optional soft cap (bytes) on pooled device scratch; 0 == uncapped.
  size_t device_pool_cap_bytes = 0;
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

  // STUB — item 4 (attention-backend registry). Mirrors interface.py:362-370
  // get_attn_backend_cls / platform-driven priority selection. No consumer yet;
  // returns 0 (no preference) until the registry lands.
  virtual int get_attn_backend_priority() const { return 0; }
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
// for an accelerator and falling back to CPU. In our single-device builds this
// is the device every engine queue runs on, so `CurrentPlatform().is_cuda()` is
// exactly the old per-queue `device.type == kCUDA` test.
Platform& CurrentPlatform();

}  // namespace vllm::platforms
