// CPU-tier contract for the Platform seam (a faithful port of
// vllm/platforms/interface.py:134-229). Mirrors the backend-registry test style
// (tests/vt/test_backend.cpp): registration + the CPU capability values, plus
// the has_device_capability lexicographic logic exercised through a synthetic
// platform (a CUDA device is not available on the CPU test tier).
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

using vllm::platforms::CurrentPlatform;
using vllm::platforms::DeviceCapability;
using vllm::platforms::GetPlatform;
using vllm::platforms::HasPlatform;
using vllm::platforms::Platform;
using vllm::platforms::RegisterPlatform;
using vllm::platforms::ResidencyPolicy;
using vt::DeviceType;
using vt::DType;

TEST_CASE("CPU platform is self-registered and advertises CPU capabilities") {
  REQUIRE(HasPlatform(DeviceType::kCPU));
  Platform& cpu = GetPlatform(DeviceType::kCPU);

  CHECK(cpu.device_type() == DeviceType::kCPU);
  CHECK(cpu.is_cpu());
  CHECK_FALSE(cpu.is_cuda());

  // Composes the vt::Backend: unified host memory, no graph capture.
  CHECK(cpu.is_unified_memory());
  CHECK(&cpu.backend() == &vt::GetBackend(DeviceType::kCPU));
  CHECK_FALSE(cpu.supports_graph_capture());

  // A CPU has no queryable compute capability (interface.py -> None).
  CHECK_FALSE(cpu.get_device_capability().present());
  CHECK_FALSE(cpu.has_device_capability(0, 0));
  CHECK_FALSE(cpu.has_device_capability(9, 0));

  // supported_dtypes order (bf16 default fallback first).
  const std::vector<DType> expected{DType::kBF16, DType::kF16, DType::kF32};
  CHECK(cpu.supported_dtypes() == expected);

  // Unified host memory: no host-weight release, no device pool.
  const ResidencyPolicy policy = cpu.residency_policy();
  CHECK_FALSE(policy.release_host_weights_after_upload);
  CHECK_FALSE(policy.uses_device_memory_pool);
  CHECK(policy.device_pool_cap_bytes == 0);

  // Attention-backend priority (item 4): CPU mirrors cpu.py's single-backend
  // preference (CPU_ATTN) then our FLASH_ATTN fallthrough (the layout the CPU
  // paged-attn kernel actually uses). The registry-driven selection is covered
  // in test_attn_backend_registry.cpp.
  const std::vector<std::string> cpu_priority{"CPU_ATTN", "FLASH_ATTN"};
  CHECK(cpu.get_attn_backend_priority() == cpu_priority);
}

TEST_CASE("CurrentPlatform resolves accelerator-first, else falls back to CPU") {
  // CurrentPlatform() answers the PROCESS-level "what accelerator is this
  // process on" question (interface.h:104): accelerator-first, CPU fallback. It
  // is NOT a per-tensor device test — a CPU queue/tensor on a GPU box keys on
  // GetPlatform(device.type), never on this (see BACKEND-PLATFORM). So the
  // fallback-to-CPU assertion can only hold on the CPU-only tier; on a GPU box
  // (or the DGX CUDA build) an accelerator IS registered and wins.
  Platform& current = CurrentPlatform();
  const bool has_accelerator =
      HasPlatform(DeviceType::kCUDA) || HasPlatform(DeviceType::kXPU) ||
      HasPlatform(DeviceType::kVULKAN) || HasPlatform(DeviceType::kMETAL);
  if (has_accelerator) {
    // Accelerator-first: the process platform is the accelerator, not CPU.
    CHECK_FALSE(current.is_cpu());
  } else {
    // CPU-only tier: the resolution falls back to CPU.
    CHECK(current.is_cpu());
    CHECK(&current == &GetPlatform(DeviceType::kCPU));
  }
  // Device-correct invariant on every tier: the CPU platform is always CPU.
  CHECK(GetPlatform(DeviceType::kCPU).is_cpu());
}

TEST_CASE("unregistered platform throws / HasPlatform reports false") {
  CHECK_FALSE(HasPlatform(DeviceType::kMETAL));
  CHECK_THROWS_AS(GetPlatform(DeviceType::kMETAL), std::runtime_error);
}

TEST_CASE("DeviceCapability comparison is lexicographic on (major, minor)") {
  CHECK(DeviceCapability{8, 6}.to_int() == 86);
  CHECK(DeviceCapability{12, 1}.present());
  CHECK_FALSE(DeviceCapability{}.present());
}

namespace {
// A synthetic platform exposing a fixed compute capability, so
// has_device_capability's lexicographic logic can be exercised without a GPU.
class FakeCapabilityPlatform final : public Platform {
 public:
  explicit FakeCapabilityPlatform(DeviceCapability cap) : cap_(cap) {}
  DeviceType device_type() const override { return DeviceType::kCUDA; }
  vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
  DeviceCapability get_device_capability() const override { return cap_; }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
  ResidencyPolicy residency_policy() const override { return {}; }

 private:
  DeviceCapability cap_;
};
}  // namespace

TEST_CASE("has_device_capability tests platform capability >= required") {
  FakeCapabilityPlatform sm121(DeviceCapability{12, 1});
  // Equal and lower requirements pass; higher major or minor fail.
  CHECK(sm121.has_device_capability(12, 1));
  CHECK(sm121.has_device_capability(12, 0));
  CHECK(sm121.has_device_capability(8, 9));
  CHECK_FALSE(sm121.has_device_capability(12, 2));
  CHECK_FALSE(sm121.has_device_capability(13, 0));
}
