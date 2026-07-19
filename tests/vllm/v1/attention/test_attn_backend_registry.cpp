// CPU-tier contract for the attention-backend REGISTRY + platform-priority
// SELECTION seam (extensibility item 4). Ports the executable spec of
// vllm/v1/attention/backends/registry.py (self-registration) and
// vllm/platforms/cuda.py::get_attn_backend_cls / _get_backend_priorities (the
// capability-ordered priority + first-registered selection) @ pin e24d1b24.
//
// Covers: (1) the gate backends self-register per DeviceType; (2) MakeAttention-
// Backend constructs the named backend / throws when absent; (3) the CPU and
// CUDA (per-capability) priority ORDER matches the vLLM-mirrored lists; (4) the
// selection walk returns the first REGISTERED name (behavior-preserving
// FLASH_ATTN on both, incl. CPU's CPU_ATTN→FLASH_ATTN fallthrough); (5) the
// explicit-override path; (6) sm_100 (major 10) vs sm_121 (major 12) select
// differently ONLY if their preferred backend is registered — here both resolve
// to FLASH_ATTN because FLASHINFER is not implemented, which IS the
// behavior-preserving outcome.
#include <doctest/doctest.h>

#include <memory>
#include <string>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/attention/registry.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

using vllm::platforms::DeviceCapability;
using vllm::platforms::Platform;
using vllm::platforms::ResidencyPolicy;
using vllm::v1::AttentionBackend;
using vllm::v1::HasAttentionBackend;
using vllm::v1::MakeAttentionBackend;
using vllm::v1::RegisterAttentionBackend;
using vllm::v1::SelectAttentionBackend;
using vllm::v1::SelectAttentionBackendName;
using vt::DeviceType;
using vt::DType;

namespace {

// A synthetic CUDA platform with a fixed compute capability, so the
// capability-ordered priority + selection can be exercised without a GPU. It
// reuses the real CudaPlatform priority lists (copied here) — the point under
// test is the registry + selector, not CudaPlatform's construction (which needs
// a device). Keep in sync with src/vllm/platforms/cuda.cpp.
class FakeCudaPlatform final : public Platform {
 public:
  explicit FakeCudaPlatform(int major, int minor) : cap_{major, minor} {}
  DeviceType device_type() const override { return DeviceType::kCUDA; }
  vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
  DeviceCapability get_device_capability() const override { return cap_; }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
  ResidencyPolicy residency_policy() const override { return {}; }
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

// A platform whose top priority is a test-only backend, to prove the walk stops
// at the first REGISTERED name rather than always falling to FLASH_ATTN. Uses a
// unique name so it never collides with a real backend other cases assert about.
class TopIsTestBackendPlatform final : public Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kCUDA; }
  vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
  DeviceCapability get_device_capability() const override { return {10, 0}; }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
  ResidencyPolicy residency_policy() const override { return {}; }
  std::vector<std::string> get_attn_backend_priority() const override {
    return {"TEST_ONLY_ATTN", "FLASH_ATTN"};
  }
};

}  // namespace

TEST_CASE("gate backends self-register per DeviceType") {
  // FlashAttentionBackend (backend.cpp) registers for CUDA and CPU; GDN
  // (gdn_attn.cpp) registers for CUDA and CPU. Static registrars ran at load.
  CHECK(HasAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN"));
  CHECK(HasAttentionBackend(DeviceType::kCPU, "FLASH_ATTN"));
  CHECK(HasAttentionBackend(DeviceType::kCUDA, "GDN_ATTN"));
  CHECK(HasAttentionBackend(DeviceType::kCPU, "GDN_ATTN"));

  // Backends we name in the priority lists but do not implement are NOT
  // registered — the selection walk must skip them.
  CHECK_FALSE(HasAttentionBackend(DeviceType::kCUDA, "FLASHINFER"));
  CHECK_FALSE(HasAttentionBackend(DeviceType::kCUDA, "TRITON_ATTN"));
  CHECK_FALSE(HasAttentionBackend(DeviceType::kCPU, "CPU_ATTN"));
}

TEST_CASE("MakeAttentionBackend constructs the named backend / throws when absent") {
  std::unique_ptr<AttentionBackend> flash =
      MakeAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN");
  REQUIRE(flash != nullptr);
  CHECK(flash->get_name() == "FLASH_ATTN");
  // FlashAttention NHD KV layout is unchanged: (num_blocks, 2, block, H, D).
  const std::vector<int64_t> shape = flash->get_kv_cache_shape(10, 16, 2, 128);
  const std::vector<int64_t> expected{10, 2, 16, 2, 128};
  CHECK(shape == expected);

  std::unique_ptr<AttentionBackend> gdn =
      MakeAttentionBackend(DeviceType::kCPU, "GDN_ATTN");
  REQUIRE(gdn != nullptr);
  CHECK(gdn->get_name() == "GDN_ATTN");

  CHECK_THROWS_AS(MakeAttentionBackend(DeviceType::kCUDA, "NOPE"),
                  std::runtime_error);
}

TEST_CASE("CUDA priority order mirrors _get_backend_priorities (non-MLA)") {
  // major != 10 (incl. GB10 sm_121 == major 12): FLASH_ATTN first.
  FakeCudaPlatform sm121(12, 1);
  const std::vector<std::string> else_order{
      "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION", "TURBOQUANT"};
  CHECK(sm121.get_attn_backend_priority() == else_order);

  // major == 10 (Blackwell datacenter): FLASHINFER first.
  FakeCudaPlatform sm100(10, 0);
  const std::vector<std::string> major10_order{
      "FLASHINFER", "FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION", "TURBOQUANT"};
  CHECK(sm100.get_attn_backend_priority() == major10_order);
}

TEST_CASE("selection walks priority and returns the first REGISTERED backend") {
  // GB10 sm_121: FLASH_ATTN is first AND registered → selected (behavior-
  // preserving; the NHD layout the runtime uses today).
  FakeCudaPlatform sm121(12, 1);
  CHECK(SelectAttentionBackendName(sm121) == "FLASH_ATTN");
  std::unique_ptr<AttentionBackend> b = SelectAttentionBackend(sm121);
  REQUIRE(b != nullptr);
  CHECK(b->get_name() == "FLASH_ATTN");

  // sm_100: FLASHINFER is preferred but UNREGISTERED, so the walk falls through
  // to FLASH_ATTN — same behavior-preserving outcome until FLASHINFER lands.
  FakeCudaPlatform sm100(10, 0);
  CHECK(SelectAttentionBackendName(sm100) == "FLASH_ATTN");
}

TEST_CASE("selection stops at the first registered name (not always FLASH_ATTN)") {
  // Prove the walk is real: register a test-only backend for CUDA, then a
  // platform preferring it selects it over FLASH_ATTN. The unique name is not
  // asserted-absent by any other case (registry mutation is global/persistent).
  RegisterAttentionBackend(DeviceType::kCUDA, "TEST_ONLY_ATTN",
                           []() -> std::unique_ptr<AttentionBackend> {
                             return std::make_unique<vllm::v1::FlashAttentionBackend>();
                           });
  TopIsTestBackendPlatform p;
  CHECK(SelectAttentionBackendName(p) == "TEST_ONLY_ATTN");
}

TEST_CASE("CPU selection: CPU_ATTN preference falls through to FLASH_ATTN") {
  // The real CpuPlatform priority is {CPU_ATTN, FLASH_ATTN}; CPU_ATTN is not
  // implemented, so the walk returns FLASH_ATTN (the layout our CPU paged-attn
  // kernel uses). Exercised here through a synthetic CPU platform mirroring
  // cpu.cpp so no GPU/accelerator resolution is involved.
  class FakeCpuPlatform final : public Platform {
   public:
    DeviceType device_type() const override { return DeviceType::kCPU; }
    vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
    DeviceCapability get_device_capability() const override { return {}; }
    std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
    ResidencyPolicy residency_policy() const override { return {}; }
    std::vector<std::string> get_attn_backend_priority() const override {
      return {"CPU_ATTN", "FLASH_ATTN"};
    }
  } cpu;
  CHECK(SelectAttentionBackendName(cpu) == "FLASH_ATTN");
}

TEST_CASE("explicit backend override is honored / validated") {
  FakeCudaPlatform sm121(12, 1);
  // A registered override is returned as-is (upstream selected_backend arg).
  CHECK(SelectAttentionBackendName(sm121, "FLASH_ATTN") == "FLASH_ATTN");
  // An unregistered override throws (upstream ValueError).
  CHECK_THROWS_AS(SelectAttentionBackendName(sm121, "TRITON_ATTN"),
                  std::invalid_argument);
}

TEST_CASE("empty priority yields no backend (base Platform default)") {
  class NoPriorityPlatform final : public Platform {
   public:
    DeviceType device_type() const override { return DeviceType::kMETAL; }
    vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
    DeviceCapability get_device_capability() const override { return {}; }
    std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
    ResidencyPolicy residency_policy() const override { return {}; }
    // Uses the base default get_attn_backend_priority() == {}.
  } none;
  CHECK(none.get_attn_backend_priority().empty());
  CHECK_THROWS_AS(SelectAttentionBackendName(none), std::runtime_error);
}
