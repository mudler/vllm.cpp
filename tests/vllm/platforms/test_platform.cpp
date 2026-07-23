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
  // ...so it belongs to no capability family either (get_device_capability None).
  CHECK_FALSE(cpu.is_device_capability_family(120));
  CHECK_FALSE(cpu.is_device_capability_family(0));

  // Portable capability predicates (work row S3): the CPU leg answers the base
  // false to every one — exactly what a `device.type == kCUDA` gate returned on a
  // CPU device, which is what makes the S3 conversions byte-identical here. None
  // of the fp8/fp4/opaque-attention/integrated/static-graph fast paths exist on
  // CPU.
  CHECK_FALSE(cpu.supports_fp8());
  CHECK_FALSE(cpu.cutlass_fp4_supported());
  CHECK_FALSE(cpu.opaque_attention_op());
  CHECK_FALSE(cpu.is_integrated_gpu());
  CHECK_FALSE(cpu.support_static_graph_mode());
  // S7 residency / FA2 POLICY — base false on CPU: it reads host weights/state
  // in place (no staging) and has no FA2 kernel, exactly what the converted
  // `device==kCUDA` gates answered on a CPU device (byte-identical).
  CHECK_FALSE(cpu.needs_weight_staging());
  CHECK_FALSE(cpu.supports_fa2_attention());
  // Proof that needs_weight_staging() is NOT is_unified_memory() in disguise: CPU
  // is UNIFIED (host==device memory) yet does NOT stage. The two predicates
  // DIVERGE here (unified true, staging false), so a staging gate keyed on
  // is_unified_memory() would wrongly send CPU down the device-staging branch —
  // the memory-property flip the dedicated policy avoids.
  CHECK(cpu.is_unified_memory());
  CHECK(cpu.needs_weight_staging() != cpu.is_unified_memory());

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

// A reserved DeviceType with no platform behind it must throw, never hand back a
// null/garbage Platform. kMETAL was the stand-in for "reserved but
// unimplemented" — but a VLLM_CPP_METAL build on a Metal-capable host now
// genuinely registers it, so the case uses a slot that is still empty there.
// kXPU is the right stand-in: it is HW-BLOCKED with no local target and no
// implementation (.agents/specs/backend-fanout-metal-vulkan-xpu.md § Scope), so
// the property under test keeps a live subject on BOTH platforms instead of
// being compiled away on macOS. Mirrors tests/vt/test_backend.cpp.
TEST_CASE("unregistered platform throws / HasPlatform reports false") {
#ifndef VLLM_CPP_METAL
  CHECK_FALSE(HasPlatform(DeviceType::kMETAL));
  CHECK_THROWS_AS(GetPlatform(DeviceType::kMETAL), std::runtime_error);
#endif
  CHECK_FALSE(HasPlatform(DeviceType::kXPU));
  CHECK_THROWS_AS(GetPlatform(DeviceType::kXPU), std::runtime_error);
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

TEST_CASE("is_device_capability_family matches any <major>.x (interface.py:441-476)") {
  // sm_120 and sm_121 share the 12.x family; a different major does not.
  FakeCapabilityPlatform sm121(DeviceCapability{12, 1});
  CHECK(sm121.is_device_capability_family(120));  // 121//10 == 120//10 == 12
  CHECK(sm121.is_device_capability_family(121));
  CHECK(sm121.is_device_capability_family(129));
  CHECK_FALSE(sm121.is_device_capability_family(100));  // 10.x
  CHECK_FALSE(sm121.is_device_capability_family(89));    // 8.x

  // The base capability predicates default to false on any platform that does not
  // override them (mirrors upstream's `Platform` base) — the FakeCapabilityPlatform
  // does not, so it answers false even though its device_type() is kCUDA. Proves
  // the defaults live on the base and the CUDA ANSWERS live in the CUDA leg.
  CHECK_FALSE(sm121.supports_fp8());
  CHECK_FALSE(sm121.cutlass_fp4_supported());
  CHECK_FALSE(sm121.opaque_attention_op());
  CHECK_FALSE(sm121.support_static_graph_mode());
  // S7 additions default false on the base too (the CUDA ANSWER lives in the leg).
  CHECK_FALSE(sm121.needs_weight_staging());
  CHECK_FALSE(sm121.supports_fa2_attention());
}

// The CUDA leg's capability ANSWERS, exercised only where a real CUDA platform is
// registered (the dgx CUDA build / a GPU box). This is the executable proof that
// each S3-converted gate is byte-identical: the predicate returns `true` on this
// CUDA device — exactly what the former `device.type == kCUDA` returned — so the
// 27B fp4-activation razor and the fp8-fused paths select the same kernels.
TEST_CASE("CUDA leg capability values (GPU build only)") {
  if (!HasPlatform(DeviceType::kCUDA)) return;  // CPU-only tier: nothing to assert
  Platform& cu = GetPlatform(DeviceType::kCUDA);
  REQUIRE(cu.get_device_capability().present());
  const int cc = cu.get_device_capability().to_int();  // 121 on GB10 (sm_121)

  // supports_fp8 == has_device_capability(8,9); GB10 (>= 8.9) -> true. This is
  // what the converted fp8-fused gates now read; it must equal the old
  // `device==kCUDA` (true) on this device.
  CHECK(cu.supports_fp8() == cu.has_device_capability(8, 9));
  CHECK(cu.supports_fp8());  // true on GB10

  // cutlass_fp4_supported: CC in [100,130). GB10 (121) -> true. This is what the
  // converted true-W4A4 fp4-activation gates (the 27B razor) now read.
  CHECK(cu.cutlass_fp4_supported() == (cc >= 100 && cc < 130));
  CHECK(cu.cutlass_fp4_supported());  // true on GB10

  // opaque_attention_op / support_static_graph_mode: unconditional true on CUDA
  // (cuda.py:570 / :662). support_static_graph_mode backs the converted decode
  // graph-capture gates.
  CHECK(cu.opaque_attention_op());
  CHECK(cu.support_static_graph_mode());

  // is_integrated_gpu: GB10 (Grace-Blackwell UMA) reports integrated; it is also
  // unified memory, so the two agree on this box. This backs the converted runner
  // device-combine/scatter gates (async sampling into device-addressable host mem).
  CHECK(cu.is_integrated_gpu() == cu.is_unified_memory());
  CHECK(cu.is_integrated_gpu());  // true on GB10

  // S7 needs_weight_staging: unconditional true on CUDA — the CUDA path stages
  // host tensors into device-resident buffers even though GB10 is physically
  // unified. This is what the converted residency / merged-projection /
  // packed-decode gates now read; it must equal the old `device==kCUDA` (true) on
  // this device. On GB10 it AGREES with is_unified_memory() (both true — GB10 is
  // unified AND stages), which is exactly why is_unified_memory() is the wrong
  // predicate: the two DIVERGE on CPU (see the CPU-leg case — unified yet
  // NON-staging), so a staging gate keyed on is_unified_memory() would flip CPU.
  CHECK(cu.needs_weight_staging());

  // S7 supports_fa2_attention: unconditional true on CUDA — the vendored FA2
  // split-KV kernel exists here. Backs the converted FA2 dtype-selection gates.
  CHECK(cu.supports_fa2_attention());

  // Family membership: GB10 is 12.x.
  CHECK(cu.is_device_capability_family(120));
  CHECK_FALSE(cu.is_device_capability_family(80));
}

// --- BACKEND-PLATFORM item 2: residency-policy CONSUMPTION -------------------
// The model (qwen3_5.cpp) no longer decides host-free / load-stream inline; it
// reads GetPlatform(<obj>.device.type).residency_policy() and derives the
// decision through ShouldReleaseHostWeights / ShouldInterleaveLoadStream — the
// exact helpers exercised here. A new GPU changes ONLY residency_policy() values;
// this logic and the model are unchanged.
namespace {
// A synthetic platform carrying an arbitrary ResidencyPolicy, so a discrete-GPU /
// unified-GPU / retain-host policy can be exercised on the CPU tier without a GPU.
class FakeResidencyPlatform final : public Platform {
 public:
  explicit FakeResidencyPlatform(ResidencyPolicy p) : policy_(p) {}
  DeviceType device_type() const override { return DeviceType::kCUDA; }
  vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
  DeviceCapability get_device_capability() const override { return {}; }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
  ResidencyPolicy residency_policy() const override { return policy_; }

 private:
  ResidencyPolicy policy_;
};
}  // namespace

using vllm::platforms::ShouldInterleaveLoadStream;
using vllm::platforms::ShouldReleaseHostWeights;

TEST_CASE("residency helpers split POLICY (platform) from KERNEL-PATH (marlin)") {
  // GB10/unified today: host weights freed after upload, pool on, uncapped.
  ResidencyPolicy gb10;
  gb10.release_host_weights_after_upload = true;
  gb10.uses_device_memory_pool = true;
  gb10.device_pool_cap_bytes = 0;

  // Host-free = policy AND marlin-committed AND env-allow (the orthogonal AND).
  CHECK(ShouldReleaseHostWeights(gb10, /*marlin=*/true, /*env=*/true));
  // KERNEL-PATH gate off (wmma re-reads host) ⇒ never free, even with the policy.
  CHECK_FALSE(ShouldReleaseHostWeights(gb10, /*marlin=*/false, /*env=*/true));
  // House A/B override (VT_MOE_HOST_FREE=0) ⇒ retain, even with the policy.
  CHECK_FALSE(ShouldReleaseHostWeights(gb10, /*marlin=*/true, /*env=*/false));
  // Load-stream interleave = policy AND marlin-committed (reproduces the old
  // `device==kCUDA && MarlinMoeEnabled()` gate for the CUDA platform).
  CHECK(ShouldInterleaveLoadStream(gb10, /*marlin=*/true));
  CHECK_FALSE(ShouldInterleaveLoadStream(gb10, /*marlin=*/false));

  // A retain-host platform (unified/CPU semantics: policy false) ⇒ neither the
  // host-free nor the interleave fires, regardless of the kernel path — the model
  // takes the materialize-all / retain branch with NO code change.
  ResidencyPolicy retain;  // all defaults false / 0
  CHECK_FALSE(ShouldReleaseHostWeights(retain, /*marlin=*/true, /*env=*/true));
  CHECK_FALSE(ShouldInterleaveLoadStream(retain, /*marlin=*/true));
}

TEST_CASE("residency_policy carries per-platform values the model consumes") {
  // The seam is additive: a platform advertises its residency policy and the
  // model reads it. Prove the policy round-trips through the Platform API and
  // drives the decisions — the discrete-GPU case sets a pool cap with no model
  // edit.
  ResidencyPolicy discrete;
  discrete.release_host_weights_after_upload = true;   // reclaim host RAM
  discrete.uses_device_memory_pool = true;
  discrete.device_pool_cap_bytes = size_t{1} << 30;    // 1 GiB soft cap
  FakeResidencyPlatform gpu(discrete);
  const ResidencyPolicy got = gpu.residency_policy();
  CHECK(got.release_host_weights_after_upload);
  CHECK(got.uses_device_memory_pool);
  CHECK(got.device_pool_cap_bytes == (size_t{1} << 30));
  CHECK(ShouldReleaseHostWeights(got, /*marlin=*/true, /*env=*/true));
  CHECK(ShouldInterleaveLoadStream(got, /*marlin=*/true));

  // CPU platform (the real one) advertises the unified-host retain/no-pool policy,
  // so its derived decisions are always retain — the consumption is device-keyed.
  const ResidencyPolicy cpu = GetPlatform(DeviceType::kCPU).residency_policy();
  CHECK_FALSE(cpu.release_host_weights_after_upload);
  CHECK_FALSE(ShouldReleaseHostWeights(cpu, /*marlin=*/true, /*env=*/true));
  CHECK_FALSE(ShouldInterleaveLoadStream(cpu, /*marlin=*/true));
}
