// CPU-tier contract for the Platform seam (a faithful port of
// vllm/platforms/interface.py:134-229). Mirrors the backend-registry test style
// (tests/vt/test_backend.cpp): registration + the CPU capability values, plus
// the has_device_capability lexicographic logic exercised through a synthetic
// platform (a CUDA device is not available on the CPU test tier).
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/platforms/cuda_arch_manifest.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

using vllm::platforms::CurrentPlatform;
using vllm::platforms::DeviceCapability;
using vllm::platforms::FindPlatformByName;
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
  // ENG-EXPERT-STREAM-DEVICE W0b (#1124): the CPU leg does NOT override this.
  //
  // That reads backwards at first glance — a CPU kernel obviously reads host
  // memory — and it is the right answer, because the predicate exists to name
  // the platforms that are NOT the CPU and can still do it. Every consumer asks
  // `is_cpu() || host_memory_is_device_addressable()`, so answering true here
  // would make the second term untestable on this tier: it would be satisfied
  // by the first for the one platform this build has.
  CHECK_FALSE(cpu.host_memory_is_device_addressable());
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

  // ISSUE #125 REGRESSION GUARD: `needs_weight_staging()` is NOT "am I a device".
  //
  // qwen3_5.cpp aliased host weight bytes into a device-tagged tensor whenever
  // `!needs_weight_staging()`, reasoning that only a non-device would skip
  // staging. That is true of CUDA and false of every OTHER device backend, which
  // all inherit the base `false` -- so Vulkan, Metal and XPU were handed HOST
  // pointers and died on the first native kernel. The correct predicate for
  // "may I alias host memory" is `is_cpu()`.
  //
  // This pins the two properties that make the distinction real, so the shape of
  // the bug cannot come back silently:
  //   * on CPU the two predicates AGREE (alias is correct either way), and
  //   * the base default for a non-CPU platform does NOT stage, which is exactly
  //     why the old predicate misclassified every non-CUDA device.
  CHECK(cpu.is_cpu());
  CHECK(cpu.is_cpu() == !cpu.needs_weight_staging());

  // supported_dtypes order (bf16 default fallback first).
  const std::vector<DType> expected{DType::kBF16, DType::kF16, DType::kF32};
  CHECK(cpu.supported_dtypes() == expected);

  // Unified host memory: no host-weight release, no device pool.
  const ResidencyPolicy policy = cpu.residency_policy();
  CHECK_FALSE(policy.release_host_weights_after_upload);
  CHECK_FALSE(policy.uses_device_memory_pool);
  CHECK(policy.device_pool_cap_bytes == 0);

  // Attention-backend priority (item 4): CPU mirrors cpu.py's single-backend
  // preference (CPU_ATTN), followed by FLASH_ATTN, whose NHD layout our CPU
  // paged-attn kernel shares. Since #1371 the first entry is registered, so the
  // walk stops there; the ORDER is what this case pins, and the registry-driven
  // selection is covered in test_attn_backend_registry.cpp.
  const std::vector<std::string> cpu_priority{"CPU_ATTN", "FLASH_ATTN"};
  CHECK(cpu.get_attn_backend_priority() == cpu_priority);
}

TEST_CASE("platform registry resolves canonical names without device-specific callers") {
  REQUIRE(FindPlatformByName("cpu") != nullptr);
  CHECK(FindPlatformByName("cpu")->device_type() == DeviceType::kCPU);
  CHECK(FindPlatformByName("not-a-platform") == nullptr);

  size_t count = 0;
  const DeviceType* priority = vllm::platforms::CurrentPlatformPriority(count);
  for (size_t i = 0; i < count; ++i) {
    if (!HasPlatform(priority[i])) continue;
    CAPTURE(vt::DeviceTypeName(priority[i]));
    CHECK(FindPlatformByName(vt::DeviceTypeName(priority[i])) ==
          &GetPlatform(priority[i]));
  }
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
      HasPlatform(DeviceType::kCUDA) || HasPlatform(DeviceType::kROCM) ||
      HasPlatform(DeviceType::kXPU) || HasPlatform(DeviceType::kVULKAN) ||
      HasPlatform(DeviceType::kMETAL);
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

// The ratchet for the ONE place a new platform is not additive (BACKEND-ROCM W0,
// found while adding kROCM: the platform registered correctly and would never
// have been selected, because CurrentPlatform() walks a hardcoded array and no
// -Werror=switch fires on an array). Two properties, both cheap, both on the
// CPU tier:
//   1. EVERY DeviceType appears in the walk. This is the one that would have
//      caught the omission.
//   2. CPU is LAST. The walk is accelerator-first by contract; a CPU entry that
//      drifted earlier would shadow every accelerator behind it.
TEST_CASE("every DeviceType is in the CurrentPlatform priority walk, CPU last") {
  size_t count = 0;
  const DeviceType* priority = vllm::platforms::CurrentPlatformPriority(count);
  REQUIRE(priority != nullptr);
  REQUIRE(count == vt::kNumDeviceTypes);

  for (size_t i = 0; i < vt::kNumDeviceTypes; ++i) {
    const DeviceType type = static_cast<DeviceType>(i);
    bool found = false;
    for (size_t j = 0; j < count; ++j) {
      if (priority[j] == type) found = true;
    }
    CHECK_MESSAGE(found, "DeviceType missing from the walk: ", vt::DeviceTypeName(type));
  }
  CHECK(priority[count - 1] == DeviceType::kCPU);
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
  // ISSUE #125, THE MISCLASSIFIED CLASS ITSELF. This stub is a NON-CPU platform
  // that does NOT stage -- exactly the combination Vulkan, Metal and XPU inherit,
  // and exactly the one `!needs_weight_staging()` mistook for "host memory, safe
  // to alias". Both predicates are false here, so they DIVERGE, and any code
  // using the staging flag to decide whether it may alias host bytes is wrong for
  // every platform shaped like this.
  //
  // (CUDA is the other half and cannot be asserted from a CPU build -- CudaPlatform
  // only compiles into a CUDA one -- but cuda.cpp overrides needs_weight_staging()
  // to true while is_cpu() stays false, so both predicates route it to UPLOAD and
  // the qwen3_5.cpp fix cannot change CUDA behaviour.)
  CHECK_FALSE(sm121.is_cpu());
  CHECK_FALSE(sm121.needs_weight_staging());
  CHECK(sm121.is_cpu() == sm121.needs_weight_staging());

  CHECK_FALSE(sm121.opaque_attention_op());
  CHECK_FALSE(sm121.support_static_graph_mode());
  // S7 additions default false on the base too (the CUDA ANSWER lives in the leg).
  CHECK_FALSE(sm121.needs_weight_staging());
  CHECK_FALSE(sm121.supports_fa2_attention());
  // W0b default. Being wrong here hands a HOST pointer to a DEVICE kernel, so
  // the base must be the refusing answer and a platform must opt in from a
  // probe. This stub does not override it and its device_type() is kCUDA, which
  // is exactly the shape that would be wrong if the answer lived at the call
  // site as a `device == kCUDA` test.
  CHECK_FALSE(sm121.host_memory_is_device_addressable());
}

namespace {
// ENG-EXPERT-STREAM-DEVICE W0b (#1124). A platform that STAGES its weights and
// can ALSO read host memory from a kernel — the GB10 shape, which has no other
// representative on a CPU tier. It exists to prove the two predicates are
// independent, which is the whole claim W0b rests on: if
// `host_memory_is_device_addressable()` were derivable from any predicate the
// seam already had, W0b would be a call-site test rather than a new one.
class FakeUnifiedAddressablePlatform final : public Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kCUDA; }
  vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
  DeviceCapability get_device_capability() const override { return {12, 1}; }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
  ResidencyPolicy residency_policy() const override { return {}; }
  bool needs_weight_staging() const override { return true; }
  bool is_integrated_gpu() const override { return true; }
  bool host_memory_is_device_addressable() const override { return true; }
};
}  // namespace

TEST_CASE("host_memory_is_device_addressable is independent of every neighbour") {
  FakeUnifiedAddressablePlatform gb10;
  Platform& cpu = GetPlatform(DeviceType::kCPU);

  // The value threads through the virtual at all: a platform that overrides it
  // answers its own value, not the base's.
  CHECK(gb10.host_memory_is_device_addressable());

  // AGAINST needs_weight_staging. Both TRUE here, and that combination is the
  // one the row exists for: the CUDA programming model still binds distinct
  // device pointers for every staged weight, AND a kernel can follow a host
  // pointer. A lane keyed on `!needs_weight_staging()` would never fire on this
  // platform; a lane keyed on this predicate does.
  CHECK(gb10.needs_weight_staging());
  CHECK(gb10.needs_weight_staging() == gb10.host_memory_is_device_addressable());

  // AGAINST is_cpu(). This is NOT a CPU, which is the entire point: the guard
  // being lifted in qwen3_5.cpp is `is_cpu()`, and it answers false here.
  CHECK_FALSE(gb10.is_cpu());
  CHECK(gb10.is_cpu() != gb10.host_memory_is_device_addressable());

  // AGAINST the CPU leg, in the other direction: the CPU IS host memory and
  // still answers false (see the CPU case for why), so the two predicates
  // DIVERGE on both platforms this build can construct. Neither is a rename of
  // the other, in either direction.
  CHECK(cpu.is_cpu());
  CHECK_FALSE(cpu.host_memory_is_device_addressable());
  CHECK(cpu.is_cpu() != cpu.host_memory_is_device_addressable());

  // AGAINST the backend seam it is often confused with.
  // `Backend::DeviceMemoryIsHostAddressable()` asks whether the HOST may
  // dereference a DEVICE allocation; this asks the reverse. The CPU backend
  // this stub borrows answers false to that one while the platform answers true
  // to this one, so they are not the same bit read twice.
  CHECK_FALSE(gb10.backend().DeviceMemoryIsHostAddressable());
  CHECK(gb10.host_memory_is_device_addressable() !=
        gb10.backend().DeviceMemoryIsHostAddressable());
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

  // W0b host_memory_is_device_addressable (#1124), probed at registration from
  // `cudaDevAttrPageableMemoryAccess AND cudaDevAttrIntegrated`.
  //
  // THE CONJUNCTION IS ASSERTED, not the value: a discrete CUDA card with HMM
  // reports pageable access and is NOT integrated, and this predicate must be
  // false there or the expert-stream lane would serve 6.95 GB per token over
  // PCIe with page migration. So the implication holds on EVERY CUDA device,
  // and it is what a discrete box would catch. The board-specific `true` below
  // is asserted only where the device also says it is integrated.
  CHECK((!cu.host_memory_is_device_addressable() || cu.is_integrated_gpu()));
  if (cu.is_integrated_gpu()) CHECK(cu.host_memory_is_device_addressable());
  // ...and the two assertions above are exactly what #1378 found insufficient.
  // Both attributes read 1 on the only CUDA box this project can reach, so either
  // term of the conjunction could be deleted and both assertions would still
  // pass -- measured, by the fresh review of #1377, which ran both mutations and
  // got GREEN twice. They now say only what THIS device reports. The RULE is
  // gated in the last case of this file, over all four attribute pairs, on a tier
  // with no CUDA device at all.

  // ...and it is NOT `needs_weight_staging()` inverted. On GB10 BOTH are true,
  // which is the combination that made this a new predicate rather than a
  // reuse: the CUDA path still stages every ordinary weight, and a kernel can
  // still follow a host pointer.
  CHECK(cu.needs_weight_staging());
  CHECK(cu.needs_weight_staging() == cu.host_memory_is_device_addressable());

  // S7 needs_weight_staging: unconditional true on CUDA — the CUDA path stages
  // host tensors into device-resident buffers even though GB10 is physically
  // unified. This is what the converted residency / merged-projection /
  // packed-decode gates now read; it must equal the old `device==kCUDA` (true) on
  // this device. On GB10 it AGREES with is_unified_memory() (both true — GB10 is
  // unified AND stages), which is exactly why is_unified_memory() is the wrong
  // predicate: the two DIVERGE on CPU (see the CPU-leg case — unified yet
  // NON-staging), so a staging gate keyed on is_unified_memory() would flip CPU.
  CHECK(cu.needs_weight_staging());

  // S7 supports_fa2_attention. NO LONGER unconditional (issue #1357): it asks
  // the generated compiled-arch manifest whether THIS device has FA2 SASS in
  // THIS build. It must still be true here, and that is the whole
  // no-change-on-the-gate-hardware claim made executable — a GB10 build requests
  // `121a`, the feature table's fa2 row contains `12.1a`, so the manifest
  // contains `121a` and this device reports capability 12,1: an exact match
  // including the arch-specific suffix. A red here on a GB10 CUDA build means
  // the manifest lost the arch, and the model would silently drop to the f32
  // graph-captured fallback.
  CHECK(cu.supports_fa2_attention());
  // And the manifest is genuinely consulted rather than ignored: the same
  // device is NOT served by a build that compiled some other architecture.
  CHECK_FALSE(vllm::platforms::ArchIsCompiled("80", cu.get_device_capability().major,
                                              cu.get_device_capability().minor));

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

// The CUDA platform's POLICY ASSEMBLY, on every host.
//
// `CudaPlatform::residency_policy()` lives in `src/vllm/platforms/cuda.cpp`, which
// compiles only in a CUDA build, so while the four assignments lived there nothing
// on a CPU-only host could reach them: #1123 recorded "delete the
// `device_memory_total_bytes` assignment" as an OWED mutation for exactly that
// reason. The assembly is now a free function in the platform header, `cuda.cpp`
// calls it with its probe, and this case pins every field it sets (#1136).
//
// What is still NOT pinned here, and is not claimed to be: the `cudaMemGetInfo`
// call and the constructor threading in `cuda.cpp`. Those need a CUDA build.
TEST_CASE("CudaResidencyPolicy assembles the CUDA policy, budget included") {
  using vllm::platforms::CudaResidencyPolicy;

  // GB10 as measured: cudaMemGetInfo total = 128452956160 (119.631 GiB).
  const size_t kGb10Total = 128452956160U;
  const ResidencyPolicy probed = CudaResidencyPolicy(kGb10Total);
  // The three fields that predate #1123, unchanged: the CUDA path frees the host
  // mirror after the Marlin build, pools device scratch, and leaves it uncapped.
  CHECK(probed.release_host_weights_after_upload);
  CHECK(probed.uses_device_memory_pool);
  CHECK(probed.device_pool_cap_bytes == 0);
  // The field #1123 added. This assertion is the one the owed mutation wanted:
  // deleting the assignment leaves 0, which the fit refusal reads as UNKNOWN, so
  // a checkpoint that cannot fit would load and die on the first forward again.
  CHECK(probed.device_memory_total_bytes == kGb10Total);
  // And it must be the ARGUMENT, not a constant: a second value moves it.
  CHECK(CudaResidencyPolicy(4096).device_memory_total_bytes == 4096);

  // A failed probe is 0 = UNKNOWN, and 0 must survive as 0 rather than being
  // substituted. The derived decisions are unaffected by the budget either way,
  // which is what makes this field additive.
  const ResidencyPolicy unknown = CudaResidencyPolicy(0);
  CHECK(unknown.device_memory_total_bytes == 0);
  CHECK(unknown.release_host_weights_after_upload);
  CHECK(ShouldReleaseHostWeights(unknown, /*marlin=*/true, /*env=*/true));
  CHECK(ShouldInterleaveLoadStream(unknown, /*marlin=*/true));
}

// --- ENG-EXPERT-STREAM-DEVICE W0b, issue #1378 --------------------------------
//
// The `PageableMemoryAccess AND Integrated` conjunction, gated over all four
// inputs on a tier with no CUDA device. Before this case the rule was asserted
// only through `CudaPlatform`, on the one box this project can reach, where BOTH
// attributes read 1 -- so the fresh review of #1377 deleted each term in turn
// (mutations M-B1 and M-B2) and the suite stayed GREEN both times. A conjunction
// that no reachable input can falsify is a comment.
//
// The DISCRETE-with-HMM row is the one that matters and the one no CUDA box here
// can produce: `pageable = 1, integrated = 0` is a card that reports pageable
// access and would serve every expert slice over PCIe with page migration. It
// must answer false, and deleting the `integrated` term is exactly what makes it
// answer true.
TEST_CASE("host_memory_is_device_addressable: the conjunction over all four attribute pairs") {
  using vllm::platforms::HostMemoryIsDeviceAddressableFromAttrs;

  // GB10 / any integrated, pageable-capable part: the ONE true row, and the one
  // the measured `dgx:gpu0` probe produced (both attributes 1).
  CHECK(HostMemoryIsDeviceAddressableFromAttrs(1, 1));

  // Discrete card with HMM. Deleting the `integrated` term flips this to true.
  CHECK_FALSE(HostMemoryIsDeviceAddressableFromAttrs(1, 0));

  // Integrated WITHOUT pageable access. `src/vllm/platforms/rocm.cpp` records
  // this as a real device class, and on it a host pointer faults. Deleting the
  // `pageable` term flips this to true.
  CHECK_FALSE(HostMemoryIsDeviceAddressableFromAttrs(0, 1));

  // Neither.
  CHECK_FALSE(HostMemoryIsDeviceAddressableFromAttrs(0, 0));

  // A FAILED `cudaDeviceGetAttribute` leaves its output at 0, which the probe
  // relies on being the conservative answer. Same two rows as above, said as the
  // failure it actually is rather than as a hypothetical device.
  CHECK_FALSE(HostMemoryIsDeviceAddressableFromAttrs(/*probe failed=*/0, 1));
  CHECK_FALSE(HostMemoryIsDeviceAddressableFromAttrs(1, /*probe failed=*/0));

  // The arguments are the raw `int`s the CUDA probe writes, not `bool`s, so any
  // non-zero value is the attribute being SET. A rule written with `== 1` would
  // pass every case above and fail here.
  CHECK(HostMemoryIsDeviceAddressableFromAttrs(2, 7));
}
