// ROCm leg of the vt::Backend seam (BACKEND-ROCM, W0 skeleton + W1 approach-(b)
// unified-memory fix).
//
// BUILD STATE, kept honest per issue #41: the W0 version of this TU was
// compiled clean AND ran its ctest gates on community boards — gfx1151 (Strix
// Halo, jimmykarily), gfx1103 (Radeon 780M, arch-btw), gfx1100 (4x 7900 XTX,
// VikashLoomba) and gfx1201 (2x R9700, bakon11). The approach-(b) DELTA in this
// revision is again written blind (no AMD GPU on the authoring machine) and
// owes the same community compile evidence before it may be called anything
// stronger. The one piece with a DECISION in it (the gfx capability parse)
// remains lifted out into include/vt/rocm/rocm_arch.h, which IS compiled and
// unit-tested on a CPU-only box (tests/vt/test_rocm_arch.cpp, 40 assertions).
//
// SHAPE: src/vt/cuda/cuda_backend.cu, virtual for virtual, including its
// unified-memory reasoning (see UnifiedMemory below) and its multi-device
// registrar. The HIP runtime API is a deliberate near-copy of the CUDA runtime
// API, which is why upstream vLLM compiles most of csrc/ for both through a
// hipify pass (cmake/hipify.py); this file is what that pass produces for the
// backend, written out by hand so a reader can diff the two.
//
// SCOPE / what is NOT here, stated plainly:
//   * `SupportsGraphCapture()` stays FALSE. hipGraph exists and is the eventual
//     mapping, but a graph that captures a wrong stream is a silent correctness
//     bug (see .agents/ on CUDA-graph capture baking stack addresses), so it is
//     not something to write blind.
//   * The async-output primitives (AllocPinned / events) inherit the vt::Backend
//     defaults. src/vt/backend.cpp documents those as already correct for
//     unified-memory backends; on a discrete AMD card they are correct but
//     synchronous, and hipHostMalloc / hipEvent* are the obvious later port.
//   * Exactly ONE op is registered (RmsNorm, see rocm_ops.hip). Everything else
//     resolves through the portable reference tier on a unified-memory part and
//     THROWS on a discrete one, which is the honest state, not a defect.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/rocm/rocm_arch.h"

namespace vt::rocm {
namespace {

void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: ") + what + ": " + hipGetErrorString(err));
  }
}

hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }

// One probe gathers the residency attributes and the capability together.
// Mirrors vt/cuda/cuda_device_caps.h in shape but is NOT cached (every consumer
// is init-time or test-time, never per-step), and stays file-local: the public
// probes below re-run it rather than exporting the struct.
struct DeviceCaps {
  bool valid = false;
  bool integrated = false;
  bool pageable_memory_access = false;
  bool managed_memory = false;
  bool concurrent_managed_access = false;
  int major = 0;
  int minor = 0;
  std::string gcn_arch;
};

DeviceCaps ProbeDevice(int index) noexcept {
  DeviceCaps caps;
  hipDeviceProp_t props{};
  if (hipGetDeviceProperties(&props, index) != hipSuccess) return caps;

  caps.gcn_arch = props.gcnArchName;
  // Prefer the string HIP itself reports over props.major/.minor: gcnArchName is
  // what upstream's _capability_from_gcn_arch parses (rocm.py:223), so both
  // layers answer from the same source. Fall back to the driver's own fields
  // when the string is unparseable rather than claiming (0, 0).
  if (const auto parsed = CapabilityFromGcnArch(caps.gcn_arch)) {
    caps.major = parsed->first;
    caps.minor = parsed->second;
  } else {
    caps.major = props.major;
    caps.minor = props.minor;
  }

  int integrated = 0;
  int pageable = 0;
  if (hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, index) != hipSuccess ||
      hipDeviceGetAttribute(&pageable, hipDeviceAttributePageableMemoryAccess, index) !=
          hipSuccess) {
    return caps;
  }
  caps.integrated = integrated != 0;
  caps.pageable_memory_access = pageable != 0;
  // The MANAGED-memory attributes decide the approach-(b) allocation branch
  // below. A failed probe of either must not unregister the device the way the
  // two probes above do: both default to 0, which disables the managed branch
  // and the unified claim — the safe direction (an APU then behaves like W0:
  // registered, discrete-shaped, no CPU fallback tier).
  int managed = 0;
  int concurrent = 0;
  if (hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, index) != hipSuccess) {
    managed = 0;
  }
  if (hipDeviceGetAttribute(&concurrent, hipDeviceAttributeConcurrentManagedAccess, index) !=
      hipSuccess) {
    concurrent = 0;
  }
  caps.managed_memory = managed != 0;
  caps.concurrent_managed_access = concurrent != 0;
  caps.valid = true;
  return caps;
}

// APPROACH (b) FROM ISSUE #41 F6, maintainer-ratified 2026-08-08: "On a device
// reporting hipDeviceAttributeIntegrated=1 (and ManagedMemory=1 +
// ConcurrentManagedAccess=1), Backend::Alloc in the ROCm backend uses
// hipMallocManaged instead of hipMalloc, and UnifiedMemory() returns true
// exactly then — host access becomes API-guaranteed rather than architecturally
// incidental."
//
// Why the W0 probe alone was not enough (measured on gfx1151, confirmed on
// gfx1103): XNACK-less RDNA3 APUs report Integrated=1 but
// PageableMemoryAccess=0, so the CUDA-shaped conjunction below read false even
// though host dereference of hipMalloc memory demonstrably worked — an
// architectural accident this project refuses to gate memory safety on. The
// managed route makes the aliasing a documented API property instead. All three
// attributes measured 1 on gfx1151 (Strix Halo) and gfx1103 (Radeon 780M).
//
// A discrete card reports Integrated=0, so this returns false — the branch is
// provably dead — on every dGPU, and the discrete path stays byte-identical to
// W0.
bool UseManagedAlloc(const DeviceCaps& caps) noexcept {
  return caps.integrated && caps.managed_memory && caps.concurrent_managed_access;
}

class RocmBackend final : public Backend {
 public:
  // Non-throwing by design: constructed during static init by the registrar,
  // which probes the device beforehand. Same contract as CudaBackend.
  RocmBackend(int device, bool unified_memory, bool managed_alloc, int major, int minor) noexcept
      : device_(device),
        unified_memory_(unified_memory),
        managed_alloc_(managed_alloc),
        major_(major),
        minor_(minor) {}

  int DeviceCapabilityMajor() const override { return major_; }
  int DeviceCapabilityMinor() const override { return minor_; }

  // Both branches satisfy the >= 64B contract Backend::Alloc owes StepArena:
  // hipMalloc aligns to at least 256 bytes as cudaMalloc does, and managed
  // allocations are at least page-granular.
  void* Alloc(size_t bytes) override {
    void* p = nullptr;
    if (managed_alloc_ && bytes > 0) {
      // Approach (b), issue #41 F6: on an integrated managed-capable device,
      // every Backend::Alloc block is a MANAGED allocation, so host access to
      // it is guaranteed by the API (ManagedMemory=1 +
      // ConcurrentManagedAccess=1 are preconditions of this branch), which is
      // exactly the property UnifiedMemory()==true advertises and the CPU
      // reference tier depends on. hipMemAttachGlobal (the API default) keeps
      // the block accessible from every stream and every device.
      //
      // The bytes==0 case stays on hipMalloc so the zero-size contract
      // (success, nullptr out) is byte-identical to W0 on every device rather
      // than resting on hipMallocManaged's less-documented zero-size behavior.
      Check(hipMallocManaged(&p, bytes, hipMemAttachGlobal), "hipMallocManaged");
    } else {
      Check(hipMalloc(&p, bytes), "hipMalloc");
    }
    return p;
  }
  // ONE free path for both Alloc branches, deliberately: the HIP runtime API
  // documents hipFree as the release call for hipMalloc AND hipMallocManaged
  // allocations (there is no hipFreeManaged; hipHostFree is for hipHostMalloc
  // only), mirroring cudaFree, whose documentation likewise covers
  // cudaMallocManaged memory. Certainty: HIGH — API-documented on both sides —
  // but stated rather than assumed because this file is written blind: if a
  // board ever reports hipErrorInvalidValue from this call on the managed
  // path, this assumption is the first suspect, and the fix would be tracking
  // the branch per-pointer, not a global switch.
  void Free(void* p) override { Check(hipFree(p), "hipFree"); }
  // Page-locked host memory so sampler D2H is a real async copy and the host
  // can read sampled token ids (default Backend::AllocPinned is Alloc →
  // hipMalloc on this backend, which is DEVICE memory — reading it as host is
  // UB and breaks multi-token greedy decode on discrete GPUs).
  void* AllocPinned(size_t bytes) override {
    void* p = nullptr;
    Check(hipHostMalloc(&p, bytes == 0 ? 1 : bytes, hipHostMallocDefault),
          "hipHostMalloc");
    return p;
  }
  void FreePinned(void* p) override {
    if (p != nullptr) Check(hipHostFree(p), "hipHostFree");
  }
  void Memset(Queue& q, void* p, int value, size_t bytes) override {
    Check(hipMemsetAsync(p, value, bytes, AsStream(q)), "hipMemsetAsync");
  }
  // hipMemcpyDefault infers direction from the pointer values, so one entry
  // point covers h2d, d2h and d2d — and pageable host pointers on an APU.
  void Copy(Queue& q, void* dst, const void* src, size_t bytes) override {
    Check(hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, AsStream(q)), "hipMemcpyAsync");
  }
  Queue CreateQueue() override {
    hipStream_t stream = nullptr;
    Check(hipStreamCreate(&stream), "hipStreamCreate");
    return Queue{Device{DeviceType::kROCM, device_}, stream};
  }
  void DestroyQueue(Queue& q) override {
    if (q.handle == nullptr) return;
    Check(hipStreamDestroy(AsStream(q)), "hipStreamDestroy");
    q.handle = nullptr;
  }
  void Synchronize(Queue& q) override {
    Check(hipStreamSynchronize(AsStream(q)), "hipStreamSynchronize");
  }

  // THE LOAD-BEARING BOOL. This is what decides whether the portable CPU
  // reference tier installs for kROCM (include/vt/op_provider.h:197-201): a CPU
  // kernel dereferences HOST pointers, so it is correct only where host and
  // device memory alias. On an APU (780M, Strix Halo) that makes every
  // unimplemented op fall back and a model run end to end; on a discrete card it
  // would be memory corruption, so the tier must stay off and unimplemented ops
  // must throw instead.
  //
  // TRUE on exactly two grounds, both API-anchored, computed by the registrar:
  //   1. The W0 conjunction, CudaBackend's own (cuda_backend.cu:295-303):
  //      Integrated AND PageableMemoryAccess. Pageable-memory-access ALONE is
  //      not enough, because a discrete part can report it while the driver
  //      services host pointers through HMM page migration — not the zero-copy
  //      aliasing this contract needs. Kept byte-for-byte for any part that
  //      reports both (NVIDIA-style integrated semantics).
  //   2. The approach-(b) managed branch (UseManagedAlloc above): every
  //      Backend::Alloc block on this device is hipMallocManaged memory, so
  //      host access is guaranteed by construction. This is the route
  //      XNACK-less RDNA3 APUs take, where ground 1 reads false (issue #41 F6).
  // Probed, never inferred from the gfx name.
  // The GDN conv-update and scan kernels registered for this backend carry
  // bf16/f16 state arms (f32 register math, boundary conversion), so both
  // compressed-state capabilities answer true. Validated on gfx1100 against
  // the f32 oracle before advertisement (the M2 e2e run exercises the bf16
  // mamba_cache_dtype path end to end).
  bool SupportsCompressedConvState() const override { return true; }
  bool SupportsCompressedGdnState() const override { return true; }

  bool UnifiedMemory() const override { return unified_memory_; }

 private:
  int device_ = 0;
  bool unified_memory_ = false;
  bool managed_alloc_ = false;
  int major_ = 0;
  int minor_ = 0;
};

// Registers every visible AMD GPU at its own Device{kROCM, i} slot, mirroring
// CudaBackend's multi-device registrar (cuda_backend.cu:304-323) — the 4x 7900
// XTX box on issue #41 is exactly why that is not a device-0 hardcode.
// `RegisterBackend(Device{kROCM,0}, ...)` writes the same slot
// `GetBackend(kROCM)` reads, so the single-GPU path is unaffected.
//
// Stays SILENT when there is no HIP runtime or no device: the platform registrar
// probes independently (static-init order across TUs is unspecified), and
// CurrentPlatform() must be able to fall through to CPU on a machine that merely
// happened to be built with -DVLLM_CPP_HIP=ON.
struct Registrar {
  Registrar() noexcept {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n <= 0) return;
    static std::vector<std::unique_ptr<RocmBackend>> backends;
    const int count = n < static_cast<int>(vt::kMaxDevicesPerType)
                          ? n
                          : static_cast<int>(vt::kMaxDevicesPerType);
    for (int i = 0; i < count; ++i) {
      const DeviceCaps caps = ProbeDevice(i);
      if (!caps.valid) continue;
      const bool managed_alloc = UseManagedAlloc(caps);
      const bool unified =
          managed_alloc || (caps.pageable_memory_access && caps.integrated);
      backends.push_back(
          std::make_unique<RocmBackend>(i, unified, managed_alloc, caps.major, caps.minor));
      RegisterBackend(Device{DeviceType::kROCM, i}, backends.back().get());
    }
  }
} registrar;

}  // namespace

// --- The HIP-free probes declared in include/vt/rocm/rocm_runtime.h ----------

bool DeviceAvailable() noexcept {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

std::string DeviceArchName(int index) noexcept {
  const DeviceCaps caps = ProbeDevice(index);
  return caps.gcn_arch;
}

bool IntegratedDevice(int index) noexcept {
  const DeviceCaps caps = ProbeDevice(index);
  return caps.valid && caps.integrated;
}

bool ManagedAllocActive(int index) noexcept {
  // Answers from the SAME pure predicate over the SAME probe the registrar
  // used, so this cannot drift from what the registered backend decided at
  // static init: the attributes are immutable hardware properties.
  const DeviceCaps caps = ProbeDevice(index);
  return caps.valid && UseManagedAlloc(caps);
}

}  // namespace vt::rocm
