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
//   * `SupportsGraphCapture()` is TRUE as of BACKEND-ROCM W1 — see the hipGraph
//     capture/replay block below and .agents/specs/rocm-decode-graph.md. The
//     concern that kept it false (a capture that bakes addresses instead of
//     re-executing over persistent buffers is a SILENT correctness bug) is now
//     an assertion rather than a worry: the mutate-src-then-replay step in
//     tests/vt/test_rocm_backend.cpp fails if replay ever returns a snapshot.
//     What is NOT claimed here is the model-level path — Platform's
//     support_static_graph_mode() still gates that, and flipping it is W2.
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
#define VT_GRAPH_DEDUP_HIP 1
#include "vt/graph_dedup.h"
#include "vt/graph_dedup_runtime.h"
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

  // --- hipGraph capture/replay (BACKEND-ROCM W1) ------------------------------
  // The hipGraph mirror of cuda_backend.cu's capture block, call for call. Every
  // name below is a long-stable HIP runtime API with the same signature and
  // semantics as its CUDA counterpart — the property that lets upstream compile
  // csrc/ for both through hipify.
  //
  // Capture contract, identical to the CUDA one (the caller must honour it, else
  // capture aborts):
  //   * every op in the region runs ASYNC on THIS stream (no Synchronize, no
  //     null-stream work, no host<->device blocking copies);
  //   * NO hipMalloc/hipFree inside the region — the scratch pool must be
  //     pre-warmed so every allocation is a pool hit. hipBLASLt's workspace is
  //     the known hazard here: LtWorkspace() in rocm_matmul_hipblaslt.hip grows
  //     it lazily inside the GEMM path, so a shape first seen during capture
  //     allocates and invalidates the capture. It fails loudly at
  //     hipStreamEndCapture rather than corrupting, which is the acceptable
  //     direction (spec .agents/specs/rocm-decode-graph.md, D1);
  //   * captured pointers stay valid + FIXED across replays (persistent
  //     buffers); only their CONTENTS change between replays.
  //
  // hipStreamCaptureModeThreadLocal matches CUDA's choice deliberately: it makes
  // an illegal op during capture a loud failure on THIS thread rather than a
  // process-wide mode change.
  bool SupportsGraphCapture() const override { return true; }
  void BeginCapture(Queue& q) override {
    Check(hipStreamBeginCapture(AsStream(q), hipStreamCaptureModeThreadLocal),
          "hipStreamBeginCapture");
  }
  void EndCapture(Queue& q) override {
    hipGraph_t graph = nullptr;
    Check(hipStreamEndCapture(AsStream(q), &graph), "hipStreamEndCapture");
    if (exec_ != nullptr) {
      Check(hipGraphExecDestroy(exec_), "hipGraphExecDestroy");
      exec_ = nullptr;
    }
    Check(hipGraphInstantiate(&exec_, graph, nullptr, nullptr, 0),
          "hipGraphInstantiate");
    Check(hipGraphDestroy(graph), "hipGraphDestroy");
  }
  void Replay(Queue& q) override {
    Check(hipGraphLaunch(exec_, AsStream(q)), "hipGraphLaunch");
  }

  // Multi-graph handle API (batched decode graph): instantiate the just-captured
  // stream graph and hand the exec back as an opaque handle the caller owns and
  // selects per padded batch size. Unlike EndCapture, nothing is stored here.
  void* EndCaptureGraph(Queue& q) override {
    hipGraph_t graph = nullptr;
    Check(hipStreamEndCapture(AsStream(q), &graph), "hipStreamEndCapture");
    // ENG-CUDAGRAPH-DEDUP (#1162), the same wiring as the CUDA leg against the same
    // shared registry and the same shared ops table: with VT_CUDA_GRAPH_DEDUP set, the
    // RAW graph is retained and keyed by topology so compatible captures share one
    // executable. Default OFF, in which case this is the pre-dedup path byte for byte.
    if (vt::GraphDedupEnabled()) {
      if (dedup_ == nullptr) {
        dedup_ = std::make_unique<vt::GraphDedupRegistry>(vt::graph_dedup_rt::Ops());
      }
      return dedup_->Register(reinterpret_cast<void*>(graph));
    }
    hipGraphExec_t exec = nullptr;
    Check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
          "hipGraphInstantiate");
    Check(hipGraphDestroy(graph), "hipGraphDestroy");
    return reinterpret_cast<void*>(exec);
  }
  // NOT ported: the CUDA leg's VT_BENCH_PROFILE_CONTROL block, which drives
  // cudaProfilerStart/Stop around a chosen replay. A rocprofiler equivalent is
  // later work and deliberately out of W1's scope (spec §2).
  void ReplayGraph(Queue& q, void* graph) override {
    if (dedup_ != nullptr && dedup_->Owns(graph)) {
      dedup_->Replay(graph, reinterpret_cast<void*>(AsStream(q)));
      return;
    }
    Check(hipGraphLaunch(reinterpret_cast<hipGraphExec_t>(graph), AsStream(q)),
          "hipGraphLaunch");
  }
  // hipError_t itself — not hipGraphExecDestroy specifically — is
  // [[nodiscard]] at C++17+ ($ROCM_PATH/include/hip/hip_runtime_api.h:293-305,
  // __HIP_NODISCARD on the `typedef enum ... hipError_t` line; the function
  // declaration at line ~8321 carries no attribute of its own). CUDA's
  // cudaError_t has no such attribute, so the mirrored bare call diagnoses
  // here (-Wunused-value) and not there. This project's -Werror does NOT
  // reach it, though: vllm_cpp_set_warnings (cmake/CompilerWarnings.cmake)
  // gates -Werror on COMPILE_LANGUAGE CXX/OBJCXX/CUDA only, no HIP branch, so
  // a bare call here would warn, not fail the build (verified against the
  // actual compile_commands.json entry for this file, and by building a
  // scratch bare-call mutation through this project's own cmake target).
  // Checked anyway, matching Free/FreePinned above: a failing destroy is a
  // leak this backend would rather report than swallow, independent of what
  // the build flags happen to enforce.
  void DestroyGraph(void* graph) override {
    if (graph == nullptr) return;
    if (dedup_ != nullptr && dedup_->Owns(graph)) {
      dedup_->Destroy(graph);
      return;
    }
    Check(hipGraphExecDestroy(reinterpret_cast<hipGraphExec_t>(graph)),
          "hipGraphExecDestroy");
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

  bool DeviceMemoryInfo(size_t* free_bytes, size_t* total_bytes) const override {
    size_t free_b = 0, tot_b = 0;
    if (hipSetDevice(device_) != hipSuccess) return false;
    if (hipMemGetInfo(&free_b, &tot_b) != hipSuccess) return false;
    if (free_bytes) *free_bytes = free_b;
    if (total_bytes) *total_bytes = tot_b;
    return true;
  }

 private:
  int device_ = 0;
  bool unified_memory_ = false;
  bool managed_alloc_ = false;
  int major_ = 0;
  int minor_ = 0;
  // Single-graph slot for the EndCapture/Replay pair. The handle API
  // (EndCaptureGraph) stores nothing here — its caller owns the exec.
  hipGraphExec_t exec_ = nullptr;
  // ENG-CUDAGRAPH-DEDUP (#1162): built on the first capture, and only when
  // VT_CUDA_GRAPH_DEDUP asked for it, so an unset environment allocates nothing.
  std::unique_ptr<vt::GraphDedupRegistry> dedup_;
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

bool HostMemoryIsDeviceAddressable(int index) noexcept {
  // The CONJUNCTION, not `pageable_memory_access` alone: see the declaration.
  // It is deliberately NOT `unified_memory_`, which the registrar widens with
  // `managed_alloc` — a hipMallocManaged device reports unified because ITS OWN
  // allocations are migratable, which says nothing about a plain host pointer.
  const DeviceCaps caps = ProbeDevice(index);
  return caps.valid && caps.integrated && caps.pageable_memory_access;
}

bool ManagedAllocActive(int index) noexcept {
  // Answers from the SAME pure predicate over the SAME probe the registrar
  // used, so this cannot drift from what the registered backend decided at
  // static init: the attributes are immutable hardware properties.
  const DeviceCaps caps = ProbeDevice(index);
  return caps.valid && UseManagedAlloc(caps);
}

}  // namespace vt::rocm
