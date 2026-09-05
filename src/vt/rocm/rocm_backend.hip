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
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
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
// A discrete card reports Integrated=0, so the managed branch is provably dead
// — on every dGPU — and the discrete path stays byte-identical to W0.
//
// THAT RATIFICATION HAS SINCE BEEN REVISITED, AND THE PARAGRAPH ABOVE IS
// HISTORY. It is kept because it states what the branch was for, and because
// its last sentence — "host dereference of hipMalloc memory demonstrably
// worked" — is the claim the alternative repair would have had to rest on.
// Read the next block for what ships.
//
// THE #2511 NARROWING, and the decision that used to live here.
//
// `gfx1151` reports PageableMemoryAccess = 0 and
// PageableMemoryAccessUsesHostPageTables = 0 (measured on the board,
// .agents/specs/rocm-gfx1151-q4k-hang.md), so it CANNOT take a recoverable page
// fault — and yet approach (b) routed every Backend::Alloc through
// hipMallocManaged, putting about 25 GiB of MIGRATABLE memory on it. A page
// migrated out from under a live queue is then a hard violation, which is why
// the eight recorded fault addresses are page-aligned and none lies inside any
// allocation the process made. Measured, one binary, one boot, the knob read at
// runtime, arms interleaved: hipMallocManaged 17 failures in 21 legs, plain
// hipMalloc 0 in 21, and the hipMalloc arm is the first thing that has ever
// completed a gate-sized run on this board.
//
// So the managed branch is now taken only where PageableMemoryAccess is 1, and
// UnifiedMemory() follows the ALLOCATOR rather than the capability. That
// withdraws the CPU reference tier on this class of part; see
// `HostAddressabilityNote` below, which is what a losing op is refused with.
//
// The DECISION is in include/vt/rocm/rocm_arch.h (`ResolveMemoryPolicy`), not
// here, for the reason that header already states about `CapabilityFromGcnArch`:
// it is the piece a wrong answer breaks silently, and there it is compiled and
// table-tested in the ordinary CPU build (tests/vt/test_rocm_arch.cpp) on a
// machine with no AMD GPU. This translation unit reads four attributes and
// calls it.
//
// VT_ROCM_MANAGED_ALLOC is tri-state: unset takes the narrowed default, `0`
// never takes the managed branch, and `1` takes it wherever the managed
// ATTRIBUTES allow — the pre-#2511 behaviour, restored whole (allocator AND
// unified claim), because a red-before arm that moves only half the
// configuration is not the configuration it claims to reproduce.
RocmManagedAllocOverride ManagedAllocOverride() {
  static const RocmManagedAllocOverride ov = [] {
    const char* e = std::getenv("VT_ROCM_MANAGED_ALLOC");
    return ParseManagedAllocOverride(e != nullptr ? std::string_view(e) : std::string_view());
  }();
  return ov;
}

// The probed attributes, in the shape the pure decision takes them.
RocmMemoryAttributes MemoryAttributes(const DeviceCaps& caps) noexcept {
  RocmMemoryAttributes a;
  a.integrated = caps.integrated;
  a.managed_memory = caps.managed_memory;
  a.concurrent_managed_access = caps.concurrent_managed_access;
  a.pageable_memory_access = caps.pageable_memory_access;
  return a;
}

RocmMemoryPolicy MemoryPolicy(const DeviceCaps& caps) {
  return ResolveMemoryPolicy(MemoryAttributes(caps), ManagedAllocOverride());
}

class RocmBackend final : public Backend {
 public:
  // Non-throwing by design: constructed during static init by the registrar,
  // which probes the device beforehand. Same contract as CudaBackend.
  RocmBackend(int device, const RocmMemoryPolicy& policy, int major, int minor) noexcept
      : device_(device),
        unified_memory_(policy.unified_memory),
        managed_alloc_(policy.managed_alloc),
        host_addressability_note_(policy.host_addressability_note),
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

  // Drains without a Queue in hand, for the portable CPU reference tier
  // (#2498). `GetOp` calls this before returning a reference-tier kernel
  // (op_provider.cpp:707-711), as does the decline path (762-765), because that
  // kernel is a HOST function that reads and writes DEVICE memory directly.
  //
  // The `{}` default in vt/backend.h documents itself as suiting "every backend
  // that submits eagerly". HIP submits eagerly and COMPLETES asynchronously:
  // CreateQueue above builds a genuine hipStreamCreate stream, so work is still
  // in flight when the host regains control. Inheriting the default therefore
  // let a host kernel read managed memory that device kernels were still
  // writing, with nothing ordering the two. Metal (metal_backend.mm) and Vulkan
  // (vulkan_backend.cpp) both override this; ROCm was the only backend
  // reporting DeviceMemoryIsHostAddressable() that did not.
  //
  // hipDeviceSynchronize rather than a stream sync BECAUSE there is no Queue
  // here: the tier's caller cannot say which stream produced the bytes it is
  // about to read, and this backend hands out a fresh stream per CreateQueue.
  // Waiting on the device waits on all of them.
  //
  // Cost is confined to the path that needs it. The call site is guarded by
  // `slot.ref_selected` (op_provider.cpp:707), so a run that reaches no
  // reference-tier op never calls this at all, and a discrete AMD board never
  // installs the tier in the first place.
  void FlushPending() override {
    Check(hipSetDevice(device_), "hipSetDevice");
    Check(hipDeviceSynchronize(), "hipDeviceSynchronize");
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
  // The sampled token id is device-mirrored (async_device_mirror(), the ROCm
  // port of the W3/W4 combine/scatter kernels in rocm_combine_tokens.hip), so
  // the between-steps host readback the depth-2 async input-combine needs is
  // valid. Mirrors the CUDA backend's override (cuda_backend.cu:229-230);
  // without this the ported kernels are dead code — runner_supports_async()
  // resolves false and async scheduling stays off (AGENTS.md "Nothing lands
  // dead").
  bool SupportsAsyncSampledTokenReadback() const override { return true; }
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
  // TRUE on exactly two grounds, both API-anchored, resolved by
  // `ResolveMemoryPolicy` (include/vt/rocm/rocm_arch.h):
  //   1. The W0 conjunction, CudaBackend's own (cuda_backend.cu:295-303):
  //      Integrated AND PageableMemoryAccess. Pageable-memory-access ALONE is
  //      not enough, because a discrete part can report it while the driver
  //      services host pointers through HMM page migration — not the zero-copy
  //      aliasing this contract needs. Kept byte-for-byte for any part that
  //      reports both (NVIDIA-style integrated semantics).
  //   2. The managed branch actually TAKEN: every Backend::Alloc block is then
  //      hipMallocManaged memory, so host access is guaranteed by construction.
  //
  // #2511 CHANGED WHICH PARTS REACH GROUND 2, and this is the sentence to read
  // if you are wondering where an APU's CPU reference tier went. Approach (b)
  // (#41 F6) took ground 2 on any integrated managed-capable device, which is
  // how XNACK-less RDNA3 APUs claimed unified while ground 1 read false. Those
  // are exactly the parts that cannot fault and recover, and the managed
  // allocation is what faults them. Ground 2 now requires PageableMemoryAccess
  // too, so under the shipped default it IMPLIES ground 1 and this bool is
  // simply the W0 conjunction. On gfx1151 and gfx1103 it is therefore FALSE,
  // the tier is withheld, and `HostAddressabilityNote()` below is what an op
  // that loses its fallback is refused with.
  // Probed, never inferred from the gfx name.
  // The GDN conv-update and scan kernels registered for this backend carry
  // bf16/f16 state arms (f32 register math, boundary conversion), so both
  // compressed-state capabilities answer true. Validated on gfx1100 against
  // the f32 oracle before advertisement (the M2 e2e run exercises the bf16
  // mamba_cache_dtype path end to end).
  bool SupportsCompressedConvState() const override { return true; }
  bool SupportsCompressedGdnState() const override { return true; }

  bool UnifiedMemory() const override { return unified_memory_; }

  // The SAME value, and deliberately not `managed_alloc_`. `ResolveMemoryPolicy`
  // sets `unified_memory = managed_alloc || (pageable_memory_access &&
  // integrated)`, and BOTH branches give host code a pointer it may
  // dereference: the managed branch by the hipMallocManaged API contract, the
  // other by the PageableMemoryAccess attribute on an integrated device. So
  // every allocation this backend hands out on a `unified_memory_` device is
  // host-addressable. Note the left operand is the branch TAKEN, not the
  // capability: reading it off the capability is precisely how the allocator
  // and the claim could disagree, which is the shape #2511 shipped for three
  // weeks.
  //
  // Answered explicitly because `DeviceMemoryIsHostAddressable()` defaults to
  // false and is what gates the portable CPU reference tier (#844, #1435). That
  // gate used to read `UnifiedMemory()`, which is a WIDER property — true on
  // CUDA/GB10, where a cudaMalloc pointer is still not host-dereferenceable.
  bool DeviceMemoryIsHostAddressable() const override { return unified_memory_; }

  // Why THIS device answers the two bools above false, when a reader might
  // reasonably have expected an APU to answer them true. Null on every device
  // where the answer is unremarkable — a discrete card, or a part that takes
  // the managed branch — so the generic refusal text stands alone there.
  //
  // `Backend`'s default is nullptr and the reference tier's refusal path
  // (src/vt/op_provider.cpp, ReferenceTierRefusalReason) appends this when it
  // is set. Without it, an op that used to fall back on Strix Halo now refuses
  // with a sentence that is true and useless: "this backend does not report its
  // device memory host-addressable". With it, the refusal names the attribute,
  // the issue, and the knob that puts the old behaviour back.
  const char* HostAddressabilityNote() const override {
    return host_addressability_note_;
  }

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
  const char* host_addressability_note_ = nullptr;
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
      backends.push_back(
          std::make_unique<RocmBackend>(i, MemoryPolicy(caps), caps.major, caps.minor));
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
  // It is deliberately NOT `unified_memory_`, which the managed branch can
  // widen — a hipMallocManaged device reports unified because ITS OWN
  // allocations are migratable, which says nothing about a plain host pointer.
  // Since #2511 the two coincide under the shipped default, and they separate
  // again the moment VT_ROCM_MANAGED_ALLOC=1 forces the managed branch back on,
  // so the distinction is still load-bearing rather than historical.
  const DeviceCaps caps = ProbeDevice(index);
  return caps.valid && caps.integrated && caps.pageable_memory_access;
}

bool ManagedAllocActive(int index) noexcept {
  // Answers from the SAME pure decision over the SAME probe the registrar used,
  // so this cannot drift from what the registered backend chose at static init:
  // the attributes are immutable hardware properties and the override is read
  // once into a function-local static. It reports the branch Alloc ACTUALLY
  // takes, which is the question a board owner is asking; #2511's narrowing
  // means the managed CAPABILITY and the managed BRANCH are no longer the same
  // answer, and reporting the capability here would have hidden exactly the
  // change this probe exists to surface.
  const DeviceCaps caps = ProbeDevice(index);
  return caps.valid && MemoryPolicy(caps).managed_alloc;
}

size_t DeviceMemoryTotalBytes(int index) noexcept {
  // Mirrors `Backend::DeviceMemoryInfo` (above) byte-for-byte, deliberately
  // NOT calling it: this is a HIP-free, backend-registration-independent
  // free function meant for the platform registrar (see the declaration),
  // where `GetBackend(kROCM)` may not have run yet.
  if (hipSetDevice(index) != hipSuccess) return 0;
  size_t free_b = 0, tot_b = 0;
  if (hipMemGetInfo(&free_b, &tot_b) != hipSuccess) return 0;
  return tot_b;
}

}  // namespace vt::rocm
