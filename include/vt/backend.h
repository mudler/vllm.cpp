// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <cstddef>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt {

// Cross-stream event handle (CUDA event; no-op on synchronous backends).
// Opaque: created by Backend::CreateEvent, released with DestroyEvent. `device`
// records the owning backend so a holder can release it without extra state.
// On synchronous/unified backends the handle stays null and every event op is a
// no-op (all prior work on a queue has already completed by the time the host
// observes it). Mirrors the `torch.Event` in vllm/v1/worker/gpu/async_utils.py.
struct Event {
  Device device;
  void* handle = nullptr;
};

class Backend {
 public:
  virtual ~Backend() = default;

  // Returns memory aligned to at least 64 bytes; StepArena depends on this.
  virtual void* Alloc(size_t bytes) = 0;
  virtual void Free(void* p) = 0;
  virtual void Memset(Queue& q, void* p, int value, size_t bytes) = 0;
  // Same-device or host<->device transfer; on CPU this is memcpy.
  virtual void Copy(Queue& q, void* dst, const void* src, size_t bytes) = 0;
  virtual Queue CreateQueue() = 0;

  // Releases a queue obtained from CreateQueue. Default no-op suits backends
  // whose queues own no resources (CPU); CUDA destroys the underlying stream.
  // Callers must destroy every queue they create on backends that need it.
  virtual void DestroyQueue(Queue&) {}

  // Blocks until all work previously submitted to the queue has completed.
  // Default no-op suits synchronous backends (CPU); async backends (CUDA)
  // override with a stream sync.
  virtual void Synchronize(Queue&) {}

  // Drains any deferred submission WITHOUT a Queue in hand. Needed because the
  // portable CPU reference tier (op_provider.cpp) runs a HOST kernel directly
  // over device memory on a unified-memory backend, and must not observe bytes
  // a batched-but-uncommitted GPU submission has not written yet. Default no-op
  // suits every backend that submits eagerly; Metal overrides it (M3c-1).
  virtual void FlushPending() {}

  // True when host and device share one memory space (CPU, GB10, Apple).
  virtual bool UnifiedMemory() const = 0;

  // True when a pointer returned by Alloc() may be DEREFERENCED BY THE HOST
  // directly -- loaded, stored, memcpy'd -- with no map/unmap call and no
  // staging bounce.
  //
  // This is STRICTLY NARROWER than UnifiedMemory(), and the difference is the
  // whole reason it exists. CUDA on GB10 reports unified memory because host and
  // device address the same physical RAM, yet a plain `cudaMalloc` pointer is
  // still not host-dereferenceable. Vulkan here allocates every buffer
  // HOST_VISIBLE|HOST_COHERENT and keeps it persistently mapped, so its pointers
  // are ordinary host memory that the GPU also reads -- which is already what
  // this backend's Copy/Memset (plain memcpy/memset) and the portable CPU
  // reference tier depend on.
  //
  // MEASURED consequence (BACKEND-VULKAN-LOADMEM): where this is true, a weight
  // that has been uploaded needs NO host mirror, because the device allocation
  // IS a host buffer. Keeping one costs a second full copy of the model --
  // 16.392 GiB of process RSS for a 7.6 GiB Qwen3-4B, against 8.622 GiB of
  // Vulkan allocation -- and on a unified box that second copy comes out of the
  // same RAM the first one does.
  //
  // Default false: a backend must OPT IN, because being wrong here hands a
  // device pointer to a host memcpy and segfaults.
  virtual bool DeviceMemoryIsHostAddressable() const { return false; }

  // Optional device free/total VRAM probe (bytes). Default false = unknown.
  // ROCm overrides it with hipMemGetInfo (src/vt/rocm/rocm_backend.hip) so model
  // code can size LRU caches without including vendor headers (device-leakage).
  //
  // CUDA does NOT override it. This comment claimed "ROCm/CUDA" until #1123
  // measured what that costs: `Gemma4MoE`'s device-expert LRU is the seam's only
  // consumer, its `FreeBytes` returns false on an absent probe and `MakeRoom`
  // then refuses the device upload (both in gemma4_moe.cpp), so on EVERY CUDA
  // device that cache admits nothing and falls back to host H2D, silently.
  // Adding the override therefore WAKES a landed residency policy and needs its own
  // measurement; that is issue #1126, and this line says what is true until then.
  //
  // The load-time GGUF fit refusal deliberately does not read this seam: it is a
  // live free/total probe, and a load-time budget must not be a function of
  // contention. It carries its own `total` on
  // `vllm::platforms::ResidencyPolicy::device_memory_total_bytes` instead.
  virtual bool DeviceMemoryInfo(size_t* /*free_bytes*/, size_t* /*total_bytes*/) const {
    return false;
  }

  // --- Device compute capability (BACKEND-CUDA-ARCH-ADDITIVITY seam-gap #4) ---
  // The architecture the backend is actually running on, as the familiar
  // `(major, minor)` pair (GB10/sm_121 -> {12, 1}). Before this, the capability
  // existed ONLY on the engine-side Platform seam
  // (src/vllm/platforms/cuda.cpp:88-91) and the kernel layer could not see it,
  // so no host launcher could dispatch per architecture. `{0, 0}` means "no
  // meaningful compute capability" and is the default for backends where the
  // notion does not apply (CPU). Mirrors vLLM's
  // `Platform.get_device_capability()` (vllm/platforms/cuda.py @ e24d1b24),
  // which likewise exposes one cached probe to everything downstream.
  virtual int DeviceCapabilityMajor() const { return 0; }
  virtual int DeviceCapabilityMinor() const { return 0; }

  // --- Async-output primitives (ENG-ASYNC-SCHED W3, async_utils.py:12-70) ------
  // The sampler-output overlap needs (a) page-locked host memory a copy engine
  // can DMA into without a staging bounce and (b) cross-stream events so a copy
  // queue can wait the main queue, record completion, and the HOST can wait ONLY
  // that copy — never the main stream. These degenerate to synchronous host ops
  // on CPU/unified backends (the base implementations below); CUDA overrides
  // them with cudaHostAlloc + cudaEvent_t. Design mirrors torch's Event/pinned
  // usage in vllm/v1/worker/gpu/async_utils.py at pin e24d1b24.

  // Page-locked host allocation for a non-blocking D2H destination. Base
  // implementation returns ordinary host memory via Alloc (correct on unified
  // memory where the copy is already a memcpy); CUDA uses cudaHostAllocDefault.
  // Released with FreePinned. `bytes` may be 0 (returns a valid 1-byte block).
  virtual void* AllocPinned(size_t bytes);
  virtual void FreePinned(void* p);

  // Cross-stream event lifecycle. Base implementations are no-ops returning a
  // null-handle Event (synchronous backends have nothing to wait on).
  // `blocking` requests an event whose HOST wait (SynchronizeEvent) SLEEPS the
  // calling thread until completion instead of busy-spinning (CUDA:
  // cudaEventBlockingSync). Used by the decode-graph slot double-buffer
  // (VT_ASYNC_EXECUTOR): the reuse wait is nearly always already-signaled at
  // depth-2, so on the rare occasion the engine runs ahead the host should sleep,
  // not burn a core spinning. Ignored on synchronous backends (no handle).
  virtual Event CreateEvent(bool blocking = false);
  virtual void DestroyEvent(Event& e);
  // Record `e` on the queue's stream: it completes once all work submitted to
  // `q` up to this point has finished (async_utils.py copy_event.record).
  virtual void RecordEvent(Event& e, Queue& q);
  // Block the HOST until `e` has completed (async_utils.py
  // copy_event.synchronize — the ONLY blocking sync, and it waits the COPY
  // queue's event, so the main queue never blocks).
  virtual void SynchronizeEvent(Event& e);
  // NON-BLOCKING completion test: has `e` already completed? Mirrors
  // torch.Event.query, which vLLM's KV-offload worker polls per step instead of
  // synchronizing (vllm/v1/kv_offload/cpu/gpu_worker.py:395-404) — a blocking
  // check there would stall the engine on every transfer. The base
  // implementation returns true, which is correct on synchronous backends (CPU):
  // all prior work has completed by the time the host can observe the event.
  virtual bool QueryEvent(Event& e);
  // Make later work on `q` wait for `e` WITHOUT blocking the host — the ordering
  // primitive behind `copy_stream.wait_stream(main_stream)` (record an event on
  // the main queue, then QueueWaitEvent it on the copy queue).
  virtual void QueueWaitEvent(Queue& q, Event& e);

  // Does this backend support a SECONDARY compute stream for overlap? The MoE
  // shared-expert overlap (qwen3_5.cpp, ENG-MOE-SHARED-AUX) forks the shared MLP
  // onto an aux stream (RecordEvent/QueueWaitEvent on a second Queue) so it runs
  // concurrently with the routed grouped-GEMMs on the main stream — mirroring
  // maybe_execute_in_parallel (multi_stream_utils.py:47-54). Base false: a
  // single-stream backend runs the shared path serially (byte-identical output,
  // no overlap). CUDA overrides true. This is the capability the model's
  // `device==kCUDA && MoeSharedAuxStreamEnabled()` gate actually asked
  // (accelerator-seam S7), CUDA true / base false. Lives on Backend (src/vt, off
  // the DSR scan) so the model file stops naming a device at the aux-stream gate.
  virtual bool SupportsAuxStream() const { return false; }

  // Whether the host may validly read the SAMPLED TOKEN ID back between steps,
  // which is what the depth-2 async input-combine path requires
  // (gpu/runner.cpp: QueueSupportsAsyncInputCombine). Base false — a DISCRETE
  // non-CUDA GPU (e.g. ROCm gfx1201) is the hazard: the non-CUDA leg of
  // sample_tokens_async Synchronizes and then host-dereferences `dev_ids`, a
  // device Alloc that is garbage off-device (the "!"-token corruption on the lab
  // R9700, 2026-08-07), so those queues MUST stay synchronous. Overridden true
  // by CPU (host and device memory are the same allocation, so the read is
  // always valid) and by CUDA (the sampled id is device-mirrored,
  // async_device_mirror()). This is the capability the runner's
  // `device == kCUDA` gate actually asked; it lives on Backend (src/vt, off the
  // DSR scan) so the device-agnostic shared layer stops naming a device — the
  // same move SupportsAuxStream made for the aux-stream gate.
  // TODO(rocm): an INTEGRATED non-CUDA GPU reports UnifiedMemory()==true (see
  // row/ROCM-UNIFIED-MEMORY-B), where the alias IS valid; such a backend may
  // override this true once a HIP sampled-token mirror or a D2H copy of dev_ids
  // lands.
  virtual bool SupportsAsyncSampledTokenReadback() const { return false; }

  // Can this backend's causal-conv1d kernels read and write a COMPRESSED (bf16)
  // conv_state IN PLACE? vLLM's default mamba_cache_dtype="auto" makes the GDN
  // conv cache the model dtype (bf16 for the gate checkpoints), and a backend
  // that cannot address it must be handed an f32 working copy instead — which
  // costs the caller a gather before and a scatter after, per GDN layer, per
  // token. Overridden true by CUDA and by Vulkan, whose kernels widen the stored
  // element to f32, accumulate in f32 registers and round once on store; that is
  // the SAME single bf16->f32->bf16 round trip the gather/scatter arm performs,
  // so the two arms agree bit-for-bit.
  // This is the capability the `device == kCUDA` clause in CheckConvCommon
  // actually asked. It lives on Backend so the shared vt op layer stops naming a
  // device, exactly as SupportsAsyncSampledTokenReadback did for the runner.
  virtual bool SupportsCompressedConvState() const { return false; }

  // The GDN recurrent (SSM) state twin of the conv clause above: f16/bf16
  // [N,Hv,Dv,Dk] state addressed in place by the GdnPrefill/GdnDecode kernels,
  // read/written in f32 registers (vLLM's mamba_cache_dtype default is bf16).
  // CheckGdnCommon used to spell this as `device == kCUDA`; asking the backend
  // keeps the shared op layer device-agnostic.
  virtual bool SupportsCompressedGdnState() const { return false; }

  // Optional graph/command capture (CUDA Graphs / Metal ICB / Vulkan CB).
  virtual bool SupportsGraphCapture() const { return false; }
  virtual void BeginCapture(Queue& q);
  virtual void EndCapture(Queue& q);
  virtual void Replay(Queue& q);

  // Multi-graph handle API (M2.5 batched decode graph): a driver that captures a
  // SET of graphs (one per padded decode batch size) owns each instantiated
  // graph as an opaque handle and selects the right one per step. EndCaptureGraph
  // returns the just-captured graph (does NOT store it in the backend);
  // ReplayGraph launches a specific one; DestroyGraph frees it. (BeginCapture is
  // shared — capture is a stream-global mode.)
  virtual void* EndCaptureGraph(Queue& q);
  virtual void ReplayGraph(Queue& q, void* graph);
  virtual void DestroyGraph(void* graph);
};

// Device-explicit resource vocabulary for new kernel adapters. Existing
// Backend::{Alloc,Free,CreateQueue,DestroyQueue} methods remain temporary
// index-0 migration shims for production call sites that predate the drop-in
// ABI. New adapter code must use these free functions so device index and queue
// cleanup are never ambient.
struct DeviceResourceOps {
  using AllocFn = void* (*)(Device, size_t);
  using FreeFn = void (*)(Device, void*);
  using CreateQueueFn = Queue (*)(Device);
  using DestroyQueueFn = void (*)(Queue&);

  AllocFn alloc = nullptr;
  FreeFn free = nullptr;
  CreateQueueFn create_queue = nullptr;
  DestroyQueueFn destroy_queue = nullptr;
};

void* Alloc(Device device, size_t bytes);
void Free(Device device, void* p);
Queue CreateQueue(Device device);
void DestroyQueue(Queue& q);

Backend& GetBackend(DeviceType type);
// Non-throwing probe: the registered backend for `type`, or nullptr when none is
// registered. `GetBackend` throws for the unregistered case, which forces every
// "is this device present?" caller into a try/catch; this is the answer without
// one. Used by the portable reference tier (op_provider.cpp) to read a device's
// UnifiedMemory() property without assuming the device exists in this build.
Backend* TryGetBackend(DeviceType type);
// Threading contract: all registration must complete before main() runs
// (backends register via static initializers). After that, GetBackend is
// lock-free reads only; no synchronization is performed.
void RegisterBackend(DeviceType type, Backend* backend);
// Static-initializer contract matches RegisterBackend. A backend-neutral
// fallback serves index 0 when no device-specific table is registered.
void RegisterDeviceResourceOps(DeviceType type, const DeviceResourceOps* ops);

// --- Multi-device registry (BACKEND-DISTRIBUTED-TP W2) ------------------------
// The type-level API above resolves ONE backend per DeviceType, which is device
// index 0 by construction — the single-GPU engine. Tensor/pipeline parallel needs
// N discrete devices of one type (device 0..N-1) each addressable by its own
// `Backend*`/resource table, mirroring vLLM spawning one worker per local GPU
// (multiproc_executor.py:176 `for local_rank in range(local_world_size)`). These
// overloads register/resolve a backend for a SPECIFIC `Device{type,index}`.
//
// BYTE-NEUTRAL for the single-device path: `Device{type,0}` shares the same
// registry slot the type-level API writes/reads, so `GetBackend(type)` and
// `GetBackend(Device{type,0})` return the identical `Backend*`, and a build that
// only ever touches index 0 is unchanged. The maximum addressable index per type
// is `kMaxDevicesPerType`.
inline constexpr size_t kMaxDevicesPerType = 16;
Backend& GetBackend(Device device);
Backend* TryGetBackend(Device device);
void RegisterBackend(Device device, Backend* backend);
void RegisterDeviceResourceOps(Device device, const DeviceResourceOps* ops);

}  // namespace vt
