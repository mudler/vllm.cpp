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

  // True when host and device share one memory space (CPU, GB10, Apple).
  virtual bool UnifiedMemory() const = 0;

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
  virtual Event CreateEvent();
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

}  // namespace vt
