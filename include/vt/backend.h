// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <cstddef>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt {

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

  // Optional cross-stream event primitives (async-decode side-stream readback,
  // VT_ASYNC_DECODE). An Event marks a point on one queue's stream so a SECOND
  // queue (the copy stream) can wait for it, and the host can later block on the
  // copy's completion — mirroring vLLM's AsyncGPUModelRunnerOutput copy stream +
  // torch.cuda.Event (gpu_model_runner.py:268-296). Default (CPU / synchronous
  // backends): CreateEvent returns nullptr and the rest are no-ops — a CPU queue
  // is already ordered and host-visible, so a readback is complete the instant
  // Copy returns. CUDA overrides these with cudaEvent_t. Ownership: the caller
  // frees every event it creates via DestroyEvent.
  virtual void* CreateEvent() { return nullptr; }
  virtual void DestroyEvent(void* /*event*/) {}
  // Record `event` at the current tail of q's stream.
  virtual void RecordEvent(Queue& /*q*/, void* /*event*/) {}
  // Make q's stream wait (device-side, non-blocking to the host) for `event`.
  virtual void StreamWaitEvent(Queue& /*q*/, void* /*event*/) {}
  // Block the HOST until `event` has completed.
  virtual void SyncEvent(void* /*event*/) {}
};

Backend& GetBackend(DeviceType type);
// Threading contract: all registration must complete before main() runs
// (backends register via static initializers). After that, GetBackend is
// lock-free reads only; no synchronization is performed.
void RegisterBackend(DeviceType type, Backend* backend);

}  // namespace vt
