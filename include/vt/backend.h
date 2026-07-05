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
};

Backend& GetBackend(DeviceType type);
// Threading contract: all registration must complete before main() runs
// (backends register via static initializers). After that, GetBackend is
// lock-free reads only; no synchronization is performed.
void RegisterBackend(DeviceType type, Backend* backend);

}  // namespace vt
