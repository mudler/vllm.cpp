// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#pragma once

#include <cstddef>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt {

class Backend {
 public:
  virtual ~Backend() = default;

  virtual void* Alloc(size_t bytes) = 0;
  virtual void Free(void* p) = 0;
  virtual void Memset(Queue& q, void* p, int value, size_t bytes) = 0;
  // Same-device or host<->device transfer; on CPU this is memcpy.
  virtual void Copy(Queue& q, void* dst, const void* src, size_t bytes) = 0;
  virtual Queue CreateQueue() = 0;

  // True when host and device share one memory space (CPU, GB10, Apple).
  virtual bool UnifiedMemory() const = 0;

  // Optional graph/command capture (CUDA Graphs / Metal ICB / Vulkan CB).
  virtual bool SupportsGraphCapture() const { return false; }
  virtual void BeginCapture(Queue& q);
  virtual void EndCapture(Queue& q);
  virtual void Replay(Queue& q);
};

Backend& GetBackend(DeviceType type);
void RegisterBackend(DeviceType type, Backend* backend);

}  // namespace vt
