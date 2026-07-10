// Ported FROM vLLM csrc/libtorch_stable/torch_utils.h:20-82 and the raw
// launcher/workspace seams inventoried in .agents/specs/dropin-kernel-abi.md,
// vLLM e24d1b24fe96. Torch-free vt adapter vocabulary; no kernel policy.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "vt/ops.h"

namespace vt::cuda {

void Check(cudaError_t error, const char* operation);

class DeviceGuard {
 public:
  explicit DeviceGuard(Device device);
  ~DeviceGuard();

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_ = 0;
  bool restore_ = false;
};

cudaStream_t Stream(const Queue& q);
int DeviceCount();

struct WorkspaceLease {
  void* data = nullptr;
  size_t capacity = 0;
  bool grew = false;
};

struct DeviceScalarLease {
  float* data = nullptr;
};

WorkspaceLease AcquireWorkspace(const Queue& q, OpId op, WorkspaceSlot slot,
                                size_t bytes, size_t alignment,
                                WorkspaceInit init);
DeviceScalarLease StageScalar(const Queue& q, OpId op, WorkspaceSlot slot,
                              float value);

// Device-explicit DestroyQueue calls this before destroying the stream. It
// synchronizes, removes every key for the exact device/id/handle tuple, and
// frees the backing allocations. Exposed for lifecycle tests and future
// backend integration; normal adapter callers use vt::DestroyQueue.
void ReleaseQueueWorkspaces(const Queue& q);
size_t WorkspaceEntryCountForTesting();

}  // namespace vt::cuda
