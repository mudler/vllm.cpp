// Ported FROM vLLM csrc/libtorch_stable/torch_utils.h:20-82 and
// csrc/core/scalar_type.hpp:14-23,80-151 @ e24d1b24fe96. Workspace lifecycle
// mirrors torch::stable::new_empty's caching-allocation role and FlashInfer's
// caller-owned, first-use-zero workspace contract (flashinfer/decode.py:676-692).
// The tiny probe is W0 test scaffolding only; no production family dispatch is
// migrated by this translation unit.

#include "vt/cuda/cuda_dropin.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vt/backend.h"

namespace vt::cuda {

void Check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda drop-in: ") + operation + ": " +
                             cudaGetErrorString(error));
  }
}

DeviceGuard::DeviceGuard(Device device) {
  VT_CHECK(device.type == DeviceType::kCUDA && device.index >= 0,
           "CUDA device guard requires a non-negative CUDA device");
  Check(cudaGetDevice(&previous_), "cudaGetDevice");
  if (previous_ != device.index) {
    Check(cudaSetDevice(device.index), "cudaSetDevice");
    restore_ = true;
  }
}

DeviceGuard::~DeviceGuard() {
  if (restore_) (void)cudaSetDevice(previous_);
}

cudaStream_t Stream(const Queue& q) {
  VT_CHECK(q.device.type == DeviceType::kCUDA && q.device.index >= 0 && q.id != 0,
           "CUDA stream requires a live CUDA queue");
  return static_cast<cudaStream_t>(q.handle);
}

int DeviceCount() {
  int count = 0;
  Check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  return count;
}

namespace {

bool IsPowerOfTwo(size_t value) { return value != 0 && (value & (value - 1)) == 0; }

size_t HashCombine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

struct WorkspaceKeyHash {
  size_t operator()(const WorkspaceKey& key) const {
    size_t hash = static_cast<size_t>(key.device.type);
    hash = HashCombine(hash, static_cast<size_t>(key.device.index));
    hash = HashCombine(hash, static_cast<size_t>(key.queue_id));
    hash = HashCombine(hash, static_cast<size_t>(key.native_handle));
    hash = HashCombine(hash, static_cast<size_t>(key.op));
    return HashCombine(hash, static_cast<size_t>(key.slot));
  }
};

struct WorkspaceEntry {
  WorkspaceKey key;
  void* base = nullptr;
  void* aligned = nullptr;
  size_t capacity = 0;
  size_t alignment = 0;
  WorkspaceInit init = WorkspaceInit::kUninitialized;
  bool zeroed_once = false;
};

class WorkspacePool {
 public:
  ~WorkspacePool() {
    // Queue destruction is the normal ownership boundary. This best-effort
    // fallback prevents an intentional process-exit leak if a caller violates
    // that contract; CUDA teardown errors cannot escape a static destructor.
    for (auto& item : entries_) {
      WorkspaceEntry& entry = *item.second;
      if (entry.base == nullptr) continue;
      int previous = 0;
      if (cudaGetDevice(&previous) != cudaSuccess) continue;
      if (cudaSetDevice(entry.key.device.index) != cudaSuccess) continue;
      (void)cudaFree(entry.base);
      (void)cudaSetDevice(previous);
    }
  }

  WorkspaceLease Acquire(const Queue& q, OpId op, WorkspaceSlot slot,
                         size_t bytes, size_t alignment, WorkspaceInit init) {
    VT_CHECK(bytes > 0, "workspace size must be positive");
    VT_CHECK(IsPowerOfTwo(alignment), "workspace alignment must be a power of two");
    VT_CHECK(bytes <= std::numeric_limits<size_t>::max() - (alignment - 1),
             "workspace allocation size overflow");

    DeviceGuard guard(q.device);
    const cudaStream_t stream = Stream(q);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    Check(cudaStreamIsCapturing(stream, &capture), "cudaStreamIsCapturing");
    const bool capturing = capture != cudaStreamCaptureStatusNone;
    const WorkspaceKey key = MakeWorkspaceKey(q, op, slot);

    std::lock_guard<std::mutex> lock(mu_);
    auto found = entries_.find(key);
    WorkspaceEntry* entry = found == entries_.end() ? nullptr : found->second.get();
    const bool needs_growth =
        entry == nullptr || bytes > entry->capacity || alignment > entry->alignment;
    if (needs_growth) {
      VT_CHECK(!capturing, "workspace growth is forbidden during CUDA graph capture");
      if (entry != nullptr) {
        VT_CHECK(entry->init == init, "workspace init policy changed for an existing slot");
      }

      const size_t alloc_bytes = bytes + alignment - 1;
      void* new_base = nullptr;
      Check(cudaMallocAsync(&new_base, alloc_bytes, stream), "cudaMallocAsync(workspace)");
      const uintptr_t raw = reinterpret_cast<uintptr_t>(new_base);
      const uintptr_t aligned = (raw + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);

      if (entry == nullptr) {
        auto owned = std::make_unique<WorkspaceEntry>();
        owned->key = key;
        owned->init = init;
        entry = owned.get();
        entries_.emplace(key, std::move(owned));
      } else if (entry->base != nullptr) {
        Check(cudaFreeAsync(entry->base, stream), "cudaFreeAsync(workspace grow)");
      }
      entry->base = new_base;
      entry->aligned = reinterpret_cast<void*>(aligned);
      entry->capacity = bytes;
      entry->alignment = alignment;
      entry->zeroed_once = false;
    } else {
      VT_CHECK(entry->init == init, "workspace init policy changed for an existing slot");
    }

    if (init == WorkspaceInit::kZeroEachUse ||
        (init == WorkspaceInit::kZeroOnFirstUse && !entry->zeroed_once)) {
      Check(cudaMemsetAsync(entry->aligned, 0, entry->capacity, stream),
            "cudaMemsetAsync(workspace init)");
      entry->zeroed_once = true;
    }
    return WorkspaceLease{entry->aligned, entry->capacity, needs_growth};
  }

  void Release(const Queue& q) {
    DeviceGuard guard(q.device);
    Check(cudaStreamSynchronize(Stream(q)), "cudaStreamSynchronize(queue cleanup)");

    std::vector<std::unique_ptr<WorkspaceEntry>> removed;
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (auto it = entries_.begin(); it != entries_.end();) {
        const WorkspaceKey& key = it->first;
        if (key.device == q.device && key.queue_id == q.id &&
            key.native_handle == reinterpret_cast<uintptr_t>(q.handle)) {
          removed.push_back(std::move(it->second));
          it = entries_.erase(it);
        } else {
          ++it;
        }
      }
    }
    for (const auto& entry : removed) {
      if (entry->base != nullptr) Check(cudaFree(entry->base), "cudaFree(workspace cleanup)");
    }
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<WorkspaceKey, std::unique_ptr<WorkspaceEntry>, WorkspaceKeyHash> entries_;
};

WorkspacePool& Pool() {
  static WorkspacePool pool;
  return pool;
}

__global__ void SetScalarKernel(float* output, float value) { *output = value; }

void* AllocOnDevice(Device device, size_t bytes) {
  DeviceGuard guard(device);
  void* result = nullptr;
  Check(cudaMalloc(&result, bytes), "cudaMalloc(device resource)");
  return result;
}

void FreeOnDevice(Device device, void* pointer) {
  DeviceGuard guard(device);
  Check(cudaFree(pointer), "cudaFree(device resource)");
}

Queue CreateQueueOnDevice(Device device) {
  DeviceGuard guard(device);
  cudaStream_t stream = nullptr;
  Check(cudaStreamCreate(&stream), "cudaStreamCreate(device resource)");
  return Queue{device, stream};
}

void DestroyQueueOnDevice(Queue& q) {
  DeviceGuard guard(q.device);
  ReleaseQueueWorkspaces(q);
  if (q.handle != nullptr) Check(cudaStreamDestroy(Stream(q)), "cudaStreamDestroy(device resource)");
}

// Layer B: intentionally raw. Its signature contains no Tensor, Queue, torch,
// registry, or workspace-manager type; a future source lift can retain this
// shape while replacing only the Layer-A adapter below.
__global__ void DropinProbeRawKernel(float* output, const float* input, int64_t rows,
                                     int64_t cols, int64_t output_row_stride,
                                     int64_t output_col_stride,
                                     int64_t input_row_stride,
                                     int64_t input_col_stride,
                                     ScalarTypeId semantic_type, KernelLayout layout,
                                     uint32_t* workspace, const float* scalar) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = rows * cols;
  if (index >= count) return;
  const int64_t row = index / cols;
  const int64_t col = index - row * cols;
  output[row * output_row_stride + col * output_col_stride] =
      input[row * input_row_stride + col * input_col_stride] + *scalar;
  if (index == 0) {
    workspace[0] = static_cast<uint32_t>(semantic_type) ^
                   static_cast<uint32_t>(layout);
  }
}

void LaunchDropinProbeRaw(float* output, const float* input, int64_t rows,
                          int64_t cols, int64_t output_row_stride,
                          int64_t output_col_stride, int64_t input_row_stride,
                          int64_t input_col_stride, ScalarTypeId semantic_type,
                          KernelLayout layout, void* workspace, const float* scalar,
                          cudaStream_t stream) {
  const int64_t count = rows * cols;
  if (count == 0) return;
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((count + kThreads - 1) / kThreads,
                                                        65535));
  DropinProbeRawKernel<<<blocks, kThreads, 0, stream>>>(
      output, input, rows, cols, output_row_stride, output_col_stride,
      input_row_stride, input_col_stride, semantic_type, layout,
      static_cast<uint32_t*>(workspace), scalar);
  Check(cudaGetLastError(), "drop-in raw probe launch");
}

void DropinProbeKernelCuda(Queue& q, Tensor& output, const Tensor& input,
                           const DropinProbeArgs& args) {
  const KernelTensorDesc output_desc = Describe(output, args.scalar_type, args.layout);
  const KernelTensorDesc input_desc = Describe(input, args.scalar_type, args.layout);
  WorkspaceLease workspace = AcquireWorkspace(
      q, OpId::kDropinProbe, args.workspace_slot, args.workspace_bytes,
      args.workspace_alignment, args.workspace_init);
  DeviceScalarLease scalar =
      StageScalar(q, OpId::kDropinProbe, args.scalar_slot, args.scalar);
  LaunchDropinProbeRaw(static_cast<float*>(output_desc.data),
                       static_cast<const float*>(input_desc.data), input_desc.shape[0],
                       input_desc.shape[1], output_desc.stride[0],
                       output_desc.stride[1], input_desc.stride[0],
                       input_desc.stride[1], input_desc.scalar_type,
                       input_desc.layout, workspace.data, scalar.data, Stream(q));
}

const DeviceResourceOps kCudaResourceOps{
    &AllocOnDevice, &FreeOnDevice, &CreateQueueOnDevice, &DestroyQueueOnDevice};

struct Registrar {
  Registrar() {
    RegisterDeviceResourceOps(DeviceType::kCUDA, &kCudaResourceOps);
    RegisterTypedOp<DropinProbeFn>(
        OpId::kDropinProbe, DeviceType::kCUDA,
        static_cast<DropinProbeFn>(&DropinProbeKernelCuda));
  }
} registrar;

}  // namespace

WorkspaceLease AcquireWorkspace(const Queue& q, OpId op, WorkspaceSlot slot,
                                size_t bytes, size_t alignment,
                                WorkspaceInit init) {
  return Pool().Acquire(q, op, slot, bytes, alignment, init);
}

DeviceScalarLease StageScalar(const Queue& q, OpId op, WorkspaceSlot slot,
                              float value) {
  WorkspaceLease storage =
      AcquireWorkspace(q, op, slot, sizeof(float), alignof(float),
                       WorkspaceInit::kUninitialized);
  SetScalarKernel<<<1, 1, 0, Stream(q)>>>(static_cast<float*>(storage.data), value);
  Check(cudaGetLastError(), "device scalar launch");
  return DeviceScalarLease{static_cast<float*>(storage.data)};
}

void ReleaseQueueWorkspaces(const Queue& q) { Pool().Release(q); }

size_t WorkspaceEntryCountForTesting() { return Pool().Size(); }

}  // namespace vt::cuda
