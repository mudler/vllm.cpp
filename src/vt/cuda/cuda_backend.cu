// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "vt/backend.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

class CudaBackend final : public Backend {
 public:
  // Non-throwing by design: constructed during static init by the registrar,
  // which probes device presence and attributes itself beforehand.
  CudaBackend(int device, bool unified_memory) noexcept
      : device_(device), unified_memory_(unified_memory) {}

  // cudaMalloc returns allocations aligned to at least 256 bytes, which
  // satisfies the >=64B contract on Backend::Alloc (StepArena depends on it).
  void* Alloc(size_t bytes) override {
    void* p = nullptr;
    Check(cudaMalloc(&p, bytes), "cudaMalloc");
    return p;
  }
  void Free(void* p) override { Check(cudaFree(p), "cudaFree"); }
  void Memset(Queue& q, void* p, int value, size_t bytes) override {
    Check(cudaMemsetAsync(p, value, bytes, AsStream(q)), "cudaMemsetAsync");
  }
  // cudaMemcpyDefault infers the direction from the pointer values, so one
  // entry point covers h2d, d2h, and d2d (and pageable host pointers on
  // unified-memory devices such as GB10).
  void Copy(Queue& q, void* dst, const void* src, size_t bytes) override {
    Check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, AsStream(q)), "cudaMemcpyAsync");
  }
  Queue CreateQueue() override {
    cudaStream_t stream = nullptr;
    Check(cudaStreamCreate(&stream), "cudaStreamCreate");
    return Queue{Device{DeviceType::kCUDA, device_}, stream};
  }
  void DestroyQueue(Queue& q) override {
    if (q.handle == nullptr) return;
    Check(cudaStreamDestroy(AsStream(q)), "cudaStreamDestroy");
    q.handle = nullptr;
  }
  void Synchronize(Queue& q) override {
    Check(cudaStreamSynchronize(AsStream(q)), "cudaStreamSynchronize");
  }
  bool UnifiedMemory() const override { return unified_memory_; }
  // SupportsGraphCapture stays at the base-class false until M2.5.

 private:
  int device_ = 0;
  bool unified_memory_ = false;
};

// Registers kCUDA during static init (registration must complete before
// main() per the backend.h contract). The probe must stay silent on machines
// that have the CUDA toolkit but no usable GPU: no throw, no print — it just
// leaves kCUDA unregistered and GetBackend(kCUDA) throws as usual.
struct Registrar {
  Registrar() noexcept {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return;
    int pageable = 0;  // cudaDevAttrPageableMemoryAccess: host and device share one memory space
    if (cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0) != cudaSuccess) {
      return;
    }
    static CudaBackend backend(0, pageable != 0);  // device 0 only for now
    RegisterBackend(DeviceType::kCUDA, &backend);
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
