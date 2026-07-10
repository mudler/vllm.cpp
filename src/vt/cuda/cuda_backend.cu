// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "vt/backend.h"
#ifdef VLLM_CPP_TRITON
#include "vt/cuda/cuda_gdn_internal.h"
#endif

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
#ifdef VLLM_CPP_TRITON
    ReleaseGdnTritonScratch(device_, q.handle);
#endif
    Check(cudaStreamDestroy(AsStream(q)), "cudaStreamDestroy");
    q.handle = nullptr;
  }
  void Synchronize(Queue& q) override {
    Check(cudaStreamSynchronize(AsStream(q)), "cudaStreamSynchronize");
  }
  bool UnifiedMemory() const override { return unified_memory_; }

  // --- CUDA-graph capture/replay (M2.5, gate-#1 decode-launch unlock) ---------
  // Capture the ops issued on the queue's stream between Begin/EndCapture into a
  // replayable graph, then Replay it per decode token. This collapses the whole
  // decode step's thousands of host-side kernel-launch + memcpy API calls (the
  // measured 88%-of-wall host-API overhead) into a single cudaGraphLaunch.
  //
  // Capture contract (the caller must honour it, else capture aborts):
  //   * every op in the region runs ASYNC on THIS stream (no Synchronize, no
  //     default-stream work, no host<->device blocking copies);
  //   * NO cudaMalloc/cudaFree inside the region — the scratch pool must be
  //     pre-warmed so every allocation is a pool hit (a miss calls cudaMalloc,
  //     which is illegal during capture and aborts it);
  //   * the captured pointers must stay valid + fixed across replays (persistent
  //     buffers); only their CONTENTS change between replays (new token id /
  //     position / slot / block-table, written by an async copy on the stream
  //     BEFORE Replay, or captured as part of the graph).
  bool SupportsGraphCapture() const override { return true; }
  void BeginCapture(Queue& q) override {
    Check(cudaStreamBeginCapture(AsStream(q), cudaStreamCaptureModeThreadLocal),
          "cudaStreamBeginCapture");
  }
  void EndCapture(Queue& q) override {
    cudaGraph_t graph = nullptr;
    Check(cudaStreamEndCapture(AsStream(q), &graph), "cudaStreamEndCapture");
    if (exec_ != nullptr) {
      cudaGraphExecDestroy(exec_);
      exec_ = nullptr;
    }
    Check(cudaGraphInstantiate(&exec_, graph, 0), "cudaGraphInstantiate");
    cudaGraphDestroy(graph);
  }
  void Replay(Queue& q) override {
    Check(cudaGraphLaunch(exec_, AsStream(q)), "cudaGraphLaunch");
  }

  // Multi-graph handle API (batched decode graph): instantiate the just-captured
  // stream graph and hand the exec back as an opaque handle the driver owns +
  // selects per padded batch size. Unlike EndCapture, nothing is stored here.
  void* EndCaptureGraph(Queue& q) override {
    cudaGraph_t graph = nullptr;
    Check(cudaStreamEndCapture(AsStream(q), &graph), "cudaStreamEndCapture");
    cudaGraphExec_t exec = nullptr;
    Check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
    cudaGraphDestroy(graph);
    return reinterpret_cast<void*>(exec);
  }
  void ReplayGraph(Queue& q, void* graph) override {
    Check(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(graph), AsStream(q)),
          "cudaGraphLaunch");
  }
  void DestroyGraph(void* graph) override {
    if (graph != nullptr) cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(graph));
  }

 private:
  int device_ = 0;
  bool unified_memory_ = false;
  cudaGraphExec_t exec_ = nullptr;  // last instantiated captured graph
};

// Registers kCUDA during static init (registration must complete before
// main() per the backend.h contract). The probe must stay silent on machines
// that have the CUDA toolkit but no usable GPU: no throw, no print — it just
// leaves kCUDA unregistered and GetBackend(kCUDA) throws as usual.
struct Registrar {
  Registrar() noexcept {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return;
    // PageableMemoryAccess alone means the driver can service ordinary host
    // pointers through HMM/UVM. Discrete Blackwell reports it too; that does
    // not make pageable system RAM equivalent to device-local memory. Require
    // an integrated GPU as well before exposing the zero-copy contract used by
    // DeviceScratch and the persistent KV/GDN caches.
    int pageable = 0;
    int integrated = 0;
    if (cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0) != cudaSuccess) {
      return;
    }
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0) != cudaSuccess) return;
    static CudaBackend backend(0, pageable != 0 && integrated != 0);  // device 0 only for now
    RegisterBackend(DeviceType::kCUDA, &backend);
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
