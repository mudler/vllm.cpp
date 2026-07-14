// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <cuda_runtime.h>
#ifdef VT_BENCH_PROFILE_CONTROL
#include <cuda_profiler_api.h>
#endif

#include <csignal>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "vt/backend.h"
#ifdef VT_BENCH_PROFILE_CONTROL
#include "vt/cuda/cuda_profiler_control.h"
#endif
#ifdef VLLM_CPP_FLASH_ATTN
#include "vt/cuda/cuda_flash_attn_fa2_internal.h"
#endif
#ifdef VLLM_CPP_TRITON
#include "vt/cuda/cuda_gdn_internal.h"
#endif

namespace vt::cuda {
namespace {

#ifdef VT_BENCH_PROFILE_CONTROL
volatile std::sig_atomic_t g_cuda_profile_arm_requested = 0;
uint32_t g_cuda_profile_target_replays = 0;
uint32_t g_cuda_profile_remaining_replays = 0;
uint32_t g_cuda_profile_target_batch = 0;
void* g_cuda_profile_pending_graph = nullptr;
void* g_cuda_profile_graph = nullptr;
uint64_t g_cuda_profile_prior_replays = 0;
uint32_t g_cuda_profile_real_batch = 0;
uint32_t g_cuda_profile_padded_batch = 0;
bool g_cuda_profile_eligible_pending = false;
bool g_cuda_profile_active = false;
bool g_cuda_profile_completed = false;

extern "C" void CudaProfilerArmSignalHandler(int signal) {
  if (signal == SIGUSR2) g_cuda_profile_arm_requested = 1;
}
#endif

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
#ifdef VLLM_CPP_FLASH_ATTN
    ReleaseFa2Scratch(device_, q.handle);
#endif
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
#ifdef VT_BENCH_PROFILE_CONTROL
    const bool eligible =
        g_cuda_profile_eligible_pending && graph == g_cuda_profile_pending_graph;
    g_cuda_profile_eligible_pending = false;
    g_cuda_profile_pending_graph = nullptr;
    if (g_cuda_profile_arm_requested != 0 && eligible) {
      g_cuda_profile_arm_requested = 0;
      if (g_cuda_profile_target_replays == 0 || g_cuda_profile_active ||
          g_cuda_profile_completed) {
        throw std::runtime_error(
            "vt cuda: invalid or duplicate graph replay profiler arm");
      }
      g_cuda_profile_remaining_replays = g_cuda_profile_target_replays;
      g_cuda_profile_graph = graph;
      g_cuda_profile_active = true;
    }
    if (g_cuda_profile_active &&
        (!eligible || graph != g_cuda_profile_graph)) {
      throw std::runtime_error(
          "vt cuda: graph replay profiler encountered an ineligible graph");
    }
    if (g_cuda_profile_active) {
      Check(cudaProfilerStart(), "cudaProfilerStart");
      if (g_cuda_profile_remaining_replays == g_cuda_profile_target_replays) {
        std::fprintf(stderr,
                     "[VT_CUDA_PROFILE] started target_replays=%u graph=%p "
                     "real_batch=%u padded_batch=%u prior_replays=%llu\n",
                     g_cuda_profile_target_replays, graph,
                     g_cuda_profile_real_batch, g_cuda_profile_padded_batch,
                     static_cast<unsigned long long>(g_cuda_profile_prior_replays));
      }
    }
#endif
    Check(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(graph), AsStream(q)),
          "cudaGraphLaunch");
#ifdef VT_BENCH_PROFILE_CONTROL
    if (g_cuda_profile_active) {
      if (g_cuda_profile_remaining_replays == 0) {
        throw std::runtime_error(
            "vt cuda: graph replay profiler counter underflow");
      }
      Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize profiler replay");
      Check(cudaProfilerStop(), "cudaProfilerStop");
      --g_cuda_profile_remaining_replays;
      if (g_cuda_profile_remaining_replays == 0) {
        g_cuda_profile_active = false;
        g_cuda_profile_completed = true;
        std::fprintf(stderr,
                     "[VT_CUDA_PROFILE] stopped captured_replays=%u graph=%p\n",
                     g_cuda_profile_target_replays, g_cuda_profile_graph);
      }
    }
#endif
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
    int pageable = 0;  // cudaDevAttrPageableMemoryAccess: host and device share one memory space
    if (cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0) != cudaSuccess) {
      return;
    }
    static CudaBackend backend(0, pageable != 0);  // device 0 only for now
    RegisterBackend(DeviceType::kCUDA, &backend);
  }
} registrar;

}  // namespace

#ifdef VT_BENCH_PROFILE_CONTROL
void ConfigureCudaGraphReplayProfiler(uint32_t replays, uint32_t batch_size) {
  if (replays == 0 || batch_size == 0) {
    throw std::invalid_argument(
        "CUDA graph replay profiler requires positive replay and batch counts");
  }
  if (g_cuda_profile_target_replays != 0) {
    throw std::logic_error("CUDA graph replay profiler is already configured");
  }
  g_cuda_profile_target_replays = replays;
  g_cuda_profile_target_batch = batch_size;
  if (std::signal(SIGUSR2, CudaProfilerArmSignalHandler) == SIG_ERR) {
    g_cuda_profile_target_replays = 0;
    g_cuda_profile_target_batch = 0;
    throw std::runtime_error("could not install CUDA profiler SIGUSR2 handler");
  }
}

void MarkCudaGraphReplayProfilerEligible(void* graph, uint32_t real_batch,
                                         uint32_t padded_batch,
                                         uint64_t prior_replays) {
  if (g_cuda_profile_completed || graph == nullptr ||
      real_batch != g_cuda_profile_target_batch ||
      padded_batch != g_cuda_profile_target_batch || prior_replays == 0) {
    return;
  }
  if (g_cuda_profile_eligible_pending) {
    throw std::logic_error(
        "CUDA graph replay profiler eligibility was not consumed");
  }
  g_cuda_profile_pending_graph = graph;
  g_cuda_profile_prior_replays = prior_replays;
  g_cuda_profile_real_batch = real_batch;
  g_cuda_profile_padded_batch = padded_batch;
  g_cuda_profile_eligible_pending = true;
}
#endif

}  // namespace vt::cuda
