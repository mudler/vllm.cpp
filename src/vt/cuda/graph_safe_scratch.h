// vllm.cpp original — CUDA-graph-safe grow-only device scratch bookkeeping.
//
// Several persistent, per-stream device scratch buffers grow on demand: the MoE
// grouped-GEMM index scratch (cuda_matmul_nvfp4.cu EnsureMoeScratch) and the
// cutlass NVFP4 / FP8 GEMM workspaces (cuda_matmul_nvfp4_cutlass.cu /
// cuda_matmul_fp8_cutlass.cu EnsureScratch). On a request larger than the current
// capacity they allocate a bigger block. The ORIGINAL strategy FREED the old block
// on growth (cudaFree / cudaFreeAsync). That is UNSAFE once CUDA-graph capture is
// live.
//
// The pure-decode forward is captured into a CUDA graph whose memset / kernel
// nodes BAKE the device pointer these Ensure* helpers returned at capture time.
// A LATER, LARGER forward — a bigger co-scheduled prefill or a larger decode batch,
// both only reachable at concurrency > 1 — grows the buffer and FREES the block the
// captured graph still references. The next replay of that graph then reads / writes
// freed memory → "an illegal memory access was encountered", surfaced at the next
// cudaEventSynchronize / cudaStreamSynchronize. This is exactly why single-stream
// (c1) never crashes (after its one prefill nothing ever grows the scratch again)
// while concurrency > 1 does, why it is independent of async scheduling and of the
// MoE WMMA path (every fp4/fp8 GEMM workspace has the same hazard), and why
// compute-sanitizer cannot reproduce it (its serialization keeps the freed block
// from being reused / released before the replay).
//
// FIX: on growth, RETIRE the old block (keep it resident for the process) instead
// of freeing it, so every pointer a captured graph baked stays valid for that
// graph's whole lifetime — the graph only ever needs the size it had at capture,
// which is <= the retired block's capacity. Growth is O(log(max/min)) — a handful
// of events over a process — so retired memory is bounded and negligible (it mirrors
// the pre-existing "blocks are never returned to the driver (leak at process exit)"
// discipline of the DevicePool and the cublasLt workspace). This header holds the
// portable, CPU-unit-testable retire bookkeeping; the actual cudaMalloc /
// cudaMallocAsync stays at each call site (only the free is replaced by a retire).
#ifndef VT_CUDA_GRAPH_SAFE_SCRATCH_H_
#define VT_CUDA_GRAPH_SAFE_SCRATCH_H_

#include <cstddef>
#include <mutex>
#include <vector>

namespace vt::cuda {

namespace detail {
inline std::vector<void*>& RetiredGraphScratchList() {
  static std::vector<void*> list;  // process-lifetime; captured-graph pointers stay valid
  return list;
}
inline std::mutex& RetiredGraphScratchMutex() {
  static std::mutex mu;
  return mu;
}
}  // namespace detail

// Retire (keep resident, NEVER free) a grow-only device scratch block whose device
// pointer may have been baked into a captured CUDA graph. Replaces the cudaFree /
// cudaFreeAsync that a grow-only Ensure* helper would otherwise call on the old
// block. A null pointer (nothing allocated yet) is ignored. Thread-safe; the caller
// need not hold its own scratch lock for this call.
inline void RetireGraphScratch(void* old_block) {
  if (old_block == nullptr) return;
  std::lock_guard<std::mutex> lock(detail::RetiredGraphScratchMutex());
  detail::RetiredGraphScratchList().push_back(old_block);
}

// Total number of blocks retired so far (diagnostics / unit tests). Monotonic for
// the process lifetime.
inline std::size_t RetiredGraphScratchCount() {
  std::lock_guard<std::mutex> lock(detail::RetiredGraphScratchMutex());
  return detail::RetiredGraphScratchList().size();
}

}  // namespace vt::cuda

#endif  // VT_CUDA_GRAPH_SAFE_SCRATCH_H_
