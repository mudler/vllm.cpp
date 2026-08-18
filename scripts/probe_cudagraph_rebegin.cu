// ENG-CUDAGRAPH-BREAK (#1163) W1 exit criterion, issue #1192.
//
// Question W0 left open: does CUDA permit cudaStreamEndCapture followed by
// cudaStreamBeginCapture on the SAME stream mid-forward, with EAGER work between
// them, on OUR stream configuration?
//
// "Our stream configuration" is mirrored exactly from src/vt/cuda/cuda_backend.cu:
//   BeginCapture     -> cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)  (:204)
//   EndCaptureGraph  -> cudaStreamEndCapture + cudaGraphInstantiate, stores nothing  (:225)
//   ReplayGraph      -> cudaGraphLaunch(exec, s)                                     (:233)
//   DestroyGraph     -> cudaGraphExecDestroy                                         (:288)
// The stream is a non-blocking stream, as vt queues are.
//
// The shape under test is the seam's, not a toy: segment, break, segment, break,
// segment. The break function is genuinely host-dependent (device -> host ->
// host arithmetic -> device), which is the class of operation a break point
// exists for, and it runs eagerly on the SAME stream that capture just left.
#include <cstdio>
#include <cuda_runtime.h>

#define N 1024

static int fails = 0;
#define CK(expr)                                                            \
  do {                                                                      \
    cudaError_t _e = (expr);                                                \
    if (_e != cudaSuccess) {                                                \
      std::printf("FAIL %s -> %d (%s)\n", #expr, (int)_e,                   \
                  cudaGetErrorString(_e));                                  \
      ++fails;                                                              \
    }                                                                       \
  } while (0)

__global__ void add_k(int* dst, const int* src, int add, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i] + add;
}

static cudaStream_t S;
static int *dA, *dB, *dC;
static int hostbuf[N];

// The break: host-dependent work, eager, on the same stream. Mirrors what a real
// break point does (a host readback that decides something, then a writeback).
static void break_fn(int mul) {
  CK(cudaMemcpyAsync(hostbuf, dB, N * sizeof(int), cudaMemcpyDeviceToHost, S));
  CK(cudaStreamSynchronize(S));
  for (int i = 0; i < N; ++i) hostbuf[i] = hostbuf[i] * mul;
  CK(cudaMemcpyAsync(dB, hostbuf, N * sizeof(int), cudaMemcpyHostToDevice, S));
  CK(cudaStreamSynchronize(S));
}

// EndCaptureGraph, byte-for-byte the cuda_backend.cu:225-232 body.
static cudaGraphExec_t EndCaptureGraph(cudaStream_t s) {
  cudaGraph_t g = nullptr;
  CK(cudaStreamEndCapture(s, &g));
  cudaGraphExec_t e = nullptr;
  CK(cudaGraphInstantiate(&e, g, 0));
  cudaGraphDestroy(g);
  return e;
}

int main() {
  cudaDeviceProp prop{};
  int dev = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  int rt = 0, drv = 0;
  cudaRuntimeGetVersion(&rt);
  cudaDriverGetVersion(&drv);
  std::printf("device=%s sm_%d%d runtime=%d driver=%d integrated=%d\n", prop.name,
              prop.major, prop.minor, rt, drv, prop.integrated);

  CK(cudaStreamCreateWithFlags(&S, cudaStreamNonBlocking));
  CK(cudaMalloc(&dA, N * sizeof(int)));
  CK(cudaMalloc(&dB, N * sizeof(int)));
  CK(cudaMalloc(&dC, N * sizeof(int)));
  if (fails) { std::printf("VERDICT: SETUP_FAILED\n"); return 3; }

  int in[N], out[N];
  for (int i = 0; i < N; ++i) in[i] = i + 1;
  CK(cudaMemcpy(dA, in, N * sizeof(int), cudaMemcpyHostToDevice));

  const int blocks = (N + 255) / 256;

  // ---------------- capture phase ----------------
  std::printf("\n[capture] segment 0: BeginCapture(thread-local)\n");
  CK(cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal));
  cudaStreamCaptureStatus st{};
  CK(cudaStreamIsCapturing(S, &st));
  std::printf("[capture] is_capturing in segment 0 = %d\n", (int)st);
  add_k<<<blocks, 256, 0, S>>>(dB, dA, 1, N);  // B = A + 1
  cudaGraphExec_t e0 = EndCaptureGraph(S);
  CK(cudaStreamIsCapturing(S, &st));
  std::printf("[capture] segment 0 ended, exec=%p, is_capturing=%d\n", (void*)e0, (int)st);
  if (fails) { std::printf("VERDICT: SEGMENT0_FAILED\n"); return 3; }

  std::printf("[capture] BREAK 0: eager host-dependent work on the SAME stream\n");
  break_fn(2);  // B = B * 2
  if (fails) { std::printf("VERDICT: EAGER_BETWEEN_SEGMENTS_FAILED\n"); return 3; }

  std::printf("[capture] segment 1: RE-BEGIN capture on the same stream  <-- EXIT CRITERION\n");
  {
    cudaError_t r = cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal);
    if (r != cudaSuccess) {
      std::printf("FAIL re-begin -> %d (%s)\nVERDICT: REBEGIN_REFUSED\n", (int)r,
                  cudaGetErrorString(r));
      return 3;
    }
  }
  add_k<<<blocks, 256, 0, S>>>(dC, dB, 3, N);  // C = B + 3
  cudaGraphExec_t e1 = EndCaptureGraph(S);
  std::printf("[capture] segment 1 ended, exec=%p\n", (void*)e1);
  if (fails) { std::printf("VERDICT: SEGMENT1_FAILED\n"); return 3; }

  // The BARE break: end then immediately re-begin with NO work in between, the
  // degenerate case of breakable_cuda_graph.py:370-374.
  std::printf("[capture] BREAK 1 (bare): re-begin with NO work between\n");
  {
    cudaError_t r = cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal);
    if (r != cudaSuccess) {
      std::printf("FAIL bare re-begin -> %d (%s)\nVERDICT: BARE_REBEGIN_REFUSED\n",
                  (int)r, cudaGetErrorString(r));
      return 3;
    }
  }
  add_k<<<blocks, 256, 0, S>>>(dC, dC, 10, N);  // C = C + 10
  cudaGraphExec_t e2 = EndCaptureGraph(S);
  std::printf("[capture] segment 2 ended, exec=%p (3 segments, 2 breaks)\n", (void*)e2);
  if (fails) { std::printf("VERDICT: BARE_SEGMENT_FAILED\n"); return 3; }

  // ---------------- replay phase: THREE replays, fresh input each ----------------
  std::printf("\n[replay] 3 replays, different input each time\n");
  for (int rep = 0; rep < 3; ++rep) {
    const int base = 100 * (rep + 1);
    for (int i = 0; i < N; ++i) in[i] = base + i;
    CK(cudaMemcpy(dA, in, N * sizeof(int), cudaMemcpyHostToDevice));

    CK(cudaGraphLaunch(e0, S));  // segment 0
    CK(cudaStreamSynchronize(S));
    break_fn(2);                 // break 0 (eager)
    CK(cudaGraphLaunch(e1, S));  // segment 1
    // break 1 is bare: nothing runs
    CK(cudaGraphLaunch(e2, S));  // segment 2
    CK(cudaStreamSynchronize(S));
    CK(cudaMemcpy(out, dC, N * sizeof(int), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < N; ++i) {
      const int want = ((base + i) + 1) * 2 + 3 + 10;
      if (out[i] != want) {
        if (bad < 3)
          std::printf("  MISMATCH rep=%d i=%d got=%d want=%d\n", rep, i, out[i], want);
        ++bad;
      }
    }
    std::printf("  replay %d: base=%d out[0]=%d want=%d mismatches=%d\n", rep, base,
                out[0], (base + 1) * 2 + 13, bad);
    if (bad) ++fails;
  }

  cudaGraphExecDestroy(e0);
  cudaGraphExecDestroy(e1);
  cudaGraphExecDestroy(e2);
  std::printf("\nfails=%d\nVERDICT: %s\n", fails, fails ? "FAILED" : "REBEGIN_HOLDS");
  return fails ? 1 : 0;
}
