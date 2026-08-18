/* ENG-CUDAGRAPH-BREAK W1 exit criterion.
 *
 * Question: does CUDA permit cuStreamEndCapture followed by cuStreamBeginCapture
 * on the SAME stream mid-forward, with EAGER work between them, under the
 * thread-local capture mode our CUDA backend uses (cuda_backend.cu:204-206)?
 *
 * Self-contained: binds the CUDA driver API through dlopen("libcuda.so"), so it
 * needs no CUDA toolkit and no headers. Uses only capturable memcpy/memset
 * stream operations, so it needs no nvcc-compiled kernel.
 *
 * The break function is genuinely host-dependent: it copies the mid buffer to
 * the host, computes v*2+3 on the CPU, and copies it back. That is the exact
 * class of operation a break point exists for.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUgraph;
typedef void *CUgraphExec;

static void *H;

static CUresult (*p_cuInit)(unsigned);
static CUresult (*p_cuDriverGetVersion)(int *);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDeviceGetName)(char *, int, CUdevice);
static CUresult (*p_cuCtxCreate)(CUcontext *, unsigned, CUdevice);
static CUresult (*p_cuStreamCreate)(CUstream *, unsigned);
static CUresult (*p_cuMemAlloc)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
static CUresult (*p_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
static CUresult (*p_cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
static CUresult (*p_cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
static CUresult (*p_cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
static CUresult (*p_cuMemsetD32Async)(CUdeviceptr, unsigned, size_t, CUstream);
static CUresult (*p_cuStreamBeginCapture)(CUstream, int);
static CUresult (*p_cuStreamEndCapture)(CUstream, CUgraph *);
static CUresult (*p_cuStreamIsCapturing)(CUstream, int *);
static CUresult (*p_cuGraphInstantiate)(CUgraphExec *, CUgraph, unsigned long long);
static CUresult (*p_cuGraphLaunch)(CUgraphExec, CUstream);
static CUresult (*p_cuGraphDestroy)(CUgraph);
static CUresult (*p_cuGraphExecDestroy)(CUgraphExec);
static CUresult (*p_cuStreamSynchronize)(CUstream);
static CUresult (*p_cuGetErrorName)(CUresult, const char **);


/* Bind by EXACT symbol name. libcuda exports several versions of the same
   entry point and they are NOT interchangeable: the legacy v1
   cuStreamBeginCapture takes no capture mode, the v1 memcpys are not
   capture-aware, and cuCtxCreate_v3 takes two extra parameters. Guessing a
   suffix binds a different function with the same name, which fails in a way
   that reads like a CUDA verdict rather than a probe bug. */
static void *sym(const char *n) {
  void *f = dlsym(H, n);
  if (!f) { printf("FAIL dlsym %s\n", n); }
  return f;
}

static const char *errname(CUresult r) {
  const char *s = "?";
  if (p_cuGetErrorName) p_cuGetErrorName(r, &s);
  return s ? s : "?";
}

static int fails = 0;
#define CK(expr)                                                               \
  do {                                                                         \
    CUresult _r = (expr);                                                      \
    if (_r != 0) {                                                             \
      printf("FAIL %s -> %d (%s)\n", #expr, _r, errname(_r));                  \
      fails++;                                                                 \
    }                                                                          \
  } while (0)

#define N 256
#define BYTES (N * (int)sizeof(int))

/* capture mode 1 == CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, what cuda_backend.cu uses */
#define MODE_THREAD_LOCAL 1

static CUstream S;
static CUdeviceptr dA, dB, dC;
static int hostbuf[N];

/* The break function: host-dependent, runs eagerly on the same stream. */
static void break_fn(void) {
  int i;
  CK(p_cuMemcpyDtoHAsync(hostbuf, dB, BYTES, S));
  CK(p_cuStreamSynchronize(S));
  for (i = 0; i < N; i++) hostbuf[i] = hostbuf[i] * 2 + 3;
  CK(p_cuMemcpyHtoDAsync(dB, hostbuf, BYTES, S));
  CK(p_cuStreamSynchronize(S));
}

int main(void) {
  int drv = 0, i, rep, cap = -1;
  char name[256];
  CUdevice dev;
  CUcontext ctx;
  CUgraph g0 = 0, g1 = 0, g2 = 0;
  CUgraphExec e0 = 0, e1 = 0, e2 = 0;
  int in[N], out[N];

  H = dlopen("libcuda.so.1", RTLD_NOW);
  if (!H) H = dlopen("libcuda.so", RTLD_NOW);
  if (!H) { printf("FAIL dlopen libcuda: %s\n", dlerror()); return 2; }

  p_cuInit = sym("cuInit");
  p_cuDriverGetVersion = sym("cuDriverGetVersion");
  p_cuDeviceGet = sym("cuDeviceGet");
  p_cuDeviceGetName = sym("cuDeviceGetName");
  p_cuCtxCreate = sym("cuCtxCreate_v2");
  p_cuStreamCreate = sym("cuStreamCreate");
  p_cuMemAlloc = sym("cuMemAlloc_v2");
  p_cuMemcpyHtoD = sym("cuMemcpyHtoD_v2");
  p_cuMemcpyDtoH = sym("cuMemcpyDtoH_v2");
  p_cuMemcpyDtoDAsync = sym("cuMemcpyDtoDAsync_v2");
  p_cuMemcpyDtoHAsync = sym("cuMemcpyDtoHAsync_v2");
  p_cuMemcpyHtoDAsync = sym("cuMemcpyHtoDAsync_v2");
  p_cuMemsetD32Async = sym("cuMemsetD32Async");
  /* v2 is the one that takes the capture MODE; v1 takes only the stream. */
  p_cuStreamBeginCapture = sym("cuStreamBeginCapture_v2");
  p_cuStreamEndCapture = sym("cuStreamEndCapture");
  p_cuStreamIsCapturing = sym("cuStreamIsCapturing");
  p_cuGraphInstantiate = sym("cuGraphInstantiateWithFlags");
  p_cuGraphLaunch = sym("cuGraphLaunch");
  p_cuGraphDestroy = sym("cuGraphDestroy");
  p_cuGraphExecDestroy = sym("cuGraphExecDestroy");
  p_cuStreamSynchronize = sym("cuStreamSynchronize");
  p_cuGetErrorName = sym("cuGetErrorName");

  if (!p_cuInit || !p_cuStreamBeginCapture || !p_cuStreamEndCapture ||
      !p_cuGraphInstantiate || !p_cuGraphLaunch || !p_cuMemcpyDtoDAsync) {
    printf("FAIL symbol bind: init=%p begin=%p end=%p inst=%p launch=%p d2d=%p\n",
           (void *)p_cuInit, (void *)p_cuStreamBeginCapture,
           (void *)p_cuStreamEndCapture, (void *)p_cuGraphInstantiate,
           (void *)p_cuGraphLaunch, (void *)p_cuMemcpyDtoDAsync);
    return 2;
  }

  CK(p_cuInit(0));
  if (fails) { printf("VERDICT: NO_GPU (cuInit failed)\n"); return 3; }
  CK(p_cuDriverGetVersion(&drv));
  CK(p_cuDeviceGet(&dev, 0));
  name[0] = 0;
  CK(p_cuDeviceGetName(name, sizeof name, dev));
  printf("driver_version=%d device=%s\n", drv, name);
  CK(p_cuCtxCreate(&ctx, 0, dev));
  CK(p_cuStreamCreate(&S, 1 /* NON_BLOCKING */));
  CK(p_cuMemAlloc(&dA, BYTES));
  CK(p_cuMemAlloc(&dB, BYTES));
  CK(p_cuMemAlloc(&dC, BYTES));
  if (fails) { printf("VERDICT: SETUP_FAILED\n"); return 3; }

  for (i = 0; i < N; i++) in[i] = i + 1;
  CK(p_cuMemcpyHtoD(dA, in, BYTES));

  /* ---- CAPTURE PHASE: segment 0, eager break, segment 1 on the SAME stream ---- */
  printf("\n[capture] segment 0: begin capture (thread-local mode)\n");
  CK(p_cuStreamBeginCapture(S, MODE_THREAD_LOCAL));
  CK(p_cuStreamIsCapturing(S, &cap));
  printf("[capture] is_capturing during segment 0 = %d (expect 1/2 = active)\n", cap);
  CK(p_cuMemcpyDtoDAsync(dB, dA, BYTES, S));
  CK(p_cuStreamEndCapture(S, &g0));
  printf("[capture] segment 0 ended, graph=%p\n", (void *)g0);
  CK(p_cuStreamIsCapturing(S, &cap));
  printf("[capture] is_capturing after end 0 = %d (expect 0 = none)\n", cap);
  CK(p_cuGraphInstantiate(&e0, g0, 0));

  printf("[capture] BREAK: eager host-dependent work on the SAME stream\n");
  break_fn();
  if (fails) { printf("VERDICT: EAGER_BETWEEN_SEGMENTS_FAILED\n"); return 3; }

  printf("[capture] segment 1: RE-BEGIN capture on the same stream  <-- exit criterion\n");
  {
    CUresult r = p_cuStreamBeginCapture(S, MODE_THREAD_LOCAL);
    if (r != 0) {
      printf("FAIL re-begin -> %d (%s)\n", r, errname(r));
      printf("VERDICT: REBEGIN_REFUSED\n");
      return 3;
    }
  }
  CK(p_cuStreamIsCapturing(S, &cap));
  printf("[capture] is_capturing during segment 1 = %d (expect active)\n", cap);
  CK(p_cuMemcpyDtoDAsync(dC, dB, BYTES, S));
  CK(p_cuStreamEndCapture(S, &g1));
  printf("[capture] segment 1 ended, graph=%p\n", (void *)g1);
  CK(p_cuGraphInstantiate(&e1, g1, 0));
  if (fails) { printf("VERDICT: SEGMENT1_FAILED\n"); return 3; }

  /* A third segment split by a BARE break (zero work between end and re-begin),
     the degenerate case of breakable_cuda_graph.py:370-374. */
  printf("[capture] bare break: end then immediately re-begin with NO work between\n");
  {
    CUresult r = p_cuStreamBeginCapture(S, MODE_THREAD_LOCAL);
    if (r != 0) {
      printf("FAIL bare re-begin -> %d (%s)\n", r, errname(r));
      printf("VERDICT: BARE_REBEGIN_REFUSED\n");
      return 3;
    }
  }
  CK(p_cuMemsetD32Async(dA, 0u, N, S)); /* harmless, keeps the graph non-empty */
  CK(p_cuStreamEndCapture(S, &g2));
  CK(p_cuGraphInstantiate(&e2, g2, 0));
  if (fails) { printf("VERDICT: BARE_SEGMENT_FAILED\n"); return 3; }

  /* ---- REPLAY PHASE: three replays with DIFFERENT inputs ---- */
  printf("\n[replay] three replays, fresh input each time\n");
  for (rep = 0; rep < 3; rep++) {
    int base = 100 * (rep + 1), bad = 0;
    for (i = 0; i < N; i++) in[i] = base + i;
    CK(p_cuMemcpyHtoD(dA, in, BYTES));

    CK(p_cuGraphLaunch(e0, S));   /* segment 0: B <- A */
    CK(p_cuStreamSynchronize(S));
    break_fn();                   /* break: B <- B*2+3 on the host */
    CK(p_cuGraphLaunch(e1, S));   /* segment 1: C <- B */
    CK(p_cuStreamSynchronize(S));

    CK(p_cuMemcpyDtoH(out, dC, BYTES));
    for (i = 0; i < N; i++) {
      int want = (base + i) * 2 + 3;
      if (out[i] != want) {
        if (bad < 3) printf("  MISMATCH rep=%d i=%d got=%d want=%d\n", rep, i, out[i], want);
        bad++;
      }
    }
    printf("  replay %d: base=%d out[0]=%d want=%d mismatches=%d\n", rep, base,
           out[0], base * 2 + 3, bad);
    if (bad) fails++;
  }

  if (p_cuGraphExecDestroy) { p_cuGraphExecDestroy(e0); p_cuGraphExecDestroy(e1); p_cuGraphExecDestroy(e2); }
  if (p_cuGraphDestroy) { p_cuGraphDestroy(g0); p_cuGraphDestroy(g1); p_cuGraphDestroy(g2); }

  printf("\nfails=%d\n", fails);
  printf("VERDICT: %s\n", fails ? "FAILED" : "REBEGIN_HOLDS");
  return fails ? 1 : 0;
}
