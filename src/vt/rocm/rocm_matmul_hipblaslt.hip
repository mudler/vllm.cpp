// ROCm dense GEMM via hipBLAS (hipblasGemmEx) + optional hipBLASLt.
// Primary path: hipblasGemmEx with the standard row-major TN trick used by
// many ROCm LLM stacks. Falls back is not needed — GemmEx is required.
//
// MatmulBT: out[M,N] = a[M,K] @ b[N,K]^T  (b = Linear weight row-major)
// Matmul:   out[M,N] = a[M,K] @ b[K,N]
// Required on ROCm 6.x for hipblasGemmEx overloads (#201/#132).
#ifndef HIPBLAS_V2
#define HIPBLAS_V2
#endif

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "vt/grow_only_stream_scratch.h"
#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"
#include "vt/rocm/rocm_getblas_dualslot.h"
#include "vt/rocm/rocm_runtime.h"
#include "vt/rocm/rocm_skinny_gemm_arch.h"

namespace vt::rocm {
namespace {

void CheckHip(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: matmul: ") + what + ": " +
                             hipGetErrorString(err));
  }
}

const char* BlasName(hipblasStatus_t st) {
  switch (st) {
    case HIPBLAS_STATUS_SUCCESS: return "SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR: return "HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM: return "INVALID_ENUM";
    default: return "UNKNOWN";
  }
}

void CheckBlas(hipblasStatus_t st, const char* what) {
  if (st != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("vt rocm: matmul: ") + what +
                             ": hipblas " + std::to_string(static_cast<int>(st)) +
                             " (" + BlasName(st) + ")");
  }
}

struct BlasCtx {
  hipblasHandle_t handle = nullptr;
};

// Per-thread handles: hipBLAS is not free-threaded; decode is single-threaded
// per engine worker. Avoid global mutex + SetStream on every GEMM (100s×/token).
// Dual-slot (#837): hop 0→1 must not hipblasDestroy GPU0 while GPU0 GEMMs may
// still be queued. Device 1 → slot 1; every other id shares slot 0.
struct HipBlasHooks {
  using handle_t = hipblasHandle_t;
  using stream_t = hipStream_t;
  static handle_t NullHandle() { return nullptr; }
  static stream_t NullStream() { return nullptr; }
  static bool IsNull(handle_t h) { return h == nullptr; }
  static bool StreamIsCapturing(stream_t s) { return vt::rocm::StreamIsCapturing(s); }
  static int GetDevice() {
    int cur = -1;
    if (hipGetDevice(&cur) != hipSuccess) return -1;
    return cur;
  }
  static void SetDevice(int d) { CheckHip(hipSetDevice(d), "hipSetDevice"); }
  static handle_t Create() {
    handle_t h = nullptr;
    CheckBlas(hipblasCreate(&h), "hipblasCreate");
    return h;
  }
  static void Destroy(handle_t h) { (void)hipblasDestroy(h); }
  static void SetStream(handle_t h, stream_t s) {
    CheckBlas(hipblasSetStream(h, s), "hipblasSetStream");
  }
};

BlasCtx GetBlas(int device, hipStream_t stream) {
  static thread_local GetBlasDualSlotEngine<HipBlasHooks> tls_slots;
  HipBlasHooks hooks;
  return BlasCtx{RocmProductGetBlasOn(tls_slots, device, stream, hooks)};
}

}  // namespace

hipblasHandle_t ProductGetBlasHandle(int device, hipStream_t stream) {
  // Same-TU call into file-local GetBlas (HipBlasHooks + tls_slots).
  return GetBlas(device, stream).handle;
}

bool ProductGetBlasStreamIsCapturing(hipStream_t stream) {
  // Exact production hook. Always-false mutation must turn the HIP probe RED.
  return HipBlasHooks::StreamIsCapturing(stream);
}

namespace {

std::string ComboName(const Tensor& a, const Tensor& b, const Tensor& out) {
  return std::string("(") + Name(a.dtype) + "," + Name(b.dtype) + ")->" + Name(out.dtype);
}

hipDataType ToBlasType(DType dt) {
  switch (dt) {
    case DType::kF32: return HIP_R_32F;
    case DType::kBF16: return HIP_R_16BF;
    case DType::kF16: return HIP_R_16F;
    default:
      throw std::runtime_error("vt rocm: matmul: unsupported dtype");
  }
}

// BF16 default: COMPUTE_32F (FAST_16BF returns NOT_SUPPORTED on gfx1201 for our BT shapes).
// Override: VT_ROCM_GEMM_COMPUTE=32f | 16bf | 16f
hipblasComputeType_t GemmCompute(DType dt) {
  if (dt != DType::kBF16 && dt != DType::kF16) return HIPBLAS_COMPUTE_32F;
  static const hipblasComputeType_t k = [] {
    if (const char* e = std::getenv("VT_ROCM_GEMM_COMPUTE")) {
      if (std::strcmp(e, "32f") == 0) return HIPBLAS_COMPUTE_32F;
      if (std::strcmp(e, "16f") == 0) return HIPBLAS_COMPUTE_16F;
      if (std::strcmp(e, "16bf") == 0) return HIPBLAS_COMPUTE_32F_FAST_16BF;
    }
    return HIPBLAS_COMPUTE_32F;
  }();
  return k;
}

// VT_ROCM_GEMV: 1=force custom M=1 BF16 GEMV, 0/unset=hipblas (faster on gfx1201 lab).
bool GemvEnabled() {
  static const bool on = [] {
    if (const char* e = std::getenv("VT_ROCM_GEMV")) return e[0] == '1';
    return false;  // hipblasGemmEx wins A/B 2026-08-09 (~52 vs ~40 t/s)
  }();
  return on;
}

// wvSplitK skinny GEMM (rocm_skinny_gemm.hip, #487): the split-K/LDS-staged
// decode path that beats the 128x128-tile rocBLAS GEMM at M<=4. Default ON for
// decode-skinny shapes; VT_ROCM_SKINNY=0 restores the BLAS path for A/B.
bool SkinnyGemmEnabled() {
  static const bool on = [] {
    if (const char* e = std::getenv("VT_ROCM_SKINNY")) return e[0] != '0';
    return true;
  }();
  return on;
}

// y[n] = alpha * dot(x[0:K], W[n,0:K]) + beta * y[n]
// grid = N — one block per output row; x cached in LDS; block-reduce over K.
__global__ void Bf16GemvBTRowKernel(__hip_bfloat16* __restrict__ y,
                                    const __hip_bfloat16* __restrict__ x,
                                    const __hip_bfloat16* __restrict__ W, int N, int K,
                                    float alpha, float beta) {
  const int n = static_cast<int>(blockIdx.x);
  if (n >= N) return;
  extern __shared__ float smem[];
  float* x_cache = smem;
  const int tid = static_cast<int>(threadIdx.x);
  constexpr int kBlock = 256;
  for (int k = tid; k < K; k += kBlock) x_cache[k] = __bfloat162float(x[k]);
  __syncthreads();

  const __hip_bfloat16* wrow = W + static_cast<int64_t>(n) * static_cast<int64_t>(K);
  float acc = 0.f;
  const int K16 = K & ~15;
  for (int k = tid * 16; k < K16; k += kBlock * 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) acc += x_cache[k + t] * __bfloat162float(wrow[k + t]);
  }
  for (int k = K16 + tid; k < K; k += kBlock) acc += x_cache[k] * __bfloat162float(wrow[k]);

  __shared__ float red[256];
  red[tid] = acc;
  __syncthreads();
#pragma unroll
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) {
    float v = alpha * red[0];
    if (beta != 0.f) v += beta * __bfloat162float(y[n]);
    y[n] = __float2bfloat16(v);
  }
}

// Fallback when K*f32 LDS too big: no x cache, threads stride K from global.
__global__ void Bf16GemvBTRowNoLdsKernel(__hip_bfloat16* __restrict__ y,
                                         const __hip_bfloat16* __restrict__ x,
                                         const __hip_bfloat16* __restrict__ W, int N, int K,
                                         float alpha, float beta) {
  const int n = static_cast<int>(blockIdx.x);
  if (n >= N) return;
  const int tid = static_cast<int>(threadIdx.x);
  constexpr int kBlock = 256;
  const __hip_bfloat16* wrow = W + static_cast<int64_t>(n) * static_cast<int64_t>(K);
  float acc = 0.f;
  for (int k = tid; k < K; k += kBlock)
    acc += __bfloat162float(x[k]) * __bfloat162float(wrow[k]);
  __shared__ float red[256];
  red[tid] = acc;
  __syncthreads();
#pragma unroll
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) {
    float v = alpha * red[0];
    if (beta != 0.f) v += beta * __bfloat162float(y[n]);
    y[n] = __float2bfloat16(v);
  }
}

void Bf16GemvBT(hipStream_t s, void* out, const void* a, const void* b, int N, int K,
                float alpha, float beta) {
  constexpr int kBlock = 256;
  if (K <= 0 || N <= 0) return;
  const size_t shmem = static_cast<size_t>(K) * sizeof(float);
  if (shmem <= 48 * 1024) {
    Bf16GemvBTRowKernel<<<N, kBlock, shmem, s>>>(
        static_cast<__hip_bfloat16*>(out), static_cast<const __hip_bfloat16*>(a),
        static_cast<const __hip_bfloat16*>(b), N, K, alpha, beta);
  } else {
    Bf16GemvBTRowNoLdsKernel<<<N, kBlock, 0, s>>>(
        static_cast<__hip_bfloat16*>(out), static_cast<const __hip_bfloat16*>(a),
        static_cast<const __hip_bfloat16*>(b), N, K, alpha, beta);
  }
}

// VT_ROCM_HIPBLASLT=1 enables. Default OFF — heuristic path aborted on gfx1201 in lab.
bool LtEnabled() {
  static const bool on = [] {
    if (const char* e = std::getenv("VT_ROCM_HIPBLASLT")) return e[0] == '1';
    return false;
  }();
  return on;
}

struct LtHandle {
  hipblasLtHandle_t h = nullptr;
  LtHandle() {
    if (hipblasLtCreate(&h) != HIPBLAS_STATUS_SUCCESS) h = nullptr;
  }
  ~LtHandle() {
    if (h) (void)hipblasLtDestroy(h);
  }
};

LtHandle& GetLt() {
  static thread_local LtHandle lt;
  return lt;
}

struct LtAlgoCache {
  hipblasLtMatmulAlgo_t algo{};
  size_t workspace = 0;
  bool ok = false;
};

// key: M,N,K,alpha_beta_flag (0=ab1.0/0.0, 1=general)
using LtKey = std::tuple<int, int, int, int>;
struct LtKeyHash {
  size_t operator()(const LtKey& k) const {
    auto [a, b, c, d] = k;
    return (static_cast<size_t>(a) * 1315423911u) ^ (static_cast<size_t>(b) * 2654435761u) ^
           (static_cast<size_t>(c) * 97531u) ^ static_cast<size_t>(d);
  }
};

std::unordered_map<LtKey, LtAlgoCache, LtKeyHash>& LtCache() {
  static thread_local std::unordered_map<LtKey, LtAlgoCache, LtKeyHash> m;
  return m;
}

// Grow-only hipBLASLt workspace, one entry per STREAM, sharing the bookkeeping
// with the activation-quant scratch (src/vt/grow_only_stream_scratch.h).
//
// It was `thread_local` with a hipFree on growth, and that is two defects
// (#2712). The workspace is handed to a STREAM-ORDERED hipblasLtMatmul below, so
// the hipFree could return a block a queued matmul still reads, and a per-call
// free is illegal under hipGraph stream capture besides. The block is now
// RETIRED instead: kept resident for the process, exactly as the sibling pool
// does and for the same reason.
//
// Keyed by stream, not by thread, because one host thread submits to two streams
// here: the Gemma4 prefill peer path issues on the expert device's queue
// (src/vt/rocm/rocm_gemma4_experts.hip:1147,1149) while the compute queue is
// live. A per-thread buffer is therefore also one buffer across a two-device
// hop, which is the hazard GetBlas above was given an explicit dual slot for.
//
// hipMalloc, not hipMallocAsync. The sibling can be stream-ordered because its
// consumers are this project's own kernels on that stream; hipBLASLt's workspace
// under graph capture is behaviour this row could not exercise, since the whole
// Lt arm is default OFF (LtEnabled above). Removing the free is the part that is
// verifiable, so it is the part that changed.
vt::GrowOnlyStreamScratch<hipStream_t>& LtWorkspacePool() {
  static vt::GrowOnlyStreamScratch<hipStream_t> pool;
  return pool;
}

void* LtWorkspace(size_t need, hipStream_t stream) {
  if (need == 0) return nullptr;
  return LtWorkspacePool().Ensure(stream, need, [](size_t bytes) -> void* {
    void* block = nullptr;
    // nullptr is the pool's "could not allocate": it publishes nothing, so the
    // caller's existing `if (!ws) return false` contract is unchanged.
    if (hipMalloc(&block, bytes) != hipSuccess) return nullptr;
    return block;
  });
}

// Grow-only A/B/C pointer tables for hipblasGemmBatchedEx, one entry per DEVICE,
// on the same bookkeeping as the workspace above (src/vt/grow_only_stream_scratch.h).
//
// This is the THIRD grow-only pool in this file, and #2837 filed it as sharing
// only the LIFETIME half of #2712's defect. Reading it, it shares the race half
// too. The shipped code held its lock across the grow and then read `tab->A`,
// `tab->B` and `tab->C` back out of the shared map entry for three
// hipMemcpyAsync calls with the lock already released, so a second caller on the
// same device that grew the tables freed the block the first was about to copy
// into. GrowOnlyStreamTriple returns the three pointers BY VALUE, so there is no
// entry left to re-read, and it retires the old block instead of freeing it --
// which is also what makes a growth legal under hipGraph stream capture, the
// contract rocm_backend.hip:322-330 already names.
vt::GrowOnlyStreamTriple<int>& PtrTablePool() {
  static vt::GrowOnlyStreamTriple<int> pool;
  return pool;
}

// The three device tables for `batch` pointers on `device`. Throws rather than
// returning empty tables: every caller below would otherwise hand a null array
// to hipblasGemmBatchedEx, and the driver's error would name the GEMM instead of
// the allocation.
vt::GrowOnlyStreamTriple<int>::Tables EnsurePtrTables(int device, int batch) {
  const size_t per_table = static_cast<size_t>(batch) * sizeof(void*);
  auto tables = PtrTablePool().Ensure(device, per_table, [](size_t bytes) -> void* {
    void* block = nullptr;
    // hipMalloc, not hipMallocAsync, for the same reason LtWorkspace above uses
    // it: these tables are consumed by hipBLAS, whose behaviour under capture
    // this project has not exercised. Removing the free is the part that is
    // verifiable, so it is the part that changed.
    if (hipMalloc(&block, bytes) != hipSuccess) return nullptr;
    return block;
  });
  if (tables.a == nullptr) {
    throw std::runtime_error("vt rocm: matmul_bt_ptr_batch: pointer tables");
  }
  return tables;
}

// C[M,N] row-major = alpha * A[M,K] @ B[N,K]^T + beta * C
// Same col-major view as hipblasGemmEx(OP_T, OP_N, m=N, n=M, k=K, B, K, A, K, C, N).
bool MatmulBTLt(hipStream_t stream, void* C, const void* A, const void* B, int M, int N, int K,
                float alpha, float beta) {
  if (!LtEnabled() || M <= 0 || N <= 0 || K <= 0) return false;
  auto& lt = GetLt();
  if (!lt.h) return false;

  const int ab_flag = (alpha == 1.f && beta == 0.f) ? 0 : 1;
  LtKey key{M, N, K, ab_flag};
  auto& cache = LtCache()[key];

  hipblasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;
  hipblasLtMatmulDesc_t matmulDesc = nullptr;
  hipblasLtMatmulPreference_t pref = nullptr;

  auto cleanup = [&]() {
    if (layoutA) (void)hipblasLtMatrixLayoutDestroy(layoutA);
    if (layoutB) (void)hipblasLtMatrixLayoutDestroy(layoutB);
    if (layoutC) (void)hipblasLtMatrixLayoutDestroy(layoutC);
    if (matmulDesc) (void)hipblasLtMatmulDescDestroy(matmulDesc);
    if (pref) (void)hipblasLtMatmulPreferenceDestroy(pref);
  };

  // Col-major views of row-major buffers (default ORDER_COL):
  //   weight B_rm[N,K] -> A_cm is K x N, ld=K
  //   act    A_rm[M,K] -> B_cm is K x M, ld=K
  //   out    C_rm[M,N] -> C_cm is N x M, ld=N
  if (hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16BF, K, N, K) != HIPBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }
  if (hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16BF, K, M, K) != HIPBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }
  if (hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16BF, N, M, N) != HIPBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }

  if (hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
      HIPBLAS_STATUS_SUCCESS) {
    cleanup();
    return false;
  }
  // C_cm = A_cm^T * B_cm  => (N x K)*(K x M) = N x M  == row C[M,N]
  const hipblasOperation_t opA = HIPBLAS_OP_T;
  const hipblasOperation_t opB = HIPBLAS_OP_N;
  (void)hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA,
                                        sizeof(opA));
  (void)hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB,
                                        sizeof(opB));

  if (!cache.ok) {
    if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) {
      cleanup();
      return false;
    }
    const uint64_t max_ws = 64ull << 20;
    (void)hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &max_ws, sizeof(max_ws));
    hipblasLtMatmulHeuristicResult_t heur{};
    int returned = 0;
    const hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(
        lt.h, matmulDesc, layoutA, layoutB, layoutC, layoutC, pref, 1, &heur, &returned);
    if (hs != HIPBLAS_STATUS_SUCCESS || returned <= 0 || heur.state != HIPBLAS_STATUS_SUCCESS) {
      cleanup();
      static thread_local int fail_log = 0;
      if (fail_log++ < 3) {
        std::fprintf(stderr, "hipBLASLt COL heuristic fail M=%d N=%d K=%d st=%d ret=%d\n", M, N,
                     K, static_cast<int>(hs), returned);
      }
      LtCache()[key] = LtAlgoCache{};
      return false;
    }
    cache.algo = heur.algo;
    cache.workspace = heur.workspaceSize;
    cache.ok = true;
    std::fprintf(stderr, "hipBLASLt COL ok M=%d N=%d K=%d ws=%zu\n", M, N, K, cache.workspace);
  }

  void* ws = LtWorkspace(cache.workspace, stream);
  if (cache.workspace > 0 && !ws) {
    cleanup();
    return false;
  }

  // Pass weight as A, act as B (matches GemmEx argument order for OP_T,OP_N)
  const hipblasStatus_t st =
      hipblasLtMatmul(lt.h, matmulDesc, &alpha, B, layoutA, A, layoutB, &beta, C, layoutC, C,
                      layoutC, &cache.algo, ws, cache.workspace, stream);
  cleanup();
  if (st != HIPBLAS_STATUS_SUCCESS) {
    cache.ok = false;
    std::fprintf(stderr, "hipBLASLt COL matmul fail st=%d M=%d N=%d K=%d\n", static_cast<int>(st),
                 M, N, K);
    return false;
  }
  return true;
}

}  // namespace

// out[M,N] = a[M,K] @ b[K,N]  (row-major contiguous / a may have row stride)
// Col-major BLAS view: C^T = B^T @ A^T
//   gemm(OP_T, OP_T, N, M, K, B, ldb=N? wait)
// Standard row-major C = A @ B via:
//   gemm(OP_N, OP_N, N, M, K, B_col, N, A_col, K, C_col, N)
// where A_row[M,K] is treated as A_col with ld=K meaning A_col is K x M...
//
// PyTorch/rocBLAS convention for row-major C(m,n) = A(m,k) @ B(k,n):
//   hipblasGemm(handle, OP_N, OP_N, n, m, k,
//               B, n, A, k, C, n)
// with A,B,C stored row-major — i.e. pass dimensions swapped so BLAS
// writes the transpose layout that coincides with row-major C.
void MatmulKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16 = a.dtype == DType::kBF16 && b.dtype == DType::kBF16;
  const bool f32 = a.dtype == DType::kF32 && b.dtype == DType::kF32;
  if (!bf16 && !f32) {
    throw std::runtime_error("vt rocm: matmul: unsupported " + ComboName(a, b, out));
  }
  if (out.dtype != DType::kF32 && out.dtype != DType::kBF16) {
    throw std::runtime_error("vt rocm: matmul: bad out dtype");
  }
  const int64_t M = a.shape[0], K = a.shape[1], N = b.shape[1];
  if (M == 0 || N == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    CheckHip(hipMemsetAsync(out.data, 0, out.Bytes(), s), "k0");
    return;
  }
  // Require contiguous B rows; A may be strided in dim0.
  if (b.stride[0] != N) {
    throw std::runtime_error("vt rocm: matmul: b must be row-contiguous");
  }
  if (a.stride[0] < K) {
    throw std::runtime_error("vt rocm: matmul: bad a stride");
  }

  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(a.dtype);
  const hipDataType ot = ToBlasType(out.dtype);

  // C_rm = A_rm @ B_rm  <=>  treat as gemm(N,N) writing C with ld=N, dims n,m,k
  // A leading dim = a.stride[0] (row stride in elements)
  CheckBlas(hipblasGemmEx(ctx.handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                          /*m=*/static_cast<int>(N), /*n=*/static_cast<int>(M),
                          /*k=*/static_cast<int>(K), &alpha, b.data, at,
                          /*ldb=*/static_cast<int>(N), a.data, at,
                          /*lda=*/static_cast<int>(a.stride[0]), &beta, out.data, ot,
                          /*ldc=*/static_cast<int>(N), GemmCompute(a.dtype),
                          HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmEx NN");
}

// out[M,N] = a[M,K] @ b[N,K]^T
// wvSplitK skinny-GEMM host entry (rocm_skinny_gemm.hip, #487). External
// vt::rocm linkage to match the definition; declared beside its only caller.
void WvSplitKBT(hipStream_t s, void* out, const void* a, const void* b, int M, int N,
                int K, int device);

// Effective YTILE after the T16 sweep knobs (VT_WVSPLIT_YTILE, default 4) —
// the gate must test the SAME tile the kernel will run.
int WvSplitKYtile();

// Row-major trick: gemm(OP_T, OP_N, N, M, K, B, K, A, a_rs, C, N)
//   BLAS: C = op(A)*op(B) with opA=T => A is KxN in col form = row B[N,K]
//   opB=N, B is KxM col = row A[M,K] with ld=a_rs
//   C is NxM col = row out[M,N] ld=N
void MatmulBTKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  EnsureQueueDevice(q);
  const bool bf16 = a.dtype == DType::kBF16 && b.dtype == DType::kBF16;
  const bool f32 = a.dtype == DType::kF32 && b.dtype == DType::kF32;
  if (!bf16 && !f32) {
    throw std::runtime_error("vt rocm: matmul_bt: unsupported " + ComboName(a, b, out));
  }
  if (out.dtype != DType::kF32 && out.dtype != DType::kBF16) {
    throw std::runtime_error("vt rocm: matmul_bt: bad out dtype");
  }
  const int64_t M = a.shape[0], K = a.shape[1], N = b.shape[0];
  if (M == 0 || N == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    CheckHip(hipMemsetAsync(out.data, 0, out.Bytes(), s), "bt k0");
    return;
  }
  if (b.stride[0] != K) {
    throw std::runtime_error("vt rocm: matmul_bt: b must be [N,K] row-contiguous");
  }
  if (a.stride[0] < K) {
    throw std::runtime_error("vt rocm: matmul_bt: bad a stride");
  }

  // Decode-skinny (M<=4) bf16: the wvSplitK port wins over the 128-tile GEMM
  // (#487). The dispatch guards mirror the donor EXACTLY (review sweep on
  // #506 found all three of these dropped, two of them memory-unsafe):
  //   * N > 8 (upstream `m > 8`, utils.py:181 — the feature-dim lower bound;
  //     at N==1 the kernel's y=1 store writes C[1] past the end of a 1-wide
  //     output);
  //   * N % YTILE == 0 (skinny_gemms.cu:1217 `M_in % _YTILE == 0` — the last
  //     wave's stores land past C[N] otherwise; the tile is the T16-effective
  //     one, NOT the donor's fixed 2: with the YT=4 default an N ≡ 2 mod 4
  //     shape would write two columns past the row under the old N%2 gate);
  //   * wave32 arch only (the gfx9 ROW_BCAST reduction arm is not ported);
  //   * K % 8 == 0 and the LDS-fit (K*M activation fits the staging buffer);
  //   everything else stays on the BLAS path.
  if (bf16 && out.dtype == DType::kBF16 && M >= 1 && M <= 4 && (K % 8) == 0 &&
      N > 8 && (N % vt::rocm::WvSplitKYtile()) == 0 && a.stride[0] == K &&
      K * M <= 32768 && SkinnyGemmEnabled() &&
      vt::rocm::SkinnyGemmArchOk(q.device.index, vt::rocm::DeviceArchName)) {
    WvSplitKBT(s, out.data, a.data, b.data, static_cast<int>(M), static_cast<int>(N),
               static_cast<int>(K), q.device.index);
    return;
  }

  // Decode: M=1 BF16 GEMV
  if (M == 1 && bf16 && out.dtype == DType::kBF16 && a.stride[0] == K && GemvEnabled()) {
    Bf16GemvBT(s, out.data, a.data, b.data, static_cast<int>(N), static_cast<int>(K), 1.f, 0.f);
    return;
  }

  // hipBLASLt (BF16 contiguous)
  if (bf16 && out.dtype == DType::kBF16 && a.stride[0] == K &&
      MatmulBTLt(s, out.data, a.data, b.data, static_cast<int>(M), static_cast<int>(N),
                 static_cast<int>(K), 1.f, 0.f)) {
    return;
  }

  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(a.dtype);
  const hipDataType ot = ToBlasType(out.dtype);

  // C = B^T @ A^T in row-major disguise:
  // gemm(OP_T, OP_N, N, M, K, B, ldb=K, A, lda=a_rs, C, ldc=N)
  CheckBlas(hipblasGemmEx(ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                          /*m=*/static_cast<int>(N), /*n=*/static_cast<int>(M),
                          /*k=*/static_cast<int>(K), &alpha, b.data, at,
                          /*ldb=*/static_cast<int>(K), a.data, at,
                          /*lda=*/static_cast<int>(a.stride[0]), &beta, out.data, ot,
                          /*ldc=*/static_cast<int>(N), GemmCompute(a.dtype),
                          HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmEx BT");
}

// out = alpha * (a @ b^T) + beta * out
void MatmulBTAlphaBetaRocm(Queue& q, void* out, const void* a, const void* b, int M, int N,
                           int K, float alpha, float beta, DType dtype) {
  if (M == 0 || N == 0) return;
  if (dtype != DType::kBF16 && dtype != DType::kF32) {
    throw std::runtime_error("vt rocm: matmul_bt_ab: dtype");
  }
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    if (beta == 0.f) {
      CheckHip(hipMemsetAsync(out, 0,
                              static_cast<size_t>(M) * static_cast<size_t>(N) * SizeOf(dtype),
                              s),
               "bt_ab k0");
    }
    return;
  }
  // Decode path: custom BF16 GEMV often beats hipblasGemmEx at M=1.
  if (M == 1 && dtype == DType::kBF16 && GemvEnabled()) {
    Bf16GemvBT(s, out, a, b, N, K, alpha, beta);
    return;
  }
  if (dtype == DType::kBF16 && MatmulBTLt(s, out, a, b, M, N, K, alpha, beta)) {
    return;
  }
  auto ctx = GetBlas(q.device.index, s);
  const hipDataType at = ToBlasType(dtype);
  CheckBlas(hipblasGemmEx(ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                          /*m=*/N, /*n=*/M, /*k=*/K, &alpha, b, at, /*ldb=*/K, a, at,
                          /*lda=*/K, &beta, out, at, /*ldc=*/N, GemmCompute(dtype),
                          HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmEx BT alpha/beta");
}

// General strided-batch BT: a may also be batched (strideA elements between batches).
// out[g,M,N] = a_g[M,K] @ b_g[N,K]^T
void MatmulBTStridedBatchFullKernelRocm(Queue& q, void* out, const void* a, const void* b,
                                        int batch, int M, int N, int K, long long strideA,
                                        DType dtype) {
  if (batch <= 0 || M == 0 || N == 0) return;
  if (dtype != DType::kBF16 && dtype != DType::kF32) {
    throw std::runtime_error("vt rocm: matmul_bt_batch: dtype");
  }
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    CheckHip(hipMemsetAsync(out, 0,
                            static_cast<size_t>(batch) * static_cast<size_t>(M) *
                                static_cast<size_t>(N) * SizeOf(dtype),
                            s),
             "bt_batch k0");
    return;
  }
  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(dtype);
  const long long strideB = static_cast<long long>(N) * K;
  const long long strideC = static_cast<long long>(M) * N;
  CheckBlas(hipblasGemmStridedBatchedEx(
                ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                /*m=*/N, /*n=*/M, /*k=*/K, &alpha, b, at, /*ldb=*/K, strideB, a, at,
                /*lda=*/K, strideA, &beta, out, at, /*ldc=*/N, strideC, batch,
                GemmCompute(dtype), HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmStridedBatchedEx BT");
}

// Batched BT via pointer arrays (no weight gather): out[g] = a @ b[g]^T
// a shared; b_ptrs[g] -> [N,K]; out_ptrs[g] -> [M,N]
void MatmulBTPointerBatchKernelRocm(Queue& q, void** out_ptrs, const void* a,
                                    void** b_ptrs, int batch, int M, int N, int K,
                                    DType dtype) {
  if (batch <= 0 || M == 0 || N == 0) return;
  if (dtype != DType::kBF16 && dtype != DType::kF32) {
    throw std::runtime_error("vt rocm: matmul_bt_ptr_batch: dtype");
  }
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    for (int g = 0; g < batch; ++g)
      CheckHip(hipMemsetAsync(out_ptrs[g], 0,
                              static_cast<size_t>(M) * static_cast<size_t>(N) * SizeOf(dtype),
                              s),
               "bt_ptr k0");
    return;
  }
  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(dtype);

  // Reuse device pointer tables (per-device, grows to max batch seen).
  const vt::GrowOnlyStreamTriple<int>::Tables tab = EnsurePtrTables(q.device.index, batch);
  std::vector<const void*> a_host(static_cast<size_t>(batch), a);
  const size_t psz = static_cast<size_t>(batch) * sizeof(void*);
  CheckHip(hipMemcpyAsync(tab.a, a_host.data(), psz, hipMemcpyHostToDevice, s), "H2D A");
  CheckHip(hipMemcpyAsync(tab.b, b_ptrs, psz, hipMemcpyHostToDevice, s), "H2D B");
  CheckHip(hipMemcpyAsync(tab.c, out_ptrs, psz, hipMemcpyHostToDevice, s), "H2D C");

  CheckBlas(hipblasGemmBatchedEx(
                ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                /*m=*/N, /*n=*/M, /*k=*/K, &alpha,
                reinterpret_cast<const void**>(tab.b), at, /*ldb=*/K,
                reinterpret_cast<const void**>(tab.a), at, /*lda=*/K, &beta,
                reinterpret_cast<void**>(tab.c), at, /*ldc=*/N, batch, GemmCompute(dtype),
                HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmBatchedEx BT");
}

// Pointer-array BT with per-batch A and B: out[g] = a[g] @ b[g]^T
void MatmulBTPointerBatchABKernelRocm(Queue& q, void** out_ptrs, void** a_ptrs,
                                      void** b_ptrs, int batch, int M, int N, int K,
                                      DType dtype) {
  if (batch <= 0 || M == 0 || N == 0) return;
  if (dtype != DType::kBF16 && dtype != DType::kF32) {
    throw std::runtime_error("vt rocm: matmul_bt_ptr_ab: dtype");
  }
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    for (int g = 0; g < batch; ++g)
      CheckHip(hipMemsetAsync(out_ptrs[g], 0,
                              static_cast<size_t>(M) * static_cast<size_t>(N) * SizeOf(dtype),
                              s),
               "bt_ptr_ab k0");
    return;
  }
  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType at = ToBlasType(dtype);

  const vt::GrowOnlyStreamTriple<int>::Tables tab = EnsurePtrTables(q.device.index, batch);
  const size_t psz = static_cast<size_t>(batch) * sizeof(void*);
  CheckHip(hipMemcpyAsync(tab.a, a_ptrs, psz, hipMemcpyHostToDevice, s), "H2D A");
  CheckHip(hipMemcpyAsync(tab.b, b_ptrs, psz, hipMemcpyHostToDevice, s), "H2D B");
  CheckHip(hipMemcpyAsync(tab.c, out_ptrs, psz, hipMemcpyHostToDevice, s), "H2D C");

  CheckBlas(hipblasGemmBatchedEx(
                ctx.handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                /*m=*/N, /*n=*/M, /*k=*/K, &alpha,
                reinterpret_cast<const void**>(tab.b), at, /*ldb=*/K,
                reinterpret_cast<const void**>(tab.a), at, /*lda=*/K, &beta,
                reinterpret_cast<void**>(tab.c), at, /*ldc=*/N, batch, GemmCompute(dtype),
                HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmBatchedEx BT AB");
}

// ---------------------------------------------------------------------------
// vt::BatchedMatmul (kBatchedMatmul) — the ROCm arm, MLA/DSA campaign W1
// (`BACKEND-ROCM`, issue #2715, spec .agents/specs/rocm-mla-dsa-ops.md).
//
// out[g] = a[g] @ b[g], all three row-major rank-3 with per-operand batch and
// row strides. This is the op the MLA projections reach for
// (`vllm/model_executor/layers/mla.py`'s `torch.bmm` over W_UK / W_UV); vLLM has
// no kernel of its own here, so the reference is the BLAS contract and the
// structural donor is `src/vt/cuda/cuda_matmul.cu:605-690`, which builds the
// same NN strided-batch call through cuBLASLt.
//
// THE OPERAND SWAP is this file's existing row-major convention, unchanged from
// `MatmulKernelRocm` above and from `MatmulBTStridedBatchFullKernelRocm`: a
// row-major X[p,q] with row stride ldx IS a column-major X^T[q,p] with leading
// dimension ldx, so row-major `C = A @ B` is the column-major `C^T = B^T · A^T`,
// i.e. `gemm(OP_N, OP_N, m=N, n=M, k=K)` with B passed FIRST. Leading dimensions
// are each operand's ROW stride (`stride[1]`) and batch strides its outer stride
// (`stride[0]`), which is what lets a strided view through unchanged — the donor
// keys its cuBLASLt plan on exactly those six values (`cuda_matmul.cu:658-669`).
//
// The dtype domain is the donor's: (bf16,bf16) and (f32,f32) in, f32 or bf16
// out. `hipblasGemmStridedBatchedEx` takes a separate type per operand, so the
// mixed-out case needs no cast pass.
void BatchedMatmulKernelRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16_in = a.dtype == DType::kBF16 && b.dtype == DType::kBF16 &&
                       (out.dtype == DType::kF32 || out.dtype == DType::kBF16);
  const bool f32_in = a.dtype == DType::kF32 && b.dtype == DType::kF32 &&
                      (out.dtype == DType::kF32 || out.dtype == DType::kBF16);
  if (!bf16_in && !f32_in) {
    throw std::runtime_error("vt rocm: batched_matmul: unsupported dtype combo " +
                             ComboName(a, b, out) +
                             "; supported: (bf16,bf16)->f32|bf16, (f32,f32)->f32|bf16");
  }
  const int64_t g = out.shape[0], M = out.shape[1], N = out.shape[2], K = a.shape[2];
  if (g == 0 || M == 0 || N == 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  if (K == 0) {
    // Empty contraction: out = 0. A wholesale memset is only sound when `out`
    // owns a dense span; a strided view is zeroed row by row (donor `:620-632`).
    if (out.IsContiguous()) {
      CheckHip(hipMemsetAsync(out.data, 0, out.Bytes(), s), "batched k=0 memset");
    } else {
      const size_t esz = SizeOf(out.dtype);
      for (int64_t bi = 0; bi < g; ++bi) {
        for (int64_t i = 0; i < M; ++i) {
          char* row = static_cast<char*>(out.data) +
                      static_cast<size_t>(bi * out.stride[0] + i * out.stride[1]) * esz;
          CheckHip(hipMemsetAsync(row, 0, static_cast<size_t>(N) * esz, s),
                   "batched k=0 row memset");
        }
      }
    }
    return;
  }

  auto ctx = GetBlas(q.device.index, s);
  const float alpha = 1.f, beta = 0.f;
  const hipDataType ab_t = ToBlasType(a.dtype);
  const hipDataType c_t = ToBlasType(out.dtype);
  CheckBlas(hipblasGemmStridedBatchedEx(
                ctx.handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                /*m=*/static_cast<int>(N), /*n=*/static_cast<int>(M),
                /*k=*/static_cast<int>(K), &alpha,
                /*A=*/b.data, ab_t, /*lda=*/static_cast<int>(b.stride[1]),
                /*strideA=*/static_cast<long long>(b.stride[0]),
                /*B=*/a.data, ab_t, /*ldb=*/static_cast<int>(a.stride[1]),
                /*strideB=*/static_cast<long long>(a.stride[0]), &beta,
                /*C=*/out.data, c_t, /*ldc=*/static_cast<int>(out.stride[1]),
                /*strideC=*/static_cast<long long>(out.stride[0]),
                /*batchCount=*/static_cast<int>(g), GemmCompute(a.dtype),
                HIPBLAS_GEMM_DEFAULT),
            "hipblasGemmStridedBatchedEx NN (batched_matmul)");
}

}  // namespace vt::rocm
