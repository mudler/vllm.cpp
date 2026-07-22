// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// cuBLASLt matmul for the CUDA backend (M0.6, correctness-grade).
// Supported dtype combos: (bf16,bf16)->f32, (bf16,bf16)->bf16, (f32,f32)->f32.
// Any other combo the public-op validation admits (e.g. f16 inputs) throws
// here, naming the combo — never silently truncates. Compute type is
// CUBLAS_COMPUTE_32F with f32 scale; all layouts are CUBLASLT_ORDER_ROW so no
// host-side transposes are needed.
// Upstream counterpart: torch.matmul/cublas path (no csrc kernel); cuBLASLt is our native equivalent.
//
// Also hosts MatmulFp8CublasLt (see the block comment near the bottom): the
// cuBLASLt FP8 (e4m3) dense GEMM — the native equivalent of vLLM's cuBLASLt fp8
// path (nvjet_sm121_qqtst_* kernels) — reusing this same handle + workspace. Its
// matmul descriptor + 3 layouts + heuristic algo can be cached per device on the
// full shape/config key (fp8_plan_cache.h), mirroring vLLM's in-graph plan reuse
// so the per-call heuristic + descriptor/layout rebuild is paid once per shape.
// DEFAULT OFF / opt-in VT_FP8_PLAN_CACHE=1: on GB10 the rebuild is NOT a
// removable wall-clock cost (bit-exact but measured production-NEUTRAL; the
// pre-GEMM GPU gap is unchanged — prefill is GPU-bound, decode is graph-captured
// so the heuristic runs at capture, not per replay-step). Bit-identical either
// way: the cuBLASLt algo selection is process-deterministic per shape.
//
// Env-gated diagnostic: VT_GEMM_ALGO_LOG=1 emits one std::cerr line per unique
// (shape, dtype-combo, epilogue) cuBLASLt algo selection (see MaybeLogGemmAlgo).
// Default OFF, zero hot-path cost when unset. It exists to compare arm-wise algo
// latching on the packed GDN BF16-BA GEMM per the 2026-07-15 forensic record;
// the portable flag/uniqueness plumbing lives in gemm_algo_log.h (CPU-tested).
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/cuda/fp8_plan_cache.h"
#include "vt/cuda/gemm_algo_log.h"
#include "vt/ops.h"

namespace vt::cuda {
namespace {

constexpr size_t kWorkspaceBytes = 32ull << 20;  // 32 MB, per the M0.6 plan

void CheckCuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: matmul: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

// Local status-name table: cublasGetStatusName lives in libcublas, which we
// do not link (only libcublasLt).
const char* StatusName(cublasStatus_t st) {
  switch (st) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_STATUS_<unknown>";
  }
}

void CheckLt(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("vt cuda: matmul: ") + what + ": cublas status " +
                             std::to_string(static_cast<int>(st)) + " (" + StatusName(st) +
                             ")");
  }
}

// cublasLt handle + 32 MB device workspace, cached per device index. Created
// lazily under a mutex on first matmul — never at static-init time (the
// pre-main discipline forbids CUDA API calls during registration). Both live
// for the whole process and are deliberately never destroyed: freeing at exit
// would race CUDA driver teardown, and one handle + workspace per device is a
// bounded leak (documented in the M0.6 plan). Note the current backend
// registers device 0 only and never switches devices; creation happens on the
// caller's current device, so true multi-device use would additionally need
// cudaSetDevice here.
struct LtContext {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
};

LtContext GetContext(int device) {
  static std::mutex mu;
  static std::unordered_map<int, LtContext> ctxs;  // values leak by design (see above)
  std::lock_guard<std::mutex> lock(mu);
  auto it = ctxs.find(device);
  if (it != ctxs.end()) return it->second;
  LtContext ctx;
  CheckLt(cublasLtCreate(&ctx.handle), "cublasLtCreate");
  cudaError_t err = cudaMalloc(&ctx.workspace, kWorkspaceBytes);
  if (err != cudaSuccess) {
    cublasLtDestroy(ctx.handle);  // creation failed part-way: don't cache a half-made context
    CheckCuda(err, "workspace cudaMalloc");
  }
  ctxs.emplace(device, ctx);
  return ctx;
}

// RAII for the per-call cublasLt descriptor objects (throw-safe cleanup).
struct LayoutGuard {
  cublasLtMatrixLayout_t v = nullptr;
  ~LayoutGuard() {
    if (v != nullptr) cublasLtMatrixLayoutDestroy(v);
  }
};
struct DescGuard {
  cublasLtMatmulDesc_t v = nullptr;
  ~DescGuard() {
    if (v != nullptr) cublasLtMatmulDescDestroy(v);
  }
};
struct PrefGuard {
  cublasLtMatmulPreference_t v = nullptr;
  ~PrefGuard() {
    if (v != nullptr) cublasLtMatmulPreferenceDestroy(v);
  }
};

void MakeRowMajor(LayoutGuard& l, cudaDataType_t t, int64_t rows, int64_t cols) {
  CheckLt(cublasLtMatrixLayoutCreate(&l.v, t, static_cast<uint64_t>(rows),
                                     static_cast<uint64_t>(cols), cols),
          "cublasLtMatrixLayoutCreate");
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  CheckLt(cublasLtMatrixLayoutSetAttribute(l.v, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                           sizeof(order)),
          "set CUBLASLT_MATRIX_LAYOUT_ORDER");
}

// Row-major layout with an EXPLICIT leading dimension (the tensor's row stride,
// which may exceed `cols` for a strided view) plus strided-batch metadata. The
// batch stride is in ELEMENTS, matching vt::Tensor::stride.
void MakeRowMajorBatched(LayoutGuard& l, cudaDataType_t t, int64_t rows, int64_t cols,
                         int64_t ld, int32_t batch, int64_t batch_stride) {
  CheckLt(cublasLtMatrixLayoutCreate(&l.v, t, static_cast<uint64_t>(rows),
                                     static_cast<uint64_t>(cols), ld),
          "cublasLtMatrixLayoutCreate (batched)");
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  CheckLt(cublasLtMatrixLayoutSetAttribute(l.v, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                           sizeof(order)),
          "set CUBLASLT_MATRIX_LAYOUT_ORDER (batched)");
  CheckLt(cublasLtMatrixLayoutSetAttribute(l.v, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch,
                                           sizeof(batch)),
          "set CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT");
  CheckLt(cublasLtMatrixLayoutSetAttribute(
              l.v, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride,
              sizeof(batch_stride)),
          "set CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET");
}

std::string ComboName(const Tensor& a, const Tensor& b, const Tensor& out) {
  return std::string("(") + Name(a.dtype) + "," + Name(b.dtype) + ")->" + Name(out.dtype);
}

// ---- Env-gated cuBLASLt algo-selection diagnostic (VT_GEMM_ALGO_LOG) --------
// Short tag for a cuBLASLt operand data type (the BF16-vs-F32 output type is the
// packed-arm variable of interest).
const char* CudaTypeTag(cudaDataType_t t) {
  switch (t) {
    case CUDA_R_32F: return "f32";
    case CUDA_R_16BF: return "bf16";
    case CUDA_R_16F: return "f16";
    case CUDA_R_8F_E4M3: return "e4m3";
    default: return "other";
  }
}

// When VT_GEMM_ALGO_LOG=1, emit ONE std::cerr line per unique (shape,
// dtype-combo, epilogue) selection naming the cuBLASLt-chosen algo config
// (id/tile/stages/splitK) and its heuristic workspace. OUR diagnostic — upstream
// logs the same selection under CUBLASLT_LOG_LEVEL / torch `_scaled_mm` verbose;
// we have no torch, so we mirror it under our own flag to compare arm-wise algo
// LATCHING on the packed GDN BF16-BA GEMM vs the F32-BA/decomposed arm, per the
// 2026-07-15 forensic record (a constant ~0.2% packed steady per-token tax whose
// one un-instrumented per-process variable is the BF16 GEMM algo selection; see
// .agents/state.md and .agents/specs/gdn-packed-decode.md). Zero cost when the
// flag is unset: GemmAlgoLogEnabled() is a cached bool and the body is skipped.
// The reads are best-effort (a driver that does not expose an attribute leaves
// its sentinel) and never throw — this is diagnostics, not a correctness path.
void MaybeLogGemmAlgo(const cublasLtMatmulHeuristicResult_t& heur, int64_t m, int64_t n,
                      int64_t k, cudaDataType_t a_t, cudaDataType_t b_t, cudaDataType_t c_t,
                      const char* epilogue) {
  if (!GemmAlgoLogEnabled()) return;  // cached bool; default OFF pays nothing here
  int32_t algo_id = -1, split_k = -1;
  uint32_t tile = 0, stages = 0;
  size_t written = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id,
                                       sizeof(algo_id), &written);
  cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile,
                                       sizeof(tile), &written);
  cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages,
                                       sizeof(stages), &written);
  cublasLtMatmulAlgoConfigGetAttribute(&heur.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k,
                                       sizeof(split_k), &written);
  // One line per unique key across every cuBLASLt GEMM site (the key embeds the
  // backend/epilogue tag, so the three paths never collide).
  static LogOncePerKey once;
  std::string key = std::string("cublasLt|m=") + std::to_string(m) + " n=" + std::to_string(n) +
                    " k=" + std::to_string(k) + "|a=" + CudaTypeTag(a_t) + " b=" +
                    CudaTypeTag(b_t) + " c=" + CudaTypeTag(c_t) + "|" + epilogue;
  if (!once.ShouldLog(key)) return;
  std::cerr << "[VT_GEMM_ALGO] backend=cublasLt m=" << m << " n=" << n << " k=" << k
            << " a=" << CudaTypeTag(a_t) << " b=" << CudaTypeTag(b_t) << " c=" << CudaTypeTag(c_t)
            << " epilogue=" << epilogue << " algoId=" << algo_id << " tile=" << tile
            << " stages=" << stages << " splitK=" << split_k << " wsSize=" << heur.workspaceSize
            << std::endl;
}

void MatmulKernelCuda(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16_in = a.dtype == DType::kBF16 && b.dtype == DType::kBF16 &&
                       (out.dtype == DType::kF32 || out.dtype == DType::kBF16);
  const bool f32_all =
      a.dtype == DType::kF32 && b.dtype == DType::kF32 && out.dtype == DType::kF32;
  if (!bf16_in && !f32_all) {
    throw std::runtime_error("vt cuda: matmul: unsupported dtype combo " +
                             ComboName(a, b, out) +
                             "; supported: (bf16,bf16)->f32|bf16, (f32,f32)->f32");
  }
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  if (m == 0 || n == 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  if (k == 0) {  // empty reduction: out = 0 (f32 and bf16 zero are all-zero bytes)
    CheckCuda(cudaMemsetAsync(out.data, 0, out.Bytes(), s), "k=0 memset");
    return;
  }

  const LtContext ctx = GetContext(q.device.index);
  const cudaDataType_t ab_type = a.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;
  const cudaDataType_t out_type = out.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;

  DescGuard desc;
  CheckLt(cublasLtMatmulDescCreate(&desc.v, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "cublasLtMatmulDescCreate");
  LayoutGuard la, lb, lc;
  MakeRowMajor(la, ab_type, m, k);
  MakeRowMajor(lb, ab_type, k, n);
  MakeRowMajor(lc, out_type, m, n);

  PrefGuard pref;
  CheckLt(cublasLtMatmulPreferenceCreate(&pref.v), "cublasLtMatmulPreferenceCreate");
  CheckLt(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
          "set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");

  cublasLtMatmulHeuristicResult_t heur{};
  int returned = 0;
  CheckLt(cublasLtMatmulAlgoGetHeuristic(ctx.handle, desc.v, la.v, lb.v, lc.v, lc.v, pref.v,
                                         /*requestedAlgoCount=*/1, &heur, &returned),
          "cublasLtMatmulAlgoGetHeuristic");
  if (returned == 0) {
    throw std::runtime_error("vt cuda: matmul: no cublasLt heuristic for [" +
                             std::to_string(m) + "," + std::to_string(k) + "]x[" +
                             std::to_string(k) + "," + std::to_string(n) + "] " +
                             ComboName(a, b, out));
  }
  MaybeLogGemmAlgo(heur, m, n, k, ab_type, ab_type, out_type, "rowmajor-NN");

  // out = 1.0 * a @ b + 0.0 * out; C and D share the same buffer and layout.
  const float alpha = 1.0f, beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, desc.v, &alpha, a.data, la.v, b.data, lb.v, &beta,
                         out.data, lc.v, out.data, lc.v, &heur.algo, ctx.workspace,
                         kWorkspaceBytes, s),
          "cublasLtMatmul");
}

// ---- cuBLASLt bf16/f32 "BT" dense GEMM (b = Linear weight [N,K]) -----------
// out[M,N] = a[M,K] @ b^T with b [N,K] row-major — K contiguous in BOTH
// operands, the TN layout vLLM's F.linear hits for its bf16 projections. On
// GB10 cuBLASLt serves this with the fast `nvjet_sm121_tst_..._TNNN` kernels
// (measured 27B GDN in_proj: 1.80 us/tok vs 2.29 us/tok for our row-major x
// row-major kMatmul, which falls to `NNNN` nvjet / sm80-cutlass kernels).
// Same column-major TN formulation as MatmulFp8CublasLtKernelCuda below (see
// its derivation comment); only the A/B dtypes differ (bf16/f32, alpha=1).
void MatmulBTKernelCuda(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16_in = a.dtype == DType::kBF16 && b.dtype == DType::kBF16;
  const bool f32_in = a.dtype == DType::kF32 && b.dtype == DType::kF32;
  if (!bf16_in && !f32_in) {
    throw std::runtime_error("vt cuda: matmul_bt: unsupported dtype combo " +
                             ComboName(a, b, out) +
                             "; supported: (bf16,bf16)->f32|bf16, (f32,f32)->f32|bf16");
  }
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[0];
  if (m == 0 || n == 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  if (k == 0) {
    CheckCuda(cudaMemsetAsync(out.data, 0, out.Bytes(), s), "bt k=0 memset");
    return;
  }

  const LtContext ctx = GetContext(q.device.index);
  const cudaDataType_t ab_type = f32_in ? CUDA_R_32F : CUDA_R_16BF;
  const cudaDataType_t out_type = out.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;

  DescGuard desc;
  CheckLt(cublasLtMatmulDescCreate(&desc.v, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "bt cublasLtMatmulDescCreate");
  const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  CheckLt(cublasLtMatmulDescSetAttribute(desc.v, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)),
          "bt set TRANSA=T");
  CheckLt(cublasLtMatmulDescSetAttribute(desc.v, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)),
          "bt set TRANSB=N");

  // Column-major TN layouts: A=weight col[K,N] ld=K (TRANSA=T => [N,K] row),
  // B=act col[K,M] ld=K (= [M,K] row), C=D=out col[N,M] ld=N (= [M,N] row).
  LayoutGuard la, lb, lc;
  CheckLt(cublasLtMatrixLayoutCreate(&la.v, ab_type, static_cast<uint64_t>(k),
                                     static_cast<uint64_t>(n), k),
          "bt Adesc (weight)");
  // ld = the activation's ROW stride (== k for a contiguous activation, so
  // contiguous callers hand cuBLASLt byte-identical layouts and get the same
  // algo). A wider stride is the MLA chunked-prefill column slice, W6.
  CheckLt(cublasLtMatrixLayoutCreate(&lb.v, ab_type, static_cast<uint64_t>(k),
                                     static_cast<uint64_t>(m), a.stride[0]),
          "bt Bdesc (act)");
  CheckLt(cublasLtMatrixLayoutCreate(&lc.v, out_type, static_cast<uint64_t>(n),
                                     static_cast<uint64_t>(m), n),
          "bt Cdesc (out)");

  PrefGuard pref;
  CheckLt(cublasLtMatmulPreferenceCreate(&pref.v), "bt cublasLtMatmulPreferenceCreate");
  CheckLt(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
          "bt set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");

  cublasLtMatmulHeuristicResult_t heur{};
  int returned = 0;
  CheckLt(cublasLtMatmulAlgoGetHeuristic(ctx.handle, desc.v, la.v, lb.v, lc.v, lc.v, pref.v,
                                         /*requestedAlgoCount=*/1, &heur, &returned),
          "bt cublasLtMatmulAlgoGetHeuristic");
  if (returned == 0) {
    throw std::runtime_error("vt cuda: matmul_bt: no cublasLt heuristic for [" +
                             std::to_string(m) + "," + std::to_string(k) + "]x[" +
                             std::to_string(n) + "," + std::to_string(k) + "]^T " +
                             ComboName(a, b, out));
  }
  // The 27B GDN in_proj_ba runs through this TN path; its BF16-vs-F32 output type
  // is the algo-latching variable the forensic record flagged.
  MaybeLogGemmAlgo(heur, m, n, k, ab_type, ab_type, out_type, "TN-bt");

  const float alpha = 1.0f, beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, desc.v, &alpha, b.data, la.v, a.data, lb.v, &beta,
                         out.data, lc.v, out.data, lc.v, &heur.algo, ctx.workspace,
                         kWorkspaceBytes, s),
          "bt cublasLtMatmul");
}

// ---- cuBLASLt strided-batched bf16/f32 GEMM (vt::BatchedMatmul) ------------
// out[G,M,N] = a[G,M,K] @ b[G,K,N] — the 1:1 counterpart of `torch.bmm`, the
// primitive MLA weight absorption is expressed in upstream
// (mla_attention.py:789 folds W_UK into the decode query, :1034 un-projects the
// latent output with W_UV). torch.bmm on CUDA bf16 resolves to cuBLAS
// `gemmStridedBatchedEx` (CUDA_R_16BF, CUBLAS_COMPUTE_32F); this is the
// cuBLASLt strided-batched form of exactly that GEMM, reusing the same handle
// and 32 MB workspace as the dense paths above.
//
// Row-major NN layouts with an EXPLICIT leading dimension, so a transposed view
// (both upstream call sites pass one) is consumed with no copy: the batch axis
// need not be the outermost storage axis, only the innermost dim must be
// unit-stride (enforced in ops.cpp).
void BatchedMatmulKernelCuda(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  const bool bf16_in = a.dtype == DType::kBF16 && b.dtype == DType::kBF16 &&
                       (out.dtype == DType::kF32 || out.dtype == DType::kBF16);
  const bool f32_in = a.dtype == DType::kF32 && b.dtype == DType::kF32 &&
                      (out.dtype == DType::kF32 || out.dtype == DType::kBF16);
  if (!bf16_in && !f32_in) {
    throw std::runtime_error("vt cuda: batched_matmul: unsupported dtype combo " +
                             ComboName(a, b, out) +
                             "; supported: (bf16,bf16)->f32|bf16, (f32,f32)->f32|bf16");
  }
  const int64_t g = out.shape[0], m = out.shape[1], n = out.shape[2], k = a.shape[2];
  if (g == 0 || m == 0 || n == 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  if (k == 0) {
    // Empty contraction: out = 0. Only safe to memset wholesale when `out` owns
    // a dense span; a strided view is zeroed row by row.
    if (out.IsContiguous()) {
      CheckCuda(cudaMemsetAsync(out.data, 0, out.Bytes(), s), "batched k=0 memset");
    } else {
      const size_t esz = SizeOf(out.dtype);
      for (int64_t bi = 0; bi < g; ++bi) {
        for (int64_t i = 0; i < m; ++i) {
          char* row = static_cast<char*>(out.data) +
                      static_cast<size_t>(bi * out.stride[0] + i * out.stride[1]) * esz;
          CheckCuda(cudaMemsetAsync(row, 0, static_cast<size_t>(n) * esz, s),
                    "batched k=0 row memset");
        }
      }
    }
    return;
  }

  const LtContext ctx = GetContext(q.device.index);
  const cudaDataType_t ab_type = a.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;
  const cudaDataType_t out_type = out.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;

  DescGuard desc;
  CheckLt(cublasLtMatmulDescCreate(&desc.v, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "batched cublasLtMatmulDescCreate");
  LayoutGuard la, lb, lc;
  const int32_t batch = static_cast<int32_t>(g);
  MakeRowMajorBatched(la, ab_type, m, k, a.stride[1], batch, a.stride[0]);
  MakeRowMajorBatched(lb, ab_type, k, n, b.stride[1], batch, b.stride[0]);
  MakeRowMajorBatched(lc, out_type, m, n, out.stride[1], batch, out.stride[0]);

  PrefGuard pref;
  CheckLt(cublasLtMatmulPreferenceCreate(&pref.v), "batched cublasLtMatmulPreferenceCreate");
  CheckLt(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
          "batched set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");

  cublasLtMatmulHeuristicResult_t heur{};
  int returned = 0;
  CheckLt(cublasLtMatmulAlgoGetHeuristic(ctx.handle, desc.v, la.v, lb.v, lc.v, lc.v, pref.v,
                                         /*requestedAlgoCount=*/1, &heur, &returned),
          "batched cublasLtMatmulAlgoGetHeuristic");
  if (returned == 0) {
    throw std::runtime_error("vt cuda: batched_matmul: no cublasLt heuristic for g=" +
                             std::to_string(g) + " [" + std::to_string(m) + "," +
                             std::to_string(k) + "]x[" + std::to_string(k) + "," +
                             std::to_string(n) + "] " + ComboName(a, b, out));
  }
  MaybeLogGemmAlgo(heur, m, n, k, ab_type, ab_type, out_type, "rowmajor-NN-batched");

  const float alpha = 1.0f, beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, desc.v, &alpha, a.data, la.v, b.data, lb.v, &beta,
                         out.data, lc.v, out.data, lc.v, &heur.algo, ctx.workspace,
                         kWorkspaceBytes, s),
          "batched cublasLtMatmul");
}

// ---- cuBLASLt FP8 (e4m3) dense GEMM ---------------------------------------
// The native equivalent of vLLM's cuBLASLt fp8 dense path (the
// `nvjet_sm121_qqtst_*` / `qq*` kernels torch._scaled_mm / cublasLt select for
// the 35B fp8 projections on GB10/sm_121a). Reuses the SAME cublasLt handle +
// 32 MB workspace as the bf16 dense GEMM above; only the matmul descriptor
// changes to the fp8 config: CUBLAS_COMPUTE_32F, e4m3 A/B, f32 scale.
//
// cuBLASLt fp8 requires the "TN" layout — the contraction dim K must be the
// contiguous (leading) dim of BOTH operands. Our activation a_fp8 [M,K] and
// weight b_fp8 [N,K] are row-major (K contiguous), so they already satisfy it
// with no host-side transpose. We compute the row-major out[M,N] as its
// column-major transpose out^T[N,M] = op(weight,T)[N,K] @ op(act,N)[K,M]:
//   A = weight  : col-major [K,N] (rows=K,cols=N,ld=K), TRANSA=OP_T  (= [N,K] row-major)
//   B = act     : col-major [K,M] (rows=K,cols=M,ld=K), TRANSB=OP_N  (= [M,K] row-major)
//   C = D = out : col-major [N,M] (rows=N,cols=M,ld=N)               (= [M,N] row-major)
// The two per-tensor static scales are folded into the host alpha (=
// input_scale*weight_scale) applied to the fp32 accumulator — identical math to
// MatmulFp8Cutlass (per-tensor scalars: dequant = fp8 * scale, and
// alpha*(A_fp8@B_fp8) reproduces sum dequant(a)*dequant(w)). If cublasLt has no
// fp8 heuristic for a given shape (e.g. tiny M on some drivers), we fall back to
// the already-16/16-validated cutlass fp8 GEMM so the correctness gate holds.
// A built fp8 GEMM plan: the matmul descriptor, the three col-major TN layouts,
// and the heuristic-selected algo. Handles are opaque cuBLASLt pointers; when
// this plan lives in the per-device cache its handles are process-lifetime (never
// destroyed — same rationale as the LtContext above: freeing at exit races the
// CUDA driver teardown; the cache is bounded by the finite set of fp8 shapes).
// When built fresh (VT_FP8_PLAN_CACHE=0 rollback), the caller destroys it after
// the matmul via Fp8PlanGuard, mirroring the pre-cache per-call behavior.
struct Fp8Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr;  // A = weight, col-major [K,N] ld=K
  cublasLtMatrixLayout_t lb = nullptr;  // B = act,    col-major [K,M] ld=K
  cublasLtMatrixLayout_t lc = nullptr;  // C = D = out, col-major [N,M] ld=N
  cublasLtMatmulHeuristicResult_t heur{};
};

// RAII teardown for a FRESHLY-built plan (used only on the VT_FP8_PLAN_CACHE=0
// rollback path). Cached plans are deliberately NOT guarded — they leak by design.
struct Fp8PlanGuard {
  Fp8Plan* p = nullptr;
  ~Fp8PlanGuard() {
    if (p == nullptr) return;
    if (p->lc != nullptr) cublasLtMatrixLayoutDestroy(p->lc);
    if (p->lb != nullptr) cublasLtMatrixLayoutDestroy(p->lb);
    if (p->la != nullptr) cublasLtMatrixLayoutDestroy(p->la);
    if (p->desc != nullptr) cublasLtMatmulDescDestroy(p->desc);
  }
};

// Build the fp8 TN descriptor + three layouts + heuristic algo for `key` on
// `ctx.handle`. On success fills *out with raw handles (caller owns lifetime) and
// returns true. Returns false iff cuBLASLt reports no fp8 heuristic for the shape
// (any partially-created handles are destroyed) — the caller then falls back to
// the cutlass fp8 GEMM. This is the exact descriptor/layout/heuristic sequence the
// pre-cache code ran inline on every call; the key fields are the only inputs.
bool BuildFp8Plan(const LtContext& ctx, const Fp8PlanKey& key, Fp8Plan* out) {
  Fp8Plan p;
  const cudaDataType_t out_type = static_cast<cudaDataType_t>(key.out_type);
  CheckLt(cublasLtMatmulDescCreate(&p.desc, static_cast<cublasComputeType_t>(key.compute_type),
                                   static_cast<cudaDataType_t>(key.scale_type)),
          "fp8 cublasLtMatmulDescCreate");
  const cublasOperation_t op_a = static_cast<cublasOperation_t>(key.trans_a);
  const cublasOperation_t op_b = static_cast<cublasOperation_t>(key.trans_b);
  CheckLt(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)),
          "fp8 set TRANSA=T");
  CheckLt(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)),
          "fp8 set TRANSB=N");

  // Column-major TN layouts (default order — NOT ORDER_ROW; fp8 needs the native
  // col-major TN form). See the derivation in the block comment above.
  const cudaDataType_t a_type = static_cast<cudaDataType_t>(key.a_type);
  CheckLt(cublasLtMatrixLayoutCreate(&p.la, a_type, static_cast<uint64_t>(key.k),
                                     static_cast<uint64_t>(key.n), key.k),
          "fp8 Adesc (weight)");
  CheckLt(cublasLtMatrixLayoutCreate(&p.lb, a_type, static_cast<uint64_t>(key.k),
                                     static_cast<uint64_t>(key.m), key.k),
          "fp8 Bdesc (act)");
  CheckLt(cublasLtMatrixLayoutCreate(&p.lc, out_type, static_cast<uint64_t>(key.n),
                                     static_cast<uint64_t>(key.m), key.n),
          "fp8 Cdesc (out)");

  PrefGuard pref;
  CheckLt(cublasLtMatmulPreferenceCreate(&pref.v), "fp8 cublasLtMatmulPreferenceCreate");
  CheckLt(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
          "fp8 set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");

  int returned = 0;
  const cublasStatus_t hst = cublasLtMatmulAlgoGetHeuristic(
      ctx.handle, p.desc, p.la, p.lb, p.lc, p.lc, pref.v, /*requestedAlgoCount=*/1, &p.heur,
      &returned);
  if (hst != CUBLAS_STATUS_SUCCESS || returned == 0) {
    if (p.lc != nullptr) cublasLtMatrixLayoutDestroy(p.lc);
    if (p.lb != nullptr) cublasLtMatrixLayoutDestroy(p.lb);
    if (p.la != nullptr) cublasLtMatrixLayoutDestroy(p.la);
    if (p.desc != nullptr) cublasLtMatmulDescDestroy(p.desc);
    return false;  // caller falls back to cutlass; nothing cached
  }
  *out = p;
  return true;
}

// Per-device fp8 plan cache. On a hit, returns the cached {desc, layouts, algo}
// by value (the handles are stable pointers into process-lifetime cached objects
// — std::unordered_map keeps element addresses stable across rehash, and the
// values are never destroyed), skipping the descriptor/layout creation + the
// ~0.8 ms heuristic entirely. On a miss, builds under the lock (a one-time cost
// per unique shape) and inserts. Returns false iff BuildFp8Plan found no fp8
// heuristic (not cached — a rare shape that falls back to cutlass). The lock is
// held only across the map access + the (cold) build, never across the matmul.
bool GetOrBuildCachedFp8Plan(const LtContext& ctx, const Fp8PlanKey& key, Fp8Plan* out) {
  static std::mutex mu;
  static std::unordered_map<Fp8PlanKey, Fp8Plan, Fp8PlanKeyHash> plans;
  std::lock_guard<std::mutex> lock(mu);
  auto it = plans.find(key);
  if (it != plans.end()) {
    *out = it->second;  // cache hit: skip desc/layout creation + heuristic
    return true;
  }
  Fp8Plan p;
  if (!BuildFp8Plan(ctx, key, &p)) return false;  // no heuristic: uncached fallback
  plans.emplace(key, p);  // process-lifetime; handles never destroyed (by design)
  *out = p;
  return true;
}

void MatmulFp8CublasLtKernelCuda(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                                 float alpha) {
  const int64_t m = a_fp8.shape[0], k = a_fp8.shape[1], n = b_fp8.shape[0];
  if (m == 0 || n == 0) return;
  cudaStream_t s = static_cast<cudaStream_t>(q.handle);
  if (k == 0) {  // empty reduction: out = 0 (f32 and bf16 zero are all-zero bytes)
    CheckCuda(cudaMemsetAsync(out.data, 0, out.Bytes(), s), "fp8 k=0 memset");
    return;
  }

  const LtContext ctx = GetContext(q.device.index);
  const cudaDataType_t out_type = out.dtype == DType::kF32 ? CUDA_R_32F : CUDA_R_16BF;

  // The FULL key that determines the descriptor + selected algo (see
  // fp8_plan_cache.h for the per-field rationale). alpha is applied per-call and
  // does NOT affect the descriptor/algo, so it is deliberately absent from the key.
  Fp8PlanKey key;
  key.device = q.device.index;
  key.m = m;
  key.n = n;
  key.k = k;
  key.out_type = static_cast<int>(out_type);
  key.a_type = static_cast<int>(CUDA_R_8F_E4M3);
  key.compute_type = static_cast<int>(CUBLAS_COMPUTE_32F);
  key.scale_type = static_cast<int>(CUDA_R_32F);
  key.trans_a = static_cast<int>(CUBLAS_OP_T);
  key.trans_b = static_cast<int>(CUBLAS_OP_N);
  key.epilogue = static_cast<int>(CUBLASLT_EPILOGUE_DEFAULT);
  key.scale_mode = 0;  // per-tensor scale folded into host alpha; no device scale ptrs

  // Cache ON (opt-in VT_FP8_PLAN_CACHE=1): reuse the per-device plan, skipping the
  // per-call descriptor/layout creation + heuristic. Cache OFF (default): build a
  // fresh plan and destroy it after the matmul (Fp8PlanGuard), exactly the
  // pre-cache per-call behavior — the shipped production path.
  const bool cache_on = Fp8PlanCacheEnabled();
  Fp8Plan plan;
  Fp8PlanGuard guard;  // engaged only on the rollback path
  bool have_plan;
  if (cache_on) {
    have_plan = GetOrBuildCachedFp8Plan(ctx, key, &plan);
  } else {
    have_plan = BuildFp8Plan(ctx, key, &plan);
    if (have_plan) guard.p = &plan;  // fresh plan: destroy after the matmul
  }
  if (!have_plan) {
    // No cublasLt fp8 kernel for this shape/config -> keep the gate robust by
    // routing to the already-validated cutlass fp8 GEMM (same fp8 math).
    ::vt::MatmulFp8Cutlass(q, out, a_fp8, b_fp8, alpha);
    return;
  }
  MaybeLogGemmAlgo(plan.heur, m, n, k, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, out_type, "TN-fp8");

  // out = alpha * op(weight) @ op(act) + 0 * C; C and D share out's buffer/layout.
  const float beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, plan.desc, &alpha, b_fp8.data, plan.la, a_fp8.data, plan.lb,
                         &beta, out.data, plan.lc, out.data, plan.lc, &plan.heur.algo,
                         ctx.workspace, kWorkspaceBytes, s),
          "fp8 cublasLtMatmul");
}

// Registers the CUDA matmul during static init (table fill only, no CUDA
// calls — see cuda_ops.cu for the rationale).
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmul, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernelCuda)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernelCuda)));
    RegisterOp(OpId::kBatchedMatmul, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<BatchedMatmulFn>(&BatchedMatmulKernelCuda)));
    RegisterOp(OpId::kMatmulFp8CublasLt, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulFp8CublasLtFn>(&MatmulFp8CublasLtKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
