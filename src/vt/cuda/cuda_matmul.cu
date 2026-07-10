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
// path (nvjet_sm121_qqtst_* kernels) — reusing this same handle + workspace.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

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

std::string ComboName(const Tensor& a, const Tensor& b, const Tensor& out) {
  return std::string("(") + Name(a.dtype) + "," + Name(b.dtype) + ")->" + Name(out.dtype);
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
  CheckLt(cublasLtMatrixLayoutCreate(&lb.v, ab_type, static_cast<uint64_t>(k),
                                     static_cast<uint64_t>(m), k),
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

  const float alpha = 1.0f, beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, desc.v, &alpha, b.data, la.v, a.data, lb.v, &beta,
                         out.data, lc.v, out.data, lc.v, &heur.algo, ctx.workspace,
                         kWorkspaceBytes, s),
          "bt cublasLtMatmul");
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

  DescGuard desc;
  CheckLt(cublasLtMatmulDescCreate(&desc.v, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "fp8 cublasLtMatmulDescCreate");
  const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  CheckLt(cublasLtMatmulDescSetAttribute(desc.v, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)),
          "fp8 set TRANSA=T");
  CheckLt(cublasLtMatmulDescSetAttribute(desc.v, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)),
          "fp8 set TRANSB=N");

  // Column-major layouts (default order — NOT ORDER_ROW; fp8 needs the native
  // col-major TN form). See the derivation in the block comment above.
  LayoutGuard la, lb, lc;
  CheckLt(cublasLtMatrixLayoutCreate(&la.v, CUDA_R_8F_E4M3, static_cast<uint64_t>(k),
                                     static_cast<uint64_t>(n), k),
          "fp8 Adesc (weight)");
  CheckLt(cublasLtMatrixLayoutCreate(&lb.v, CUDA_R_8F_E4M3, static_cast<uint64_t>(k),
                                     static_cast<uint64_t>(m), k),
          "fp8 Bdesc (act)");
  CheckLt(cublasLtMatrixLayoutCreate(&lc.v, out_type, static_cast<uint64_t>(n),
                                     static_cast<uint64_t>(m), n),
          "fp8 Cdesc (out)");

  PrefGuard pref;
  CheckLt(cublasLtMatmulPreferenceCreate(&pref.v), "fp8 cublasLtMatmulPreferenceCreate");
  CheckLt(cublasLtMatmulPreferenceSetAttribute(pref.v, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
          "fp8 set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");

  cublasLtMatmulHeuristicResult_t heur{};
  int returned = 0;
  const cublasStatus_t hst = cublasLtMatmulAlgoGetHeuristic(
      ctx.handle, desc.v, la.v, lb.v, lc.v, lc.v, pref.v, /*requestedAlgoCount=*/1, &heur,
      &returned);
  if (hst != CUBLAS_STATUS_SUCCESS || returned == 0) {
    // No cublasLt fp8 kernel for this shape/config -> keep the gate robust by
    // routing to the already-validated cutlass fp8 GEMM (same fp8 math).
    ::vt::MatmulFp8Cutlass(q, out, a_fp8, b_fp8, alpha);
    return;
  }

  // out = alpha * op(weight) @ op(act) + 0 * C; C and D share out's buffer/layout.
  const float beta = 0.0f;
  CheckLt(cublasLtMatmul(ctx.handle, desc.v, &alpha, b_fp8.data, la.v, a_fp8.data, lb.v, &beta,
                         out.data, lc.v, out.data, lc.v, &heur.algo, ctx.workspace, kWorkspaceBytes,
                         s),
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
    RegisterOp(OpId::kMatmulFp8CublasLt, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulFp8CublasLtFn>(&MatmulFp8CublasLtKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
