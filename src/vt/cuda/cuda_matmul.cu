// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// cuBLASLt matmul for the CUDA backend (M0.6, correctness-grade).
// Supported dtype combos: (bf16,bf16)->f32, (bf16,bf16)->bf16, (f32,f32)->f32.
// Any other combo the public-op validation admits (e.g. f16 inputs) throws
// here, naming the combo — never silently truncates. Compute type is
// CUBLAS_COMPUTE_32F with f32 scale; all layouts are CUBLASLT_ORDER_ROW so no
// host-side transposes are needed.
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

// Registers the CUDA matmul during static init (table fill only, no CUDA
// calls — see cuda_ops.cu for the rationale).
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmul, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
