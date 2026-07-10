// vllm.cpp — cutlass sm120 per-tensor W8A8 fp8 GEMM drop-in (+ static act quant).
//
// This is a 1:1 lift of vLLM's `cutlass_scaled_mm_sm120_fp8`
// (csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8{,_dispatch}
// .cu(h) + scaled_mm.cuh `cutlass_3x_gemm_sm120` @ e24d1b24) — the per-tensor
// fp8 e4m3 W8A8 GEMM vLLM selects on GB10/sm_121 for the 35B's FP8 projections.
// The only changes to the host surface: torch::stable::Tensor -> vt::Tensor
// (data_ptr -> .data, torch workspace/DeviceGuard -> cudaMallocAsync + our
// stream). The GEMM math + CollectiveBuilder config are the vLLM sm120 fp8
// config (RowMajor A, ColumnMajor B, bf16 D, f32 accumulate).
//
// DEVIATION (recorded): vLLM applies the two per-tensor scales through its
// ScaledEpilogue as out = scale_a·(scale_b·acc) (a full col/row-broadcast EVT).
// For PER-TENSOR (scalar) scales this collapses to a single accumulator multiply
// out = alpha·acc, alpha = input_scale·weight_scale — so we fold both scales into
// one host scalar and use cutlass's default LinearCombination epilogue (alpha_ptr
// on device, exactly like the NVFP4 cutlass drop-in). Numerically within fp8
// tolerance of the two-stage form; the checkpoint scales ARE per-tensor.
//
// Isolated TU (heavy cutlass templates) — built only for sm_12{0,1}a. Pairs with
// QuantFp8Static (below), the static per-tensor activation quant that mirrors
// vLLM's static_scaled_fp8_quant (is_scale_inverted=False: x/input_scale, clamp,
// RNE hardware cvt). See .agents/specs/cutlass-dropin-feasibility.md.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "cutlass/cutlass.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include "vt/ops.h"

using namespace cute;

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: matmul_fp8_cutlass: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Persistent per-stream GEMM scratch (alpha scalar + grown-on-demand cutlass
// workspace + bf16 output staging). Replaces the per-call cudaMallocAsync /
// cudaFreeAsync churn (up to 3 async allocs + frees PER GEMM; ~33k over a
// prefill). Reuse is safe under the forward's single-stream ordering: a buffer
// handed to one GEMM is fully consumed (its kernels complete) before the next
// GEMM on the SAME stream is issued, so one persistent buffer per stream never
// aliases live data. Buffers grow monotonically and leak at process exit (like
// the cublasLt workspace). VT_CUTLASS_NOPOOL=1 restores per-call alloc (A/B).
struct StreamScratch {
  float* alpha = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  void* bf16 = nullptr;
  size_t bf16_bytes = 0;
};

bool CutlassPoolEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_CUTLASS_NOPOOL");
    return !(e != nullptr && e[0] == '1');
  }();
  return on;
}

StreamScratch& ScratchFor(cudaStream_t s) {
  static std::mutex mu;
  static std::unordered_map<cudaStream_t, StreamScratch> m;
  std::lock_guard<std::mutex> lk(mu);
  return m[s];
}

// Ensure `*buf` holds >= `need` bytes on `s`; regrow (async) if short. Returns
// the (possibly new) pointer.
void* EnsureScratch(void** buf, size_t* have, size_t need, cudaStream_t s, const char* what) {
  if (need > *have) {
    if (*buf != nullptr) Check(cudaFreeAsync(*buf, s), "cudaFreeAsync scratch grow");
    Check(cudaMallocAsync(buf, need, s), what);
    *have = need;
  }
  return *buf;
}

// Persistent device alpha scalar for stream `s` (allocated once).
float* PersistentAlpha(cudaStream_t s) {
  StreamScratch& sc = ScratchFor(s);
  if (sc.alpha == nullptr) Check(cudaMallocAsync(&sc.alpha, sizeof(float), s), "cudaMallocAsync alpha");
  return sc.alpha;
}

#define VT_CUTLASS_CHECK(status)                                                       \
  do {                                                                                 \
    cutlass::Status s_ = (status);                                                     \
    if (s_ != cutlass::Status::kSuccess) {                                             \
      throw std::runtime_error(std::string("vt cuda: matmul_fp8_cutlass: cutlass ") + \
                               cutlassGetStatusString(s_));                            \
    }                                                                                  \
  } while (0)

// ---- vLLM sm120 fp8 config (scaled_mm.cuh cutlass_3x_gemm_sm120 +
// scaled_mm_sm120_fp8_dispatch.cuh), raw-pointer surface. -------------------
// M>256: KernelScheduleAuto, EpilogueScheduleAuto, 128x128x128 (sm120_fp8_config
// _default). M<=256: KernelTmaWarpSpecializedPingpong, 64x64x128 (config_M64 —
// "SM120 Cooperative kernel requires Tile M >= 128; for smaller tiles use
// Pingpong"). vLLM's M16/M32 custom-EpilogueTile refinements are perf-only for
// tiny M and are covered correctly (predicated) by the M64 pingpong tile.
struct sm120_fp8_config_default {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
};

struct sm120_fp8_config_M64 {
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _64, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
};

template <typename Config, typename OutType>
struct Fp8GemmSm120 {
  using ElementAB = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementAB>::value;  // 16
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementAB>::value;  // 16

  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using TileShape = typename Config::TileShape;
  using ClusterShape = typename Config::ClusterShape;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutD,
          AlignmentD, typename Config::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementAB, LayoutA, AlignmentA, ElementAB,
          LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename Config::KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                           CollectiveMainloop, CollectiveEpilogue,
                                           cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// args_from_options (vLLM c3x::cutlass_gemm_caller), raw-pointer surface. The
// per-tensor scales are folded into the epilogue's alpha_ptr (device scalar).
template <typename Gemm>
typename Gemm::Arguments ArgsFromRawPtrs(void* D, const void* A, const void* B, const float* alpha,
                                         int M, int N, int K) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A), stride_A, static_cast<ElementB const*>(B), stride_B},
      {{},
       static_cast<ElementD const*>(D),
       stride_D,
       static_cast<ElementD*>(D),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha);
  return arguments;
}

template <typename Gemm>
void RunGemm(void* D, const void* A, const void* B, const float* alpha, int M, int N, int K,
             cudaStream_t stream) {
  Gemm gemm;
  auto arguments = ArgsFromRawPtrs<Gemm>(D, A, B, alpha, M, N, K);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  const bool pool = CutlassPoolEnabled();
  if (workspace_size > 0) {
    if (pool) {
      StreamScratch& sc = ScratchFor(stream);
      workspace = EnsureScratch(&sc.workspace, &sc.workspace_bytes, workspace_size, stream,
                                "cudaMallocAsync workspace");
    } else {
      Check(cudaMallocAsync(&workspace, workspace_size, stream), "cudaMallocAsync workspace");
    }
  }
  VT_CUTLASS_CHECK(gemm.can_implement(arguments));
  VT_CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));
  VT_CUTLASS_CHECK(gemm.run(arguments, workspace, stream));
  if (workspace && !pool) Check(cudaFreeAsync(workspace, stream), "cudaFreeAsync workspace");
}

// Dispatch by M (vLLM cutlass_gemm_sm120_fp8_dispatch). OutType = bf16.
template <typename OutType>
void Fp8GemmDispatch(void* D, const void* A, const void* B, const float* alpha, int m, int n, int k,
                     cudaStream_t stream) {
  if (m <= 256) {
    RunGemm<typename Fp8GemmSm120<sm120_fp8_config_M64, OutType>::Gemm>(D, A, B, alpha, m, n, k,
                                                                       stream);
  } else {
    RunGemm<typename Fp8GemmSm120<sm120_fp8_config_default, OutType>::Gemm>(D, A, B, alpha, m, n, k,
                                                                           stream);
  }
}

// alpha lives on device (cutlass epilogue reads alpha_ptr): pool-backed async
// alloc + a 1-thread write; no host<->device sync on the hot path (NVFP4 drop-in
// discipline).
__global__ void SetScalar(float* p, float v) { *p = v; }

// bf16 -> f32 for the f32-output projections (q/k/v, in_proj_qkv/z sinks): the
// cutlass epilogue emits bf16, so an f32 out is the bf16-rounded value cast up.
__global__ void CastBf16ToF32Kernel(float* out, const __nv_bfloat16* in, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step)
    out[i] = __bfloat162float(in[i]);
}

void MatmulFp8CutlassKernelCuda(Queue& q, Tensor& out, const Tensor& a_fp8, const Tensor& b_fp8,
                                float alpha) {
  const int m = static_cast<int>(a_fp8.shape[0]);
  const int k = static_cast<int>(a_fp8.shape[1]);
  const int n = static_cast<int>(b_fp8.shape[0]);
  if (m == 0 || n == 0) return;
  cudaStream_t s = AsStream(q);

  const bool pool = CutlassPoolEnabled();
  float* d_alpha = nullptr;
  if (pool) {
    d_alpha = PersistentAlpha(s);
  } else {
    Check(cudaMallocAsync(&d_alpha, sizeof(float), s), "cudaMallocAsync alpha");
  }
  SetScalar<<<1, 1, 0, s>>>(d_alpha, alpha);

  const bool out_f32 = (out.dtype == DType::kF32);
  void* d_out = out.data;
  void* bf16_scratch = nullptr;
  if (out_f32) {
    const size_t need = static_cast<size_t>(m) * n * sizeof(__nv_bfloat16);
    if (pool) {
      StreamScratch& sc = ScratchFor(s);
      bf16_scratch = EnsureScratch(&sc.bf16, &sc.bf16_bytes, need, s, "cudaMallocAsync bf16 scratch");
    } else {
      Check(cudaMallocAsync(&bf16_scratch, need, s), "cudaMallocAsync bf16 scratch");
    }
    d_out = bf16_scratch;
  }

  Fp8GemmDispatch<cutlass::bfloat16_t>(d_out, a_fp8.data, b_fp8.data, d_alpha, m, n, k, s);

  if (out_f32) {
    const int64_t total = static_cast<int64_t>(m) * n;
    const int blocks = static_cast<int>((total + 255) / 256);
    CastBf16ToF32Kernel<<<blocks, 256, 0, s>>>(
        static_cast<float*>(out.data), static_cast<const __nv_bfloat16*>(bf16_scratch), total);
    if (!pool) Check(cudaFreeAsync(bf16_scratch, s), "cudaFreeAsync bf16 scratch");
  }

  if (!pool) Check(cudaFreeAsync(d_alpha, s), "cudaFreeAsync alpha");
  Check(cudaGetLastError(), "matmul_fp8_cutlass launch");
}

// ---- Static per-tensor fp8 activation quant (vLLM static_scaled_fp8_quant) ---
// out_fp8[i] = fp8_e4m3(clamp(x[i]/input_scale, -448, 448)). __NV_SATFINITE cvt
// saturates == clamp-then-cvt; RNE == vLLM's hardware cvt. Tin f32/bf16.
__device__ __forceinline__ uint8_t F32ToFp8Dev(float f) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3));
}
__device__ inline float LoadIn(const float* p, int64_t i) { return p[i]; }
__device__ inline float LoadIn(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }

template <typename Tin>
__global__ void QuantFp8StaticKernel(uint8_t* out, const Tin* x, float input_scale, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const float inv = 1.0f / input_scale;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step)
    out[i] = F32ToFp8Dev(LoadIn(x, i) * inv);
}

void QuantFp8StaticKernelCuda(Queue& q, Tensor& out_fp8, const Tensor& x, float input_scale) {
  const int64_t n = x.shape[0] * x.shape[1];
  if (n == 0) return;
  cudaStream_t s = AsStream(q);
  const int blocks = static_cast<int>(std::min<int64_t>((n + 255) / 256, 65535));
  switch (x.dtype) {
    case DType::kF32:
      QuantFp8StaticKernel<float><<<blocks, 256, 0, s>>>(out_fp8.Ptr<uint8_t>(), x.Ptr<float>(),
                                                         input_scale, n);
      break;
    case DType::kBF16:
      QuantFp8StaticKernel<__nv_bfloat16><<<blocks, 256, 0, s>>>(
          out_fp8.Ptr<uint8_t>(), x.Ptr<__nv_bfloat16>(), input_scale, n);
      break;
    default: VT_CHECK(false, "cuda quant_fp8_static: unsupported x dtype (f32/bf16 only)");
  }
  Check(cudaGetLastError(), "quant_fp8_static launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulFp8Cutlass, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulFp8CutlassFn>(&MatmulFp8CutlassKernelCuda)));
    RegisterOp(OpId::kQuantFp8Static, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<QuantFp8StaticFn>(&QuantFp8StaticKernelCuda)));
  }
};
Registrar g_registrar;

}  // namespace
}  // namespace vt::cuda
