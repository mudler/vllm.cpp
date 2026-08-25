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
// QuantFp8Static, the static per-tensor activation quant that mirrors vLLM's
// static_scaled_fp8_quant (is_scale_inverted=False: x/input_scale, clamp, RNE
// hardware cvt) — which lives in `src/vt/cuda/cuda_quant_fp8.cu` and is compiled
// UNCONDITIONALLY for CUDA, because it needs no cutlass and this TU's arch gate
// was silently withholding it from every other CUDA arch (issue #960).
// See .agents/specs/cutlass-dropin-feasibility.md.
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

#include "vt/cuda/fp8_per_tensor_dispatch.h"
#include "vt/cuda/graph_safe_scratch.h"
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
    // RETIRE the old block instead of freeing it: this per-stream cutlass FP8 GEMM
    // workspace pointer is baked into the captured pure-decode CUDA graph. A later,
    // larger forward (a bigger co-scheduled prefill or a larger decode batch — only
    // at concurrency > 1) grows the workspace; freeing the old block would dangle the
    // captured graph's pointer → illegal memory access on the next replay. See
    // graph_safe_scratch.h.
    RetireGraphScratch(*buf);
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

// ---- vLLM sm120 fp8 configs (scaled_mm.cuh cutlass_3x_gemm_sm120 +
// scaled_mm_sm120_fp8_dispatch.cuh), raw-pointer surface. -------------------
// All four of upstream's rungs, `scaled_mm_sm120_fp8_dispatch.cuh` at the pin
// `5559679229`:
//
//   M<=16   sm120_fp8_config_M16     :127-138  16x64x128,  EpilogueTile 16x32
//   M<=32   sm120_fp8_config_M32     :112-123  32x64x128,  EpilogueTile 32x32
//   M<=256  sm120_fp8_config_M64     :94-108   64x64x128,  EpilogueTile auto
//   else    sm120_fp8_config_default :81-90    128x128x128, EpilogueTile auto
//
// The three small rungs all use `KernelTmaWarpSpecializedPingpong` because the
// "SM120 Cooperative kernel requires Tile M >= 128; for smaller tiles use
// Pingpong" (upstream's own comment at :96-98).
//
// THE TWO SMALL-M RUNGS WERE MISSING FROM THIS FILE UNTIL #1866, on the
// recorded ground that they "are perf-only for tiny M and are covered correctly
// (predicated) by the M64 pingpong tile". Both halves of that were true and the
// conclusion was still wrong, because TINY M IS DECODE: every M from 1 to 256
// took the 64-row tile, so a batch-1 step computed a 64-row tile for one row
// and #1857's 9-row spec-decode verify computed one for nine. A predicated tile
// gets the VALUE right, which is exactly why nothing in the correctness suite
// could see it. See .agents/specs/perf-fp8-small-m-dispatch.md.
//
// The two small rungs need an explicit CUTLASS `EpilogueTile` where the other
// two take `EpilogueTileAuto`; upstream carries that as a separate wrapper
// (`cutlass_3x_gemm_sm120_custom`, :18-77) whose ONLY difference from the plain
// one is that parameter, so here it is one more `Config` member instead of a
// second template.
struct sm120_fp8_config_default {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
};

struct sm120_fp8_config_M64 {
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _64, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
};

struct sm120_fp8_config_M32 {
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_32, _64, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using EpilogueTile = Shape<_32, _32>;
};

struct sm120_fp8_config_M16 {
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_16, _64, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using EpilogueTile = Shape<_16, _32>;
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

  // The EpilogueTile comes from the Config, not from `EpilogueTileAuto`: the
  // M16/M32 rungs pin 16x32 / 32x32 exactly as upstream's
  // `cutlass_3x_gemm_sm120_custom` does, and the other two carry
  // `EpilogueTileAuto` in the same slot, so this line is unchanged for them.
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          typename Config::EpilogueTile, ElementAccumulator,
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

// Dispatch by M (vLLM cutlass_gemm_sm120_fp8_dispatch, :155-179). OutType = bf16.
//
// The ladder itself lives in `vt/cuda/fp8_per_tensor_dispatch.h` and this is a
// plain switch over its answer. That split is the point: the boundaries are
// what can be wrong here, a wrong boundary produces a SLOW rather than a WRONG
// answer, and nothing that runs on a host with no GPU could otherwise see one.
// `tests/vt/test_fp8_per_tensor_dispatch.cpp` gates them by value.
template <typename OutType>
void Fp8GemmDispatch(void* D, const void* A, const void* B, const float* alpha, int m, int n, int k,
                     cudaStream_t stream) {
  const Fp8PerTensorConfig config =
      Fp8Sm120ConfigForM(static_cast<int64_t>(m), Fp8CutlassSmallMEnabled());
  Fp8PerTensorCountDispatch(config);
  switch (config) {
    case Fp8PerTensorConfig::kM16:
      return RunGemm<typename Fp8GemmSm120<sm120_fp8_config_M16, OutType>::Gemm>(
          D, A, B, alpha, m, n, k, stream);
    case Fp8PerTensorConfig::kM32:
      return RunGemm<typename Fp8GemmSm120<sm120_fp8_config_M32, OutType>::Gemm>(
          D, A, B, alpha, m, n, k, stream);
    case Fp8PerTensorConfig::kM64:
      return RunGemm<typename Fp8GemmSm120<sm120_fp8_config_M64, OutType>::Gemm>(
          D, A, B, alpha, m, n, k, stream);
    case Fp8PerTensorConfig::kDefault:
    case Fp8PerTensorConfig::kCount:
      break;
  }
  RunGemm<typename Fp8GemmSm120<sm120_fp8_config_default, OutType>::Gemm>(D, A, B, alpha, m, n, k,
                                                                         stream);
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

// ---- Static per-tensor fp8 activation quant: NOT HERE ANY MORE (issue #960) --
// `QuantFp8Static`'s CUDA kernel used to live below this line, and that was the
// defect. This TU is compiled ONLY when `VT_CUTLASS_FP8_ARCHS` is non-empty, so
// a kernel with no cutlass dependency whatsoever inherited cutlass's arch set
// and `OpId::kQuantFp8Static` went UNREGISTERED for `DeviceType::kCUDA` on every
// other CUDA arch — where the resolver then fell through to the portable CPU
// reference tier and dereferenced device pointers (SIGSEGV; #844 is the same
// defect from the fallback's end). It now lives in
// `src/vt/cuda/cuda_quant_fp8.cu`, which is in the unconditional
// `if(VLLM_CPP_CUDA)` source list. Do not move it back:
// `scripts/check-cuda-op-arch-gate.py` fails if you do.

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulFp8Cutlass, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulFp8CutlassFn>(&MatmulFp8CutlassKernelCuda)));
  }
};
Registrar g_registrar;

}  // namespace
}  // namespace vt::cuda
