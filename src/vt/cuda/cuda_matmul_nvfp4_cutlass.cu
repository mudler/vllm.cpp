// vllm.cpp — cutlass sm120a NVFP4 block-scaled fp4xfp4 GEMM drop-in.
//
// This is a 1:1 lift of vLLM's `cutlass_scaled_fp4_mm_sm120a`
// (csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu @
// e24d1b24) — the near-peak Blackwell/GeForce block-scaled fp4 GEMM (cutlass
// example 79b), the kernel vLLM selects on GB10/sm_121 for W4A4 NVFP4. The only
// change is the host surface: torch::stable::Tensor -> vt::Tensor (data_ptr ->
// .data, torch::stable::empty(workspace)/DeviceGuard -> cudaMallocAsync + our
// stream). The GEMM math and the CollectiveBuilder config are verbatim.
//
// Isolated TU (heavy cutlass templates, ~34s compile) — built only for
// sm_12{0,1}a. Pairs with SwizzleBlockscale (below), the lift of vLLM's
// swizzle_blockscale (nvfp4_utils.py:13-53) that lays the linear fp8 block
// scales into the atom layout Sm1xxBlkScaledConfig::tile_atom_to_shape_SF{A,B}
// reads. See .agents/cutlass-dropin-feasibility.md and qwen27b-w4a4-notes.md §7.

#include <cuda_bf16.h>
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
    throw std::runtime_error(std::string("vt cuda: matmul_nvfp4_cutlass: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Persistent per-stream GEMM scratch (alpha scalar + grown-on-demand cutlass
// workspace + bf16 output staging). Replaces the per-call cudaMallocAsync /
// cudaFreeAsync churn (up to 3 async allocs + frees PER GEMM). Reuse is safe
// under the forward's single-stream ordering: a buffer handed to one GEMM is
// fully consumed before the next GEMM on the SAME stream is issued. Buffers grow
// monotonically and leak at process exit. VT_CUTLASS_NOPOOL=1 restores per-call.
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

void* EnsureScratch(void** buf, size_t* have, size_t need, cudaStream_t s, const char* what) {
  if (need > *have) {
    if (*buf != nullptr) Check(cudaFreeAsync(*buf, s), "cudaFreeAsync scratch grow");
    Check(cudaMallocAsync(buf, need, s), what);
    *have = need;
  }
  return *buf;
}

float* PersistentAlpha(cudaStream_t s) {
  StreamScratch& sc = ScratchFor(s);
  if (sc.alpha == nullptr) Check(cudaMallocAsync(&sc.alpha, sizeof(float), s), "cudaMallocAsync alpha");
  return sc.alpha;
}

#define VT_CUTLASS_CHECK(status)                                                    \
  do {                                                                              \
    cutlass::Status s_ = (status);                                                  \
    if (s_ != cutlass::Status::kSuccess) {                                          \
      throw std::runtime_error(std::string("vt cuda: matmul_nvfp4_cutlass: cutlass ") + \
                               cutlassGetStatusString(s_));                         \
    }                                                                               \
  } while (0)

// ---- vLLM Fp4GemmSm120 collective config (verbatim, :53-127) ---------------
struct sm120_fp4_config_M256 {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = void;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

struct sm120_fp4_config_default {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = cutlass::gemm::PersistentScheduler;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};

// ---- Alternative SM120 blockscaled configs for the VT_FP4_GEMM_CFG autotune
// sweep (perf lever; the 27B/35B prefill fp4 W4A4 GEMM is the single biggest
// prefill bucket) ------------------------------------------------------------
// Selected at runtime for the LARGE-M (prefill) branch only; the small-M
// (decode) branch stays on sm120_fp4_config_M256 unchanged. Every alternative
// keeps the collective's numerics-relevant knobs IDENTICAL to the default —
// same ClusterShape<1,1,1>, same K-tile 128, same
// KernelScheduleAuto/EpilogueScheduleAuto, same alpha / scale-swizzle /
// epilogue plumbing — and differs ONLY in the (M,N) tile shape and/or the tile
// scheduler. The sm120 block-scaled fp4 MMA accumulates over K in the same
// order for every output element regardless of the M/N tiling (there is no
// split-K here: the tile scheduler hands each CTA whole output tiles), so
// cfg 1..4 are byte-for-byte equal to cfg 0 in result — only speed differs.
//
// Constraints grounded in flashinfer's sm120 dense block-scaled GEMM
// (dense_blockscaled_gemm_sm120_b12x.py): SM120/GB10 GeForce has NO
// tcgen05 / 2-CTA / multi-cluster tensor MMA, so ClusterShape is ALWAYS
// <1,1,1>; tile_k is fixed at 128 (= sf_vec_size*8); tile_m/tile_n must be
// divisible by 64. Candidate (M,N) tiles mirror flashinfer's own SM120 tile
// set (_SM100_MMA_TILER_MN_CANDIDATES, cluster forced to (1,1) on sm120):
// 128x128 is flashinfer's large-M default (_select_default_sm120_mma_tiler
// returns (128,128) for m>128), 128x256 the big-N tile for the wide gate/up
// projections (N=17408). 256x256 is intentionally NOT offered: the sm120
// cooperative kernel spreads a tile_m*tile_n f32 accumulator over 256 MMA
// threads, so 256x256 = 256 regs/thread exceeds the ~232-reg budget and would
// fail can_implement.

// cfg 1: flashinfer's SM120 large-M default tile (128x128), persistent sched.
struct sm120_fp4_config_alt1 {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = cutlass::gemm::PersistentScheduler;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

// cfg 2: big-N tile (128x256) for the wide gate/up projections, persistent.
struct sm120_fp4_config_alt2 {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = cutlass::gemm::PersistentScheduler;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _256, _128>;
  using PerSmTileShape_MNK = Shape<_128, _256, _128>;
};

// cfg 3: default tile (256x128) with the NON-persistent (data-parallel) tile
// scheduler — a scheduler A/B against cfg 0 at the same tile.
struct sm120_fp4_config_alt3 {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = void;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};

// cfg 4: flashinfer's 128x128 tile with the NON-persistent tile scheduler.
struct sm120_fp4_config_alt4 {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = void;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

// Runtime selector. VT_FP4_GEMM_CFG in [0..kFp4GemmMaxCfg]: 0 (unset / empty /
// out-of-range) = today's EXACT default path (byte-identical); 1..4 pick the
// alternatives above for the large-M (prefill) branch. Read once, cached.
constexpr int kFp4GemmMaxCfg = 4;
int Fp4GemmCfg() {
  static const int cfg = [] {
    const char* e = std::getenv("VT_FP4_GEMM_CFG");
    if (e == nullptr || e[0] == '\0') return 0;
    const int v = std::atoi(e);
    if (v < 0 || v > kFp4GemmMaxCfg) return 0;  // guard: unknown → default
    return v;
  }();
  return cfg;
}

template <typename Config, typename OutType>
struct Fp4GemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          typename Config::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename Config::KernelSchedule>::CollectiveOp;

  using TileScheduler = typename Config::TileScheduler;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                           CollectiveMainloop, CollectiveEpilogue,
                                           TileScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// ---- args_from_options (vLLM :129-175), raw-pointer surface ----------------
template <typename Gemm>
typename Gemm::Arguments ArgsFromRawPtrs(void* D, const void* A, const void* B, const void* A_sf,
                                         const void* B_sf, const float* alpha, int M, int N,
                                         int K) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_SFB =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A), stride_A, static_cast<ElementB const*>(B), stride_B,
       static_cast<ElementSFA const*>(A_sf), layout_SFA,
       static_cast<ElementSFB const*>(B_sf), layout_SFB},
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
void RunGemm(void* D, const void* A, const void* B, const void* A_sf, const void* B_sf,
             const float* alpha, int M, int N, int K, cudaStream_t stream) {
  Gemm gemm;
  auto arguments = ArgsFromRawPtrs<Gemm>(D, A, B, A_sf, B_sf, alpha, M, N, K);

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

// Dispatch by M (vLLM cutlass_fp4_gemm_dispatch, :202-231). OutType = bf16.
template <typename OutType>
void Fp4GemmDispatch(void* D, const void* A, const void* B, const void* A_sf, const void* B_sf,
                     const float* alpha, int m, int n, int k, cudaStream_t stream) {
  auto next_pow_2 = [](uint32_t v) {
    if (v <= 1) return 1u;
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
  };
  uint32_t const mp2 = std::max<uint32_t>(16u, next_pow_2(static_cast<uint32_t>(m)));
  if (mp2 <= 256) {
    // Small-M (decode) branch — unchanged, not part of the prefill sweep.
    RunGemm<typename Fp4GemmSm120<sm120_fp4_config_M256, OutType>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    return;
  }
  // Large-M (prefill) branch — config selectable via VT_FP4_GEMM_CFG for the
  // autotune sweep. cfg 0 (unset/default) dispatches sm120_fp4_config_default,
  // i.e. the exact prior path (byte-identical result). cfg 1..4 are the same
  // math at a different tile/scheduler (see the config structs above).
  switch (Fp4GemmCfg()) {
    case 1:
      RunGemm<typename Fp4GemmSm120<sm120_fp4_config_alt1, OutType>::Gemm>(
          D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
      return;
    case 2:
      RunGemm<typename Fp4GemmSm120<sm120_fp4_config_alt2, OutType>::Gemm>(
          D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
      return;
    case 3:
      RunGemm<typename Fp4GemmSm120<sm120_fp4_config_alt3, OutType>::Gemm>(
          D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
      return;
    case 4:
      RunGemm<typename Fp4GemmSm120<sm120_fp4_config_alt4, OutType>::Gemm>(
          D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
      return;
    default:
      RunGemm<typename Fp4GemmSm120<sm120_fp4_config_default, OutType>::Gemm>(
          D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
      return;
  }
}

// ---- SwizzleBlockscale kernel (lift of vllm swizzle_blockscale) ------------
// Linear fp8 block scale [rows, cols] -> swizzled [Mp=round_up(rows,128),
// Kp=round_up(cols,4)] in the cutlass atom layout. Mapping (vLLM reshape
// [Mp/128,4,32,Kp/4,4] then permute(0,1,4,3,2,5)):
//   dst = ((((m/128)*(Kp/4) + k/4)*32 + m%32')*4 + (m%128)/32)*4 + k%4
// with m%32' = (m%128)%32. One thread per swizzled (dst) slot; padding = 0.
__global__ void SwizzleBlockscaleKernel(uint8_t* dst, const uint8_t* src, int rows, int cols,
                                        int Mp, int Kp) {
  const int total = Mp * Kp;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int mo = idx / Kp;           // padded row
  const int ko = idx % Kp;           // padded col (group)
  uint8_t val = 0;
  if (mo < rows && ko < cols) val = src[mo * cols + ko];
  const int a0 = mo / 128;
  const int a1 = (mo % 128) / 32;    // in [0,4)
  const int a2 = (mo % 128) % 32;    // in [0,32)
  const int a3 = ko / 4;             // in [0, Kp/4)
  const int a4 = ko % 4;             // in [0,4)
  const int dst_idx = ((((a0 * (Kp / 4) + a3) * 32 + a2) * 4 + a1) * 4 + a4);
  dst[dst_idx] = val;
}

void SwizzleBlockscaleKernelCuda(Queue& q, Tensor& out_swizzled, const Tensor& in_linear) {
  const int rows = static_cast<int>(in_linear.shape[0]);
  const int cols = static_cast<int>(in_linear.shape[1]);
  const int Mp = static_cast<int>(out_swizzled.shape[0]);
  const int Kp = static_cast<int>(out_swizzled.shape[1]);
  cudaStream_t s = AsStream(q);
  const int total = Mp * Kp;
  if (total == 0) return;
  constexpr int kBlock = 256;
  const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
  SwizzleBlockscaleKernel<<<grid, kBlock, 0, s>>>(out_swizzled.Ptr<uint8_t>(),
                                                  in_linear.Ptr<uint8_t>(), rows, cols, Mp, Kp);
  Check(cudaGetLastError(), "swizzle_blockscale kernel launch");
}

// Write a scalar to device WITHOUT a synchronous pageable H2D memcpy (a 4-byte
// H2D copy of `alpha` would serialize every small/decode GEMM). Fully stream-
// ordered, async.
__global__ void SetScalar(float* p, float v) { *p = v; }

// bf16 -> f32 for the f32-output projections (q/k/v/gate/up sinks): the cutlass
// epilogue only emits bf16/half, so an f32 out is produced by casting a bf16
// scratch. Same value the bf16 epilogue rounds to.
__global__ void CastBf16ToF32Kernel(float* out, const __nv_bfloat16* in, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step)
    out[i] = __bfloat162float(in[i]);
}

// ---- MatmulNvfp4Cutlass registered op --------------------------------------
void MatmulNvfp4CutlassKernelCuda(Queue& q, Tensor& out, const Tensor& a_packed,
                                  const Tensor& a_sf_sw, const Tensor& b_packed,
                                  const Tensor& b_sf_sw, float alpha) {
  const int m = static_cast<int>(a_packed.shape[0]);
  const int k = static_cast<int>(a_packed.shape[1] * 2);
  const int n = static_cast<int>(b_packed.shape[0]);
  cudaStream_t s = AsStream(q);

  // alpha lives on device (cutlass epilogue reads alpha_ptr). Pool-backed async
  // alloc + a 1-thread write kernel; no host<->device sync on the hot path.
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

  Fp4GemmDispatch<cutlass::bfloat16_t>(d_out, a_packed.data, b_packed.data, a_sf_sw.data,
                                       b_sf_sw.data, d_alpha, m, n, k, s);

  if (out_f32) {
    const int64_t total = static_cast<int64_t>(m) * n;
    const int blocks = static_cast<int>((total + 255) / 256);
    CastBf16ToF32Kernel<<<blocks, 256, 0, s>>>(
        static_cast<float*>(out.data), static_cast<const __nv_bfloat16*>(bf16_scratch), total);
    if (!pool) Check(cudaFreeAsync(bf16_scratch, s), "cudaFreeAsync bf16 scratch");
  }

  if (!pool) Check(cudaFreeAsync(d_alpha, s), "cudaFreeAsync alpha");
  Check(cudaGetLastError(), "matmul_nvfp4_cutlass launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kSwizzleBlockscale, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<SwizzleBlockscaleFn>(&SwizzleBlockscaleKernelCuda)));
    RegisterOp(OpId::kMatmulNvfp4Cutlass, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MatmulNvfp4CutlassFn>(&MatmulNvfp4CutlassKernelCuda)));
  }
};
Registrar g_registrar;

}  // namespace
}  // namespace vt::cuda
