// vllm.cpp — the CUDA arm of `vt::MatmulFp8BlockScaled`: the sm120-family
// block-scaled (fine-grained 128x128) FP8 GEMM whose scales apply in the
// MAINLOOP, once per K-block, into an f32 accumulator.
//
// VT-MATMUL-FP8-BLOCK-CUDA, milestone M5 of
// https://github.com/mudler/vllm.cpp/issues/1189. Spec:
// `.agents/specs/vt-matmul-fp8-block-cuda.md`. The op, its signature and its
// validation are M2's (`770e49486`, `.agents/specs/vt-matmul-fp8-block-ref.md`)
// and are unchanged; this file adds one device arm behind them.
//
// A 1:1 lift of vLLM's `cutlass_scaled_mm_blockwise_sm120_fp8` at the pin
// `5559679229bc961848b121ccdeaa8fa5d79bec98`:
// `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/
//  scaled_mm_blockwise_sm120_fp8_dispatch.cuh` — `cutlass_3x_gemm_fp8_blockwise`,
// the three `sm120_blockwise_fp8_config_*`, `cutlass_gemm_caller_blockwise` and
// `cutlass_gemm_blockwise_sm120_fp8_dispatch`, in that file's own order. The
// changes to the HOST surface only: `torch::stable::Tensor` -> `vt::Tensor`
// (`data_ptr()` -> `.data`), and the torch workspace + `DeviceGuard` -> a
// per-stream grow-only scratch on our stream. The GEMM math, the collective
// builders, the tile shapes, the scale granularities and the M heuristic are
// upstream's, unchanged.
//
// WHY CUTLASS AND NOT DEEPGEMM OR TRITON. Upstream ranks the block-FP8 kernels
// FlashInfer, DeepGEMM, CUTLASS, Marlin, Triton, Humming
// (`vllm/model_executor/kernels/linear/__init__.py`); DeepGEMM is auto-disabled
// for `qwen3_5_text` on device-capability family 120
// (`vllm/utils/deep_gemm.py`), and Marlin is excluded at `cc >= 89`. CUTLASS is
// what runs, and unlike the ACTIVATION QUANT of #1189 M1 it AGREES with the
// reference in both placement and polarity — cutlass 4.5.0's
// `include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp` is
// literally `accum(i) += tmp_accum(i) * tCrScaleAViewAsC(i) *
// tCrScaleBViewAsC(i)`. See `include/vt/ops.h` above
// `vt::MatmulFp8BlockScaled` for the whole chain.
//
// THE SCALES ARE MAINLOOP ARGUMENTS. `ptr_SFA` and `ptr_SFB` below are members
// of `GemmKernel::MainloopArguments`. The epilogue is a plain
// `LinearCombination` with no alpha of its own, because an epilogue has exactly
// ONE degree of freedom per output element and this scheme has `cdiv(K,128)` of
// them. Do not "simplify" the scales into the epilogue: that is the defect
// `tests/vt/test_ops_matmul_fp8_block_cpu.cpp` G4 exists to catch on the CPU
// arm, and it would be just as unrepresentable here.
//
// NOT MEASURED ON HARDWARE. This kernel has never executed. What is established
// is that it compiles for sm_120a/sm_121a in the CI `cuda-fat-build` lane and
// that the host-side decisions in `fp8_block_scaled_dispatch.h` are correct
// against upstream's source. The numerical comparison against the CPU reference
// arm — `tests/vt/test_ops_matmul_fp8_block_cuda.cpp`, upstream's own
// `test_w8a8_block_fp8_cutlass_matmul` ported whole — is written, registered,
// and skips without a device. It is OWED. No speed claim is made anywhere.
//
// Isolated TU (heavy cutlass templates), built only for `VT_CUTLASS_FP8_ARCHS`
// (12.0a, 12.1a), which is also the arch gate: upstream's `enable_sm120_family`
// kernel wrapper is NOT mirrored because our per-source gencode already
// restricts this TU to the sm120 family, exactly as the per-tensor sibling
// `cuda_matmul_fp8_cutlass.cu` does. A CUDA arch outside that cell therefore
// leaves `OpId::kMatmulFp8BlockScaled` unregistered and
// `dense_fp8_block::BlockFp8Runnable` keeps refusing by name — which is the
// honest answer and not the #960/#844 fall-through.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

#include "vt/cuda/fp8_block_scaled_dispatch.h"
#include "vt/cuda/graph_safe_scratch.h"
#include "vt/ops.h"

using namespace cute;

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: matmul_fp8_block_scaled: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

#define VT_CUTLASS_CHECK(status)                                                          \
  do {                                                                                    \
    cutlass::Status s_ = (status);                                                        \
    if (s_ != cutlass::Status::kSuccess) {                                                \
      throw std::runtime_error(std::string("vt cuda: matmul_fp8_block_scaled: cutlass ") + \
                               cutlassGetStatusString(s_));                               \
    }                                                                                     \
  } while (0)

// ---- Per-stream grow-only scratch -----------------------------------------
// The same discipline both sibling cutlass TUs use, and for the same recorded
// reason: a captured pure-decode CUDA graph BAKES the device pointer these
// helpers returned at capture time, so a later larger forward that grows a
// buffer must RETIRE the old block rather than free it (graph_safe_scratch.h).
// Three buffers: the cutlass workspace, the transposed activation scale, and a
// bf16 staging buffer for an f32 `out`.
struct StreamScratch {
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  void* act_scale_t = nullptr;
  size_t act_scale_t_bytes = 0;
  void* bf16 = nullptr;
  size_t bf16_bytes = 0;
};

StreamScratch& ScratchFor(cudaStream_t s) {
  static std::mutex mu;
  static std::unordered_map<cudaStream_t, StreamScratch> m;
  std::lock_guard<std::mutex> lk(mu);
  return m[s];
}

void* EnsureScratch(void** buf, size_t* have, size_t need, cudaStream_t s, const char* what) {
  if (need > *have) {
    RetireGraphScratch(*buf);
    Check(cudaMallocAsync(buf, need, s), what);
    *have = need;
  }
  return *buf;
}

// ---- The activation-scale transpose ---------------------------------------
// `cutlass_gemm_caller_blockwise` does NOT read the scale tensors' strides: it
// deduces `layout_SFA`/`layout_SFB` from the problem shape, so the memory each
// pointer refers to must already be in the deduced layout. That layout is
// COLUMN-major `[M, k_tiles]` for the activation scale (see
// `Fp8BlockScaledActScaleIndex` and the derivation above it), while
// `vt::QuantFp8Group` emits row-major — the layout the CPU reference arm reads
// and `dense_fp8_block::MatmulFp8BlockScaledD` allocates. Upstream sidesteps
// this one rung earlier by building its `QuantFP8` with
// `column_major_scales=True` (`kernels/linear/scaled_mm/cutlass.py`) and says so
// in its own CUTLASS test. We transpose instead; the row's spec records the
// alternative under `## Owed`, because changing the op's output contract would
// change two landed rows' allocations.
//
// UNCONDITIONAL. There is no `k_tiles == 1` fast path, where the two layouts
// coincide and a skipped transpose would be invisible.
__global__ void TransposeActScaleKernel(float* dst, const float* src, int m, int k_tiles) {
  const int64_t total = static_cast<int64_t>(m) * k_tiles;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += step) {
    const int64_t row = i / k_tiles;
    const int64_t k_tile = i - row * k_tiles;
    dst[Fp8BlockScaledActScaleIndex(row, k_tile, m)] = src[i];
  }
}

// bf16 -> f32 for an f32 `out`. The collective is instantiated for bf16 only —
// upstream's entry accepts bf16 and fp16 and nothing wider
// (`vllm/_custom_ops.py`, `cutlass_scaled_mm`) — so an f32 `out` is the
// bf16-rounded value cast up, exactly as the per-tensor sibling does for its f32
// sinks. NOTHING HERE COMPUTES IN F32 BEYOND THE ACCUMULATOR, which is
// upstream's `ElementAccumulator = float`.
__global__ void CastBf16ToF32Kernel(float* out, const __nv_bfloat16* in, int64_t n) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += step)
    out[i] = __bfloat162float(in[i]);
}

// ---- vLLM `cutlass_3x_gemm_fp8_blockwise`, raw-pointer surface -------------
template <class OutType, int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK,
          class MmaTileShape, class ClusterShape, class EpilogueScheduler,
          class MainloopScheduler, bool swap_ab_ = false>
struct BlockwiseFp8GemmSm120 {
  static constexpr bool swap_ab = swap_ab_;
  // Re-exposed so the host-side refusal predicate can be held to THESE numbers
  // at compile time rather than to a comment; see the static_asserts under the
  // three configs below (#1437).
  static constexpr int kScaleGranularityM = ScaleGranularityM;
  static constexpr int kScaleGranularityN = ScaleGranularityN;
  static constexpr int kScaleGranularityK = ScaleGranularityK;
  using MmaTileShapeType = MmaTileShape;
  using ElementAB = cutlass::float_e4m3_t;

  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  // ColumnMajor for B matches the CUTLASS convention: our `b_fp8` is the
  // checkpoint's `[N,K]` row-major bytes, which IS `[K,N]` column-major, so the
  // pointer is handed over verbatim. That is upstream's `B.T` without the view.
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;  // no bias: upstream refuses one on this path
  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  using ScaleConfig = cute::conditional_t<
      swap_ab,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, cute::UMMA::Major::K,
                                                 cute::UMMA::Major::MN>,
      cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, cute::UMMA::Major::MN,
                                                 cute::UMMA::Major::K>>;

  // Deduced, so SFA and SFB cannot be swapped here even when A and B are.
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using ElementScalar = float;
  using DefaultOperation =
      cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC,
                                                   ElementScalar, RoundStyle>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      ElementC, cute::conditional_t<swap_ab, LayoutC_Transpose, LayoutC>, AlignmentC, ElementD,
      cute::conditional_t<swap_ab, LayoutD_Transpose, LayoutD>, AlignmentD, EpilogueScheduler,
      DefaultOperation>::CollectiveOp;

  using CollectiveMainloop = cute::conditional_t<
      swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementB, cute::tuple<LayoutB_Transpose, LayoutSFA>, AlignmentB,
          ElementA, cute::tuple<LayoutA_Transpose, LayoutSFB>, AlignmentA, ElementAccumulator,
          MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB,
          cute::tuple<LayoutB, LayoutSFB>, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler>::CollectiveOp>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue>;
};

// ---- The three configs (upstream's tiles, schedules and granularities) -----
template <typename OutType>
struct Sm120BlockwiseConfigDefault {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = BlockwiseFp8GemmSm120<OutType, 1, 128, 128, TileShape, ClusterShape,
                                     EpilogueSchedule, KernelSchedule>;
};

template <typename OutType>
struct Sm120BlockwiseConfigPingpong {
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = BlockwiseFp8GemmSm120<OutType, 1, 128, 128, TileShape, ClusterShape,
                                     EpilogueSchedule, KernelSchedule>;
};

template <typename OutType>
struct Sm120BlockwiseConfigSwapAb {
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = BlockwiseFp8GemmSm120<OutType, 128, 1, 128, TileShape, ClusterShape,
                                     EpilogueSchedule, KernelSchedule, true>;
};

// ---- The refusal predicate's SIGNATURE, held to these three configs --------
//
// `vt::cuda::Fp8BlockScaledRefusalFor` takes `(n, k, block_n, block_k)`. It has
// NO `m` parameter, so it cannot express a constraint on `m` — and that is only
// correct while `can_implement` asks nothing of `m` in any config this file can
// dispatch to. Until #1437's review that tie was a comment in the header, which
// is exactly the shape of thing that goes stale when a fourth config lands.
//
// The derivation, once: `RunBlockwiseGemm` builds the problem shape `(m,n,k)`
// unswapped and `(n,m,k)` under swap, and
// `sm120_mma_tma_blockwise_scaling.hpp::can_implement` asks
//
//     M % ScaleGranularityM == 0
//     N % ScaleGranularityN == 0
//     K % size<2>(TileShape{}) == 0
//
// so the HOST extent each clause lands on is
//
//     unswapped:  M -> m at ScaleGranularityM,  N -> n at ScaleGranularityN
//     swapped:    M -> n at ScaleGranularityM,  N -> m at ScaleGranularityN
//
// `m` is free exactly when the granularity it meets is 1, and `n` carries the
// 128 in both orientations. `kTileK` is read off the config's OWN `TileShape`
// with the same `size<2>` expression `can_implement` uses, so it cannot agree
// with the header by being copied from it.
//
// A config with `ScaleGranularityM != 1` on the unswapped path — or
// `ScaleGranularityN != 1` on the swapped one — would bind `m`, and this build
// then STOPS HERE rather than shipping a predicate that silently cannot ask.
//
// The same treatment covers the BLOCK GEOMETRY refusals, `kBlockN` and
// `kBlockK`, which rested on prose alone. Those two turn away any checkpoint
// whose `weight_block_size` is not `[128, 128]` BECAUSE the collective is
// instantiated for that geometry and nothing else, so their constants are
// correct only while they equal the configs' own scale-vector sizes.
// `MatmulFp8BlockScaledKernelCuda` computes `k_tiles` from the RUNTIME
// `block_k` while CUTLASS deduces `layout_SFA`/`layout_SFB` from
// `ScaleGranularityK` (`blockwise_scale_layout.hpp`, `ceil_div(K, SFVecSizeK)`),
// and the same holds along N. Let those two numbers drift apart and the
// checkpoint's scale grid is read on a stride it was not written with — wrong
// numbers rather than a refusal, and no token gate can see it.
template <typename Cfg>
struct Sm120BlockwiseConfigBinding {
  using G = typename Cfg::Gemm;
  static constexpr int kNGranularity = G::swap_ab ? G::kScaleGranularityM : G::kScaleGranularityN;
  static constexpr int kMGranularity = G::swap_ab ? G::kScaleGranularityN : G::kScaleGranularityM;
  static constexpr int kKGranularity = G::kScaleGranularityK;
  static constexpr int kTileK = cute::size<2>(typename G::MmaTileShapeType{});
};

using BindDefault = Sm120BlockwiseConfigBinding<Sm120BlockwiseConfigDefault<cutlass::bfloat16_t>>;
using BindPingpong = Sm120BlockwiseConfigBinding<Sm120BlockwiseConfigPingpong<cutlass::bfloat16_t>>;
using BindSwapAb = Sm120BlockwiseConfigBinding<Sm120BlockwiseConfigSwapAb<cutlass::bfloat16_t>>;

static_assert(BindDefault::kMGranularity == 1 && BindPingpong::kMGranularity == 1 &&
                  BindSwapAb::kMGranularity == 1,
              "an sm120 blockwise config now constrains m, and Fp8BlockScaledRefusalFor(n, k, "
              "block_n, block_k) has no m parameter to express it with");
static_assert(BindDefault::kNGranularity == kFp8BlockScaledScaleBlockN &&
                  BindPingpong::kNGranularity == kFp8BlockScaledScaleBlockN &&
                  BindSwapAb::kNGranularity == kFp8BlockScaledScaleBlockN,
              "an sm120 blockwise config constrains n at a granularity "
              "kFp8BlockScaledScaleBlockN does not name");
static_assert(BindDefault::kTileK == kFp8BlockScaledTileK &&
                  BindPingpong::kTileK == kFp8BlockScaledTileK &&
                  BindSwapAb::kTileK == kFp8BlockScaledTileK,
              "an sm120 blockwise config has a TileShape K that kFp8BlockScaledTileK does not "
              "name");
// The block geometry the `kBlockN` / `kBlockK` refusals pin the CHECKPOINT to
// is the geometry these configs are instantiated for. `kTileK` above is a
// different claim: it is the mainloop's tile, not the scale vector, and the two
// are equal here only by arithmetic coincidence.
static_assert(BindDefault::kNGranularity == kFp8BlockScaledBlockN &&
                  BindPingpong::kNGranularity == kFp8BlockScaledBlockN &&
                  BindSwapAb::kNGranularity == kFp8BlockScaledBlockN &&
                  BindDefault::kKGranularity == kFp8BlockScaledBlockK &&
                  BindPingpong::kKGranularity == kFp8BlockScaledBlockK &&
                  BindSwapAb::kKGranularity == kFp8BlockScaledBlockK,
              "an sm120 blockwise config reads the weight scale on a block geometry that "
              "kFp8BlockScaledBlockN / kFp8BlockScaledBlockK do not name, so the block_n / "
              "block_k refusals would admit a checkpoint whose scale grid this kernel then "
              "reads on the wrong stride");

// ---- vLLM `cutlass_gemm_caller_blockwise`, raw-pointer surface -------------
// `a` is the activation `[M,K]` fp8 bytes; `b` is the weight `[N,K]` fp8 bytes
// read as `[K,N]` column-major; `a_scales` is the TRANSPOSED activation scale
// (column-major `[M,k_tiles]`); `b_scales` is the checkpoint's
// `[cdiv(N,128), cdiv(K,128)]` row-major grid, verbatim.
template <typename Gemm>
void RunBlockwiseGemm(void* d_ptr, const void* a_ptr, const void* b_ptr, const float* a_scales,
                      const float* b_scales, int m, int n, int k, cudaStream_t stream) {
  static constexpr bool swap_ab = Gemm::swap_ab;
  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using LayoutSFA = typename Gemm::LayoutSFA;
  using LayoutSFB = typename Gemm::LayoutSFB;
  using ScaleConfig = typename Gemm::ScaleConfig;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using ElementBlockScale = typename Gemm::ElementBlockScale;

  StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(
      StrideC{}, swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  LayoutSFA layout_SFA = swap_ab ? ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, 1))
                                 : ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = swap_ab ? ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m, k, 1))
                                 : ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  typename GemmKernel::MainloopArguments mainloop_args{};
  mainloop_args.layout_SFA = layout_SFA;
  mainloop_args.layout_SFB = layout_SFB;
  if constexpr (swap_ab) {
    mainloop_args.ptr_A = static_cast<ElementAB const*>(b_ptr);
    mainloop_args.dA = b_stride;
    mainloop_args.ptr_B = static_cast<ElementAB const*>(a_ptr);
    mainloop_args.dB = a_stride;
    mainloop_args.ptr_SFA = static_cast<ElementBlockScale const*>(b_scales);
    mainloop_args.ptr_SFB = static_cast<ElementBlockScale const*>(a_scales);
  } else {
    mainloop_args.ptr_A = static_cast<ElementAB const*>(a_ptr);
    mainloop_args.dA = a_stride;
    mainloop_args.ptr_B = static_cast<ElementAB const*>(b_ptr);
    mainloop_args.dB = b_stride;
    mainloop_args.ptr_SFA = static_cast<ElementBlockScale const*>(a_scales);
    mainloop_args.ptr_SFB = static_cast<ElementBlockScale const*>(b_scales);
  }

  auto prob_shape = swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);
  auto* c_ptr = static_cast<ElementD*>(d_ptr);
  typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, c_stride, c_ptr, c_stride};

  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::TileSchedulerArguments scheduler{};
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape,
                                      mainloop_args,
                                      epilogue_args,
                                      hw_info,
                                      scheduler};

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  VT_CUTLASS_CHECK(gemm_op.can_implement(args));

  const size_t workspace_size = gemm_op.get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    StreamScratch& sc = ScratchFor(stream);
    workspace = EnsureScratch(&sc.workspace, &sc.workspace_bytes, workspace_size, stream,
                              "cudaMallocAsync workspace");
  }
  VT_CUTLASS_CHECK(gemm_op.run(args, workspace, stream));
}

// ---- The op ---------------------------------------------------------------
void MatmulFp8BlockScaledKernelCuda(Queue& q, Tensor& out, const Tensor& a_fp8,
                                    const Tensor& a_scale, const Tensor& b_fp8,
                                    const Tensor& b_scale, int block_n, int block_k) {
  const int m = static_cast<int>(a_fp8.shape[0]);
  const int k = static_cast<int>(a_fp8.shape[1]);
  const int n = static_cast<int>(b_fp8.shape[0]);

  // Refused BEFORE anything is allocated or launched, and counted, so a refusal
  // is distinguishable from a dispatch that never happened. `vt::ops.cpp` has
  // already checked ranks, dtypes, contiguity, devices and the two `cdiv` scale
  // shapes; what is left is what THIS kernel cannot implement and the CPU
  // reference arm of the same op can.
  const Fp8BlockScaledRefusal refusal = Fp8BlockScaledRefusalFor(n, k, block_n, block_k);
  if (refusal != Fp8BlockScaledRefusal::kNone) {
    Fp8BlockScaledCountRefusal();
    throw std::runtime_error("vt cuda: " +
                             Fp8BlockScaledRefusalMessage(refusal, n, k, block_n, block_k));
  }
  if (m == 0 || n == 0) return;

  cudaStream_t s = AsStream(q);
  StreamScratch& sc = ScratchFor(s);

  // The activation scale, transposed into the column-major layout the deduced
  // `layout_SF*` expects. `cdiv`, because the op's contract allows a ragged
  // final K-block even though the activation quant upstream demands divisibility
  // (`utils/fp8_utils.py`: `cdiv` on the weight, an assert on the activation).
  const int k_tiles = (k + block_k - 1) / block_k;
  const size_t scale_bytes = static_cast<size_t>(m) * k_tiles * sizeof(float);
  float* a_scale_t = static_cast<float*>(EnsureScratch(&sc.act_scale_t, &sc.act_scale_t_bytes,
                                                       scale_bytes, s,
                                                       "cudaMallocAsync act scale transpose"));
  {
    const int64_t total = static_cast<int64_t>(m) * k_tiles;
    const int blocks = static_cast<int>((total + 255) / 256);
    TransposeActScaleKernel<<<blocks, 256, 0, s>>>(
        a_scale_t, static_cast<const float*>(a_scale.data), m, k_tiles);
  }

  const bool out_f32 = (out.dtype == DType::kF32);
  void* d_out = out.data;
  if (out_f32) {
    const size_t need = static_cast<size_t>(m) * n * sizeof(__nv_bfloat16);
    d_out = EnsureScratch(&sc.bf16, &sc.bf16_bytes, need, s, "cudaMallocAsync bf16 staging");
  }

  using OutType = cutlass::bfloat16_t;
  const Fp8BlockScaledConfig config = Fp8BlockScaledConfigFor(m);
  const float* b_scale_ptr = static_cast<const float*>(b_scale.data);
  switch (config) {
    case Fp8BlockScaledConfig::kSwapAb:
      RunBlockwiseGemm<typename Sm120BlockwiseConfigSwapAb<OutType>::Gemm>(
          d_out, a_fp8.data, b_fp8.data, a_scale_t, b_scale_ptr, m, n, k, s);
      break;
    case Fp8BlockScaledConfig::kPingpong:
      RunBlockwiseGemm<typename Sm120BlockwiseConfigPingpong<OutType>::Gemm>(
          d_out, a_fp8.data, b_fp8.data, a_scale_t, b_scale_ptr, m, n, k, s);
      break;
    case Fp8BlockScaledConfig::kDefault:
    case Fp8BlockScaledConfig::kCount:
      RunBlockwiseGemm<typename Sm120BlockwiseConfigDefault<OutType>::Gemm>(
          d_out, a_fp8.data, b_fp8.data, a_scale_t, b_scale_ptr, m, n, k, s);
      break;
  }
  // AFTER cutlass reported success, so a call that threw does not overstate
  // dispatch. `tests/vt/test_fp8_block_scaled_dispatch.cpp` G5 pins the
  // ordering from the refusal side, which is the half a device-free host can
  // observe.
  Fp8BlockScaledCountDispatch(config);

  if (out_f32) {
    const int64_t total = static_cast<int64_t>(m) * n;
    const int blocks = static_cast<int>((total + 255) / 256);
    CastBf16ToF32Kernel<<<blocks, 256, 0, s>>>(static_cast<float*>(out.data),
                                               static_cast<const __nv_bfloat16*>(d_out), total);
  }
  Check(cudaGetLastError(), "matmul_fp8_block_scaled launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(
        OpId::kMatmulFp8BlockScaled, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<MatmulFp8BlockScaledFn>(&MatmulFp8BlockScaledKernelCuda)));
  }
};
Registrar g_registrar;

}  // namespace
}  // namespace vt::cuda
