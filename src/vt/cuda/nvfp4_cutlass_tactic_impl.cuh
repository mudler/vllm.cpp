// vllm.cpp — FlashInfer-parity SM12 NVFP4 CUTLASS tactic implementation.
//
// Ported from installed FlashInfer 0.6.12
// `data/include/flashinfer/gemm/fp4_gemm_template_sm120.h:99-195,216-327`
// and `fp4_gemm_cutlass_template_sm120.h:47-147` (Apache-2.0). The host-only
// FlashInfer/Torch wrappers are replaced by LaunchParams; kernel types,
// explicit TMA epilogue, block-scaled cooperative mainloop, swap semantics,
// scheduler selection and workspace contract remain equivalent. PDL is
// deliberately disabled at the eager adapter seam until all local producers
// participate in the same dependency chain (see RunImpl).
#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "vt/cuda/nvfp4_cutlass_tactics.h"

namespace vt::cuda::nvfp4 {
namespace detail {

using namespace cute;

template <int TileM, int TileN, int TileK, bool SwapAB>
struct Fp4GemmSm120 {
  using OutElementType = cutlass::bfloat16_t;
  using ThreadBlockShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  using Arch = cutlass::arch::Sm120;
  using ClusterShape = Shape<_1, _1, _1>;

  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementC = void;
  using LayoutC =
      std::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using FusionOperation =
      cutlass::epilogue::fusion::LinearCombination<OutElementType, float, void, float>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          Arch, cutlass::arch::OpClassTensorOp, ThreadBlockShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
          ElementC, LayoutC, AlignmentC, OutElementType, LayoutC, AlignmentC,
          cutlass::epilogue::TmaWarpSpecialized, FusionOperation>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          Arch, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA, AlignmentA, ElementB,
          LayoutB, AlignmentB, ElementAccumulator, ThreadBlockShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using DefaultKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StaticPersistentScheduler>;
  using StreamKKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using DefaultGemm = cutlass::gemm::device::GemmUniversalAdapter<DefaultKernel>;
  using StreamKGemm = cutlass::gemm::device::GemmUniversalAdapter<StreamKKernel>;
};

template <typename Gemm>
typename Gemm::Arguments PrepareArgs(void* d, const void* a, const void* b, const void* a_sf,
                                     const void* b_sf, const float* alpha, int m, int n, int k) {
  using ScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementD = typename Gemm::ElementD;

  typename Gemm::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.epilogue.thread.alpha_ptr = alpha;
  args.problem_shape = make_shape(m, n, k, 1);
  args.mainloop.ptr_A = static_cast<const cutlass::float_e2m1_t*>(a);
  args.mainloop.ptr_B = static_cast<const cutlass::float_e2m1_t*>(b);
  args.mainloop.ptr_SFA = static_cast<const cutlass::float_ue4m3_t*>(a_sf);
  args.mainloop.ptr_SFB = static_cast<const cutlass::float_ue4m3_t*>(b_sf);
  args.epilogue.ptr_C = static_cast<const void*>(d);
  args.epilogue.ptr_D = static_cast<ElementD*>(d);
  args.mainloop.dA =
      cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {m, k, 1});
  args.mainloop.dB =
      cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {n, k, 1});
  args.epilogue.dC =
      cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {m, n, 1});
  args.epilogue.dD = args.epilogue.dC;
  args.mainloop.layout_SFA = ScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = ScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);
  if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>) {
    args.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>) {
    using RasterOrder = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = RasterOrder::Heuristic;
  }
  args.hw_info.cluster_shape = dim3(1, 1, 1);
  args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  return args;
}

template <typename Gemm>
size_t WorkspaceSizeImpl(void* d, const void* a, const void* b, const void* a_sf,
                         const void* b_sf, const float* alpha, int m, int n, int k) {
  Gemm gemm;
  auto args = PrepareArgs<Gemm>(d, a, b, a_sf, b_sf, alpha, m, n, k);
  return Gemm::get_workspace_size(args);
}

template <typename Gemm>
void RunImpl(void* d, const void* a, const void* b, const void* a_sf, const void* b_sf,
             const float* alpha, int m, int n, int k, void* workspace, size_t workspace_bytes,
             cudaStream_t stream) {
  Gemm gemm;
  auto args = PrepareArgs<Gemm>(d, a, b, a_sf, b_sf, alpha, m, n, k);
  const size_t required = Gemm::get_workspace_size(args);
  if (required > workspace_bytes) {
    throw std::runtime_error("NVFP4 tactic workspace is undersized: required " +
                             std::to_string(required) + ", got " +
                             std::to_string(workspace_bytes));
  }
  const cutlass::Status can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("NVFP4 tactic cannot implement shape: ") +
                             cutlassGetStatusString(can_implement));
  }
  const cutlass::Status initialized = gemm.initialize(args, workspace, stream);
  if (initialized != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("NVFP4 tactic initialization failed: ") +
                             cutlassGetStatusString(initialized));
  }
  // The eager C++ adapter produces activation FP4, swizzled scales and a device
  // alpha immediately before this call. Unlike FlashInfer's runner boundary,
  // those producers do not participate in a complete PDL dependency chain.
  // Ordinary stream serialization is asynchronous to the host and is required
  // for correctness (the exact real-shape raw A/B gate catches stale reads).
  const cutlass::Status ran = gemm.run(args, workspace, stream, nullptr, false);
  if (ran != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("NVFP4 tactic launch failed: ") +
                             cutlassGetStatusString(ran));
  }
}

template <int TileM, int TileN, int TileK, bool SwapAB, bool StreamK>
using ConfiguredGemm = std::conditional_t<
    StreamK, typename Fp4GemmSm120<TileM, TileN, TileK, SwapAB>::StreamKGemm,
    typename Fp4GemmSm120<TileM, TileN, TileK, SwapAB>::DefaultGemm>;

template <int TileM, int TileN, int TileK, bool SwapAB, bool StreamK>
size_t WorkspaceSize(const LaunchParams& params) {
  using Gemm = ConfiguredGemm<TileM, TileN, TileK, SwapAB, StreamK>;
  if constexpr (SwapAB) {
    return WorkspaceSizeImpl<Gemm>(params.d, params.b, params.a, params.b_sf, params.a_sf,
                                   params.alpha, params.n, params.m, params.k);
  }
  return WorkspaceSizeImpl<Gemm>(params.d, params.a, params.b, params.a_sf, params.b_sf,
                                 params.alpha, params.m, params.n, params.k);
}

template <int TileM, int TileN, int TileK, bool SwapAB, bool StreamK>
void Run(const LaunchParams& params) {
  using Gemm = ConfiguredGemm<TileM, TileN, TileK, SwapAB, StreamK>;
  if constexpr (SwapAB) {
    RunImpl<Gemm>(params.d, params.b, params.a, params.b_sf, params.a_sf, params.alpha, params.n,
                  params.m, params.k, params.workspace, params.workspace_bytes, params.stream);
  } else {
    RunImpl<Gemm>(params.d, params.a, params.b, params.a_sf, params.b_sf, params.alpha, params.m,
                  params.n, params.k, params.workspace, params.workspace_bytes, params.stream);
  }
}

template <int TileM, int TileN, int TileK, bool SwapAB, bool StreamK>
Candidate MakeCandidate(const TacticDescriptor& descriptor) {
  return Candidate{descriptor, &WorkspaceSize<TileM, TileN, TileK, SwapAB, StreamK>,
                   &Run<TileM, TileN, TileK, SwapAB, StreamK>};
}

}  // namespace detail
}  // namespace vt::cuda::nvfp4
