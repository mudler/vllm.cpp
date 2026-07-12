// Immutable W1 four-candidate implementation used by VT_FP4_FULL_TACTICS=0.
// Keep this separate from the FlashInfer-parity explicit epilogue/scheduler TUs
// so the W2 same-binary fallback preserves the exact pre-W2 kernel semantics.

#include <stdexcept>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "vt/cuda/nvfp4_cutlass_tactics.h"

namespace vt::cuda::nvfp4 {
namespace {

using namespace cute;

struct M256Config {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = void;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape = Shape<_128, _128, _128>;
};

struct DefaultConfig {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = cutlass::gemm::PersistentScheduler;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape = Shape<_256, _128, _128>;
};

template <int TileM, int TileN, int TileK>
struct TileConfig {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = cutlass::gemm::PersistentScheduler;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  using PerSmTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
};

template <typename Config>
struct W1Gemm {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementD = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using Arch = cutlass::arch::Sm120;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          Arch, cutlass::arch::OpClassBlockScaledTensorOp, typename Config::PerSmTileShape,
          typename Config::ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator, ElementC, cutlass::layout::RowMajor,
          128 / cutlass::sizeof_bits<ElementC>::value, ElementD, cutlass::layout::RowMajor,
          128 / cutlass::sizeof_bits<ElementD>::value,
          typename Config::EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          Arch, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, cutlass::layout::RowMajor, 32,
          ElementB, cutlass::layout::ColumnMajor, 32, ElementAccumulator,
          typename Config::MmaTileShape, typename Config::ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename Config::KernelSchedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                       CollectiveEpilogue,
                                                       typename Config::TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename Gemm>
typename Gemm::Arguments Args(const LaunchParams& params) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using ScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {params.m, params.k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {params.n, params.k, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {params.m, params.n, 1});
  auto layout_sfa =
      ScaledConfig::tile_atom_to_shape_SFA(make_shape(params.m, params.n, params.k, 1));
  auto layout_sfb =
      ScaledConfig::tile_atom_to_shape_SFB(make_shape(params.m, params.n, params.k, 1));

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {params.m, params.n, params.k, 1},
      {static_cast<const ElementA*>(params.a), stride_a, static_cast<const ElementB*>(params.b),
       stride_b, static_cast<const cutlass::float_ue4m3_t*>(params.a_sf), layout_sfa,
       static_cast<const cutlass::float_ue4m3_t*>(params.b_sf), layout_sfb},
      {{}, static_cast<const ElementD*>(params.d), stride_d, static_cast<ElementD*>(params.d),
       stride_d}};
  args.epilogue.thread.alpha_ptr = params.alpha;
  return args;
}

template <typename Config>
size_t WorkspaceSize(const LaunchParams& params) {
  using Gemm = typename W1Gemm<Config>::Gemm;
  Gemm gemm;
  auto args = Args<Gemm>(params);
  return gemm.get_workspace_size(args);
}

template <typename Config>
void Run(const LaunchParams& params) {
  using Gemm = typename W1Gemm<Config>::Gemm;
  Gemm gemm;
  auto args = Args<Gemm>(params);
  const size_t required = gemm.get_workspace_size(args);
  if (required > params.workspace_bytes) {
    throw std::runtime_error("W1 NVFP4 tactic workspace is undersized");
  }
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("W1 NVFP4 tactic cannot implement shape: ") +
                             cutlassGetStatusString(status));
  }
  status = gemm.initialize(args, params.workspace, params.stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("W1 NVFP4 tactic initialization failed: ") +
                             cutlassGetStatusString(status));
  }
  status = gemm.run(args, params.workspace, params.stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("W1 NVFP4 tactic launch failed: ") +
                             cutlassGetStatusString(status));
  }
}

template <typename Config>
Candidate MakeW1Candidate(int id) {
  return Candidate{*TacticDescriptorForId(id), &WorkspaceSize<Config>, &Run<Config>};
}

}  // namespace

const CandidateGroup& W1Tactics() {
  static const CandidateGroup group{{
      MakeW1Candidate<M256Config>(kW1TacticIds[0]),
      MakeW1Candidate<DefaultConfig>(kW1TacticIds[1]),
      MakeW1Candidate<TileConfig<128, 128, 256>>(kW1TacticIds[2]),
      MakeW1Candidate<TileConfig<128, 256, 128>>(kW1TacticIds[3]),
  }};
  return group;
}

}  // namespace vt::cuda::nvfp4
