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

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

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
  if (workspace_size > 0) {
    Check(cudaMallocAsync(&workspace, workspace_size, stream), "cudaMallocAsync workspace");
  }
  VT_CUTLASS_CHECK(gemm.can_implement(arguments));
  VT_CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));
  VT_CUTLASS_CHECK(gemm.run(arguments, workspace, stream));
  if (workspace) Check(cudaFreeAsync(workspace, stream), "cudaFreeAsync workspace");
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
    RunGemm<typename Fp4GemmSm120<sm120_fp4_config_M256, OutType>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    RunGemm<typename Fp4GemmSm120<sm120_fp4_config_default, OutType>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
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
  float* d_alpha = nullptr;
  Check(cudaMallocAsync(&d_alpha, sizeof(float), s), "cudaMallocAsync alpha");
  SetScalar<<<1, 1, 0, s>>>(d_alpha, alpha);

  Fp4GemmDispatch<cutlass::bfloat16_t>(out.data, a_packed.data, b_packed.data, a_sf_sw.data,
                                       b_sf_sw.data, d_alpha, m, n, k, s);

  Check(cudaFreeAsync(d_alpha, s), "cudaFreeAsync alpha");
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
