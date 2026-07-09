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
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

// ---- Autotunable tile set (mirrors flashinfer's sm120/121 fp4 cutlass configs)
// The flashinfer FlashInferCutlassNvFp4LinearKernel autotuner (the WHAT vLLM
// selects among with enable_flashinfer_autotune=True on GB10) tunes over 8 CTA
// tile shapes, all with 1x1x1 cluster on this target. Source (flashinfer 0.5.x):
//   - jit/gemm/core.py::gen_gemm_sm120_module_cutlass_fp4 cta_m_n_k_list, and
//   - include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h
//     CutlassFp4GemmRunner::getConfigs() (CutlassTileConfigSM120 enum).
// The 8 (CTA_M, CTA_N, CTA_K) tiles (K already the true cutlass tile-K, i.e. the
// enum's "…64B"/"…128B" doubled to 128/256):
//   128x32x128  128x32x256  128x64x128  128x64x256
//   128x128x128 128x128x256 256x128x128 128x256x128
//
// We instantiate the four N>=128 tiles — {128x128x128, 128x128x256,
// 256x128x128, 128x256x128}. The four narrow-N tiles (N in {32,64}) do NOT
// compile through our block-scaled CollectiveBuilder: the scale-factor TMA needs
// the CTA tile N to equal the SF atom N (= max(128, TileN)), so N<128 trips
// "TMA requires CTA_Tile and SLayout top-level size equivalence" (cute
// copy_traits_sm90_tma.hpp). flashinfer supports them via a different epilogue
// (arch::OpClassTensorOp + explicit TmaWarpSpecialized, ElementC=void) which we
// intentionally do NOT adopt here — it would rework the proven fixed-dispatch
// configs. The N>=128 subset is exactly where the measured 6.1%/M=4096 prefill
// headroom lives (large-M dense projections); narrow-N tiles only help tiny
// M/N where the M<=256 config already ties/beats vLLM. Two of the four
// (128x128x128, 256x128x128) are the fixed-dispatch configs above; the template
// realizes 128x128x256 and 128x256x128 (schedules stay Auto as the two proven
// configs — the builder resolves KernelTmaWarpSpecializedCooperativeBlockScaled
// -Sm120 per tile, matching flashinfer's KernelTmaWarpSpecializedCooperative).
// flashinfer also tunes StreamK + swap_ab variants; we take the DP-persistent
// subset (large-M prefill favors the DP scheduler) to bound nvcc time + binary.
template <int TileM, int TileN, int TileK,
          typename Sched = cutlass::gemm::PersistentScheduler>
struct sm120_fp4_tile {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = Sched;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  using PerSmTileShape_MNK = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
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
    RunGemm<typename Fp4GemmSm120<sm120_fp4_config_M256, OutType>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    RunGemm<typename Fp4GemmSm120<sm120_fp4_config_default, OutType>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

// ---- Per-shape tile autotune (VT_FP4_AUTOTUNE, default OFF) -----------------
// Mirrors vLLM's FlashInferCutlassNvFp4LinearKernel autotune (choose_one over
// the tile candidates): on first sight of a (Mbucket,N,K) shape, micro-bench the
// instantiated tiles on the REAL operands and cache the winner keyed by the
// shape. The projection shapes repeat every layer/step, so the one-time tune
// amortizes to ~0. Default OFF; the fixed 2-way M-dispatch above is the
// baseline/fallback (and candidate 0/1 are byte-identical to it). OutType is
// always bf16 here (the epilogue only emits bf16; f32-out projections go through
// a bf16 scratch upstream), so the candidate table is bf16-only.
using Fp4RunFn = void (*)(void*, const void*, const void*, const void*, const void*,
                          const float*, int, int, int, cudaStream_t);

template <typename Config>
void RunTileBf16(void* D, const void* A, const void* B, const void* A_sf, const void* B_sf,
                 const float* alpha, int m, int n, int k, cudaStream_t s) {
  RunGemm<typename Fp4GemmSm120<Config, cutlass::bfloat16_t>::Gemm>(D, A, B, A_sf, B_sf, alpha, m,
                                                                    n, k, s);
}

struct Fp4Candidate {
  const char* name;
  Fp4RunFn run;
};

// The candidate set = flashinfer's four N>=128 sm120 tiles. Index 0/1 are the
// fixed-dispatch baselines (identical to the OFF path); 2/3 are the extra tiles.
const std::vector<Fp4Candidate>& Fp4Candidates() {
  static const std::vector<Fp4Candidate> c = {
      {"128x128x128", &RunTileBf16<sm120_fp4_config_M256>},     // baseline for M<=256
      {"256x128x128", &RunTileBf16<sm120_fp4_config_default>},  // baseline for M>256
      {"128x128x256", &RunTileBf16<sm120_fp4_tile<128, 128, 256>>},
      {"128x256x128", &RunTileBf16<sm120_fp4_tile<128, 256, 128>>},
  };
  return c;
}

bool Fp4AutotuneEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP4_AUTOTUNE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

bool Fp4AutotuneVerbose() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP4_AUTOTUNE_VERBOSE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

uint32_t NextPow2M(int m) {
  auto next_pow_2 = [](uint32_t v) {
    if (v <= 1) return 1u;
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
  };
  return std::max<uint32_t>(16u, next_pow_2(static_cast<uint32_t>(m)));
}

struct Fp4ShapeKey {
  uint32_t mp2;
  int n;
  int k;
  bool operator==(const Fp4ShapeKey& o) const { return mp2 == o.mp2 && n == o.n && k == o.k; }
};
struct Fp4ShapeKeyHash {
  size_t operator()(const Fp4ShapeKey& x) const {
    return (std::hash<uint32_t>{}(x.mp2) * 1000003u) ^ (std::hash<int>{}(x.n) * 9176u) ^
           std::hash<int>{}(x.k);
  }
};

constexpr float kFp4Inf = 3.4e38f;

// Time one candidate on the real operands (warmup + iters); kFp4Inf on failure
// (unsupported tile / cutlass error) so the selector skips it. Runs write D
// (garbage between candidates); the caller re-runs the winner for the real out.
float Fp4TimeCandidate(const Fp4Candidate& cand, void* D, const void* A, const void* B,
                       const void* A_sf, const void* B_sf, const float* alpha, int m, int n, int k,
                       cudaStream_t s) {
  constexpr int kWarm = 3, kIter = 10;
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  try {
    for (int i = 0; i < kWarm; ++i) cand.run(D, A, B, A_sf, B_sf, alpha, m, n, k, s);
    Check(cudaEventCreate(&e0), "autotune event create e0");
    Check(cudaEventCreate(&e1), "autotune event create e1");
    Check(cudaEventRecord(e0, s), "autotune record e0");
    for (int i = 0; i < kIter; ++i) cand.run(D, A, B, A_sf, B_sf, alpha, m, n, k, s);
    Check(cudaEventRecord(e1, s), "autotune record e1");
    Check(cudaEventSynchronize(e1), "autotune sync");
    float ms = 0.0f;
    Check(cudaEventElapsedTime(&ms, e0, e1), "autotune elapsed");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / kIter;
  } catch (const std::exception&) {
    if (e0 != nullptr) cudaEventDestroy(e0);
    if (e1 != nullptr) cudaEventDestroy(e1);
    cudaGetLastError();  // clear any sticky launch error from the failed candidate
    return kFp4Inf;
  }
}

// Pick + cache the best tile candidate index for a (Mbucket,N,K) shape.
int Fp4SelectPlan(void* D, const void* A, const void* B, const void* A_sf, const void* B_sf,
                  const float* alpha, int m, int n, int k, cudaStream_t s) {
  const uint32_t mp2 = NextPow2M(m);
  const Fp4ShapeKey key{mp2, n, k};
  static std::mutex mu;
  static std::unordered_map<Fp4ShapeKey, int, Fp4ShapeKeyHash> plans;
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = plans.find(key);
    if (it != plans.end()) return it->second;
  }
  const auto& cands = Fp4Candidates();
  const int baseline = (mp2 <= 256) ? 0 : 1;
  std::vector<float> t(cands.size(), kFp4Inf);
  for (size_t i = 0; i < cands.size(); ++i)
    t[i] = Fp4TimeCandidate(cands[i], D, A, B, A_sf, B_sf, alpha, m, n, k, s);
  int best = baseline;
  for (size_t i = 0; i < cands.size(); ++i)
    if (t[i] < t[best]) best = static_cast<int>(i);
  // Hysteresis: only leave the proven baseline when the winner is clearly faster
  // (>1%). Rejects measurement noise and keeps token output stable run-to-run
  // (the 16/16 token-exact gate); the measured headroom is ~6.1%, so a real
  // winner clears 1% easily.
  const int chosen = (t[best] < t[baseline] * 0.99f) ? best : baseline;
  if (Fp4AutotuneVerbose()) {
    fprintf(stderr,
            "[VT_FP4_AUTOTUNE] M=%d(mp2=%u) N=%d K=%d -> %s (%.1f us); baseline %s (%.1f us)\n", m,
            mp2, n, k, cands[static_cast<size_t>(chosen)].name, t[static_cast<size_t>(chosen)] * 1000.0f,
            cands[static_cast<size_t>(baseline)].name, t[static_cast<size_t>(baseline)] * 1000.0f);
  }
  {
    std::lock_guard<std::mutex> lk(mu);
    plans.emplace(key, chosen);
  }
  return chosen;
}

// bf16-out fp4 GEMM entry: fixed dispatch (default) or per-shape autotuned tile.
void Fp4GemmRunBf16(void* D, const void* A, const void* B, const void* A_sf, const void* B_sf,
                    const float* alpha, int m, int n, int k, cudaStream_t s) {
  if (!Fp4AutotuneEnabled()) {
    Fp4GemmDispatch<cutlass::bfloat16_t>(D, A, B, A_sf, B_sf, alpha, m, n, k, s);
    return;
  }
  const int idx = Fp4SelectPlan(D, A, B, A_sf, B_sf, alpha, m, n, k, s);
  Fp4Candidates()[static_cast<size_t>(idx)].run(D, A, B, A_sf, B_sf, alpha, m, n, k, s);
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

  Fp4GemmRunBf16(d_out, a_packed.data, b_packed.data, a_sf_sw.data, b_sf_sw.data, d_alpha, m, n, k,
                 s);

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
