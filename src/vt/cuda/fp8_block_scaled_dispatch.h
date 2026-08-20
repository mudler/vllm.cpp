// vllm.cpp — the HOST-SIDE decisions of the block-scaled FP8 CUTLASS GEMM
// (VT-MATMUL-FP8-BLOCK-CUDA, #1189 milestone M5,
// .agents/specs/vt-matmul-fp8-block-cuda.md).
//
// This header holds ONLY the pure pieces of `vt::MatmulFp8BlockScaled`'s CUDA
// arm: which tile config a given M selects, which shapes the CUTLASS collective
// cannot implement and how it says so, what index the scale layouts CUTLASS
// deduces assign to an element, and the per-config dispatch counters. It has no
// CUDA dependency of any kind. The kernel itself — the heavy CUTLASS templates
// and the registration — lives in `src/vt/cuda/cuda_matmul_fp8_block_cutlass.cu`,
// which is compiled only for `VT_CUTLASS_FP8_ARCHS` (12.0a, 12.1a).
//
// WHY THE SPLIT. It is the arrangement `fp8_plan_cache.h` and
// `graph_safe_scratch.h` beside it already use, and the reason is the same here
// and sharper: no CI job in this tree has a GPU, and the machines that write
// this code mostly have neither a GPU nor `nvcc`. A decision that can only be
// exercised on hardware is a decision nothing gates. The heuristic, the refusal
// predicate and the two layout formulas are exactly the parts that can be wrong
// in a way a successful compile cannot see, so they are the parts that get a
// red-first test running on every machine
// (`tests/vt/test_fp8_block_scaled_dispatch.cpp`). The numerical comparison
// against `vt::MatmulFp8BlockScaled`'s CPU reference arm is the load-bearing
// gate and it needs a device; it lives in
// `tests/vt/test_ops_matmul_fp8_block_cuda.cpp` and it is OWED, not done.
//
// UPSTREAM, pinned vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`:
//   * the M heuristic and the three configs:
//     `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/
//      scaled_mm_blockwise_sm120_fp8_dispatch.cuh` —
//     `cutlass_gemm_blockwise_sm120_fp8_dispatch`,
//     `sm120_blockwise_fp8_config_{default,pingpong,swapab}`.
//   * the host-side reroute away from CUTLASS on a misaligned weight:
//     `vllm/_custom_ops.py` `cutlass_scaled_mm`, `cutlass_compatible_b`.
//   * the 128-hardcoded shape checks: `csrc/.../c3x/scaled_mm_helper.hpp`
//     `dispatch_scaled_mm`.
//   * the deduced scale layouts: CUTLASS 4.5.0
//     `include/cutlass/detail/blockwise_scale_layout.hpp`
//     `Sm1xxBlockwiseScaleConfig::tile_atom_to_shape_SFA` / `..._SFB`.
#ifndef VT_CUDA_FP8_BLOCK_SCALED_DISPATCH_H_
#define VT_CUDA_FP8_BLOCK_SCALED_DISPATCH_H_

#include <atomic>
#include <cstdint>
#include <string>

// The two scale-layout index formulas below are called from a device kernel as
// well as from the host test, ON PURPOSE: the formula a red-first host test
// pins has to be the same object code the kernel runs, or the test pins a copy.
// Annotating exactly those two functions is the whole CUDA-awareness of this
// header; under a plain host compiler the macro is empty and the header needs no
// CUDA toolkit.
#if defined(__CUDACC__)
#define VT_FP8_BLOCK_SCALED_HD __host__ __device__
#else
#define VT_FP8_BLOCK_SCALED_HD
#endif

namespace vt::cuda {

// ---------------------------------------------------------------------------
// The block geometry and the alignment floor
// ---------------------------------------------------------------------------

// The ONLY scale granularity the sm120 blockwise collectives are built for.
// `ScaleGranularity` is a compile-time template parameter of
// `cutlass_3x_gemm_fp8_blockwise`, every upstream sm120 config instantiates it
// at (1,128,128) or (128,1,128), and `dispatch_scaled_mm` hardcodes 128 in its
// `ceil_div` shape checks. A different block size is a different kernel, not a
// different argument.
inline constexpr int kFp8BlockScaledBlockN = 128;
inline constexpr int kFp8BlockScaledBlockK = 128;

// CUTLASS operand alignment, in ELEMENTS. `AlignmentA` and `AlignmentB` are
// `128 / cutlass::sizeof_bits<float_e4m3_t>::value` = 16, and both fp8 operands
// carry K as their contiguous (A, RowMajor) or leading (B, ColumnMajor) extent,
// so a K that is not a multiple of 16 cannot be implemented. `AlignmentD` is
// `128 / cutlass::sizeof_bits<bfloat16_t>::value` = 8 along N. Upstream draws
// the line at 16 on BOTH and reroutes to Triton there rather than refusing
// (`_custom_ops.py`, `cutlass_compatible_b = b.shape[0] % 16 == 0 and
// b.shape[1] % 16 == 0`, where `b` is `B.T` so those are K and N); this tree
// has no Triton block arm to reroute to, so it refuses at the same line.
inline constexpr int kFp8BlockScaledAlignK = 16;
inline constexpr int kFp8BlockScaledAlignN = 16;

// The sm120 collective's OWN floor, which is a different and stricter thing.
// `cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp::can_implement`
// (CUTLASS 4.5.0, and byte-identical at upstream's pinned v4.4.2) adds three
// lines the TMA-alignment checks above do not imply:
//
//     // Ensure complete scale blocks
//     implementable = implementable && (M % ScaleGranularityM == 0);
//     implementable = implementable && (N % ScaleGranularityN == 0);
//     // We expect full tiles in K
//     implementable = implementable && (K % size<2>(TileShape{}) == 0);
//
// The sm90 sibling
// `sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp` has NO such
// block, only four `check_alignment` calls, which is why this is an sm120
// limitation and not a property of block-wise FP8.
//
// WORKING THE THREE CONFIGS THROUGH. The collective reads its granularities off
// the deduced scale layouts -- `ScaleGranularityM = size<0,0>(LayoutSFA)`,
// `ScaleGranularityN = size<0,0>(LayoutSFB)` (the same file, :129-131) -- and
// `RunBlockwiseGemm` hands it `(m,n,k)` unswapped and `(n,m,k)` under swap:
//
//   default  128x128x128, `Sm120BlockwiseScaleConfig<1,128,128>`, prob (m,n,k):
//            M%1 trivial, N%128 -> n % 128 == 0, K%TileShape_K -> k % 128 == 0.
//   pingpong  64x128x128, same granularities, prob (m,n,k): identical.
//   swapab   128x32x128,  `Sm120BlockwiseScaleConfig<128,1,128>`, prob (n,m,k):
//            M%128 -> n % 128 == 0, N%1 trivial, K%TileShape_K -> k % 128 == 0.
//
// So THE SWAP DOES NOT MOVE THE CONSTRAINT. `n` is the constrained extent in all
// three -- as the collective's N unswapped and as its M under swap -- and `m` is
// never constrained by these lines in any of them. TileShape_K is 128 in all
// three configs, so one constant covers K as well.
inline constexpr int kFp8BlockScaledScaleBlockN = 128;
inline constexpr int kFp8BlockScaledTileK = 128;

// ---------------------------------------------------------------------------
// Which tile config runs
// ---------------------------------------------------------------------------

// The three sm120 blockwise configs, in upstream's own order of test.
enum class Fp8BlockScaledConfig : uint8_t {
  kSwapAb = 0,  // 128x32x128, granularity (128,1,128), Cooperative
  kPingpong,    // 64x128x128, granularity (1,128,128), Pingpong
  kDefault,     // 128x128x128, granularity (1,128,128), KernelScheduleAuto
  kCount,
};

// upstream `cutlass_gemm_blockwise_sm120_fp8_dispatch`:
//   bool swap_ab = (M <= 64) || (M % 4 != 0);
//   if (!swap_ab) { if (M <= 256) pingpong; else default; } else swapab;
//
// `M % 4 != 0` IS A CORRECTNESS CONDITION, not a tuning knob, and it is why all
// three configs are ported rather than only the largest. In the unswapped
// configuration the activation scale has `ScaleGranularityM = 1`, so its
// deduced layout has stride 1 along M (see `Fp8BlockScaledActScaleIndex`); a TMA
// load of that stream needs a 16-byte-aligned row, which for f32 is `M % 4 == 0`.
// Swapping A and B moves the activation scale to the SFB stream, where the
// granularity along the swapped problem's N is 1 but the tile is chosen for it.
// Decode is M = 1 and therefore takes the swapped path on every step.
inline Fp8BlockScaledConfig Fp8BlockScaledConfigFor(int64_t m) {
  if (m <= 64 || (m % 4) != 0) return Fp8BlockScaledConfig::kSwapAb;
  if (m <= 256) return Fp8BlockScaledConfig::kPingpong;
  return Fp8BlockScaledConfig::kDefault;
}

inline const char* Fp8BlockScaledConfigName(Fp8BlockScaledConfig c) {
  switch (c) {
    case Fp8BlockScaledConfig::kSwapAb:
      return "swapab_128x32x128";
    case Fp8BlockScaledConfig::kPingpong:
      return "pingpong_64x128x128";
    case Fp8BlockScaledConfig::kDefault:
      return "default_128x128x128";
    case Fp8BlockScaledConfig::kCount:
      break;
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// What this arm refuses, and why each refusal is upstream's own line
// ---------------------------------------------------------------------------

enum class Fp8BlockScaledRefusal : uint8_t {
  kNone = 0,
  kBlockN,  // block_n != 128
  kBlockK,  // block_k != 128
  kAlignK,  // K % 16 != 0 — CUTLASS cannot implement it; upstream reroutes
  kAlignN,  // N % 16 != 0 — same
  kTileK,   // K % 128 != 0 — the sm120 collective expects full tiles in K
  kScaleBlockN,  // N % 128 != 0 — the sm120 collective wants complete scale blocks
};

// A RAGGED BLOCK IS A REFUSAL ON sm120, AND THIS HEADER SAID THE OPPOSITE UNTIL
// [#1437](https://github.com/mudler/vllm.cpp/issues/1437). `tile_atom_to_shape_SFB`
// does size the scale grid with `ceil_div(N, 128)`, so a short final N-block is
// EXPRESSIBLE — but the sm120 collective refuses to implement it anyway, three
// lines further down in `can_implement` (see `kFp8BlockScaledScaleBlockN`
// above). Upstream's own CUTLASS test is `M=32, N=576, K=7168` — 576 is
// 4*128 + 64 — precisely because DSV3's `kv_a_proj_with_mqa` has that shape
// (`tests/kernels/quantization/test_block_fp8.py`,
// `test_w8a8_block_fp8_cutlass_matmul`), and on 2026-08-20 that case threw
// `cutlass Invalid status` on a GB10 while THIS PREDICATE ACCEPTED IT. Upstream
// gates the same module at `get_device_capability() < (9, 0)` only, so cc 12.1
// passes its gate and its test cannot succeed there either; the shape is served
// by the sm90/sm100 collectives, which carry no such constraint.
//
// ORDER. The coarse 16-alignment floor is asked BEFORE the 128 floors even
// though 128 is the stricter number, because the two answer to different
// authorities and a reader deserves the one that applies to them: `% 16` is
// UPSTREAM's line, where vllm reroutes to triton rather than refusing
// (`_custom_ops.py`, `cutlass_compatible_b`), and `% 128` is CUTLASS's sm120
// line, which upstream never reaches because it never asks. Both messages name
// the other, so neither answer sends a reader to a shape that is still refused.
// THE SIGNATURE HAS NO `m`, AND THAT IS A CLAIM ABOUT THE CONFIGS rather than
// about this function. `can_implement` constrains `m` in none of the three —
// the granularity `m` meets is 1 on the unswapped path and 1 under swap, worked
// through at `kFp8BlockScaledScaleBlockN` above — so there is nothing for a
// fourth parameter to ask. A config with `ScaleGranularityM != 1` unswapped, or
// `ScaleGranularityN != 1` swapped, WOULD bind `m`, and this signature would
// then have to change. That condition is not left to this comment:
// `src/vt/cuda/cuda_matmul_fp8_block_cutlass.cu` static_asserts it against the
// three configs' own granularities and TileShapes, so a config that broke it
// fails the build rather than the fleet.
inline Fp8BlockScaledRefusal Fp8BlockScaledRefusalFor(int64_t n, int64_t k, int block_n,
                                                      int block_k) {
  if (block_n != kFp8BlockScaledBlockN) return Fp8BlockScaledRefusal::kBlockN;
  if (block_k != kFp8BlockScaledBlockK) return Fp8BlockScaledRefusal::kBlockK;
  if ((k % kFp8BlockScaledAlignK) != 0) return Fp8BlockScaledRefusal::kAlignK;
  if ((n % kFp8BlockScaledAlignN) != 0) return Fp8BlockScaledRefusal::kAlignN;
  if ((k % kFp8BlockScaledTileK) != 0) return Fp8BlockScaledRefusal::kTileK;
  if ((n % kFp8BlockScaledScaleBlockN) != 0) return Fp8BlockScaledRefusal::kScaleBlockN;
  return Fp8BlockScaledRefusal::kNone;
}

// The message names the dimension, the remainder, and where upstream draws the
// same line — because the CPU arm of this same op ACCEPTS every one of these
// shapes, so a reader who hits this is being told the two arms have different
// domains and needs to know that is upstream's situation too.
inline std::string Fp8BlockScaledRefusalMessage(Fp8BlockScaledRefusal refusal, int64_t n,
                                                int64_t k, int block_n, int block_k) {
  const std::string head = "matmul_fp8_block_scaled: no CUDA kernel for this shape. ";
  switch (refusal) {
    case Fp8BlockScaledRefusal::kNone:
      return std::string();
    case Fp8BlockScaledRefusal::kBlockN:
      return head + "block_n is " + std::to_string(block_n) + " and the sm120 blockwise " +
             "CUTLASS collective is instantiated for 128 only (the scale granularity is a " +
             "compile-time template parameter, and vllm csrc/.../c3x/scaled_mm_helper.hpp " +
             "hardcodes 128 in its shape checks). The CPU reference arm takes any positive " +
             "block_n.";
    case Fp8BlockScaledRefusal::kBlockK:
      return head + "block_k is " + std::to_string(block_k) + " and the sm120 blockwise " +
             "CUTLASS collective is instantiated for 128 only (the scale granularity is a " +
             "compile-time template parameter, and vllm csrc/.../c3x/scaled_mm_helper.hpp " +
             "hardcodes 128 in its shape checks). The CPU reference arm takes any positive " +
             "block_k.";
    case Fp8BlockScaledRefusal::kAlignK:
      return head + "K is " + std::to_string(k) + ", which leaves a remainder of " +
             std::to_string(k % kFp8BlockScaledAlignK) +
             " modulo 16, and both fp8 operands need K aligned to 16 elements " +
             "(cutlass AlignmentA/AlignmentB = 128 / sizeof_bits<e4m3> = 16). vllm refuses " +
             "the same shape one rung higher and reroutes it to triton " +
             "(vllm/_custom_ops.py, cutlass_compatible_b); there is no triton block arm " +
             "here. The CPU reference arm runs this shape. Rounding K up to a multiple of " +
             "16 is not enough on this arch: see the K % 128 refusal below, which is the " +
             "sm120 collective's own full-tile-in-K line.";
    case Fp8BlockScaledRefusal::kAlignN:
      return head + "N is " + std::to_string(n) + ", which leaves a remainder of " +
             std::to_string(n % kFp8BlockScaledAlignN) +
             " modulo 16, and the output needs N aligned to 16 elements " +
             "(vllm/_custom_ops.py, cutlass_compatible_b, requires it of both weight " +
             "extents). Rounding N up to a multiple of 16 is not enough on this arch: see " +
             "the N % 128 refusal below, which is the sm120 collective's own " +
             "complete-scale-block line and is stricter. The CPU reference arm runs this " +
             "shape.";
    case Fp8BlockScaledRefusal::kTileK:
      return head + "K is " + std::to_string(k) + ", which leaves a remainder of " +
             std::to_string(k % kFp8BlockScaledTileK) +
             " modulo 128, and the sm120 blockwise collective expects FULL TILES IN K: " +
             "cutlass gemm/collective/sm120_mma_tma_blockwise_scaling.hpp can_implement " +
             "requires K % size<2>(TileShape{}) == 0, and TileShape K is 128 in all three " +
             "sm120 configs (default 128x128x128, pingpong 64x128x128, swapab 128x32x128), " +
             "so K must be a multiple of 128. THIS IS AN sm120 LIMITATION, NOT A GENERAL " +
             "ONE: the sm90 sibling sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_" +
             "scaling.hpp checks TMA alignment only and takes a ragged final K-block. " +
             "vllm.cpp #1437 has the derivation. This op's own contract tiles K with cdiv " +
             "and the CPU reference arm runs this shape.";
    case Fp8BlockScaledRefusal::kScaleBlockN:
      return head + "N is " + std::to_string(n) + ", which leaves a remainder of " +
             std::to_string(n % kFp8BlockScaledScaleBlockN) +
             " modulo 128, and the sm120 blockwise collective wants COMPLETE SCALE BLOCKS: " +
             "cutlass gemm/collective/sm120_mma_tma_blockwise_scaling.hpp can_implement " +
             "requires N % ScaleGranularityN == 0 and M % ScaleGranularityM == 0, so N must " +
             "be a multiple of 128. THE SWAP DOES NOT MOVE THIS: N is the constrained " +
             "extent in all three sm120 configs — the collective's N at granularity 128 on " +
             "the unswapped path, and its M at granularity 128 under swap_ab, which is the " +
             "path every M <= 64 takes. THIS IS AN sm120 LIMITATION, NOT A GENERAL ONE: the " +
             "sm90 sibling sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp " +
             "checks TMA alignment only and takes a ragged final scale block. It refuses " +
             "DSV3's kv_a_proj_with_mqa, N = 576 = 4*128 + 64, which is upstream's own " +
             "cutlass test case (tests/kernels/quantization/test_block_fp8.py, " +
             "test_w8a8_block_fp8_cutlass_matmul) and which this arm therefore cannot serve " +
             "on an sm120 device at all; vllm.cpp #1437 has the derivation. The CPU " +
             "reference arm runs this shape.";
  }
  return head + "unclassified refusal";
}

// ---------------------------------------------------------------------------
// The scale layouts CUTLASS deduces from the PROBLEM SHAPE
// ---------------------------------------------------------------------------
//
// `cutlass_gemm_caller_blockwise` never reads the scale tensors' strides. It
// builds `layout_SFA` / `layout_SFB` from the problem shape through
// `Sm120BlockwiseScaleConfig`, so the memory each pointer refers to has to
// already be in the layout the config deduced. Reading
// `Sm1xxBlockwiseScaleConfig::tile_atom_to_shape_SFA` and `..._SFB` out of
// CUTLASS 4.5.0 `include/cutlass/detail/blockwise_scale_layout.hpp`:
//
//   majorSF == UMMA::Major::MN -> stride ((0,1),(0, ceil_div(extent, vec)))
//   majorSF == UMMA::Major::K  -> stride ((0, ceil_div(K, vecK)),(0,1))
//
// The unswapped config is <1,128,128, MN, K> and the swapped one is
// <128,1,128, K, MN>. Working both through, the ACTIVATION scale lands on the
// MN-major side in BOTH and the WEIGHT scale lands on the K-major side in BOTH:
//
//   activation scale index (row, k_tile) = row + k_tile * M   -> COLUMN-major
//   weight     scale index (n_blk, k_tile) = n_blk * k_tiles + k_tile -> row-major
//
// The weight side is already the checkpoint's layout: `weight_scale_inv` ships
// `[cdiv(N,128), cdiv(K,128)]` row-major and nothing re-lays it out, which is
// why upstream can hand CUTLASS `Bs.T` — a transposed VIEW whose `data_ptr()` is
// `Bs`'s own.
//
// The activation side is not. `vt::QuantFp8Group` emits row-major
// `[M, cdiv(K,128)]` (#1189 M1), which is what the CPU reference arm reads and
// what `dense_fp8_block::MatmulFp8BlockScaledD` allocates. Upstream avoids the
// copy one rung earlier by asking its quantizer for the other layout —
// `CutlassFp8BlockScaledMMKernel` builds `QuantFP8` with
// `column_major_scales=True` — and its own CUTLASS test comments the fact. This
// arm TRANSPOSES instead, into per-stream scratch: `M * cdiv(K,128)` floats,
// 40 of them for a decode step of the target checkpoint's `q_proj`. Changing the
// op's output contract to carry the column-major layout is recorded as owed in
// the row's spec, because two landed rows depend on the current one.

VT_FP8_BLOCK_SCALED_HD inline int64_t Fp8BlockScaledActScaleIndex(int64_t row, int64_t k_tile,
                                                                    int64_t m) {
  return row + k_tile * m;
}

VT_FP8_BLOCK_SCALED_HD inline int64_t Fp8BlockScaledWeightScaleIndex(int64_t n_block,
                                                                     int64_t k_tile,
                                                                     int64_t k_tiles) {
  return n_block * k_tiles + k_tile;
}

// ---------------------------------------------------------------------------
// The dispatch counter
// ---------------------------------------------------------------------------
//
// #1189's gate design records why a counter and not only a value comparison: a
// x1.02 and a x1.10 scale perturbation on the per-tensor fp8 tower were
// demonstrably REACHED and still produced 16/16 identical tokens
// (`tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp`). A token gate
// therefore cannot say whether anything ran here, and a silent dequant to bf16
// is numerically BETTER than the quantized path, so it is invisible to every
// value comparison in this tree. The counter is not: it advances only from
// inside the kernel, only after cutlass reports success, and it names WHICH
// config ran — so a heuristic that quietly stopped selecting `swapab` for M = 1
// is visible without a second hardware run to find it.
//
// `dense_fp8_block::BlockGemmCount` answers the sibling question one layer up
// (did the block-wise LINEAR path run at all) and is unaffected by this.
struct Fp8BlockScaledStats {
  uint64_t swap_ab = 0;
  uint64_t pingpong = 0;
  uint64_t deflt = 0;
  uint64_t refused = 0;

  uint64_t dispatched() const { return swap_ab + pingpong + deflt; }
};

namespace detail {
inline std::atomic<uint64_t>* Fp8BlockScaledCounters() {
  // One array of four across every translation unit: a function-local static in
  // an inline function is one object ([basic.def.odr]). Index 0..2 are the
  // configs in `Fp8BlockScaledConfig` order, index 3 is the refusals.
  static std::atomic<uint64_t> counters[4] = {};
  return counters;
}
}  // namespace detail

inline void Fp8BlockScaledCountDispatch(Fp8BlockScaledConfig config) {
  const auto slot = static_cast<size_t>(config);
  if (slot >= static_cast<size_t>(Fp8BlockScaledConfig::kCount)) return;
  detail::Fp8BlockScaledCounters()[slot].fetch_add(1, std::memory_order_relaxed);
}

inline void Fp8BlockScaledCountRefusal() {
  detail::Fp8BlockScaledCounters()[3].fetch_add(1, std::memory_order_relaxed);
}

// Read as ONE snapshot. A test that read the four counters through four calls
// could attribute a sibling's increment to its own call; reading them together
// and differencing two snapshots cannot.
inline Fp8BlockScaledStats Fp8BlockScaledStatsSnapshot() {
  auto* c = detail::Fp8BlockScaledCounters();
  Fp8BlockScaledStats s;
  s.swap_ab = c[0].load(std::memory_order_relaxed);
  s.pingpong = c[1].load(std::memory_order_relaxed);
  s.deflt = c[2].load(std::memory_order_relaxed);
  s.refused = c[3].load(std::memory_order_relaxed);
  return s;
}

}  // namespace vt::cuda

#endif  // VT_CUDA_FP8_BLOCK_SCALED_DISPATCH_H_
