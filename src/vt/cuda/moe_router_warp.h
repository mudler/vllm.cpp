// vllm.cpp original (vt runtime, inventory deviation §9.1). Portable contract
// for the single-warp MoE router top-k (spec:
// .agents/specs/moe-router-topk-single-warp.md, issue #378). The CUDA kernel
// itself stays in cuda_moe.cu; what lives here is the part that can be WRONG in
// a way no compiler catches -- the lane->expert map and the two register
// reduction trees -- so a host test can execute it without a GPU.
//
// SHAPE port of vLLM's register-resident topkGating
// (csrc/libtorch_stable/moe/topk_softmax_kernels.cu:279-592 @ 555967922): one
// warp per token, no shared memory, no barriers. NOT a math port -- every
// arithmetic decision stays the incumbent block kernel's; see the spec §4 for
// the point-by-point table of where vLLM differs and why we do not follow it.
//
// WHY THE MAP IS WHAT IT IS. The incumbent block reduction is
//
//     for (int s = kBlock / 2; s > 0; s /= 2) {
//       if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
//       __syncthreads();
//     }
//
// over kBlock == 256 lanes (cuda_moe.cu:90-93). Levels s = 128, 64, 32 are all
// multiples of the 32-wide warp, so `t` and `t + s` always have the SAME lane id
// `L = t % 32`: those three levels never cross a lane. They combine exactly the
// entries red[L + 32q] for q in [0, 8), and they combine them by the standard
// halving recursion on q. Levels s = 16..1 run entirely inside warp 0 and are
// what __shfl_down_sync(0xffffffffu, v, s) reproduces.
//
// So if lane L holds slot q == expert (L + 32q) and reduces the slots with the
// SAME halving tree, the result is bit-identical to the block tree by
// structural congruence: identical operands, identical operations, identical
// association. Not by an appeal to associativity -- float addition is not
// associative, which is exactly why the previous row (6a8c5cf9) believed a
// register-resident router was off-limits. It was off-limits for vLLM's OWN
// partition (lane L owns the CONTIGUOUS experts [VPT*L, VPT*L + VPT),
// topk_softmax_kernels.cu:344-346), which does reassociate. A different map
// does not.
//
// Written in the tree's leaf order the offsets are 32 * {0,4,2,6,1,5,3,7}, the
// bit-reversal of {0..7}. The stride form implemented here is the same
// statement and is the one that can be checked by eye against the incumbent
// loop.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define VT_MOE_ROUTER_HD __host__ __device__
// nvcc only. The host build runs -Wall -Wextra -Werror, and -Wall enables
// -Wunknown-pragmas, so a bare `#pragma unroll` here would fail the host TU
// that the portable test compiles.
#define VT_MOE_ROUTER_UNROLL _Pragma("unroll")
#else
#define VT_MOE_ROUTER_HD
#define VT_MOE_ROUTER_UNROLL
#endif

namespace vt::cuda {

// The warp width the map is derived against. The whole derivation above depends
// on the incumbent block's kBlock (256) being a multiple of this, and on the
// three cross-warp levels therefore being lane-local.
inline constexpr int kMoeRouterWarpWidth = 32;

// Warps per CTA, one token per warp. Mirrors the traced upstream instantiation
// topkGating<8, 256, /*WARPS_PER_CTA=*/4, 16, 32, ...> (ROWS_PER_WARP == 1,
// ROWS_PER_CTA == 4, topk_softmax_kernels.cu:311-317).
inline constexpr int kMoeRouterWarpsPerCta = 4;

// THE LANE MAP. Lane `lane` holds, in register slot `slot`, the expert
// `lane + 32 * slot`. Equivalently: slot q of the warp covers the 32
// CONSECUTIVE experts [32q, 32q + 32), so every load is fully coalesced.
//
// This is deliberately NOT vLLM's contiguous `lane * VPT + slot`. That map
// reassociates the softmax sum and is the mutation the host test discriminates
// against; see MoeRouterWarpContiguousExpert below.
VT_MOE_ROUTER_HD inline constexpr int MoeRouterWarpExpert(int lane, int slot) {
  return lane + kMoeRouterWarpWidth * slot;
}

// vLLM's own partition, present ONLY so the test can instantiate it and prove
// the oracle can tell the two apart. Never dispatched.
VT_MOE_ROUTER_HD inline constexpr int MoeRouterWarpContiguousExpert(int lane, int slot,
                                                                   int values_per_thread) {
  return lane * values_per_thread + slot;
}

// Values (experts) per lane for a row of `e` experts, or 0 when the row is not
// eligible for the warp kernel.
//
// Only E in {32, 64, 128, 256} is admitted, because those are the widths whose
// bit-exactness is DERIVED in the spec (§5), one VPT at a time. E > 256 is
// excluded on purpose: above the incumbent's 256-wide block the seed loops at
// cuda_moe.cu:71,83 accumulate SEVERAL experts per thread in ascending order,
// so e[t] is itself an association that would have to be reproduced. Anything
// not derived falls through to the unchanged block kernel.
VT_MOE_ROUTER_HD inline constexpr int MoeRouterWarpValuesPerThread(int64_t e) {
  return e == 32 ? 1 : (e == 64 ? 2 : (e == 128 ? 4 : (e == 256 ? 8 : 0)));
}

// Per-lane sum over the slots, in the incumbent block tree's association.
//
// Congruent to the s = 128, 64, 32 levels of cuda_moe.cu:90-93 restricted to
// one lane. For VPT < 8 the levels the block spent on its zero-padded seeds
// (threads t >= e never enter the loop body at :83, so red[t] keeps its +0.0f)
// are dropped, which is bit-exact because x + (+0.0f) == x for every value
// these seeds can hold: expf yields +0.0f or a positive value and never -0.0f,
// the one float whose sign bit that addition would flip.
//
// The OCCUPIED seeds rest on the same argument, which the spec originally
// glossed. The incumbent's per-thread seed is not the loaded value but
// `float acc = 0.0f; ... acc += ex` (cuda_moe.cu:84-89), i.e. (+0.0f) + E_t,
// while `r[q] = v[q]` below is a bare copy. Those differ only for -0.0f (which
// expf never produces) and for a SIGNALLING NaN, which the addition would quiet
// and the copy would not (expf returns only quiet NaNs). Same contained class as
// the pad leaves above; spec §5.
template <int VPT>
VT_MOE_ROUTER_HD inline float MoeRouterWarpTreeSum(const float (&v)[VPT]) {
  float r[VPT];
  VT_MOE_ROUTER_UNROLL
  for (int q = 0; q < VPT; ++q) r[q] = v[q];
  VT_MOE_ROUTER_UNROLL
  for (int s = VPT / 2; s > 0; s >>= 1) {
    VT_MOE_ROUTER_UNROLL
    for (int q = 0; q < s; ++q) r[q] += r[q + s];
  }
  return r[0];
}

// Per-lane max over the slots, in the incumbent block tree's association.
//
// The fmaxf(-INFINITY, .) seed is applied per element BEFORE the tree because
// that is what the incumbent does (`float m = -INFINITY; ... m = fmaxf(m,
// Load(...))` at cuda_moe.cu:70-71). Verbatim reproduction is the reason, and
// the only reason. An earlier version of this comment claimed the seed "is what
// makes an all-NaN row behave identically"; review FALSIFIED that by mutation.
// Deleting the seed changes ONLY the `mx` intermediate -- the weights and the
// indices stay byte-identical on every case, all-NaN included. fmaxf already
// returns the non-NaN operand, so the tree erases NaN with or without the seed;
// the seed bites only when a lane holds nothing BUT NaN, and then only by making
// mx -INFINITY instead of NaN, which expf(l - mx) washes out (NaN either way ->
// every prob clamped to 0.0f at :100). Do not "simplify" it away: its only guard
// in the gate is the mx intermediate check in
// tests/vt/test_moe_router_warp_map.cpp (spec §5, §8.1).
//
// Dropping the block's -INFINITY pad leaves for VPT < 8 is bit-exact because
// x = fmaxf(x, -INFINITY) for every non-NaN x, including -0.0f (checked
// exhaustively over all 2^32 non-NaN x during review), and after the seed no
// NaN remains.
template <int VPT>
VT_MOE_ROUTER_HD inline float MoeRouterWarpTreeMax(const float (&v)[VPT]) {
  float r[VPT];
  VT_MOE_ROUTER_UNROLL
  for (int q = 0; q < VPT; ++q) r[q] = fmaxf(-INFINITY, v[q]);
  VT_MOE_ROUTER_UNROLL
  for (int s = VPT / 2; s > 0; s >>= 1) {
    VT_MOE_ROUTER_UNROLL
    for (int q = 0; q < s; ++q) r[q] = fmaxf(r[q], r[q + s]);
  }
  return r[0];
}

// VT_MOE_ROUTER_WARP selector. DEFAULT ON; "0" restores the incumbent block
// kernel for a same-binary A/B. Same spelling as the sibling default-on levers
// in this backend (cuda_paged_attn.cu:2504-2507 Fa2PrefillEnabled), and read
// fresh per launch so an in-process test can flip it.
inline bool MoeRouterWarpFlagIsOn(const char* value) {
  return value == nullptr || value[0] != '0';
}

}  // namespace vt::cuda
