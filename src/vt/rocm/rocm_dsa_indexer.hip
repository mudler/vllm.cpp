// ROCm/HIP arm of the DSA "Lightning Indexer" selection pair —
// `vt::DsaIndexerLogits` and `vt::DsaTopkSelect`.
// Row MODEL-MM-QWEN4-EXP, W5 of `.agents/specs/qwen4-exp-rocm-ops.md`,
// issue ISSUE-LOCAL-01M2B8QF2TDRZ7KDA3MHA7KMQP.
//
// SELF-REGISTERING translation unit in the established additive pattern
// (`rocm_qwen4_exp.hip` W1, `rocm_qwen4_exp_ple.hip` W3, `rocm_qwen4_exp_qsa.hip`
// W4): no existing kernel TU and no shared op array is edited, and in particular
// `rocm_ops.hip` is NOT, because that file is a surface every ROCm row writes and
// AGENTS.md "Records" calls such a file a lock. The file name mirrors its CUDA
// sibling `src/vt/cuda/cuda_dsa_indexer.cu` and its CPU sibling
// `src/vt/cpu/cpu_dsa_indexer.cpp` one for one.
//
// ─── TWO ROWS OWN THIS PAIR, AND THE ARM IS LANDED ONCE ──────────────────────
// `.agents/specs/rocm-mla-dsa-ops.md` prices these same two ops as its W2 under
// row `BACKEND-ROCM` / #2715, and `.agents/specs/qwen4-exp-rocm-ops.md` owes them
// as its W5 under `MODEL-MM-QWEN4-EXP`. Both records are correct about the gap
// and neither had claimed it. This file is the ONE arm; the two specs are
// reconciled onto it in the same change rather than a second kernel being
// written. There are two consumers, not two behaviours:
//   - `src/vllm/model_executor/models/qwen4_exp_qsa_block.cpp:391,403` composes
//     the QSA indexer from the pair (and calls `DsaTopkSelect` UNCONDITIONALLY,
//     even at `nb == 0`, because the all-`-1` selection IS upstream's
//     `num_complete_blocks == 0` branch), and
//   - the GLM-5.3 sparse step, whose absence `rocm_ops.hip:370-375` already
//     recorded from the MLA/DSA side.
//
// ─── WHAT vLLM's AMD BACKEND SAYS ABOUT THESE TWO ────────────────────────────
// Spec D3 makes vLLM's first-party AMD backend the mirror wherever it defines
// behaviour, and spec D3e MEASURED that the ENTIRE 268-line divergence between
// `vllm/models/qwen4_exp/amd/ops/qsa.py` and `nvidia/ops/qsa.py` at the pin
// `e126687a9a` lands on exactly these two ops and on nothing else. So unlike W4's
// pair, this file is the one where the mirror had something to say. It was read
// line by line before anything here was written, and spec D3g records the result
// in full. In short, because a reader of this file will ask:
//
// 1. **THE SCORING KERNEL: vLLM's AMD REWRITE CONVERGES ON THE DECOMPOSITION
//    THIS TREE ALREADY HAS, and that is the finding.** `nvidia/`'s
//    `_qsa_mqa_paged_kernel` pads the small head axis to a tensor-core-compatible
//    `MAX_N`, loads Q once as a `[BLOCK_D, MAX_N]` tile and computes
//    `tl.dot(keys, query)` per column tile with `TILES_PER_PROG` software
//    pipelining at `num_warps=2`. `amd/` DELETES `TILES_PER_PROG`, `STAGES` and
//    `MAX_N`, gives each program one column tile of 32, and loops
//    `for head in tl.static_range(0, NUM_HEADS)` accumulating
//    `score += tl.maximum(tl.sum(keys * query[None, :], axis=1), 0.0)` at
//    `num_warps=4`. That is a PER-HEAD SCALAR DOT, a per-head ReLU and a sum over
//    heads, with NO matrix intrinsic anywhere — which is the MFMA-is-CDNA-only
//    constraint answered upstream by not needing a matrix unit at all. Our CUDA
//    donor is already a per-head scalar dot with a per-head ReLU and a sum over
//    heads (`cuda_dsa_indexer.cu:70-150`), and this arm keeps that shape. So the
//    mirror and the donor AGREE on the algorithm, and D3's "where they agree,
//    port ours" applies to the arithmetic; what the mirror settles is the thing
//    D3 exists for, that a tensor-core tiling must NOT be reached for here.
//
//    THREE SUB-DIFFERENCES EXIST AND NONE IS ARITHMETIC. (a) vLLM's QSA indexer
//    is WEIGHT-FREE: it has no per-head `weights` operand and applies one scalar
//    `score /= score_divisor` AFTER the head sum. `vt::DsaIndexerLogits` is the
//    DeepSeek-V4 lightning indexer, whose fold `weights * q_scale *
//    softmax_scale * n_head_scale` (deepseek_v2.py:840) multiplies BEFORE it —
//    and the QSA consumer degenerates onto it exactly, passing an all-ones
//    `weights`, a null `q_scale` and `n_head_scale = 1`
//    (qwen4_exp_qsa_block.cpp:374-390, which argues that equality in its own
//    comments). The op is the more general of the two; nothing is lost. (b)
//    vLLM's AMD arm reduces its head-dim dot with `tl.sum`, a tree, while both
//    of our arms keep the CPU oracle's ASCENDING sequential order — kept for the
//    reason D3a §1 and D3d §2 already ratified twice on this row: the CPU arm is
//    the only oracle that can fail this arm, and adopting a lowering shape would
//    move it away from that oracle while matching nothing. (c) vLLM masks
//    out-of-window columns to `-inf` inside the same store; so do we.
//
// 2. **THE TOP-K: vLLM's AMD divergence is a ROUTE, not a behaviour.**
//    `qsa_select_paged_tokens` gains `current_platform.is_cuda()` in its
//    `use_cooperative_topk` predicate and a third branch calling
//    `ops.top_k_per_row_decode` where neither CUDA kernel exists. All three
//    branches answer the same question — the `block_topk` largest logits of the
//    row — and differ in which selection kernel answers it (`cooperative_topk`,
//    `persistent_topk`, or `sampler.cu:717`'s insertion/radix sort). **What the
//    AMD branch does NOT do is redefine the op**, and two parts of
//    `vt::DsaTopkSelect`'s contract are this tree's and must not move: ties break
//    toward the SMALLER key index, and the emission is ASCENDING BY KEY. Both are
//    load-bearing rather than cosmetic — the tie rule because the ReLU makes an
//    exact `0.0` an ordinary logit value, the ascending emission because it is
//    what makes a FULL selection reproduce dense attention bit for bit inside
//    `vt::MlaDecodeAttention`, whose online softmax reduces in visit order. They
//    are shared by the CPU oracle, the CUDA arm, the 3-million-shape fuzz of
//    `.agents/specs/dsa-topk-bounds.md` and the QSA consumer, and adopting
//    upstream's rank-ordered emission on ONE DEVICE would make this arm answer a
//    DIFFERENT OP. That is the D3a §3 / D3d §1 shape: vLLM does inside its kernel
//    what this tree's op contract puts outside it. Recorded in spec D3g, not
//    resolved silently.
//
// ─── WHAT THE TRANSCRIPTION CHANGED, AND WHY EACH CHANGE ─────────────────────
// The HIP spelling table is W3's and W4's: `cuda_runtime.h` -> `hip/hip_runtime.h`,
// `cuda_bf16.h` -> `hip/hip_bf16.h`, `__nv_bfloat16` -> `__hip_bfloat16`,
// `cudaStream_t` -> `hipStream_t`, `cudaGetLastError` -> `hipGetLastError`,
// `vt::cuda` -> `vt::rocm`; the donor's per-dtype template becomes the `DTag`
// runtime tag its three ROCm siblings already carry. Beyond that, exactly THREE
// things differ from the donor and NONE of them is arithmetic:
//
// 1. **NO CROSS-LANE PRIMITIVE, ANYWHERE — and this one removes a reassociation
//    as well as a width assumption.** The donor reduces the per-head DOT with
//    `__shfl_xor_sync(0xffffffffu, dot, off)` over 16/8/4/2/1, which is a 32-lane
//    assumption wearing a mask argument: correct on gfx1151 (RDNA3.5, wavefront
//    32) and wrong on a 64-lane CDNA part compiled from the same source. W4 set
//    this precedent for the same reason. Here the replacement is not a different
//    reduction, it is NO reduction: **one thread owns a whole head and walks the
//    dot sequentially in ascending `d`**, which is the CPU oracle's own order
//    rather than a tree over it. The parallelism moves to the head axis and the
//    key axis, where `kHeadLanes * kKeyRows = 256` threads are busy at the
//    released 64-head geometry.
// 2. **THE HEAD SUM IS FOLDED BY ONE THREAD IN ASCENDING LANE ORDER**, from a
//    `[kKeyRows][kHeadLanes]` shared array, exactly as the donor folds its
//    per-warp partials in ascending warp order. Deterministic BY CONSTRUCTION:
//    the partition is fixed per thread, there is no atomic, and lane `x` owns
//    heads `x, x + kHeadLanes, ...`. **WHEN `index_n_heads <= kHeadLanes` — which
//    the released 64-head config is — lane order IS head order, so the whole op
//    is bit-identical to the CPU oracle**, dot and head sum alike. Above 64 heads
//    the fold groups `h` with `h + 64` first and the association differs from the
//    host's; the `5e-4` cross-device band is the bar there. The gate MEASURES
//    both arms rather than resting on this paragraph: it reports the differing-
//    BYTE count beside the NMSE at 64 heads and runs a 96-head arm on NMSE.
// 3. **`CUDART_INF_F` -> `__int_as_float(0x7f800000)`.** The donor's `-inf` fill
//    for an out-of-window column comes from a CUDA math-constants header with no
//    HIP spelling. The bit pattern is written directly rather than through a
//    device-library macro so the sentinel cannot depend on a library detail —
//    W4's `nanf("")` change, with the same reasoning. `0x7f800000` IS the f32
//    `+inf` those headers define.
//
// The TOP-K kernel needed none of the three: the donor already reduces through a
// `__shared__` tree with `__syncthreads()` and uses no shuffle at all, and its
// comparator is a TOTAL order ("larger logit wins; on an exact tie the SMALLER
// index wins"), so its answer does not depend on the reduction order in the first
// place. It is transcribed with the HIP spellings and the `-inf` change only.
//
// ─── PRECISION ──────────────────────────────────────────────────────────────
// `vt::DsaIndexerLogits` CONTAINS NO TRANSCENDENTAL. Every operation is a
// multiply, an add, a `max` against zero and a compare; the build pins
// `-ffp-contract=off` for HIP (CMakeLists.txt:414) so nothing re-associates into
// an FMA; and both the dot and (at <= 64 heads) the head sum run in the host's
// ascending order. So this arm is EXPECTED bit-identical to its CPU oracle and
// the gate measures that. `vt::DsaTopkSelect` does no arithmetic at all — every
// output is an INDEX — so its bar is EXACT INTEGER EQUALITY and not a tolerance.
// That is not a tightening chosen to pass: a value band is a discrete selector's
// blind spot, because a wrong-but-nearby index moves no float at all. See the
// gate's own header in tests/vt/test_backend_cross_device.cpp.
//
// ─── NO MFMA, NO DP4A, NO WAVEFRONT ASSUMPTION ──────────────────────────────
// Neither kernel uses a matrix intrinsic, a dot-product intrinsic or a cross-lane
// primitive, and nothing in this file reads `warpSize`. MFMA is CDNA-only and
// RDNA has no equivalent; upstream's own AMD arm reaches for no matrix unit
// either, which is finding 1 above.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/dtype.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType, the op declarations

namespace vt::rocm {
namespace {

void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: dsa_indexer: ") + what + ": " +
                             hipGetErrorString(err));
  }
}

hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }

// Local dtype tag + accessors, the arrangement `rocm_qwen4_exp_ple.hip` and
// `rocm_qwen4_exp_qsa.hip` already carry: a local copy rather than a hoist,
// because hoisting would edit a translation unit several other rows are working
// in — the shared-file lock AGENTS.md "Records" names.
enum class DTag : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

DTag TagOf(DType d, const char* op, const char* what) {
  switch (d) {
    case DType::kF32: return DTag::kF32;
    case DType::kF16: return DTag::kF16;
    case DType::kBF16: return DTag::kBF16;
    default:
      VT_CHECK(false,
               std::string("rocm ") + op + ": unsupported " + what + " dtype (f32/f16/bf16)");
      return DTag::kF32;
  }
}

__device__ inline float LoadAt(const void* p, DTag tag, int64_t i) {
  switch (tag) {
    case DTag::kF32: return static_cast<const float*>(p)[i];
    case DTag::kF16: return __half2float(static_cast<const __half*>(p)[i]);
    default: return __bfloat162float(static_cast<const __hip_bfloat16*>(p)[i]);
  }
}

// The f32 `+inf` bit pattern, spelled directly. See change 3 in the header.
__device__ inline float PosInf() { return __int_as_float(0x7f800000); }

// ─── LOGITS ─────────────────────────────────────────────────────────────────
// Grid (num_tokens, ceil(num_keys / kKeyRows)). Block is
// dim3(kHeadLanes, kKeyRows): thread (x, y) owns key `blockIdx.y * kKeyRows + y`
// and heads `x, x + kHeadLanes, ...` of that key, and walks each head's whole
// `head_dim` dot sequentially in ascending `d`. Lane 0 of each row folds the
// `kHeadLanes` partials in ascending lane order and performs the single store.
//
// `kHeadLanes` is 64 because the released `index_n_heads` is 64: one head per
// lane makes the fold order the host's head order exactly (header, change 2).
// It is also a whole number of wavefronts on BOTH families, 2 on RDNA's 32 and 1
// on CDNA's 64, which is a launch-shape remark and not a correctness premise —
// nothing here reads `warpSize`.
constexpr int kHeadLanes = 64;
constexpr int kKeyRows = 4;

__global__ __launch_bounds__(kHeadLanes * kKeyRows) void DsaIndexerLogitsKernel(
    float* __restrict__ logits, const void* __restrict__ q, DTag q_tag,
    const void* __restrict__ k, DTag k_tag, const void* __restrict__ weights, DTag w_tag,
    const float* __restrict__ q_scale, const int32_t* __restrict__ win_start,
    const int32_t* __restrict__ win_end, int64_t lg_s0, int64_t q_s0, int64_t q_s1,
    int64_t k_s0, int64_t w_s0, int64_t qs_s0, int num_tokens, int num_keys, int n_heads,
    int head_dim, float gfold) {
  const int t = static_cast<int>(blockIdx.x);
  if (t >= num_tokens) return;  // whole block, so no __syncthreads is skipped
  const int x = static_cast<int>(threadIdx.x);
  const int y = static_cast<int>(threadIdx.y);
  const int s = static_cast<int>(blockIdx.y) * kKeyRows + y;

  // `tl.store(logits_ptrs, scores, mask=in_window)`
  // (triton_fp8_mqa_logits.py:155-156) writes only the in-window columns and
  // leaves the rest at the row's pre-filled -inf. Here EACH BLOCK OWNS ITS OWN
  // COLUMNS and writes both cases from the single store below — deliberately,
  // rather than having one block pre-fill the whole row, which would race every
  // other block of the same row. This is the donor's arrangement.
  const int lo = max(0, win_start[t]);
  const int hi = min(num_keys, win_end[t]);
  const bool live = s < num_keys && s >= lo && s < hi;

  float acc = 0.0f;
  if (live) {
    for (int h = x; h < n_heads; h += kHeadLanes) {
      const int64_t q_off = static_cast<int64_t>(t) * q_s0 + static_cast<int64_t>(h) * q_s1;
      const int64_t k_off = static_cast<int64_t>(s) * k_s0;
      // ASCENDING, SEQUENTIAL, ON ONE THREAD — the CPU oracle's own order, and
      // what replaces the donor's 32-lane shuffle tree (header, change 1).
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += LoadAt(q, q_tag, q_off + d) * LoadAt(k, k_tag, k_off + d);
      }
      // `weights * q_scale * softmax_scale * n_head_scale` (deepseek_v2.py:840),
      // associated as the CPU arm associates it: `(w * per_head) * gfold`. A null
      // q_scale is upstream's unquantized arm, where the per-head factor is
      // exactly 1 — and it is the ONE member of the fold a SELECTION could see,
      // the other two being global positive constants.
      const float w = LoadAt(weights, w_tag, static_cast<int64_t>(t) * w_s0 + h);
      const float per_head =
          q_scale != nullptr ? q_scale[static_cast<int64_t>(t) * qs_s0 + h] : 1.0f;
      // `tl.maximum(scores, 0.0)` (:129/:150) — the ReLU is the load-bearing
      // nuance: without it a strongly ANTI-CORRELATED head would pull the logit
      // DOWN instead of contributing nothing. It is also why an exact 0.0 is an
      // ordinary logit value here and the selector's tie rule has to be pinned.
      acc += w * per_head * gfold * (dot > 0.0f ? dot : 0.0f);
    }
  }

  __shared__ float partial[kKeyRows][kHeadLanes];
  partial[y][x] = acc;
  __syncthreads();
  if (x == 0 && s < num_keys) {
    float* row = logits + static_cast<int64_t>(t) * lg_s0;
    if (live) {
      // Ascending lane order, one thread, no atomic — deterministic by
      // construction, and at `n_heads <= kHeadLanes` this IS ascending head
      // order and therefore the host's exact summation.
      float sum = 0.0f;
      for (int i = 0; i < kHeadLanes; ++i) sum += partial[y][i];
      row[s] = sum;
    } else {
      row[s] = -PosInf();
    }
  }
}

// ─── TOP-K ──────────────────────────────────────────────────────────────────
// One block per query row, transcribed from `cuda_dsa_indexer.cu:150-260`. The
// short-context branch is a straight ascending fill. The full branch runs
// upstream's selection semantics without a sort: for each of `topk` output slots
// one thread scans and the block reduces to the best remaining (logit, index)
// pair under "larger logit wins; on an exact tie the SMALLER index wins".
// `topk` is 2048 on the released config and the scan is over the candidate
// range, so this is quadratic in the worst case and DELIBERATELY so — it is a
// REFERENCE-GRADE device kernel whose job is to give the same answer as the CPU
// arm on any input, not to be the shipping selection kernel. That is the donor's
// position and this arm inherits it rather than re-deciding it; a
// threshold-based selection is a later brick and it will be gated against
// exactly this.
constexpr int kTopkThreads = 256;

__global__ __launch_bounds__(kTopkThreads) void DsaTopkSelectKernel(
    int32_t* __restrict__ indices, int32_t* __restrict__ counts,
    const float* __restrict__ logits, const int32_t* __restrict__ win_start,
    const int32_t* __restrict__ win_end, int64_t idx_s0, int64_t lg_s0, int num_keys,
    int topk) {
  const int t = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  int32_t* dst = indices + static_cast<int64_t>(t) * idx_s0;
  const float* row = logits + static_cast<int64_t>(t) * lg_s0;

  // `-1` is the "no token" sentinel the topk buffer is pre-filled with
  // (sparse_attn_indexer.py:431-432).
  for (int i = tid; i < topk; i += kTopkThreads) dst[i] = -1;
  __syncthreads();

  const int lo = max(0, win_start[t]);
  const int hi = min(num_keys, win_end[t]);
  const int n = max(0, hi - lo);

  if (n <= topk) {
    // SHORT CONTEXT: every candidate, ascending. This branch is what makes a
    // sparse layer identical to dense attention while the whole context fits in
    // `index_topk`, and it is ALSO the branch the QSA consumer relies on at
    // `nb == 0`, where every window is empty and this writes the all-`-1`,
    // zero-count selection that IS upstream's `num_complete_blocks == 0` case
    // (modeling_qwen4_exp.py:698-700, qwen4_exp_qsa_block.cpp:403).
    for (int i = tid; i < n; i += kTopkThreads) dst[i] = static_cast<int32_t>(lo + i);
    if (tid == 0) counts[t] = static_cast<int32_t>(n);
    return;
  }

  // FULL TOP-K. The pair (pick_val, pick_idx) is the previous pick under the
  // ordering "(logit desc, index asc)", so a candidate is still available iff it
  // sorts strictly after it. Comparing against the last pick rather than keeping
  // a visited set is what keeps this O(topk * n) with O(1) state.
  __shared__ float best_val[kTopkThreads];
  __shared__ int best_idx[kTopkThreads];
  __shared__ float pick_val;
  __shared__ int pick_idx;
  if (tid == 0) {
    pick_val = PosInf();
    pick_idx = -1;
  }
  __syncthreads();

  for (int slot = 0; slot < topk; ++slot) {
    const float bar_v = pick_val;
    const int bar_i = pick_idx;
    float bv = -PosInf();
    int bi = -1;
    for (int s = lo + tid; s < hi; s += kTopkThreads) {
      const float v = row[s];
      // Strictly after the previous pick in (logit desc, index asc) order.
      const bool avail = bar_i < 0 || v < bar_v || (v == bar_v && s > bar_i);
      if (!avail) continue;
      if (bi < 0 || v > bv || (v == bv && s < bi)) {
        bv = v;
        bi = s;
      }
    }
    best_val[tid] = bv;
    best_idx[tid] = bi;
    __syncthreads();
    // Fixed-shape shared-memory tree, ascending stride, same order every launch.
    // The comparator is a TOTAL order, so the answer does not depend on the
    // order anyway — which is why this reduction needed no change on RDNA while
    // the logits kernel's shuffle tree did.
    for (int stride = kTopkThreads >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float ov = best_val[tid + stride];
        const int oi = best_idx[tid + stride];
        const bool better = oi >= 0 && (best_idx[tid] < 0 || ov > best_val[tid] ||
                                        (ov == best_val[tid] && oi < best_idx[tid]));
        if (better) {
          best_val[tid] = ov;
          best_idx[tid] = oi;
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      pick_val = best_val[0];
      pick_idx = best_idx[0];
      // Written in DESCENDING-RANK order here and sorted into ascending KEY
      // order below. The ascending emission is the op's contract and NOT a
      // cosmetic: it is what makes a full selection reproduce dense attention's
      // reduction order bit for bit.
      dst[slot] = static_cast<int32_t>(best_idx[0]);
    }
    __syncthreads();
  }

  // Ascending key order over the `topk` chosen entries. One thread, an insertion
  // pass — `topk` is small against the candidate scan above and this keeps the
  // ordering rule in one readable place.
  if (tid == 0) {
    for (int i = 1; i < topk; ++i) {
      const int32_t v = dst[i];
      int j = i - 1;
      while (j >= 0 && dst[j] > v) {
        dst[j + 1] = dst[j];
        --j;
      }
      dst[j + 1] = v;
    }
    counts[t] = static_cast<int32_t>(topk);
  }
}

void DsaIndexerLogitsKernelRocm(Queue& q, Tensor& logits, const Tensor& q_states,
                                const Tensor& k, const Tensor& weights,
                                const Tensor& win_start, const Tensor& win_end,
                                const DsaIndexerLogitsArgs& args) {
  static constexpr const char* kOp = "dsa_indexer_logits";
  const int64_t T = q_states.shape[0];
  const int64_t H = q_states.shape[1];
  const int64_t D = q_states.shape[2];
  const int64_t S = k.shape[0];
  const DTag q_tag = TagOf(q_states.dtype, kOp, "q");
  const DTag k_tag = TagOf(k.dtype, kOp, "k");
  const DTag w_tag = TagOf(weights.dtype, kOp, "weights");

  const dim3 block(kHeadLanes, kKeyRows);
  const dim3 grid(static_cast<unsigned>(T),
                  static_cast<unsigned>((S + kKeyRows - 1) / kKeyRows));
  DsaIndexerLogitsKernel<<<grid, block, 0, AsStream(q)>>>(
      logits.Ptr<float>(), q_states.data, q_tag, k.data, k_tag, weights.data, w_tag,
      args.q_scale != nullptr ? args.q_scale->Ptr<float>() : nullptr,
      win_start.Ptr<int32_t>(), win_end.Ptr<int32_t>(), logits.stride[0], q_states.stride[0],
      q_states.stride[1], k.stride[0], weights.stride[0],
      args.q_scale != nullptr ? args.q_scale->stride[0] : 0, static_cast<int>(T),
      static_cast<int>(S), static_cast<int>(H), static_cast<int>(D),
      // The two GLOBAL scalars of the fold, multiplied together once on the host
      // exactly as the CPU arm does (`gfold`), so the device sees the same
      // constant and not a second rounding of it.
      args.softmax_scale * args.n_head_scale);
  Check(hipGetLastError(), "dsa_indexer_logits launch");
}

void DsaTopkSelectKernelRocm(Queue& q, Tensor& indices, Tensor& counts, const Tensor& logits,
                             const Tensor& win_start, const Tensor& win_end) {
  const int64_t T = logits.shape[0];
  const int64_t S = logits.shape[1];
  const int64_t topk = indices.shape[1];
  DsaTopkSelectKernel<<<static_cast<unsigned>(T), kTopkThreads, 0, AsStream(q)>>>(
      indices.Ptr<int32_t>(), counts.Ptr<int32_t>(), logits.Ptr<float>(),
      win_start.Ptr<int32_t>(), win_end.Ptr<int32_t>(), indices.stride[0], logits.stride[0],
      static_cast<int>(S), static_cast<int>(topk));
  Check(hipGetLastError(), "dsa_topk_select launch");
}

// Registers the ROCm arms during static init, the pattern every `rocm_*.hip`
// registrar in this directory uses. UNCONDITIONAL and at preprocessor depth 0:
// an op registered from a feature-gated TU disappears on a build that does not
// compile the feature, the resolver falls through, and on `gfx1151` there is
// nothing to fall through TO — the portable reference tier is not installed
// there (spec D2), so the call refuses by name instead. That refusal is the
// intended failure and a conditional registration would turn it into a silent
// absence.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kDsaIndexerLogits, DeviceType::kROCM,
               reinterpret_cast<void*>(
                   static_cast<DsaIndexerLogitsFn>(&DsaIndexerLogitsKernelRocm)));
    RegisterOp(OpId::kDsaTopkSelect, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<DsaTopkSelectFn>(&DsaTopkSelectKernelRocm)));
  }
};
const Registrar registrar;

}  // namespace
}  // namespace vt::rocm
