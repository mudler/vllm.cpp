// vllm.cpp — the RADIX top-k selection core, shared by the host and the device.
//
// SPEC-DFLASH2 W3 / D2 (revisited), [#1867](https://github.com/mudler/vllm.cpp/issues/1867).
//
// WHAT THIS IS. The arithmetic half of `vt::TopKValuesIndices` on CUDA: a
// monotone float->uint32 key, a four-round 8-bit radix narrowing over histograms
// of those keys, and the tie rule that picks among keys equal to the k-th
// largest. It carries NO parallelism. `src/vt/cuda/cuda_sample.cu` supplies the
// block-cooperative loops that BUILD the histograms and COLLECT the survivors;
// this header supplies every value the answer depends on, so the answer can be
// gated on a host with no `nvcc` (which the authoring host is, and has been
// since W3 — see `## Owed` of .agents/specs/dflash2-spec-decode.md).
//
// UPSTREAM. Mirrored from FlashInfer, which is the kernel vLLM itself dispatches
// to for this exact call. `vllm/model_executor/layers/logits_processor.py:48-52`
// (`_topk`) @ merge `b389ac29465b33f9e9c534df221ea3c129e9793f` reads
//
//     def _topk(scores, k):
//         impl = _flashinfer_topk()
//         if impl is None or not scores.is_cuda:
//             return torch.topk(scores, k, dim=-1)
//         return impl(scores, k, sorted=True, deterministic=True)
//
// and its own docstring at `logits_processor.py:29-35` says why: "The top-k
// spans the vocabulary, where the radix kernel is about twice torch.topk."
// `flashinfer.top_k` lands in `RadixTopKKernel_Unified`
// (`flashinfer/topk.cuh:1094`) — the kernel #1857's SGLang column measured at
// 40 us for the identical 8 x 248320, K=16 shape — whose selection core is
// `RadixSelectFromSharedMemory` (`flashinfer/topk.cuh:576-758`) over the ordered
// keys of `RadixTopKTraits<float>` (`flashinfer/topk_common.cuh:26-48`). Those
// two functions are what this header ports:
//
//   * `RadixTopKKey`        <- `RadixTopKTraits<float>::ToOrdered`, topk_common.cuh:35-39
//   * `RadixTopKPickBucket` <- the suffix-sum bucket search, topk.cuh:683-691
//   * `kRadixTopKRounds`    <- `NUM_ROUNDS = ORDERED_BITS / RADIX_BITS`, topk.cuh:581-585
//
// FlashInfer is read at `0.6.12`, git
// `d768c14e7cf5dd5df45a8a1de78ae815879f108a`, which is the wheel vLLM resolves
// through `vllm.utils.flashinfer`. It does NOT become a repository oracle here:
// AGENTS.md admits FlashInfer as part of the vLLM executing chain, so the
// anchors above are cited the way a CUTLASS or a cuBLASLt anchor is. Both files
// are pinned by CONTENT as well as by revision, because a wheel's headers carry
// no `git log` to check a line number against:
//   topk.cuh        sha256 65909b2e939e92b19c3645cc768a13455a06db2e0364405ff2f9a83827776182
//   topk_common.cuh sha256 4ed389d1535ab2e1b759eddca9fac3c46346b47711be2b72bbb1b99b21524f2d
//
// WHY THE PORT HAPPENED AT ALL, given that D2 refused it. D2 refused FlashInfer's
// KERNEL — 3380 lines of multi-CTA with a grid barrier over a persistent
// workspace, and three tie-break modes — and named its own revisit condition: "Revisit only if W3 measures the
// top-k as the selector's dominant cost here, as it is upstream." #1867 is that
// measurement. What is ported here is the ALGORITHM, in the shape our fixed
// K=16-over-248320 needs, not the general kernel; D2's reasons for refusing the
// kernel are untouched and still hold.
//
// THE KEY, and the two places it is NOT FlashInfer's. `RadixTopKKey` maps a
// float to a uint32 whose UNSIGNED order is the float's own order, so a radix
// digit search over the keys IS a value search. FlashInfer's transform is the
// standard one (flip every bit of a negative, flip only the sign bit of a
// non-negative); ours adds two normalizations, each because the CPU reference
// `src/vt/cpu/cpu_ops.cpp::TopKValuesIndicesKernel` — which is the authoritative
// contract, not this file — answers differently without them:
//
//   * NEGATIVE ZERO maps to the key of POSITIVE ZERO. Untouched, the transform
//     ranks +0.0 (key 0x80000000) strictly above -0.0 (key 0x7FFFFFFF), while
//     the CPU comparator's `x != y` is FALSE for the pair and falls through to
//     ascending index. A row holding both at the k-th boundary would come back
//     in a different ORDER on the two arms, which is precisely the class of
//     divergence this op's contract exists to forbid.
//   * EVERY NaN maps to 0xFFFFFFFF, the maximum. `torch.topk(largest=True)`
//     orders NaN first and the CPU comparator says so explicitly. Untouched, the
//     transform happens to rank a POSITIVE NaN above +inf and a NEGATIVE NaN
//     below -inf, so half the NaNs would sort first and half last. Collapsing
//     them all to one key gives NaN-first for every payload and every sign, with
//     ties among NaNs broken by ascending index — the CPU comparator's `return a
//     < b` branch, reproduced rather than approximated.
//
// The second normalization is also what lets the CUDA arm answer the NaN row
// that [#1489](https://github.com/mudler/vllm.cpp/issues/1489) measured it
// failing: `TopKValuesIndicesRowKernel`'s `fmaxf`/`fminf` bracket returned the
// non-NaN operand and its `r[j] > thr` survivor test was false for a NaN, so
// that kernel could never select one. The radix arm has no such blind spot. That
// is a CLAIM ABOUT THIS HEADER, gated here on the host; whether the DEVICE agrees
// is owed to the same GPU run that owes #1867's timing, and until then the
// device cases in `tests/vt/test_ops_topk_values_indices.cpp` stay narrowed
// exactly as #1489 left them.
//
// WHAT THE ROUNDS DO. `kRadixTopKRounds` = 32/8 = 4. Round r histograms bits
// [24-8r, 32-8r) of every key that MATCHES the prefix fixed by rounds 0..r-1;
// `RadixTopKPickBucket` walks that histogram from bucket 255 down, and stops at
// the first bucket whose "count at or above me" reaches `remaining_k` while
// "count strictly above me" does not. That bucket's digit joins the prefix and
// `remaining_k` drops by the count strictly above. After four rounds the prefix
// is all 32 bits, so it IS the key of the k-th largest element — exactly, with
// no bracket, no tolerance and no iteration budget. The kernel it replaces
// searched the same threshold by ternary bisection over float VALUES under a
// `kThreshMaxIter = 64` budget, which is where #1867's 683 us/step went.
#ifndef VLLM_CPP_VT_RADIX_TOPK_H_
#define VLLM_CPP_VT_RADIX_TOPK_H_

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define VT_RADIX_TOPK_HD __host__ __device__ inline
#else
#define VT_RADIX_TOPK_HD inline
#endif

namespace vt {

// 8 bits per round over a 32-bit key: 256 buckets, 4 rounds. These are
// FlashInfer's `RADIX_BITS`/`RADIX` and its derived `NUM_ROUNDS`
// (`topk.cuh:581-585`), not independent choices — a different radix would change
// the histogram width the kernel allocates in shared memory and nothing else
// about the answer.
constexpr int kRadixTopKBits = 8;
constexpr int kRadixTopKRadix = 1 << kRadixTopKBits;  // 256
constexpr int kRadixTopKRounds = 32 / kRadixTopKBits;  // 4

// --- THE CANDIDATE BUFFER, AND WHY ITS SIZE IS A DEVICE QUERY -------------
//
// The kernel compacts the columns that survive a round into shared memory so the
// later rounds do not re-read global. How many survive is DATA, and the number
// is much larger than the first version of this file assumed. Round 0's digit is
// `key >> 24`: the sign bit plus the TOP SEVEN of the eight exponent bits, so
// one bucket spans TWO ADJACENT EXPONENTS — a 4x range of magnitudes, not one
// octave. Measured on the rows `tests/vt/test_ops_radix_topk.cpp` actually runs,
// at 248320 columns and K = 16:
//
//   row                       round-0 bucket    round-1 bucket
//   production LCG rows 0..7   92701..93938        456..522
//   tie-dense LCG rows 0..3    92593..93401        452..500
//   gaussian sd=1.0                   5657              1
//   gaussian sd=3.0                    973              4
//   gaussian sd=6.0                  22829              1
//   gaussian sd=10.0                   160              4
//   every column equal              248320         248320
//
// Two things follow, and the kernel is built on both. A cap of any size that
// fits shared memory is defeated by the round-0 bucket on the production shape —
// 93k columns is 45x the 2048 the first version chose and still 5.7x
// `kRadixTopKCandCapMax` — so the buffer alone cannot deliver the compaction.
// And the SAME rows narrow to about 490 columns one round later, which fits with
// three orders of magnitude to spare. The kernel therefore compacts after round
// 0 when that fits, and otherwise re-compacts after round 1 from the round-1
// histogram, which it already has. That second stage is FlashInfer's
// `collect_with_threshold_non_last_round` (`flashinfer/topk.cuh:2566-2592`):
// emit the columns strictly above the round's threshold bin, carry the columns
// equal to it into the next buffer.
//
// Bytes per candidate: one key and one column index.
constexpr int kRadixTopKCandBytes = static_cast<int>(sizeof(uint32_t) + sizeof(int));  // 8

// The ceiling on the candidate buffer, in candidates. This is FlashInfer's
// `FILTERED_TOPK_SMEM_INPUT_SIZE = 16 * 1024` (`flashinfer/topk.cuh:2267`),
// whose `FILTERED_TOPK_SMEM_DYNAMIC = sizeof(int) * 2 * 16K` is the 128 KB of
// DYNAMIC shared that `LaunchFilteredTopKUnified` opts into with
// `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
// smem_size)` (`topk.cuh:3088-3105`). 16K candidates at 8 bytes is 128 KB here
// too, which is why the buffer CANNOT be a static `__shared__` array: ptxas caps
// static shared at 48 KB on every architecture. Mirroring the opt-in is the
// reason this header carries no per-device constant — the launcher asks
// `cudaDevAttrMaxSharedMemoryPerBlockOptin` what the part allows, exactly as
// FlashInfer's own sizing does (`GetRadixTopKAvailableOrderedSmemBytes`,
// `topk.cuh:42-59`; the query at `topk.cuh:1480-1481`).
constexpr int kRadixTopKCandCapMax = 16 * 1024;

// How many candidates fit, given the DYNAMIC shared-memory budget the launcher
// obtained and the `k` (key, index) winner pairs that share the same allocation.
// Returns 0 when nothing is left over, which is a legal answer: the kernel's
// global fallback needs no candidate buffer at all.
VT_RADIX_TOPK_HD int RadixTopKCandCap(uint32_t dynamic_smem_bytes, int k) {
  const uint32_t pairs = static_cast<uint32_t>(k) * static_cast<uint32_t>(kRadixTopKCandBytes);
  if (dynamic_smem_bytes <= pairs) return 0;
  const uint32_t cap = (dynamic_smem_bytes - pairs) / static_cast<uint32_t>(kRadixTopKCandBytes);
  return cap > static_cast<uint32_t>(kRadixTopKCandCapMax) ? kRadixTopKCandCapMax
                                                          : static_cast<int>(cap);
}

// The dynamic shared-memory allocation the launcher must request for a given
// cap: the k winner pairs first, then the candidate buffer.
VT_RADIX_TOPK_HD uint32_t RadixTopKDynamicSmemBytes(int cand_cap, int k) {
  return static_cast<uint32_t>(cand_cap + k) * static_cast<uint32_t>(kRadixTopKCandBytes);
}

// The monotone key. `RadixTopKKey(a) < RadixTopKKey(b)` iff `a` sorts BELOW `b`
// under this op's contract (NaN first, then descending value, then ascending
// index — the index half is not the key's business).
//
// Ported from `RadixTopKTraits<float>::ToOrdered`, flashinfer/topk_common.cuh:35-39,
// plus the two normalizations the file header argues for.
VT_RADIX_TOPK_HD uint32_t RadixTopKKey(float x) {
  uint32_t bits;
  // No `std::memcpy` and no `__float_as_uint`: this function compiles on the
  // host and inside a `__global__`, and a union punt is the one spelling both
  // accept. It is not type-punning UB in either C++ or CUDA C++ as compiled
  // here, and every alternative costs one arm or the other.
  union {
    float f;
    uint32_t u;
  } pun;
  pun.f = x;
  bits = pun.u;

  // EVERY NaN to the maximum key. An exponent of all ones with a non-zero
  // mantissa is a NaN whatever the sign bit says, so this catches the negative
  // ones the raw transform would have sent to the bottom.
  if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) return 0xFFFFFFFFu;
  // NEGATIVE ZERO to positive zero's key. `-0.0f == 0.0f` in the CPU
  // comparator, so the two must not be separable here either.
  if (bits == 0x80000000u) bits = 0u;

  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// The digit round `round` reads. Round 0 reads the TOP 8 bits.
VT_RADIX_TOPK_HD uint32_t RadixTopKBucket(uint32_t key, int round) {
  return (key >> (32 - kRadixTopKBits * (round + 1))) & 0xFFu;
}

// The mask that isolates the bits rounds 0..round-1 have already fixed. Round 0
// fixes nothing, so its mask is zero and every key matches — which is why the
// first round histograms the whole row.
VT_RADIX_TOPK_HD uint32_t RadixTopKPrefixMask(int round) {
  return round == 0 ? 0u : (0xFFFFFFFFu << (32 - kRadixTopKBits * round));
}

// Does `key` still belong to the candidate set after rounds 0..round-1?
VT_RADIX_TOPK_HD bool RadixTopKPrefixMatches(uint32_t key, uint32_t prefix, int round) {
  const uint32_t mask = RadixTopKPrefixMask(round);
  return (key & mask) == (prefix & mask);
}

// One round's decision: given `hist[256]` — how many candidate keys carry each
// digit value — and how many of the top-k are still unplaced, return the digit
// the k-th largest carries and write back the count still unplaced BELOW that
// digit.
//
// Ported from flashinfer/topk.cuh:683-691, which spreads the same test across
// 256 threads (`count_ge >= remaining_k && count_gt < remaining_k`). Here it is
// one descending walk, because the parallel form buys nothing: 256 adds against
// the 248320 loads the histogram itself cost is not a term in the runtime, and a
// serial walk is the same code on the host and on the device, which is the
// property the whole header exists for.
//
// `remaining_k` is >= 1 and no larger than the total count in `hist` — the
// caller establishes both, and the walk relies on it: the loop is guaranteed to
// stop, because bucket 0's `count_ge` is the total.
VT_RADIX_TOPK_HD uint32_t RadixTopKPickBucket(const uint32_t* hist, uint32_t remaining_k,
                                              uint32_t* out_remaining) {
  uint32_t count_gt = 0;  // candidates in buckets strictly ABOVE the one under test
  for (int b = kRadixTopKRadix - 1; b >= 0; --b) {
    const uint32_t count_ge = count_gt + hist[b];
    if (count_ge >= remaining_k) {
      *out_remaining = remaining_k - count_gt;
      return static_cast<uint32_t>(b);
    }
    count_gt = count_ge;
  }
  // Unreachable when the precondition holds. Answering bucket 0 rather than
  // reading past the array keeps a violated precondition a WRONG ANSWER on the
  // host, where a test can see it, instead of an out-of-bounds read on a device,
  // where nothing can.
  *out_remaining = remaining_k;
  return 0u;
}

// Does `a` outrank `b` in the final k-element ordering? Descending KEY, ties by
// ascending INDEX. Descending key is descending value with NaN first, by
// construction of `RadixTopKKey`, so this one predicate is the whole contract
// that `src/vt/cpu/cpu_ops.cpp::TopKValuesIndicesKernel`'s comparator states in
// float terms.
VT_RADIX_TOPK_HD bool RadixTopKOutranks(uint32_t key_a, int64_t idx_a, uint32_t key_b,
                                        int64_t idx_b) {
  if (key_a != key_b) return key_a > key_b;
  return idx_a < idx_b;
}

}  // namespace vt

#endif  // VLLM_CPP_VT_RADIX_TOPK_H_
