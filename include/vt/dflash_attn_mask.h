#pragma once
// SPEC-DFLASH2 (#2784, discharging #1900): the DFlash block attention's MASK BOUND.
//
// A DFlash draft layer states its attention as (`causal`, `sliding_window`),
// resolved from the checkpoint by `ResolveQwen3DFlashAttnModes`. Turning that
// pair into one query row's visible key range is the arithmetic below, and it
// was written out ELEVEN times: twice in `src/vt/cpu/cpu_ops.cpp` and nine
// times in `src/vt/cuda/cuda_ops.cu`. All eleven copies agreed, and all eleven
// were wrong in the same way.
//
// THE DEFECT THEY SHARED. Every copy guarded the window on `causal &&`, so a
// NON-CAUSAL layer carrying a window attended over the entire combined
// sequence. The published DFlash2 drafter is exactly that shape -- five
// `sliding_attention` layers, `sliding_window: 2048`, `is_causal: false` -- so
// on the checkpoint this engine ships, every draft layer ran unwindowed.
// Below 2048 tokens of context that is a NO-OP and the arm is correct by
// accident; above it the draft attends to keys it was never trained to see, and
// #2784 measured acceptance falling from 0.77 at 2,307 prompt tokens to 0.06 at
// 8,159, where speculation is 23% SLOWER than no speculation at all. The verify
// is lossless, so no output token moves and no token gate in this repository
// can see it. Only ACCEPTANCE falls.
//
// WHAT UPSTREAM DOES, which is not "no window" and is not "the causal window".
// `_maybe_symmetrize_window` (vllm/v1/attention/backends/flash_attn.py:319-330 @
// the parity pin 5559679229) makes a causal `(w, 0)` window SYMMETRIC `(w, w)`
// when attention is non-causal, "so bidirectional queries attend in both
// directions", and `:665-696` puts that value in the metadata every DFlash
// group reads. The window and the causality are two independent resolutions all
// the way down (`qwen3_dflash.py:84-146`, and `:221-234` constructs
// `Attention(per_layer_sliding_window=...)` irrespective of `causal`).
//
// WHY IT LIVES IN A HEADER, in the words `vt/dflash_attn_grid.h` already used
// for the same hazard: it is the part that can be wrong in a way no CUDA-free
// machine could otherwise catch. `tests/vt/test_dflash_attn_mask.cpp` executes
// it on the CPU, so the nine CUDA call sites carry no arithmetic of their own
// and cannot drift from the two CPU ones again.
#include <cstdint>

#if defined(__CUDACC__)
#define VT_DFLASH_MASK_HD __host__ __device__
#else
#define VT_DFLASH_MASK_HD
#endif

namespace vt {

// One query row's INCLUSIVE visible key range on its own key axis. `hi < lo`
// means nothing is visible, which the callers already handle (an empty j loop,
// or the MMA kernel's block-uniform early return).
struct DFlashMaskSpan {
  int64_t lo = 0;
  int64_t hi = -1;
};

// `ii` is the query row's position on the key axis, bottom-right aligned --
// the kernels' `ii` (intra-block) or `ii_comb` (context ++ block). `num_keys`
// is the length of that axis. `window` is the layer's `sliding_window`, 0 for a
// full-attention layer.
//
// THE CAUSAL ARM IS BYTE-IDENTICAL TO THE PRE-#2784 BOUND, BY CONSTRUCTION AND
// NOT BY MEASUREMENT: when `causal` is true the pre-window `hi` is already `ii`,
// and `ii <= ii + (window - 1)` for every `window > 0`, so the upper clamp below
// can never move it. That is what makes this safe on every DFlash1 checkpoint,
// none of which declares `is_causal`. `test_dflash_attn_mask.cpp` sweeps the
// pair rather than trusting this paragraph.
VT_DFLASH_MASK_HD inline DFlashMaskSpan DFlashMaskSpanOf(int64_t ii, int64_t num_keys,
                                                         bool causal, int64_t window) {
  DFlashMaskSpan s;
  s.lo = 0;
  s.hi = causal ? ii : (num_keys - 1);
  if (window > 0) {
    const int64_t wlo = ii - (window - 1);
    if (wlo > s.lo) s.lo = wlo;
    // The symmetric half. On a causal layer this is inert (see above); on a
    // non-causal one it is the bound the whole row exists to restore.
    const int64_t whi = ii + (window - 1);
    if (whi < s.hi) s.hi = whi;
  }
  return s;
}

}  // namespace vt
