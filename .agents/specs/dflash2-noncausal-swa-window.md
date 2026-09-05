# SPEC-DFLASH2 — a non-causal SWA draft layer must attend within its window ([#2784](https://github.com/mudler/vllm.cpp/issues/2784))

Row: `SPEC-DFLASH2`. Issue:
[#2784](https://github.com/mudler/vllm.cpp/issues/2784). Discharges the
`## Owed` entry [#1900](https://github.com/mudler/vllm.cpp/issues/1900) recorded
in [`dflash2-draft-block-fa2.md`](dflash2-draft-block-fa2.md) (W11), which named
this exact kernel property and deferred it as its own unit of work.

## The finding

[#2784](https://github.com/mudler/vllm.cpp/issues/2784) measured, on `dgx:gpu0`
(GB10, driver 580.173.02), one binary, `VT_DFLASH_PAGED=0`, 512 output tokens,
T=0, 4 prompts per rung, 3 reps, draft budget 16, EXL3 target + EXL3 DFlash2
draft:

| prompt tokens | no draft | + draft k=16 | acceptance rate | accepted/pass |
|---:|---:|---:|---:|---:|
| 324 | 16.86 | 86.77 | 0.81 | 12.96 |
| 2,307 | 16.92 | 79.66 | 0.77 | 12.32 |
| **8,159** | **16.34** | **12.57** | **0.06** | **0.96** |

The three 8,159 reps are `12.60 / 12.54 / 12.57`, a 0.5% spread, so it is a
stable state and not variance.

**The cause is a mask, not the data.** Every DFlash draft attention kernel in
this tree drops the sliding window whenever the layer is NON-CAUSAL, and the
shipped DFlash2 drafter is exactly that shape.

## The mechanism, and why the shape is what it is

The published drafter declares, in its own `config.json` (read from three
independent copies on the NAS, byte-identical in these fields):

```json
"layer_types": ["sliding_attention", ...x5], "sliding_window": 2048, "is_causal": false
```

`ResolveQwen3DFlashAttnModes` (`src/vllm/model_executor/models/qwen3_dflash_weights.cpp`)
mirrors upstream and resolves the two answers INDEPENDENTLY, so every one of the
five layers comes out `(causal = false, sliding_window = 2048)`. Its own comment
states the consequence upstream intends:

> "a non-causal SWA layer still attends within its window"

Every kernel then contradicts that comment. The mask bound is guarded on
`causal &&`:

```c
const int64_t jhi = causal ? ii_comb : (N - 1);
int64_t jlo = 0;
if (causal && window > 0) jlo = ii_comb - (window - 1) > 0 ? ii_comb - (window - 1) : 0;
```

so on this checkpoint `jlo` is 0 and `jhi` is `N - 1` at every layer: the draft
attends over the WHOLE combined sequence, with no window at all. Eleven sites
carry that guard, and all eleven are DFlash attention kernels —
`src/vt/cpu/cpu_ops.cpp` (`DFlashBlockAttentionKernel`,
`DFlashPagedBlockAttentionKernel`) and nine in `src/vt/cuda/cuda_ops.cu`
(`DFlashBlockAttentionKernel`, `DFlashAttnQBlockKernel`,
`DFlashBlockAttentionWarpKernel`, `DFlashAttnKeyLaneKernel`,
`DFlashAttnChunkKernel`, `DFlashAttnMmaKernel` block staging and per-row bounds,
`DFlashPagedBlockAttentionKernel`, `DFlashPagedBlockAttentionWarpKernel`). The
W11 paged seam drops it a twelfth time, in the mask TRANSLATION rather than in a
kernel: `DflashBlockPagedMaskOf`
(`src/vllm/model_executor/models/qwen3_dflash_internal.h`) emits no
`window_size` when `causal` is false.

**The defect is invisible below the window and grows above it**, which is the
measured shape and is why no short-context gate could see it:

Counted for the LAST query row of a 1+16 draft block, `ii = C + 16`, against a
key axis of `C + 17`. The symmetric window is `[ii-2047, ii+2047]`, and its
right half falls off the end of that axis, so what actually binds is the left
bound alone:

| context C | keys upstream admits | keys we admitted | extra |
|---:|---:|---:|---:|
| 324 | 341 (the whole axis) | 341 | **none — identical** |
| 2,307 | 2,064 | 2,324 | 260, about 11% |
| 8,159 | 2,064 | 8,176 | **6,112 keys upstream masks, 75% of the attended mass** |

At 324 the whole sequence lies inside +/-2047 of every query row, so the window
is a no-op and the arm is correct by accident. At 2,307 it barely binds and
acceptance moves 0.81 -> 0.77. At 8,159 three quarters of the attended mass is
keys the drafter was never trained to see, and acceptance falls to 0.06. The
comparator [#2784](https://github.com/mudler/vllm.cpp/issues/2784) cites
(`pangoleen/qwen3.8-27b-dgx-spark-dflash2`, the same `dflash_config` on the same
hardware class) applies the window and declines gently across its whole ladder.

**Why no gate caught it.** The verify is lossless, so wrong draft tokens change
no output token; only ACCEPTANCE falls. That is the defect class this row exists
to remove, and the one `RefuseDflash1ArgmaxOnDflash2Block` was written for. The
benchmark harness runs at `--input-len 16`, below the window, where the bug is
provably inert.

## Upstream anchors

Read at the parity pin `5559679229` (`.agents/upstream-sync.md`), local checkout
`/home/mudler/_git/vllm`.

- `vllm/model_executor/models/qwen3_dflash.py:84-146` — `_resolve_layer_attention`
  returns `(sliding_window, causal)` as two INDEPENDENT resolutions. A layer that
  is `sliding_attention` gets a window whatever `is_causal` says.
- `vllm/model_executor/models/qwen3_dflash.py:221-234` —
  `Attention(..., per_layer_sliding_window=sliding_window)` is constructed with
  the window irrespective of `causal`; `self.causal = causal` is a separate
  field carried into the metadata builder.
- `vllm/v1/attention/backends/flash_attn.py:319-330` — **`_maybe_symmetrize_window`,
  the rule this row ports.** Verbatim:

  > "Make a causal sliding window ``(w, 0)`` symmetric ``(w, w)`` when attention
  > is non-causal, so bidirectional queries attend in both directions. Leaves
  > full-attention ``(-1, -1)`` and already-symmetric windows untouched."

  ```python
  non_causal = isinstance(causal, torch.Tensor) or causal is False
  if window is not None and window[0] >= 0 and window[1] == 0 and non_causal:
      return (window[0], window[0])
  return window
  ```
- `vllm/v1/attention/backends/flash_attn.py:665-696` — the builder applies it to
  the group's own `sliding_window` and stores `sliding_window=effective_sliding_window`
  in the metadata, so a non-causal DFlash group runs with `(w-1, w-1)`.

So upstream's answer for `(causal = false, window = w)` is **the SYMMETRIC
window `(w-1, w-1)`**, never "no window". Our engine's own
`vt::AttentionWindow` contract already spells that form:

> "(W-1, 0) is a causal decoder window of W tokens and (W-1, W-1) is the
> symmetric encoder form" (`include/vt/ops.h`)

and `PagedAttentionArgs::window_size` already documents "visible keys are
intersected with `[p-left, p+right]` after the causal/full bound is applied".
Both paged backends already implement the `right` bound
(`src/vt/cpu/cpu_paged_attn.cpp:223-225`,
`src/vt/cuda/cuda_paged_attn.cu:220-224`), so the W11 seam needs only the
translation repaired, not a new kernel.

## Scope

1. **One shared mask bound, in a header, gated on the CPU.**
   `include/vt/dflash_attn_mask.h` — `vt::DFlashMaskSpanOf(ii, num_keys, causal,
   window)` returns the inclusive `[lo, hi]` key range for one query row. It is
   `__host__ __device__` under `__CUDACC__`, following the precedent of
   `include/vt/dflash_attn_grid.h`, whose header comment states the reason this
   file exists in the same words: the mapping "lives here, in a header, because
   it is the part that can be wrong in a way no CUDA-free machine could
   otherwise catch".
2. **All eleven kernel sites call it**, so the eleven copies of the mask stop
   being eleven places the rule can drift.
3. **`DflashBlockPagedMaskOf` emits the symmetric window** for a non-causal
   layer that carries one, and the two prose passages that asserted the opposite
   are corrected.
4. **Tests**, below.

Out of scope: the target's own attention. No shipped target declares a
non-causal SWA layer, and every one of the eleven guarded sites is a DFlash
kernel, so the blast radius of this change is the DFlash draft alone.

## Design

```c
struct DFlashMaskSpan { int64_t lo; int64_t hi; };

DFlashMaskSpan DFlashMaskSpanOf(int64_t ii, int64_t num_keys, bool causal, int64_t window) {
  lo = 0;
  hi = causal ? ii : (num_keys - 1);
  if (window > 0) {
    lo = max(lo, ii - (window - 1));
    hi = min(hi, ii + (window - 1));
  }
  return {lo, hi};
}
```

`ii` is the query row's own position on the key axis (bottom-right aligned, the
kernels' `ii` / `ii_comb`), `num_keys` the length of that axis.

**The causal arm is byte-identical by construction, not by measurement.** For
`causal = true` the pre-window bound is already `hi = ii`, and `ii <= ii +
(window - 1)` for every `window > 0`, so the new `min` can never move it. That
is what makes this change safe to land on the DFlash1 checkpoints, none of which
declare `is_causal`, and it is asserted rather than asserted-about
(`test_dflash_attn_mask.cpp`).

The MMA kernel keeps two bounds: a per-row mask and a block-level STAGING range
that is the union over the block's rows. Both `lo` and `hi` are non-decreasing
in `ii`, so the union is `[span(ii_first).lo, span(ii_last).hi]` — the same two
calls, at the block's first and last row.

## Risks

- **The CUDA edits cannot be compiled on this host** (no nvcc, and the fleet's
  `dgx:gpu0` read `unhealthy (no contact 2h48m)` with twelve queued jobs for the
  whole session). Mitigated by routing every site through ONE header function
  that the CPU suite compiles and executes, which is the precedent
  `dflash_attn_grid.h` set for exactly this hazard; the per-site edits then
  carry no arithmetic of their own. It is a real limitation and is recorded as
  such, not discharged.
- **Drafted tokens move on the published DFlash2 checkpoint.** That is the
  point of the change, and it is why W11 declined to make it in flow. Any
  golden that pins DFlash2 draft tokens at a context above 2048 must be
  re-taken; below 2048 the mask is provably inert, so short fixtures cannot
  move.
- **The symmetric window leaves the FA-2 fast lane.**
  `src/vt/cuda/cuda_paged_attn.cu:2856` admits that lane only for
  `args.causal && left >= 0 && right == 0`, so the W11 paged seam falls to a
  general kernel on this checkpoint. Correct but slower; a speed row, not this
  one.

## Tests

Red-before, in ascending strength:

1. `tests/vt/test_dflash_attn_mask.cpp` — the bound itself. The non-causal
   window is symmetric; the causal arm is unchanged for every `(ii, window)`
   pair over a swept range (the byte-identity claim above, executed).
2. `tests/vt/test_ops_dflash_paged_block_attn.cpp` — **the behavioural gate, and
   the one that would have caught #2784.** `vt::DFlashPagedBlockAttention` with
   `causal = false`, `window = w`, and a context much longer than `w`: two
   contexts identical inside every query row's symmetric window and DIFFERENT
   outside it must produce IDENTICAL outputs. It fails today because the keys
   outside the window are attended.
3. `tests/vllm/models/test_qwen3_dflash_block_route.cpp` — the translation:
   `DflashBlockPagedMaskOf(false, w)` must carry `(w-1, w-1)`. The file's
   existing byte-for-byte route-equivalence case then proves the seam and the
   bespoke op agree on the NEW semantics, which is what keeps the two arms from
   parting.

## Gates

```sh
ctest --test-dir <build> -R 'dflash' --output-on-failure
```

## Stop conditions

- If any CPU token fixture below 2048 context moves: STOP. The mask is provably
  inert there and a movement means the causal arm was not byte-identical.
- If the acceptance re-measurement on `dgx:gpu0` does not recover the 8k rung:
  STOP and report; the mechanism is then not the whole cause.

## Now

`ACTIVE`. Spec committed; implementation and CPU gates follow in this branch.
The DEVICE re-measurement of the 8,159 rung is OWED and is not claimed here —
see `## Owed`.

## Owed

- **The device measurement.** `#2784`'s ladder re-run on `dgx:gpu0` with the fix
  in, at 324 / 2,307 / 8,159 and above, to show acceptance recovering rather
  than inferred to. The fleet gave no lease this session (`dgx:gpu0` unhealthy,
  no contact 2h48m, twelve jobs queued), so the claim in this spec is a
  SOURCE-GROUNDED root cause plus a CPU behavioural gate, and the throughput
  number is not quoted.
- **The onset bisect** (3k / 4k / 6k). The mechanism predicts a smooth onset
  beginning at exactly 2048 and worsening monotonically, which is a falsifiable
  prediction the bisect would test. Owed with the measurement above.
- **A long-context acceptance gate.** Nothing in the harness runs above
  `--input-len 16`, so no gate in this tree can see an acceptance defect that is
  inert below 2048. That is the reason #2784 reached a user rather than a CI
  run, and it is a bigger unit of work than this fix.
