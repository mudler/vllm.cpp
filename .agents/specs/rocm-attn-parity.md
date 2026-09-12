# ROCm attention parity: the value-dtype probability and the FP32 Q/K carrier

Owning row: `BACKEND-ROCM-BF16-MOE`
Issue: [#3115](https://github.com/mudler/vllm.cpp/issues/3115), which asks for the
decode attention and Q/K preamble parity of the BF16 MoE row.
Base: `cd1ab3909a25d58a9676afa08263366b09ee0cb8` on
`row/BACKEND-ROCM-BF16-MOE-attn-parity`.
Integration: one pull request, with this spec committed before any product edit.
The test-only replay instrument `e5c7bfc4d` is already at the base and is the
measurement this repair answers.

## Now

State: `ACTIVE`. Both repairs are in and measured on native gfx1100 at layer 0
step 0 of the L33/C2/R0 workload: row A0 (native output vs the primary's 2902
words) is 0, row B (post-RoPE Q/K) is 1 Q / 0 K, and row C (the kernel on the
primary's own Q/K/V) is 0 of 8448. The fresh review then found four defects in the
repair, all repaired here: the op-layer stride check that made the `VT_FUSED_CHAIN_ADOPT=0`
hand-call throw for every `Dh > 1`, the fixed 64-key tile that ignored the arm the
primary would execute, a focused case that could not tell a 32-key tile from a
64-key one on this capture, and an instrument that exited mid-run. `## Design 1`
carries the arm table; `## Owed` names the arms that are implemented but have no
capture to measure them against.

## Problem and scope

Two arithmetic differences produce the 2902-word parity gap on native gfx1100 at
layer 0 step 0 of the L33/C2/R0 workload (66 tokens, `hq = hkv = 1`, `dh = 128`,
`scale 0.08838834764831845`).

1. **The kernel keeps the softmax probability in f32.** The primary narrows it to
   the value dtype before the value dot and keeps only the running sum in f32.
   `PagedAttnDecodeOptBf16T` accumulated `o_reg = o_reg * corr + pw * v_reg` with
   `pw` in f32 (pre-repair site `src/vt/rocm/rocm_paged_attn.hip:527-531`, now
   `:636-640`).
2. **The preamble narrows the normalized Q/K to BF16 before RoPE.** The primary
   carries the normalized value in f32 through the rotation and narrows once, at
   the store, reading a BF16 cos/sin cache. The native preamble rounds twice.

Both repairs are mirroring repairs. Neither invents behavior: each replaces a
native rounding decision with the primary's own.

**In scope.** `src/vt/rocm/rocm_paged_attn.hip` and the ROCm attention preamble,
plus the test cases that witness them, plus this spec.

**Out of scope.** The CUDA, Metal and CPU realizations of the same two
arith-metics; the two rocWMMA attention kernels; the Tier-0 fused-chain composite
for `kAttnQkNormRope`. Each is named under `## Sibling sites` or `## Owed` with
its reason and is not silently changed.

## Primary anchors

Pin `e126687a9a828d513c01a07cd69f025f27d63280`
(`.agents/upstream-sync.md` §"parity-pin"); source tree
`/home/vikash/oracle/gfx1100-active-2773/source`.

The executing primary arm for this workload is Triton, not the ROCm custom
kernel: `vllm/platforms/rocm.py` gates its decode predicate on `gqa_ratio >= 3`
and this fixture has `qg == 1`, so `chunked_prefill_paged_decode.py` logs the
fallback and a 33-token request (`max_query_len > 1`) enters
`prefix_prefill.py::context_attention_fwd`.

| fact | anchor |
|---|---|
| prefill softmax probability, f32 | `vllm/v1/attention/ops/prefix_prefill.py:443` |
| prefill running sum taken BEFORE the narrowing | `prefix_prefill.py:445` |
| prefill probability narrowed to the value dtype | `prefix_prefill.py:471` |
| prefill value dot accumulates the narrowed probability | `prefix_prefill.py:473` |
| prefill denominator update keeps the f32 sum | `prefix_prefill.py:475` |
| prefill epilogue divides by the denominator | `prefix_prefill.py:478` |
| decode probability narrowed inside the dot | `chunked_prefill_paged_decode.py:265` |
| decode running sum taken BEFORE the narrowing | `chunked_prefill_paged_decode.py:251` |
| Qwen3 q/k norm then RoPE | `vllm/model_executor/models/qwen3.py:150-167` |
| the cos/sin cache is narrowed to the query dtype | `vllm/model_executor/layers/rotary_embedding/base.py:105-131` |
| the compiled preamble reads that BF16 cache | `RotaryEmbedding.forward_static`, `base.py:150-190` |

The last three rows are why the primary's boundary is
`bf16(rope_f32(rmsnorm_f32(bf16 qkv)))` against a BF16 cos/sin cache. The
validated CPU transcription of that boundary
(`/home/vikash/.cache/attn-parity-plan/check_attention_models.py`, model A)
reproduces the primary's captured output **exactly, 0 of 8448 words**, and its
preamble model reproduces the primary's captured Q to **1 of 8448 words**. Those
two numbers license the CPU model as the primary-arithmetic reference without
executing the primary, and they are the targets this repair is measured against.

## Measurement this repair starts from

Operator-verified and independently reproduced, identical in
`/home/vikash/.cache/moe-attn-parity/gpu-run-at-head.log` and
`operator-rerun.log`; 1322 of 1322 assertions, 1 case, 3 skipped. Native gfx1100,
L33/C2/R0 layer 0 step 0.

| row | meaning | words / 8448 |
|---|---|---|
| A0 | native run output vs primary output | 2902 |
| B | native run post-RoPE Q / K vs primary Q / K | 1569 / 1542 |
| C0 | replay self-consistency, native Q/K/V at the primary's geometry | 0 |
| C | native `PagedAttention` on the PRIMARY's own Q/K/V vs its output | 2918 |
| D | native `RmsNorm`+`RopeFromCache` on the primary's qkv vs primary Q/K | 1569 / 1542 |
| D' | the same through `RopeNeox` | 1776 / 1719 |
| E | native cos/sin table vs primary `cos_sin` | 0 |
| F | primary qkv q/k slices vs native pre-norm q/k | 0 |

C0 is the guard: the replay reproduces the native run's own output byte-for-byte
at the primary's cache geometry, so row C measures the kernel and not the
transplant. E and F exclude the cos/sin source and the projection, so the
preamble term is exactly the pre-RoPE BF16 store.

### Where this head stands

Native gfx1100, the same workload, the same instrument, after both repairs and
after the fresh review's four findings were repaired. `VT_FUSED_CHAIN_ADOPT=1` and
`=0` produce byte-identical dumps; 8 of 8 cases pass at either setting.

| row | meaning | before | at this head |
|---|---|---|---|
| A0 | native run output vs primary output | 2902 | **0** |
| B | native run post-RoPE Q / K vs primary Q / K | 1569 / 1542 | 1 / 0 |
| C0 | replay self-consistency at the primary's geometry | 0 | 0 |
| C | native `PagedAttention` on the PRIMARY's own Q/K/V | 2918 | **0** |
| D | composite preamble (RmsNorm+RopeFromCache) on primary qkv | 1569 / 1542 | 1569 / 1542 (not the executing path) |
| D2 | production fused op on the primary's qkv | — | 1 / 0 |
| D3 | hand-call `vt::AttnQkNormRope` on the primary's qkv | — | 1 / 0, and 0 words against D2 |
| E | native cos/sin table vs primary `cos_sin` | 0 | 0 |
| F | primary qkv q/k slices vs native pre-norm q/k | 0 | 0 |

Row A0 at 0 is the row's target: the native output now reproduces the primary's
captured attention output byte-for-byte on the L33/C2/R0 layer-0 step-0 workload,
with the production token gate unmoved (`[66,1,70,57,33,81,63,69]` at length 33,
concurrency 2, request 0, and the same 27 records as before).

## Design 1 — the value-dtype probability, at the primary's reference max

### The literal one-line mirror is FALSIFIED

The first implementation of this design narrowed `pw` at the accumulate, exactly
as `prefix_prefill.py:471` narrows `p`. Measured on gfx1100 at the same workload,
row C moved **2918 -> 3011 of 8448 words**: the literal mirror makes parity worse.

It was falsified on CPU first, and the CPU model is exact for this workload. The
preserved transcription in `/home/vikash/.cache/attn-parity-plan/` is a faithful
model of the native kernel: its warp-online variant reproduces the device's
2918 exactly. Parameterized (`/home/vikash/.cache/moe-attn-parity/check_variants.py`,
`diag.py`), it gives:

| form | words / 8448 |
|---|---|
| tile max, narrow | **0** |
| tile max, no narrow | 2917 |
| warp-online, no narrow (the device's 2918) | 2918 |
| warp-online, narrow (the literal mirror) | 3011 |
| sequential-online, narrow | 2058 |

The second column is the reason. Narrowing is only the primary's boundary when
the exponent's reference max is the same one the primary used. `p` is narrowed
at the scale of `qk - m_ij`: the primary's `m_ij` is
`maximum(m_i, tl.max(qk, axis=1))` (`prefix_prefill.py:442`) — a running max
advanced once per key tile, 64 keys on the measured arm — while the native kernel
advanced a running max **once per key, per warp** (pre-repair site
`rocm_paged_attn.hip:525-533`, now `:636-640`). A bf16 rounding is
relative, so narrowing `exp(qk - m_warp_running)` and then rescaling by
`exp(m_warp - gm)` at the combine is not the same rounding as narrowing
`exp(qk - m_ij)`. For a row whose keys fit one tile the two scales coincide, which
is why the tile-max model is exact at 0 — and why this workload's rows, at 33
keys, are the case where getting the scale right is worth the whole 2918.

Keeping the literal mirror would ship a change that measurably worsens the row's
metric, so it is reverted, not committed.

### The repaired design

Mirror the primary's rule together with the reference max it is taken against:

* the value accumulate multiplies a probability narrowed to the **value dtype**;
* that probability is `exp(s - m_ij)` where `m_ij` is the running max advanced
  **once per contiguous key tile**, exactly the primary's
  `maximum(m_i, tl.max(qk, axis=1))` (`prefix_prefill.py:442`,
  `chunked_prefill_paged_decode.py:244`);
* `corr` (the primary's `alpha`) stays f32 — `prefix_prefill.py:449`, `:258`;
* the running sum keeps the un-narrowed f32 probability — `prefix_prefill.py:445`
  against `:471`, `chunked_prefill_paged_decode.py:251` against `:265`;
* the warp-combine rescale stays f32, because the primary's `acc = acc * alpha` is
  f32 at `:449`/`:258` and has no narrowed analogue.

### The tile is the primary's per-row arm, not a constant

The first repair fixed the tile at 64 keys. That is the tile of ONE arm, and the
same native kernel serves three: `chunked_prefill_paged_decode.py:317` dispatches
the call on the batch's max query length and the decode kernel it launches returns
for a row with `query_len > 1` (`filter_by_query_len=True`, `:93-97`, `:503`), so
the arm is decided per **row**:

| arm (per row) | primary kernel | context tile | chunk / decode tile |
|---|---|---|---|
| `query_len > 1`, empty context | `prefix_prefill._fwd_kernel` | — | `BLOCK_N` = 64, anchored at the chunk start |
| `query_len > 1`, cached context | `prefix_prefill._fwd_kernel` | `TRITON_BLOCK_SIZE` = 32, anchored at key 0 | `BLOCK_N` = 64, anchored at the chunk start |
| `query_len == 1` | Triton `kernel_paged_attention_2d` | — | `min(block_size, 128)`, anchored at key 0 |
| any, non-power-of-two physical block size | both of the above | 32 | 32 |

Anchors and widths are read from the launcher, not chosen: `prefix_prefill.py:955-966`
(`BLOCK_M`/`BLOCK_N`), `:965` (`TRITON_BLOCK_SIZE = 32`, bound as the kernel's
`BLOCK_SIZE` at `:1007`), `:231-343` (the context loop from key 0: its running max
at `:312`, the narrowed probability at `:338`, the value dot at `:340`, and lanes
at or past `cur_batch_ctx_len` masked at `:288-290`) and `:369` (the chunk loop
from the chunk start);
`chunked_prefill_paged_decode.py:444-445` (`TRITON_BLOCK_SIZE = min(block_size, 128)`,
or 32 when the physical block size is not a power of two), `:147-149` (`start_n = j *
BLOCK_SIZE`) and `:244`. A key outside the causal or sliding-window bound is
skipped rather than dropped from the grid: the primary masks such a lane to `-inf`
(`prefix_prefill.py:288-290`) or `-10000` (`chunked_prefill_paged_decode.py:236`),
neither of which can raise a tile max, and `exp(x - m)` underflows to 0 for either.
Tiles wholly below the window's left bound are skipped so a windowed model keeps its
`O(window)` walk.

Concretely, the key walk in `PagedAttnDecodeOptBf16T` becomes per-tile two-phase
over the arm's key grid: each of the eight warps computes its strided keys' scores
for the tile, the tile max is reduced across the CTA (one `__syncthreads` per tile,
where the pre-repair loop anchored at `jmin` and had none), then every warp applies
the shared `m_ij`, the shared `alpha = FastExp(m_prev - m_ij)`, and accumulates
`ProbInValueDtype<bf16>(p) * v_reg` with the f32 `p` in `lsum`. The key-to-warp map
stays the tile-and-warp-strided one the kernel already uses, so all eight warps
stay busy on a 33-key row and the loop keeps its shape. A tile narrower than
`kDecWarps` leaves the extra warps with no keys and an all-`-inf` partial max, which
cannot raise the CTA max.

One helper expresses the narrowing for both value dtypes the file uses
(`src/vt/rocm/rocm_paged_attn.hip` uses only `float` and `__hip_bfloat16`; there
is no f16 KV path in the file):

```cpp
template <typename TV>
__device__ inline float ProbInValueDtype(float p) {
  if constexpr (std::is_same_v<TV, __hip_bfloat16>) {
    return __bfloat162float(__float2bfloat16(p));  // p.to(v.dtype)
  } else {
    return p;  // p.to(f32) is the identity
  }
}
```

Narrowing to f32 is the identity, so a site templated on the value dtype is
correct for both arms and no dispatch is added.

**Cost, stated because it is real.** The tile max needs the tile's scores before
the exponent, so the loop needs one `__syncthreads` per tile and a score register
per key per warp. The registers are sized for the widest tile any arm asks for
(128 keys, `min(block_size, 128)`), and the measured cost of that on the compiler's
own resource report is nil: `PagedAttnDecodeOptBf16T<4>/<8>/<16>` go from 57/65/93
VGPRs before this change to 52/56/92 after, with 0 spill bytes and 16 waves/SIMD
occupancy on gfx1100 either way. What the finer decode tile DOES cost is one
synchronization per `min(block_size, 128)` keys instead of one per 64, so a model
whose physical block size is 16 or 32 now pays twice or four times as many CTA
synchronizations on the decode arm — the primary's own structure and the price of
the mirror. This change makes no performance claim and the decode arm's throughput
must be measured before the arm is accepted as free.

### Sibling sites in `src/vt/rocm/rocm_paged_attn.hip`

Line numbers are the POST-repair file's (`794298535` + the arm repair); the
pre-repair table named the same kernels 62-205 lines earlier.

Every site below needs both halves of the repaired design: the value-dtype
probability **and** the reference max it is narrowed against, at the arm's own
tiles. This change implements the pair at the measured site only; the rest are
owed, because each is a separate kernel rewrite of the same shape and none of them
is measured on this workload. The literal one-line narrowing was measured at the
measured site and made parity worse, so it is not applied anywhere.

| site | kernel | value dtype | disposition |
|---|---|---|---|
| `:638` | `PagedAttnDecodeOptBf16T` | bf16 | **changed** — the measured site; the arm table above; d=128 at `:2356`, d=256 at `:2363`, d=512 at `:2370` |
| `:221` | `PagedAttnOnline` | `TKV` | **owed** — the generic fallback, and the primary's arm for neither measured dtype |
| `:687` | `PagedAttnDecodeGqaBf16` | bf16 | **owed** — the QG-fused decode sibling (`:2327`, `:2334`, `:2341`); needs the same arm-tiled walk |
| `:830` | `PagedAttnDecodeGqaF32Q` | `TKV` | **owed** — the f32-query / bf16-KV arm (`:2422`, `:2430`, `:2440`, `:2448`) |
| `:982` | `PagedAttnPrefillFlashTile` | bf16 | **owed** — scalar online-V branch of the flash tile |
| `:1513` | `PagedAttnPrefillSharedK` | `TKV` | **owed** — the scoreless shared-K prefill |
| `:1283` | `PagedAttnPrefillWmmaWave` | bf16 | **owed** — body compiles only under `VT_ROCWMMA_OK` (`:9`, `:1292`, gfx1200/1201), so on gfx1100 neither the edit nor its result can be compiled or run here |
| `:1754` | `PagedAttnPrefillSharedKWmma` | bf16 | **owed** — same guard (`:1763`), and the host admits it only through `PrefillSharedKWmmaHostOk()` on gfx1200/1201 (`:41`, `:2193`) |

The last two are also both reachable only from default-OFF lab toggles
(`VT_ATTN_PREFILL_FLASH`, `VT_ATTN_PREFILL_SHAREDK_WMMA`). Every owed site is
recorded under `## Owed`, not silently skipped.

**Not a site.** `FastExp` (`:197`) is left alone: the CPU model shows `fast_exp`
and `np.exp` both reach 0 in the tile form, so it is not on this path's critical
term. The reciprocal epilogue (`:676`) is a 2^-24-scale difference against the
primary's division and is likewise not the term.

## Design 2 — the FP32 Q/K carrier through RoPE

The native preamble today is `RmsNorm` into a BF16 buffer followed by
`RopeFromCache` over that buffer (`include/vt/recipes.h:357-385` binds the 2-D
norm view and the 3-D rope view to the same BF16 buffer; the hand-call at
`include/vllm/model_executor/models/dense_attn_block.h:642-648` is the same
sequence). On ROCm the fused branch at `dense_attn_block.h:582-613` executes that
composite and not a fused kernel, because at the pre-repair head
`src/vt/rocm/rocm_ops.hip:321` registered `OpId::kAttnQkNormRopeGate` and
**not** `OpId::kAttnQkNormRope` (the repair adds the registration at `:331`), so
`vt::FusedChain` takes its `OpRegistered` guard (`src/vt/ops.cpp:1448-1452`) and
falls through to `FusedChainComposite` (`src/vt/ops.cpp:1468`).

**Design: register the recipe's own fast op on ROCm, with the f32 carrier.**

`vt::OpId::kAttnQkNormRope` is registered for `DeviceType::kROCM`, implemented by
`AttnQkNormRopeKernelRocm` in `src/vt/rocm/rocm_ops.hip`. The kernel:

1. loads the raw `q3`/`k3` row (BF16), the per-head norm weight (BF16) and the
   cos/sin cache row (BF16 at the position `positions[token]` supplies);
2. computes `mean(x^2)` over the head with the **same** f32 accumulation and the
   same `kBlock` binary-tree shared reduction the shipped `RmsNormRowKernel` uses
   (`src/vt/rocm/rocm_rmsnorm.hip:118-147`), then
   `inv = 1 / sqrtf(mean + eps)`;
3. forms `v * inv * w[j]` in f32, rotates the first `rot` elements in f32 exactly
   as `RopeFromCacheK` does (`src/vt/rocm/rocm_dense_basic.hip:699-704`, NeoX and
   GPT-J pair orders both), and narrows **once**, at the store, with the bf16
   store helper the file already uses;
4. leaves elements at or beyond `rot` normalized and narrowed once, which is the
   same BF16 word the shipped `RmsNorm` store produced.

The kernel is modelled on the registered sibling `AttnQkNormRopeGateK`
(`src/vt/rocm/rocm_gdn_fused.hip:95-169`), which is already a fused
norm-plus-partial-NeoX-RoPE preamble on this backend. The `attn_f32` arm passes
f32 states, an f32 norm weight and the f32 cache; for that arm the kernel's
arithmetic is the shipped `RmsNorm` + `RopeFromCache` arithmetic with no
intermediate rounding at all, so its bytes are unchanged by construction.

**Both realizations must agree.** With the op registered, `vt::FusedChain` takes
the fast path for the fused branch, and the hand-call fallback at
`dense_attn_block.h:645-653` is changed to dispatch to the same
`vt::AttnQkNormRope` when the op is registered on the device and the bf16 cache
is in use. On a backend that registers no fast op (CPU), and on the `RopeNeox`
default branch (no cache), both realizations keep exactly today's sequence. So
the fused path and the hand-call fallback agree byte-for-byte on every backend,
which is what the recipe's byte-exact composite contract asks for
(`include/vt/recipes.h:378-383`) and what `tests/vllm/models/test_qwen3_forward.cpp`
already gates for the adoption switch.

**The Tier-0 composite stays as it is, and is owed.** Carrying f32 through the
composite would require an operand slot that holds an f32 intermediate
(`kMaxFusedOperands` is 8, `include/vt/fused_recipe.h:107-108`, and the recipe
already spends all eight) plus a `RopeFromCache` that reads an f32 state against
a BF16 cache. That is a recipe and op-contract change on four backends, which
this row does not own. On ROCm the composite is no longer the executing path
once the fast op is registered; every other backend keeps today's bytes. This is
recorded under `## Owed`.

## Risks

* **This changes the numerics of existing users of the six attention kernels.**
  Every ROCm paged-attention arm whose value dtype is bf16 moves toward the
  primary and away from its previous bytes. Committed ROCm device goldens and
  near-tie anchors on those arms may need re-derivation; the production gate
  `test_rocm_moe_bf16` is re-run in this change and any moved golden is reported.
* **This changes the numerics of the ROCm BF16 Qwen3-dense preamble** for every
  model with a qk-norm and the cos/sin cache (which is on by default,
  `dense_attn_block.h:92-98`). The `attn_f32` arm is unaffected by construction.
* **The kernel term may not reach exactly zero.** The primary's value dot is a
  tile `tl.dot` with its own accumulation order; the native arm keeps `FastExp`,
  a warp-strided key walk and a reciprocal epilogue. A residual of a few words is
  possible and must be attributed before any further edit.
* **A greedy anchor can move.** The change alters bf16 words; a token is decided
  by the full logits, so movement is possible in principle. The gate's recorded
  tokens are checked and any movement is reported rather than papered over.
* **The decode arm's tile follows the physical block size, so a model with a
  block size other than 64 changes bytes on every decode step.** That is the
  mirror, not a side effect: the primary's Triton decode kernel tiles by
  `min(block_size, 128)`, so the native arm tiled by 64 was narrowing against a
  reference max the primary never used whenever `block_size != 64`. It also costs
  one CTA synchronization per tile, so a block size of 16 or 32 pays more of them
  per decode step. `tests/vt/test_ops_paged_attn.cpp` and the production gate are
  re-run here and neither moved; a model whose committed decode golden was captured
  with `block_size != 64` would need its golden re-derived, and none was found.
* **Performance.** The preamble repair replaces three launches per layer with
  one; the kernel repair adds one bf16 round per key and one synchronization per
  tile. Neither is a perf claim and no perf axis is accepted by this change.

## Tests

**Red first, focused, in the committed instrument**
(`tests/vllm/models/test_rocm_moe_bf16.cpp`, case "ROCm paged attention replays
the primary's captured attention boundary"):

* kernel: `CHECK(replay_diff.different == 0)` — red at 2918 before Design 1,
  and still red at 3011 under the falsified literal mirror;
* preamble: `CHECK(run_q.different <= 1 && run_k.different <= 1)` — red at
  1569/1542 before Design 2;
* preamble on the primary's own qkv through the production op, which must reach
  the CPU model's 1 word — red before Design 2;
* hand-call (D3): `vt::AttnQkNormRope` on that same primary qkv must reach the
  same 1 word and must equal the fused realization byte-for-byte — red (it threw)
  before the op-layer stride repair.

The instrument gains an observer on `OpId::kAttnQkNormRope` for the pre-norm and
post-RoPE native bytes, because after Design 2 the production preamble no longer
calls `kRopeFromCache`; the existing `kRopeFromCache`/`kRmsNorm` observers stay
and now measure the composite fallback.

**The arm's tile width, on synthesized data.** The capture cannot witness the tile
choice: its 33-token rows are all `prefix_prefill` with an empty context, so
`max(keys 0..31) == max(keys 0..32)` and a 32-key tile reproduces the same 8448
bytes. Three device cases build a workload where the width IS load-bearing — each
key's value row is a single bf16-exact integer in its own output lane, `q = e0`,
and each K row is 0 or a bf16-exact high value at one key, so a key that shares a
tile with the high key is narrowed at a different scale than the same key alone in
a tile. Each case compares the device against a host transcription of the primary's
own key walk for that arm AND against the same transcription under the neighbouring
widths, so the case fails in both directions:

| case | arm | device vs the arm's tiles | device vs the neighbouring tilings |
|---|---|---|---|
| "uses the primary's 64-key prefill tile" | `query_len` 65, empty context, block 16 | 0 / 33280 words | 1408 words at 32 and at 16 |
| "uses the primary's 32-key context tile" | `query_len` 3, 37-key context, block 16 | 0 / 1536 words | 192 words at a uniform 64 |
| "uses the primary's decode tile" | `query_len` 1, `seq_len` 40, block 16 | 0 / 1024 words | 64 words at 32 and at 64 |

The thresholds are a noise floor of 8 words (the host/device difference in the
order the f32 running sum is accumulated) against a signal floor of 32, and the
measured separation is 0 against 64 at its narrowest.

**The adoption switch, in one process.** Case "ROCm bf16 qk-norm-rope: the
hand-call realization matches the fused recipe" runs the recipe's fast realization
(`vt::FusedChain`, what `VT_FUSED_CHAIN_ADOPT=1` executes) and the hand-call the
fallback branch reaches (`VT_FUSED_CHAIN_ADOPT=0`) over one synthesized q/k and
requires byte identity, so the documented same-binary A/B lever is a gate and not
a claim. The instrument's own case additionally pins, per run, that the registered
op executed and that the Tier-0 composite did not.

**Mutation.** Each repair is reverted in place, its focused case must redden, and
the file is restored with a sha256 check.

**Suites.** The instrument's CPU-only case (no device), the instrument's device
case under `flock /home/vikash/gpu.lock` with `HIP_VISIBLE_DEVICES=0` and both
settings of `VT_FUSED_CHAIN_ADOPT`, `tests/vt/test_ops_paged_attn.cpp`, and the
production gate `tests/test_rocm_moe_bf16` with `VT_ROCM_MOE_FIXTURE`,
`VT_ROCM_MOE_ORACLE` and `VT_FUSED_CHAIN_ADOPT=1`.

**The instrument's own hygiene.** Every environment-gated case is decorated
`doctest::skip(...)`, so an absent variable reports the case skipped in doctest's
summary and the remaining cases still run; the process exits 77 only when
something was skipped and nothing failed.

## Gates and evidence

* Focused red and green runs, with the exact command and exit status.
* The device run's per-row numbers for C, B, D, D2 and D3, in the same table shape
  as `## Measurement this repair starts from`.
* The two `VT_FUSED_CHAIN_ADOPT` runs' dumped binaries compared byte-for-byte, with
  the adoption flag the only differing field in the report.
* The production gate's recorded tokens, compared to
  `[66,1,70,57,33,81,63,69]` at length 33, concurrency 2, request 0.
* `TMPDIR=/home/vikash/.cache/moe-attn-parity/tmp-preflight
  GIT_CONFIG_GLOBAL=/dev/null GIT_CEILING_DIRECTORIES=$TMPDIR
  PYTHONPATH=/home/vikash/.cache/rdna3-moe-impl/numpy-only-python`, the `python3`
  wrapper adding `--jobs 4` to `check-tree-compiles.py`, stdin closed, with the
  exit line appended to `/home/vikash/.cache/moe-attn-parity/preflight-repair.log`.

## Owed

* **The two arms with no capture to measure them against.** The decode arm
  (`query_len == 1` → Triton `kernel_paged_attention_2d` tiled by
  `min(block_size, 128)`) and the chunked arm's context phase
  (`prefix_prefill._fwd_kernel` tiled by `TRITON_BLOCK_SIZE = 32`) are implemented
  from the pinned source and measured against a host transcription of it on
  synthesized data, but **no primary capture exists for either**: the primary
  attention capture is step-0 only, so every recorded row has `query_len > 1` and
  an empty context, and the capture recipe that produced it is gone — gap G2 in
  `/home/vikash/.cache/attn-parity-plan/PLAN.md` (`run-operator.py` reads
  `attention-capture-recipe/command.json` and `rocm_moe_oracle_attention_capture.py`,
  neither of which exists, and the oracle worktree the argv binds is gone too).
  A decode-step or a chunked-context capture would need a rewritten recipe; until
  one exists, these two arms are implemented-but-unmeasured against the primary.
* **A batch that mixes the arms.** The arm is per row, so one launch can serve
  `query_len > 1` rows through the chunked geometry and `query_len == 1` rows
  through the decode geometry, which is what the primary does too
  (`filter_by_query_len`). Each arm is pinned on its own batch here and the choice
  is a per-row branch, but no case builds a mixed batch, so that combination is
  implemented and unmeasured.
* **The sliding-window arms.** The walk now anchors its tiles where the primary
  anchors them, but the left bound itself stays this backend's own
  `jmin = p - window_left`, which CUDA shares (`cuda_paged_attn.cu:203-206`) and
  the primary states as `p - j < SLIDING_WINDOW`. No windowed model is measured in
  this row, so whether those two bounds agree is open and untouched here.
* The five unmeasured attention kernels named in `## Sibling sites` keep both
  the f32 probability and the per-key running max; each needs the same arm-tiled
  walk before it mirrors the primary.
* The two rocWMMA attention kernels (`rocm_paged_attn.hip:1283`, `:1754`) keep
  the f32 probability until a gfx1200/1201 host can compile and run them.
* The Tier-0 `kAttnQkNormRope` composite keeps the pre-RoPE BF16 store. It is the
  realization on every backend without a registered fast op, and changing it
  needs an f32 operand slot in the recipe and a mixed-dtype `RopeFromCache`.
* The CUDA, Metal and CPU realizations of both arithmetics are not repaired by
  this change and are not measured here.
* `FastExp`, the warp-strided key order and the reciprocal epilogue remain native
  traits; only a measured residual would justify touching them.
* The decode arm's throughput is not measured. The finer tile synchronizes once per
  `min(block_size, 128)` keys instead of once per 64, so a model with a physical
  block size of 16 or 32 pays more CTA synchronizations per decode step.

## Stop conditions

* If row C does not reach 0, the residual is attributed against the parameterized
  CPU model before any further product edit, or the repair returns with the
  residual reported as an open gap.
* If the production gate's recorded tokens move, the golden is reported as moved
  with the failing positions; it is never silently re-derived.
* If the primary's boundary cannot be reproduced without changing an op contract
  outside this row, the preamble repair stops and returns `BLOCKED` with the
  contract named.
