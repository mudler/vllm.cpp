# SPEC-DFLASH2 W11 — the draft block's attention reaches the FA-2 split-KV lane

Issue: [#1890](https://github.com/mudler/vllm.cpp/issues/1890).
Parent wave spec: [`dflash2-spec-decode.md`](dflash2-spec-decode.md).
Sibling: [`dflash2-spec-as-decode.md`](dflash2-spec-as-decode.md) (W10, #1857) —
the same shape of fix on the TARGET VERIFY lane, and the precedent this wave
mirrors term for term.

## Now

`ACTIVE` — implementation in flight on `row/SPEC-DFLASH2-DRAFT-BLOCK-FA2`.

## The gap, measured (#1890, from the corrected #1857 attribution)

nsys head-to-head against SGLang on the identical checkpoint, `main` @
`1724be38e`, build `/usr/local/vcpp/build7`, **artifact-verified**
(`-DVLLM_CPP_CUTLASS_FETCH=ON`, `CUDA FA2 compiled-arch manifest: [121a]`,
`nm -C … | grep -c SpecDecodeFA2Bf16` = 1). 384 tokens, temperature 0, 106
steps.

| lane | kernel | us/call | calls/step | ms/step |
|---|---|---:|---:|---:|
| target verify (post-W10) | `flash_fwd_splitkv_kernel` | **17.1** | 16.2 | 0.32 |
| draft block | `DFlashPagedBlockAttentionWarpKernel` | **449.7** | 5.1 | **2.29** |
| SGLang, both lanes | `BatchPrefillWithPagedKVCache` | 14.3 | 21.3 | 0.36 |

16.2 + 5.1 = **21.3** attention calls/step on both engines. Same work, same call
count, one of our two lanes **31x** slower per call. It is the second-largest
lever in the residual `+4.72 ms/step (+4.1%)` against SGLang; #1866 (FP8 GEMM
algo selection, +3.04) is the largest and #1867 (radix TopK, +0.65) the third.

**Why the draft lane is slow is not a mystery, and it is not the mask.** The
shipped D14 kernel (`cuda_ops.cu::DFlashPagedBlockAttentionWarpKernel`) launches
`grid(Nq, Hq)` blocks of **32 threads** — one warp per (query row, q-head) — and
each warp walks the **whole** combined key sequence serially, one `__shfl_xor`
butterfly per key. At the published draft shape that is `9 x 32 = 288` warps for
a `C ~ 600-1000` key loop: no KV-dimension parallelism at all, and roughly three
warps per SM on a ~100-SM GB10. The D14 comment records its own measurement
("median ~460 us/call") and the 449.7 above is that number, unchanged — the
warp rewrite removed a `__syncthreads` storm and left the occupancy defect.

Split-KV **is** the missing parallelism, which is exactly the W10 finding one
lane over.

## Why this row owns it

#1890 assigns it to `SPEC-DFLASH2` explicitly, and the draft block forward
(`ForwardPagedBody`, `qwen3_dflash.cpp`) is this row's code. No dense-attn seam
row owns a draft-only kernel.

## THE KV-LAYOUT FINDING (the question #1890 asks first)

> Establish whether the draft's KV layout (block table, page size, head dims,
> dtype) admits the same split-KV launcher; if it does not, say exactly which
> property blocks it rather than forcing a partial lane.

**It does not admit `LaunchSpecDecodeFA2Bf16` as it stands, and the blocking
property is not any of the four named.** It is KV RESIDENCY:

> **The draft block's own (1+k) K and V rows are not in a paged cache.**

`vt::DFlashPagedBlockAttention` takes SEVEN tensors where a paged attention takes
four, and the two extra ones are the point: `block_key` and `block_value`,
`[Tq, Hkv, D]` contiguous, produced **inside** the per-layer loop from that
layer's own hidden (`input_layernorm → qkv_proj → k_norm → RoPE`,
`qwen3_dflash.cpp`, the `ForwardPagedBody` layer loop). The paged store
(`DflashDeviceKVStore::pool_k/pool_v`) holds only the CONTEXT rows, which come
from a **different projection input** — the aux-combined target features through
the shared `hidden_norm` (`PrecomputeContextKVDeviceBf16`). The two are not the
same tensor computed twice; they are two different maps of two different inputs.

`LaunchSpecDecodeFA2Bf16` reads K and V **exclusively** through
`block_table` + `page_block_size` from the paged pool. Handing it
`store.pool_k/pool_v` would not be slow-but-right: every block row would vanish
from the attention and each draft query would attend to context only. That is a
wrong answer, not a slow one, which is why this is a blocker and not a widening.

The four properties #1890 names are all *widenings* rather than blocks, and each
is listed here so the next reader does not have to re-derive it:

| property | draft value | launcher requires | verdict |
|---|---|---|---|
| page size | 16 (`kDflashPageSize`) | `block_size % 16 == 0` | **admits** |
| block table | `[1, max_pages]` i32 identity, `stride[1] == 1` | `[num_reqs, max_pages]`, `stride[1] == 1` | **admits** |
| dtype | bf16 q / bf16 pool / bf16 out | bf16 throughout | **admits** |
| head dim | 128 | hard `d == 256` | widening — `flash_fwd_split_hdim128_bf16{,_causal}_sm80.cu` are both compiled |
| GQA topology | Hq 32 / Hkv 8 (ratio 4) | `hq ∈ {16, 24}` | widening — the ratio arms are an enumerated list, not a kernel constraint |
| mask | `causal=false` on full layers | hard `args.causal` | widening — non-causal split-KV is a compiled instantiation |
| window | SWA layers carry `sliding_window` | hard `!window_size.has_value()` | widening — `LOCAL_SWITCH` instantiates `Is_local` |

**The route that closes it** is therefore not "call the W10 launcher". It is:
put the block K/V **into** the pool, and then the draft block attention IS a
paged attention — and can be spelled with the ops this tree already ships.

## Upstream anchors

Unchanged from the parent wave: `backend.py:718-736` @ `b389ac2946` (the reorder
threshold `SpecAsDecodeQueryLen` mirrors), `flashinfer.py:852-860`
(`supports_spec_as_decode`), `_make_xqa_draft_block_mask` `:114-140`. This wave
adds no policy. The DFlash2 architecture itself remains BEYOND-PIN
(`vllm/model_executor/models/qwen3_dflash2.py` @ vllm-project/vllm#52816 head
`19c9351904df4c63042671bc67a866ca48dc7d6f`), and this wave does not advance the
parity pin — it changes which of our own ops serve a forward whose math is
already ported and gated.

The structural claim it mirrors is upstream's own: a draft block attention over
a paged context is `append_paged_kv_cache` followed by a paged attention, which
is why SGLang issues ONE kernel for both lanes.

## Design

### 1. The draft block attention is a paged attention (model side)

Two existing ops replace one bespoke op, per layer:

```
vt::ReshapeAndCache(q, k3, v3, pool_k[l], pool_v[l], slot_map);   // write [C, C+Tq)
vt::PagedAttention(q, a3, q3, pool_k[l], pool_v[l], block_table, seq_lens_ext,
                   cu_seqlens, pa);                               // read  [0, C+Tq)
```

This is what upstream does. SGLang's draft attention is
`append_paged_kv_cache` + `BatchPrefillWithPagedKVCache` — the 21.3 calls/step
in the table above are ONE kernel serving both lanes precisely because both
lanes are paged reads. Our two-lane split exists only because the draft block's
K/V never entered a page.

**The mask maps exactly, with no new mask code.** `PagedAttentionArgs` already
carries FlashAttention's bottom-right alignment: for query token `local` of a
request with `query_len` rows and `seq_lens[r]` keys, the absolute position is
`p = seq_len - query_len + local` (`ops.h`, the `window_size` comment; the CPU
arm computes exactly this in `cpu_paged_attn.cpp`). With `seq_len = C + Tq` and
`query_len = Tq` that is `p = C + local`, which is `ii_comb` in
`DFlashPagedBlockAttentionKernel` verbatim. So:

| draft layer mode | `DFlashPagedBlockAttentionArgs` | `PagedAttentionArgs` |
|---|---|---|
| full attention | `causal=false` | `causal=false`, `window_size=nullopt` |
| SWA, `W > 0` | `causal=true, sliding_window=W` | `causal=true`, `window_size={W-1, 0}` |
| plain causal | `causal=true, sliding_window<=0` | `causal=true`, `window_size=nullopt` |

`causal=false` with a window maps to `nullopt` too, because the block kernel
ignores `window` when `!causal` (`jlo` is guarded on `causal && window > 0`) and
the paged arm has to resolve the SAME mask or the two routes diverge. That is
the whole content of the row: it says what the two routes must agree on, not
what the right answer is.

**An earlier revision of this spec said more than that, and the more was
false.** It read "`causal=false` with a window is not a case", which asserts that
dropping the window is correct. Against the pin it is not:
`vllm/model_executor/models/qwen3_dflash.py:89-146` resolves the window and the
causal flag as two INDEPENDENT answers and `:221-234` passes
`per_layer_sliding_window` into `Attention` irrespective of `causal`, so a
non-causal SWA layer attends within its window. This repository's own loader
already says the same in prose
(`src/vllm/model_executor/models/qwen3_dflash_weights.cpp:181-183`), and the
kernels and that comment cannot both be right. It is a repo-wide property of our
attention kernels rather than anything W11 introduced — the same
`causal && window > 0` guard stands at `src/vt/cpu/cpu_ops.cpp:2917,2994` and at
nine sites in `src/vt/cuda/cuda_ops.cu`. Filed as
[#1900](https://github.com/mudler/vllm.cpp/issues/1900), owed below, and NOT
fixed here: changing it moves every attention kernel in the tree and needs its
own red-before evidence. When #1900 lands, this row and the byte-identity case
that pins it survive unchanged — both routes move together.

**The pool write is safe, and it is not new state.** Slots `[C, C+Tq)` are
beyond `seq_lens`, so nothing reads them as context. Their only other writer is
`ScatterProjectedContextRows`, which writes accepted rows at exactly
`[num_ctx, num_ctx+count)` and only then advances the device `seq_lens`.
Speculative bytes left in those slots are therefore always overwritten before
they can be read, in the one order the store allows.

**Capture safety is unchanged in kind.** Two more persistent device buffers join
the ones the graph already reads in place:

- `g_slot_map` — i64 `[Tq]`, refreshed OUTSIDE any capture to `[C, C+Tq)`;
- `g_seq_ext` — i32 `[1]`, refreshed OUTSIDE any capture to `C + Tq`.

Both sit beside `g_dpos`, which the driver already refreshes in place each step.
The captured graph keeps reading the store's own `seq_lens` for nothing and
`g_seq_ext` for the attention bound; the growing context still enters purely as
a device VALUE, so one captured graph still replays as the context grows.

### 2. The admission (host-testable, `qwen3_dflash_internal.h`)

`ClassifyDflashBlockAttn(DflashBlockAttnEligibility) -> DflashBlockAttnRoute`,
pure and host-compilable, mirroring `vt::PagedAttnUniformSpecShape` and
`ClassifyDenseFa2` (#1879):

- `kPagedSeam` requires: `num_reqs == 1`, `tq > 1`, `block_size > 0`,
  `head_dim > 0`, `hkv > 0`, `hq % hkv == 0`, bf16 query/pool/out,
  `ctx_len >= 0`, `ctx_len + tq <= max_pages * block_size` (capacity — the
  write must fit AND the block table must reach the last position the extended
  `seq_lens` declares), `block_table_col_stride == 1`, and `enabled` (the env
  switch).
- `kBlockKernel` otherwise, which is the shipped path byte-for-byte.

The predicate deliberately does NOT read the FA-2 lane's own conjuncts. The
route is correct on every backend (that is what makes it CPU-gateable and
bit-identical, below); WHICH kernel serves it is the CUDA dispatch's business,
exactly as `uniform_spec_query_len` is "a ROUTING HINT, not a semantic change"
(`ops.h`).

The route is part of the CAPTURED SHAPE: `DflashDeviceKVStore::g_route` records
the classification the graph was captured under, and a step that classifies
differently resets and recaptures — the same handling `g_final_hidden` already
gets. Without it a store that fills to `max_slots - Tq` would flip route under a
live graph.

### 3. The CUDA lane (`cuda_paged_attn.cu`, `cuda_flash_attn_fa2.cu`)

`fa2_spec_decode` gains a second, ADDITIVE arm — the DRAFT BLOCK arm — gated by
its own switch `VT_FA2_DFLASH_BLOCK`, so the shipped W10 d256 verify arm is
dispatch-identical:

```
fa2_dflash_block =
    PagedAttnUniformSpecShape(num_tokens, num_reqs, args.uniform_spec_query_len)
    && d == 128 && block_size % 16 == 0 && block_table.stride[1] == 1
    && num_kv_heads > 0 && hq % num_kv_heads == 0
    && (no window OR (causal && window.left >= 0 && window.right == 0))
    && bf16 q/kv/out && Fa2DflashBlockEnabled()
```

The window term is a REFUSAL, not a convenience. `PagedAttentionArgs`
intersects the window with the causal bound, so `causal && right > 0` still
means `j <= p`, while FA-2's local mask would honour the window's right edge
instead. No draft layer produces that shape, and refusing it is cheaper than
encoding the intersection in the launcher.

and `LaunchSpecDecodeFA2Bf16` widens from "d256, causal, no window" to a head
dim (128 or 256) plus one of THREE mask presentations, each a compiled
instantiation of the vendored split-KV kernel:

| draft layer mode | `is_causal` | `window_left` | `window_right` | dispatch |
|---|---|---:|---:|---|
| full | false | -1 | -1 | `<bf16, D, false>` (`Is_local` false) |
| SWA `W>0` | false | `W-1` | 0 | `<bf16, D, false>` (`Is_local` TRUE) |
| plain causal | true | -1 | 0 | `<bf16, D, true>` |

`is_causal=false` for the windowed arm is REQUIRED, not a choice:
`LOCAL_SWITCH` is `(left >= 0 || right >= 0) && !Is_causal`
(`flash_fwd_launch_template.h`), and the kernel is instantiated with
`Is_local && !Is_causal`, so a causal dispatch carrying a left window would
compile the mask away and silently ignore the window. The FA-2 local mask is
bottom-right aligned against `seqused_k`, which is the same alignment
`PagedAttentionArgs` documents, so the three rows above are the same three rows
as the mask table in §1.

`kBlockN` is 128 at head dim 128 and 64 at 256 (`set_params_splitkv`), which is
the only other head-dim-dependent constant in that launcher.

### Options considered and rejected

- **`Append_KV` (the vendored `mha_fwd_kvcache` in-kernel append).** The
  vendored splitkv kernel DOES support `knew_ptr`/`vnew_ptr` with a
  `block_table` (`flash_fwd_kernel.h`, and `BOOL_SWITCH(params.knew_ptr !=
  nullptr, Append_KV)` in `flash_fwd_launch_template.h`), and it would fold the
  pool write into the attention launch. Rejected: no launcher in this tree has
  ever set `knew_ptr`, the presentation needs the `cu_seqlens_k`-as-lengths mode
  (`is_seqlens_k_cumulative=false`, `block_info.h`) that no launcher here uses
  either, and the append's interaction with `num_splits > 1` is untested. It
  buys one kernel launch and costs the entire CPU-gateable bit-identity
  argument, because CPU has no such op. `vt::ReshapeAndCache` is 1.4 us/call in
  the same profile.
- **Widening the D14 warp kernel with a KV-split.** That is writing a third
  flash kernel by hand next to two the tree already has, against
  `AGENTS.md ## Shared seams` ("Never write a parallel path by hand").
- **Leaving the bespoke op and only speeding it up.** Same objection, and it
  keeps the draft off every future lane improvement the verify gets.

### Scope boundaries

OUT: any change to the W10 d256 verify arm; the DFlash1 materialized
`[context;block]` path (`VT_DFLASH_PAGED=0`); the multi-request
`ForwardWithCtxKVDev` path; the D2 context-free `DFlashBlockAttention` op; any
kernel authored here; `#1866`/`#1867`; any speed claim.

`vt::DFlashPagedBlockAttention` and both its kernels STAY. They are the
rollback arm (`VT_FA2_DFLASH_BLOCK=0`) and the bit-identity reference the new
route is gated against.

## Numerics

**WHICH GATE CARRIES THE NUMERICS CLAIM.** The byte-for-byte op equivalence in
`test_qwen3_dflash_block_route`, and nothing else. The drafted-token comparison
through the production runner does NOT carry it: that fixture drafts a constant
(#1894), so it is a reachability and inertness gate, and its own case says so.

**CPU: bit-identical, by construction and by gate.** `PagedAttentionKernel`
(`cpu_paged_attn.cpp`) and `DFlashPagedBlockAttentionKernel` (`cpu_ops.cpp`) are
the same three-pass online softmax in the same order: `dot` accumulated over `e`
ascending in f32, `dot *= scale`, running max over `j` ascending, `exp(p - m)`
summed over `j` ascending, `inv = 1/denom`, `acc[e] += (prob*inv) * v` over `j`
ascending, stored at the out dtype. The bf16 widening is `BF16ToF32` on both
sides (`LoadF32` / `KvElem<kBF16>`, `WidenRowToF32` for the query row) and the
store is `F32ToBF16` on both (`StoreF32` / `StoreRowF32`). `ReshapeAndCache` is
a raw element copy, so the block rows land in the pages as the identical bf16
bits the block tensor held. Same operands, same order, same widths ⟹ same
bytes. **This is asserted, not argued**: the wave's cross-arm case runs one
store through both routes and requires byte equality on the logits.

**CUDA: the same near-tie class the lane already carries, and one polarity
better.** The shipped default is ALREADY not bit-identical to the block kernel —
D14's own comment says so ("NOT bit-identical … the head_dim partial-sum
grouping over 32 lanes differs"), and `VT_DFLASH_ATTN_BLOCK=1` is the existing
bit-identical rollback for exactly this op. FA-2 reassociates the QK^T/PV
reductions and, at `num_splits > 1`, the split combine reorders f32 partials —
the same class W10 shipped and `VT_FA2_DECODE_GQA_SWAP` before it.

**Spec-decode output is exact by construction under greedy verify**: the target
verify is untouched, so only WHICH proposals are accepted can shift, inside the
ratified ±4 acceptance gate. The GPU token battery is nonetheless OWED below —
"exact by construction" is an argument, and this repository does not accept an
argument where a measurement is possible.

## Risks

**The route does not ask whether the FA-2 lane will serve it, and that is a
deliberate choice with a cost.** `ClassifyDflashBlockAttn` reads shape, dtype
and capacity only. On a CUDA build where the draft-block arm cannot engage — FA2
compiled out, `VT_FA2_DFLASH_BLOCK=0` at the vt layer, or a topology the
admission refuses — the routed read is still CORRECT, but it lands on the
prefill class (`num_tokens > num_reqs`, head dim 128, so the correctness-grade
CUDA-core flash) rather than on the warp kernel it left. Whether that is faster
or slower than 449.7 us/call is **not measured**, and this wave does not claim
it either way.

The alternative was to thread the lane's own conjuncts into the model-side
predicate, as #1865's repair did for the dtype selection. It was rejected here
because those conjuncts are CUDA-only: making them part of the route would take
the CPU arm off the seam too, and the CPU arm is the entire byte-for-byte
equivalence gate. The mitigation is the switch — `VT_FA2_DFLASH_BLOCK=0`
restores the bespoke op in the same binary — and the owed A/B is what decides
whether a narrower admission is wanted. Reconciling it onto a threaded
`fa2_platform` flag is a follow-up if the measurement asks for one, not a
silent widening now.

## Reachability

Production entry point: `include/vllm.h` → the server's spec-decode step →
`runner.cpp` `Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV` →
`ForwardPagedBody` → the routed attention. The wave's chain case drives the
production DFlash2 runner fixture (CPU) and reads the route counters, so
deleting the routing call site reds it. Recorded as mutation M6 below.

## Tests

- `tests/vllm/models/test_qwen3_dflash_block_route.cpp` (new):
  1. **classifier** — the admission table, each conjunct falsified in turn
     (the switch, num_reqs, tq, all three dtypes, GQA divisibility, block-table
     column stride, block size, head dim), plus the capacity term at BOTH
     edges: exactly-full admits and one row over refuses, so a threshold moved
     either way fails one of the two.
  2. **equivalence, BYTE-FOR-BYTE** — six cases covering every (causal,
     window-present) pair the DFlash layer resolver produces: non-causal with
     no window, causal-SWA with a BINDING window, plain causal, and
     NON-CAUSAL WITH A WINDOW — the pair `z-lab/Qwen3.8-27B-DFlash2` and this
     repository's runner fixture actually resolve (`sliding_attention` layers
     plus a top-level `is_causal: false`), and the one the first battery
     omitted. Plus GQA and MHA, a page-straddling context, a page-aligned
     single-page context, and an EMPTY context. Each compares
     `vt::DFlashPagedBlockAttention` against the
     `ReshapeAndCache` + `vt::PagedAttention` pair element-for-element and
     reports the first differing index. The pool rows the speculative write
     lands on are POISONED beforehand, so a route that quietly attends over its
     own scratch is caught rather than averaged away.
- `tests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp` (extended):
  3. **the chain** — through the production runner fixture, every draft-block
     attention call is on the paged seam, none on the bespoke op, and the total
     is a multiple of the draft's layer count (a route that fired on some
     layers and not others would still move the counter).
  4. **the same-binary A/B** — `VT_FA2_DFLASH_BLOCK=0` reaches the forward,
     moves the lane (proved by the counters, not assumed), completes without
     throwing, and leaves the engine's output unmoved. The case states in its
     own comment that its token comparison is NOT a numerics gate on this
     fixture, and why (#1894).
- RED FIRST, carried by commit order: the tests, the classifier and the
  counters land in one commit with `ClassifyDflashBlockAttn` returning
  `kBlockKernel` unconditionally — the pre-W11 behaviour exactly. Cases 1, 3
  and 4 are red there. The next commit supplies the classifier body and the
  route, and they go green.

Mutations, each restored byte-for-byte against a `sha256sum` manifest printed
after the restore. MEASURED, not predicted:

| # | mutation | op gate | runner gate |
|---|---|---|---|
| M1 | capacity `<=` → `<` (threshold, TIGHTER) | **RED** 1/28 | green |
| M2 | capacity → always true (threshold, LOOSER) | **RED** 2/28 | green |
| M3 | drop the three bf16 conjuncts | **RED** 3/28 | green |
| M4 | window `sliding_window - 1` → `sliding_window` | **RED** 4/28 | green |
| M4b | mask `causal` forced true | **RED** 6/28 | green |
| M5 | neutralise the paged K/V write before the read | **RED** 12/28, 5 cases | green |
| M6 | **reachability** — the routing call site in `ForwardPagedBody` | green | **RED** at `0 == N` |
| M7 | drop the `uniform_spec_query_len` routing hint | green | **RED** at `7 == 23` and `7 > 16` |
| M8 | hand the read the store's `seq_lens` instead of the extended bound | green | **RED**, refuses by name |
| M9 | the slot ARITHMETIC one page up (`ctx_len + i` → `+ 16`) | **RED** 14/30, 6 cases | green |
| M10 | the slot ARGUMENT one page up at both production sites | green | **RED**, refuses by name |
| M10a | the same, at the EAGER site (`qwen3_dflash.cpp:1525`) ALONE | green | **RED** 7/8, refuses by name |
| M10b | the same, at the CUDA-GRAPH REFRESH site (`:1627`) ALONE | green | green — a DIFFERENT pair of suites reds, below |
| M11 | drop the `causal &&` guard in `DflashBlockPagedMaskOf` | **RED** 2/30, 1 case | green |
| M12 | the graph refresh SKIPPED on replay (`if (st.g_state != 2)`) | green | green — and so are the other two, #1902 |

Five of these changed the wave — M5, M6, M7 and, through the fresh review, M9
and M10 — and they are recorded because a mutation that only confirms what you
already believed has told you nothing:

- **M6 first ran GREEN.** The counter was recorded beside the CLASSIFICATION,
  so deleting the whole routed branch still moved it — a counter measuring a
  class rather than a capability, which is the exact defect
  `.agents/reachability.md` exists for. The increments moved inside each branch
  and M6 now reds the production-runner gate.
- **M5 and M8 first ran GREEN**, and the reason is not W11's: this repository's
  DFlash2 runner fixture drafts a CONSTANT (`19 19 19` at all eight steps), so
  ANY drafted-token comparison through it is a tautology against a numerics
  change. Filed as [#1894](https://github.com/mudler/vllm.cpp/issues/1894) and
  owed below. M5 is closed by binding the write and the read into ONE function
  (`DflashBlockPagedAttention`) that the op-level gate calls, so the equivalence
  cases now catch it. M8 is closed by a host-readable invariant check inside
  that function: `seq_ext` must read `ctx_len + tq`, and on a host-addressable
  device it is checked and refused by name.

  **BINDING THE PAIR CLOSED THE CALL-DELETED VARIANT ONLY, AND THE FIRST
  RECORD OF M5 OVERSTATED IT.** The slot map stayed a PARAMETER of that
  function, so the equivalence gate built its own correct one and compared
  against that: the write happened, the read happened, and WHERE it happened was
  ungated. The W11 fresh review measured the cost — moving the production slot
  arithmetic one page (`st.num_ctx + i` -> `+ 16`) at both sites left
  `test_qwen3_dflash_block_route` 28/28 and `test_dflash2_runner_reach` 162/162
  GREEN while every draft query attended to the context plus stale pool rows and
  never saw its own block K/V. A wrong answer on every backend. M9, M10 and the
  remediation below are that finding's repair.
- **M10 WAS RECORDED AS ONE MUTATION AND IT IS TWO, AND ITS TWO-COLUMN ROW
  CREDITS THE WRONG SUITE.** "At both production sites" reads as if the runner
  gate covered both, and it does not. The fresh re-review flagged it, and this
  repair measured the two sites separately on the same build. The EAGER site
  (`qwen3_dflash.cpp:1525`) ALONE produces the whole of M10's recorded red —
  `test_dflash2_runner_reach` 7/8 test cases failed, 17/78 assertions, refusing
  by name from the `VT_CHECK` in `DflashBlockPagedAttention` (M10a); it also
  reds `test_qwen3_dflash_decode_graph_seam` 2/4 and `test_qwen3_dflash2_draft`
  6/43, which reach the same eager lane. The CUDA-GRAPH refresh site (`:1627`)
  ALONE leaves the runner gate FULLY GREEN at 8/8 162/162 and reds a different
  pair: `test_qwen3_dflash_decode_graph_seam` 3/4 and `test_qwen3_dflash2_draft`
  1/43, each THREW rather than CHECK-failed, with the same refusal by name
  (M10b). Both sites are covered. The row credited one suite with both, which is
  the smaller version of the same mistake as the paragraph below.
- **M12 IS THE ONE THAT STAYED GREEN, AND IT IS A REAL GAP.** Skipping the graph
  refresh on replay only — `if (st.g_state != 2) { …Copy… }`, a wrong answer on
  CUDA at every draft step after the first — left ALL FOUR suites green:
  `test_qwen3_dflash_decode_graph_seam` 4/4 23/23,
  `test_qwen3_dflash2_draft` 43/43 449/449, `test_dflash2_runner_reach` 8/8
  162/162, `test_qwen3_dflash_block_route` 13/13 30/30, byte-for-byte the
  baseline. Reproduced here from the fresh re-review's finding, restored against
  a `sha256sum` manifest taken before the mutation. Filed as
  [#1902](https://github.com/mudler/vllm.cpp/issues/1902) and owed below; the
  narrative under F1 says why no CPU gate can close it.
- **M7 WAS RECORDED GREEN/GREEN, AND THAT WAS WRONG.** The reasoning looked
  sound — `uniform_spec_query_len` is a lane selector, inert on CPU by the
  field's own contract ("a backend that ignores the field is still correct",
  `ops.h`) — and it was never MEASURED against this branch's own W10 case. The
  W11 fresh review measured it. Deleting the assignment reds
  `test_dflash2_runner_reach` at
  `CHECK( arrivals == spec_as_decode_steps + paged_seam_calls ) -> CHECK( 7 == 23 )`
  and `CHECK( arrivals > paged_seam_calls ) -> CHECK( 7 > 16 )`, because the
  hint feeds `PagedAttnUniformSpecShape` and that is what gates the classified
  arrival counter (`src/vt/ops.cpp:3883`) on EVERY backend, not only on CUDA.
  The hint is a lane selector on CUDA; the CLASSIFICATION it carries is
  backend-independent, and W10's own repair (#1865) is what made it countable.
  The debt this row recorded under `## Owed` for M7 therefore does not exist and
  is removed. The lesson is the one this table's preamble already states: a
  mutation you predict instead of running has told you nothing, and predicting
  GREEN is the expensive direction.

### The fresh review's three findings, and the remediation each got

The W11 fresh review confirmed the substance — the equivalence is genuinely
OLD-vs-NEW with real poisoning, M4/M5/M6 reproduce, the counters count
execution rather than intent, and the route is reached from `include/vllm.h` on
the default configuration — and returned three findings. All three are repaired
in this branch, and each repair carries its own measured mutation above.

**F1, the slot map (HIGH).** Both production sites spelled the slot arithmetic
out by hand, and `DflashBlockPagedAttention` took the result as a PARAMETER, so
the gate that exists to compare the two routes built its own correct one.

Of the two remediations the review named — extend the host-readable invariant to
cover the slot map, or derive the slots inside the gated function — this branch
takes the SECOND, adapted to the one constraint that rules out the literal form
of it. The gated function cannot fill the tensor itself: on the CUDA-graph path
a replay never calls it, the buffer's ADDRESS is baked into the capture, and the
refresh has to happen outside the captured region. So the derivation moves to
`detail::DflashBlockPagedInputsOf(ctx_len, tq)`, which produces BOTH values from
ONE context length, and the two obligations split cleanly:

- the ARITHMETIC is gated by the byte-for-byte battery, because
  `CheckRouteEquivalence` now builds arm B's tensors from that function while
  arm A derives nothing from it. This is host arithmetic with no device term, so
  a CPU red is a CUDA red (M9);
- the ARGUMENT is refused inside `DflashBlockPagedAttention`, which re-derives
  the canonical pair from the `ctx_len` it reads off the STORE and compares. The
  comparison is between two HOST values, so unlike the pre-existing `seq_ext`
  read it carries NO `kCPU` guard and never dereferences a device pointer: on
  every step that ENTERS that function it holds on CUDA exactly as it holds on
  CPU (M10).

The CPU-readable `seq_ext` check stays and gains a sibling on the slot map's two
ends, which closes a third variant neither of the above sees: the host values
were right and the UPLOAD did not land on the tensor this call reads.

The first remediation was rejected on its own terms rather than on effort: a
host-readable check of a DEVICE slot map needs a D2H copy, which is a
synchronisation per layer per draft step and is forbidden outright inside a
capture. It would have bought a guard that is CPU-only again, which is the
property the finding objected to.

**"EVERY STEP THAT ENTERS THAT FUNCTION" IS NOT EVERY DRAFT STEP, AND THE FIRST
RECORD OF THIS OVERSTATED IT.** The guard is per CALL, not per STEP. On
`st.g_state == 2` the driver calls `st.g_graph.Replay(queue)` and returns
(`src/vllm/model_executor/models/qwen3_dflash.cpp:1636-1660`), so
`ForwardPagedBody` — and `detail::DflashBlockPagedAttention` with it — is
entered on the EAGER lane and on the ONE warm-then-capture step per request, and
on NO replay step. Every draft step after the first is therefore unguarded,
while the persistent buffers those steps read are refreshed by a SECOND
production site (`qwen3_dflash.cpp:1627-1631`) that on a replay step has nothing
downstream to check it. The fresh re-review measured the gap rather than arguing
it: making that refresh skip on replay only (`if (st.g_state != 2) { …Copy… }`),
which on CUDA is a wrong answer at every step after the first, left ALL FOUR
suites green — `decode_graph_seam` 4/4 23/23, `dflash2_draft` 43/43 449/449,
`runner_reach` 8/8 162/162, `block_route` 13/13 30/30.

No CPU gate can close it, which is why this is owed rather than repaired here:
the capture-capable CPU backend's `ReplayGraph` is a log push that executes
nothing (`tests/vllm/models/decode_graph_seam_harness.h:117`), so a CPU replay
step performs no attention at all, right or wrong. The owed proof is a
device-side one. Both remediations that would close it are a redesign of the
capture path with their own red-before evidence, and neither is refused on
effort. Reading `g_seq_ext` back before `Replay` is a D2H SYNCHRONISATION on the
hot draft path — the property the paragraph above already rejected when it was
per-layer, and this one sits on the path W11 exists to make faster. Moving the
refresh inside the guarded function is impossible by construction, because a
replay never calls it and the buffer ADDRESSES are baked into the capture.
Filed as
[#1902](https://github.com/mudler/vllm.cpp/issues/1902) and owed below.

The claim as first written is corrected in two of its three places by this
commit — `src/vllm/model_executor/models/qwen3_dflash_internal.h` and the
paragraph above. The third is the body of commit `638fb4d62` ("it holds on CUDA
exactly as it holds on CPU"), which is history and cannot be edited; this spec
is the correction of record for it, the same way it is for the append-only
`.agents/issue-index.md` rows below.

**F2, M7 (MEDIUM).** Recorded green/green from reasoning, never run. It reds.
The table and the narrative above are corrected and the owed item is deleted.
The `.agents/issue-index.md` row for #1890 also carries the old claim; that file
is append-only, so the row stands and this spec is the correction of record.

**F3, the missing mask (MEDIUM).** The battery covered `(false,0)`, `(true,5)`,
`(true,0)` and the shape variations, but never `causal == false && window > 0` —
which is exactly what the production resolver yields for the published DFlash2
checkpoint. Deleting the `causal &&` guard in `DflashBlockPagedMaskOf` left both
suites green. The new case restores the claim to something true: one case per
(causal, window-present) pair, and the guard-deletion now reds it (M11). The
`.agents/issue-index.md` row for #1890 says "five mask and layout cases"; that
file is append-only too, and this spec is the correction of record for the count
as well as for M7.

The case's own header, and the mask table above, first said MORE than the case
measures: that dropping the window when `!causal` is the correct answer. It is
not, at the pin. Both passages are corrected to describe the byte-identity the
case actually gates, and the kernel behaviour is
[#1900](https://github.com/mudler/vllm.cpp/issues/1900), owed below.

## Gates

First result, on `row/SPEC-DFLASH2-DRAFT-BLOCK-FA2` @ `f39d7ef66` (PR #1896):
full CPU suite **611/611**, and every non-Windows CI check green — 15 pass, 8
skipping, including `cuda-fat-build` (1h53m22s, `success`), which is what
compiled the two `.cu` regions this box has no `nvcc` for.

Result after the fresh-review repair, at `d961aef9b`: full CPU suite
**612/612**, on a tree that also carries `origin/main` @ `749e5c8d9` (#1893,
the FP8 small-M dispatch), which is where the extra case comes from. The build
is CI's own CPU configuration — no `CMAKE_BUILD_TYPE`, so no `NDEBUG` and every
`assert` live — because a Release gate over an assert-firing bug is a latent
failure rather than a pass. The four mutations M7 and M9-M11 were measured on
that same configuration and each restore was verified byte-for-byte against a
`sha256sum` manifest.

Result after the fresh RE-review repair (this commit), on a tree that carries
`origin/main` @ `ced0ab639`: full CPU suite **614/614**. The four
focused suites are `test_qwen3_dflash_decode_graph_seam` 4/4 23/23,
`test_qwen3_dflash2_draft` 43/43 449/449, `test_dflash2_runner_reach` 8/8
162/162 and `test_qwen3_dflash_block_route` 13/13 30/30 — byte-for-byte the
counts the re-review reported, on the same CI CPU configuration (no
`CMAKE_BUILD_TYPE`, every `assert` live). M10a, M10b and M12 were measured on
that build, each restored from a copy taken before the mutation and verified
with `sha256sum -c` against a manifest of all three touched files. This repair
changes no product behaviour: two comment blocks, one `TEST_CASE` name, the
spec, and two appended `.agents/issue-index.md` rows.

`windows-msvc-cpu` and `windows-msvc-vulkan` are red, and they are NOT this
change: the failing step is the Windows focused gate and the failure is
`test_openai_api_server.exe exited with status -1073740791` (`0xC0000409`,
`STATUS_STACK_BUFFER_OVERRUN`), which is
[#584](https://github.com/mudler/vllm.cpp/issues/584) verbatim. Checked rather
than assumed, because `gh pr checks` reports a CANCELLED job as `fail`: over the
last 15 pull requests that job is red on all of them, 11 with
`jobConclusion=failure` and 4 cancelled, and the failure text on #1889 and #1861
— both predating this branch — is byte-identical down to the exit code.

```sh
scripts/agent-preflight.sh
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure -R 'dflash|paged_attn|route'
ctest --test-dir build --output-on-failure          # full CPU gate
python3 scripts/agent-integration.py --base origin/main
```

## Owed (operator-run; this wave claims none of it)

- ~~**The first CUDA compile of the edited `.cu` regions.**~~ **DISCHARGED by
  CI**, and it is the one owed item this wave does not hand on. There is no
  `nvcc` on the implementer's box, so `cuda_paged_attn.cu` and
  `cuda_flash_attn_fa2.cu` were edited and not compiled locally — the debt W10
  (#1858) and its repair (#1879) both carried. The `cuda-fat-build` job compiled
  them on PR #1896 (`f39d7ef66`, 1h53m22s, `success`), across the ten-SM
  gencode set its own audit step pins. A compile is not a run: what remains owed
  is whether the lane ENGAGES and what it costs, below.
- **The GPU number.** `DFlashPagedBlockAttentionWarpKernel` should leave the
  kernel table and `flash_fwd_splitkv_kernel` should absorb its 5.1 calls/step.
  #1890 expects ~2.3 ms/step. **This wave claims nothing.** The build recipe is
  the one the corrected #1857 attribution pinned, and the artifact assertion
  comes BEFORE any timing:
  ```sh
  cmake -DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a \
        -DVLLM_CPP_CUTLASS_FETCH=ON …
  grep 'FA2 compiled-arch manifest' build.log      # must be non-empty
  nm -C build/examples/vllm-server | grep -c SpecDecodeFA2Bf16   # must be 1
  ```
  A profile from a build whose manifest reads `[]` measures the fallback, which
  is the exact failure that retracted the first #1857 table.
- **The `VT_FA2_DFLASH_BLOCK` on/off A/B** on that binary, same-binary.
- **The GPU token battery** — DFlash / DFlash2 / DSpark greedy fixtures, route
  ON vs OFF, on the engaged lane. CPU byte-identity is gated here; the GPU arm
  is a near-tie class and needs the ±4 acceptance gate run.
- **The `num_splits` heuristic is sized off the store's CAPACITY, not its
  occupancy.** `LaunchSpecDecodeFA2Bf16` feeds `max_seqlen_k = max_blocks *
  page_size`, which for the draft store is a fixed 4096 whatever `C` currently
  is — the same property the W10 verify arm already has. At `Tq = 9`, batch 1
  and 32 CTAs per split the heuristic will split, possibly further than a
  ~600-1000-key context wants. Whether that costs anything, and whether the
  split combine is entered at all on this shape, is a measurement rather than a
  derivation; nothing here is tuned on it.
- **[#1894](https://github.com/mudler/vllm.cpp/issues/1894) — the DFlash2
  runner fixture drafts a CONSTANT**, so every drafted-token comparison through
  it (including the landed W8 lane-comparison case) is a tautology against a
  numerics change. Found by this wave's mutation pass, filed, and NOT fixed in
  flow: changing the fixture's weights moves five landed cases at once and needs
  its own red-before evidence, which is a different unit of work. Owned by
  `SPEC-DFLASH2`.
- **[#1902](https://github.com/mudler/vllm.cpp/issues/1902) — the paged-seam
  guard never runs on a REPLAY step**, so the CUDA-graph refresh site
  (`qwen3_dflash.cpp:1627-1631`) is unguarded on every draft step after the
  first, and the reviewer's skip-on-replay mutation left all four suites green.
  Not closable on CPU — the capture-capable CPU backend's `ReplayGraph` executes
  nothing — so the owed proof is DEVICE-SIDE: a CUDA run over at least two draft
  steps that reds on a stale or absent replay-step refresh, or a test backend
  whose replay re-executes the recorded work. Found by the fresh re-review of
  this wave, filed, and NOT fixed in flow: both remediations redesign the
  capture path, and one of them puts a per-step D2H synchronisation on the path
  this row exists to make faster. Owned by `SPEC-DFLASH2`.
- **[#1900](https://github.com/mudler/vllm.cpp/issues/1900) — non-causal SWA
  layers drop their window in our attention kernels**, while upstream attends
  within it (`vllm/model_executor/models/qwen3_dflash.py:89-146,221-234`) and
  this repository's own loader comment says upstream's answer
  (`qwen3_dflash_weights.cpp:181-183`). Repo-wide and pre-existing —
  `src/vt/cpu/cpu_ops.cpp:2917,2994` plus nine sites in
  `src/vt/cuda/cuda_ops.cu` — and surfaced by this wave only because W11 wrote a
  normative claim about it into this spec and into
  `test_qwen3_dflash_block_route.cpp` without an upstream anchor. Those two
  passages are corrected here to describe what the battery MEASURES (the two
  routes agree byte for byte under the kernel's current mask semantics) rather
  than to assert that those semantics are right. The kernel change itself is NOT
  fixed in flow: it moves every attention kernel in the tree and it changes
  drafted tokens on the published DFlash2 checkpoint, which is its own unit of
  work. Owner is `SPEC-DFLASH2` or the attention-kernel row, to be picked by
  whoever takes it.

## Stop conditions

- If any CPU token fixture moves: STOP. The route was claimed bit-identical and
  is not; report `NEEDS_DECISION` with the first differing byte.
- If the GPU battery shows a token divergence the ±4 acceptance gate does not
  cover: STOP with the measured divergence, do not widen the gate.
- If the re-profile shows `DFlashPagedBlockAttentionWarpKernel` still in the
  table on a manifest-verified build: the classification is not reaching the
  dispatch; `NEEDS_DECISION` with the narration line attached, no guessing.
