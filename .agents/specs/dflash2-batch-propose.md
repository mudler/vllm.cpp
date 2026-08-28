# SPEC-DFLASH2 W12 — the batched propose: why c=4 -> c=8 is flat, and what would fix it

**Row:** `SPEC-DFLASH2` (wave W12, after W11
[dflash2-draft-block-fa2.md](dflash2-draft-block-fa2.md) and the #2010 repair).
**Issues:** [#2087](https://github.com/mudler/vllm.cpp/issues/2087) (the wave),
[#2088](https://github.com/mudler/vllm.cpp/issues/2088) and
[#2089](https://github.com/mudler/vllm.cpp/issues/2089) (found in the same read,
owned by the parent spec's `## Owed`),
[#2111](https://github.com/mudler/vllm.cpp/issues/2111) (D2's bound, and the
1.78x premise it was aimed at).
**Parent spec:** [dflash2-spec-decode.md](dflash2-spec-decode.md).
**Kind:** this document is the INVESTIGATION and the wave's scope. No product
code lands with it: every claim it makes about cost is arithmetic over code that
was read, and the row's own rules make a throughput claim a device measurement.
`## Gates` names the runs that would settle it.

## Why

Measured 2026-08-27 on `dgx:gpu0` (GB10), idle, under `rc` leases.
Qwen3.8-27B NVFP4 + DFlash2 k=8, 1024 in / 512 out, `--max-num-seqs 16`,
`vllm bench serve --dataset-name random --backend openai-chat`. Ours on `main`
after #1994/#1997/#2000/#2010:

| c | ours out tok/s | vLLM | SGLang |
|---|---|---|---|
| 1 | 25.14 | 24.36 | 25.20 |
| 2 | 40.12 | 38.59 | 44.93 |
| 4 | 60.25 | 64.25 | 77.06 |
| 8 | **63.3** | **80.0** | 109.24 |

c=4 -> c=8 we gain 5%, vLLM gains 25%, SGLang 42%. The c=8 rung is controlled:
ours is 4 runs (61.76, 62.64, 63.35, 65.51; spread 5.9%), vLLM 3 runs (78.12,
79.97, 82.00; spread 4.8%), so the 21% deficit clears both spreads by 4x.

**THE RESOLUTION FLOOR AT c=8 IS ~6%, AND IT BINDS THIS WHOLE DOCUMENT.** Our
own spread is 5.9% and vLLM's is 4.8%, so no effect below roughly 6% is
readable at that rung and no experiment here may be scored on one. Several
attributions on this row have already died to sub-noise effects. SGLang's
109.24 is **n = 1** and carries no error bar at all; it bounds the ambition, it
cannot settle a comparison.

**The rung we LEAD is c=1, and it is the only rung that takes the fast path.**
That is the whole finding in one sentence.

## What the code does

`src/vllm/v1/worker/gpu/runner.cpp:3378` is the ONLY production caller of
`Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV` (the other call sites are
tests). It builds `stores` with one entry per PROPOSING ROW
(`runner.cpp:3345-3355`), so `P` equals the number of rows that proposed this
step — 1 at c=1, and up to 8 at c=8.

The fast path is gated on `P == 1`
(`src/vllm/model_executor/models/qwen3_dflash.cpp:1577`). It runs
`ForwardPagedBody` (`:1461-1508`): `Tq = 1 + k = 9` query rows read the
request's persistent paged store through the shared paged seam
(`DflashBlockPagedAttention`, `qwen3_dflash_internal.h:222-287`), and the whole
step is CUDA-graph captured and replayed (`qwen3_dflash.cpp:1602-1604`,
`:1862-1886`).

Every `P > 1` step falls to `qwen3_dflash.cpp:1888-1930`, whose own comment
says it is "not capture-targeted":

1. `2 x L` context buffers of `[C, kdim]` are allocated, where `C` is the SUM of
   every proposing row's full context length (`:1892-1895`);
2. every context row of every request is gathered out of its paged store with
   `4 x P x L` `IndexSelect`/`IndexCopy` launches (`:1908-1922`);
3. `ForwardWithCtxKVDev` (`:664`) runs. Per LAYER it allocates `qcb`, `kcb`,
   `vcb` (`:792-794`) and `acomb` (`:811`) of `[Ncomb = C + Tq, ...]`, memsets
   `qcb` (`:795`), scatters the context K/V in (`:800-807`), and calls
   `vt::DFlashBlockAttention` (`:818`).

`vt::DFlashBlockAttention`'s CUDA grid is over `t = query.shape[0]`
(`src/vt/cuda/cuda_ops.cu:2634`, `:2643`, `:2650`), i.e. over all `Ncomb` rows.
**The draft computes an attention output for every context row of every request
in the batch and then throws them away** — `qwen3_dflash.cpp:820-827`
`IndexSelect`s only the `Tq` block rows back out. The `qcb.Zero` at `:795` is
the tell: the comment beside it says "context query rows are unused (their attn
output is discarded)", and they are still computed.

Every draft layer of the campaign subject is non-causal (`is_causal false`,
5 layers, hidden 5120 — `dflash2-spec-decode.md`), so `jhi = qe - qs - 1` and
the window is not applied (`cuda_ops.cu:1580-1582`, `:2402-2409`, `:2435`; that
window drop is #2088, filed separately). Each of the `Ncomb` query rows
therefore attends over its own request's ENTIRE span.

| route | attention pairs per layer per step |
|---|---|
| `P == 1` paged | `(1 + k) x C` |
| `P > 1` fallback | `sum_r (ctx_r + 1 + k)^2` |

At `ctx_r ~ 1300` and `k = 8` that is a ~150x blow-up PER ROW, on top of `O(C)`
bytes of gather, memset and index traffic per layer, and the loss of the
CUDA-graph lane. It enters at c=2 and grows with `c` because `C` does.

## Why three profiled waves did not see it

This is not a subtle cost. It is a ~150x term on the production path, and W9,
W10 and W11 each profiled the draft phase without reporting it. The reason is
[#2089](https://github.com/mudler/vllm.cpp/issues/2089), and it is worth stating
as a lesson rather than as a defect line.

W11 added `DflashBlockRouteStats` so a gate could assert which attention lane a
draft block took, and deliberately put the counter INSIDE the branch that runs
rather than beside the classification, because "a counter measuring a class, not
a capability" is the failure it exists to avoid
(`src/vllm/model_executor/models/qwen3_dflash.cpp:1486-1493`). That argument was
right and it was applied to one branch. **Both increments sit inside the
`P == 1` branch** — `qwen3_dflash.cpp:1487` (`kPagedSeam`) and `:1502`
(`kBlockKernel`). The `P > 1` fallback at `:1888-1930` increments neither.

So at every concurrency above one the route counters read ZERO for both lanes
while production runs a third route that nothing names. Every wave that
profiled this path drove it at c=1 — the `## Owed` instrument in W9, the c1
speed gate, the e2e gate, which drives one request at a time
([dflash2-draft-fixed-cost.md](dflash2-draft-fixed-cost.md) Lever B is an
explicit "per step at P=1" census) — and at c=1 the instrument reports a lane
that is genuinely fast.

**An instrument that only counts the fast path cannot report that a slow path
exists.** It is the `.agents/verification.md` weak-gate shape from the other
side: not a gate that passes a wrong artifact, but a counter whose zero is
indistinguishable from "this lane did not run" and from "this lane was never
asked". #2089 lands with or before the D1 change below, or the change cannot be
gated.

## What it is NOT

**Not the scheduler's CPU cost, and the mirror claim needed qualifying.** The
propose runs inside `execute_model` (`runner.cpp::propose_drafts_block`), after
the schedule is fixed; `P` is a count of proposing rows and nothing in the
scheduler chooses the draft route.

The prior finding — `.agents/parity-ledger.md:467`, "our V1 waiting-queue
admission + token-budget accounting is a faithful 1:1 mirror of pinned vLLM's" —
was re-read against the pin for this wave and is **partially refuted**. Its own
scope line is honest (waiting loop plus two defaults, measured with NO
speculator), but three specs restate it without that scope, and on a spec-decode
run two upstream mechanisms are absent: `pad_spec_decode`
(`vllm/v1/core/sched/scheduler.py:826-843`, `:1022-1025`) and the dynamic-SD
lookup (`:1122-1125`). Both were already recorded as deferrals in
`include/vllm/v1/core/sched/scheduler.h:54-56` with no issue; filed as
[#2090](https://github.com/mudler/vllm.cpp/issues/2090).

Also verified: no `O(num_running^2)` term exists in our `schedule()` that
upstream lacks. Every running-set touch is `O(N)` and matched
(`scheduler.cpp:488-635`, `:556-562`, `:781`, `:1128-1135`); the added constants
are `std::map` string compares at `N = 8`, microseconds. **Scheduler CPU cost is
not the stall.**

`pad_spec_decode` is a REAL divergence and is nonetheless **inert for this
ladder**: it fires when a newly admitted request has `num_new_tokens == 1`,
which is a full prefix-cache hit, and the #1574 recipe runs
`--no-enable-prefix-caching` (prefix caching plus DFlash2 also kills the engine
at c=1, [#2042](https://github.com/mudler/vllm.cpp/issues/2042)). Recorded so
the next reader does not chase it, and measurable by E6.

**Not the allocations.** #2010's author flagged `qwen3_dflash.cpp:1888-1927` as
a per-step device-allocation burst. `DBuf` draws from the shared size-class
`DevicePool` (`include/vllm/model_executor/models/dense_device_glue.h:109-127`,
`include/vllm/model_executor/models/device_pool.h`), which is UNCAPPED on GB10
(`include/vllm/platforms/interface.h:118`), so after warm-up these are pool hits
and not `cudaMalloc`/`cudaFree` syncs. The cost is the WORK, not the allocation.
The observation was right about the line and wrong about the mechanism.

**Not the draft-context fallback (#1943).** At `--max-num-seqs 16` the store
sizing resolves 8 GiB / 16 = 512 MiB per request
(`qwen3_dflash.cpp:1032`, `:1140-1180`), which at 5 layers holds far more than
`max_model_len`, so no row reaches `ctx.disabled` at 1024 + 512 tokens
(`runner.cpp:3188-3200`). Confirm with the startup sizing line before relying on
this.

**Not #1867.** `TopKValuesIndicesRowKernel` is 708 us/step at 8 rows on ~48 SMs;
at c=8 it has 64 rows and MORE parallelism to absorb, so it is a per-step
constant, not a scaling term. Worth fixing; not this.

## Design — what would fix it

Batch the paged propose so `P > 1` takes the route `P == 1` takes: attention
computed for `P x (1 + k)` query rows against per-request paged context, and
NEVER for the context rows.

- **D1 — narrow.** Keep the materialized combined K/V, but give
  `vt::DFlashBlockAttention` a separate QUERY cu so `Q` is `[Tq, ...]` while
  `K`/`V` stay `[Ncomb, ...]`. This deletes the `Ncomb`-sized `qcb` and `acomb`,
  the memset, the query `IndexCopy` and the output `IndexSelect`, and ~99% of the
  attention work. It touches one `vt` op signature and its CPU and CUDA kernels.
  It leaves the `O(C)` gather (step 2 above) in place.
- **D2 — full.** One shared paged pool for every request's draft context, a
  batched block table and per-request `seq_lens`, so the batched propose IS the
  `P == 1` path with `num_reqs > 1`. `vt::DFlashPagedBlockAttention` already
  carries `pa.num_reqs` (`qwen3_dflash.cpp:1500`, set to 1 today). This
  additionally deletes the gather, and it is the same unified pool
  [#2007](https://github.com/mudler/vllm.cpp/issues/2007) needs, so the two rows
  should agree on the allocation before either lands.
- **D3 — REJECTED: loop the `P == 1` path per request.** It re-reads the draft's
  ~1.5 GB of weights and the ~0.72 GB packed head once per row
  ([dflash2-draft-fixed-cost.md](dflash2-draft-fixed-cost.md) Lever A), which at
  P=8 is ~18 GB/step of weight traffic. The whole point of a batch is to read
  the weights once.

D1 first: it is the smaller change, it is CPU-gateable byte-for-byte against the
current kernel (same online softmax over the same rows, minus the rows whose
output is discarded), and it settles whether the attention is the term before
the pool work is designed.

## Gates

Correctness first, and none of these can be run without the box.

1. **CPU byte-for-byte.** D1's output over the `Tq` block rows must equal the
   current path's `IndexSelect`ed rows, bit for bit, on the CPU backend, over
   the existing `P > 1` fixtures in
   `tests/vllm/v1/spec_decode/test_dflash_propose.cpp:335`, `:373`.
2. **Red-first.** Delete the query-cu argument's effect (make `Q` span `Ncomb`
   again) and the focused gate must go red on COST, not on tokens — which it
   cannot, so the red-first case is a launch-shape assertion: the attention op
   must be called with `query.shape[0] == Tq`, and a production-runner test
   driving `P > 1` must assert it.
3. **Route counter.** #2089 must land with or before this, or the gate cannot
   see which lane the batch took.
4. **Device throughput.** The c=1/2/4/8 ladder rerun on one binary, idle,
   reproduced 2-3x, against the same vLLM and SGLang denominators.

   **The vLLM denominator is legitimate, and 21% is a FLOOR on the gap rather
   than a ceiling.** AGENTS.md forbids `--enforce-eager` as a denominator, so
   the 80.0 was checked against that rule before it was used.
   [#2039](https://github.com/mudler/vllm.cpp/issues/2039) establishes that
   vLLM **cannot** capture CUDA graphs for a DFlash2 draft: it dies in
   draft-graph capture with a `ConstraintViolationError`
   (`qwen3_dflash2.py:277`) and the engine never boots. `--enforce-eager` is
   therefore FORCED and not chosen, which makes it that engine's production
   configuration for this model and a valid denominator. The sharpening runs
   our way: a graphed vLLM would be faster still, so the 21% deficit at c=8 is
   a lower bound. Do not restate it as "the gap".

## The discriminating experiments, in priority order

None of these needs a code change. Each is stated with what it shows if the
hypothesis is TRUE and if it is FALSE.

**E1 — the phase split.** Run c=1, c=4 and c=8 with `VT_SPEC_TRACE=2`. The
runner prints `[spec-phase-dev] pre= fwd= select= walk=` per step
(`runner.cpp:3423-3428`, `:3441-3446`), with a queue drain at each seam.

- TRUE: `fwd` at c=8 is roughly 2x `fwd` at c=4 and many times the c=1 value,
  and it dominates the step. That is the fallback's `O(C)` attention.
- FALSE: `fwd` is flat across c and the growth is in `pre`, `select`, `walk`, or
  outside the draft phase entirely. Then #2087 is a real defect but not this
  stall, and the next suspect is the verify batch.

**E2 — speculation off.** Run the c=1/2/4/8 ladder with the
`--speculative-config` removed.

- TRUE: non-speculative c=4 -> c=8 scales like vLLM's. The stall is entirely in
  the draft path.
- FALSE: non-speculative c=8 stalls too. Then the draft is at most part of it and
  the target's decode batching is the other part; profile the verify.

**E3 — the context sweep.** c=8 at 1024 in and at 256 in, same output length.
The fallback attention is `O(sum_r ctx_r^2)`; the target decode is ~`O(1)` per
row.

- TRUE: c=8 throughput improves much more than proportionally as the input
  shortens, and much more than vLLM's does on the same sweep.
- FALSE: both engines improve about the same. Then the cost is not
  context-quadratic and D1's arithmetic is wrong.

  **Read against the ~6% floor.** A 4x context reduction should move a
  context-quadratic term by far more than that, so this experiment is only
  worth running at a large sweep — 1024 against 256, not 1024 against 768. Score
  it on the RATIO between the two engines' improvements, not on ours alone.

**E4 — k=1.** c=4 and c=8 at `num_speculative_tokens=1`. `Tq` drops from `9P` to
`2P` while `C` is unchanged, so the fallback's cost barely moves; the verify
batch shrinks 4.5x.

- TRUE: the c=4 -> c=8 stall PERSISTS at k=1. The term is `C`, not the verify.
- FALSE: the stall disappears at k=1. The term is the `(1+k)` verify batch, and
  #1943's untrimmed fallback drafts are the first thing to check.

  **Read against the ~6% floor.** "Persists" and "disappears" are the only two
  readings this experiment supports; a partial move is not resolvable at c=8 and
  must not be reported as a fraction. Repeat both rungs 3-4x, as the ladder that
  produced the table above was.

**E5 — the window, for #2088.** Read `layer_types`, `swa_window_size` and
`sliding_window` off the campaign draft's `config.json` and print the resolved
per-layer `(causal, sliding_window)`.

- TRUE: five layers resolve `sliding_attention` with a positive window and
  `causal == false`. The window is being dropped and #2088 is live.
- FALSE: the layers resolve `sliding_window == 0`. #2088 is inert for this
  checkpoint, the attention is legitimately full-span, and D1's `Ncomb` argument
  is unaffected — only the magnitude changes.

**E6 — the graph-dispatch fraction.** c=4 and c=8, reading
`GraphDispatchStats::uniform_spec_steps` / `total` and `spec_as_decode_steps`
(`src/vllm/v1/worker/gpu/cudagraph_dispatch.h:187-201`). A ragged batch is a
WHOLE-STEP cliff here: `GraphEligibleQueryLen`
(`cudagraph_dispatch.h:161-175`) returns `nullopt` for the entire step if any
one request has `drafts + 1 != q`, which drops the batch onto the
`num_splits=1` prefill ladder (`runner.cpp:1784-1799`,
`.agents/specs/dflash2-spec-as-decode.md:103-111`). One odd request poisons all
eight, so this is a second concurrency-amplified mechanism that is INDEPENDENT
of #2087.

- TRUE: the uniform fraction falls materially from c=4 to c=8. Raggedness is a
  term, and #2090 plus the #1943 sync-mode fallback are where it comes from.
- FALSE: the fraction is flat and high at both. The verify lane is exonerated
  and everything is downstream in the draft.

Run E1, E2 and E6 first: E1 and E2 either put the cost inside `fwd` at `P > 1`
or send this spec back, and E6 costs nothing to read alongside them.

## Risks

- **R1.** D1 changes a `vt` op's signature, which is a shared seam. Every other
  caller of `vt::DFlashBlockAttention` (`qwen3_dflash.cpp:551`, the MiniMax-H3
  device paths) must keep its current behaviour, and the default when the query
  cu is absent must be exactly today's.
- **R2.** The five CUDA kernels behind this op (reference, warp, chunk,
  key-lane, MMA) each carry their own mask arithmetic. A query-cu that is
  applied in four of them and forgotten in the fifth is an acceptance-only
  defect the token gate cannot see. Every kernel needs the CPU-equivalence case.
- **R3.** [#2028](https://github.com/mudler/vllm.cpp/issues/2028) is a CUDA
  illegal memory access under sustained c=8 load on this exact path. It may share
  a cause with this wave and it may not; do not assume either. A rerun of the
  ladder can die before it measures.

## Owed

- **O1.** Every device number in this spec is arithmetic over code that was
  read, not a measurement. `## Gates` and the experiment list carry what a lease
  owes.
- **O2.** #2088 and #2089 are filed and NOT fixed here; both are listed under
  `## Owed` in the parent spec [dflash2-spec-decode.md](dflash2-spec-decode.md).
- **O3. CLOSED, and it was WRONG.** The `(Hq, Hkv, head_dim)` of the campaign
  draft ARE recorded in this tree, and were when this entry was written:
  `tests/vllm/models/test_qwen3_dflash2_draft.cpp:129-171` carries
  `z-lab/Qwen3.8-27B-DFlash2`'s `config.json` verbatim. `Hq = 32`, `Hkv = 8`,
  `head_dim = 128`, so **`kdim = 1024`** and GQA is 4:1; also `L = 5`,
  `hidden_size = 5120`, `vocab_size = 248320`, `sliding_window = 2048`,
  `selector_rank = 256`, `selector_top_k = 16`. The last two match what was read
  directly off the NAS checkpoint on 2026-08-28, independently.

  Two things this entry cost while it stood. Any byte figure derived with the
  `kdim ∈ [512, 5120]` bound below is HALF the real value if it used 512 — the
  true `kdim` is at the top of that range, so the `1.28 GB / 4.7 ms` row of the
  `B_saved` table is the applicable one. And a sweep of this row's levers on
  2026-08-28 repeated the claim and therefore could not size the attention
  lever, having to mark its magnitude speculative on an unknown that was
  already committed. Read the tree before recording something as unrecorded.
- **O4.** D1's CUDA half compiles and RUNS on **one** architecture. Built on
  `dgx:gpu0` under an `rc` lease with `-DVLLM_CPP_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=121a -DVLLM_CPP_CUTLASS_FETCH=ON`, linking a working
  server binary, with `test_ops_dflash_block_attn` 19/19 and 6,368,877 assertions.
  That is `sm_121a` alone, and it provably cannot reach the
  `#if __CUDA_ARCH__ < 800` branch of `DFlashAttnMmaKernel` — where D1's
  `(void)qcu;` lives and where an unused-parameter `-Werror` would fire — because
  `__CUDA_ARCH__` is 1210 there. `cuda-fat-build` closed that residual: GREEN on
  `3e541640c` (2026-08-27, 1h58m), which compiles every architecture this repo
  ships, including the sub-`sm_80` ones that take that branch. So the CUDA half
  COMPILES everywhere and RUNS on one architecture. Nothing has executed the
  kernels on a second architecture, and O6 is why nothing in CI ever will.
- **O5.** D1 changes NO token, so no experiment in this spec is settled by it.
  E1, E2, E3, E4 and E6 remain exactly as written, and Gate 4's ladder is what
  turns D1 from an arithmetic argument into a measurement. Read the ladder
  against the ~6% resolution floor `## Why` records. E2 has since been RUN
  (spec-OFF 10.83 / 38.97 / 69.27 at c=1/4/8) and its reading is recorded above;
  E1 was traced only far enough to give the ctx-2048 c=1 figure. E3, E4 and E6
  remain exactly as written.
- **O6.** [#2108](https://github.com/mudler/vllm.cpp/issues/2108) — this
  repository has NO GPU CI runner, so every test that appears to gate a device
  path is either skipped or silently running on the CPU backend, and both shapes
  report green. It bit this wave twice. `test_dflash_propose` builds its queue
  with `vt::Queue Cpu()`, so its 10/10 on a GPU box said nothing about CUDA and
  was briefly read as if it had; and `test_dflash2_runner_reach` is 7-red under
  CUDA on `main` — identical failure counts pre- and post-D1, so pre-existing and
  not D1's — with nothing having ever executed it there. The consequence for THIS
  row is precise: `ForwardWithCtxKVDev` at `P > 1` with real device tensors, the
  exact path D1 changed, is covered by no gate at all. The op cases cover the
  kernels in isolation and DO run on CUDA; `test_dflash_propose` covers the model
  path on CPU; their composition on a device is covered only by an end-to-end
  throughput run, which is blind to the acceptance-only defect class this row
  keeps hitting. Owned by #2108, not repaired here.

- **O7.** [#2112](https://github.com/mudler/vllm.cpp/issues/2112) — **E6 is
  not runnable as written, and Gate 3 cannot be read on the ladder.** Every
  caller of `vllm::v1::GetGraphDispatchStats()` and of
  `vllm::detail::GetDflashBlockRouteStats()` is a TEST; there is no print, no
  env-gated dump and no metric, so from a running server neither
  `uniform_spec_steps` nor the W11/#2089 route counters can be observed. E6 says
  to read the first pair at c=4 and c=8 and ranks itself in the first group to
  run; it costs a code change instead. This is the #2089 shape one level out —
  the counter was widened to cover both lanes and still has no readout on the
  workload it was built for, which makes it a diagnostic that measures a class
  rather than a capability. Not repaired here; owned by this row. **DISCHARGED**
  by SPEC-DFLASH2 W13 ([dflash2-mixed-step-readout.md](dflash2-mixed-step-readout.md)):
  `VT_GRAPH_STATS=N` prints `[graph-dispatch]` and `[dflash-route]` from the
  shared step path, gated by a production reachability case that reads both
  lines off real fd 2 and by the two mutations that delete the call site and
  bypass the classifier.

## What D1 landed

D1 is implemented (#2087, the D1 bullet of `## Design`). `vt::DFlashBlockAttention`
takes `DFlashBlockAttentionArgs::cu_seqlens_q`; null is byte-for-byte the old
behaviour and every pre-W12 caller passes null. `ForwardWithCtxKVDev` sets it, so
its query stays `[Tq, ...]` while K/V span `[Ncomb, ...]`, and the `Ncomb`-sized
query buffer, its memset, the `Ncomb`-sized output buffer, the query `IndexCopy`
and the output `IndexSelect` are gone.

Gate 1 is MET on the CPU backend and was measured as an actual before/after, not
as a self-comparison: the same `P = 2` device-KV fixture digests to
`h=2918102966398552862` over 48 floats on `f6563e9dd` (pre-D1) and on the D1
head, from one identical instrumentation patch applied to both trees. The
`P = 1` case digests to `h=13229198400555904305` on both. Gate 2 is MET as the
launch-shape assertion it names, at the model entry and through a
two-concurrent-request drive of the production engine. Gate 3 (#2089) landed
with it.

Gate 4 is MET, measured 2026-08-27 on `dgx:gpu0` (GB10) under an `rc` lease.
Both arms are the SAME SESSION at the same context: `build18` = `main` at
`ca3dcda21`, `build19` = the D1 head `3e541640c`; ctx 2048, 1024 in / 512 out,
`--max-num-seqs 16`, `--num-blocks 3744`, speculation ON, every rung 100% ok
with `ima=0 discont=0 dflash=5`.

| c | pre-D1 | D1 | delta | vLLM |
|---|---|---|---|---|
| 1 | 22.44 | 22.31 | -0.6% (CONTROL, inert) | 24.36 |
| 4 | 59.24 | 63.08 | +6.5% | 64.25 |
| 8 | **69.23** | **76.23** | **+10.1%** | 80.0 |

The c=8 deficit against vLLM closes from -13.5% to -4.7%, and c=4 -> c=8 scaling
goes 1.17x -> 1.21x against vLLM's 1.25x.

**Read the caveats before the numbers.** Every rung is n=1, and `## Why` records
a ~6% resolution floor at c=8. c=8's +10.1% clears it by 1.7x; **c=4's +6.5%
barely clears it and is not a safe claim on its own**; c=1 is a two-sided
CONTROL rather than a result, and it is inert exactly as the code predicts,
because `P == 1` takes `ForwardPagedBody` -> `vt::DFlashPagedBlockAttention`, an
op D1 never touches and never calls. The 3-4 reps per rung that produced the
`## Why` table are still OWED.

**Two earlier figures for this row are WITHDRAWN, and the reason is worth
keeping.** A first pass reported +20.4% at c=8 and a -21% -> -4.7% close. Both
compared the D1 arm against `build17` — a different build at a different context
length — and both are wrong; the same-session pre-D1 c=8 is 69.23, not 63.3. The
same error produced the claim that speculation was NET-NEGATIVE at c=8: against
the same-session 69.23, spec-ON is BREAK-EVEN with E2's 69.27 spec-OFF number,
not behind it. A cross-build comparison is not an A/B, and a ctx confound that
was first noticed on one rung applied to all three.

**D1 does not close the scaling gap, and that is the case for D2.** E2's
spec-OFF arm scales 1.78x from c=4 to c=8; pre-D1 spec-ON scaled 1.17x and D1
spec-ON scales 1.21x. So the draft still damps concurrency scaling well below
what the same engine does with speculation off. D1 removed the attention term
`## Design` names; the `O(C)` per-layer gather it deliberately left in place,
and the loss of the CUDA-graph lane, are still there. D2 is what those need.

**Acceptance is UNCHANGED**, which is what bit-identical CPU equivalence
predicts and the only device check that can see an acceptance-only defect: the
verify is lossless, so a mis-indexed query cu would emit the target's tokens and
cost acceptance alone, reading as a throughput number rather than as a fault.
Measured at c=8, k=8, `VT_SPEC_TRACE=1`: pre-D1 1.7861 (n=561) and 1.9674
(n=521); D1 1.9342 (n=532). The D1 value sits INSIDE the pre-D1 arm's own 10.2%
spread, so the +8.3% that a single pre-D1 run first suggested was sampling noise
— temperature 1.0 with top_k 20 drafts different tokens every run. One run per
arm would have reported a shift that is not there.

Device correctness is gated by `test_ops_dflash_block_attn` on that build:
**19/19 cases, 6,368,877 assertions, 0 skipped** — the first execution anywhere
of D1's two device cases, including the bf16 tensor-core case, since f32 can
never reach `DFlashAttnMmaKernel` by dispatch.

## D2 — the bound, measured against the code it would delete (#2111)

D2 was scoped next and is **not being implemented**, for a reason that is
arithmetic over this tree rather than a preference. Filed as
[#2111](https://github.com/mudler/vllm.cpp/issues/2111). Nothing here is a new
device measurement; every figure is derived from code that was read and from the
ladders already recorded above.

**The two terms D2 removes are bounded at about 2% of the c=8 step, which is
under this document's own ~6% resolution floor for that rung.**

Post-D1 the `P > 1` lane's `O(C)` work is exactly two `IndexSelect`/`IndexCopy`
stages: the gather in `ForwardBlockLogitsWithDeviceKV` (pool -> `tmpk`/`tmpv` ->
`ckv`, read and write, K and V: 16 B per context row per `kdim` element per
layer) and the combined scatter in `ForwardWithCtxKVDev` (`ckv` -> `kcb`/`vcb`:
8 B on the same basis). So

```
B_saved = 24 * L * C * kdim
```

`L = 5` (every run reports `dflash=5`) and `C ~ 8 x 1300 = 10400` at c=8 on this
recipe. `kdim = Hkv * Dh` is the term O3 still owes, so it is bounded rather
than read: `kdim <= hidden_size = 5120`, realistically 512 to 1024 under GQA.

| `kdim` | `B_saved` | at GB10's ~273 GB/s |
|---|---|---|
| 512 (realistic) | 639 MB | **2.3 ms** |
| 1024 (realistic upper) | 1.28 GB | 4.7 ms |
| 5120 (`Hkv == Hq`, not a real geometry) | 6.39 GB | 23.4 ms |

The other term D2 restores is the CUDA-graph lane, and a capture removes host
dispatch rather than device work. The `P > 1` lane issues about 255 launches at
`P = 8`, `L = 5` (`4*P*L` gather, `2*L` scatter, the per-layer body, the head),
so at ~6 us of dispatch each its **entire ceiling is ~1.5 ms**.

Against that, the D1 arm's c=8 step is `76.23 / (8 * 1.9342) = 4.93` steps/s, or
**203 ms**. Reaching 112 tok/s means removing **64.8 ms**; reaching SGLang's
scaling from our c=4 (89.4 tok/s) means removing **29.9 ms**. D2's two terms
together are ~3.8 ms on realistic geometry: 5.9% of the first and 12.7% of the
second.

**And 1.78x is not a target any engine on this box demonstrates.** No engine
reaches it with speculation ON — vLLM scales 1.245x, SGLang 1.418x, ours 1.208x,
against speculation-OFF's 1.777x. The reason is structural and not an
implementation difference: at `k = 8` and c=8 the verify batch is 72 target rows
against speculation-off's 8, so a speculative step does about 9x the target work
by construction while the ladder counts accepted tokens. Reading 1.208 against
1.777 charges the draft for the verify batch's own growth. The reachable
denominator is SGLang's 1.418.

**Where the residual is, is unattributed, and E1 is what would attribute it.**
With acceptance 1.9342 and the speculation-off ladder as the target-only
baseline, the non-speculative step grows 102.6 -> 115.5 ms from c=4 to c=8
(+12.9 ms) while the speculative step grows 122.6 -> 203.0 ms (+80.4 ms). That
80.4 ms is split between the verify batch (36 -> 72 target rows) and the draft
phase and nothing has measured the split. E1 prints exactly that split, needs no
code, and O5 records that it was traced only at c=1. **E6 is unread and it now
HAS its readout**: SPEC-DFLASH2 W13 ([#2112](https://github.com/mudler/vllm.cpp/issues/2112),
[#2117](https://github.com/mudler/vllm.cpp/issues/2117),
[dflash2-mixed-step-readout.md](dflash2-mixed-step-readout.md)) landed
`VT_GRAPH_STATS=N`, which prints both counter families from
`GPUModelRunner::execute_model` every `N` steps. E6 costs a rerun of the rung
with the variable set and no code change. It also reads MORE than it asked for:
`ragged_steps` is split three ways, so `ragged_mixed` names the admission
population #2117 mechanism 1 is about and separates it from #1943's uneven
verify widths, which the flat count could not. Run E1 first, then E6.

**D2's own precondition is also unmet.** `## Design` conditions the shared pool
on agreeing the allocation with [#2007](https://github.com/mudler/vllm.cpp/issues/2007),
which is open and unowned. The pool is additionally a residency change:
`MakeDeviceKVStore` allocates per request on `first_sight`, so residency follows
live concurrency today, whereas one arena addressable by a single block table
must exist for `max_num_reqs` up front — up to the whole 8 GiB
`kDflashCtxTotalBudgetBytes` aggregate — on the box #1647 OOM-rebooted.

**The change surface, recorded so the next reader does not derive it again.** D2
is not hard, it is: a per-layer arena `[max_num_reqs * max_pages, block_size,
Hkv, Dh]` with a page-range allocator, the store keeping a `page_base` and a
`block_table` row of `page_base + p` instead of today's identity; slots in
`ScatterProjectedContextRows` becoming `page_base * block_size + num_ctx + i`;
`ForwardPagedBody` generalized to P stores (`cu_seqlens [P+1]`, `seq_lens [P]`,
`block_table [P, max_pages]`, `slot_map [P*Tq]`) with
`detail::DflashBlockPagedInputsOf` and its three `VT_CHECK`s generalized per
request; `DflashBlockEligibility`'s hardcoded `e.num_reqs = 1`; and, for the
graph lane, a capture keyed on `(P, Tq)` owned above the store, because
`DflashDeviceKVStore::g_*` is per request and the batch composition changes every
step. A shared pool is required at all only because `vt::PagedAttention` and
`vt::ReshapeAndCache` each take ONE K/V tensor while P requests own P pools;
everything else those ops need is already there.

**Caveats.** Acceptance is measured only at c=8 (1.9342, n=532) and assumed equal
at c=4; it is a draft/target property rather than a concurrency one, but the
assumption is load-bearing for the step-time figures. Every ladder rung is n=1.
The bandwidth figure is GB10's nominal LPDDR5X.

## Stop conditions

Stop and report if E1 puts the growth outside `fwd`, or if E2 shows the
non-speculative ladder stalling the same way. Either result refutes the scope
above and the wave should be re-cut rather than implemented.

**D2 is stopped here, and this is what stopped it.** Not E1 or E2 — a third
reading the stop conditions did not anticipate: the terms D2 deletes are bounded
below the rung's own resolution floor before it is built, so implementing it
could not produce a readable result either way. `## D2 — the bound` states the
arithmetic; #2111 owns it. E1 and E6 at c=4 and c=8 are what the wave owes next,
and they cost one lease and no code.
