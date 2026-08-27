# SPEC-DFLASH2 W12 — the batched propose: why c=4 -> c=8 is flat, and what would fix it

**Row:** `SPEC-DFLASH2` (wave W12, after W11
[dflash2-draft-block-fa2.md](dflash2-draft-block-fa2.md) and the #2010 repair).
**Issues:** [#2087](https://github.com/mudler/vllm.cpp/issues/2087) (the wave),
[#2088](https://github.com/mudler/vllm.cpp/issues/2088) and
[#2089](https://github.com/mudler/vllm.cpp/issues/2089) (found in the same read,
owned by the parent spec's `## Owed`).
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
- **O3.** The `(Hq, Hkv, head_dim)` of the campaign draft are not recorded
  anywhere in this tree; the byte figures a reviewer might derive from `Ncomb`
  need them. Record them from the checkpoint header with the E5 read.

## Stop conditions

Stop and report if E1 puts the growth outside `fwd`, or if E2 shows the
non-speculative ladder stalling the same way. Either result refutes the scope
above and the wave should be re-cut rather than implemented.
