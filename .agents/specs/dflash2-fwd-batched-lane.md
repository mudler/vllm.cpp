# SPEC-DFLASH2 — the draft FORWARD is 76% of the draft phase, and its batched lane diverges from vLLM

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding).
**Issue:** [#2202](https://github.com/mudler/vllm.cpp/issues/2202).
**Related:** [#2154](https://github.com/mudler/vllm.cpp/issues/2154) (acceptance
collapse, same code), [#2155](https://github.com/mudler/vllm.cpp/issues/2155)
(the selector; its mechanism was retracted and this row inherits the real
target), [#2152](https://github.com/mudler/vllm.cpp/issues/2152) (no c=8 reading
is currently admissible), [#2007](https://github.com/mudler/vllm.cpp/issues/2007)
(the arena a batched paged store would need),
[#545](https://github.com/mudler/vllm.cpp/issues/545) (the host crashed three
times during this work).
**Kind:** performance, with a mirror obligation inside it.

## Now

`ACTIVE`. L1 landed as `fe21faf63`, the seam adoption as `c9b2049bc` (#2207),
and **L2 as `150b37852`** (#2212). The prior text here said L2 must not merge
until it ran green on a GPU; it had already merged, and `scripts/now.py` renders
this section, so the derived surface reported a landed change as an unmergeable
branch (#2234).

L2 measured **-11% on `fwd`** (35.19 -> 31.3 ms) with a terminal control
matching to 1.1%. Its CUDA build exists on `dgx:gpu0` as `build-l2`, and the
kernel signature is the identity: `DFlashAttnMmaKernel` there mangles to
`...fbll` (13 params, carrying `tiles_per_req`) against `...fbl` in the pre-L2
`build23`. That is a stronger identity than a commit label, because it proves
the FEATURE is compiled in rather than that a directory was named after a SHA
(`vllm_version()` returns `0.0.3+cuda` for every commit, so the binaries carry
no tree identity of their own).

**Next gate: L3, the batched capture lane.** The row deferred it until L2 had
priced the attention shape, and L2 has. At `P > 1` the draft forward is not
capture-targeted, so at c=8 the term that is 76% of the draft phase runs EAGER,
while vLLM replays a FULL draft graph at every batch size and pads to
`max_num_reqs` with `PAD_SLOT_ID` (`spec_decode/dflash/speculator.py:456-458`,
`:589-618` at pin `5559679229`). This is a porting gap under "mirror vLLM",
not a new design.

## The measurement

`VT_SPEC_TRACE=2` brackets each draft segment with a real `Synchronize`; level 1
does not, which is why its `sample=` figure absorbed the whole step and
misdirected #2155. One leg, c=8, ctx 2048, boot `bc7ae2cb`, n=771 phases:

| segment | median | share |
|---|---|---|
| `pre` | 6.84 ms | 14% |
| **`fwd`** | **36.19 ms** | **76%** |
| `select` | 4.22 ms | 9% |
| `walk` | 0.02 ms | 0.04% |

Level-2 syncs serialize, so the absolutes are inflated against level 1's ~19 ms
wall. **The ratios are the result.**

## The draft's dimensions, and a stale record

`tests/vllm/models/test_qwen3_dflash2_draft.cpp:129-171` carries the draft's
`config.json` verbatim: `L=5`, `H=5120`, `Hq=32`, `Hkv=8`, `Dh=128`, so
`kdim=1024` and GQA is 4:1; `vocab=248320`; `sliding_window=2048`;
`selector_rank=256`; `selector_top_k=16`. The last two match what was read
directly off the NAS checkpoint, independently.

`.agents/specs/dflash2-batch-propose.md` O3 states these are "not recorded
anywhere in this tree" and bounds `kdim ∈ [512, 5120]`. **That is stale**, and
any estimate built on 512 is half the real figure. Two consequences: #2088's
dropped sliding window is INERT for speed here (2048 > the ~1200-row context),
and levers priced on 8-way KV re-reads are overstated 2x.

## The structural finding

`ForwardBlockLogitsWithDeviceKV` has two lanes. `P == 1` reads the persistent
paged store directly and is CUDA-graph captured. `P > 1` falls through to a
materialised combined-context forward that is, in its own words, "not
capture-targeted".

**Upstream has ONE lane.** Read at the pin (`5559679229`, verified against
`.agents/upstream-sync.md`): vLLM attends the draft against the PAGED cache with
a batched block table at every batch size, rebuilding only metadata per step and
copying no K/V bytes (`llm_base_proposer.py:682-708`, `backend.py:736-757`,
`flash_attn.py:1040-1053`). DFlash's cross-attention context is written INTO the
pages — `dflash.py:140-165` passes `block_table_ptr=cad.block_table_tensor` to
one fused kernel, and `qwen3_dflash.py:602-619` scatters via
`reshape_and_cache_flash`. There is no `P == 1` special case anywhere in
`llm_base_proposer.py`, `dflash.py` or `eagle.py`.

Our own comments name both halves: `qwen3_dflash.cpp:221` says we mirror
upstream's context store "minus the paged-cache write", and that omitted write
is the entire upstream mechanism.

So the end state is a mirror obligation, not a discretionary optimisation. What
this spec disputes is only the ORDER, on cost grounds below.

## Cost, at the real dimensions

Per draft step, P=8, C≈9600, kdim=1024, L=5:

| term | formula | bytes | est. ms | share of `fwd` |
|---|---|---|---|---|
| context gather + scatter | `24*L*C*kdim` | 1.18 GB | 4.33 | 12% |
| draft weight sweep (P-independent) | body 1.5 GB + head 0.72 GB | 2.22 GB | 8.1 | 22% |
| host dispatch, fully eager | ~314 launches | — | ~1.6 | 4% |
| attention K/V staging | see below | 223-892 MB | 0.8-3.3 | 2-9% |
| **residual, inside kernels** | | | **~19-21** | **~55%** |

The copies are 12% of `fwd`, about 2.1% of the c=8 step — **below the rung's own
resolution**. That independently reconfirms #2111's stop conclusion at the now
known `kdim`, and it is why this spec does NOT reopen the batched paged store
(W12 D2) on copy arithmetic.

## The unbounded term

`LaunchDFlashBlockAttention` set `mgrid.x = ceil(Tq_tot / 64) = 2` at c=8, and
`DFlashAttnMmaKernel` stages the UNION of its block's rows' key ranges. Block 0
spanned all eight requests, so its union was the whole combined sequence:
~303 sequential 32-key tiles against the ~38 each of its rows actually needs,
with roughly 87% of MMA lanes masked. The tile loop is
`__syncthreads` / plain global load / `__syncthreads` with no async copy, so the
latency is fully exposed. The `P == 1` lane never hits this — it routes onto the
FA-2 split-KV decode lane.

## Waves

**L1 — contiguous context copy. LANDED (`fe21faf63`).** Both index maps were the
identity, so four device ops per (request, layer) became one `Backend::Copy`
each for K and V. Byte-for-byte identical; gated by a new `P == 2` invariant
case, since the batched lane had NO test at all.

**L2 — per-request query tiling. COMMITTED, UNVERIFIED.** A block can no longer
span a request, so its union is one request's key run. The mapping lives in
`include/vt/dflash_attn_grid.h` and is coverage-tested on the CPU, because a
wrong mapping drops output rows silently. Size unknown; it is a candidate for
part of the ~19-21 ms residual.

**L3 — the batched paged store.** The mirror-correct end state. Deletes the
copies and most of the dispatch, and is the only wave that reaches the FA-2
decode lane and a capturable graph at `P > 1`. Blocked on #2007 (one arena
instead of per-request pools) and on `DflashBlockEligibility` hardcoding
`e.num_reqs = 1`. Do not start it until L2 has measured how much of the residual
was the attention shape.

## Tests

L1 and L2 each landed their own gate, both CPU-runnable and both
mutation-proven. The batched lane had no coverage before L1: every case in
`test_qwen3_dflash_decode_graph_seam.cpp` built one store.

## Gates

- `[spec-phase-dev] fwd=` is the axis. **Not step throughput**: the c=8 rung's
  zero-draft-block rate varies 0.0%-87.7% across runs of one binary (#2154) and
  #2152's admissibility work is unfinished.
- The synthetic `vt::DFlashBlockAttention` bench at
  `(t=72, N=9672, hq=32, hk=8, d=128)` against 8x `(t=9, N=1209)` sizes L2
  without a full-model lease.
- Any device measurement records `uptime` and `boot_id` beside it, and refuses to
  fold across a `boot_id` change (#545).

## Owed

All three items this section carried are DISCHARGED, and are recorded here
rather than deleted so the next reader can tell "done" from "never written".

- ~~The seam policy item.~~ Landed as `c9b2049bc` (#2207): both hot bodies call
  `layers::UnquantizedMlpGateUpMethod` and all three sites take
  `MergedQkvEnabled()` + `vt::QkvSplit`. This is what makes a quantized gate-up
  arm reachable for the draft at all, which #2224 needs.
- ~~O3 in `dflash2-batch-propose.md` is stale.~~ Closed there against the config
  literal.
- ~~A stale anchor for the `P == 1` gate.~~ `dflash2-request-scoped-context.md`
  now names the symbol and records that the line moved, per `.agents/porting.md`.
  The correction this bullet itself proposed (`:1614`) had ALREADY drifted to
  `:1716` by the time it was read, which is the argument for the symbol rule
  stated twice over.

## Stop conditions

Return `NEEDS_DECISION` if L2 measures small AND the residual stays unexplained.
That would mean `fwd`'s bulk is inside kernels this row has not identified, and
L3's cost/benefit should be re-derived before committing to the arena.

## Outcome

Filled in when the row reaches `DONE`.
