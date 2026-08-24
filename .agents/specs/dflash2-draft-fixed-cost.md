# SPEC-DFLASH2 W9 — the draft phase's flat ~23 ms: what the records settle, what lands, what a lease still owes

**Row:** `SPEC-DFLASH2` (wave W9, after W8 [dflash2-device-propose.md](dflash2-device-propose.md)).
**Issue:** [#1849](https://github.com/mudler/vllm.cpp/issues/1849).
**Parent spec:** [dflash2-spec-decode.md](dflash2-spec-decode.md).
**Kind:** one pull request, spec committed first (the W8 shape; the row-claim
"separate pull requests" answer covered the row's structured spec, and every
wave since has carried its own spec as the first commit of its own pull
request).

## Why

#1849 measures the DFlash2 draft phase at a flat 22.4–23.1 ms at every
`num_speculative_tokens` K in {1,2,4,8} (dgx GB10, `main` `8b107701b`
post-W8, real-text #1574 workload, c1), with the verify lane exonerated
(step grows only ~9 ms from q=2 to q=9). At accept 3.56 and step 121.8 ms,
halving the draft phase is ~+10% throughput — the whole distance to SGLang
on the real-text row (#1574: we tie vLLM at 0.2% and trail SGLang ~10%).

The issue names two levers and asks for measurement first on both. This spec
records the measurements that the repository's own records and code settle,
lands the two things they justify, and names exactly what still needs the
box.

## Upstream chain

Same beyond-pin anchor as the whole row: vllm-project/vllm#52816, MERGED at
head `3406ec1d`, merge commit `b389ac29465b33f9e9c534df221ea3c129e9793f`.
Read again for this wave, from the local vLLM clone at the merge commit:

- `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` —
  `_generate_draft` runs the backbone (`_run_model`, hidden out, NO head),
  gathers `last_hidden_states[sample_indices[:num_sample]]`, runs
  `compute_candidates` on the k gathered rows, `candidate_selector` (torch
  ops under `@support_torch_compile`), then ONE Triton `_selector_walk_kernel`
  per request. The drafted tokens stay in `self.draft_tokens` ON DEVICE — no
  D2H in the draft step at all.
- `vllm/model_executor/models/qwen3_dflash2.py` — `compute_candidates` =
  `candidate_logits_processor.get_top_k_tokens(self.lm_head, hidden, top_k)`,
  i.e. the head GEMM through `lm_head.quant_method.apply`, the target's own
  logits path (D14's anchor, unchanged).
- The quantized-head path for the r0b0tlab subject:
  `vllm/model_executor/layers/quantization/modelopt.py` accepts
  `ParallelLMHead` and pins the Marlin W4A16 kernel for `NVFP4`
  ([perf-27b-lmhead-nvfp4.md](perf-27b-lmhead-nvfp4.md) `## Upstream anchors`,
  verified against the pinned oracle `555967922`).

## Lever A — the shared head's dtype, settled from the artifact and the code

**The #1849 title premise — "the shared bf16 head is read twice per step" —
does not describe the measured subject.** Established from records, not
assumption:

1. **What the artifact carries.** `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` @
   `36f717a2` stores `lm_head` as ModelOpt W4A16_NVFP4 `group_size` 16 —
   `lm_head.weight` U8 + `lm_head.weight_scale` F8_E4M3 +
   `lm_head.weight_scale_2` F32 `[]`, one of the 193 W4A16 modules.
   Header-verified over all four shards on 2026-08-21
   (`.agents/quantization-matrix.md` `QUANT-QWEN38-27B-NVFP4-ARM`;
   [qwen38-27b-quant-arms.md](qwen38-27b-quant-arms.md) W5's 2001-name
   accounting, with the committed manifest `.inc` fixtures). NOT bf16, and
   NOT FP8 — so there is no FP8 head to mirror and no quantization to
   invent.
2. **What upstream does with it.** Computes with it PACKED, through
   `lm_head.quant_method.apply` (Marlin W4A16), for the target's own logits
   AND for `compute_candidates` — the anchors above.
3. **What our tree does with it.** The same, since #1628 (parent spec D14):
   the target's verify head takes `MatmulNvfp4F32D` (`qwen3_5.cpp:3106`),
   the draft's shared head lands packed in `Qwen3DFlashWeights::lm_head_fp4`
   and takes `dense_nvfp4::MatmulNvfp4W4A16D`; `VT_LMHEAD_FP4` defaults ON.
   Lever A's "quantize the shared head" is therefore ALREADY LANDED, and the
   #1849 arithmetic changes:

   - Head traffic per step ≈ 2 × ~0.72 GB packed (0.64 GB codes + 0.08 GB
     FP8 scales at 248320×5120 g16), ≈ 5.2 ms at ~273 GB/s across BOTH
     phases — not 18.6 ms of bf16.
   - The draft-phase bandwidth floor re-derives to ≈ 9.2 ms: 5 draft layers
     ~1.5 GB + packed head ~0.72 GB + the fc aux-combine
     [5120, 5×5120] bf16 0.26 GB + context-KV projections and gathers
     ~0.1 GB. The resident selector codebooks (2 × 127 MB) and the embed
     table are gathered, not swept.
   - The unattributed residual is therefore ≈ 13–14 ms — LARGER than the
     ~8 ms #1849 derived under its bf16 assumption — and (see Lever B) it is
     not launch overhead, so it sits INSIDE kernels. Naming which ones needs
     the box; `## Owed` O1 carries it with the instrument this wave lands.

**The bf16-target arm's ceiling, recorded honestly.** On a bf16 safetensors
target (the z-lab arm) the checkpoint stores `lm_head.weight` BF16 and
upstream serves it BF16 through `UnquantizedLinearMethod`. Mirroring forbids
inventing a quantization there (D12's silent-acceptance failure is exactly a
head the target does not compute with), so on that arm the two bf16 head
reads (~2 × 2.54 GB ≈ 18.6 ms/step at ~273 GB/s) are the mirrored cost and
stand as a ceiling. Recorded; no code change.

**The second copy — half real, and the real half is memory, not bandwidth.**
The draft's shared head/embed never alias the target's:

- HOST, bf16 arm: `LoadDflashSharedLmHead`'s bf16 loop and the loader's
  embed read each `memcpy` the full `[248320, 5120]` bf16 tensor into a
  draft-owned buffer — ~5.1 GB of anonymous host memory whose content is
  byte-identical to the file mapping the target already borrows
  (`BorrowStTensorBytes`, ENG-LOAD-DIRECT-UPLOAD). On the r0b0tlab arm the
  head takes the packed borrow (zero copy) but the EMBED copy is live —
  ~2.5 GB on the campaign box, which OOM-reboots under memory pressure and
  ran the #1574 row at reduced sizing. **This wave fixes the host half**:
  both loads go borrow-first with the existing fail-closed
  `BorrowStTensorBytes` seam (`## Design` D2).
- DEVICE, both arms: two residents where one would do. `ResidentWeight`
  caches per `OwnedTensor` (`w.d_dev`), and the two Marlin dense bodies keep
  separate address-keyed resident caches (`qwen3_5.cpp:2760` vs
  `dense_nvfp4_gemm.h`, the O29 pair), so the target's head and the draft's
  stage ~0.72 GB (packed, repacked-for-Marlin) or ~2.54 GB (bf16) EACH.
  Deduplicating requires the two Marlin bodies to converge on one resident —
  exactly the O29 convergence the parent spec already owes — and is NOT done
  here (`## Owed` O3).
- BANDWIDTH: unaffected either way. The verify head read and the draft head
  read are two GEMMs over the full weight regardless of how many device
  copies exist (no cache tier holds 0.72 GB), and upstream reads the head
  twice per step too. The second copy costs memory only.

## Lever B — the launch/sync census, and why the residual is not launches

The post-W8 sample phase, counted in code
(`runner.cpp::propose_drafts_block` → `Dflash2SelectCandidatesDevice` →
`Dflash2WalkPathDevice`), per step at P=1:

| # | work | size |
|---|---|---|
| 1 | `vt::Embedding` + block-position H2D (outside the capture) | Tq i32, ~36 B up |
| 2 | ONE `st.g_graph.Replay` (backbone + conv + head, captured) | the floor |
| 3 | sample-index H2D + 2 × `vt::IndexSelect` (logits rows, hidden rows) | 32 B up; ~16 MB moved |
| 4 | `vt::TopKValuesIndices` (P·k blocks over 248320 columns) | 1 launch |
| 5 | `vt::MulScalar` + `vt::SoftCap` on [P·k, 16] values | 2 launches, 2 KB |
| 6 | `vt::MatmulBT` hidden→rank-256 projection | 1 launch |
| 7 | anchors H2D + `vt::Dflash2SelectorEdges` | 8 B up; 1 launch |
| 8 | `vt::Dflash2PathWalk` + the ONE download (P·k i64) + sync | 64 B down |

Total: one replay, ~10 eager launches, three H2D of ~76 bytes, one D2H of
64 bytes, one queue synchronize. At realistic launch cost that is well under
half a millisecond — it CANNOT be the ~13 ms residual. Upstream's granularity
is the same (eager torch topk/mul/tanh_softcap/matmul/einsum plus one Triton
walk kernel); there is no fused selector on their side to port.

Two genuine structural divergences from upstream, both examined and neither
taken, with the arithmetic:

- **R1 — head-after-gather.** Upstream gathers the k sample rows of hidden
  and runs the head GEMM at M=k outside its graph; we compute the full
  (1+k)-row logits inside the captured forward and gather logits rows after.
  The head GEMM is weight-bandwidth-bound at both M, so the delta is the
  extra logits row (~1 MB) plus the gathered-logits copy (~16 MB) ≈ 60 µs —
  and taking it would pull the step's largest single kernel OUT of the CUDA
  graph. Rejected: costs the graph more than it saves in traffic.
- **R2 — device-resident drafts.** Upstream's walk writes `draft_tokens` on
  device and the next verify consumes it there; no D2H and no sync in the
  draft step. Our runner downloads the P·k ids and synchronizes, because the
  host builds the verify input batch. The sync is where the 23 ms is
  OBSERVED (it drains the queued replay + selector), not what causes it;
  eliminating it means feeding the verify input device-to-device, which is
  the async-scheduling row's architecture (W7,
  [spec-decode-async-scheduling.md](spec-decode-async-scheduling.md)), not a
  propose-chain trim. Named for that row; not taken here.

**What Lever B therefore needs first is attribution, on the box.** The
`VT_SPEC_TRACE=1` sample number aggregates pre-phase GPU work + replay +
selector + walk into one figure. This wave lands the split (`## Design` D1)
so the owed lease run reads four numbers instead of one and the next
hypothesis (Marlin head at small M? the conv? the paged attention? the
top-k's one-block-per-row shape at P·k blocks?) is selected by measurement.

## Scope

In scope:

- **D1**: `VT_SPEC_TRACE=2` — device-segment attribution of the DFlash2
  draft phase (`[spec-phase-dev]`: pre / fwd / select / walk), additive to
  the level-1 line, off by default, production propose path only.
- **D2**: borrow-first loading for the draft's SHARED bf16 embed table and
  bf16 lm_head (safetensors arm), through the existing
  `BorrowStTensorBytes` fail-closed seam. Byte-identical content; host-copy
  elimination only.

Out of scope, recorded: the device-side head/resident dedup (O29
convergence, `## Owed` O3); R1 and R2 above; any change to the GGUF arm
(`LoadGgufSharedEmbedAndHeadBf16` widens and is D12's refused case); the
DSpark lane's behaviour (it shares `SharedHeadSource::LoadInto`, so it
inherits D2's borrow through the same seam, and nothing else); any speed
claim (`## Gates`).

## Design

### D1 — `[spec-phase-dev]`: the draft phase split at its seams

`propose_drafts_block`'s trace latch becomes a LEVEL (`VT_SPEC_TRACE`,
parsed as an integer; any nonzero value below 2 keeps today's behaviour
exactly, so every existing `VT_SPEC_TRACE=1` recipe is unchanged). At level
>= 2, on the DFlash2 branch only, the runner brackets the four phase seams
with `Backend::Synchronize(queue_)` and wall-clocks the segments:

```
[spec-phase-dev] pre=%.2fms fwd=%.2fms select=%.2fms walk=%.2fms
```

- `pre` — the pre-phase device work (aux combine + accepted-prefix append),
  measured by a sync at the old `t_fwd0` boundary.
- `fwd` — the block forward (embed refresh + graph replay, or the eager
  paged body), synced after `ForwardBlockLogitsWithDeviceKV` returns.
- `select` — `Dflash2SelectCandidatesDevice` (gathers, top-K, scalars,
  projection, edge lattice), synced after it returns.
- `walk` — `Dflash2WalkPathDevice`, whose own download already synchronizes.

The added syncs run ONLY at level >= 2: the level-1 lane keeps its
overlap-preserving shape, so an A/B against the #1849 K-ladder stays valid.
Synchronization changes no value anywhere — the segments see the same bits —
and the CPU arm's syncs are no-ops, so the line is CPU-gateable. The line is
emitted per proposing step, beside (not instead of) the level-1 line. On a
CUDA box the sync semantics (each segment waits the queue) are exactly the
`b.Synchronize` the runner already uses at its sampling boundary; their
verification on real hardware rides O1's lease run.

### D2 — borrow the shared embed and head instead of copying them

`LoadDflashSharedLmHead`'s bf16 loop gains a `BorrowStTensorBytes` attempt
before its `memcpy` fallback, exactly the `LoadBf16Direct` shape
(`dense_weight_loaders.h:363`): on success the draft's `lm_head` views the
file mapping (keep-alive: `StTensor::mapping`), `nk = true` set after, and
no owned buffer exists; on any mismatch — synthetic `StTensor`, size
mismatch, `VT_LOAD_DIRECT_UPLOAD=0` — the seam fails closed and the copy
runs as today.

The embed read moves from the loader-file-local `LoadNamedBf16` into a new
exported sibling beside the head loader:

```cpp
// qwen3_dflash.h
OwnedTensor LoadDflashSharedEmbedBf16(const std::vector<SafetensorsFile>& shards,
                                      const std::string& name);
```

Same lookup (exact name, first shard that has it), same BF16 refusal text as
the read it replaces, `nk = false`, borrow-first with the same fallback.
`SharedHeadSource::LoadInto` calls it; `LoadNamedBf16` keeps its remaining
callers (if any) untouched. Exported because a file-local borrow is a lever
a reviewer's deletion mutation could remove with every gate green — the test
must reach the exact function production calls.

Safety, argued once: both tensors are whole-range verbatim reads the draft
never mutates (a draft declaring `input_embedding_scale != 1.0` is already
refused by name at load, so no in-place scale exists), the mapping outlives
the draft via the borrow's keep-alive, and `ResidentWeight` uploads from the
mapping directly — the same contract the target's own weights have run under
since ENG-LOAD-DIRECT-UPLOAD.

What this is worth: ~5.1 GB of anonymous host memory on the bf16-target arm
(embed + head), ~2.5 GB on the r0b0tlab arm (embed; the packed head already
borrows). It does not move step time and is not claimed to.

## Tests

- **T1 (D2, head)** — in `tests/vllm/models/test_qwen3_dflash2_draft.cpp`:
  an on-disk bf16 target through `OpenShards`; `LoadDflashSharedLmHead`
  must return a head whose `bytes.data()` IS the shard's `StTensor::data`
  (pointer equality into the mapping), bytes equal, `nk` true; under the
  `SetLoadDirectUploadOverrideForTesting(false)` arm it must NOT alias and
  must still be byte-equal (the fallback). RED first: today's memcpy fails
  the aliasing assertion.
- **T2 (D2, embed)** — same fixture, `LoadDflashSharedEmbedBf16`: aliasing,
  byte equality, `nk` false, and the not-BF16 refusal text unchanged. RED
  first: the function does not exist.
- **T3 (D1)** — new binary `tests/vllm/v1/spec_decode/test_dflash2_draft_phase_trace.cpp`
  (its own binary because `VT_SPEC_TRACE` is latched process-wide before
  `main`, and the existing reach binary latches "1"): drives the same
  `LoadedEngine` CPU fixture as `test_dflash2_runner_reach.cpp` with
  `VT_SPEC_TRACE=2`, captures real fd 2, and asserts (a) the
  `[spec-phase-dev]` line appears with all four fields parseable and
  non-negative, once per proposing step; (b) the level-1 `[spec-phase]` and
  `first=[...]` lines STILL appear (level 2 is additive); (c) token identity
  under the instrument: the drafted blocks satisfy the same structural
  properties the level-1 reach gate holds them to (a whole block of k real
  vocabulary ids per step, at least one beyond top-k), and two engine runs
  under level 2 draft identical blocks. A literal cross-binary golden is
  deliberately not stored — the instrument's inability to move a value is by
  construction (`Synchronize` has no data effect), and a stored copy of the
  level-1 blocks would be a second description of the fixture that drifts.
  RED first: the line does not exist.
- **T4 (D1, label binding; #1851 F1)** — in the T3 binary: the fresh review
  proved by a surviving mutant that T3's assertions cannot see a MONOTONIC
  SLIDE of the select/walk seam (sync + `t_sel` stamp moved before
  `Dflash2SelectCandidatesDevice`): every segment stays non-negative, no
  draft moves, and `walk` silently absorbs the selector — which mislabels
  the exact numbers the `## Owed` O1 lease run picks its kernel hypothesis
  from. T4 binds the label to the work: a test seam at the top of
  `Dflash2SelectCandidatesDevice` (`VT_SPEC_TEST_SELECT_SPIN_MS`, a
  steady_clock spin — read per call, NOT latched, so the one case that sets
  it can scope it with setenv/unsetenv) injects a 200 ms wall-clock floor
  INSIDE the selector's bracket, and the case asserts on every traced step
  that `select` >= 150 ms and `walk` < 150 ms, and that the spun run drafts
  byte-identical blocks to an unspun run. The seam is INERT in production:
  no `VT_SPEC_TRACE` level sets the variable, the unset cost is one getenv
  per proposing step, and it delays without computing. RED first: without
  the seam the case fails at `select >= 150` (select reads ~0.02 ms).
- **T5 (D1, the level boundary from below; #1851 F2)** — in the reach
  binary (which latches `VT_SPEC_TRACE=1` pre-main, so it is the one
  process that can observe level 1): the level-1 `[spec-phase]` and
  `[spec-propose]` lines must be PRESENT on the same real-fd-2 capture
  (a dead capture must not read as absence) and `[spec-phase-dev]` must be
  ABSENT. Before T5, mutating `propose_trace_level >= 2` to `>= 1` left
  every suite green — "the syncs run ONLY at level >= 2" was asserted, not
  gated.

## Mutations (named in advance for the fresh reviewer)

- M1: restore the head `memcpy` (delete the borrow attempt) → T1 red.
- M2: make `LoadDflashSharedEmbedBf16` copy unconditionally → T2 red.
- M3: delete the level-2 branch in `propose_drafts_block` → T3 red (line
  absent) while the level-1 lane stays green.
- M4: print the line but emit garbage segments (e.g. swap two boundaries so
  a segment is negative) → T3's non-negativity/parse assertions red.
- M5 (reachability): delete `SharedHeadSource::LoadInto`'s call to
  `LoadDflashSharedEmbedBf16` (call `LoadNamedBf16` again) → T2 alone stays
  green (it calls the function directly), so T2 ALSO asserts through the
  loader: a `FromModelDir` case (the existing #1628 on-disk-target harness)
  whose draft embed aliases the mapping. The reviewer's scratch deletion of
  the production call site must red that case.
- M6 (#1851 F1, the surviving mutant, now killed): slide the select/walk
  seam — move `if (dev_trace) trace_b.Synchronize(queue_); t_sel = ...`
  to BEFORE the `Dflash2SelectCandidatesDevice` call → T4 red on both
  halves (`select` reads 0; `walk` carries the 200 ms floor) while T3 and
  the token-identity case stay green.
- M7 (#1851 F2, the surviving mutant, now killed): `propose_trace_level
  >= 2` → `>= 1` in the runner's `dev_trace` → T5 red at the
  `[spec-phase-dev]`-absent assertion.

## Gates

The full CPU gate (the parent row's suites: `test_qwen3_dflash2_draft`,
`test_dflash2_runner_reach`, `test_dflash_propose`, the GGUF suites, plus
the new T3 binary) green, `scripts/agent-preflight.sh --staged` green.
No speed number is claimed at this wave: the GPU step-time delta and the
segment attribution are operator-run under a lease (`## Owed`).

## Owed

- **O1 — the on-box attribution run.** Operator, `rc` lease on `dgx:gpu0`:
  re-run the #1849 K-ladder recipe with `VT_SPEC_TRACE=2` and record the
  four segments at K=1 and K=8; then one `ncu`/`nsys` pass over the fattest
  segment (nsys with the graph-kernel flag — the default hides graphed
  kernels). This selects among the named suspects: the Marlin W4A16 head at
  M<=9, the grouped conv, the paged attention body, the one-block-per-row
  top-k. Issue #1849 stays open until this lands.
- **O2 — the step-time delta.** Any claimed reduction from this wave or its
  follow-ons is measured operator-run on the #1574 real-text workload,
  same-binary A/B. No number is claimed here.
- **O3 — the device-side shared-head resident dedup.** Blocked on the O29
  Marlin-body convergence in the parent spec; ~0.72 GB (packed) or ~2.54 GB
  (bf16) of device memory. Not a step-time lever (bandwidth is per-read, not
  per-copy).
- **O4 — CUDA semantics of the level-2 syncs.** The CPU gate pins the line,
  its fields and its reachability; that each segment brackets exactly the
  intended device work on a real queue is verified in O1's lease run.

## Evidence (implementer, CPU box, 2026-08-24)

Build: `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release` scratch tree, gcc, no CUDA.

- RED first, T3: at the spec commit's tree plus the tests, before the runner
  change, `test_dflash2_draft_phase_trace` fails
  `REQUIRE(dev.size() > 0)` — the `[spec-phase-dev]` line does not exist
  (1 failed / 25 passed).
- GREEN: after the implementation, `test_dflash2_draft_phase_trace` 2/2 cases,
  150 assertions; `test_dflash2_runner_reach` 4/4, 110 (the fixture
  extraction moved no behaviour); `test_qwen3_dflash2_draft` 43/43, 449.
- Mutations, each applied to a scratch edit, rebuilt, run, and restored with
  `git diff` verifying zero residue:
  - M1 (head borrow deleted, `if (false && BorrowStTensorBytes...)`):
    `test_qwen3_dflash2_draft` 447/449 — T1's aliasing CHECK and the
    FromModelDir borrow-delta case red. This is also T1's red-vs-pre-change
    evidence: the mutated body IS the pre-W9 memcpy behaviour.
  - M2 (embed borrow deleted, same form): 447/449 — T2's aliasing CHECK and
    the borrow-delta case red.
  - M5 (production call site: `SharedHeadSource::LoadInto` reverted to an
    inlined copy-based embed read): 448/449 — ONLY the FromModelDir
    borrow-delta case red, T2 direct stays green, which is exactly the reach
    property that case exists for.
  - M3 (`dev_trace = false`): `test_dflash2_draft_phase_trace` red at
    `REQUIRE(dev.size() > 0)`.
  - M4 (select segment printed as `t_fwd1 - t_sel`): red at
    `CHECK(v >= 0.0)`, once per traced step.
- Full gate: recorded in the pull request (ctest over the full suite on this
  box, #618's known flake exempt as the task authority names).

The CUDA semantics of the level-2 syncs and every speed number stay OWED
(`## Owed` O1/O2/O4) — nothing here is a GPU measurement.

## Evidence (repair wave, #1851 F1/F2, CPU box, 2026-08-24)

The fresh review of #1851 left two instrument-coverage findings, each proven
by a surviving mutant; this wave adds T4/T5 and the T4 seam, and kills both.

- RED first, T4: with the test in place and `Dflash2SelectCandidatesDevice`
  restored to the pre-seam bytes, `test_dflash2_draft_phase_trace` fails
  8x at `CHECK(0.02 >= 150)` (2 passed / 1 failed cases).
- GREEN: seam restored — `test_dflash2_draft_phase_trace` 3/3 cases,
  202 assertions; `test_dflash2_runner_reach` 5/5, 118.
- M6 (the reviewer's slide, re-applied to a scratch edit): T4 red at
  `CHECK(0 >= 150)` (8x) AND `CHECK(200.xx < 150)` (8x — the walk segment
  carries the spin), T3 and the token-identity case green, which is the
  discrimination the finding asked for. Restored; sha256 equal.
- M7 (`>= 2` → `>= 1`): `test_dflash2_runner_reach` red exactly at T5's
  `captured.find("[spec-phase-dev]") == npos` (4 passed / 1 failed cases).
  Restored; sha256 equal.
- T5's presence half (level-1 lines on the same capture) keeps a dead
  capture from reading as a pass; its red is M7, the boundary mutant it
  exists to kill.

## Now

`ACTIVE` — W9 in flight on `row/SPEC-DFLASH2-draft-fixed-cost`.
