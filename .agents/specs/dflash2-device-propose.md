# SPEC-DFLASH2 W8 — the propose path becomes device-resident, and the paged+graph draft forward re-arms

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding), wave W8.
**Owning spec:** [dflash2-spec-decode.md](dflash2-spec-decode.md). This file is
the wave's own spec so that the shared row file takes only a pointer edit; the
row's `## Work breakdown` names this wave and links here.
**Issues:** [#1837](https://github.com/mudler/vllm.cpp/issues/1837) (the
25 MB/step round trip and the fallen-back paged+graph draft forward) and
[#1838](https://github.com/mudler/vllm.cpp/issues/1838) (the aux
D2H→cast→H2D→D2H context-accumulation chain). One flow, one pull request, by
operator direction at dispatch: both issues reshape the same propose path, and
half a reshape would leave the seam carrying both marshaling conventions at
once.
**Pull request shape:** ONE pull request, spec commit first. The row's original
claim recorded separate spec and implementation pull requests for the ROW; the
dispatching operator directed one pull request for THIS wave, and commit order
inside it still proves the spec came first.

## Why

Measured 2026-08-24 on dgx GB10 (#1574 workload, chat, c1): DFlash2 step
141.5 ms median ITL against vLLM ~127 and SGLang ~121 at equal acceptance
(~8.2 of 9). AR is at parity (89.2 ms vs 88), so the whole remaining gap is
per-step propose overhead. `VT_SPEC_TRACE` attributes only
`backbone=19.0ms sample=1.6ms` of it; the remaining ~31 ms is the untimed
propose pre-phase plus sync overhead. The issue bodies carry the full traced
census; the short form:

- The paged/graph draft fast path requires `final_out == nullptr`
  (`src/vllm/model_executor/models/qwen3_dflash.cpp::ForwardBlockLogitsWithDeviceKV`,
  the P==1 branch guard). A DFlash2 draft passes `&block_hidden`
  (`src/vllm/v1/worker/gpu/runner.cpp::propose_drafts_block`), so every step
  takes the materialized fallback: per-step re-materialization of the whole
  context, an eager 5-layer forward, and a full 8.94 MB f32 logits D2H
  (`ForwardWithCtxKVDev`'s tail download).
- The selector then host-copies the 8 sample rows (~8 MB), re-uploads them
  (`Qwen3DFlash2Model::ComputeCandidates`'s `DBuf dev_logits` upload), runs the
  device TopK, downloads ids and values, and the walk uploads again
  (`vllm::v1::Dflash2WalkPath`). ~10 full-queue `Synchronize`s per step.
- The propose pre-phase downloads the whole `[T, H*taps]` bf16 aux tap, casts
  it on the host, `CombineAuxFeatures` uploads it back, GEMMs, and downloads
  the result again; the accepted-prefix gather then copies rows on the host and
  `AppendContextKVDevice` re-uploads them
  (`runner.cpp::propose_drafts_block` steps 1–3,
  `qwen3_dflash.cpp::CombineAuxFeatures`, `::AppendContextKVDevice`).
- The in-code debt note names this exactly
  (`runner.cpp::propose_drafts_block`, the "COST, named rather than
  discovered" comment): the `final_out` host contract costs a DFlash2 draft
  the CUDA-graph draft step "until W4 computes the candidates inside the
  forward instead of after it". W4 landed the device walk and never moved the
  candidates. This wave is that move.

## Upstream chain

Merged upstream at `b389ac29465b33f9e9c534df221ea3c129e9793f` (the merged
vllm-project/vllm#52816 head the row's `## Upstream chain` pins). Read via
`git show` on the fetched object, not a checkout:

- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py::DFlashSpeculator.propose`
  and `._generate_draft`: the propose consumes the target's aux hidden device
  tensors directly (`combine_hidden_states(torch.cat(aux_hidden_states,
  dim=-1))`, `self.hidden_states[:num_target_tokens].copy_(...)`), precomputes
  and stores context KV device-side, and runs the query forward either through
  `DFlashCudaGraphManager.run_fullgraph` or eagerly. No hidden state and no
  logit crosses the host boundary during propose.
- `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py::DFlash2Speculator._generate_draft`:
  the sample-row gather is a device index
  (`last_hidden_states[self.sample_indices[:num_sample]]`), the candidate
  top-K runs on device through `compute_candidates`, the edge scores through
  `self.model.model.candidate_selector(...)`, and `_selector_walk_kernel`
  writes `draft_tokens` on device. Only the K draft token ids come back.
- `vllm/model_executor/layers/logits_processor.py::LogitsProcessor.get_top_k_tokens`
  (:241-286): head → optional padding mask → `_topk` → id rebase by
  `org_vocab_start_index` → `values.float()` → scale → soft cap, all on the
  device tensors. Our `vt::TopKValuesIndices` + `vt::MulScalar` + `vt::SoftCap`
  sequence is the same pipeline in the same order.

Nothing here advances the parity pin; the port stays BEYOND-PIN exactly as the
row records.

## Scope

In scope:

1. **(#1837) Device handles out of the block forward.** A new optional
   `DflashBlockDeviceOut` out-parameter on
   `Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV` (and on the shared
   `ForwardWithCtxKVDev` core behind it) that exposes the post-final-norm
   hidden (bf16) and the logits (f32) as DEVICE tensors with pool-backed
   ownership (`DBuf::ReleaseShared`). Requesting it does NOT disqualify the
   P==1 paged branch, so a DFlash2 draft re-enters the D13 paged forward and
   its CUDA-graph capture. The `final_out` host contract stays for the parity
   and test surfaces that use it; the production DFlash2 propose stops using
   it.
2. **(#1837) The device-resident selector.**
   `vllm::v1::Dflash2SelectCandidatesDevice` +
   `vllm::v1::Dflash2WalkPathDevice`, composed from device cores
   `Qwen3DFlash2Model::ComputeCandidatesDevice` and
   `::SelectorEdgeScoresDevice`: sample-row `vt::IndexSelect` on device, device
   `vt::TopKValuesIndices`, `vt::MulScalar` + `vt::SoftCap` on the surviving
   values, `vt::Dflash2SelectorEdges` and `vt::Dflash2PathWalk`
   device-to-device. The only download in the whole selector+walk is the final
   `[num_reqs, k]` i64 draft token ids. The existing host-vector entries
   (`Dflash2SelectCandidates`, `Dflash2WalkPath`, `ComputeCandidates`,
   `SelectorEdgeScores`) become marshaling shells over the SAME device cores,
   so there is one implementation of the math and the unit-test surface keeps
   its contract.
3. **(#1838) The device-resident pre-phase.**
   `Qwen3DFlashModel::CombineAuxFeaturesDevice` consumes
   `exec_state_.spec_aux.tensor` (bf16, already on device) directly — one
   `vt::MatmulBT`, no casts, no round trip — and
   `::AppendContextKVDeviceRows` gathers the accepted-prefix rows with a
   device `vt::IndexSelect` (indices the rejection output already determines
   on the host) and feeds the context-KV projection device-side. The host
   `CombineAuxFeatures` and `AppendContextKVDevice` become shells over the
   same cores. This serves DFlash1 and DFlash2 alike: the pre-phase is the
   shared DFlash block seam.
4. **Sync coalescing where it falls out.** The rejection sampler's two
   Synchronized downloads become two copies and one `Synchronize`
   (`src/vllm/v1/spec_decode/rejection_sampler.cpp::RejectionSampler::forward`).
   The `[spec-phase]` trace gains a `pre=` term so the pre-phase this wave
   moves is attributed instead of untimed.

Out of scope: any throughput claim (owed, operator-run, `## Owed`); the DSpark
and MTP lanes; the ngram proposer; any change to the verify or the rejection
rule itself; the W7 async-scheduling wave (#1824, open PR #1835), which this
wave does not touch and does not depend on.

## Design

### The shapes

```
struct DflashBlockDeviceOut {            // qwen3_dflash.h
  vt::Tensor logits;                     // [Tq, draft_vocab] f32, device
  vt::Tensor hidden;                     // [Tq, H] bf16, device (post-final-norm)
  std::shared_ptr<void> keep_logits;     // pool-backed ownership when the call
  std::shared_ptr<void> keep_hidden;     //   owns the storage; empty when the
};                                       //   store's persistent graph buffers own it

struct Dflash2CandidateSetDevice { ... } // ids [rows,K] i64 + values [rows,K] f32, device
struct Dflash2ProposeStateDevice { ... } // candidates + edge scores [B,L,K,K] f32, device
```

Headers expose `vt::Tensor` views plus `std::shared_ptr<void>` keeps (the
`DBuf::ReleaseShared` carrier), so no header grows a `dense_attn` include.

### The paged/graph lane with device outputs

The P==1 paged branch keeps its exact guard conjuncts (`final_out == nullptr`
among them); the new `device_out` parameter is allowed through. Three sub-lanes:

- **Eager paged** (CPU, `VT_DFLASH_GRAPH=0`, `VLLM_CPP_CUDAGRAPH=0`, or
  capture unsupported): `ForwardPagedBody` gains an optional
  `std::optional<DBuf>* out_final_hidden`; the dnorm buffer moves out instead
  of dying in scope, and both DBufs transfer into `device_out` via
  `ReleaseShared`. No host download happens when `device_out` is set.
- **Capture step**: the eager warm pass runs exactly as today and its buffers
  are still dropped before `BeginCapture` — keeping them alive would change
  the pool free-list at capture time and turn the capture's pool hits into a
  forbidden mid-capture `cudaMalloc`. The capture then files the segment with
  the persistent `st.g_logits` and the new `st.g_final_hidden`, and the step
  finishes with ONE immediate `st.g_graph.Replay(queue)`: under real CUDA the
  captured buffers hold no computed values until a replay, and on the
  capture-capable CPU harness the "capture" executed eagerly so the replay is
  a no-op over already-correct values. Either way `device_out` hands out
  views of the persistent buffers holding the same bits the warm pass
  computed. Cost: one extra draft forward on the capture step only, once per
  request lifetime.
- **Replay steps**: replay refreshes `st.g_logits`/`st.g_final_hidden` in
  place (fixed addresses); `device_out` views them with empty keeps (the
  store owns them, and the selector consumes them in the same step). The
  DFlash1 lane (no `device_out`) is byte-identical to before on all three
  sub-lanes.

The materialized fallback (`P>1`, a separate mask embedding, parity dumps, or
`VT_DFLASH_PAGED=0`) fills `device_out` at `ForwardWithCtxKVDev`'s tail — the
same dnorm and logits DBufs it already computes, released instead of
downloaded — so the device selector is the ONE selector lane for every DFlash2
propose, and `device_out` plus `final_out` together are refused by name
(no caller wants both; a silent both would download what the reshape exists to
keep resident).

When `device_out` is set the function returns an EMPTY host logits vector.
The DFlash1 `sample` fallback never runs for a DFlash2 block
(`RefuseDflash1ArgmaxOnDflash2Block` guards emptiness exactly as before), and
the `[spec-phase]` trace prints the device logits' numel instead of the host
vector's size.

### The device selector

`Dflash2SelectCandidatesDevice(block_logits_dev, block_hidden_dev, anchors,
num_reqs, k, weights, config, queue)`:

1. Build the sample-row index `{r*(1+k)+1+j}` on the host (integers the caller
   already owns), upload `[P*k]` i32, `vt::IndexSelect` the f32 logits rows and
   the bf16 hidden rows. This is upstream's
   `last_hidden_states[self.sample_indices[:num_sample]]`, and it replaces the
   host `std::copy` gather.
2. `ComputeCandidatesDevice`: `vt::TopKValuesIndices` (padding mask arg
   unchanged), then `vt::MulScalar(values, output_multiplier)` then
   `vt::SoftCap(values, final_logit_softcapping)` when the cap is set — the
   same order the host loop ran, and the CPU kernels are the same float
   arithmetic (`LoadF32(x,i) * s`; `c * std::tanh(v / c)` with the commutative
   multiply), so the values are bit-identical to the host loop's.
3. `SelectorEdgeScoresDevice`: `vt::MatmulBT` of the bf16 sample hidden
   against `hidden_projection` directly — the host path's f32 download,
   re-upload and `CastBf16` was an exact bf16→f32→bf16 round trip, so
   projecting the original bf16 bits is the same GEMM input — then
   `vt::Dflash2SelectorEdges` with the device ids/values/anchors. The scores
   stay on device.
4. `Dflash2WalkPathDevice`: `vt::Dflash2PathWalk` over the device scores and
   candidates, then the single `[B*L]` i64 download, with the same i32-range
   refusal the host walk carries.

**The org-vocab id rebase stays the host caller's step, by the op's own
documented polarity** (`vt::TopKValuesIndicesArgs`: "applied by the CALLER ...
an id-space rebase of the result rather than a property of the search").
`org_vocab_start_index` is structurally 0 on every path this engine ships
(single device, no vocab-parallel sharding — the long-form argument lives in
`Qwen3DFlash2Model::ComputeCandidates`'s comment and is not repeated here), so
the device lane's un-rebased ids ARE the final ids, and the host
`ComputeCandidates` shell keeps applying the rebase to its downloaded ids so
the synthetic shard-index tests keep their contract. The day this engine pads
or shards an LM head, the device lane needs the rebase on device in the SAME
edit — recorded beside the existing padded-head note rather than duplicated
as a second registry.

### The device pre-phase

`CombineAuxFeaturesDevice(aux_bf16_tensor)` → `[T, H]` bf16 device
(`ResidentWeight(fc)` + one `vt::MatmulBT`). The host chain it replaces —
bf16 aux D2H, host bf16→f32 scalar loop, f32 H2D, `CastBf16`, GEMM,
`CastF32`, f32 D2H, per-row host gather, f32 H2D, `CastBf16` inside
`PrecomputeContextKVDevice` — is a sequence of exact bf16→f32→bf16 round
trips around the identical GEMM, so the device path is bit-identical by
construction and the tests below pin it rather than assume it.

`AppendContextKVDeviceRows(store, combined_dev, rows, new_positions, ...)`:
uploads the `[append]` i32 row index, `vt::IndexSelect`s the accepted-prefix
rows, and runs the SAME projection+scatter tail as `AppendContextKVDevice` —
`PrecomputeContextKVDevice` splits into a bf16-tensor core plus its old
f32-host shell, and the paged-slot `IndexCopy` tail is shared. The
ascending-position ordering, the position-contiguity refusal and the
capacity refusal are unchanged and now sit in the shared tail.

The runner keeps every host-side decision it has today: the row sort by
position, the accepted-prefix count, the discontinuity `VT_CHECK`, the
discard handling and the block assembly are integer bookkeeping over host
step inputs and stay put. Only the FLOATS stop commuting.

## Risks and decisions

- **D1 — the paged forward becomes the DFlash2 production lane, and paged vs
  materialized must be bit-identical for a CONV draft.** The claim is already
  in the tree for DFlash1 ("Bit-identical to ForwardWithCtxKVDev over the same
  context", pinned by `tests/vllm/v1/spec_decode/test_dflash_propose.cpp`'s
  D11 cases); the conv is a per-row stream op called identically in both
  bodies. This wave pins the DFlash2 case with its own test (T2 below) instead
  of inheriting the DFlash1 claim. **Stop condition:** if the DFlash2 paged
  forward diverges bitwise from the materialized one on the CPU fixtures, this
  wave STOPS and returns NEEDS_DECISION with the measured divergence — no
  tolerance papers over it.
- **D2 — the capture-step replay.** Handing out device views on the capture
  step requires computed values in the persistent buffers; the warm pass's
  buffers cannot outlive the pool-warm contract. The one-replay-after-capture
  design (above) is the smallest shape that satisfies both, costs one forward
  once per request lifetime, and reads identically on the CPU harness and on
  CUDA. Rejected alternatives: keeping warm buffers alive (breaks the pool
  free-list precondition the capture comment documents), a D2D copy epilogue
  into pre-allocated outputs (adds a per-replay copy forever to save one
  forward once).
- **D3 — the host entries survive as shells.** `Dflash2SelectCandidates`,
  `Dflash2WalkPath`, `ComputeCandidates`, `SelectorEdgeScores`,
  `CombineAuxFeatures`, `AppendContextKVDevice` all delegate to the device
  cores. This is not a parallel path: the math has exactly one implementation,
  and the shells are the marshaling the unit tests and the
  `DflashProposeBlock` test entry need. `DflashProposeBlock` itself keeps its
  host contract (it is a test/demo entry; the runner is the production entry)
  and stays draft-identical to the runner lane because every lane runs the
  same cores over the same bits.
- **D4 — contract changes to existing tests are limited and argued.** The one
  pinned host contract this wave deliberately moves is the runner's propose
  call shape (`final_out=&block_hidden` → `device_out`), and the focused test
  that pinned the old shape (T1's red form, below) moves WITH the production
  shape in the same commit — the W7 precedent for `test_runner`'s async
  contract. No other existing test changes; the existing DFlash2 suites'
  drafted tokens are the identity gate and must stay green unmodified.
- **D5 — `spec_aux` dtype.** The device combine consumes the aux tap as bf16
  (`Qwen3_5AuxTaps.tensor` is documented bf16 `[T, H*taps]`); the runner's old
  host loop assumed the same thing silently (`std::vector<uint16_t>`). The
  device path asserts the dtype by name instead of assuming it.
- **D6 — CUDA-arm evidence is owed, not claimed.** This box has no GPU. The
  graph lane is held on the capture-capable CPU harness
  (`tests/vllm/models/decode_graph_seam_harness.h`), which holds ROUTING and
  the capture step's numerics but cannot see a real replay recompute; the
  `[DFLASH-GRAPH]` counters printing captures/replays for a DFlash2 draft on
  a real device, and the TPOT number, are owed operator-run (`## Owed`).

## Tests

Red-first, on this box (CPU), before implementation:

- **T1 (RED captured, then moved with the production shape — D4):** under
  `vllm_test::StaticGraphCpu`, a DFlash2 draft's propose forward in the
  runner's production shape must capture through the seam:
  `vt::GraphBreakStats.segments_captured == 1`, `full_scopes == 1`, replays
  advancing on later proposes. On the current tree the DFlash2 shape passes
  `final_out` and takes the fallback, so the stats stay 0 — that is the
  captured red, and it is the defect #1837 names. The green form calls the
  same forward with `device_out`, which IS the new production shape.
- **T2 (identity, new API):** DFlash2 P==1 `device_out` paged-eager logits and
  hidden, downloaded, are BIT-IDENTICAL to the `VT_DFLASH_PAGED=0`
  materialized lane and to the `ForwardBlockLogitsWithContext` full-recompute
  reference (D1's pin).
- **T3 (identity, new API):** `Dflash2SelectCandidatesDevice` +
  `Dflash2WalkPathDevice` drafts are BIT-IDENTICAL to the host
  `Dflash2SelectCandidates` + `Dflash2WalkPath` drafts over the same block
  outputs. Mutation (a) below reds exactly this.
- **T4 (identity, new API):** `CombineAuxFeaturesDevice` +
  `AppendContextKVDeviceRows` produce a store whose subsequent block-forward
  logits are BIT-IDENTICAL to the host `CombineAuxFeatures` + host gather +
  `AppendContextKVDevice` store's.
- **T5 (engine, lanes agree):** the `test_dflash2_runner_reach` engine fixture
  drafts identically with `VT_DFLASH_PAGED=1` and `=0` — after this wave the
  two spellings select different lanes and must agree to the bit.
- **Existing suites unmodified** are the standing identity gate:
  `test_qwen3_dflash2_draft`, `test_dflash2_runner_reach` (its measured
  2-of-8 scalar-arm divergence in particular), `test_dflash2_argmax_guard`,
  `test_dflash_propose`, `test_qwen3_dflash_decode_graph_seam`,
  `test_qwen3_dflash_forward`, the runner spec-decode suites, and the
  committed parity goldens' gate. A drafted token that moves anywhere in these
  is a red this wave must not ship.

## Mutations (the fresh reviewer's set, named in advance)

- **(a)** Swap the device sample-row gather to include the anchor row (`+0`
  instead of `+1..+k`): the pinned sample-row case ("the SAMPLE rows are
  +1..+k, never the anchor row") and the reach suite must red. T3 deliberately
  does NOT red here and that is a property, not a gap: the gather has exactly
  ONE implementation, both of T3's lanes share it, and their equality survives
  any shared defect — which is why the absolute pins and the production reach
  gate carry this mutation. RUN 2026-08-24: `test_qwen3_dflash2_draft` 39/40
  (`st.candidates.ids[j*K] == 1 + j` NOT correct, four assertions),
  `test_dflash2_runner_reach` 1/4 cases red (109/110 assertions;
  reviewer-verified — the 3/4 shape belongs to mutation (c)). Restored;
  suites green again (404/404, 110/110).
- **(b)** Break the DFlash2 capture admission (drop `device_out` from the
  paged guard's allowance, or force the fallback): T1's
  `segments_captured == 1` must red. RUN 2026-08-24: adding
  `device_out == nullptr` to the paged guard read `CHECK( 0 == 1 )` on the
  segment, scope and replay assertions. Restored; green.
- **(c)** Reachability: delete the runner's `Dflash2SelectCandidatesDevice`
  call site: the drafts come back empty, `RefuseDflash1ArgmaxOnDflash2Block`
  fires by name, and the reach suite reds loudly. RUN 2026-08-24: `if (false
  && dflash2)` at the call site threw the named refusal on every engine case
  (3/4 cases red, `threw` non-empty). Restored; green.
- **(d)** Reachability (#1838): delete the runner's
  `AppendContextKVDeviceRows` call site. **RUN 2026-08-24, and the first run
  was a FINDING, not a pass**: with the call deleted the reach suite stayed
  GREEN (4/4) — `dflash_ctx_len_` kept advancing, so the position-discontinuity
  check never fired, the store stayed empty, `ctx_cu` stayed self-consistent at
  {0,0}, and every propose ran CONTEXT-FREE with well-formed drafts. A lossless
  verify makes that invisible to every token gate: only acceptance falls. This
  is the same shape as the pre-existing check's blind spot, so the repair is a
  second production invariant beside it: the runner's counter and the DEVICE
  store's `num_ctx` must agree
  (`propose_drafts_block`, "the context-KV append is dead or double-run"),
  a host integer comparison with no transfer. Re-run with the invariant in
  place, the same mutation throws by name on the second propose and the reach
  suite reds on every case. Restored; suite green (110/110).

## Gates

- Focused: the suites above, run red-first where the surface exists, then
  green, on this box.
- Full: `scripts/agent-preflight.sh` clean before every push
  (`test_cpu_x86_llamacpp_floor` is the known #618 flake; a red there is
  re-run serially before being read).
- Speed: NOT claimed here. See `## Owed`.

## Owed

- **O1 — the GPU numbers, operator-run.** The #1574 workload TPOT/ITL with
  this wave, against the recorded vLLM 62.8 tok/s / 15.49 ms and SGLang
  66.9 / 14.72; the `[DFLASH-GRAPH]` counters printing captures and replays
  for a DFlash2 draft on a real device; and the sync census re-taken. This
  wave's estimate is #1837's 8–12 ms/step plus #1838's 2–5 ms/step, and an
  estimate is not a result.
- **O2 — the real-replay numerics.** The CPU harness cannot see a replayed
  segment recompute (its replay recomputes nothing); the DFlash2 graph lane's
  replay-vs-eager bit-identity on device rides the same operator lease as O1.

## Evidence (implementer, CPU box, 2026-08-24)

- **RED, captured before the implementation** on the spec-commit tree
  (`ac7b987f6` + the red-form test only): the W8 graph case drove
  `ForwardBlockLogitsWithDeviceKV` in the runner's then-production DFlash2
  shape (`final_out=&block_hidden`) under `vllm_test::StaticGraphCpu` and read
  `CHECK( s.segments_captured == 1 ) is NOT correct! values: CHECK( 0 == 1 )`
  — the `final_out` contract disqualified the paged branch and nothing was
  ever captured. That is #1837's defect, observed as a failing assertion.
- **GREEN after**: the same assertion set over the new `device_out` shape reads
  1 segment, 1 full scope, replays advancing (1 on the capture step — the D2
  self-replay — then +1 per propose), and the capture-lane logits+hidden
  BIT-IDENTICAL to the materialized fallback.
- **The identity chain, all green and all exact** (`test_qwen3_dflash2_draft`
  40 cases / 404 assertions; `test_dflash2_runner_reach` 4 cases / 110):
  eager-paged == materialized (T2); device selector+walk == host selector+walk
  draft-for-draft over shifted blocks with the Muse scalars on (T3); device
  aux pre-phase == host aux pre-phase, store-for-store through the block
  forward, with an out-of-order accepted-prefix gather (T4); paged lane ==
  materialized lane at the ENGINE, block-for-block (T5). The D1 stop condition
  never fired: the paged DFlash2 forward is bit-identical to the materialized
  one on these fixtures.
- **Every pre-existing DFlash suite passes unmodified** — including the
  value-sensitive measured margins (`walk != argmax`, the 5-of-6 and 2-of-8
  scalar-flip counts), which a one-ulp drift in the moved value postprocess
  would have moved.
- The mutation set (a)-(d) is recorded under `## Mutations`; the implementer's
  own runs are in the pull request evidence, and the fresh reviewer re-runs
  them independently.

## Now

Spec and implementation in one pull request (operator-directed shape). CPU
gates green; the GPU numbers and on-device counters stay owed under `## Owed`.
