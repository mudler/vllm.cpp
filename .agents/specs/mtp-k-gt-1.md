# SPEC-MTP-K-GT-1: MTP speculation depth k>1

Row: `SPEC-MTP-K-GT-1`, under `ROAD-V1-D3-SPEC-K`
([roadmap_v1.md](../roadmap_v1.md) row 12a) and `SPEC-MTP`
([engine-matrix.md](../engine-matrix.md)).

Issue: [#81](https://github.com/mudler/vllm.cpp/issues/81), milestone M1
("MTP k>1"). The issue is already listed in
[issue-index.md](../issue-index.md) against `SPEC-MTP`, so this row adds no
index line.

Git integration: no `.agents/developer-preferences.md` exists in the shared
checkout, and no split case applies, so this row uses the repository default of
ONE pull request. Commit order proves the spec came first.

## Verdict

`--num-speculative-tokens 3` is accepted today and silently produces one draft
token per request. Nothing refuses it and nothing logs it, so a user who
configures depth 3 pays for depth 3 in KV reservation and in graph capture and
receives depth 1 in throughput.

This row makes MTP depth genuinely configurable. Setting k gives k. The
deliverable is the working depth, not the refusal.

## Our baseline: the present-tense defect, verified

Verified against `332aed738` in this worktree, not taken from the issue text.

- `src/vllm/v1/worker/gpu/runner.cpp:2164-2176` stashes exactly one draft
  token per request: `out.draft_token_ids.push_back({drafts[i]})`, with the
  comment `k=1: one token/request`.
- `src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp:14-86`
  (`MtpProposePrefill`) returns `std::vector<int32_t>` of length `num_reqs`.
  There is one value per request and no place to put a second.
- The rest of the engine honours the configured k.
  `src/vllm/v1/core/kv_cache_utils.cpp:1002` sizes snapshot slots by
  `num_speculative_tokens`. `src/vllm/v1/worker/gpu/cudagraph_dispatch.h:45-46`
  captures the verify shape at `1 + num_speculative_tokens`.
  `src/vllm/entrypoints/model_loader.cpp:842-843` widens the KV spec by
  `ResolvedNumSpeculativeTokens()`.
- `grep -rn num_speculative_tokens src/ include/` returns 91 lines over 23
  distinct files and no `throw`, no `VT_CHECK` and no refusal keyed on a value
  above 1 for the MTP method.

The other three proposers already honour k and are NOT in scope for any
refusal, checked rather than assumed:

- n-gram returns 0..k per request per step
  (`src/vllm/v1/worker/gpu/runner.cpp:2273-2282` moves the whole per-request
  vector).
- DFlash proposes a `1 + k` block
  (`src/vllm/v1/worker/gpu/runner.cpp:2182-2201`, `num_query_per_req = 1 + k`
  at `:2196`).
- DSpark proposes `num_query_per_req()` rows
  (`src/vllm/v1/worker/gpu/runner.cpp:2208-2230`, `layout.num_query_per_req()`
  at `:2222`).

Each of those resolves its config through its own `Resolve*` entry point, so a
refusal placed on the MTP entry point cannot reach them.

## Scope

IN:

- Phase 1. Refuse an MTP k above 1 with a message naming the missing part, at
  `LoadedEngine::ResolveSpecConfig`. Landed as its own commit so the defect is
  closed even if Phase 2 stalls.
- Phase 2. Port vLLM's autoregressive multi-step propose for the MTP head,
  remove the Phase 1 refusal in the same flow, and make the configured depth
  reach the scheduler as k draft tokens per request.
- Per-depth acceptance telemetry, required by #81's M1 checklist.
- CPU tests that prove the configured depth took effect, and the greedy
  token-equivalence of k=1, k=3 and spec-OFF.

OUT, and recorded under `## Owed`:

- M3 `SPEC-DYNAMIC` (`num_speculative_tokens_per_batch_size`).
- M4 the acceptance-EMA adaptive controller.
- M5 the graphed `1 + k` verify shape.
- Any llama.cpp-side work. llama.cpp is comparison context for WHY depth
  matters, never a mirror source here.

## Upstream chain

Pinned oracle `5559679229bc961848b121ccdeaa8fa5d79bec98`
([upstream-sync.md](../upstream-sync.md)), checkout `/home/mudler/_git/vllm`.
Every anchor below was read at that revision.

`vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`:

| Anchor | What it defines |
|---|---|
| `:129-274` `propose` | The whole shape: one prefill forward, then the k=1 early exit at `:236-238`, then `prepare_decode_inputs`, then `_multi_step_decode`, then `return self.draft_tokens[:num_reqs]` |
| `:236-238` | The early exit this tree ported. `num_speculative_steps == 1` returns `draft_tokens[:num_reqs, :1]` |
| `:335-370` `_prefill` | Sample step 0 from the `last_token_indices` rows, then carry BOTH the sampled hidden and the positions of those rows into the per-request decode buffers |
| `:374-419` `_multi_step_decode` | `for step in range(1, num_speculative_steps)`: rebuild slot mappings and draft attention metadata each step, set `current_draft_step`, generate |
| `:426-471` `_generate_draft` | One draft forward over `num_reqs` tokens, sample, then `update_draft_inputs` |
| `:597-671` `prepare_decode_inputs` | The step-1 entry state. `input_ids[r] = draft_tokens[r,0]`, `positions[r] = min(positions[r]+1, max_model_len-1)`, `seq_lens[r] = min(target_seq_lens[r] - num_rejected[r] + 1, max_model_len)`, `query_start_loc = identity` |
| `:674-771` `update_draft_inputs` | `draft_tokens[r, step] = token`; return early when `step >= num_speculative_steps - 1`; else feed the token and the hidden forward and advance position and `seq_len` by one |
| `:52-59` `advance_draft_positions` | True for standard MTP. Gemma4 MTP is the false case and is not a Qwen3.5 arch, so this port takes the true branch only |

`vllm/v1/spec_decode/llm_base_proposer.py:682` is the v0-runner sibling loop
named in the dispatch brief. The executing chain for the v1 GPU runner this
tree mirrors is the `autoregressive/speculator.py` path above, reached from
`vllm/v1/worker/gpu/model_runner.py` `speculator.propose(...)`, so the port
follows `autoregressive/speculator.py`. Recorded because the two disagree in
buffer shape and only one of them is our mirror.

`vllm/config/speculative.py:973-987` resolves the default k for this
architecture to `n_predict`, which both gate checkpoints set to 1. So k=1 stays
the DEFAULT after this row. k>1 becomes reachable, not automatic.

## Port map (design)

### 4.1 Phase 1, where the refusal goes and why

At `LoadedEngine::ResolveSpecConfig`, on the `mtp` arm
(`src/vllm/entrypoints/model_loader.cpp:812-833`), beside the per-method
requirements the other three arms already carry.

Four reasons, and two rejected alternatives.

1. Every entry point reaches it. The CLI, the `--speculative-config` JSON, the
   C ABI and the in-memory `LoadedEngine` constructors all resolve here, and the
   `LoadedEngine` constructor runs it as its FIRST member initialisation
   (`src/vllm/entrypoints/model_loader.cpp:971`). One check covers all of them.
2. It fails at engine construction, BEFORE the KV pool is widened for a depth
   that will not be used (`resolved_spec_config_` is initialised before
   `kv_cfg_` in the same member list) and before any graph capture.
3. The thing that cannot serve depth is THIS ENGINE's propose path, not the
   config value. A capability gate belongs on the engine's resolution seam.
4. The `ngram`, `dflash` and `dspark` arms return before this line, so the
   refusal cannot reach the three proposers that genuinely serve k>1.

Rejected: `SpeculativeConfig::ResolveMtp`
(`include/vllm/config/speculative.h:75-95`). It is a faithful mirror of
`speculative.py`'s resolution, and upstream imposes no depth bound there, so a
throw inside it would be a divergence in the mirrored type. It also broke the
scheduler tests, which construct an MTP config at k=3 through `ResolveMtp`
(`tests/vllm/v1/test_scheduler.cpp:1214,1299,1334`) to exercise the
scheduler half, which HAS always served depth. That red was the correct signal
that the config type is not where the capability lives.

Rejected: the propose call site in `GPUModelRunner::propose_drafts`. It refuses
after the load, repeatedly, on a configuration already paid for, and it needs a
method predicate to avoid catching the other proposers, which is a second place
for the method routing to drift.

The refusal is ONE `if` block and its message. Phase 2 deletes exactly that
block. It widens no bound and leaves nothing stranded.

### 4.2 Phase 2, the multi-step propose

The k=1 path stays byte-identical. The prefill forward, its input preparation
and its argmax are untouched, and `k == 1` still returns after the prefill.

New host routines, mirroring the two Triton kernels one for one:

- `prepare_decode_inputs` (`speculator.py:597-671`) in
  `src/vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_decode_inputs.cpp`,
  beside the landed `prepare_prefill_inputs.cpp`. Same file, same directory and
  same header shape as its prefill sibling, because upstream keeps them in one
  module.
- `update_draft_inputs` (`speculator.py:674-771`) in the same translation unit.

The propose splits in two. `MtpProposePrefill` keeps its exact signature and
delegates to an internal `ProposePrefill` that also returns the two values
`_prefill` carries forward and the old signature dropped on the floor: the
`last_token_indices` rows and their positions. `MtpProposeDrafts` is the new
public entry, taking k, `max_model_len` and `block_size`, and returning
`[num_reqs * k]` row-major. Keeping the old entry point means the landed k=1
tests still exercise the shipping k=1 path rather than a rewritten one.

The decode step's attention metadata is built from the target's, which is what
upstream does through `self.block_tables.compute_slot_mappings` over the same
block tables:

- `query_start_loc` becomes the identity `0..num_reqs`, one token per request.
- `seq_lens` comes from `prepare_decode_inputs` and then advances one per step.
- `block_table_tensor` is the target's, unchanged. The draft KV layer
  (`fa_draft`) has the same block geometry as the target's full-attention
  group, so the same block ids address it.
- `slot_mapping[r] = block_table[r][pos / block_size] * block_size + pos %
  block_size`, with `block_size` read from `draft_attn_kv_[0].block_size`.

The hidden-state carry needs a device row gather (`hidden_states[
last_token_indices]`, `speculator.py:363-371`). `vt` device buffers are built
in the anonymous namespace of `src/vllm/model_executor/models/qwen3_5.cpp`, so
the gather is exposed as a method on `Qwen3_5MTPModel` rather than reimplemented
in the speculator. That keeps the device-buffer idiom in one file and does not
create a parallel path.

### 4.2a What depth does to the CUDA-graph layer

Two hazards were raised against this row. Both were checked against the tree
rather than accepted, and they resolve differently.

**This row does not make the spec graph slot key collide. The stronger claim that
it does NOT collide at all is WITHDRAWN, because it was over-strong and a fresh
review was right to reject it.**

`src/vllm/model_executor/models/qwen3_5.cpp:9276` (and its dense sibling at
`:9698`) keys the slot ring on `S` alone, and `:9253` sets `S = B` exactly on a
spec step while padding `B` onto the capture ladder otherwise, so `4 x (1+1)` and
`2 x (1+3)` are the same key.

The withdrawn argument reasoned only INSIDE the spec-admission branch. It said
that the spec graph is admitted only by
`IsUniformDecodeBatch(..., num_speculative_tokens)`, that `num_spec()` is a
constant for the engine's lifetime, and therefore that within one slot map the
query length is fixed and `S` determines the request count uniquely. Each step is
true about the spec branch and the conclusion still does not follow, because the
predicate that reaches the ring is
`uniform_decode = input.pure_decode || (spec_graph && ...)`
(`qwen3_5_moe.cpp:143-148`, `qwen3_5_dense.cpp:172-177`). The ring is shared by
BOTH disjuncts, so two query lengths do reach one map: at k=1, 8 requests
pure-decode and 4 requests spec both give `S = 8`, and at k=3, 2 requests spec
gives `S = 8` as well. `SizeSlot` invalidates on `fa_cols` and on `aux_taps`
(`qwen3_5.cpp:9280-9330`) and on nothing keyed to spec-versus-pure.

What survives is the claim this row actually needs: the ambiguity is PRE-EXISTING
and this row does not widen it. It arrived with SPEC-DSPARK W8 (#442), which made
a uniform spec batch capturable at all by re-expressing
`ValidateGdnDecodeGraphState` (`qwen3_5.cpp:469-497`). Configurable depth adds no
new way for two query lengths to share one key, because `num_spec()` is still one
value per engine and the slot map is a member of `Qwen3_5DecodeGraph::Impl`, so
two engines at different depths do not share one. It is also CPU-untestable here,
for the same reason the rest of the graph layer is: no CPU gate observes graph
capture.

The residual is folded into
[#1020](https://github.com/mudler/vllm.cpp/issues/1020), which already owes the
`(S, q)` re-key, and the `## Owed` entry below states it. Folded rather than filed
separately because it is one repair in one commit: a predicate widened to the
step's actual query length is safe only once the ring distinguishes those query
lengths, and the ring needs to distinguish them only because both branches share
it.

One stale comment the same audit found IS repaired in flow, because it is a false
statement rather than a design question. `qwen3_5.cpp:9177` and `:9601` both said
"spec never captures, ValidateGdnDecodeGraphState rejects a spec batch" and
concluded that the spec-segmentation copies beside them are inert. #442 made both
halves false. The comment now says those copies are inert on the pure-decode path
only and load-bearing on a captured spec step, and it names #1020 for the
residual.

**Shape COUNT is unchanged, shape SIZE grows with depth.** A spec step takes
`S = num_reqs x (1 + k)` with `num_reqs` bounded by `max_num_seqs`, so the
number of distinct captured shapes stays `max_num_seqs` at every k. Each shape
is `(1 + k)` times larger, so the persistent per-shape logits buffer scales with
depth. This row therefore does not multiply the shape count. The separate
concern that a spec step takes the exact `num_reqs x (1 + k)` instead of padding
onto `DecodeGraphSizes` is not this row's, and nothing here makes it worse.

**A step whose ACTUAL depth differs from the configured depth falls silently to
eager.** Verified: the predicate above compares against the configured k, so any
other uniform query length misses it and the verify runs eager with no log and
no counter. This is present-tense at k=1 and this row makes it reachable in one
NEW shape. `src/vllm/v1/core/sched/scheduler.cpp:616-622` truncates a request's
drafts to what the step's token budget fits, so at k=3 a clamped step can give
every request 2 drafts, produce a uniform query length of 3, and miss a
predicate expecting 4. At k=1 truncation is all-or-nothing and the batch simply
stops being uniform, which is a shape no graph exists for anyway. The cost is
measured, not hypothetical: capturing the `1+k` verify moved the 35B cells from
0.870x to 0.986x (`c5615cfe0`), and falling out of the graph gives that back.

This is NOT fixed here, and the reason is precise rather than convenient. The
fix is to make the predicate read the step's actual uniform query length and to
key the slot ring on `(S, q)` so the widened predicate stays unambiguous. That
widens the capture set, and the capture set is a persistent-memory decision that
has to be measured on a GPU before it lands. The GPU is held by another session
for the whole of this flow, and no CPU gate can observe graph capture at all.
Filed and listed under `## Owed` with what it must show.

### 4.2b Per-step depth is a value, not a constant read from config

The propose loop takes its depth as an explicit parameter, resolved once per
step at the runner call site rather than read from `spec_config_` inside the
loop. This costs nothing now and it is the seam a future scheduler-supplied
depth needs, mirroring `vllm/v1/core/sched/scheduler.py:1121-1126`, where the
scheduler decides how many tokens to speculate for a step. No adaptive
mechanism is built here, and none is implied.

### 4.2c `spec_step_idx` stays 0 on every draft step, as upstream leaves it

`Qwen3_5MTP.forward` selects its layer with `spec_step_idx % num_mtp_layers`
(`vllm/model_executor/models/qwen3_5_mtp.py:162`), which reads like something the
multi-step loop should advance. It does not. `grep -rn spec_step_idx
vllm/v1/worker/gpu/` returns NOTHING at the pin: the v1 GPU runner's speculator
never passes it, so every draft step takes the default 0 and reuses layer 0. Our
`ForwardPaged` defaults the same way and the loop leaves it alone, which is the
faithful mirror. It is also inert on both gate checkpoints, whose
`mtp_num_hidden_layers` is 1, so `step % 1` is 0 regardless. Recorded because the
modulo makes the absence look like an omission on our side rather than upstream's
current behaviour.

### 4.3 Shared seams

- The propose stays behind `GPUModelRunner::propose_drafts`, which
  `ModelRegistry::Forward`'s caller already routes to. No new propose path.
- Drafts still leave through `DraftTokenIds` and the existing
  `take_draft_token_ids` seam, which already carries variable-length drafts for
  the n-gram proposer, so the scheduler and the verify path need no change.
- The draft forward stays on `Qwen3_5MTPModel::ForwardPaged`.

Nothing here needs a seam that cannot represent the behaviour.

### 4.4 Telemetry

`spec_drafts_proposed_` and `spec_drafts_accepted_`
(`include/vllm/v1/worker/gpu/runner.h:671-672`) count tokens and cannot answer
"how deep did acceptance reach". #81's M1 asks for per-depth acceptance, and
M4's controller needs exactly that signal, so this row adds per-depth counters
beside them and exposes them read-only.

### 4.5 What the CPU tier can and cannot run, measured

Two constraints were found by running the gate, not by reading the code.

**A GDN-hybrid spec VERIFY cannot run on CPU with bf16 state.**
`vt::CausalConv1dSpecUpdate` requires an f32 conv state and rejects bf16 off
CUDA (`src/vt/ops.cpp:1806`), and the model dtype makes the conv state bf16. No
spec verify step had ever executed on CPU in this tree before this row, because
the k=1 engine tests ran with a NULL drafter and never produced a draft to
verify. The CPU depth gate therefore runs the f32-state arm through the escape
`MakeQwen3_5KVCacheSpec` already reads (`VT_GDN_STATE_BF16=0`,
`src/vllm/model_executor/models/qwen3_5_common.cpp:54-63`). This is sound for
what is under test: the MTP head is itself `layer_type="full_attention"`
(`qwen3_5_mtp.py:105-112`) and every draft decode step touches only the draft's
own paged full-attention KV layer, so the depth loop never reaches the GDN
rollback path. The bf16 arm stays covered by the owed DGX gate.

**A configured k that the propose does not honour produces WRONG TOKENS on the
GDN hybrid path, not merely lost throughput.** Measured, as a mutation of the
landed code: clamping `propose_drafts` back to one draft per request while the
engine is configured for k=3 made the greedy output DIVERGE from
speculative-off (`{19,5,3,9,15,6,14,15,7,3}` against `{19,5,3,9,18,2,1,1,1,1}`),
where the correct implementation matches it exactly. That is the configuration
this tree shipped before this row, and it is stronger than the throughput
argument the issue makes: the Phase 1 refusal was closing a correctness hole and
not only a billing one. It is also the same class of defect as
[#1020](https://github.com/mudler/vllm.cpp/issues/1020), where an actual depth
below the configured one silently de-graphs, so both have one root: the engine
is sized from the configured k and nothing checks that the propose delivered it.

## Risks/decisions

| Risk | Why it matters | Control |
|---|---|---|
| A token-identity gate cannot see a silently clamped k | It is exactly today's bug: k=3 emits the same greedy tokens as k=1 because greedy plus accept-iff-equal is depth-independent | Every depth test asserts a POSITIVE witness of depth beside the identity. The witness has to be counted where the WORK happens, not read off the emitted draft list: `spec_mtp_draft_decode_forwards() == spec_mtp_propose_calls() * (k - 1)`. Draft-list length and per-depth counter size were tried first and a fresh review broke both with a one-forward padded propose, recorded in `## Outcome` |
| The draft KV slot for an advanced position may collide with the target's | The draft writes at positions the target has not reached | Mirrors upstream exactly, which writes the same slots in its own draft KV layer. The draft KV is a separate group (`fa_draft`) |
| `max_model_len` overrun at depth | Position and `seq_len` can pass the bound at the tail of a sequence | Mirrored clamps from `speculator.py:637-645` and `:732-737` |
| A test that constructs the proposer by hand proves nothing about reachability | AGENTS.md "Nothing lands dead" | The depth tests drive `GPUModelRunner::execute_model`, the runner the loader builds, with the depth arriving through the same `SpeculativeConfig` the loader resolves |
| A discarded (still-prefilling) row's draft-decode steps write draft KV at positions the target has not reached | Its draft is thrown away, but the k-1 decode forwards still carry its row, so they write `position+1 .. position+k-1` into the draft KV layer | Mirrors upstream, which carries every row through `_multi_step_decode` for the same reason (a uniform batch shape). The scheduler reserves `num_lookahead_tokens` blocks for every scheduled request, so those positions are allocated, and `draft_decode_slot_mapping` refuses loudly rather than silently addressing another request's page if that ever stops holding |
| The GPU is held by another session | The k=2..4 three-way on real checkpoints cannot run in this flow | Recorded under `## Owed` with the exact conditions it must show. Not silently skipped |

## Tests to port

Red-before is captured for every claimed guarantee.

CPU, in this flow. All three suites run on the synthetic Qwen3.5 dense model, so
none needs a checkpoint or a GPU.

| Suite | What it pins | Result |
|---|---|---|
| `test_speculative_mtp_depth` (new) | The mirrored config type: the k=1 default, the upstream divisibility rule, and that the type CARRIES a depth rather than clamping it | 4 cases, 20 assertions |
| `test_prepare_decode_inputs` (new) | The two Triton-kernel ports: the step-1 entry state, the between-step advance, the final-step early return, both `max_model_len` clamps, and the draft slot mapping | 8 cases, 33 assertions |
| `test_mtp_depth` (new) | Depth through the PRODUCTION loader at k=1, 2, 3, 4, with a real `mtp.*` head loaded by `LoadQwen3_5MTP`, the greedy equivalence of spec-OFF, k=1 and k=3, and the draft-decode-forward equality that is the actual depth witness | 5 cases, 47 assertions |

Red-before evidence, all with real counts:

1. Phase 1, at the engine seam: without the refusal,
   `CHECK_THROWS_AS(LoadedEngine(..., k=3))` did not throw at all. 13
   assertions, 4 failed, `Status: FAILURE!`.
2. Phase 2, at the engine seam: with `propose_drafts` mutated back to the
   pre-fix clamp (`push_back({drafts[base]})`), the depth witness read
   `VerifiedDepth == 1` against 3, at k=2 and k=4 likewise. 24 assertions, 5
   failed, `compile_err = 0`. The k=3 identity ALSO broke under that mutation,
   which is recorded in section 4.5 because it is a stronger fact than the issue
   claims.
3. The two kernel ports are new functions, so their "red" would be a link error
   and would prove nothing. Five mutations instead, each rebuilt and rerun with
   `compile_err = 0` and a DISTINCT failure count, so no two were the same edit:
   the position clamp off by one (1 failed), `seq_len` dropping `num_rejected`
   (4), the final-step early return moved by one (5), the slot mapping ignoring
   `position / block_size` (2), and the between-step `seq_len` advance removed
   (3). Restored byte-for-byte and green afterwards.

## Gates

Full CPU gate: `scripts/agent-preflight.sh` plus the built test suite. The suite result is `100% tests passed, 0 tests failed out of 495`, with 2 checkpoint-gated skips (`test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e`).

Build note, recorded because it changed the evidence's arm. The first full CPU
build here was `CMAKE_BUILD_TYPE=Debug` and reached 49 GB across roughly 500 test
binaries, which filled the host disk to 100% and left one test executable
UNLINKED. `ctest` then reported that test as `***Not Run` rather than as a build
failure, which is the ENOSPC-presents-as-a-verdict shape. The full suite is
therefore run at `Release`, which is the configuration `scripts/build-cpu-release.sh`
itself uses. The three new suites were additionally run under `Debug` while they
were being written, so the assertion-bearing arm is covered too.

**The `.agents/issue-index.md` checker conflict is RESOLVED, and the premise this
row recorded for it was wrong.** An earlier revision of this section said the two
checkers cannot both be satisfied by any edit to that file, so the duplicated
`#995` row was RESTORED after being collapsed, and the conflict was filed as
[#1027](https://github.com/mudler/vllm.cpp/issues/1027) and listed as owed.
[#1025](https://github.com/mudler/vllm.cpp/pull/1025) landed on `origin/main` and
falsified that premise.

The mechanism is worth stating, because it is the union-merge hazard this
repository already knows about seen from the checker's side.
`check-issue-index-append-only.py:47-56` diffs
`merge-base(origin/main, HEAD)..HEAD`, NOT the file's own history. The two
checkers therefore conflict only while the duplicate sits in the MERGE BASE. Once
`main` itself merged the two rows by key, the merge base became a file with one
well-formed `#995` row, and a branch that takes main's version and appends its own
rows removes nothing at all. Both checkers are then green on one tree, with
neither widened nor argued away.

Taking main's version is also what the Records rule requires of a keyed record:
take the complete target-branch version, apply the scoped edit again, and verify
that unrelated keys stay byte-for-byte equal. The union driver's automatic result
is the wrong one here and it is wrong in the silent direction. `git merge-tree`
against `origin/main` produced a tree with TWO `#995` rows again, because a union
merge re-adds a line one side deleted. That is what this branch would have
resurrected, and it would have turned an INHERITED red into this branch's own
defect. After the repair the same `git merge-tree` yields one row.

#1027 is now a duplicate of #1022, which #1025 closed. It is still OPEN on GitHub
and this flow has no authority to close it, so that remains an operator action.
Its index row is written to say what actually happened rather than to repeat the
false premise, which is safe because the row has never reached `origin` and no
other branch carries it, so no union merge can duplicate it.

`test_cpu_x86_llamacpp_floor`'s contended-leg case is load-dependent and is not
repaired here either. Measured on ONE tree at `1554494c3`, five runs: FAIL at
loadavg 157 with `NO_QUIET_WINDOW` (4) instead of `GIVING_UP` (2), FAIL at 19,
then PASS at 21, 25, 23 and 22. That is
[#618](https://github.com/mudler/vllm.cpp/issues/618) verbatim, and the loadavg
157 run was this branch's own full build. The attribution is measured rather than
asserted, which matters because a single red would otherwise read as this diff's
fault: nothing in this change is on that harness's path.

### Final gate, at `1554494c3` against `origin/main` `4f2d91756`

`scripts/agent-preflight.sh`, by block:

| Block | ok | FAIL |
|---|---:|---:|
| Session role | 1 | 0 |
| Record gates | 25 | 1 (`check-agent-record`, the inherited #995 duplicate) |
| Mutation suites | 42 | 2 (`test_agent_record`, same duplicate; `test_cpu_x86_llamacpp_floor`, the load flake above) |
| Committed range vs `origin/main` | 2 | 1 (`doc-checkpoint range`, below) |
| Commit trailers vs `origin/main` | 2 | 0 |

The trailer block EXECUTED rather than skipping. It is guarded on
`git merge-base --is-ancestor origin/main HEAD`
(`scripts/agent-preflight.sh:226-228`), and an earlier run on this branch skipped
it silently because `origin/main` had advanced mid-flow. That is why this branch
carries two merges rather than one.

`ctest` on the same head: `100% tests passed, 0 tests failed out of 495`, with
two checkpoint-gated skips.

`doc-checkpoint range` named commit `4ad2bec87`, the Phase 1 refusal: it changed
`src/vllm/entrypoints/` and did not update `docs/USAGE.md` in the SAME commit,
which it genuinely owed, because it changed what `--speculative-config` accepts.
`check-doc-checkpoint.py:401` walks `rev-list --reverse --no-merges base..head`
(`:374`) PER COMMIT, so no later commit can pay that debt, and the only in-branch
repair is to amend the commit. The earlier flow was directed not to rebase and
reported it. It is now REPAIRED: the branch was unpushed, the amend was
authorised for this branch only, and the commit (now `c5511d12f`) carries the
`docs/USAGE.md` line it owed, which is that for `mtp` a `num_speculative_tokens`
above the resolved default of 1 is refused at load because the multi-step draft
propose is not ported. The Phase 2 commit replaces that text with the depth text
now shipping, so the flag row is correct after every commit in the range rather
than only at the end.

The amend was verified by TREE IDENTITY, not by inspection. The rebase preserved
both merge commits (`git rebase -i --rebase-merges`), and
`git diff <old-head> <new-head>` is EMPTY: the rewritten branch's final tree is
`3ad6ecfe0` on both sides, byte for byte. So the only thing the history edit
changed is which commit carries the documentation line, which is exactly what the
checker measures.

Known pre-existing red, NOT caused by this row and not repaired here.
`check-env-doc` and `test_check_env_doc` failed on pristine `332aed738`
(`VT_MOE_EXPERT_STREAM`, `VT_MOE_EXPERT_STREAM_SLOTS`,
`VT_MOE_EXPERT_STREAM_SLOT_BYTES` undocumented, arrived with #993), tracked by
[#995](https://github.com/mudler/vllm.cpp/issues/995) and owned by
`ENG-EXPERT-STREAM`. Reproduced on the pristine base before any edit here, which
is how it was attributed rather than assumed. `45b022cdc` (#997) landed the
documentation on `origin/main` mid-flow and is merged in, so the final gate on
this branch has both green.

### Records this row deliberately does NOT write

`.agents/porting-inventory.md` section 9 gains no entry. The one deviation this
port carries is that the draft-decode arrays are built at the request count
instead of padded to `max_num_reqs`, because our draft decode is not
graph-captured and its only consumer requires the exact width. It is recorded in
the header that owns it, which is where a reader of the code meets it, and the
precedent is in-tree: the prefill sibling's own tail-pad deviation lives in
`BlockTable::compute_slot_mapping`'s comment and not in section 9, whose entries
are architecture-scale choices. A reviewer who disagrees should say so rather
than assume it was missed.

`.agents/NOW.md` is written at operator cadence only and is not a per-row
lifecycle write, so it stays untouched.

`.agents/engine-matrix.md` DOES gain a row, and the first attempt here argued it
should not. The argument was that a matrix row edits a file every concurrent pull
request also edits, which AGENTS.md names as a lock, and that this spec plus
roadmap row 12a were already a sufficient per-row surface. `check-agent-record`
refused that immediately: `roadmap_v1.md: references unknown stable row
SPEC-MTP-K-GT-1`. It is right and the argument was wrong. AGENTS.md's Records
section says every inventory item records its state in the correct matrix, and a
stable ID that only the roadmap knows about is exactly the dangling record the
checker exists to catch. Recorded because a reviewer will otherwise wonder
whether the row was added deliberately or by trial and error. It was the second,
and the checker was the one that knew.

## Dependencies

Nothing new. The row consumes what already landed: `prepare_prefill_inputs`
(SPEC-MTP I5b), `Qwen3_5MTPModel::ForwardPaged` (I5c), the `fa_draft` draft KV
group, the greedy rejection sampler and its `num_rejected` accounting (I3), and
the `DraftTokenIds` / `take_draft_token_ids` seam, which already carried
variable-length drafts for the n-gram proposer. No new `vt` op, no new CUDA
kernel, no new upstream pin. The OWED GPU gate depends on `dgx.casa` and on the
pinned oracle being able to run `--speculative-config mtp` at k=2..4, which is
upstream's own supported configuration.

## Work breakdown

| Step | What | State |
|---|---|---|
| W1 | The spec, its upstream anchors, and the two graph-layer hazards checked rather than assumed | LANDED (`98ab752f5`) |
| W2 | The refusal at `LoadedEngine::ResolveSpecConfig`, red-first at the engine seam. Independently landable, so the defect closes even if W3 stalls | LANDED (`4ad2bec87`) |
| W3 | `prepare_decode_inputs` + `update_draft_inputs` + `draft_decode_slot_mapping`, the two Triton-kernel ports, mutation-gated | LANDED (`93d6bc329`) |
| W4 | `MtpProposeDrafts` (the multi-step loop) + `Qwen3_5MTPModel::GatherHiddenRows` + the runner emitting k drafts + per-depth telemetry, and the W2 refusal REMOVED rather than widened | LANDED (`93d6bc329`) |
| W5 | The CPU depth gate through the production loader at k=1..4, plus the public-document projections | LANDED (`93d6bc329`) |
| W6 | The DGX three-way at k=2..4 on the 27B and 35B, on the DEFAULT bf16 GDN state | OWED, see below. The GPU lock was held for the whole flow |
| W7 | The matched-k throughput A/B and the acceptance-versus-depth curve | OWED, #81 M2 |

## Stop conditions

- Return `NEEDS_DECISION` if the k>1 semantics cannot be settled against the
  pinned vLLM rather than guessing them.
- Do not take the GPU lock. If `dgx.casa` holds `$HOME/gpu.lock`, record the
  GPU gate as owed and land the CPU evidence.
- Do not widen or delete a checker to make a red green.

## Owed

| Owed | What it must show | Who |
|---|---|---|
| DGX three-way greedy gate at k=2, 3, 4 on Qwen3.6-27B and 35B | our-ON == our-OFF == vLLM-ON, token for token on the golden prefix, at EACH depth, with nonzero acceptance and the per-depth counters populated at every depth up to k, plus spec-OFF byte-identical. It must run the DEFAULT bf16 GDN state, because the CPU gate here runs the f32 arm for the reason in section 4.5, so the GDN speculative rollback at depth is UNEXERCISED until this runs | `SPEC-MTP-K-GT-1`, [#81](https://github.com/mudler/vllm.cpp/issues/81) M1 |
| Silent de-graphing when the actual depth differs from the configured k, AND the `S`-only slot-ring key ([#1020](https://github.com/mudler/vllm.cpp/issues/1020)) | The spec-graph predicate reads the step's ACTUAL uniform query length instead of `num_spec()` (`runner.cpp:1383`), and the graph slot ring is keyed on `(S, q)` in the SAME change. The re-key is owed on its own merits and NOT only as a consequence of widening the predicate, which is the correction section 4.2a records: `uniform_decode = input.pure_decode \|\| (spec_graph && ...)` (`qwen3_5_moe.cpp:143-148`, `qwen3_5_dense.cpp:172-177`) already routes TWO query lengths to one `impl_->slots[S]` (`qwen3_5.cpp:9276`, dense `:9698`), and `SizeSlot` invalidates on `fa_cols` and `aux_taps` only (`:9280-9330`). At k=1, 8 requests pure-decode and 4 requests spec both key on `S = 8`. That is pre-existing since SPEC-DSPARK W8 (#442) and this row does not widen it, but it is not the benign thing the first spec revision claimed. Plus a measured before-and-after on the capture-set size and persistent logits memory, and a counter or log for the eager fallback so it can never again be invisible | `SPEC-MTP-K-GT-1`, [#1020](https://github.com/mudler/vllm.cpp/issues/1020) |
| Close [#1027](https://github.com/mudler/vllm.cpp/issues/1027) as a duplicate of [#1022](https://github.com/mudler/vllm.cpp/issues/1022) | NOT a code or record debt. The defect #1027 describes is FIXED on `origin/main` by [#1025](https://github.com/mudler/vllm.cpp/pull/1025), and this branch takes main's single well-formed row rather than resurrecting the duplicate, so `check-agent-record` and `check-issue-index-append-only` are BOTH green here. What remains is one remote write this flow has no authority for. Detail and the mechanism are in `## Gates` | operator, [#1027](https://github.com/mudler/vllm.cpp/issues/1027) |
| M2 speed A/B at matched k | concurrency-1 and concurrency>1 throughput against vLLM same-config at matched k, plus the acceptance-versus-depth curve for prose and for code | [#81](https://github.com/mudler/vllm.cpp/issues/81) M2 |
| M3 `SPEC-DYNAMIC` | `num_speculative_tokens_per_batch_size` and the dense batch-size to k lookup, mirrored from `vllm/config/speculative.py:177` and `vllm/v1/spec_decode/dynamic/utils.py:7,77` | [#81](https://github.com/mudler/vllm.cpp/issues/81) M3 |
| M4 adaptive depth (acceptance EMA) | The controller is EXPLICITLY optional and out of scope here. It needs: an acceptance EMA that rises slowly and decays about twice as fast, a mapping from that EMA to a depth, default OFF, and an A/B on a mixed prose and code workload that it must WIN before it ships as a default. It is from-scratch, so it also owes a [porting-inventory.md](../porting-inventory.md) section 9 record. CONSTRAINT carried from #81: if the spec verify step is ever graphed, a graph is captured per K (`vllm/v1/worker/gpu/cudagraph_utils.py:200-220`), so K must be quantised to a small captured ladder and never a free 1..9. The ladder and the capture set are therefore designed together, not separately | [#81](https://github.com/mudler/vllm.cpp/issues/81) M4 |
| M5 graphed `1 + k` verify shape | Coupled to M4's ladder for the reason above | [#81](https://github.com/mudler/vllm.cpp/issues/81) M5 |

## Now

`ACTIVE`. Phase 1 landed as its own commit and was removed by Phase 2 in the same
flow, so no released state refuses a depth it can serve. The CPU evidence is
complete and the GPU gate is owed above, so the row is NOT `DONE`: no speed
number is claimed at k>1, which is the whole point of depth.

## Outcome (partial, CPU half)

**Measured.** Depth reaches the verify path at k=1, 2, 3 and 4 through
`LoadedEngine`, and the greedy token stream is identical to speculative-off at
every one of them (`test_mtp_depth` 5/5, 47 assertions).

**Corrected: the per-depth counters do NOT distinguish "the loop ran k times"
from "the loop ran more than once", and an earlier revision of this section
claimed they do.** A fresh review disproved it by mutation. Return straight after
the prefill and pad all k columns with the step-0 draft, so that ONE forward runs
in total, and the suite stayed green at 5/5, 34/34, `Status: SUCCESS`, with
`compile_err = 0` and the mutation demonstrably applied. The witness was
`spec_drafts_proposed_by_depth().size()`, grown from
`step.num_draft_tokens_per_req[i]` (`runner.cpp:1780-1782`), which is the LENGTH
of the emitted draft list and therefore a pure function of the runner's slicing at
`runner.cpp:2209-2212`. Anything that hands the verify path k tokens satisfies it.
The reviewer also probed the counters in both arms at k=3 and got identical
output, `proposed: 7 7 7 accepted: 0 0 0` on the real loop and
`proposed: 7 7 7 accepted: 0 0 0` on the padded fake. The equality across depths
held trivially at zero, and index k-1 was reached by list length alone. The
shipped code did run k steps. The GATE was the defect.

**The witness that replaced it: draft decode forwards, counted where the work
happens.** `MtpProposeDrafts` returns `MtpDraftProposal`, whose
`num_draft_decode_forwards` is incremented AFTER each draft decode forward
returns. The runner accumulates it beside a count of propose calls and exposes
both read-only, so the depth assertion is the exact equality
`spec_mtp_draft_decode_forwards() == spec_mtp_propose_calls() * (k - 1)`, with
`spec_mtp_propose_calls() > 0` required first so the equality cannot hold
vacuously. No draft-list shape produces that equality. Re-run of the reviewer's
exact mutation against it: `Status: FAILURE!`, 3 of 5 cases failed, 4 of 47
assertions failed, `compile_err = 0`, exit code 1, and the four failures are
exactly the four depth equalities (`0 == 16` at k=3 with 8 calls, `0 == 20` at
k=3 with 10 calls, `0 == 8` at k=2, `0 == 24` at k=4). Every one of the other 43
assertions, including every list-length and every token-identity assertion, still
PASSED under the mutation, which is the reviewer's finding reproduced rather than
argued.

**Rejected: non-zero acceptance at depth >= 2 as the CPU witness.** It is the
right assertion and it is unavailable here. Acceptance is measured at ZERO at
every depth on the synthetic gate model, in both arms, so the acceptance profile
cannot separate them.

**Rejected: distinctness of the k drafted tokens.** A correct drafter may repeat
a token, and on this 24-entry vocabulary it often does, so the property belongs to
the model rather than to the loop.

**What the CPU half therefore establishes, and what it does not.** It establishes
that k drafts are PROPOSED, that k-1 draft decode forwards run per propose call at
every k tested, that k drafts reach the verify path, and that the greedy stream
does not move. It does NOT establish that the accept path works at depth, because
not one draft is ever accepted at any depth on this model. The prefix-monotonicity
assertions over `spec_drafts_accepted_by_depth()` are consequently vacuous, and
they are kept because the invariant is the right one to state. The owed DGX gate
demanding non-zero acceptance at every depth is not belt-and-braces. It is the
only place the accept path at depth is exercised at all.

**Rejected: the refusal inside `SpeculativeConfig::ResolveMtp`.** It reddened
the scheduler tests, which build an MTP config at k=3 to exercise a scheduler
half that has always served depth. The red was correct: the config type is a
mirror of upstream and upstream bounds no depth there.

**Rejected: token identity as the depth gate.** Greedy plus accept-if-equal
makes the emission independent of k, so identity passes on a drafter clamped to
1. Every depth assertion is paired with a positive witness for that reason.

**Why each default has its value.** k defaults to the checkpoint's
`mtp_num_hidden_layers`, which is 1 on both gate checkpoints, so this row
changes no default. vLLM resolves the same default the same way
(`speculative.py:977-979`), and until the owed throughput A/B says depth wins,
raising the default would be a speed claim without a measurement.
