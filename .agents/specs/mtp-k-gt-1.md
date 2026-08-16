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

## 0. Verdict

`--num-speculative-tokens 3` is accepted today and silently produces one draft
token per request. Nothing refuses it and nothing logs it, so a user who
configures depth 3 pays for depth 3 in KV reservation and in graph capture and
receives depth 1 in throughput.

This row makes MTP depth genuinely configurable. Setting k gives k. The
deliverable is the working depth, not the refusal.

## 1. The present-tense defect, verified

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

## 2. Scope

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

## 3. Upstream anchors

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

## 4. Design

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

`MtpProposePrefill` gains the two values `_prefill` carries forward and the
current signature drops on the floor: the sampled hidden rows and the positions
of the `last_token_indices` rows. The k=1 caller ignores both, so its behaviour
does not move.

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

**The spec graph slot key does NOT collide, and this row does not make it
collide.** `src/vllm/model_executor/models/qwen3_5.cpp:9237` keys the slot ring
on `S` alone, and `:9214` sets `S = B` exactly on a spec step, so `4 x (1+1)`
and `2 x (1+3)` are the same key. That is only reachable if two different
uniform query lengths can reach the same slot map. They cannot. The spec graph
is admitted only by `IsUniformDecodeBatch(..., num_speculative_tokens)`
(`src/vllm/model_executor/models/qwen3_5_moe.cpp:145-148` and the dense sibling
`qwen3_5_dense.cpp:174-177`), which requires the uniform query length to equal
`1 + num_spec()` EXACTLY. `num_spec()` is a constant for the engine's lifetime,
so within one slot map the query length is fixed and `S` determines the request
count uniquely. The slot map is a member of `Qwen3_5DecodeGraph::Impl`, reached
through `qwen.decode_graph()`, so it is per-model and two engines at different
depths do not share one. This row changes none of that: it makes depth
configurable, not variable within a process.

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

## 5. Risks

| Risk | Why it matters | Control |
|---|---|---|
| A token-identity gate cannot see a silently clamped k | It is exactly today's bug: k=3 emits the same greedy tokens as k=1 because greedy plus accept-iff-equal is depth-independent | Every depth test asserts a POSITIVE witness of depth (drafts of length k, and per-depth counters above zero) beside the identity |
| The draft KV slot for an advanced position may collide with the target's | The draft writes at positions the target has not reached | Mirrors upstream exactly, which writes the same slots in its own draft KV layer. The draft KV is a separate group (`fa_draft`) |
| `max_model_len` overrun at depth | Position and `seq_len` can pass the bound at the tail of a sequence | Mirrored clamps from `speculator.py:637-645` and `:732-737` |
| A test that constructs the proposer by hand proves nothing about reachability | AGENTS.md "Nothing lands dead" | The depth tests drive `GPUModelRunner::execute_model`, the runner the loader builds, with the depth arriving through the same `SpeculativeConfig` the loader resolves |
| The GPU is held by another session | The k=2..4 three-way on real checkpoints cannot run in this flow | Recorded under `## Owed` with the exact conditions it must show. Not silently skipped |

## 6. Tests and gates

Red-before is captured for every claimed guarantee.

CPU, in this flow:

1. `ResolveMtp` refuses k>1 by name, and accepts k=1 (Phase 1). Deleted by
   Phase 2 and replaced by its inverse: `ResolveMtp` ACCEPTS k>1 and carries
   the value.
2. `prepare_decode_inputs` and `update_draft_inputs` against their upstream
   kernels, including the `max_model_len` clamps and the final-step early
   return.
3. Depth reaches the runner: at k=3 through `GPUModelRunner::execute_model`,
   every non-discarded request's stashed draft has length 3, and the per-depth
   counters record proposals at depths 1, 2 and 3. At k=1 the length is 1. This
   is the test that fails today.
4. Greedy token equivalence: spec-OFF, k=1 and k=3 emit the identical token
   sequence over the same synthetic model and prompt, position for position.
   Greedy plus accept-iff-equal makes the emission depth-independent, so this
   is an equality and not a tolerance.

Full CPU gate: `scripts/agent-preflight.sh` plus the built test suite.

Known pre-existing red, NOT caused by this row and not repaired here:
`check-env-doc` and `test_check_env_doc` fail on pristine `332aed738`
(`VT_MOE_EXPERT_STREAM`, `VT_MOE_EXPERT_STREAM_SLOTS`,
`VT_MOE_EXPERT_STREAM_SLOT_BYTES` undocumented, arrived with #993). Tracked by
[#995](https://github.com/mudler/vllm.cpp/issues/995) and owned by
`ENG-EXPERT-STREAM`. Reproduced on the pristine base before any edit in this
worktree.

## 7. Stop conditions

- Return `NEEDS_DECISION` if the k>1 semantics cannot be settled against the
  pinned vLLM rather than guessing them.
- Do not take the GPU lock. If `dgx.casa` holds `$HOME/gpu.lock`, record the
  GPU gate as owed and land the CPU evidence.
- Do not widen or delete a checker to make a red green.

## 8. Owed

| Owed | What it must show | Who |
|---|---|---|
| DGX three-way greedy gate at k=2, 3, 4 on Qwen3.6-27B and 35B | our-ON == our-OFF == vLLM-ON, token for token on the golden prefix, at EACH depth, with nonzero acceptance and the per-depth counters populated at every depth up to k, plus spec-OFF byte-identical | `SPEC-MTP-K-GT-1`, [#81](https://github.com/mudler/vllm.cpp/issues/81) M1 |
| Silent de-graphing when the actual depth differs from the configured k ([#1020](https://github.com/mudler/vllm.cpp/issues/1020)) | The spec-graph predicate reads the step's ACTUAL uniform query length instead of `num_spec()` (`runner.cpp:1383`), and the graph slot ring is keyed on `(S, q)` in the SAME change, because today's single-query-length admission is the only thing that keeps the `S`-only key unambiguous (`qwen3_5.cpp:9214,9237`). Plus a measured before-and-after on the capture-set size and persistent logits memory, and a counter or log for the eager fallback so it can never again be invisible | `SPEC-MTP-K-GT-1`, [#1020](https://github.com/mudler/vllm.cpp/issues/1020) |
| M2 speed A/B at matched k | concurrency-1 and concurrency>1 throughput against vLLM same-config at matched k, plus the acceptance-versus-depth curve for prose and for code | [#81](https://github.com/mudler/vllm.cpp/issues/81) M2 |
| M3 `SPEC-DYNAMIC` | `num_speculative_tokens_per_batch_size` and the dense batch-size to k lookup, mirrored from `vllm/config/speculative.py:177` and `vllm/v1/spec_decode/dynamic/utils.py:7,77` | [#81](https://github.com/mudler/vllm.cpp/issues/81) M3 |
| M4 adaptive depth (acceptance EMA) | The controller is EXPLICITLY optional and out of scope here. It needs: an acceptance EMA that rises slowly and decays about twice as fast, a mapping from that EMA to a depth, default OFF, and an A/B on a mixed prose and code workload that it must WIN before it ships as a default. It is from-scratch, so it also owes a [porting-inventory.md](../porting-inventory.md) section 9 record. CONSTRAINT carried from #81: if the spec verify step is ever graphed, a graph is captured per K (`vllm/v1/worker/gpu/cudagraph_utils.py:200-220`), so K must be quantised to a small captured ladder and never a free 1..9. The ladder and the capture set are therefore designed together, not separately | [#81](https://github.com/mudler/vllm.cpp/issues/81) M4 |
| M5 graphed `1 + k` verify shape | Coupled to M4's ladder for the reason above | [#81](https://github.com/mudler/vllm.cpp/issues/81) M5 |

## Now

`ACTIVE`. Phase 1 landed as its own commit and removed by Phase 2 in the same
flow, so no released state refuses a depth it can serve. CPU evidence complete,
GPU gate owed above.
