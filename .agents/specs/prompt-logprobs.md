# `SAMPLE-PROMPT-LOGPROBS` — the runner prompt-logits source

*(Live spec, 2026-08-09. Base `origin/main` `bd6b3936`. Pin vLLM 0.26.0.dev0
`555967922`. Issue [#223](https://github.com/mudler/vllm.cpp/issues/223). Owner claim
`CLAIM-SAMPLE-PROMPT-LOGPROBS-W1`. Row
`SAMPLE-PROMPT-LOGPROBS` (`.agents/engine-matrix.md:132`, `PARTIAL`). Closes the
named residual of `ROAD-V1-C7` recorded in
[roadmap-v1-completion.md](roadmap-v1-completion.md) §1 and §3.4.)*

## Scope

Port vLLM's `_get_prompt_logprobs_dict` onto our runner so
`ModelRunnerOutput.prompt_logprobs_dict` is actually populated, making
`SamplingParams.prompt_logprobs` a working feature instead of a silent no-op.

**In scope (W1):** the runner-side source — logits at prompt positions, the
`log_softmax` + gather, the chunked-prefill `in_progress` accumulation, the
`-1` → `vocab_size` widening, emit-on-final-chunk, and the request-lifetime
bookkeeping. Gated on the CPU reference backend end-to-end through the engine to
`RequestOutput.prompt_logprobs`.

**Out of scope (named residuals, own follow-ups):** the OpenAI `echo` +
`prompt_logprobs` response serialization in
`serving_completion.cpp` / `serving_chat.cpp` (W2 — the helper
`BuildCompletionLogProbs` already handles the echoed-prompt position shape,
`serving_utils.cpp:122`); `logprobs_mode` variants beyond `raw_logprobs`
(`SAMPLE-LOGPROB-TOKEN-IDS`); prompt logprobs for prompt-**embeds** requests
(upstream skips them too, `gpu_model_runner.py:5635-5637`); any GPU speed axis
(upstream itself declares this path "a rare feature, prioritize simple,
maintainable loop over optimal performance", `:5622-5623`).

## Upstream chain

- `vllm/v1/worker/gpu_model_runner.py:5612-5719` — `_get_prompt_logprobs_dict`,
  the function being ported.
- `vllm/v1/worker/gpu_model_runner.py:3841-3845` — the call site: it runs on
  `hidden_states[:num_scheduled_tokens]` with `scheduler_output.num_scheduled_tokens`,
  after the sampled ids are written back.
- `vllm/v1/worker/gpu_model_runner.py:677-679,1199,1305-1310` — the
  `num_prompt_logprobs` map: seeded at `add_request` from
  `sampling_params.prompt_logprobs` (`vocab_size` when `-1`), popped when the
  request finishes, and deleted when its prefill completes (`:5710-5712`).
- `vllm/v1/sample/sampler.py:309,396` — `compute_logprobs` (log_softmax) and
  `gather_logprobs`, reused verbatim for prompt positions.
- `vllm/v1/engine/logprobs.py:121-206` — the consumer we already ported.

## Our baseline

Everything below the runner is DONE and gated (issue #223 lists the anchors):
`LogprobsProcessor::UpdatePromptLogprobs`/`pop_prompt_logprobs`
(`src/vllm/v1/engine/logprobs.cpp:75,100`), the scheduler slice
(`src/vllm/v1/core/sched/scheduler.cpp:915-938`), and
`RequestOutput.prompt_logprobs` (`src/vllm/v1/engine/output_processor.cpp:224`).
`grep -rn prompt_logprobs src/vllm/v1/worker/` returns nothing: the source is the
whole gap.

## Design

### Where the prompt logits come from

Upstream keeps `hidden_states` and calls `self.model.compute_logits(...)` on the
prompt slice. Our forward applies `lm_head` **inside** the model, so the question
is how to make it produce a row at every prompt position.

The forward already carries a row list — `ModelForwardInput::logits_indices` — so
naming the prompt positions there looks like the 1:1 move. **It is not
available.** Every model's gather-before-`lm_head` is guarded on
`logits_indices.size() < T` (21 sites across 19 model files, e.g.
`qwen3_5.cpp:6249`, `laguna.cpp:1247`, `deepseek_v2.cpp:598`): the gather is an
optimization for "far fewer rows than tokens", and a list that is not strictly
shorter than the token count is silently ignored, returning all `T` rows in
token-stream order instead. Scoring a whole prompt is exactly the case that
trips it — one sampler row plus `n-1` prompt rows for an `n`-token prompt is
`n` rows for `n` tokens. Widening that contract would mean editing every model.

So a step that owes prompt logits takes the **full-logits path** — the same one
`VT_LOGITS_GATHER=0` already selects — by passing an empty gather list. The
forward returns `[num_actual_tokens, vocab]`, `assemble_sample_logits` re-gathers
the sampler's rows through its existing case (B), and a prompt row is found by
its token-stream index. This is also the shape upstream computes: its
`compute_logits` call at `:5680-5682` runs `lm_head` over the whole prompt
slice, so there is no work here that vLLM does not also do.

**Inertness.** `step.prompt_logprob_indices` is empty on every step where no
request asked, and the gather flag is then exactly the expression it has always
been. No extra row, no extra allocation, no changed branch.

**Recorded deviation.** On a step where a request DID ask, the sampled rows come
out of a full-`T` `lm_head` rather than a gathered one. Same op per row, but a
different GEMM shape, so those rows are not guaranteed bit-identical to the same
step without the flag. It cannot reach a request that did not ask, and the same
is true upstream for the mirror-image reason (it adds a second `compute_logits`
over the same weights).

### Per-request row selection (1:1 `:5638-5670`)

For request `r` with `num_prompt_logprobs = k`, at `num_computed_tokens = start_idx`
and `num_scheduled_tokens = num_tokens`:

```
start_tok            = start_idx + 1
num_remaining_tokens = num_prompt_tokens - start_tok
num_logits           = num_tokens <= num_remaining_tokens
                         ? num_tokens                    (a chunk; more remain)
                         : num_remaining_tokens          (final chunk -> emit)
```

`num_logits <= 0` produces nothing (the exact-prefill edge, `:5668-5671`). The
rows are `query_start_loc[req_idx] + [0, num_logits)`; the target token for row
`i` is prompt token `start_tok + i` — the *next* token, which is what the logprob
is scored against.

### Accumulation + emit (1:1 `:5646-5652,5698-5712`)

A request's `LogprobsTensors` covers `num_prompt_tokens - 1` positions and
`k + 1` columns, allocated once on first sight and held in
`GPUModelRunner::in_progress_prompt_logprobs_` (our stand-in for upstream's
`request.in_progress_prompt_logprobs_cpu`; we have no per-request state object on
the runner). Each step writes the slice `[start_idx, start_idx + num_logits)`.
The final chunk moves the whole tensor into
`ModelRunnerOutput.prompt_logprobs_dict` and drops both the in-progress entry and
the `num_prompt_logprobs` entry. Aborted/finished requests are dropped in
`update_states`' removal pass (`:1199`).

### Scoring (1:1 `:5688-5697`)

`raw_logprobs = log_softmax(prompt_logits)` via `vt::ComputeLogprobs` — the same
op the sampler uses (`sampler.cpp:284`) — then `GatherLogprobs(raw, n, vocab, k,
tgt_token_ids)`. `GatherLogprobs` stays file-local; the pair is exposed as one
`Sampler::compute_prompt_logprobs` entry point, which is where upstream's
`self.sampler.compute_logprobs` / `gather_logprobs` calls live anyway and keeps
the device-buffer helper in the file that already owns it. Prompt
positions bypass every logits processor, which is why upstream notes
`processed_*` and `raw_*` coincide here (`:5691-5693`).

### `prompt_logprobs == -1`

Widened to `vocab_size` at admission, exactly as upstream (`:1306-1310`). This
differs from the `num_logprobs` sample-side handling, which preserves our `-1`
sentinel for the sampler to consume — that deviation is already recorded in
`input_batch.h`; the prompt path takes upstream's form because `GatherLogprobs`
needs a concrete `k`.

## Port map

| Upstream (`555967922`) | Local anchor |
|---|---|
| `gpu_model_runner.py:5626-5686` (row selection: `start_tok`, chunk vs final, `num_logits <= 0`) | `src/vllm/v1/worker/gpu/prepare_inputs.cpp` — the `num_prompt_logprobs` block building `StepInputs::prompt_logprob_rows` / `prompt_logprob_indices` |
| `gpu_model_runner.py:5688-5697` (`compute_logprobs` + `gather_logprobs` over prompt rows) | `Sampler::compute_prompt_logprobs`, `src/vllm/v1/sample/sampler.cpp` |
| `gpu_model_runner.py:5645-5651,5698-5706` (`in_progress_prompt_logprobs_cpu`, slice write) | `GPUModelRunner::collect_prompt_logprobs` + `in_progress_prompt_logprobs_`, `src/vllm/v1/worker/gpu/runner.cpp` |
| `gpu_model_runner.py:5665-5667,5709-5712` (emit on final chunk, drop both maps) | same function, the `final_chunk` branch |
| `gpu_model_runner.py:1305-1310` (seed `num_prompt_logprobs`, `-1` → `vocab_size`) | `InputBatch::add_request`, `src/vllm/v1/worker/gpu/input_batch.cpp` |
| `gpu_model_runner.py:1199` (drop with the request) | `InputBatch::remove_request` + `GPUModelRunner::drop_stale_prompt_logprobs` |
| `gpu_model_runner.py:3841-3845` (call site, after write-back) | `GPUModelRunner::sample_tokens` and `sample_tokens_async`, after `assemble_sample_logits` |
| `gpu_model_runner.py:5680-5682` (`compute_logits` over the prompt slice) | the full-logits route in `execute_model` (see Design) — no per-model change |

## Dependencies

None outstanding. Everything below the runner was already landed and gated
(`SAMPLE-LOGPROBS` W5, `ecda3ce1`): `LogprobsProcessor::UpdatePromptLogprobs`,
the scheduler slice, `RequestOutput.prompt_logprobs`. No new kernel, no vt op,
no ABI change, no dependency on the GPU: the whole row is gateable on the CPU
reference backend. Blocks nothing; `SAMPLE-LOGPROB-TOKEN-IDS` and the W2 `echo`
serialization sit downstream of it but neither is required to close W1.

## Work breakdown

- **W1 (this change).** The runner source, end to end to
  `RequestOutput.prompt_logprobs`, CPU-gated. Row `PARTIAL` → `ACTIVE`.
- **W2 (follow-up, own PR).** The OpenAI `echo` + `prompt_logprobs` response
  serialization on `/v1/completions` (`completion/serving.py:520-560`:
  `clamp_prompt_logprobs`, prompt-then-output token/logprob concatenation, the
  `max_tokens == 0` prompt-only arm) and the chat counterpart. Serving-layer
  only — `BuildCompletionLogProbs` already emits the echoed-prompt position
  shape (`serving_utils.cpp:122`). Closes the row to `DONE`.
- **Not in this row.** `logprobs_mode` variants (`SAMPLE-LOGPROB-TOKEN-IDS`),
  prompt-embeds requests (upstream skips them at `:5635-5637`), any speed axis.

## Risks/decisions

1. **Perturbing the production path.** Mitigated by strict inertness (above) and
   by a no-prompt-logprobs regression assertion in the gate.
2. **Chunked prefill off-by-one.** The `start_tok`/`num_remaining_tokens`
   arithmetic is the subtle part; the gate exercises a prompt split across
   chunks, plus the `num_logits <= 0` exact-prefill edge, both RED-first.
3. **Row ordering.** The prompt rows must be appended in the same request order
   the slicing loop assumes. Asserted in the runner and covered by a
   two-concurrent-request case.

### Recorded by the 2026-08-10 fresh review (PR #235)

The first four are latent — nothing reachable today executes them — and are
written down rather than fixed so the next change on this path starts from the
real lifetime model instead of re-deriving it. The fifth is an OPEN gap with an
owed gate (see Gates).

4. **`gather == false` hands the device sampler a HOST pointer.** With the
   full-logits route taken, every registry returns `HostLogits(...)`, so
   `assemble_sample_logits` case (B) (`runner.cpp:1348-1366`) builds
   `vt::Tensor::Contiguous(fl.host.data(), kF32, queue_.device, …)` — a host
   allocation labelled with the CUDA device — and hands it to the on-device
   sampler kernels. `vt::Tensor` carries no residency flag, so nothing catches
   it. Only GB10 **unified memory** has been reasoned about, where a host pointer
   is addressable from the device and this is sound. On a **discrete** GPU it is
   not obviously sound, and it has NOT been executed there. Before this row that
   path was reachable only behind the explicit `VT_LOGITS_GATHER=0` opt-out;
   this row makes it default-reachable on any step where a request asks for
   prompt logprobs. Owed: the CUDA smoke gate in Gates. Not narrowed further
   here because the box this row was implemented and gated on has no GPU.
5. **The in-progress tensor is keyed to input-batch membership, upstream's is
   not.** Upstream hangs `in_progress_prompt_logprobs_cpu` off
   `CachedRequestState`, which outlives the input batch; ours lives in
   `GPUModelRunner::in_progress_prompt_logprobs_` and
   `drop_stale_prompt_logprobs` erases anything the batch no longer carries.
   `update_states` removes any request not scheduled this step, and the
   scheduler can do that to a RUNNING, partially-prefilled request when the
   token budget runs out (`scheduler.cpp:391-396`). Its accumulated rows would
   then be dropped and silently refilled with zeros on re-admission. Latent only
   because re-admitting such a request is itself unimplemented
   (`prepare_inputs.cpp:65-67`); it becomes real the moment it is.
6. **`drop_stale_prompt_logprobs` runs only from `collect_prompt_logprobs`.** It
   is therefore skipped on a `num_reqs == 0` flush step and on the pooling early
   return, so an abort that empties the batch leaves the tensor resident until
   some later step calls in. Bounded (one tensor per aborted request until the
   next collect) and no correctness impact — the sweep is by req_id, so a later
   request reusing that id still finds it gone before it is read.
7. **A re-admitted preempted request would emit its payload twice.** The final
   chunk erases `num_prompt_logprobs` (`:5709-5712`); a preempted request
   re-admitted through the NEW-request path re-seeds it from
   `sp.prompt_logprobs`, so the prompt would be scored and appended a second
   time. Upstream seeds only on the new-request path too (`:1305-1310`), but its
   map is keyed to a request object that is not rebuilt on preemption. Latent
   for the same reason as 5.

## Tests to port

Added to `tests/vllm/v1/test_llm_engine.cpp` — the engine-level gate file whose
tiny synthetic Qwen3.5-MoE fixture and `Harness` are exactly the vehicle, and
`RequestOutput.prompt_logprobs` is an engine-level output. CPU reference backend
throughout. `Harness` gains an optional batched-token budget so a prompt can be
forced to chunk; it defaults to today's value, so every existing case is
unchanged.

1. **Payload shape** — `prompt_logprobs=k` on an N-token prompt yields
   `num_prompt_tokens` positions, position 0 `None`, positions `1..N-1` carrying
   `k+1` entries (or `k` when the target is already in the top-k), with the
   prompt token itself always present. 1:1 `logprobs.py:162-187`.
2. **Values** — prompt position `i` is scored against the model's distribution
   at position `i-1`, which is the distribution the SAMPLER sees when the same
   prompt is truncated to `[0, i)`. The test runs that truncated request and
   compares the two through the independent sampled-logprobs path, on the
   truncated run's greedy pick (rank 1 on both sides, and present in the
   all-vocab prompt row by construction).
3. **Chunked prefill** — the same prompt with a chunk size that splits it emits
   the identical tensor as the single-chunk run.
4. **Exact-prefill edge** — a chunk boundary landing exactly at
   `num_prompt_tokens - 1` produces no extra rows and still emits.
5. **Two concurrent requests** with different `k` get their own tensors.
6. **Inertness** — a request without `prompt_logprobs` yields
   `prompt_logprobs == nullopt` and byte-identical sampled tokens to current main.
7. **`-1`** widens to `vocab_size` columns (case 3 uses it).

Added by the 2026-08-10 review repair (PR #235), because 6 as written could not
deliver what it promised and 4 walked its edge in the only shape where the bug
is invisible:

8. **The route decision itself** — on a step where NO request asked, the forward
   must produce `step_num_logits()` rows, not one per token. Case 6 compares
   prompt-logprobs-on against -off inside one build, so forcing the full-logits
   route on every step moves both arms together and the file stays green (it
   did: 17/17, 346 assertions, under exactly that mutation). The assertion is
   therefore on the decision, read through `GPUModelRunner::last_forward_rows()`
   / `step_num_logits()` after each `engine.step()`, plus the mirror-image
   assertion that a step which DID ask takes the other route — otherwise the
   first is a tautology about `ForwardLogits`.
9. **The exact-prefill edge in a MIXED batch** — a zero-row final chunk beside a
   request that never asked and contributes more than one token. Case 4 walks
   the same edge with a single request, where the full-logits row count and the
   sampler row count coincide at 1 by accident, so it cannot see the assertion
   fire. RED-first: the step threw
   `collect_prompt_logprobs: a prompt-logprob step must carry full logits` out
   of `engine.step()` and killed the whole batch.

Existing gates that must stay green: `test_logprobs`, `test_serving`,
`test_llm_engine`, `test_scheduler`, `test_sampler`, `test_input_batch`.

## Gates

CPU reference backend. No GPU axis is claimed by this row: the feature is
correctness-only and upstream ships it as an explicitly unoptimized path
(`gpu_model_runner.py:5622-5623`). Exact invocation:

```sh
cmake -S . -B build-gate -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_METAL=OFF
cmake --build build-gate -j 18
./build-gate/tests/test_llm_engine
ctest --test-dir build-gate -j 6 --output-on-failure
```

`test_llm_engine` is the focused gate (RED captured before, GREEN after); the
`ctest` run is the full no-regression gate. A failure under `-j` is re-run
serially before it is called a regression:

```sh
ctest --test-dir build-gate -R '^test_engine_core_proc$' --output-on-failure
```

### `PENDING` — CUDA smoke gate (risk 4)

**Owed, not satisfied, not waived.** Everything above runs on the CPU reference
backend, where `vt::Tensor::Contiguous(fl.host.data(), …, queue_.device, …)` is
a host pointer labelled with the CPU device and the sampler is a host kernel —
so it cannot observe risk 4 at all. The row therefore has an open gate:

The vehicle is the server, not `vllm-cli` — no example binary exposes the
parameter (`grep -rn prompt_logprobs examples/` is empty), but
`protocol.cpp:522` maps the request field onto `SamplingParams`, so the runner
path runs even though W2's response serialization does not yet echo the payload
back:

```sh
# CUDA build, real checkpoint, DEFAULT flags (NOT VT_LOGITS_GATHER=0).
./build-cuda/examples/vllm-server --model <ckpt> &
# A: the asking request must not fault.
curl -s localhost:8000/v1/completions -d \
  '{"model":"m","prompt":"...","max_tokens":16,"temperature":0,"prompt_logprobs":5}'
# B: the same request WITHOUT the field, as the comparison arm.
curl -s localhost:8000/v1/completions -d \
  '{"model":"m","prompt":"...","max_tokens":16,"temperature":0}'
```

Pass = A does not fault AND A's completion is coherent and agrees with B. A
sampler reading a host pointer as device memory produces garbage, not a
near-tie, so coherence is the discriminating signal; the "Recorded deviation"
above (full-`T` vs gathered `lm_head` is a different GEMM shape) permits a
late-token divergence between A and B and does not permit incoherence.

Run it on **both** a unified-memory part (GB10, where the host pointer is
device-addressable and the path is expected to work) and a **discrete** part,
which is the one that has never been reasoned about, let alone executed. Until
both legs run, no CUDA claim is made for `prompt_logprobs`; `docs/USAGE.md` says
so in the same words. Blocked here on hardware: the box this row was implemented
and gated on has no GPU, and the GPU box is held by another claim.

## Evidence

Captured in the PR body: the RED run, the GREEN run, the full `ctest` summary,
and the reviewer's mutation results.

## Stop conditions

- Return `NEEDS_DECISION` if closing this needs the OpenAI `echo` serialization
  (it does not — that is W2 and its own row entry).
- Return `NEEDS_CONTEXT` if the tiny CPU fixture cannot express a chunked prefill
  (the scheduler's `max_num_batched_tokens` is settable per-harness, so it can).
- Never widen the gather list on a step where no request asked; if that becomes
  unavoidable, stop and re-spec.

## Outcome

*(W1, 2026-08-09. Row `PARTIAL` → `ACTIVE`; not `DONE` — the OpenAI `echo`
serialization is the named residual.)*

**Measured.** `tests/vllm/v1/test_llm_engine.cpp` §9, CPU reference backend,
clean Release build. RED first, against the unmodified engine with the new cases
in place: 4 of 6 failing, every prompt-logprob payload exactly 1 entry long — the
leading `None` `LogprobsProcessor::FromNewRequest` seeds, and nothing else. GREEN
after: `test_llm_engine` 17/17, 346 assertions.

**Review found a real bug (2026-08-10, PR #235, `VERDICT: FAIL`).** The
inertness guard and the full-logits assertion were keyed to DIFFERENT
conditions: the gather was decided on `prompt_logprob_indices.empty()` while
`collect_prompt_logprobs` asserted `fl.rows == num_actual_tokens` whenever
`prompt_logprob_rows` was non-empty. On the exact-prefill edge those disagree by
construction — `prepare_inputs` keeps a `final_chunk` entry with `num_rows == 0`
that contributes no gather index (`:5668-5673`) — so the step correctly kept the
gathered `lm_head` and the assertion fired anyway as soon as any OTHER request
in that step contributed more than one token. `VT_CHECK` throws, and the
exception escapes `engine.step()`, so it killed the whole batch rather than the
asking request. Fixed by moving the check to the slice it guards, inside
`if (r.num_rows > 0)`: a zero-row final chunk needs no logits at all. Keying the
gather on `prompt_logprob_rows` instead was rejected — it would force the slow
route onto a step that scores nothing. Regression: test 9 above, RED-first.
Post-repair on the pre-rebase tree: `test_llm_engine` 19/19, 360 assertions; on
the tree rebased onto `8a6704a2` (which brings main's own two `logprobs=-1`
cases into the same file): 21/21, 384 assertions.

**Rejected as insufficient: the on-vs-off inertness case.** Test 6 promised
"byte-identical sampled tokens to current main" and could not deliver it —
forcing the full-logits route on EVERY step left the file green at 17/17 · 346,
because both arms of an in-build comparison move together whenever the change is
to the SHARED route. Replaced by test 8, which asserts the decision through a
const observation seam on the runner (`last_forward_rows()` /
`step_num_logits()`); that same mutation now fails it.

**Open, not claimed: risk 4.** The full-logits route hands the sampler a host
pointer wearing the CUDA device label. Sound on unified memory, unreasoned on a
discrete GPU, and unexecutable here — no GPU on this box. Recorded as an owed
`PENDING` gate rather than asserted either way, and `docs/USAGE.md` says the
same. Risks 5-7 are latent divergences from upstream's lifetime model, written
down and deliberately not fixed in a review-repair change.

**Rejected: naming the prompt positions in `logits_indices`.** The obvious 1:1
of upstream's second `compute_logits` call, and the design this spec was
committed with. It does not work: every model's gather-before-`lm_head` is
guarded on `logits_indices.size() < T`, so a list that is not strictly shorter
than the token count is *silently ignored* — the forward returns all `T` rows in
token-stream order and the row-count assertion still passes, because `1 + (n-1)`
sampler-plus-prompt rows happens to equal `n`. The sampler then reads prompt
position 0 as its own row. Caught by the value cross-check, not by any shape
assertion, which is the argument for having written that case. Widening the
contract would mean editing 21 sites across 19 model files inside a
prompt-logprobs change; the full-logits path already existed, is what upstream
computes anyway, and touches no model.

**Why the value gate compares against the sampler.** A hand-computed
`log_softmax` reference would only re-derive the code under test. Truncating the
prompt to `[0, i)` puts the *same* distribution in front of the ordinary sampled
-logprobs path, so the two must agree — an independent route to the same number.
It is what caught the rejected design.

**Found and split out:** [#231](https://github.com/mudler/vllm.cpp/issues/231) —
`logprobs=-1` (the SAMPLED path) segfaults, because the sampler emits a
raw-vocab `LogprobsTensors` with empty ids/ranks that `UpdateSampleLogprobs`
indexes anyway. Pre-existing, unrelated to this row, its own issue rather than a
silent fix; the widening landed separately on `main` before this branch was
rebased onto it. The `-1` PROMPT path is unaffected: the runner widens to
`vocab_size` at admission, as upstream does, so it produces the ordinary shape.

**Default.** No flag. `prompt_logprobs` was already a request-level parameter
that validated and plumbed; this makes it do what it says. A step where nobody
asked is unchanged.

---

# W2 — the payload reaches a client

*(Live spec addition, 2026-08-23. Base `origin/main` `bacb71109`. Pin vLLM
0.26.0.dev0 `555967922`. Issue
[#1815](https://github.com/mudler/vllm.cpp/issues/1815), context
[#1775](https://github.com/mudler/vllm.cpp/issues/1775) /
[#821](https://github.com/mudler/vllm.cpp/issues/821). Owner claim
`CLAIM-SAMPLE-PROMPT-LOGPROBS-W2`. Row `SAMPLE-PROMPT-LOGPROBS`, `ACTIVE`.)*

## W2 scope

W1 made the runner compute prompt logits and filled
`RequestOutput.prompt_logprobs`. Nothing then read that field: neither
`CompletionResponseChoice` nor `ChatCompletionResponse` had a place to put it,
and neither serving path looked. `{"prompt_logprobs": 4}` therefore returned HTTP
200 with the field absent from the body — the #925 defect class, a key accepted
and silently ignored.

W2 closes exactly that hop, plus the three request validators upstream runs
before the value is read.

IN: the two response fields, `clamp_prompt_logprobs`, the three validators, the
two serving call sites, the upstream tests.

OUT, and deliberately: OpenAI `echo` (prepending the prompt text/tokens to the
emitted payload). It remains this row's named residual, and the row stays
`ACTIVE` rather than `DONE`.

## W2 upstream anchors

| what | upstream `file:line` @ `555967922` |
|---|---|
| `CompletionResponseChoice.prompt_logprobs` | `vllm/entrypoints/openai/completion/protocol.py:601` |
| `ChatCompletionResponse.prompt_logprobs` (TOP-LEVEL) | `vllm/entrypoints/openai/chat_completion/protocol.py:126` |
| `check_logprobs`, the shared prefix | `completion/protocol.py:474-494` == `chat_completion/protocol.py:763-783` |
| the completion-only suffix (`logprobs`, no -1) | `completion/protocol.py:495-499` |
| the chat-only suffix (`top_logprobs`, -1 allowed, needs the `logprobs` bool) | `chat_completion/protocol.py:784-796` |
| `clamp_prompt_logprobs` | `vllm/entrypoints/generate/base/serving.py:305-317` |
| completion emit (clamp at :520, choice at :588) | `vllm/entrypoints/openai/completion/serving.py:520,588` |
| chat emit | `vllm/entrypoints/openai/chat_completion/serving.py:1070` |
| `Logprob` wire shape `{logprob, rank, decoded_token}` | `vllm/logprobs.py:12-24,27` |

## W2 design

`PromptLogprobsToJson` in `src/vllm/entrypoints/openai/protocol.cpp` renders
`list[dict[int, Logprob] | None]` the way `model_dump()` does: an array whose
entries are `null` or an object keyed by the DECIMAL token id, each value
`{logprob, rank, decoded_token}` with explicit `null`s rather than omitted keys.
The key order is our `LogprobsOnePosition::order`, which is the Python dict
insertion order upstream serializes.

`ValidateLogprobsPrefix(j, count_field)` runs on the RAW body before any value is
read, mirroring the `mode="before"` model validator. It throws
`std::invalid_argument`, which `api_server.cpp` already maps to
`400 BadRequestError`, so upstream's `VLLMValidationError -> 400` shape is
preserved including the message text.

**The two validators share a prefix and then DIVERGE, and merging them is a
defect.** The first cut of this change had one shared function whose only
parameter was the count field's NAME, on the reading that the two endpoints
differ only there. They do not, and the check that caught it was reading
upstream rather than running a test:

| | completion `logprobs` | chat `top_logprobs` |
|---|---|---|
| `-1` | refused, `must be a positive value` (`:495-499`) | ALLOWED, the "every vocabulary entry" sentinel (`:784-790`) |
| other negative | same message | `must be a positive value or -1` |
| needs the `logprobs` flag | n/a (`logprobs` IS the count) | yes, `when using \`top_logprobs\`, \`logprobs\` must be set to true` (`:792-796`) |

The merged version would have taken `top_logprobs: -1` off the HTTP surface —
a capability this tree ships and gates (`test_serving.cpp`, "serving_chat:
top_logprobs=-1 returns every vocab entry per token", `logprobs-all-sentinel.md`
§Scope). **No existing test would have gone red**, because that case sets
`req.top_logprobs = -1` on the struct and never parses a body. So the shared
prefix takes the count field's name only for the integer-ness loop that genuinely
covers both, and each parser carries its own suffix inline, exactly as upstream
lays it out.

The refusals, each with upstream's exact wording:

| endpoint | body | message |
|---|---|---|
| both | a non-numeric `prompt_logprobs` or count | ``` `<field>` must be an integer.``` |
| both | `prompt_logprobs` with `stream` and value `> 0` or `== -1` | ``` `prompt_logprobs` are not available when `stream=True`.``` |
| both | `prompt_logprobs < 0` and `!= -1` | ``` `prompt_logprobs` must be a positive value or -1.``` |
| completions | `logprobs < 0`, `-1` included | ``` `logprobs` must be a positive value.``` |
| chat | `top_logprobs < 0` and `!= -1` | ``` `top_logprobs` must be a positive value or -1.``` |
| chat | `top_logprobs == -1` or `> 0` without `logprobs: true` | ``` when using `top_logprobs`, `logprobs` must be set to true.``` |

`prompt_logprobs: 0` WITH `stream` parses, because upstream's condition is
`> 0 or == -1` and not "is set"; `top_logprobs: 0` parses without the `logprobs`
flag for the same reason. A JSON bool passes the integer check, because `bool` is
an `int` subclass in Python and `isinstance(v, (int, float))` accepts it.

This closes the request-validation half of
[#249](https://github.com/mudler/vllm.cpp/issues/249) — the cap half already
landed at `src/vllm/v1/engine/input_processor.cpp:137` — and the
completion-surface divergence `logprobs-all-sentinel.md` records under `## Scope`:
we accepted `logprobs: -1` there and then emitted empty `top_logprobs` maps,
because `BuildCompletionLogProbs`'s `idx > -1` breaks on the first entry.

## W2 recorded deviations

- **`prompt_logprobs: -1` is served here, and refused upstream.** Upstream widens
  `-1` to the vocabulary size and then compares it against `--max-logprobs`,
  which defaults to 20 (`vllm/sampling_params.py:806-814`), so the whole-vocabulary
  request is a 400 there. Our `ModelConfig` carries no separate `max_logprobs`
  and the cap IS the vocabulary size
  (`src/vllm/v1/engine/input_processor.cpp:143`, upstream's own meaning for
  `max_logprobs == -1`). That is the pre-existing deviation of the cap, not of
  this row, and it is why the `-1` arm of
  `tests/entrypoints/openai/completion/test_completion.py:281-308` is ported as a
  PARSE assertion rather than as upstream's `BadRequestError`.
- **The n>1 arm of that same upstream test is not ported over the socket.**
  `AsyncLLM` never fans a request out into `n` children, so an `n>1` completion
  returns one choice there regardless of this row
  ([#1816](https://github.com/mudler/vllm.cpp/issues/1816)). Measured, not
  assumed: the identical body without `prompt_logprobs` also returns one choice.
  The per-choice property is gated instead over the SYNC `LLMEngine`, which does
  fan out.

## W2 tests ported

| upstream | ours |
|---|---|
| `tests/entrypoints/openai/completion/test_completion.py:78` (`prompt_logprobs is None` when unasked) | `test_api_server.cpp` "not requested → an explicit null, on both endpoints"; `test_serving.cpp` tail case |
| `test_completion.py:281-308` (`test_prompt_logprobs_completion`) | `test_api_server.cpp` "requested → a real per-position distribution" + `test_serving.cpp` "prompt_logprobs rides every n>1 choice" |
| `tests/entrypoints/openai/completion/test_completion_error.py:615-625` (`test_non_numeric_logprobs_rejected`, parametrized over both fields) | `test_protocol.cpp` "prompt_logprobs / logprobs request validation mirrors upstream" |
| `completion/protocol.py:483-499` validators (no dedicated upstream test) | the same case, plus the socket 400s |

The value assertions are the point. A test that only checks the array EXISTS
passes on zeros, and zeros are what a dropped payload looks like. Both e2e cases
therefore assert that the entries at a position are a subset of one `log_softmax`
distribution — `sum(exp(logprob)) <= 1` — which an array of zeros fails by
summing to the entry count, and that the `rank == 1` entry carries the position's
maximum.

## W2 evidence

RED, on the pinned base `bacb71109` with the implementation reverted and only the
socket test present (`test_api_server.cpp` compiles against the unmodified tree,
so this is a behavioural red rather than a build failure):

| subcase | red |
|---|---|
| not requested | `REQUIRE(j.at("choices").at(0).contains("prompt_logprobs"))` → `false` |
| requested | `json.exception.out_of_range.403 key 'prompt_logprobs' not found` |
| chat top-level | same, on the response object |
| the 400s | `stream=true` returned **200 and streamed SSE**; `-2` returned **200** |

GREEN, same tree with the change: `test_openai_protocol` 32/32 · 223,
`test_openai_serving` 43/43 · 615, `test_openai_logprobs` 7/7 · 40,
`test_openai_api_server` 76/76 · 1007. The five new cases carry 173 assertions
between them and none reports `assertions: 0`.

Reachability mutations, each with `compile_rc=0` printed and the deletion
confirmed by `git diff --stat`, each restored byte-for-byte against a pre-taken
`sha256sum`:

| # | deleted | result |
|---|---|---|
| M1 | `choice.prompt_logprobs = prompt_logprobs;` (`serving_completion.cpp`) | RED — api_server 14/15, serving 2/3 |
| M2 | `response.prompt_logprobs = final_res.prompt_logprobs;` (`serving_chat.cpp`) | RED — api_server 4/5 |
| M3 | `ValidateLogprobsPrefix(j, "logprobs");` | RED — protocol 17/22, api_server 6/11 |
| M4 | the `prompt_logprobs` line of `to_json(CompletionResponseChoice)` | RED — api_server 4/5, protocol 0/1 |
| M5 | `ValidateLogprobsPrefix(j, "top_logprobs");` | RED — protocol 12/16 |
| M6 | `ClampPromptLogprobs(prompt_logprobs);` | **GREEN — not caught**, see `## Owed` |
| M7 | the whole chat-only suffix | RED — protocol 19/22 |
| M8 | the chat suffix REPLACED by the completion rule (the first cut's defect) | RED — and it reports `assertions: 11 \| 11 passed` with `Status: FAILURE!`, because the parse THROWS where the case expects a value. Grepping only `assertions:` would have read this mutation as a pass |

M8 is the reason the split exists, gated rather than argued.

## Owed

- [#1816](https://github.com/mudler/vllm.cpp/issues/1816) — `AsyncLLM` never fans
  out `n>1`, so every `n > 1` request to the production server is silently served
  as `n = 1`. Found here, not fixed here: it needs its own row, spec and fresh
  review rather than an in-flow repair.
- [#1817](https://github.com/mudler/vllm.cpp/issues/1817) — the
  `ClampPromptLogprobs` call sites are reached but not measured (mutation M6 stays
  green). The function itself is gated directly; no CPU fixture here can produce
  the `-inf` the clamp exists for.
- OpenAI `echo` + prompt_logprobs serialization stays this row's named residual,
  tracked by [#223](https://github.com/mudler/vllm.cpp/issues/223)'s row entry in
  `.agents/engine-matrix.md`.

## W2 outcome

**What this unblocks.** [#1775](https://github.com/mudler/vllm.cpp/issues/1775)
records its first blocker as "no production path exposes a logit vector", so our
distribution and pinned llama.cpp's cannot be diffed. They can now: a
`POST /v1/completions` with `{"prompt": <the prompt>, "max_tokens": 0 or 1,
"prompt_logprobs": -1}` returns, for every prompt position, the full-vocabulary
distribution the model assigned to the NEXT id — which is exactly the
teacher-forced comparison the oracle side of #1775 already ran, and it comes out
of the production server rather than a probe. Feed the divergent prompt's own ids
back as the prompt to put both engines on the same trajectory.

**Not claimed.** This ships the instrument, not the measurement. No #1775 diff
was run here — that needs the 27B Q4_K_M checkpoint and a GPU, and the fleet was
held by two other sessions.

**Why a shared validator rather than two.** The two upstream before-validators
differ only in the count field's name; the refusal set, the messages and the
order are identical. One function with the field name as a NAMED argument makes
the difference visible at both call sites instead of duplicating 40 lines that
would drift.
