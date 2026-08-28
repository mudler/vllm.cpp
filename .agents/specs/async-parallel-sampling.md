# `SAMPLE-N-ASYNC` — parallel-sampling fan-out on the SERVED engine

Issue: [#1816](https://github.com/mudler/vllm.cpp/issues/1816).
Owning matrix rows: [`SAMPLE-N`](../feature-matrix.md) and
[`SAMPLE-BEST-OF`](../feature-matrix.md), whose `Residuals` both name
"async-streaming per-child collation" — this row is that residual.

## Scope

`AsyncLLM` is the engine every OpenAI HTTP route runs on. It never fans a
request out into `n` children, so every `n > 1` request to the production
server is silently answered with ONE choice. `LLMEngine` — which no HTTP route
uses — does fan out, and `tests/vllm/entrypoints/openai/test_serving.cpp`
gates the property there. The covered engine is not the served one.

IN scope:

- The parallel-sampling fan-out on all three `AsyncLLM::add_request` overloads
  (string, token-ids, multimodal), sharing ONE `RequestOutputCollector` across
  the `n` children, as `async_llm.py:_add_request` does.
- The abort path: the id `add_request` hands back is now the PARENT id, and
  aborting it must abort every child in the OutputProcessor and in EngineCore.
- The `/v1/completions` streaming SSE source, which reads
  `response.outputs.front()` and therefore DROPS every other child's delta the
  moment a frame carries more than one. This is a pre-existing serving defect
  that only the fan-out can expose; it is small and clear, so it is repaired in
  flow ([#2119](https://github.com/mudler/vllm.cpp/issues/2119)).

OUT of scope, and why:

- `/v1/chat/completions` STREAMING with `n > 1`, which this row makes WORSE and
  defers as a REGRESSION rather than leaving alone. Upstream keeps per-choice
  `previous_num_tokens`, `previous_text`, role frames and a SEPARATE tool /
  reasoning parser instance per choice index
  (`chat_completion/serving.py:404-802`). Our `ChatSseStream` holds one of each
  as a scalar member and reads `response.outputs.front()`
  (`serving_chat.cpp:443`). Before the fan-out that route ran ONE sequence and
  streamed its one choice; after it the engine runs `n` children and `n-1` of
  them are generated and then discarded. Measured over a socket on this branch:
  `{"n":3,"stream":true}` returns 200 with 4 frames carrying index `{0}` only,
  while the same body without `stream` returns 3 choices and
  `completion_tokens=9` — 3x the decode compute for one visible choice, plus the
  KV blocks the discarded children hold. `finish_reason` is also set only under
  `if (response.finished)` (`:485-488`), true for `n>1` only after every child
  finishes, so three requested choices yield ONE finish reason at the end.
  Making the state per-index is a parser-lifetime change with its own review
  surface, not a repair of this fan-out. Owed below,
  [#2120](https://github.com/mudler/vllm.cpp/issues/2120). The chat
  NON-streaming `n > 1` arm is IN scope: it became live with the fan-out and had
  no test on any route, so this row gates it over the socket.
- `AsyncLLM::add_request_wave`, BOTH overloads. It is a local extension with no
  upstream counterpart and no production caller — `examples/bench/bench_core.h`
  is its only user. Owed below,
  [#2121](https://github.com/mudler/vllm.cpp/issues/2121). That issue's index
  row cites `async_llm.cpp:133,166`, which this pull request staled by inserting
  the fan-out call sites above them; the overloads are at `:147` and `:180` at
  this head. The index is append-only, so the correction lives here and in a
  comment on the issue, and the durable anchor is the symbol, not the line.
- `LLMEngine`, the scheduler, and the sampler. Untouched.

## Upstream chain (pin `5559679229`, [upstream-sync.md](../upstream-sync.md))

| Behaviour | Upstream `file:line` |
|---|---|
| The fan-out itself | `vllm/v1/engine/async_llm.py:386-399` |
| One collector for the whole parent | `vllm/v1/engine/async_llm.py:377` |
| `n == 1` short-circuit (byte-identical path) | `vllm/v1/engine/async_llm.py:382-384` |
| Child = copy of the parent, only `request_id` + `sampling_params` replaced | `vllm/v1/engine/async_llm.py:392-395` |
| Child registration: OutputProcessor BEFORE EngineCore | `vllm/v1/engine/async_llm.py:401-414` |
| Children share the parent's `external_req_id` | `vllm/v1/engine/output_processor.py:551` (the child's `external_req_id` is untouched by the fan-out above) |
| Abort by external id fans to every internal child | `vllm/v1/engine/output_processor.py:487-489` |
| Streaming emits ONE choice per SSE chunk | `tests/entrypoints/openai/completion/test_completion.py:419-423` |
| `best_of != n` + `stream` DOWNGRADES to non-streaming | `vllm/entrypoints/openai/serving_completion.py:253-260` **@ `56e96b37e4^`** — not at the pin; see below |
| A downgraded request still answers on the SSE transport | `vllm/entrypoints/openai/completion/serving.py:269-278` (present at the pin) |
| The chat route does NOT downgrade, though it carries `best_of` | `vllm/entrypoints/openai/serving_chat.py:355` + `protocol.py:568` **@ `56e96b37e4^`** |

**Why one anchor is off-pin, and why that is not a licence to invent.**
`best_of` does not exist at `5559679229`, so the pin cannot state its stream
rule. `56e96b37e4` ("[V0 Deprecation] Remove `best_of`", vllm#29090,
2025-11-21) removed the field; its PARENT is the last revision that defines the
behaviour, and it reads:

```python
# Similar to the OpenAI API, when n != best_of, we do not stream the
# results. Noting that best_of is only supported in V0. In addition,
# we do not stream the results when use beam search.
stream = (request.stream
          and (request.best_of is None or request.n == request.best_of)
          and not request.use_beam_search)
```

`protocol.h:203-210` declares our `best_of` an implementation of exactly that
classic OpenAI / vLLM-V0 contract, so this IS the contract's stream behaviour
and not a local design choice. The commit that removed the field deleted only
the `best_of` clause and KEPT beam search as a downgrade, so the two ranked
modes were treated identically. The 400 at `completion/serving.py:136-139`
is NOT precedent for refusing `best_of`: `65a4da1504` (vllm#36160) introduced
it on 2026-03-08, almost four months after `best_of` was gone. Beam search
keeps that 400 here because the PIN is what we mirror and the pin refuses it;
`best_of` gets the downgrade because that is the only rule upstream ever
applied to it.

**The route asymmetry is UPSTREAM'S, and is deliberately preserved.**
`/v1/chat/completions` accepts `best_of` here and applies the same
`SelectBestOf` ranking non-streamed (`serving_chat.cpp:1015-1037`, where the
absence of a downgrade is now annotated in the code too), but it does
NOT downgrade. That mirrors upstream: at `56e96b37e4^`, `ChatCompletionRequest`
carries `best_of` (`protocol.py:568`, class at `:528`) and passes it into
`SamplingParams` (`:892`), yet `serving_chat.py:355` streams on the RAW
`request.stream` with no `best_of` or beam-search guard at all. Only the
completions route ever downgraded. Adding a downgrade to chat would be
inventing a rule upstream declined to write, so the asymmetry is named here
instead. CONSEQUENCE, stated rather than discovered later: while
[#2120](https://github.com/mudler/vllm.cpp/issues/2120) is open the chat
streaming arm collapses `n > 1` onto one choice, and when #2120 is fixed that
arm will stream all `best_of` children rather than the ranked top-`n`. That is
upstream's shape, and it is #2120's business to reconcile, not this row's.

The child naming, `n == 1` child params, `seed + idx`, and the FINAL_ONLY vs
streaming aggregation are already ported — `ParentRequest`
(`src/vllm/v1/engine/parallel_sampling.cpp`, 1:1 `parallel_sampling.py`) and
`OutputProcessor::add_request`'s `parent_req` argument both exist and are used
by `LLMEngine::FanOutParallelSampling`. Nothing in this row invents semantics;
it routes the served engine through machinery that is already here.

## Design

1. `AsyncLLM::PublishParallelSampling(request, prompt, collector)` — one private
   helper, called by each of the three `add_request` overloads under
   `if (request.sampling_params.n > 1)`, placed exactly where
   `LLMEngine::add_request` places its `FanOutParallelSampling` call so the
   `n == 1` path stays byte-identical.
2. The helper builds the shared `ParentRequest`, then the `n` children
   (`{idx}_{parent}`, `n == 1` params, `seed + idx` — all from
   `ParentRequest::get_child_info`), registers every child with the
   OutputProcessor against the ONE collector and the shared parent, and then
   enqueues the `n` core requests in a single `add_requests_async` call.
   The batch enqueue is a deliberate deviation from upstream's per-child
   `await`: a mid-loop enqueue failure would otherwise leave earlier children
   running in EngineCore with no frontend state, which `process_outputs`
   silently ignores forever. `rollback_requests` undoes the whole admission.
   Before the first registration it rejects every colliding child id, the same
   guard `PublishPreparedWave` makes: `add_request` throws on a duplicate live
   id and the rollback erases BY id, so without the pre-check a collision would
   put the pre-existing request into the rollback set and destroy state its own
   caller still holds. Unconstructible over HTTP today — the serving layer mints
   the parent id — but the fan-out is also an ABI seam where the caller names
   it, so the asymmetry is closed rather than recorded.
3. It returns `AsyncRequest{parent_id, collector}` — the PARENT id, which is
   what the client aborts and what `RequestOutput::request_id` already carries
   (`RequestState::make_request_output` reads `parent_req->external_req_id()`).
4. `OutputProcessor::add_request` gives every child the PARENT's
   `external_req_id`, mirroring upstream where the fan-out replaces only
   `request_id`. That makes `external_req_ids_[parent] == {0_p, 1_p, ...}`.
5. `OutputProcessor::abort_requests` resolves each incoming id through
   `external_req_ids_` first, so aborting the parent aborts every child and
   returns every child's core id. For a single-sequence request the map holds
   `{id: [id]}`, so the resolution is an identity and the path is unchanged.
6. `CompletionSseStream::next` emits ONE chunk per `CompletionOutput` in the
   frame instead of `outputs.front()` only, buffering the rest in a pending
   deque, and counts generated tokens per choice index for the hold-back check
   while still SUMMING them for `usage`.
7. `create_completion` DOWNGRADES `best_of != n` together with `stream: true`:
   it serves the request non-streamed, ranks with `SelectBestOf`, and returns
   the aggregated body wrapped in ONE SSE frame plus `[DONE]`. That is
   upstream's rule, `serving_completion.py:253-260 @ 56e96b37e4^` plus the
   wrapper at `completion/serving.py:269-278`. See below for why trimming
   inside the stream is not available, and why a 400 was the wrong answer.
   Beam search keeps the pin's 400 (`completion/serving.py:136-139`), because
   the pin is what we mirror and the pin refuses beam search.
8. ADAPTATION, annotated. Upstream derives `output_kind` from the RAW
   `request.stream` — `CompletionRequest.to_sampling_params` sets it at
   `protocol.py:1401-1403 @ 56e96b37e4^` (class at `:1086`, method at `:1304`;
   the chat twin is `:915-917`, class `:528`) — so a downgraded request still
   asks EngineCore for DELTA outputs and upstream aggregates them itself. Our
   non-streaming arm reads the FINAL `RequestOutput` instead, so inheriting the
   raw binding would deliver only the last delta's text, and the downgrade's
   whole point is that the client gets the SAME body the non-streamed request
   returns. `create_completion` therefore binds `output_kind` to the EFFECTIVE
   stream (`sampling_params.output_kind = kFinalOnly` when
   `request.stream && !stream_results`, `serving_completion.cpp`). This changes
   no behaviour upstream's own aggregator relies on: it is a consequence of
   WHERE the aggregation happens, not of what the response contains. The
   `best_of == n` and no-`best_of` streaming paths are untouched.

### Why the downgrade, and not a trim inside the stream

The fan-out exposed a defect the row would otherwise have shipped: `best_of`
sets `sp.n = best_of` (`protocol.cpp:318-320`) and `SelectBestOf` runs only in
the non-streaming arm (`serving_completion.cpp`), so a streaming
`{"n":2,"best_of":4}` began answering FOUR choices where the same body without
`stream` answers two. Before this row it answered one, because nothing fanned
out. Under-delivery became over-delivery.

Trimming to `n` INSIDE the stream is the obvious repair and it does not exist.
The rank is by FINAL cumulative logprob, which a delta stream does not have
until each candidate's last token, and a frame already sent cannot be recalled.
The three ways to pretend otherwise are each worse:

- Buffer every frame to the end and emit the top `n`. That turns a streaming
  request into a blocking one whose first byte arrives last, with no signal to
  the client. The request succeeds and the latency contract silently does not.
- Stream the first `n` children by index. That returns `n` choices which are
  NOT the `n` the non-streaming arm returns, labelled as if they were the best.
  The client cannot tell.
- Rank on running cumulative logprob. The leader changes as tokens arrive, so
  frames already sent belong to candidates that later lose.

So the two arms cannot agree on per-frame CONTENT. Upstream settles it by
agreeing on the BODY and giving up the frames: `stream = (request.stream and
(request.best_of is None or request.n == request.best_of) and not
request.use_beam_search)` (`serving_completion.py:253-260 @ 56e96b37e4^`), with
`completion/serving.py:269-278` wrapping the aggregated response in one SSE
frame plus `[DONE]` whenever the client asked to stream and the server did not.
The client gets exactly the body it would have got without `stream`, on the
transport it asked for.

An earlier head of this row answered 400 instead, on the reasoning that
upstream could not settle the question because `best_of` is not a field on its
`CompletionRequest` at the pin (0.26 dropped it; `protocol.h:203-210` declares
ours a local extension implementing the classic OpenAI / vLLM-V0 contract).
That reasoning was wrong: the last revision that DEFINES `best_of` is
`56e96b37e4^`, and it downgrades. The 400 at `completion/serving.py:136-139` is
not precedent either — `65a4da1504` (vllm#36160, 2026-03-08) added it almost
four months after the field was gone, so at the one revision where both ranked
modes coexisted upstream downgraded both and refused neither. The 400 also had
a measured blast radius: `{"n":1,"best_of":4,"stream":true}`, the canonical
OpenAI `best_of` call, answered 400 at that head and 200 before this row.

NARROW: only `best_of != n` downgrades. `best_of` unset or `== n` has nothing
to rank and streams `n` children exactly as a plain `n > 1` request does.

## Risks

- **A shared collector under `n` producers.** `RequestOutputCollector::Merge`
  already keys by `CompletionOutput::index` and keeps distinct indices
  independent; the `n>1` cumulative merge has a direct unit test. The fan-out
  makes that path live rather than hypothetical.
- **Abort semantics widening.** Resolving through `external_req_ids_` changes
  `abort_requests` for every caller. It is an identity for `n == 1` because the
  map is 1:1 there, and the shutdown sweep passes internal child ids which are
  not map keys and fall through unchanged.
- **`generate()` termination.** Under FINAL_ONLY the parent suppresses output
  until the last child finishes, so the `while (!output.finished)` driver sees
  exactly one terminal RequestOutput carrying `n` outputs.

## Tests to port

| Upstream | Here |
|---|---|
| `tests/entrypoints/openai/completion/test_completion.py:349` `test_parallel_no_streaming` | `test_api_server.cpp` — `n=3` over a REAL SOCKET: 3 choices, `index == idx`, every `finish_reason` set, usage sums the children |
| `tests/entrypoints/openai/completion/test_completion.py:398` `test_parallel_streaming` | `test_api_server.cpp` — `n=3` `stream:true` over a real socket: exactly one choice per chunk, `n` finish reasons, indices `0..n-1` |
| — (`best_of` is OURS at the PIN; upstream has no such field there and `grep -rn best_of tests/entrypoints/` at the pin returns nothing. The stream rule is NOT ours: `serving_completion.py:253-260 @ 56e96b37e4^`) | `test_api_server.cpp` — `best_of > n` returns exactly `n` ranked choices non-streaming; with `stream:true` it DOWNGRADES to one `text/event-stream` SSE frame (Content-Type asserted) carrying the whole ranked response (`n=2,best_of=4` → 2 choices; `n=1,best_of=4` → 1 choice), while `best_of == n` still streams deltas |
| — (abort, ours) | `test_api_server.cpp` — a client disconnect on an `n>1` stream leaves NO unfinished request, asserted SYNCHRONOUSLY after `abort()` so natural drain cannot pass it |
| — (call-site coverage, ours) | `test_async_llm.cpp` — the tokens and multimodal `add_request` overloads fan out too; no OpenAI route reaches either, so the socket gate cannot see them |
| — (chat, ours) | `test_api_server.cpp` — `/v1/chat/completions` NON-streaming `n>1` returns `n` indexed choices over a socket. The chat STREAMING arm is the regression owed to #2120 |
| — (chat `best_of`, ours) | `test_api_server.cpp` — `/v1/chat/completions` `n=2,best_of=4` non-streamed returns exactly 2 ranked choices. Added by the repair round: a mutation showed the chat `SelectBestOf` trim was reachable by no test at all |
| `serving_chat.py:355` + `protocol.py:568` @ `56e96b37e4^` (chat carries `best_of` and streams anyway) | `test_api_server.cpp` — `/v1/chat/completions` `n=2,best_of=4,stream:true` STREAMS `chat.completion.chunk` deltas and does NOT downgrade, pinning the upstream-sourced asymmetry with `/v1/completions`. Asserts only that it streamed, so it stays true once #2120 is fixed |

Every `/v1/completions` case enters through `ApiServer`'s registered route over
`httplib`, and the disconnect case enters through `ApiServer::handle_completions`,
which is the body that route calls (`api_server.cpp:1172`). Both are production
entry points. The `test_async_llm.cpp` cases enter through `AsyncLLM` itself,
which is the ABI seam `include/vllm.h` and the multimodal serving path use, and
which no OpenAI route reaches for those two overloads.

Measured, not asserted — each mutation was applied, rebuilt (`BUILD_RC=0`) and
run, then the tree restored by byte-copy and verified by sha256:

| Mutation | Result |
|---|---|
| delete the fan-out call site in the STRING overload | `test_openai_api_server` 4 failed / RC=1 (`REQUIRE( 1 == 2 )`, `REQUIRE( 1 == 3 )` x3) |
| delete it in the TOKENS overload | `test_async_llm` RC=1, `REQUIRE( 1 == 3 )` |
| delete it in the MULTIMODAL overload | `test_async_llm` RC=1, `REQUIRE( 1 == 3 )` |
| force `request_index=0` on every child | NEVER RC=0 — the shared slot is undefined behaviour, so the exit STATUS is not a stable datum. Observed on the full binary: `test_openai_api_server` RC=135 (SIGBUS) and `test_async_llm` RC=139 (SIGSEGV) in this row's run; a later re-review measured RC=139 full and RC=1 case-scoped. Recorded as a SIGNAL (the process dies or the assertion fails; it never passes), because a UB exit code re-measured is a different number and pinning one would gate on noise |
| delete the external->internal resolution in `abort_requests` | `test_openai_api_server` RC=1, 1 case / 3 assertions failed |
| delete `state.external_req_id = parent_req->external_req_id()` | `test_openai_api_server` RC=1, 1 case / 3 assertions failed |
| replace `stream_results` with the raw `request.stream` (kill the `best_of != n` downgrade) | `test_openai_api_server` RC=1, 1 case / 1 assertion failed |
| delete the single-frame wrapper on the downgraded response | `test_openai_api_server` RC=1, 1 case / 2 assertions failed |
| force `trim_best_of = false` on the CHAT route | `test_openai_api_server` RC=1, 1 case / 1 assertion failed — **this one SURVIVED at first**: before this repair added the chat `best_of` socket case, deleting the chat ranking left every suite green, so the chat arm of SAMPLE-BEST-OF was landing UNGATED. Found by re-running the mutation set, not by reading |

The last two are why the disconnect case asserts synchronously. Both mutations
leave the `n` children running to `max_tokens`, so a poll loop would watch them
retire naturally and report success. The first version of the abort case did
exactly that: it posted a plain NON-streaming request, aborted nothing, and
polled to zero. It survived both mutations at RC=0.

## Gates

```sh
ctest --test-dir build -R "openai_api_server|async_llm|output_processor|serving" --output-on-failure
scripts/agent-preflight.sh --fail-on-skip
```

## Owed

- [#2119](https://github.com/mudler/vllm.cpp/issues/2119) — `/v1/completions`
  streaming dropped every `CompletionOutput` past the first. FIXED in this row's
  pull request; the issue exists because it is a distinct defect from #1816.
- [#2120](https://github.com/mudler/vllm.cpp/issues/2120) — a REGRESSION this
  row causes, not a pre-existing gap it leaves alone. `/v1/chat/completions`
  streaming collapses `n > 1` onto one choice's parser and text state. BEFORE
  the fan-out that route ran ONE sequence and streamed its one choice; AFTER
  it the engine runs `n` children, `n-1` are generated and then DISCARDED, and
  the client still sees one choice — so the row buys `n`x the decode compute
  and the KV blocks for no visible output. Deferred because per-index parser
  lifetimes are a separate review surface (see OUT of scope).
- [#2121](https://github.com/mudler/vllm.cpp/issues/2121) —
  `AsyncLLM::add_request_wave` does not fan out `n > 1`.
- [#2145](https://github.com/mudler/vllm.cpp/issues/2145) — the fan-out
  DEEP-copies the prompt `n` times. `EngineCoreRequest::prompt_token_ids` is a
  `std::vector<int32_t>` by value (`types.h:79`), so `EngineCoreRequest child =
  request` copies the whole prompt per child and `FromEngineCoreRequest` copies
  it again — `O(n * prompt_len)` where upstream's `copy(request)`
  (`async_llm.py:393`, `llm_engine.py:283`) is SHALLOW and copies none. FIVE
  comments claimed the copy "shares" the prompt or the mm inputs; a sweep for
  the wording found and corrected all five — `async_llm.cpp` at the string
  overload's entry, at the mm overload's entry and at the fan-out copy site,
  and `llm_engine.cpp` at its mm overload's entry and in
  `FanOutParallelSampling` — and the copy itself is NOT changed, because the cheap mirror is a shared immutable token
  buffer on `EngineCoreRequest` that every engine path reads.

  DECIDED, on the multimodal path: #2145 is NOT widened, and no second issue is
  filed. The mm overload copies the same `EngineCoreRequest`, so its
  `prompt_token_ids` copy is already #2145 — and it is #2145's WORST case, not
  a new one, because an mm prompt is the placeholder-EXPANDED token list. The
  mm-only residual is the `mm_features` vector itself
  (`std::vector<MultiModalFeatureSpec>` by value, `types.h:93`), and that copy
  does NOT duplicate the encoder payload: `MultiModalFeatureSpec` holds it
  behind `std::shared_ptr<ImageKwargs>` / `<AudioKwargs>`
  (`multimodal/inputs.h:80-81`), so a child copy bumps a refcount. What is
  duplicated per child is the spec vector and two short strings per spec —
  bounded by the placeholder COUNT, not by pixels. One shared token buffer on
  `EngineCoreRequest` closes the mm path with the text path; a separate issue
  for the residual would be filing noise.

- [#2150](https://github.com/mudler/vllm.cpp/issues/2150) —
  `ParentRequest::get_outputs` indexes `output_aggregator_` with
  `completion_output.index` unchecked and drains it through `*slot` on a
  possibly-empty optional (`parallel_sampling.cpp:75-82`). A faithful 1:1 port
  of `parallel_sampling.py:100-126`, which is equally unchecked — but upstream's
  list raises `IndexError` and yields a visible `None`, while our
  `std::vector<std::optional<...>>` gives UB on both. NOT fixed here and NO
  guard added: `idx` is `0..n-1` by construction and the vector is sized `n`, so
  a check would be dead code, which this repository does not land. Filed because
  it is the reason the `request_index=0` mutation's exit status is unstable, and
  because the right fix is probably a debug-configuration assertion rather than
  a release-path branch.

## Stop conditions

Stop and report `NEEDS_DECISION` if the fix requires per-index parser lifetimes
in `ChatSseStream`, or any change to `LLMEngine`, the scheduler or the sampler.

## Now

`ACTIVE` — implementation in `row/sample-n-async`, pull request
[#2146](https://github.com/mudler/vllm.cpp/pull/2146) on
`row/sample-n-async-trailer-repair`. (#2139 was the earlier pull request for
this row and is CLOSED.)
