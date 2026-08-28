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
7. `create_completion` REFUSES `best_of > n` together with `stream: true`, with
   a 400 that names the reason. See below; this is the one place the streaming
   and non-streaming arms deliberately disagree about whether a body is valid.

### Why streaming `best_of > n` is refused rather than trimmed

The fan-out exposed a defect the row would otherwise have shipped: `best_of`
sets `sp.n = best_of` (`protocol.cpp:318-320`) and `SelectBestOf` runs only in
the non-streaming arm (`serving_completion.cpp`), so a streaming
`{"n":2,"best_of":4}` began answering FOUR choices where the same body without
`stream` answers two. Before this row it answered one, because nothing fanned
out. Under-delivery became over-delivery.

Trimming to `n` in the stream is the obvious repair and it does not exist. The
rank is by FINAL cumulative logprob, which a delta stream does not have until
each candidate's last token, and a frame already sent cannot be recalled. The
three ways to pretend otherwise are each worse than refusing:

- Buffer every frame to the end and emit the top `n`. That turns a streaming
  request into a blocking one whose first byte arrives last, with no signal to
  the client. The request succeeds and the latency contract silently does not.
- Stream the first `n` children by index. That returns `n` choices which are
  NOT the `n` the non-streaming arm returns, labelled as if they were the best.
  The client cannot tell.
- Rank on running cumulative logprob. The leader changes as tokens arrive, so
  frames already sent belong to candidates that later lose.

So the two arms cannot agree on CONTENT. They can only agree visibly or
disagree invisibly. A 400 is a disagreement the caller can see and act on; a
wrong choice set is one nobody can detect. That is the trade, and it is
deliberate: a user whose body works without `stream` and 400s with it is being
told the ranking cannot be computed, not that the parameter is unknown.

Upstream cannot settle this — `best_of` is not a field on its
`CompletionRequest` at the pin (0.26 dropped it; `protocol.h:201-211` declares
ours a local extension implementing the classic OpenAI / vLLM-V0 contract). But
upstream carries exactly ONE other ranked-selection mode, and refuses it with
`stream` for exactly this reason and at exactly this point in the handler:
`if request.stream and request.use_beam_search` (`completion/serving.py:136-139`).
Mirroring that refusal is closer to the mirror rule than inventing a third
semantics vLLM never had.

NARROW: only `best_of > n` is refused. `best_of` unset or `== n` has nothing to
rank and streams `n` children exactly as a plain `n > 1` request does.

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
| — (`best_of` is OURS; upstream has no such field and `grep -rn best_of tests/entrypoints/` at the pin returns nothing) | `test_api_server.cpp` — `best_of > n` returns exactly `n` ranked choices non-streaming, and is REFUSED with 400 when `stream:true` |
| — (abort, ours) | `test_api_server.cpp` — a client disconnect on an `n>1` stream leaves NO unfinished request, asserted SYNCHRONOUSLY after `abort()` so natural drain cannot pass it |
| — (call-site coverage, ours) | `test_async_llm.cpp` — the tokens and multimodal `add_request` overloads fan out too; no OpenAI route reaches either, so the socket gate cannot see them |
| — (chat, ours) | `test_api_server.cpp` — `/v1/chat/completions` NON-streaming `n>1` returns `n` indexed choices over a socket. The chat STREAMING arm is the regression owed to #2120 |

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
| force `request_index=0` on every child | `test_openai_api_server` RC=135, `test_async_llm` RC=139 — the shared slot corrupts memory |
| delete the external->internal resolution in `abort_requests` | `test_openai_api_server` RC=1, `CHECK( 3 == 0 )` x3 |
| delete `state.external_req_id = parent_req->external_req_id()` | `test_openai_api_server` RC=1, `CHECK( 3 == 0 )` x3 |

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
- [#2120](https://github.com/mudler/vllm.cpp/issues/2120) — `/v1/chat/completions`
  streaming still collapses `n > 1` onto one choice's parser and text state.
- [#2121](https://github.com/mudler/vllm.cpp/issues/2121) —
  `AsyncLLM::add_request_wave` does not fan out `n > 1`.
- [#2145](https://github.com/mudler/vllm.cpp/issues/2145) — the fan-out
  DEEP-copies the prompt `n` times. `EngineCoreRequest::prompt_token_ids` is a
  `std::vector<int32_t>` by value (`types.h:79`), so `EngineCoreRequest child =
  request` copies the whole prompt per child and `FromEngineCoreRequest` copies
  it again — `O(n * prompt_len)` where upstream's `copy(request)`
  (`async_llm.py:393`) is SHALLOW and copies none. The comment on both fan-out
  sites claimed the copy "shares the prompt token ids"; this row corrects the
  comment in `async_llm.cpp` AND in the `llm_engine.cpp` line it was inherited
  from, and does NOT change the copy, because the cheap mirror is a shared
  immutable token buffer on `EngineCoreRequest` that every engine path reads.

## Stop conditions

Stop and report `NEEDS_DECISION` if the fix requires per-index parser lifetimes
in `ChatSseStream`, or any change to `LLMEngine`, the scheduler or the sampler.

## Now

`ACTIVE` — implementation in `row/sample-n-async`, pull request
[#2139](https://github.com/mudler/vllm.cpp/pull/2139).
