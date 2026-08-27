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

- `/v1/chat/completions` STREAMING with `n > 1`. Upstream keeps per-choice
  `previous_num_tokens`, `previous_text`, role frames and a SEPARATE tool /
  reasoning parser instance per choice index
  (`chat_completion/serving.py:404-802`). Our `ChatSseStream` holds one of each
  as a scalar member. Making that per-index is a parser-lifetime change with its
  own review surface, not a repair of this fan-out. Owed below,
  [#2120](https://github.com/mudler/vllm.cpp/issues/2120).
- `AsyncLLM::add_request_wave`. It is a local extension with no upstream
  counterpart and no production caller — `examples/bench/bench_core.h` is its
  only user. Owed below, [#2121](https://github.com/mudler/vllm.cpp/issues/2121).
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
| `tests/entrypoints/openai/completion/test_completion.py:633` `n=2` batch arm | `test_api_server.cpp` — `best_of` > `n` returns exactly `n` ranked choices |
| — (abort, ours) | `test_api_server.cpp` — a client disconnect on an `n>1` stream leaves NO unfinished request |

Every case enters through `ApiServer`'s registered `/v1/completions` route over
`httplib`, which is the production entry point. Deleting the fan-out call site
in `AsyncLLM::add_request` makes all of them red.

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

## Stop conditions

Stop and report `NEEDS_DECISION` if the fix requires per-index parser lifetimes
in `ChatSseStream`, or any change to `LLMEngine`, the scheduler or the sampler.

## Now

`ACTIVE` — implementation in `row/async-llm-n-fanout`.
