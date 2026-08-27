# Chat SSE: the role frame waits for the first engine result

**Row:** `SERVE-STREAM-USAGE` (the row that owns `ChatSseStream`) ·
**Issue:** [#1982](https://github.com/mudler/vllm.cpp/issues/1982) ·
**Kind:** bug fix in flow.

## Now

`SERVE-STREAM-USAGE` keeps its recorded state. This change repairs one
divergence inside the row's existing surface. It adds no capability and moves
no lifecycle state.

## Scope

`ChatSseStream::next` emits the `/v1/chat/completions` role frame before it
reads anything from the engine. Make the default path buffer the first engine
result, exactly as the continuous-usage path already does, so that:

1. the role frame reaches the client only after the first engine output exists;
2. a request that fails before its first token raises at the stream seam
   before any frame is written.

Out of scope, and named so that the boundary is visible:

- `/v1/completions`. `src/vllm/entrypoints/openai/serving_completion.cpp::CompletionSseStream`
  (the hold-back in its `next`, at `:94-99`) already withholds
  the empty chunked-prefill delta, mirroring
  `vllm/entrypoints/openai/completion/serving.py:368-374`. It is not touched.
- The wire shape of the role frame. Same `delta.role`, same empty
  `delta.content`, same position ahead of every content frame, same
  continuous-usage attachment. Only its arrival time changes.
- The sync `LLMEngine` chat path. It renders every frame after generation ends,
  so no ordering question exists there.
- Streaming error frames. Upstream converts an exception inside the generator
  into a `data: {"error": …}` frame plus `data: [DONE]`
  (`chat_completion/serving.py:827-833`). We have no such seam on either
  endpoint. That gap is [#1992](https://github.com/mudler/vllm.cpp/issues/1992),
  recorded under `## Owed` below; it is a second defect, not this one.

## Upstream chain

Read at the parity pin `555967922`
(`.agents/upstream-sync.md`), in `/home/mudler/_git/vllm`.

| Upstream anchor | What it fixes here |
|---|---|
| `vllm/entrypoints/openai/chat_completion/serving.py::OpenAIServingChat.chat_completion_stream_generator` `:477` | `async for res in result_generator:` — the loop that must produce a result before anything is yielded. |
| the same function, `:484-486` | The reason, in upstream's own words: "We need to do it here, because if there are exceptions in the result_generator, it needs to be sent as the FIRST response (by the try...catch)." |
| the same function, `:487-534` | `if first_iteration:` builds and yields the role chunk, inside the loop body. |
| the same function, `:827-833` | `except GenerationError` / `except Exception` yields the streaming error response. This is the arm the ordering exists to protect. |
| `vllm/benchmarks/lib/endpoint_request_func.py:404-408` | `if choices := data.get("choices"):` then `if ttft == 0.0:` — TTFT is stamped on the first frame carrying a `choices` key, whatever `delta.content` holds. |

## The measurement consequence

Our role frame carries `choices[0].delta.content = ""` and no `usage`, so it
satisfies the TTFT guard above. Emitted before any engine work, it makes
`vllm bench serve --backend openai-chat` record the HTTP round trip to an empty
frame instead of the time to a first token. The number is near zero and does not
move with load. vLLM and SGLang emit their role frame after the first result, so
their rows on the same harness measure the real quantity and ours does not.
This blocks the `#1574` three-engine comparison, whose harness uses
`--backend openai-chat`.

The artifact flatters this engine and only this engine, which is the property
that makes it a correctness problem rather than a benchmarking footnote.

## Reversing a recorded decision

`.agents/specs/stream-options.md` recorded the current behavior on purpose, in
two places:

- `:110-113` — "Chat continuous usage may buffer the first `RequestOutput` long
  enough to know the prompt-ID count before emitting the role frame; this
  mirrors upstream, which emits that role frame only after the first result
  arrives."
- `:197-199` — "Chat's role frame must not invent a prompt count. In continuous
  mode it waits for/buffers the first engine result, matching upstream's
  first-iteration ordering rather than reporting zero."

Both sentences are true about continuous usage and both stop there. The recorded
reasoning treated the buffering as a means to a native prompt count, so it
scoped the wait to the mode that needs that count. Upstream's ordering has a
second and a third purpose that the record did not weigh:

- upstream's stated purpose, error ordering (`:484-486`), which is independent
  of usage mode;
- the TTFT stamping rule above, which did not exist in the analysis at all.

The narrow reading is therefore wrong rather than outdated. This change updates
both passages in `stream-options.md` so that the record and the code agree.

## Design

One edit in `src/vllm/entrypoints/openai/serving_chat.cpp::ChatSseStream`, in its
`next`: remove the `if (usage_.include_continuous_usage)` guard around the
first-result buffering loop, so both modes run it.

The loop already has the shape both modes need. It calls `WaitOutput`, returns
a standalone ping frame when the keepalive interval expires, records
`prompt_tokens_`, and stores the first result carrying an output (or a finished
result) in `buffered_response_`. The main content loop already consumes
`buffered_response_` before it waits again, so no result is dropped or
duplicated.

Nothing else moves. The frame the default path emits is byte-identical, because
the default path attaches no `usage` and `prompt_tokens_` is not serialized
there.

### What this costs a real client

The first byte of the SSE body now arrives when the first token is ready rather
than when the request is admitted. A client that renders a typing indicator on
the role frame loses that early signal by the true prefill time. The first
*token* is not delayed: the token that used to arrive in frame two now arrives
in frame three, at the same instant, because both frames are written from one
`RequestOutput` that the stream already held.

TTFT as measured by `vllm bench serve --backend openai-chat` gets worse, and
should. It was previously measuring an HTTP round trip.

Worker-thread occupancy does not change. `create_chat_completion` still returns
without waiting, and the wait moves from the second `next()` call to the first,
on the same cpp-httplib worker thread that was going to block either way
(`src/vllm/entrypoints/openai/api_server.cpp::ApiServer::register_routes`, the
chunked content provider). `AsyncLLM` keeps batching every other request. No
concurrency decision is needed.

## Tests

New file `tests/vllm/entrypoints/openai/test_chat_stream_first_frame.cpp`,
driving the production `ApiServer::handle_chat_completions` dispatch over a real
`AsyncLLM` whose model runner the test controls.

1. **Frame ordering.** The runner blocks inside `sample_tokens` until the test
   releases it, and counts the steps it has sampled. A background thread calls
   `next()` once. While the runner is gated, the test asserts its own
   precondition (`sampled_steps() == 0`, so no token exists) and then asserts
   that no frame has arrived. It releases the runner and asserts that the frame
   then arrives, that it carries a `choices` key, and that a token existed by
   the time it did.

   A shape assertion on the role frame would pass before the change and prove
   nothing, so the discriminating assertion is the negative one taken while the
   runner is gated.

2. **Error ordering.** The runner throws inside `sample_tokens`. The engine
   guard posts `ENGINE_CORE_DEAD`, `AsyncLLM` propagates the error to the
   request's collector, and the collector rethrows on the consumer thread. The
   test asserts that the *first* `next()` call throws and that it wrote no
   frame. Before the change the first call returns the role frame and does not
   throw.

3. **No regression in the existing frame set.** `test_sse_keepalive`,
   `test_api_server` and `test_serving` keep the role frame first among the data
   frames, keep the continuous-usage counts, and keep the keepalive contract.

### Reachability

The production entry point is `ApiServer::handle_chat_completions`, which the
`/v1/chat/completions` route calls. Both new cases enter through it, not
through `ChatSseStream`, which is in an anonymous namespace and unreachable by
name.

The reachability mutation deletes the buffering loop's production call site in a
scratch copy and reruns the focused gate. The gate must go red.

## Gates

1. Focused: the new target plus every neighbouring OpenAI suite.
2. Full CPU CTest on a CUDA-OFF build.
3. `scripts/agent-preflight.sh`, exit code captured explicitly. A tail that
   reads clean here has exited 1 before, so the code is the verdict.

No GPU axis is claimed. This change alters an HTTP arrival time, and the online
gate runs on `/v1/completions`, which this change does not touch.

## Evidence

Measured 2026-08-26 on `linux/x86_64`, GCC 13.3.0, CMake `Release`,
`-DVLLM_CPP_CUDA=OFF -DVLLM_CPP_BUILD_TESTS=ON`, in
`.wt/chat-role-frame-order` off base `21fe11cf1`.

| What | Command | Result |
|---|---|---|
| Red, parent tree | `./build/tests/test_chat_stream_first_frame` | 2 cases failed, 4 assertions; the role frame appears in the failure text of both cases |
| Green, fixed head | the same | 2 cases passed, 21 of 21 assertions |
| Focused | `ctest -R 'test_chat_stream_first_frame\|test_sse_keepalive\|test_openai_api_server\|test_openai_serving\|test_openai_serving_chat_stream\|test_openai_protocol\|test_openai_conformance\|test_openai_logprobs'` | 8 of 8 passed, exit 0 |
| Full CPU CTest | `ctest -j 4 --output-on-failure` | 628 of 628 passed, exit 0 |
| Preflight | `scripts/agent-preflight.sh` | exit 1, one gate: `test_cpu_x86_llamacpp_floor`. Zero SKIPs. Every other gate `ok`, including `commit-trailers` and `commit-style` |

The one preflight failure is [#618](https://github.com/mudler/vllm.cpp/issues/618),
not this change. Its contended-leg case is load dependent, and the recorded
signature is exactly what it printed: `NO_QUIET_WINDOW` (4) where it expects
`GIVING_UP` (2), here at `load=33.97` while the box carried other sessions'
builds. Run alone at load 21.38 the same file passes 10 of 10 in 20.8 s.

### Mutations, on the final head

The tree was restored from a byte copy after each one and `sha256sum -c`
confirmed both source files, and the suite was re-run green afterwards. The
mutation counts below were taken at 20 assertions, before one diagnostic
assertion was added to the puller thread; neither mutation touches it.

| Mutation | Result |
|---|---|
| Reinstate `if (usage_.include_continuous_usage)` around the buffering loop | 2 cases failed, 4 assertions — the same 4 as the red |
| Delete the production call site `out.sse_stream = std::move(result.sse_stream)` in `ApiServer::handle_chat_completions` | 2 cases failed at 6 assertions; the gate cannot reach the code without the route handler |

The second is the reachability mutation. It is what separates "the class works"
from "a client reaches it".

## Owed

- [#1982](https://github.com/mudler/vllm.cpp/issues/1982) — this change closes
  the ordering half.
- [#1992](https://github.com/mudler/vllm.cpp/issues/1992) — a streaming error
  frame. Neither `ChatSseStream::next` nor
  `CompletionSseStream::next` converts an engine exception into
  `data: {"error": …}` + `data: [DONE]` the way
  `chat_completion/serving.py:827-833` does. The exception reaches the cpp-httplib
  content provider (`src/vllm/entrypoints/openai/api_server.cpp`, the
  `catch (...)` in the chunked provider), which logs to `stderr` and truncates
  the body. After this change no role frame precedes that truncation, so the
  client no longer sees a well-formed start to a request that died; it sees an
  empty 200. That is an improvement and not a fix.

## Stop conditions

- Return `NEEDS_DECISION` if the buffering cannot be done without holding a
  worker thread longer than the current path already holds it. Measured: it
  cannot happen, because the wait moves rather than accumulates.
- Return `NEEDS_CONTEXT` if the pinned oracle tree cannot be read. It could:
  `/home/mudler/_git/vllm` is a shallow clone whose tree at `555967922` is
  complete.
