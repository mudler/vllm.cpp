# Spike: OpenAI stream options and usage frames

**Row:** `SERVE-STREAM-USAGE` · **claim:**
`CLAIM-SERVE-STREAM-USAGE-1` · **state:** `SPIKE`.

## Scope

Port vLLM's complete text-generation `stream_options` contract for both
`/v1/completions` and `/v1/chat/completions`:

- parse `include_usage: bool | null` and
  `continuous_usage_stats: bool | null`, with absent/null values resolving to
  false;
- reject a non-empty `stream_options` object unless `stream=true`;
- when `include_usage=true`, emit one final SSE JSON frame before `[DONE]`
  whose `choices` array is empty and whose prompt/completion/total counts come
  from native prompt/output token IDs, never decoded-text re-tokenization;
- when both fields are true, attach cumulative usage to every emitted choice
  frame (and to chat's initial role frame), while retaining the same final
  usage frame;
- ignore `continuous_usage_stats=true` when `include_usage` is false, matching
  `should_include_usage`;
- expose vLLM's `--enable-force-include-usage` server flag, which forces both
  final and continuous usage even when the request omits `stream_options`;
- preserve the default wire shape: requests without usage enabled continue to
  emit only choice frames followed by `[DONE]`.

This row covers completion/chat usage only. Cached-prompt detail fields,
system fingerprints, Responses/Anthropic/audio endpoints, and server-wide
usage metrics belong to their own protocol/config/endpoint rows. It does not
change sampling, scheduling, tokenization, model execution, or finish reasons.

## Upstream chain

- Shared schema: pinned vLLM
  `vllm/entrypoints/openai/engine/protocol.py:241-243` defines
  `StreamOptions(include_usage, continuous_usage_stats)`.
- Completion request/validation:
  `vllm/entrypoints/openai/completion/protocol.py:45-66,471-478`.
- Chat request/validation:
  `vllm/entrypoints/openai/chat_completion/protocol.py:193-214,731-737`.
- Selection semantics:
  `vllm/entrypoints/serve/utils/api_utils.py:276-289`; force mode returns
  `(true, true)`, otherwise continuous usage is conditional on final usage.
- Completion execution:
  `vllm/entrypoints/openai/completion/serving.py:298-305,359-454` counts prompt
  IDs and cumulative output IDs, attaches continuous usage to choice frames,
  then emits the empty-choice final usage frame.
- Chat execution:
  `vllm/entrypoints/openai/chat_completion/serving.py:459-512,570-760` includes
  zero-completion usage on the role frame, cumulative usage on content/tool
  frames, and the final empty-choice frame.
- Server flag/config plumbing:
  `vllm/entrypoints/openai/cli_args.py:139-140` and the two serving
  constructors at completion `serving.py:63-74` and chat
  `serving.py:122-163`.
- The binding client already sends
  `stream_options={"include_usage": true}` at
  `vllm/benchmarks/lib/endpoint_request_func.py:174-186`, consumes the final
  count at `:235-247`, and otherwise falls back to approximate decoded-text
  tokenization in `vllm/benchmarks/serve.py:589-605`.

No GPU kernel or dependency dispatch participates in this feature. Runtime
trace proof remains part of the owning online gate because that gate compares
the complete serving path, but the usage-frame implementation itself is a
host-side protocol/serialization path.

## Our baseline

`CompletionRequest` and `ChatCompletionRequest` currently drop the unknown
`stream_options` key in
`src/vllm/entrypoints/openai/protocol.cpp:178-325`. Response structs already
carry optional usage and serialize it when present at
`include/vllm/entrypoints/openai/protocol.h:251-258,370-376` and
`src/vllm/entrypoints/openai/protocol.cpp:404-411,493-500`.

Both synchronous and production AsyncLLM streaming paths count output token
IDs but currently transition directly from the final choice frame to `[DONE]`:
completion `src/vllm/entrypoints/openai/serving_completion.cpp:18-96,143-211`
and chat `src/vllm/entrypoints/openai/serving_chat.cpp:231-351,430-536`.
Non-streaming usage is already native-ID based.

The missing requested frame became observable in the clean 27B online gate at
commit `8289cbd5b3d318ef700632e77d8d846467e086a8`. Its first local repetition
completed c1/c2/c4/c8/c16 and all 192 c32 HTTP requests with no errors. At c32,
request index 158 delivered one first-token frame plus 127 ITLs (128 choice
frames), but decoded text re-tokenized to 126 IDs. The fail-closed validator
therefore rejected total output 24,574 instead of 24,576 and stopped before
vLLM. Raw SHA-256 is
`af4fc15fd5af8db625c4a83b6c6213c139054854bfe18957ad549253f8fc53a9`;
client-log SHA-256 is
`52e142e04ed66d718ddb2022bd79ce9409b03dc305b51a6631ad86679652fde1`.
The server exited, `/tmp/gpu` was released, and the partial arm supplies no
ratio.

## Port map

| Pinned vLLM surface | Local implementation surface |
|---|---|
| `engine/protocol.py::StreamOptions` | `StreamOptions` plus optional request fields in `include/vllm/entrypoints/openai/protocol.h`; JSON parsing/validation in `src/vllm/entrypoints/openai/protocol.cpp` |
| `api_utils.py::should_include_usage` | typed usage-selection helper in `include/vllm/entrypoints/openai/serving_utils.h` and `src/vllm/entrypoints/openai/serving_utils.cpp` |
| completion stream generator | sync and `CompletionSseStream` paths in `src/vllm/entrypoints/openai/serving_completion.cpp` |
| chat stream generator | sync and `ChatSseStream` paths in `src/vllm/entrypoints/openai/serving_chat.cpp` |
| final/continuous `UsageInfo` frames | existing `CompletionStreamResponse` / `ChatCompletionStreamResponse` serializers in `protocol.cpp`; final frames use empty vectors, not sentinel choices |
| `enable_force_include_usage` | serving constructor booleans and `examples/server/main.cpp --enable-force-include-usage` |
| FastAPI/Pydantic validation | existing cpp-httplib dispatch exception-to-400 seam in `src/vllm/entrypoints/openai/api_server.cpp` |

The local pull-stream state machines retain one explicit pending-usage state
between the final choice and `[DONE]`. Chat continuous usage may buffer the
first `RequestOutput` long enough to know the prompt-ID count before emitting
the role frame; this mirrors upstream, which emits that role frame only after
the first result arrives.

## Tests to port

| Upstream executable specification | Local tier / case |
|---|---|
| `tests/entrypoints/openai/completion/test_completion.py:400-553` (`test_completion_stream_options`) | protocol parse/validation cases in `tests/vllm/entrypoints/openai/test_protocol.cpp`; sync serving and production AsyncLLM/API final/continuous/default frame-order/count cases in `test_serving.cpp` and `test_api_server.cpp` |
| `tests/entrypoints/openai/chat_completion/test_chat.py:348-445` (`test_chat_completion_stream_options`) | chat default/final/continuous role+content+empty-choice frame cases in the same focused serving/API suites |
| `tests/entrypoints/openai/chat_completion/test_enable_force_include_usage.py:39-68` | constructor/CLI force mode with omitted request options; server help contract in `examples/CMakeLists.txt` |
| `tests/entrypoints/openai/test_chunked_prompt.py:41-116` | usage-only chunk/cumulative-count behavior over the synthetic engine; its logprobs assertions remain owned by the separate logprobs row and are not silently claimed here |
| `vllm/benchmarks/lib/endpoint_request_func.py:174-247` | real pinned-client regression: the regenerated online gate must read native completion counts for every request, including the retained c32 prompt that re-tokenizes to 126 |

Every focused test asserts ordering (`finish choice -> usage choices=[] ->
[DONE]`), exact native-ID counts, usage omission by default, monotonic
continuous counts, null/empty options, and the invalid non-stream forms. A test
may not replace native counts with the requested maximum or ITL length.

## Gates

1. **Protocol/unit:** focused protocol, serving, API-server and server-help
   targets pass with exact upstream-named cases; default SSE snapshots remain
   unchanged.
2. **Sanitizers/concurrency:** focused ASan+UBSan and TSan AsyncLLM/API streams
   pass; disconnect before the pending usage frame aborts/cleans exactly once.
3. **CPU suite/record:** clean CUDA-OFF build plus full CTest, record checker,
   13 mutation tests, `git diff --check`, and static/syntax checks pass.
4. **Binding regression:** after merge, regenerate both SHA-bound corpora and
   run `SERVE-GATE-ONLINE` for 27B then 35B under separate whole-model flocks.
   Every request must report exactly 128 native output tokens, including the
   retained 27B c32 text-merging case; all repetitions, memory return and paired
   traces remain mandatory.
5. **Performance:** the extra terminal frame may not regress any online axis.
   The fresh identical vLLM denominator and all-axis comparison are binding;
   request-level final usage and continuous usage are also toggles on the same
   binary for an isolated HTTP serialization A/B. Timing stops on choice-token
   arrival, while whole-request throughput still includes consuming the final
   frame. Run 2-3 uncontended repetitions and retain the spread.
6. **Memory/lifecycle:** final/continuous frames do not retain completed engine
   requests; peak memory and post-arm available-memory return stay no worse
   than vLLM in the two-model campaign.

Until gate 4 closes, both `SERVE-STREAM-USAGE` and `SERVE-GATE-ONLINE` remain
open; a successful CPU implementation is `GATING`, never `DONE`.

## Dependencies

- Rows: `SERVE-OAI-BASIC`, `SERVE-ASYNC-LLM`, and `SERVE-GATE-ONLINE`.
- Existing types: `UsageInfo`, native `RequestOutput.prompt_token_ids`, and
  delta `CompletionOutput.token_ids`.
- Toolchain: nlohmann/json, cpp-httplib, doctest, the existing CPU sanitizer
  builds; no new dependency.
- Hardware/data: CPU tests use the synthetic Qwen engine. Closure needs the
  GB10, both exact NVFP4 checkpoints, the pinned vLLM 0.24.0 environment and
  commit-bound corpus/evidence root.
- Licensing: vLLM and the local port are Apache-2.0; copied semantics retain
  upstream path/commit headers.

## Work breakdown

| Leaf | Owned change | Exit |
|---|---|---|
| W0 protocol | `StreamOptions`, completion/chat parsing, null/default and non-stream validation | ported protocol cases pass |
| W1 completion | shared selection helper; sync + AsyncLLM continuous/final usage state | completion serving/API cases pass |
| W2 chat | role/content continuous counts; sync + AsyncLLM final usage state | chat serving/API cases pass |
| W3 force/config | constructor plumbing and `--enable-force-include-usage` | omitted-options force test + help contract pass |
| W4 closure | full CPU/sanitizer gates, exact online rerun, ledger/matrix/roadmap/README status | two-model every-axis evidence classifies the row |

W0-W3 are one protocol feature and one claim because their request schema,
wire-order invariant and shared helper cannot be independently enabled without
creating a knowingly asymmetric completion/chat API. W4 is an independent
hardware checkpoint after the implementation commit.

## Risks and decisions

- Decoded text is not a token-count oracle: adjacent decoded token fragments
  may merge when re-encoded. Native IDs are the only accepted usage source.
- A usage frame has `choices=[]`. Clients that index choice zero must do so only
  for choice frames; the pinned vLLM client already follows this rule.
- `continuous_usage_stats` never implies final usage by itself; upstream gates
  it behind `include_usage`.
- Null booleans resolve false, but a non-empty options object is still invalid
  on a non-stream request, matching Pydantic's before-validator.
- The final usage frame follows the finish-reason choice and precedes `[DONE]`.
  Emitting it after `[DONE]`, attaching it only to the finish choice, or
  estimating it from text would not be wire-compatible.
- Chat's role frame must not invent a prompt count. In continuous mode it waits
  for/buffers the first engine result, matching upstream's first-iteration
  ordering rather than reporting zero.
- The failed `8289cbd` arm is diagnostic only. It cannot be patched in place or
  reused after the implementation changes the commit SHA.
