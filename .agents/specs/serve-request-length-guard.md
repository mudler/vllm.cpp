# SPEC-SERVE-REQUEST-LENGTH-GUARD — a refusing byte bound at the request boundary

**Issue:** [#1541](https://github.com/mudler/vllm.cpp/issues/1541), filed by the
operator at the merge of [#1539](https://github.com/mudler/vllm.cpp/pull/1539)
and listed under `## Owed` in
[`bpe-quadratic-merge.md`](bpe-quadratic-merge.md).
**Kind:** one refusal added at three request handlers, plus one derived quantity
read off the tokenizer. No kernel, no device code, no model numerics, no new
config key, no new CLI flag. Every gate in this spec runs on a CPU host.
**Row:** `SERVE-REQUEST-LENGTH-GUARD` in
[`engine-matrix.md`](../engine-matrix.md), section "Serving and the OpenAI API".
**Base:** `db648fb88`. Every local line number in this document is read there.
**Pull request shape:** ONE pull request. No answer is recorded for this row
under `## Git integration` in `.agents/developer-preferences.md`, none of
AGENTS.md's three split cases applies — no helper needs a base-reachable spec,
the scope is three call sites, and every wave writes product code — so the
repository default applies.

## Now

**`GATING`.** The guard is implemented, reachable from three registered routes,
and green on the CPU tier. The operator reruns `## Gates` and owns the promotion
to `DONE`. Nothing in this row needs a GPU, a lease, or a checkpoint mount.

## Scope

In scope:

- `vllm::tok::Tokenizer::MaxTokenBytes()`, the longest stored token text in the
  loaded vocabulary, computed once in
  `src/vllm/tokenizer/tokenizer.cpp::FinalizeTables`.
- `ApiServer::max_prompt_bytes()` and `ApiServer::refuse_oversized_prompt`, the
  derived bound and the refusal, in
  `src/vllm/entrypoints/openai/api_server.cpp`.
- The bound's derivation moving into `ApiServer::set_tokenizer`, which is the
  one place that already receives both of its factors and is called by the
  single production wiring seam `ConfigureUtilityEndpoints`
  (`src/vllm/entrypoints/openai/server_main.cpp:1640`).
- Three call sites: `ApiServer::handle_completions`,
  `ApiServer::handle_chat_completions` and `ApiServer::handle_tokenize`.
- Red-first socket cases in
  `tests/vllm/entrypoints/openai/test_api_server.cpp`.

Out of scope, each with a stated reason:

- **Authentication.** There is none anywhere in `src/vllm/entrypoints/`, which
  is why an unauthenticated caller reaches the tokenizer at all. Adding one is a
  separate product decision with its own surface, and a bound is worth having
  whether or not it lands.
- **A raw request-BODY byte bound.** Argued and rejected under `## Design`: it
  would refuse legitimate multimodal chat requests, whose inline base64 media is
  never tokenized as text.
- **A bound on the NUMBER of messages or prompts.** vLLM has one
  (`VLLM_MAX_COMPLETION_PROMPTS`); we do not accept a prompt LIST at all
  (`CompletionRequest::prompt` is a bare `std::string`,
  `include/vllm/entrypoints/openai/protocol.h:190`), so there is nothing to
  bound. Named under `## Owed` so the gap is visible if the list form lands.
- **`/v1/embeddings`, `/v1/audio/*`, `/v1/videos*`, `/detokenize`, the C ABI.**
  Enumerated under `## Design`, with what each one is and is not exposed to.
- **`src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen`.** Untouched by
  design. It is the post-encode token check and it stays exactly where it is.

## Our baseline

Read at `db648fb88`, and checked rather than assumed:

- **The only size bound in the stack is httplib's default.**
  `CPPHTTPLIB_PAYLOAD_MAX_LENGTH` at `third_party/httplib/httplib.h:129-130` is
  `100 * 1024 * 1024`. Nothing in `src/`, `include/`, `cmake/` or
  `CMakeLists.txt` overrides it, and nothing calls `set_payload_max_length`.
- **There is no authentication anywhere in `src/vllm/entrypoints/`.** A
  case-insensitive grep for `api_key`, `api-key`, `bearer`, `authorization` and
  `authenticat` over `src/vllm/entrypoints/` and `include/vllm/entrypoints/`
  returns nothing, exit status 1.
- **`/tokenize` needs no engine and no model.** It is registered whenever a
  tokenizer is attached (`src/vllm/entrypoints/openai/api_server.cpp`,
  `register_routes`), and its handler encodes directly.
- **The generate paths pay the encode BEFORE the length check.**
  `src/vllm/v1/engine/input_processor.cpp:259-260` encodes and `:265` validates,
  so `max_model_len` bounds nothing until the expensive step is already paid.
- **The cost is no longer quadratic and is still unbounded.** `67823aee2` took
  64 KB in one pretoken from 23,620.695 ms to 7.797 ms and moved the exponent
  from about 2 to about 1 ([`bpe-quadratic-merge.md`](bpe-quadratic-merge.md)
  `## Outcome`). Linear against a 100 MB body is a smaller problem than
  quadratic, not the absence of one.

**The longest stored token text, measured over the four committed goldens** with
`json.load` on each `tokenizer.json` and `len(k.encode('utf-8'))` over
`model.vocab` and `added_tokens`:

| golden | vocab entries | longest stored token, bytes | which |
|---|---:|---:|---|
| `tests/parity/goldens/tokenizer_qwen36/tokenizer.json` | 248,044 | **256** | a 128-space run, `Ġ` x 128, `Ġ` being 2 UTF-8 bytes |
| `tests/parity/goldens/tokenizer_deepseek_v2/tokenizer.json` | 100,000 | **256** | the same shape |
| `tests/parity/goldens/tokenizer_muse_glimmer/tokenizer.json` | 200,000 | **192** | `Âł` x 48 |
| `tests/parity/goldens/tokenizer_mistral/tokenizer.json` | 32,768 | **48** | `▁` x 16, the metaspace mark being 3 UTF-8 bytes |

At a 40,960-token `max_model_len` the derived bound is therefore 10,485,760
bytes on a Qwen3.6-class checkpoint and 1,966,080 on a Mistral-class one — a
10x to 53x reduction against httplib's 100 MB, on a quantity that adapts to the
checkpoint instead of being transcribed.

## Upstream chain

**vLLM has NO equivalent for a prompt BYTE bound.** That is a search result with
the paths named, not a conclusion. Read at pin `5559679229bc961848b121ccdeaa8fa5d79bec98`:

| Layer | What it bounds | Anchor | Why it is not the equivalent |
|---|---|---|---|
| uvicorn/h11 | the request line + headers | `vllm/entrypoints/openai/cli_args.py:292-294` `h11_max_incomplete_event_size`, default 4 MB at `vllm/entrypoints/serve/utils/constants.py:9` | vLLM's docstring says "header or body", but h11 applies the cap to the UNDRAINED receive buffer only (`h11/_connection.py:485`, whose own comment reads "431 is Request header fields too large which is pretty much the only situation where we can get here"), and a body arrives as drained DATA events. It does not bound a body |
| request validation | the COUNT of prompts in a list | `vllm/entrypoints/openai/completion/protocol.py:536-553` `validate_prompt_list_length`, `VLLM_MAX_COMPLETION_PROMPTS` default 1024 at `vllm/envs.py:110,1095-1100` | a count, not bytes. It IS the register this refusal mirrors: refused inside a pydantic `model_validator`, so ahead of the router's `check_model`, with the limit named in the message |
| request boundary | the BYTES of an audio upload | `vllm/entrypoints/speech_to_text/base/utils.py:38-46` `read_upload_with_limit`, `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` default 25 at `vllm/envs.py:79,972-976` | a refusing byte bound at the request boundary, checked from `Content-Length` before materialization — but on the audio-upload surface, not on a text prompt |
| input processing | the TOKEN count, after the encode | `vllm/v1/engine/input_processor.py:387-432` `_validate_prompt_len` | the very ordering this row exists to get ahead of |

So the SHAPE is mirrored — refuse during request validation, name the limit,
never truncate — and the NUMBER is ours, which is why `## Design` derives it
instead of transcribing one.

## Port map

| Upstream | Ours today | After |
|---|---|---|
| `completion/protocol.py:536-553` (refuse in request validation, message names the limit) | nothing between the body and the encode | `ApiServer::refuse_oversized_prompt`, called from three handlers before any tokenization |
| `speech_to_text/base/utils.py:38-46` (a refusing BYTE bound at the boundary) | nothing on the text surfaces | the same shape, on the prompt text |
| `envs.py:110` / `envs.py:79` (an ARBITRARY policy number, therefore configurable) | — | a DERIVED number, therefore not configurable. See `## Design` |
| `input_processor.py:387-432` `_validate_prompt_len` | `src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen` | unchanged, and deliberately so |
| no counterpart | no way to ask a tokenizer its worst-case bytes per token | `vllm::tok::Tokenizer::MaxTokenBytes()` |

## Design

**The bound is `max_model_len * MaxTokenBytes()`, and it is derived rather than
chosen.** The token texts of an encode concatenate back to the input, so a
prompt of `B` bytes costs at least `B / MaxTokenBytes()` tokens. Any prompt
longer than `max_model_len * MaxTokenBytes()` therefore exceeds `max_model_len`
tokens, and `ValidatePromptLen` would refuse it after the encode. The guard
refuses it before. **It rejects nothing the server would have served**; it only
moves an already-certain refusal ahead of the work that pays for it.

`MaxTokenBytes()` is the longest STORED token text, which is an OVER-estimate of
the decoded bytes one token can carry, and the over-estimate is the direction
soundness needs. On the byte-level family a stored token holds one mapped
codepoint per input byte, at 1-2 UTF-8 bytes each; on the SentencePiece family
it holds the literal text with the metaspace mark (3 bytes) standing in for one
space and `<0xNN>` (6 bytes) for one fallback byte. Both are at least as long as
what they decode to, so the bound is never tighter than the true one.

**It is FIXED, not configurable, and that is an argument rather than an
omission.** Below the derived value the guard would refuse prompts the server
would serve, which is a behaviour change and not a defence. Above it the guard
is inert on the generate paths, because `ValidatePromptLen` refuses anyway. It
already adapts per checkpoint, through both of its factors, which is what a
config key would otherwise be used for. vLLM makes its analogues configurable
because each of them is an arbitrary policy number — 1024 prompts, 25 MB of
audio — with nothing to derive it from; ours has a derivation, so there is
nothing to tune. This is a divergence from vLLM's register and it is recorded
here as one. No config key, no CLI flag, and therefore no `docs/USAGE.md` key
row.

**Rejected: a raw request-BODY byte bound.** It is checkable earlier still, and
it would bound the JSON parse as well as the encode. It is wrong here because
`/v1/chat/completions` accepts inline base64 media
(`src/vllm/entrypoints/openai/chat_mm.cpp`, `DecodeDataUri`), which is never
tokenized as text: a bound derived from text-token arithmetic would refuse
legitimate multimodal requests, and a bound loose enough not to would not bound
the text. The guard therefore measures the text that reaches the tokenizer.

**Rejected: truncation.** Named as forbidden by
[`bpe-quadratic-merge.md`](bpe-quadratic-merge.md) `## Defence in depth`, and
restated because it is the shortcut somebody reaches for. A shortened prompt
returns model output for text the caller did not send. The refusal is tested by
its ABSENCE of a token list, not only by its status code, so a truncating guard
cannot satisfy the gate.

**Rejected: putting it in `ValidatePromptLen`.** It needs the token count the
expensive step produces (`input_processor.cpp:259-260` then `:265`), so placing
the guard there reproduces the exact ordering that made the original defect
reachable.

### Which surfaces, and which not

| Surface | Covered | Why |
|---|---|---|
| `POST /tokenize` | **yes** | the surface this row exists for: no engine, no model, no credential. Measured on the FINAL prompt, after the chat form's template render, because that string is what the encode is handed |
| `POST /v1/completions` | **yes** | `request.prompt.size()`, before `check_model` and before `create_completion` |
| `POST /v1/chat/completions` | **yes** | the SUM of `ChatMessage::content` over the messages, which is what the chat template concatenates into the one prompt. Inline base64 media lives in `content_parts` and is correctly not counted |
| `POST /detokenize` | no | it takes token ids, not text; its cost is bounded by the id count and it never reaches the BPE merge loop |
| `POST /v1/embeddings` | no | reaches the tokenizer through a different seam (`EmbedFn`), is registered only when an embedder is attached, and `ApiServer` does not parse its inputs into a shape the bound can read. NAMED GAP, `## Owed` |
| `POST /v1/audio/transcriptions` | no | a multipart audio upload, not text. vLLM bounds this one by BYTES (`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`); we do not, and that is a separate mirror. NAMED GAP, `## Owed` |
| `POST /v1/audio/speech`, `POST /v1/videos`, `POST /v1/videos/sync` | no | opt-in routes registered only with their backing attached, and their cost is dominated by generation rather than tokenization. NAMED GAP, `## Owed` |
| the C ABI (`include/vllm.h`) | no | an in-process caller is not an unauthenticated remote one, and it already chooses its own prompt. The bound is a property of the HTTP boundary |
| `/v1/messages` | not applicable | no such route exists in this tree; `grep -rn "v1/messages" src/ include/` returns nothing |

**A guard on one surface is a guard with holes, and the holes above are named
rather than implied.** The three covered surfaces are the three that hand a
caller-supplied STRING to `Tokenizer::Encode` from a registered route.

### The error shape

`MakeError(400, "BadRequestError", ...)` — the register every sibling refusal in
`src/vllm/entrypoints/openai/api_server.cpp` uses, and the status
`InputValidationError` already maps to for the post-encode token refusal
(mirroring `serve/utils/error_response.py:62-65`). The message names what
arrived, the limit, and the derivation:

```
prompt length 289 bytes exceeds the maximum allowed prompt length of 288 bytes
(max_model_len 32 x 9 bytes, the longest token in this tokenizer's vocabulary).
A prompt this long cannot fit in 32 tokens, so it is refused here rather than
tokenized first. The request is refused, not truncated.
```

## Reachability

The chain, at this row's own merge commit:

```
server_main.cpp:1640  ConfigureUtilityEndpoints(server, tokenizer, max_model_len, ...)
  -> api_server.cpp   ApiServer::set_tokenizer(&tokenizer, max_model_len)
                        -> max_prompt_bytes_ = max_model_len * tok.MaxTokenBytes()
  -> register_routes  POST /tokenize, POST /v1/completions, POST /v1/chat/completions
                        -> handle_* -> refuse_oversized_prompt
```

Every gate case drives it over a real socket through
`ConfigureUtilityEndpoints`, the one seam `server_main.cpp` uses, so the tests
enter through the production entry point rather than constructing the guard by
hand. `## Outcome` records the deletion mutation.

## Dependencies

**Code:** none. Host code only, no oracle run, no lease, no checkpoint mount.
The four goldens the derivation was measured on are already committed.

**Record:** none, and this row appends no [`issue-index.md`](../issue-index.md)
row of its own. #1541's row already landed at
[`issue-index.md`](../issue-index.md), appended by the closing commit of
`SPEC-BPE-QUADRATIC-MERGE`, and the index is append-only: a second row for the
same issue is what `scripts/check-agent-record.py` reports as `issue #1541
listed twice`. That row names no owning row ID and is owned through `## Owed` in
[`bpe-quadratic-merge.md`](bpe-quadratic-merge.md), which stays true.

## Work breakdown

Non-overlapping, four commits in ONE pull request.

| W | Deliverable | Reviewable on its own because |
|---|---|---|
| **W1** | The socket cases of `## Tests to port` items 1-4, against UNCHANGED code | they are red, and that red is the finding: `/tokenize` answers 200 to a prompt the server can never serve |
| **W2** | `Tokenizer::MaxTokenBytes()` | one derived quantity, with the four measured goldens behind it; nothing calls it yet |
| **W3** | The bound, the refusal, and the three call sites | W1 turns green and nothing else moves |
| **W4** | The records: this spec's `## Outcome`, the matrix row, `docs/STATUS.md`, `docs/FEATURES.md`, the gate-command baseline | the code is frozen |

## Risks/decisions

| Risk | Why it is real | Control |
|---|---|---|
| The bound refuses a servable prompt | it is new refusal behaviour on a shipped route | it is derived to be an OVER-estimate on both tokenizer families; the "at the bound" case asserts a prompt of exactly `max_prompt_bytes` is tokenized whole |
| A future edit turns the refusal into a truncation | truncation is the obvious shortcut, and a status-only test would accept it | the `/tokenize` case asserts the response carries NO `tokens` and NO `count` key, so a 200 with a shortened encode fails it. Mutation recorded in `## Outcome` |
| The guard is unreached because `set_tokenizer` was never called | `max_prompt_bytes()` is 0 in that state and the guard is silently inert | the "no tokenizer attached" case pins the inert state EXPLICITLY, and every refusal case drives `ConfigureUtilityEndpoints`, the seam `server_main.cpp` actually uses. Deletion mutation in `## Outcome` |
| `/tokenize` now refuses something vLLM accepts | vLLM's `/tokenize` has no bound at all | deliberate and recorded. Counting tokens for text the server can never serve is not a servable use, and the message says why. `## Stop conditions` makes a real complaint a `NEEDS_DECISION` rather than a quiet widening |
| A hostile `max_model_len` overflows the product | `int64_t * size_t` | `set_tokenizer` clamps to `SIZE_MAX` instead of wrapping |
| The chat sum misses text the template adds | a template emits per-message framing, so the rendered prompt exceeds the sum | the residual is bounded by the message COUNT, which the 100 MB body bound still caps. Named under `## Owed` |

## Tests to port

There is no upstream test to port: vLLM has no prompt-byte bound, so item 5 is
the closest mirror rather than a port, and this list says so instead of claiming
a provenance it does not have. All of these run over a real socket unless
stated, because a guard that only fires when a test calls the parse function is
`.agents/reachability.md`'s unpassed-parameter shape.

1. **One byte over the bound is refused, and nothing is tokenized.** `POST
   /tokenize` with `max_prompt_bytes + 1` bytes: 400, `BadRequestError`, the
   message naming both the length received and the limit, and the body carrying
   NEITHER `tokens` NOR `count`. That last pair is the truncation detector.
   RED today: the route answers 200.
2. **A prompt AT the bound is tokenized in FULL.** The same route, exactly
   `max_prompt_bytes` bytes: 200, and `count` equal to
   `Tokenizer::EncodeWithSpecialTokens(...).size()`. Passes today, and must keep
   passing: it is what stops the bound being satisfied by a shortening step
   hidden anywhere on the path.
3. **`/v1/completions` and `/v1/chat/completions` refuse on BYTES.** Both
   already answer 400 on an over-long prompt, from `ValidatePromptLen`, AFTER
   the encode. The cases therefore assert the MESSAGE: the byte limit named, and
   `maximum model length` absent. RED today on the message.
4. **The bound is decided before the model lookup.** `/v1/completions` with an
   unknown model AND an oversized prompt answers 400, not 404 — mirroring vLLM,
   where `validate_prompt_list_length` is a pydantic `model_validator` and so
   runs ahead of the router's `check_model`. RED today: 404.
5. **The derivation, pinned as a derivation.** `max_prompt_bytes()` equals
   `max_model_len * Tokenizer::MaxTokenBytes()` exactly; it is 0 with no
   tokenizer attached and 0 at `max_model_len <= 0`, which is the same "no
   context length is known" state `ValidatePromptLen` early-outs on. A literal
   in items 1-4 would still pass if the bound came from somewhere else.

## Gates

The CPU tier proves all of it; nothing here needs a GPU or a lease.

```sh
cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON
ninja -C build test_openai_api_server test_openai_conformance test_openai_serving \
      test_bpe test_bpe_equivalence test_detokenizer test_tokenizer_parity \
      test_tokenizer_parity_mistral test_tokenizer_parity_deepseek \
      test_tokenizer_parity_gpt4o test_tokenizer_metaspace_split
./build/tests/test_openai_api_server
./build/tests/test_bpe
./build/tests/test_bpe_equivalence
```

Run each suite as its own executable so `Status:` can be read beside
`assertions:`, and assert a NON-ZERO case count: a `-tc` filter typo reports
`0 cases ran` and `SUCCESS!`.

The tokenizer suites are in the gate because `FinalizeTables` is edited, and the
serving suites because three handlers are. `scripts/agent-preflight.sh` is the
full gate.

## Stop conditions

- Stop and return `NEEDS_DECISION` if the derived bound refuses any request the
  server would otherwise have served. That would mean the derivation is wrong,
  not that the constant needs loosening.
- Stop and return `NEEDS_DECISION` if a real caller needs `/tokenize` to count
  tokens for text longer than the bound. The answer is a config key with a
  `docs/USAGE.md` row, and that is a developer decision rather than a quiet
  widening.
- Stop if the guard needs a change to
  `src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen`. A design that
  needs it is the wrong design, by this row's second binding constraint.
- Do not take a GPU or an `rc` lease for anything in `## Tests to port`.

## Owed

Named gaps, none of them a defect this row leaves behind:

- **`/v1/embeddings`, `/v1/audio/transcriptions`, `/v1/audio/speech` and the
  `/v1/videos*` routes carry no length bound.** Enumerated under `## Design`.
  The transcription one has a direct vLLM mirror to port
  (`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`, `speech_to_text/base/utils.py:38-46`); the
  others do not. Owned by this row.
- **The chat guard measures the summed message text, not the rendered prompt.**
  A template's per-message framing is not counted, so a request with very many
  tiny messages is bounded only by the 100 MB body limit. Owned by this row.
- **No prompt-LIST form exists to bound.** If `CompletionRequest::prompt` ever
  becomes a list, `VLLM_MAX_COMPLETION_PROMPTS`
  (`completion/protocol.py:536-553`) is the mirror to port at the same time.
  Owned by this row.

## Outcome

Recorded on the branch, before the merge. Base `db648fb88`, branch
`row/SERVE-REQUEST-LENGTH-GUARD`, five commits.

### What landed

`vllm::tok::Tokenizer::MaxTokenBytes()` is the longest stored token text,
computed once in `FinalizeTables`. `ApiServer::set_tokenizer` derives
`max_prompt_bytes_ = max_model_len * MaxTokenBytes()`, clamped on overflow and
0 when either factor is unknown. `ApiServer::refuse_oversized_prompt` returns a
400 `BadRequestError` naming the length received, the limit and the derivation,
called from `handle_tokenize` immediately before the encode and from
`handle_completions` / `handle_chat_completions` after the body parses and
before `check_model`.

**No config key, no CLI flag, no environment variable.** `## Design` carries the
argument. `docs/USAGE.md` gains a troubleshooting entry for the message rather
than a key row, because there is no key.

### Red, then green

| stage | command | cases | assertions | `Status:` | exit |
|---|---|---:|---:|---|---:|
| RED, at `bfe25bc51` | `test_openai_api_server -tc="api_server: an oversized prompt is REFUSED at the request boundary"` | 1 | 41, **15 failed** | `FAILURE!` | **1** |
| GREEN, at `a5dbf60e3` | the same command | 1 | 41, 0 failed | `SUCCESS!` | 0 |

The red is the finding, not a step toward one: `/tokenize` answered **200** with
a token list to a 289-byte prompt against a 288-byte bound, on a route that
needs no engine, no model and no credential. `/v1/completions` and
`/v1/chat/completions` already answered 400 — from `ValidatePromptLen`, after
the encode — so their cases assert the MESSAGE, and it is the message that was
red. `/v1/completions` with an unknown model answered 404, so the bound was not
yet ahead of the model lookup.

**The first red run reported 6 assertions, not 41**, because `.at("error")` on a
200 body threw out of the whole `TEST_CASE` and hid four of the five subcases.
The accessors are non-throwing for that reason, and the number recorded above is
the one taken after the fix.

### The full focused gate, at `a5dbf60e3`

One fresh `cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON
-DVLLM_CPP_BUILD_EXAMPLES=OFF`, then `ninja -C build -j 16`: **1122 of 1122
targets, zero compiler warnings**, so no stale binary is printing this green.
Every suite run as its own executable so `Status:` could be read beside
`assertions:`, and every one reports a NON-ZERO case count.

| suite | cases | assertions | `Status:` | exit |
|---|---:|---:|---|---:|
| `test_openai_api_server` | 68 | 874 | `SUCCESS!` | 0 |
| `test_openai_conformance` | 23 | 252 | `SUCCESS!` | 0 |
| `test_openai_serving` | 42 | 556 | `SUCCESS!` | 0 |
| `test_bpe` | 24 | 971 | `SUCCESS!` | 0 |
| `test_bpe_equivalence` | 2 | 334 | `SUCCESS!` | 0 |
| `test_detokenizer` | 12 | 221 | `SUCCESS!` | 0 |
| `test_tokenizer_parity` | 4 | 1175 | `SUCCESS!` | 0 |
| `test_tokenizer_parity_mistral` | 6 | 421 | `SUCCESS!` | 0 |
| `test_tokenizer_parity_deepseek` | 6 | 2461 | `SUCCESS!` | 0 |
| `test_tokenizer_parity_gpt4o` | 5 | 1000 | `SUCCESS!` | 0 |
| `test_tokenizer_metaspace_split` | 7 | 28 | `SUCCESS!` | 0 |

### The two mutations

Both applied in the worktree against a pre-taken sha256
(`1788a62b18f42df259ac4f761bff69a85fa01980c8917940d80985b753284429` for
`src/vllm/entrypoints/openai/api_server.cpp`), with the compiler return code and
the applied diff printed each time, and both restored byte-for-byte to that hash
with the suite re-run green after.

**REACHABILITY — delete all three production call sites.** `git diff --no-index`
against the pristine copy: `1 file changed, 9 deletions(-)`, the nine lines being
the three `if (auto refusal = refuse_oversized_prompt(...)) { return *refusal; }`
blocks. `compile_rc=0`, so this is a real mutation and not a build failure
wearing a pass.

| case | result |
|---|---|
| the five socket cases | **41 assertions, 15 failed, `FAILURE!`, exit 1** |
| the derivation case | 10 assertions, 0 failed, `SUCCESS!`, exit 0 |

**That split is itself the finding, and it is why the socket cases exist.** The
derivation case constructs the bound and reads it back; it stays green with the
guard reaching nothing, so on its own it measures a class rather than a
capability — exactly the shape `.agents/reachability.md` names. Only the cases
that enter through the registered route over a real socket detect the deletion.

**TRUNCATION — replace the refusal with `resize(max_prompt_bytes_)` in
`handle_tokenize` and `handle_completions`.** `git diff --no-index`:
`1 file changed, 4 insertions(+), 4 deletions(-)`. `compile_rc=0`. Result: **41
assertions, 13 failed, `FAILURE!`, exit 1**, and the two assertions that fire
FIRST on the truncating build are the truncation detectors —
`CHECK_FALSE(j.contains("tokens"))` and `CHECK_FALSE(j.contains("count"))` — so
the gate distinguishes a refusal from a shortened success rather than only
checking that the request did not hang.

### The post-merge rerun

`origin/main` advanced to `5539686c7` (`MUSIC3-DIT-ARM-REACH`, #1131) while this
row was being gated, so it was merged at `73964532d` and everything was run
again on the merged head. The incoming change touches the MiniMax-Music3 speech
model, its device header, one test and `docs/USAGE.md`; the only file both sides
edit is `docs/USAGE.md`, where the two additions land in different sections.

Rebuild from the same directory: **566 of 566 targets, zero compiler warnings**,
then `ninja: no work to do` on a confirming pass. All twelve suites re-run as
their own executables: identical case and assertion counts to the table above,
every one `SUCCESS!` and exit 0.

Every record checker re-run with `--base 5539686c7`: `check-commit-trailers`,
`check-commit-style`, `check-doc-checkpoint`, `check-now-current`,
`check-issue-index-append-only`, `check-pr-size`, `check-agent-record`,
`check-public-doc-tables`, `check-symbol-anchors`, `check-test-registration`,
`check-surface-coverage`, `check-conflict-markers`, `check-env-doc`,
`check-readme-structure`, `check-role-discipline`, `check-prompt-contract`,
`check-supported-models`, `check-quickstart-recipes`, `check-site`,
`check-model-checklist`, `check-fusion-consistency`,
`check-runner-routing-consistency`, `check-oracle-pins`, `check-snapshot-pins`,
`check-gate-commands --check` -- 25 of 25 rc 0. The range gates EXECUTED against
the real landing tree rather than skipping on a stale base.

**Two known non-findings are recorded rather than left implicit.**
`scripts/check-windows-portability.py` exits 1 on this tree AND at `db648fb88`
with the identical message, `CMakeLists.txt: MSVC /W4 /WX policy is negated on
the C/C++ compile by /w`; this row does not touch `CMakeLists.txt`.
`scripts/agent-preflight.sh` reported `test_cpu_x86_llamacpp_floor` failing at a
1-minute load average of 30.96 on a 20-core box shared with another session's
build; re-run at load 15.21 it is 10 of 10 `OK`, exit 0. That is the load
sensitivity #618 records, not a regression.

### Why each default has its value

**The bound is `max_model_len * MaxTokenBytes()` and not a round number.** It is
the largest prompt that could still fit in the model's context, so it is the
tightest bound that provably refuses nothing `ValidatePromptLen` would have
accepted. Anything smaller is a policy choice that changes behaviour; anything
larger is inert on the generate paths. At a 40,960-token context it is 10 MB on
a Qwen3.6-class checkpoint and 1.9 MB on a Mistral-class one, against httplib's
100 MB.

**`MaxTokenBytes()` reads the STORED text, not the decoded text.** The stored
form is never shorter, so the bound is never tighter than the true one. An
under-estimate would refuse a servable prompt, which is the one failure this
design cannot have.

**The refusal is 400 `BadRequestError`.** The register of every sibling refusal
in the same file, and the status `InputValidationError` already maps to for the
post-encode token refusal. 413 was considered and not taken: this is not a
payload-size limit at the transport, it is a request-validation refusal, and it
must carry the OpenAI `ErrorResponse` body an SDK reads.

**It is not configurable.** Argued in `## Design`. This diverges from vLLM's
register for its own analogues, and the divergence is recorded rather than
hidden: vLLM's numbers are arbitrary policy and must be tunable; ours is derived
and has nothing to tune.

### What was rejected

- **A raw request-BODY byte bound.** Earlier still, and it would bound the JSON
  parse too — but `/v1/chat/completions` carries inline base64 media that is
  never tokenized as text, so a text-derived bound would refuse legitimate
  multimodal requests and a bound loose enough not to would not bound the text.
- **Truncation.** Forbidden by the row's first binding constraint, and the gate
  detects it (above) rather than trusting the prohibition.
- **`ValidatePromptLen`.** Forbidden by the second, and it needs the token count
  the expensive step produces.
- **A number transcribed from vLLM.** `h11_max_incomplete_event_size`'s 4 MB was
  the tempting one. It bounds the header block, not the body, and taking it
  would have been a constant with a citation that does not support it.

### Limitations, disclosed rather than closed

1. **No end-to-end measurement of what the bound saves.** After `67823aee2` a
   64 KB prompt encodes in tens of milliseconds, so the guard protects against a
   regression and against a 100 MB body rather than against today's measured
   cost. No throughput or latency figure is claimed, and `docs/BENCHMARKS.md` is
   therefore not edited.
2. **Four routes and the C ABI are unbounded**, listed under `## Owed` with the
   one that has a direct vLLM mirror to port named.
3. **The chat arm measures the summed message text, not the rendered prompt.**
   Also under `## Owed`.
