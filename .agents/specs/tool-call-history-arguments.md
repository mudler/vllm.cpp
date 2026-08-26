# SERVE-TOOL-HISTORY-ARGS — normalize OpenAI tool arguments before render

Issue: [#526](https://github.com/mudler/vllm.cpp/issues/526)

Row: `SERVE-TOOL-HISTORY-ARGS`

Reconstructed on origin/main `6354755ba570848c9f8f1e1fb47d732833346c26`
(Researcher `60ee` / `12d9` / Bakon GO `eeb9`). Predicate sources on
`fork/main`: `7bc45c9f`, `d2a6f260`, `f2ddafa3`. Not cherry-picked.

Status: **spec checkpoint.** Implementation is a later commit in the same PR.
Live Gemma withheld. No GPU.

## Problem

OpenAI Chat Completions represents
`assistant.tool_calls[].function.arguments` as a JSON-encoded string. Current
main `src/vllm/entrypoints/chat_template.cpp::BuildMessages` copies that string
verbatim into the Jinja `messages` value.

That diverges from pinned vLLM. `vllm/entrypoints/chat_utils.py`
`_postprocess_messages` decodes assistant tool history before every HF
chat-template render:

- non-empty string → `json.loads`;
- empty / absent / JSON `null` → `{}`;
- already-structured values stay structured;
- malformed JSON raises instead of entering the prompt.

Gemma 4 makes the divergence observable. Native history is:

```text
<|tool_call>call:terminal{command:<|"|>date<|"|>}<tool_call|>
```

String-passthrough renders JSON-quoted keys. The model can imitate that form;
the Gemma parser then treats quotes as part of the key. The client executor is
not the repair surface.

## Upstream anchors

- Pinned vLLM `_postprocess_messages` (chat_utils.py; decode assistant
  `function.arguments` before `apply_chat_template`).
- OpenAI Chat Completions wire: `function.arguments` is a JSON string.
- Google Gemma 4 canonical tool template (mapping required; string arguments
  are a deserialize-before-render error).
- Issue [#526](https://github.com/mudler/vllm.cpp/issues/526).
- Researcher parent RED on exact `6354755b` (`e0d2`): overlay
  `14760709…`, parent binary `1f4152c2…`, log `92d2d87a…`; 27 cases / 5 failed,
  65 assertions / 6 failed. Wire-string and non-assistant complements stay
  green on parent.

## Scope

In:

1. Mirror pinned vLLM assistant-history post-processing at the
   protocol-to-renderer boundary only.
2. Keep `FunctionCall::arguments` as an OpenAI JSON string on the wire.
3. Render-context JSON: empty / `null` → `{}`; valid object/list/scalar kept;
   malformed → `ChatTemplateError` before generation, no payload echo.
4. CPU regressions listed under Tests.
5. Docs: FEATURES tool-call cell notes #526; spec in `.agents/specs/`.

Out:

- Hermes Agent or any downstream client repair.
- Changing parser extraction, tool schemas, or OpenAI response wire type.
- GPU kernels, Q2S/M0, SWA_PHYSICAL child, wvSplitK.
- Live `:8010` / Gemma multi-turn until a separate pinned GO.

## Design

`DecodeHistoricalToolArguments(encoded, index, name)` in
`chat_template.cpp`. `BuildMessages` assigns the decoded value only when
`m.role == "assistant"`. Non-assistant tool-call-shaped messages stay strings.
`to_json(FunctionCall)` is untouched.

## Tests

`tests/vllm/entrypoints/test_chat_template.cpp` plus fixture
`tests/fixtures/gemma4_tool_chat_template.jinja` (sha256 `aa3185df…`):

1. Gemma object history: `{"command":"date"}` renders
   `call:terminal{command:<|"|>date<|"|>}` and not `call:terminal{"command"`.
2. Nested values through Gemma `format_argument`.
3. `""` and `"null"` both render `{}`.
4. Malformed `{"command":` throws `ChatTemplateError`; payload absent from
   `what()`.
5. Wire serialization still a string.
6. `role=user` malformed tool-call-shaped message is not normalized.
7. Existing Qwen3.5 multi-turn fixture stays green and emits decoded
   `<parameter=city>Rome`.

Parent must RED cases 1–4 and 7 for the intended reason (`e0d2` banked).
Child must PASS the full `test_chat_template` suite.

## Gates

```bash
cmake --build <cpu-build> --target test_chat_template -j2
ctest --test-dir <cpu-build> -R '^test_chat_template$' --output-on-failure
```

HIP OFF. No server. `git diff --check` on touched files. No push. Live Gemma
is a later GO.

## Evidence

- Parent RED: Researcher `e0d2` on exact pin (hashes above).
- Child CPU: recorded in the implementation commit / RESULT after rebuild.
- `.env` is **missing** in this worktree. Environment-gated host/path values
  stay PENDING. Not invented.

## Risks and stop conditions

- Templates that expected raw strings: stop and compare the same template
  under pinned vLLM; do not add a model-specific heuristic.
- Malformed client history: fail closed; do not coerce into the prompt.
- Do not change `FunctionCall::arguments` wire type.
- Stop: no combined spec+product commit.
- Stop: no live daemon bounce without exclusive `:8010` owner + pinned GO.
- Stop: no Hermes source changes.

## Git integration

Default **one PR**, two commits, no force-push:

1. This spec + exactly one append-only `.agents/issue-index.md` row for #526.
2. Product / tests / FEATURES as a later child commit on the same branch.

No push until Researcher review of the immutable child HEAD.

## Now

Helper claimed `SERVE-TOOL-HISTORY-ARGS` on
`wip/reconcile-eeb9-20260823`. Spec checkpoint next. Implementation follows
in a separate commit. SWA child not started.
