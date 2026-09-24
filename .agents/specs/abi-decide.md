# ABI-DECIDE: Unify vllm_systemone + vllm_score into vllm_decide (ABI v29)

## Now

`SPIKE` — spec committed, implementation pending.

## Upstream chain

No vLLM upstream anchor. This is a vllm.cpp-internal ABI refactor. The
category-level pattern (one C ABI function per capability class) is already
established: `vllm_complete` for LLMs, `vllm_embed` for embeddings,
`vllm_gliner_ner` for NER, `vllm_transcribe` for ASR. The decision/scoring
capability class currently has two model-specific entry points instead of one.

## Our baseline

ABI v28 ships four functions for the decision/scoring capability class:

- `vllm_systemone(vllm_engine*, const char* request_json, char** out_json)` —
  kev/laya decision pipeline (architecture `KevModel` or `LayaModel`)
- `vllm_systemone_free(char* json)`
- `vllm_score(vllm_engine*, const char* request_json, char** out_json)` —
  cua-s1 scoring pipeline (architecture `CuaS1Forms`)
- `vllm_score_free(char* json)`

Declared in `include/vllm.h:1175-1201`, implemented in
`src/capi/vllm_c.cpp:1692-1897`.

The HTTP server endpoints (`/v1/systemone`, `/v1/score`) use callback function
pointers (`decision_`, `score_`) set in `api_server.cpp`, not the C ABI
directly. The C ABI is consumed by the LocalAI purego backend.

## Port map

Replace the four v28 functions with two v29 functions:

```c
VLLM_API vllm_status vllm_decide(vllm_engine* engine,
                                   const char* request_json,
                                   char** out_json);

VLLM_API void vllm_decide_free(char* json);
```

`vllm_decide` dispatches by architecture:

- `KevModel` or `LayaModel` → decision pipeline (current `vllm_systemone` body)
- `CuaS1Forms` → scoring pipeline (current `vllm_score` body)
- other → `VLLM_ERR_INVALID_ARGUMENT` with a message naming the architecture

The request JSON is the raw HTTP body. The engine architecture determines the
pipeline. The caller does not specify a "mode" — the architecture IS the mode.

Ownership semantics are unchanged: `*out_json` is a heap-allocated
NUL-terminated string on success (caller frees with `vllm_decide_free`), NULL
on error.

### Files to change

1. `include/vllm.h` — replace v28 block comment + 4 declarations with v29
   block comment + 2 declarations
2. `src/capi/vllm_c.cpp` — replace `vllm_systemone`/`vllm_systemone_free`/
   `vllm_score`/`vllm_score_free` with `vllm_decide`/`vllm_decide_free`

### Files NOT changed

- HTTP server (`api_server.cpp`, `systemone.cpp/.h`) — uses callbacks, not C ABI
- Model code (`kev_registry.cpp`, `qwen3_5.cpp`, etc.) — internal to the engine
- `docs/FEATURES.md`, `docs/USAGE.md` — updated in the record PR (#3299) which
  references v28; a follow-up doc update to v29 can ride in this PR or a later
  record-only PR

## Tests to port

The existing C ABI tests for `vllm_systemone` and `vllm_score` need to call
`vllm_decide` instead. These tests live in `tests/vllm/` and exercise the
engine through the C ABI. The test assertions (JSON response shape, decision
values, score values) remain identical — only the function name changes.

## Dependencies

None. This is a pure rename-and-merge of two C ABI functions into one. No
new library dependencies, no new model code.

## Work breakdown

1. Edit `include/vllm.h`: replace v28 comment block and 4 declarations with
   v29 comment block and 2 declarations
2. Edit `src/capi/vllm_c.cpp`: merge the two function bodies into one
   `vllm_decide` that dispatches by architecture; replace the two `_free`
   functions with one `vllm_decide_free`
3. Update any C ABI test that calls `vllm_systemone` or `vllm_score` to call
   `vllm_decide` instead
4. Build (CPU), run checkers

## Gates

- `check-agent-record.py` — PASSES (roadmap row + spec committed)
- `check-commit-style.py` — PASSES (subject, body, trailers)
- CPU build compiles with the new function signatures
- Existing C ABI tests pass with `vllm_decide` replacing the two old functions

## Risks

- **ABI break for LocalAI**: the purego bindings must be updated in lockstep.
  PR #12241 on LocalAI will be rewritten to call `vllm_decide` instead of
  `vllm_systemone`/`vllm_score`. The two PRs (vllm.cpp + LocalAI) land
  together.
- **No semantic change**: the JSON request/response format is unchanged. Only
  the C function name changes. The HTTP endpoints are unaffected because they
  use callbacks.

## Git integration

One pull request: spec + implementation in the same PR. The spec commit lands
first, then the code commit, so commit order proves spec-before-code.

## Outcome

Merged as PR #3301 (squash `3abc69ffd`).

**What shipped.** `vllm_systemone` / `vllm_systemone_free` and `vllm_score` /
`vllm_score_free` (ABI v28) were replaced by `vllm_decide` /
`vllm_decide_free` (ABI v29). The engine dispatches by architecture
internally, matching the category-level pattern (`vllm_complete` for LLMs,
`vllm_embed` for embeddings, `vllm_gliner_ner` for NER). The JSON
request/response format is unchanged; only the C function name changed.

**What was rejected.** Keeping a model-per-function ABI surface was rejected
because it mirrors the model, not the capability category, and scales linearly
with each new decision model. A single `vllm_decide` with a `question_type`
field in the request JSON was chosen instead, because the HTTP layer already
used `question_type` for routing and the C ABI only needed to match.

**Defaults.** ABI version bumped v28 → v29. The `Assisted-by` trailer records
`AGENT:regolo/glm5.2 [maki]`. LocalAI PR #12247 updates the purego bindings to
ABI v29 in lockstep.

## Owed

None.
