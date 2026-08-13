# Spike — Reasoning parsers (`SAMPLE-REASONING`)

Row: `.agents/engine-matrix.md` → `SAMPLE-REASONING`
Upstream anchor: `vllm/reasoning/__init__.py:22` (the `_REASONING_PARSERS_TO_REGISTER`
registry) + `vllm/reasoning/abs_reasoning_parsers.py:26` (`ReasoningParser` ABC),
`:213` (`ReasoningParserManager`).
Pinned oracle: `/home/mudler/_git/vllm` @ `555967922` (vLLM 0.26.0.dev0).
Claim: `CLAIM-SAMPLE-REASONING`.
Issue: [#605](https://github.com/mudler/vllm.cpp/issues/605) (W3 resequencing + brick 1).

## Scope

A **reasoning parser** splits a model's raw generated string into a REASONING
span (chain-of-thought, usually wrapped in `<think>…</think>`-style markers) and
the user-visible CONTENT. It surfaces the reasoning as the OpenAI
`reasoning_content` response field (streamed on `delta.reasoning`, non-streamed
on `message.reasoning_content`), and it **gates reasoning-conditioned structured
output**: a grammar/tool constraint must not begin until the reasoning span has
closed (`is_reasoning_end`).

This is the serving/entrypoints text-parsing seam, the exact analogue of the
tool-parser seam in `src/vllm/entrypoints/openai/tool_parsers/`. It is a
device-neutral pure function of the detokenized stream — CPU-only, no GPU. In
scope: the `<think>` reasoning/content split (non-streaming + streaming), the
`--reasoning-parser`/C-ABI/template-auto selection surface, and the coverage of
upstream's ~28 registered parser names. Out of scope for this row: the xgrammar
token-ID gate internals (token-ID methods) and `reasoning_effort` sampling hints.

## Upstream chain

The `ReasoningParser` ABC methods (`abs_reasoning_parsers.py`) and their text-only
mapping onto our seam:

| upstream (`abs_reasoning_parsers.py`) | purpose | our seam (`reasoning_parsers/abstract.h`) |
|---|---|---|
| `extract_reasoning(model_output, request) -> (reasoning-or-None, content-or-None)` :146 | non-streaming whole-string split | `ExtractedReasoning extract_reasoning(model_output, request)` |
| `extract_reasoning_streaming(prev_text, cur_text, delta_text, prev_ids, cur_ids, delta_ids) -> DeltaMessage-or-None` :166 | incremental per-delta split | `std::optional<DeltaMessage> extract_reasoning_streaming(prev_text, cur_text, delta_text, request)` |
| `is_reasoning_end(input_ids) -> bool` :73 | has the reasoning span closed? (gates structured output) | `bool is_reasoning_end(const std::string& text)` (TEXT form) |
| `is_reasoning_end_streaming(input_ids, delta_ids)` :90 | decode-step form | folded into the streaming path |
| `extract_content_ids(input_ids)` :115 | token-ID content slice (xgrammar) | DROPPED (token-ID method) |
| `count_reasoning_tokens(token_ids)` :127 | reasoning-token accounting | DROPPED (token-ID method) |
| `reasoning_start_str` / `reasoning_end_str` :48,55 | span delimiters | per-parser (`start_token`/`end_token` on BaseThinking) |

`BaseThinkingReasoningParser` (`basic_parsers.py:18`) is the `<think>…</think>`
family core: non-streaming `partition(start)` then split on `end` (a start-less
stream still treats its leading span as reasoning up to the first `end`);
streaming is a 4-way membership test over `{start,end} × {previous,delta}`; a lone
marker delta is swallowed (`None`); `is_reasoning_end` = last `end` after last
`start`.

Upstream's registry (`__init__.py:22`) = 28 names across ~24 modules, grouped by
mechanism:
- **A. Text `<think>`-family** (`BaseThinking` subclasses): `deepseek_r1`,
  `mistral` (`[THINK]`/`[/THINK]`), `olmo3`, `step3`, `step3p5`, `poolside_v1`,
  `ernie45` (end-only), `hunyuan_a13b`, `hy_v3`, `kimi_k2` (end-only `◁/think▷`).
- **B. End-only / append-think** (`minimax_m2` family): `minimax_m2`,
  `minimax_m2_append_think`, `minimax_m3`.
- **C. Delegating / thinking-gated adapters**: `deepseek_v3`
  (`deepseek_v3_reasoning_parser.py:20` — R1 or Identity by `thinking`), `holo2`
  (`:83` — v3 defaulting thinking=True), `IdentityReasoningParser`
  (`identity_reasoning_parser.py:17`, internal passthrough), `cohere_command3/4`,
  `granite` (`granite_reasoning_parser.py:18` — regex `Here is my thought
  process:` / `Here is my response:`, NOT special-token), `openai_gptoss`
  (`<|channel|>analysis…<|end|>` channels + structural-tag prep).
- **D. Engine-backed reasoning adapters** (`*ParserReasoningAdapter`, thin
  re-exports of `vllm/parser/engine/registered_adapters.py`, `engine_based_streaming
  = True`): `qwen3`+`mimo`, `gemma4`, `glm45`+`glm47`, `seed_oss`, `deepseek_v4`,
  `nemotron_v3`, `inkling` — the SAME streaming parser engine tracked by
  `TOOLS-STREAMING-PARSER` (engine core + assembly + 12 configs already landed).

## Our baseline

**This row read `INVENTORIED` with empty anchors, but a mature seam is ALREADY
LANDED on `main`** (shipped under the tool-parser wave-B2 change `da933828`, the
ABI-v5 change `eb9d1291`, and the `think_auto` default `5fffe7e6` — none of which
advanced THIS row). What exists today:

- Base ABC + registry: `reasoning_parsers/abstract.{h,cpp}` (`ReasoningParser`,
  `ExtractedReasoning`, `get_reasoning_parser`, `reasoning_parser_names`).
- `BaseThinkingReasoningParser`: `reasoning_parsers/basic.{h,cpp}`.
- 7 registered names (pre-W1): `think_auto` (our auto-detect default),
  `deepseek_r1`, `mistral`, `minimax_m2`, `minimax_m2_append_think`, `step3`,
  `olmo3`.
- Template auto-detection + `--reasoning-parser` resolution:
  `reasoning_parsers/detect.{h,cpp}`.
- C ABI v5 selection: `vllm_model_params.reasoning_parser` (`include/vllm.h`,
  `src/capi/vllm_c.cpp`).
- Serving wiring: reasoning-before-tools routing + `reasoning` SSE delta
  (`serving_chat.cpp`, `protocol.{h,cpp}`; `include_reasoning` request field).
- Unit tests: `test_{base_thinking,deepseek_r1,mistral,minimax_m2,step3,olmo3,
  detect,think_auto}.cpp` + `reasoning_test_utils.h` (mirrors `tests/reasoning/utils.py`).

So W1's "first brick" (base + registry + DeepSeek-R1 `<think>` split + streaming
+ no-think edge + upstream test) was **already satisfied**. This spike backfills
the row to reality and lands the next additive brick.

**Documented deviations** (already in `reasoning_parsers/abstract.h`, all
TEXT-ONLY-scoped, mirroring the tool-parser seam): the ctor drops the tokenizer;
the token-ID methods (xgrammar) are replaced by a TEXT `is_reasoning_end(text)`
(equivalent for well-formed streams since think markers are self-delimiting
special tokens); the streaming signature drops the three token-ID spans; the
lazy/plugin registry collapses to the hand-wired factory. Streaming carries the
span on `DeltaMessage.reasoning`; empty content collapses to `nullopt`.

## Port map

Exact files (W1, this change):
- NEW `include/vllm/entrypoints/openai/reasoning_parsers/identity.h` +
  `src/vllm/entrypoints/openai/reasoning_parsers/identity.cpp` — port of
  `identity_reasoning_parser.py:17` (passthrough: `(None, model_output)`;
  content-wrap streaming; `is_reasoning_end` always True).
- NEW `include/vllm/entrypoints/openai/reasoning_parsers/deepseek_v3.h` +
  `src/vllm/entrypoints/openai/reasoning_parsers/deepseek_v3.cpp` — port of
  `deepseek_v3_reasoning_parser.py:20,83` (thinking-gated delegation to
  `DeepSeekR1ReasoningParser` / `IdentityReasoningParser`; `holo2` = the
  thinking-default variant). Name-only factory mirrors DEFAULT construction:
  `deepseek_v3`→thinking=False→Identity, `holo2`→thinking=True→R1.
- EDIT `reasoning_parsers/abstract.cpp` — 2 factory branches (`deepseek_v3`,
  `holo2`) + the name list (7→9).
- EDIT `CMakeLists.txt` (2 src lines), `tests/CMakeLists.txt` (1 test).

Remaining waves reuse this same file pattern (one `.{h,cpp}` per family + one
factory branch + one test), or, for group D, wrap the already-landed
`parser/engine/` adapters.

## Tests to port

| upstream test | covers | our test | status |
|---|---|---|---|
| `test_base_thinking_reasoning_parser.py` | BaseThinking family | `test_base_thinking.cpp` | DONE |
| `test_deepseekr1_reasoning_parser.py` | `<think>` split + streaming + no-think edge | `test_deepseek_r1.cpp` | DONE |
| `test_deepseekv3_reasoning_parser.py` | v3 thinking-gated selection + identity passthrough + v4 alias | `test_deepseek_v3.cpp` | **THIS CHANGE** |
| `test_mistral_reasoning_parser.py` | `[THINK]` | `test_mistral.cpp` | DONE |
| `test_minimax_m2_reasoning_parser.py` (+append) | end-only + append-think | `test_minimax_m2.cpp` | DONE |
| `test_step3p5_reasoning_parser.py` | step3 | `test_step3.cpp` (step3); step3p5 pending | PARTIAL |
| `test_olmo3_reasoning_parser.py` | olmo3 | `test_olmo3.cpp` | DONE |
| `test_qwen3_reasoning_parser.py` | qwen3 + mimo: engine-backed `<think>` + implicit `<tool_call>` reasoning end, thinking-off passthrough, multi-token deltas | `test_qwen3.cpp` | **DONE (W3 brick 1, 2026-08-13)** |
| `test_{granite,gptoss,glm4_moe,gemma4,hunyuan,hy_v3,kimi_k2,cohere,nemotron_v3,minimax_m3,holo2}_reasoning_parser.py` | remaining families | — | MISSING (W3 rest / W2) |

The v4-alias sub-case is SKIPPED-with-reason in `test_deepseek_v3.cpp`
(engine-backed adapter, W3). Each ported parser is gated over the exact upstream
input strings, non-streaming AND the per-delta streaming reconstruction,
RED-first.

## Gates

- CORRECTNESS: doctest parity — each parser's `(reasoning, content)` over the
  exact upstream input strings (non-streaming + streaming reconstruction),
  RED-first. The oracle here is the parser CONTRACT (a pure text function); the
  upstream test cases ARE the oracle (upstream tests = executable spec), needing
  no GPU and no live vLLM run. An e2e `reasoning_content` check on a live
  `<think>` model is a follow-on once a reasoning checkpoint is wired.
- BUILD: CPU `-Werror` clean (full `vllm` lib + the reasoning test targets).
- PERF: NOT APPLICABLE (serving text parse, off the GPU hot path);
  `docs/BENCHMARKS.md` records the disposition.

## Dependencies

- No new third-party or build dependency; pure C++20 over the existing
  `entrypoints/openai/protocol.h` (`DeltaMessage`, `ChatCompletionRequest`).
- Group-D (engine-backed) adapters depend on the `TOOLS-STREAMING-PARSER`
  engine (`parser/engine/`, already `ACTIVE` — core + assembly + 12 configs
  landed); W3 wires their reasoning face over it, not a fresh text parser.
- Reasoning-gated grammar (W4) depends on the structured-output row (the grammar
  FSM must hold until `is_reasoning_end`); cross-referenced, not owned here.
- `chat_template_kwargs.thinking` threading (W4) depends on the request-params
  plumbing (`adjust_request` hook); `include_reasoning` already exists.

## Work breakdown

- **W0** — this spike; backfill `SAMPLE-REASONING` to reflect the landed seam.
- **W1** (this change) — `deepseek_v3` + `holo2` + internal `identity` delegate,
  registered, upstream test ported (`test_deepseek_v3.cpp`), RED-first. Coverage
  7 → 9 registered names.
- **W2** — remaining text families: `poolside_v1`, `ernie45`, `granite`,
  `hunyuan_a13b`, `step3p5`, `hy_v3`, `kimi_k2`, `minimax_m3`, `cohere_command3/4`,
  `openai_gptoss` (each a direct text port + its upstream test).
- **W3** — the engine-backed reasoning adapters (`qwen3`/`mimo`, `gemma4`,
  `glm45/47`, `seed_oss`, `deepseek_v4`, `nemotron_v3`, `inkling`, `nano_v3`) as
  reasoning faces over the already-landed `TOOLS-STREAMING-PARSER` engine.
  **Brick 1 DONE 2026-08-13 (#605):** `qwen3` + `mimo` (one class, two registry
  names). Landed the reusable base `ParserEngineReasoningAdapter`
  (`adapters.py:35`) that every remaining W3 name plugs into, the `Qwen3Parser`
  engine subclass (`qwen3.py:201`, now also serving `seed_oss` exactly as upstream
  does), and the two engine methods the reasoning face needs —
  `ParserEngine::extract_reasoning_streaming` (`parser_engine.py:519`) and the TEXT
  form of `is_reasoning_end` (`parser_engine.py:595`). Coverage 10 -> 12 names;
  covers 20 of the 76 recipe uses. Next bricks, by recipe demand: `glm45`+`glm47`
  (11), `gemma4` (6), `nemotron_v3` (3), `deepseek_v4` (2), `inkling` (2),
  `seed_oss`, `nano_v3`.

### Ordering amendment 2026-08-13 — W3 runs BEFORE W2 (#605)

The waves above were numbered without demand data. The recipe-surface sweep
supplies it, and it inverts the order.

Measured over `vllm-project/recipes` @ `86c7777aa699482ef1ebd0c5da9fc540ccc00a40`
(157 official model recipes), `--reasoning-parser` is passed **76 times across 20
distinct values**. We resolve 15 of those uses; the other **61 abort startup** with
`unknown reasoning parser`.

| Wave | Recipe uses it covers | Notable |
|---|---:|---|
| **W3** (engine-backed adapters) | **43 / 76** | `qwen3` 18, `glm45` 11, `gemma4` 6, `deepseek_v4` 2, `nemotron_v3` 3, `inkling` 2 |
| W2 (remaining text families) | 18 / 76 | `kimi_k2` 4, `poolside_v1` 4, `hy_v3` 2, `step3p5` 2, `minimax_m3` 1 |
| — | 0 | `ernie45`, `granite`, `cohere_command3/4`, `openai_gptoss` — **zero recipe demand**, all in W2 |

`qwen3` alone (18 uses) outweighs every W2 name combined except `kimi_k2` and
`poolside_v1`. It is also the parser the published Qwen3.5 and Qwen3.6 recipes pass
to models **we already ship token-exact and gated** — so today the engine serves
the model and rejects its own recipe's flag.

**Therefore: W3 runs first, and within W3 the first brick is
`Qwen3ParserReasoningAdapter`** (upstream `vllm/reasoning/qwen3_engine_reasoning_parser.py`,
re-exported from `vllm/parser/engine/registered_adapters.py`), which serves BOTH
`qwen3` and `mimo` — `vllm/reasoning/__init__.py:87` registers `mimo` onto the same
class, so two names land for one port. `glm45` and `glm47` share
`Glm47MoeParserReasoningAdapter` the same way (`__init__.py:55,59`).

Three names belong in W3 and were missing from its list:

- `nano_v3` (1 recipe use) — add to W3.
- `kimi_k3` (1) and `ling3` (1) — **post-pin**; they are not in the registry at
  `5559679229bc961848b121ccdeaa8fa5d79bec98`. Recorded here so they are not
  rediscovered; they land with the next pin advance, not before.

W2 is not cancelled — it is resequenced behind W3. The four zero-demand names stay
in scope because upstream registers them and we mirror upstream; they are simply
not what a user hits first.
- **W4** — request-time `chat_template_kwargs` threading (`adjust_request`) +
  reasoning-gated grammar FSM hold (cross-ref structured-output).

## Risks/decisions

- **Decision**: the `deepseek_v3` runtime `chat_template_kwargs.thinking` override
  is not threaded through the name-only factory; we mirror the DEFAULT
  construction (`deepseek_v3`→Identity, `holo2`→R1) and document it at the port
  site. Threading it is W4. Risk: a caller expecting `deepseek_v3` +
  `{"thinking":true}` to split will instead get passthrough until W4 — mitigated
  by exposing `holo2` for the thinking path.
- **Risk (DISCHARGED 2026-08-13, #605)**: README stated a stale reasoning-family
  count — "7" after W1 made it 9, then `muse_glimmer` made it 10 without a bump.
  The W3 brick-1 change takes it to 12 and updates `README.md`, `docs/USAGE.md`
  and `docs/STATUS.md` in the SAME change. The pinned count in
  `test_detect.cpp` is what forces this: a factory branch added without listing
  the name fails there.
- **Risk/decision**: token-ID methods (`extract_content_ids`,
  `count_reasoning_tokens`) remain dropped under the text-only seam; needed only
  if we port the xgrammar token-ID gate. Deferred with the structured-output row.
- **Concurrency**: the reasoning-parser seam is quiescent (its prior claims DONE);
  the ACTIVE `TOOLS-STREAMING-PARSER` work lives in the SEPARATE `parser/engine/`
  dir — no file overlap.
