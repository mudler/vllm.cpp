ID: ISSUE-GH-695
Title: PR #683: engine-backed tool parsers (inkling) run skip_special_tokens=true, stripping structural markers before ParserEngine
Row: TOOLS-PARSER-BREADTH
State: OPEN
Kind: bug
GitHub: 695
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-13
Updated: 2026-08-13
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Summary
>
> PR #683 registers `inkling` as an engine-backed tool parser, but on the actual OpenAI serving path the model's structural tool-call markers are stripped before the parser sees them, silently breaking tool-calling for `inkling` (and any engine-backed parser).
>
> ## Mechanism (file:line at PR #683 head 945ce0f7)
>
> - `OpenAIServingChat::MakeToolParser` returns `nullptr` for every engine-backed name (`src/vllm/entrypoints/openai/serving_chat.cpp:546-548`), so `inkling` is served by `MakeParserEngine` / `ParserEngine`, not the text-seam `ToolParser`.
> - Upstream `ParserEngine.adjust_request` sets `skip_special_tokens=False`; the C++ `ParserEngine` has **no** `adjust_request` equivalent, and `ToolParser::adjust_request` (e.g. `tool_parsers/kimi_k2.cpp:87-93`) is **never invoked** anywhere in the serving path.
> - `ChatCompletionRequest.skip_special_tokens` defaults `true` (`include/vllm/entrypoints/openai/protocol.h:461`) and is forwarded verbatim by `to_sampling_params` (`protocol.cpp:583`); `serving_chat.cpp` applies no override for the engine path.
> - `inkling`'s structural markers (`<|message_model|>`, `<|content_invoke_tool_json|>`, …) are tokenizer specials, so a default OpenAI request strips the grammar in the detokenizer before `ParserEngine` runs.
>
> ## Test gap
>
> The 18 new tests feed literal marker text directly to the adapter, so they exercise the adapter but NOT the serving/detokenizer path — they pass while the production path fails.
>
> ## Fix direction (for the implementer's spec)
>
> Mirror upstream `ParserEngine.adjust_request` and the existing `kimi_k2` precedent: when an engine-backed tool parser is active with tools (`has_tools && tool_choice != none`), force `skip_special_tokens=false`. Add a serving-path regression test proving a default OpenAI ChatCompletion request preserves `inkling` markers end-to-end through the detokenizer — not another adapter unit test.
>
> ## Provenance
>
> Found in review of PR #683 by the `research` agent; verified in source and diagnosed by the coordinator. Fix to be done by a fresh implementer + fresh reviewer per AGENTS.md.
>
> Kind: bug

## Resolution

-
