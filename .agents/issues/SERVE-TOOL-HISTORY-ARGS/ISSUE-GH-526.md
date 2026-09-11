ID: ISSUE-GH-526
Title: OpenAI multi-turn tool history reaches chat templates with string-valued arguments
Row: SERVE-TOOL-HISTORY-ARGS
State: OPEN
Kind: bug
GitHub: 526
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-12
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `SERVE-TOOL-HISTORY-ARGS`
>
> ## Bug
>
> The OpenAI Chat Completions protocol carries `assistant.tool_calls[].function.arguments` as a JSON-encoded string. vllm.cpp preserves that string through `ChatMessage` and `BuildMessages()` and hands it directly to Jinja. This diverges from pinned vLLM `vllm/entrypoints/chat_utils.py::_postprocess_messages`, which `json.loads()` each historical argument string before chat-template rendering (empty/null becomes `{}`).
>
> Gemma 4 exposes the failure because its canonical DSL requires an arguments mapping. A valid first call such as `{"command":"date"}` is rendered into history as `<|tool_call>call:terminal{"command":"date"}<tool_call|>` rather than canonical `<|tool_call>call:terminal{command:<|"|>date<|"|>}<tool_call|>`. On the next turn Gemma imitates the quoted form; the parser faithfully returns a literal quote-wrapped key such as `{"\"command\"":"\"ls\""}`, and clients cannot find `command`. Retries recursively escape it.
>
> ## Reproduction evidence
>
> - `~/llms/logs/agentic_tool_loop_20260811-212603.json`: turns 1-2 have valid arguments; turn 3 is the first completed-history turn and emits the literal quote-wrapped key/value.
> - The same pattern repeats in `agentic_tool_loop_20260811-202653.json`.
> - Fresh-context controls emit valid `{command: ...}`-derived JSON.
> - Current vllm.cpp `src/vllm/entrypoints/chat_template.cpp` stores `fn["arguments"] = tc.function.arguments`.
> - Pinned vLLM converts strings to objects in `vllm/entrypoints/chat_utils.py:1915-1951`.
> - Google/vLLM Gemma 4 canonical templates now fail closed unless historical arguments are mappings.
>
> ## Expected behavior
>
> Before rendering any chat template, vllm.cpp should mirror vLLM message post-processing:
>
> 1. Decode valid JSON argument strings to JSON values.
> 2. Normalize empty/null arguments to `{}`.
> 3. Reject invalid JSON instead of teaching the model malformed history.
> 4. Preserve already-structured arguments internally.
> 5. Gate a complete user → assistant tool call → tool result → next-generation Gemma render, plus non-Gemma template compatibility.
>
> This must be fixed in vllm.cpp, not in Hermes or another client.

## Resolution

-
