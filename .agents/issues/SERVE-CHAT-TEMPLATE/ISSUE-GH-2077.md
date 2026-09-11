ID: ISSUE-GH-2077
Title: server: GGUF models fall back to the naive role-join prompt because the chat template is never loaded from GGUF metadata
Row: SERVE-CHAT-TEMPLATE
State: CLOSED
Kind: bug
GitHub: 2077
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-27
Updated: 2026-08-28
Closed: 2026-08-28

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What is wrong
>
> `server_main.cpp` loads the chat template exclusively from `tokenizer_config.json` via `LoadChatTemplateFromConfig`. When `--model` points to a `.gguf` file, the path resolves to `<file>.gguf/tokenizer_config.json`, which does not exist. The server falls back to `DefaultChatPromptFallback`, a naive `user: <content>\nassistant:` join.
>
> The codebase already has `LoadChatTemplateFromGguf` (`src/vllm/entrypoints/chat_template.cpp:374`), which reads the `tokenizer.chat_template` metadata key from the GGUF file. The server never calls it.
>
> ## Impact
>
> A GGUF model started with `vllm-server --model <file>.gguf` does not understand the naive prompt format. The model interprets the bare `assistant:` as a generation prompt, echoes it back, and loops on `\nassistant:\n` until `max_tokens` is exhausted. Every response is garbage, and tokens-per-second reads as near zero because every token is wasted on the loop.
>
> Observed on `kind_tharp` (ROCm 7.14, Qwen3.5-4B-Q4_K_M.gguf):
>
> ```
> server: no chat template (cannot open tokenizer_config.json:
>   /models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf/tokenizer_config.json);
>   falling back to the simple role-join prompt
> ```
>
> Streaming response (before fix, `max_tokens=20`):
>
> ```
> data: {"choices":[{"delta":{"content":"\n"},...}]}
> data: {"choices":[{"delta":{"content":"assistant"},...}]}
> data: {"choices":[{"delta":{"content":":"},...}]}
> data: {"choices":[{"delta":{"content":"\n"},...}]}
> data: {"choices":[{"delta":{"content":"assistant"},...}]}
> ... (loops until max_tokens)
> ```
>
> The GGUF file carries `tokenizer.chat_template` (7816 chars) in its metadata at key `tokenizer.chat_template`, so the template is available. The server just never reads it.
>
> ## Reproduction
>
> ```sh
> vllm-server --model Qwen3.5-4B-Q4_K_M.gguf --host 0.0.0.0 --port 6001
>
> curl http://localhost:6001/v1/chat/completions \
>   -H "Content-Type: application/json" \
>   -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20,"stream":true}'
> ```
>
> The response loops on `assistant:\n` instead of producing content.
>
> ## Fix
>
> In `server_main.cpp`, when `LoadChatTemplateFromConfig` throws and the model is a `.gguf` file, fall back to `LoadChatTemplateFromGguf` before giving up on the template. The function already exists and is declared in `include/vllm/entrypoints/chat_template.h:155`.

## Resolution

GitHub records closing pull request #2079 (https://github.com/mudler/vllm.cpp/pull/2079) merged on 2026-08-28 as commit `23490accdb0cc28d8c120bb4c477b640866de551`. GitHub closed issue #2077 on 2026-08-28.
