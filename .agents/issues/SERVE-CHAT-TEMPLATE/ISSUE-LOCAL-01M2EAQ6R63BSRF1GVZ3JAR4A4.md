ID: ISSUE-LOCAL-01M2EAQ6R63BSRF1GVZ3JAR4A4
Title: The OpenAI chat path re-parses the Jinja chat template on every request, which is most of the short-prompt TTFT gap to exllamav3 on GB10
Row: SERVE-CHAT-TEMPLATE
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

`MakeChatTemplatePromptFn` (`src/vllm/entrypoints/chat_template.cpp:302`) captures the template STRING, and every request calls `apply_chat_template`, which runs `minja::Parser::parse(template_str, ...)` (`chat_template.cpp:144`) before it renders.

Measured on the Qwen3.8-27B EXL3 target's own `chat_template.jinja` (8952 chars), rendering the 592-char prompt the variadic benchmark's S band sends:
- On an AMD Ryzen 9 9950X3D at -O2, parse took 45.2 ms and render took 0.10 ms per request (warm, mean of 15).
- On dgx:gpu0 (GB10), the server's `--verbose` stage log put HTTP ingress to `stage=templated` at a median of 130 ms, with some requests at 180-207 ms, over 10 warm requests at c = 1. The engine part (queued to first SSE token) was 533 ms. Evidence is in `/workspace/exl3-short-ttft/stages-*/` on the rc share.

Against exllamav3 on the same prompt and box, our TTFT p50 is 669 ms and theirs is 595 ms (`/workspace/exl3-short-ttft/20260913-205841/`). The re-parse is more than that whole 74 ms gap.

Upstream compiles a chat template once and caches it. transformers' `_compile_jinja_template` is `lru_cache`d, and vLLM's renderer resolves the template once per model (`vllm/renderers/hf.py`). The fix is to parse once, when the prompt fn is built, and render the cached `TemplateNode` per request. That requires minja's render to be safe to call concurrently on one shared root with per-call contexts, which has to be verified in `third_party/minja/minja.hpp` rather than assumed.

## Resolution

2026-09-14: fixed by 7c5ce4eab on row/SERVE-CHAT-TEMPLATE-PARSE-CACHE. MakeChatTemplatePromptFn parses once at build and renders the shared minja TemplateNode per request; parse errors still throw per request with the same ChatTemplateError. On the Qwen3.8 template (x86 -O3) a request went from 46.19/48.03 ms to 0.031/0.041 ms with byte-identical output on six shapes. TSan and ASan/UBSan clean on 41/41. A fresh review returned PASS (M1 re-parse, M3 shared Context and M4 build-time throw killed; M2 benign; M5 error ordering survived and is recorded as untested). The operator reran test_chat_template, test_chat_prompt, test_capi, test_chat_mm, test_openai_api_server and test_openai_serving: 6/6. The GB10 end-to-end TTFT effect is measured by the EXL3 benchmark rerun, not here.
