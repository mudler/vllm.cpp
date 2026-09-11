ID: ISSUE-GH-398
Title: sm_110: response_format json_schema returns 1 token + finish_reason=error (json_object also flagged error); reproduced on pristine main 0eb049f7
Row: ENG-STRUCTURED-OUTPUT
State: OPEN
Kind: UNKNOWN
GitHub: 398
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-11
Updated: 2026-08-11
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> On sm_110 (Jetson AGX Thor), a `response_format: json_schema` request returns **one token and `finish_reason: "error"`**. All 15 of our JSON schemas fail identically. Unconstrained generation on the same build and model is healthy.
>
> ## Three-way isolation
>
> Same server process, same model, same prompt, only `response_format` varying:
>
> | request | `finish_reason` | tokens | content |
> |---|---|---|---|
> | `response_format: {type: json_schema, strict: true, ...}` | **`error`** | **1** | `"` |
> | `response_format: {type: json_object}` | **`error`** | 11 | `{\n  "greeting": "Hello!"\n}` -- **valid JSON** |
> | *(no `response_format`)* | `stop` | 29 | `Zdręś, jak to jest w Polsce. W Polsce mówimy: **"Cześć"**.` |
>
> The middle row is the one that localises this. The **same model on the same build emits well-formed JSON** when asked for `json_object` -- so this is not the model failing to produce JSON. But that request is *also* flagged `finish_reason: "error"`, so the error is raised whenever `response_format` is present at all; the `json_schema` path additionally terminates after one token.
>
> ## Reproduction
>
> ```
> git clone --depth 1 https://github.com/mudler/vllm.cpp.git
> # HEAD 0eb049f7e3fe522a1e8763c59be5bcfbbab53139, matches origin/HEAD
> # git status --porcelain -> empty, git diff --stat -> empty, 0 untracked
>
> cmake -S . -B build-110 -GNinja -DCMAKE_BUILD_TYPE=Release \
>       -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF
> ninja -C build-110 examples/vllm-server        # 237 s, 403 TUs, 0 warnings, 0 errors
>
> ./build-110/examples/vllm-server --model <qwen3-1.7b-nvfp4a16> \
>       --host 127.0.0.1 --port 30890 --max-model-len 4096 \
>       --max-num-seqs 8 --num-blocks 256
> ```
>
> Model: `apolloparty/Qwen3-1.7B-NVFP4A16`. Device: Jetson AGX Thor, sm_110, CUDA 13.2, driver-only host, server run inside `nvcr.io`-derived image as an MPS client.
>
> Across our 15 committed JSON schemas issued as `response_format: json_schema strict`: **15/15 fail**, every one `Expecting value: line 1 column 1 (char 0)` with `completion_tokens=1`. The harness self-test confirms all 15 schemas are discriminating (unconstrained output is rejected by each), so the checker is not vacuous.
>
> The server logs nothing beyond the request summary -- `finish_reason=error`, no message, no exception:
>
> ```
> INFO Finished request chatcmpl-30 prompt_tokens=16 completion_tokens=1 \
>      total_tokens=17 finish_reason=error elapsed_s=0.00927641
> ```
>
> ## Why I believe this is yours and not mine
>
> I first saw this on a working tree carrying ~1,100 lines of our own local modification, **~634 of them in `src/vllm/tokenizer/`** (pretokenizer, unicode_data). A grammar backend keys on the token vocabulary, so patched pretokenizer internals are a very plausible way to break constrained decoding while leaving unconstrained generation healthy -- exactly the symptom shape here. I did not report it then.
>
> I rebuilt from the pristine clone above and **the behaviour is byte-for-byte identical**: 1 token, `finish_reason: "error"`, 15/15 schemas failing, `json_object` still returning valid JSON under an `error` finish reason. Our tokenizer work is not the cause.
>
> ## Against the documented contract
>
> `docs/STATUS.md:127`, quoted without elision because the caveat in it is yours and belongs in the quote:
>
> > Structured output | **Supported (subset), engine-enforced; xgrammar backend W1 (CPU, not yet production-wired)** | JSON schema, JSON object, regex, choice, GBNF grammar. Constrained decoding runs in the production engine (native grammar backend, per-step logits bitmask) and is reachable from OpenAI `response_format` and the C ABI (ABI v2 `structured_*` fields).
>
> The "not yet production-wired" caveat is about the **xgrammar** backend specifically. The path in play here is the native grammar backend, which the same line says runs in the production engine and is reachable from `response_format` -- which is exactly the request that fails.
>
> `docs/FEATURES.md:275` -- `Structured output / grammars | structured_json, structured_grammar | reachable`.
>
> ## A question rather than a claim
>
> **Has structured output been exercised on sm_110, or only on sm_12x?** I ask because two other features turned out to be arch-gated in ways not obvious from the call site: `marlin-nvfp4` excluded `11.0` until #326, and `fa2` still reads `8.0,8.6,8.7,8.9,12.0a,12.1a` (`cmake/CudaArchFeatures.cmake:349`), so `VLLM_CPP_FLASH_ATTN` is undefined on this arch and the whole FA2 block compiles out. An arch-conditional path in the grammar/bitmask machinery would fit the pattern -- but I have not read that code and am not asserting it. If the expectation is that structured output is arch-independent, then this is a plain defect and the arch is incidental.
>
> ## What I have not done
>
> I have not instrumented the engine, read the grammar backend, or bisected. This is a reproduction on a clean tree with the failure localised to the `response_format` path, not a root cause. Happy to run any diagnostic you want on this hardware -- including a patched build with logging in the bitmask path, which is cheap here (full single-arch build is under four minutes).
>

## Resolution

-
