ID: ISSUE-GH-1933
Title: **The GGUF arm resolved `deepseek-llm`, `deepseek-v3` and `joyai-llm` to `kLlama3` as a documented "close APPROXIMATION" while the exact pre-tokenizer for each was in the tree.** Found closing [#1924](https://github.com/mudler/vllm.cpp/issues/1924), and it contradicts that issue's own scope note ("The GGUF path is unaffected because it carries its own vocabulary") — the vocabulary is its own, the pre-tokenizer was not. `deepseek-llm` is `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM` = `kDeepSeek`, which landed at `66a44f9bf` and was never wired to the pre name; `deepseek-v3` and `joyai-llm` (plus `hunyuan-dense`, which was refused by name entirely) are `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM` = the `kDeepSeekV3` #1924 adds. NOT a rare-boundary difference: the V3 alternation binds an ASCII punctuation character to the letters after it, so `def foo(x): return x` keeps `(x` as one piece where `kLlama3` splits it, and `$var`/`_name` are one piece against two; `kDeepSeek` isolates every newline and splits digits one at a time against `kLlama3`'s groups of three, so every multi-digit number in a prompt got a different id. Same shape as [#347](https://github.com/mudler/vllm.cpp/issues/347). The artifact it bites is the DeepSeek-V4-Flash GGUF (`antirez/ds4` q2-imatrix, pre `joyai-llm`), which the old comment named by hand. `laguna` KEEPS `kLlama3` and its approximation note: llama.cpp has no `laguna` pre name, so nothing exact exists to resolve it onto, and that distinction is pinned in the test rather than left to the reader. FIXED IN FLOW with #1924, because the exact V3 pipeline half of it needs lands in the same change
Row: MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm
State: UNKNOWN
Kind: bug
GitHub: 1933
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:734`

### Frozen archive evidence

> | [#1933](https://github.com/mudler/vllm.cpp/issues/1933) | `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm` | **The GGUF arm resolved `deepseek-llm`, `deepseek-v3` and `joyai-llm` to `kLlama3` as a documented "close APPROXIMATION" while the exact pre-tokenizer for each was in the tree.** Found closing [#1924](https://github.com/mudler/vllm.cpp/issues/1924), and it contradicts that issue's own scope note ("The GGUF path is unaffected because it carries its own vocabulary") — the vocabulary is its own, the pre-tokenizer was not. `deepseek-llm` is `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM` = `kDeepSeek`, which landed at `66a44f9bf` and was never wired to the pre name; `deepseek-v3` and `joyai-llm` (plus `hunyuan-dense`, which was refused by name entirely) are `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM` = the `kDeepSeekV3` #1924 adds. NOT a rare-boundary difference: the V3 alternation binds an ASCII punctuation character to the letters after it, so `def foo(x): return x` keeps `(x` as one piece where `kLlama3` splits it, and `$var`/`_name` are one piece against two; `kDeepSeek` isolates every newline and splits digits one at a time against `kLlama3`'s groups of three, so every multi-digit number in a prompt got a different id. Same shape as [#347](https://github.com/mudler/vllm.cpp/issues/347). The artifact it bites is the DeepSeek-V4-Flash GGUF (`antirez/ds4` q2-imatrix, pre `joyai-llm`), which the old comment named by hand. `laguna` KEEPS `kLlama3` and its approximation note: llama.cpp has no `laguna` pre name, so nothing exact exists to resolve it onto, and that distinction is pinned in the test rather than left to the reader. FIXED IN FLOW with #1924, because the exact V3 pipeline half of it needs lands in the same change | bug |

## Resolution

-
