ID: ISSUE-GH-2291
Title: **The W7a converter and the published artifact disagree on three tensors, and one of the three is a silent value transform.** Found while implementing W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)), whose own scope sentence assumed they agreed, and fixed in the same flow. Read at source from llama.cpp PR [#27752](https://github.com/ggml-org/llama.cpp/pull/27752) head `8a8d0bcc4d5fdf024c457526245bec4bc3a12adc` (`conversion/glm5next.py`, sha256 `bfacba27746096e7bb3ca4a2549c9026d3475e226c7f3edf230c37ffadc7b6b3`) plus the `DeepseekV2Model` it inherits, and confirmed against the staged UD-Q2_K_XL header table. (1) `.dt_bias` is RENAMED to `.dt_proj.bias` before the generic map runs, so the file carries `blk.N.ssm_dt.bias` and no bare `ssm_dt`. (2) `kv_b_proj` is SPLIT into `attn_k_b` and `attn_v_b` with the k half TRANSPOSED, so the file carries two tensors at DIFFERENT shapes — ne `[256, 512, 64]` and `[512, 256, 64]` — and no `attn_kv_b.weight`; because `qk_nope_head_dim == v_head_dim == 256`, a fixture at equal head dims cannot tell a correct split from a swapped one, so the gate asserts both SHAPES and the nearer-own-half property rather than sizes. (3) `ssm_a` holds `-exp(A_log)`, not `A_log` — the dangerous one, because the tensor is present, the shape is right and the values are plausible floats, so nothing structural fires: a loader that inverts gets NaN on every KDA decay, one that does not runs a sign-flipped forget gate and generates fluent wrong text, and no oracle for this model runs on any device this project reaches to tell the difference. Fixed in the converter, in the C++ name map (the split needs its own 1:1 table, since one HF name maps to two GGUF names and a dict cannot carry one key twice) and in the new loader, which refuses a non-negative `ssm_a` by name. `tests/scripts/test_convert_glm5_next_gguf.py` was RED on the tensor set before the converter moved
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 2291
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:890`

### Frozen archive evidence

> | [#2291](https://github.com/mudler/vllm.cpp/issues/2291) | `MODEL-MM-glm5-next-glm5-next-for-conditional-generation` | **The W7a converter and the published artifact disagree on three tensors, and one of the three is a silent value transform.** Found while implementing W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)), whose own scope sentence assumed they agreed, and fixed in the same flow. Read at source from llama.cpp PR [#27752](https://github.com/ggml-org/llama.cpp/pull/27752) head `8a8d0bcc4d5fdf024c457526245bec4bc3a12adc` (`conversion/glm5next.py`, sha256 `bfacba27746096e7bb3ca4a2549c9026d3475e226c7f3edf230c37ffadc7b6b3`) plus the `DeepseekV2Model` it inherits, and confirmed against the staged UD-Q2_K_XL header table. (1) `.dt_bias` is RENAMED to `.dt_proj.bias` before the generic map runs, so the file carries `blk.N.ssm_dt.bias` and no bare `ssm_dt`. (2) `kv_b_proj` is SPLIT into `attn_k_b` and `attn_v_b` with the k half TRANSPOSED, so the file carries two tensors at DIFFERENT shapes — ne `[256, 512, 64]` and `[512, 256, 64]` — and no `attn_kv_b.weight`; because `qk_nope_head_dim == v_head_dim == 256`, a fixture at equal head dims cannot tell a correct split from a swapped one, so the gate asserts both SHAPES and the nearer-own-half property rather than sizes. (3) `ssm_a` holds `-exp(A_log)`, not `A_log` — the dangerous one, because the tensor is present, the shape is right and the values are plausible floats, so nothing structural fires: a loader that inverts gets NaN on every KDA decay, one that does not runs a sign-flipped forget gate and generates fluent wrong text, and no oracle for this model runs on any device this project reaches to tell the difference. Fixed in the converter, in the C++ name map (the split needs its own 1:1 table, since one HF name maps to two GGUF names and a dict cannot carry one key twice) and in the new loader, which refuses a non-negative `ssm_a` by name. `tests/scripts/test_convert_glm5_next_gguf.py` was RED on the tensor set before the converter moved | bug |

## Resolution

-
