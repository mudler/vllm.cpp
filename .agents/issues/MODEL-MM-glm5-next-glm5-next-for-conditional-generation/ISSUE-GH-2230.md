ID: ISSUE-GH-2230
Title: **Three refusal messages named LANDED waves as owing, and one denied an artifact that exists — and the gate was PINNING all three.** Fixed IN FLOW under W5 of [#2223](https://github.com/mudler/vllm.cpp/issues/2223). (1) The forward refusal read "W3 the NoPE MLA block -- `MlaBlockDims::Validate` still refuses `qk_rope_head_dim == 0`", which W3 (#2213, `e511a614b`) made false by relaxing exactly that validator; it named W2's sigmoid forget gate and W4's unweighted mHC collapse as owed too, both landed (`199c44578`, `6c715de00`). W1 wrote the message and no later wave touched the file — `git log --oneline -- src/vllm/model_executor/models/glm5_next_registry.cpp` ends at W1's `47a2b35a5`. (2) The GGUF loader refusal read "NO `.gguf` of this model exists anywhere ... (O7)"; `unsloth/GLM-5.3-Flash-GGUF` rev `d425e572f` is published and four arms are staged. (3) The KV-cache refusal said the KDA layers carry "three separate conv states"; they carry ONE — the checkpoint's `self_attn.{q,k,v}_conv1d` concatenate into one grouped depthwise conv (`modeling_glm5_next.py:620-628`, `glm5_next_kda.h` "THREE LAYOUT FACTS"), and a spec written from that sentence would triple the group. THE MECHANISM: `test_glm5_next_scaffold.cpp` asserted all three sentences, so the gate passed *because* nothing had corrected them — a refusal message is this row's only user-visible surface and its assertions were pinning stale text rather than checking it. The repair adds the NEGATIVES (`MlaBlockDims::Validate still refuses` and `NO `.gguf` of this model exists` must NOT appear) so a revision that reintroduces either reds
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 2230
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:867`

### Frozen archive evidence

> | [#2230](https://github.com/mudler/vllm.cpp/issues/2230) | `MODEL-MM-glm5-next-glm5-next-for-conditional-generation` | **Three refusal messages named LANDED waves as owing, and one denied an artifact that exists — and the gate was PINNING all three.** Fixed IN FLOW under W5 of [#2223](https://github.com/mudler/vllm.cpp/issues/2223). (1) The forward refusal read "W3 the NoPE MLA block -- `MlaBlockDims::Validate` still refuses `qk_rope_head_dim == 0`", which W3 (#2213, `e511a614b`) made false by relaxing exactly that validator; it named W2's sigmoid forget gate and W4's unweighted mHC collapse as owed too, both landed (`199c44578`, `6c715de00`). W1 wrote the message and no later wave touched the file — `git log --oneline -- src/vllm/model_executor/models/glm5_next_registry.cpp` ends at W1's `47a2b35a5`. (2) The GGUF loader refusal read "NO `.gguf` of this model exists anywhere ... (O7)"; `unsloth/GLM-5.3-Flash-GGUF` rev `d425e572f` is published and four arms are staged. (3) The KV-cache refusal said the KDA layers carry "three separate conv states"; they carry ONE — the checkpoint's `self_attn.{q,k,v}_conv1d` concatenate into one grouped depthwise conv (`modeling_glm5_next.py:620-628`, `glm5_next_kda.h` "THREE LAYOUT FACTS"), and a spec written from that sentence would triple the group. THE MECHANISM: `test_glm5_next_scaffold.cpp` asserted all three sentences, so the gate passed *because* nothing had corrected them — a refusal message is this row's only user-visible surface and its assertions were pinning stale text rather than checking it. The repair adds the NEGATIVES (`MlaBlockDims::Validate still refuses` and `NO `.gguf` of this model exists` must NOT appear) so a revision that reintroduces either reds | bug |

## Resolution

-
