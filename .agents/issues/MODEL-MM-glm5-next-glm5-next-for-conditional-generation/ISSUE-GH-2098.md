ID: ISSUE-GH-2098
Title: **GLM-5.3-Flash's mHC head collapse is an unweighted mean, and DeepSeek-V4's gated `HcHeadCollapse` is the wrong final projection.** W4 of [#1998](https://github.com/mudler/vllm.cpp/issues/1998). `Glm5NextTextHyperHead.forward` is `hidden_streams.mean(dim=2)` and its own docstring says "Unlike DeepSeek-V4" (`modular_glm5_next.py:368-372` @ transformers v5.16.1); the checkpoint carries no `hc_head.*` tensor at any layer, so there are no weights a gated collapse could read. The other three mHC pieces ARE V4's and are reused. Landed `src/vllm/model_executor/models/glm5_next_mhc.{h,cpp}` gated against goldens RUN out of the pinned reference
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: UNKNOWN
Kind: feature
GitHub: 2098
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:857`

### Frozen archive evidence

> | [#2098](https://github.com/mudler/vllm.cpp/issues/2098) | `MODEL-MM-glm5-next-glm5-next-for-conditional-generation` | **GLM-5.3-Flash's mHC head collapse is an unweighted mean, and DeepSeek-V4's gated `HcHeadCollapse` is the wrong final projection.** W4 of [#1998](https://github.com/mudler/vllm.cpp/issues/1998). `Glm5NextTextHyperHead.forward` is `hidden_streams.mean(dim=2)` and its own docstring says "Unlike DeepSeek-V4" (`modular_glm5_next.py:368-372` @ transformers v5.16.1); the checkpoint carries no `hc_head.*` tensor at any layer, so there are no weights a gated collapse could read. The other three mHC pieces ARE V4's and are reused. Landed `src/vllm/model_executor/models/glm5_next_mhc.{h,cpp}` gated against goldens RUN out of the pinned reference | feature |

## Resolution

-
