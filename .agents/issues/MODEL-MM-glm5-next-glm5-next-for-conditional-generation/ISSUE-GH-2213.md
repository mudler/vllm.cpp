ID: ISSUE-GH-2213
Title: **NoPE MLA and the DSA k-pool indexer — the geometry every later wave waits on.** W3 of [#1998](https://github.com/mudler/vllm.cpp/issues/1998). Two things, and each one fails quietly. (1) `MlaBlockDims::Validate` required every dimension `> 0` while `Glm5NextTextConfig.validate_architecture` REQUIRES `qk_rope_head_dim == 0` ("Expecting NoPE for the DSA attention layers"), so the two validators were exact complements over one field and no value satisfied both — O11, pinned executably in `test_glm5_next_scaffold.cpp` and now discharged: 0 is the ABSENT rotary, `head_size()` collapses to `kv_lora_rank` (512, not 576), and the block's rope branches become NOT TAKEN rather than zero-width work. Kimi-Linear is the near miss and is untouched: it keeps `qk_rope_head_dim = 64` and skips only the rotation. (2) `Glm5NextTextIndexer` scores LEARNED POOLED candidates, not raw tokens — `index_kpool` consecutive valid tokens compressed by a per-channel 4-way softmax with an intra-pool position embedding, `index_topk // index_kpool` pools selected, expanded back to raw indices, and the ragged tail appended raw and UNSCORED at width `index_topk + index_kpool - 1` = 2051. `deepseek_v4_dsa.cpp` has no pooling stage at all, so reusing it selects the wrong candidate set and yields plausible indices either way. `index_kpool` is **4** on the published artifact and 16 in the config class. Landed `src/vllm/model_executor/models/glm5_next_dsa.{h,cpp}` gated against goldens RUN out of transformers v5.16.1 at seq_len 21 vs index_topk 8 — STRICTLY past the threshold, because at or below it a top-k selects everything and the pooling is unobservable — asserting SET equality of the selected indices over 17 discriminating rows with a smallest margin of 2.58e-3. SACRED inertness proven by the six-arm DeepSeek byte-identity probe, base `150b37852` vs head, all six fingerprints identical
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: UNKNOWN
Kind: feature
GitHub: 2213
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:865`

### Frozen archive evidence

> | [#2213](https://github.com/mudler/vllm.cpp/issues/2213) | `MODEL-MM-glm5-next-glm5-next-for-conditional-generation` | **NoPE MLA and the DSA k-pool indexer — the geometry every later wave waits on.** W3 of [#1998](https://github.com/mudler/vllm.cpp/issues/1998). Two things, and each one fails quietly. (1) `MlaBlockDims::Validate` required every dimension `> 0` while `Glm5NextTextConfig.validate_architecture` REQUIRES `qk_rope_head_dim == 0` ("Expecting NoPE for the DSA attention layers"), so the two validators were exact complements over one field and no value satisfied both — O11, pinned executably in `test_glm5_next_scaffold.cpp` and now discharged: 0 is the ABSENT rotary, `head_size()` collapses to `kv_lora_rank` (512, not 576), and the block's rope branches become NOT TAKEN rather than zero-width work. Kimi-Linear is the near miss and is untouched: it keeps `qk_rope_head_dim = 64` and skips only the rotation. (2) `Glm5NextTextIndexer` scores LEARNED POOLED candidates, not raw tokens — `index_kpool` consecutive valid tokens compressed by a per-channel 4-way softmax with an intra-pool position embedding, `index_topk // index_kpool` pools selected, expanded back to raw indices, and the ragged tail appended raw and UNSCORED at width `index_topk + index_kpool - 1` = 2051. `deepseek_v4_dsa.cpp` has no pooling stage at all, so reusing it selects the wrong candidate set and yields plausible indices either way. `index_kpool` is **4** on the published artifact and 16 in the config class. Landed `src/vllm/model_executor/models/glm5_next_dsa.{h,cpp}` gated against goldens RUN out of transformers v5.16.1 at seq_len 21 vs index_topk 8 — STRICTLY past the threshold, because at or below it a top-k selects everything and the pooling is unobservable — asserting SET equality of the selected indices over 17 discriminating rows with a smallest margin of 2.58e-3. SACRED inertness proven by the six-arm DeepSeek byte-identity probe, base `150b37852` vs head, all six fingerprints identical | feature |

## Resolution

-
