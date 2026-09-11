ID: ISSUE-GH-699
Title: dots3-note (280B-A16B multimodal MoE, text+image+video+audio): scope the port. Upstream SUBCLASSES DeepSeek — `Dots3NoteModel(DeepseekV32Model)`, `Dots3NoteMoE(DeepseekV2MoE)` — so our gated MLA, DSA indexer, `noaux_tc` router, Qwen3-VL vision and Voxtral audio carry most of it; net-new is sliding-window MLA over 33 of 46 layers with a SECOND latent geometry (576 vs 1088 rows), the padded/heterogeneous MLA KV spec, the headwise attention gate, the pyramid MoE ViT and the `dots` audio stem. Two blockers, both recorded rather than worked around: BEYOND-PIN (vLLM `main` only, #51255, still being patched) and ORACLE-MEMORY-INFEASIBLE (~576 GB bf16 / ~290 GB fp8 against a 119-122 GiB ceiling on every host we own, and no smaller checkpoint exists)
Row: MODEL-MM-dots3-note-dots3-note-for-causal-lm
State: UNKNOWN
Kind: feature
GitHub: 699
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:195`

### Frozen archive evidence

> | [#699](https://github.com/mudler/vllm.cpp/issues/699) | `MODEL-MM-dots3-note-dots3-note-for-causal-lm` | dots3-note (280B-A16B multimodal MoE, text+image+video+audio): scope the port. Upstream SUBCLASSES DeepSeek — `Dots3NoteModel(DeepseekV32Model)`, `Dots3NoteMoE(DeepseekV2MoE)` — so our gated MLA, DSA indexer, `noaux_tc` router, Qwen3-VL vision and Voxtral audio carry most of it; net-new is sliding-window MLA over 33 of 46 layers with a SECOND latent geometry (576 vs 1088 rows), the padded/heterogeneous MLA KV spec, the headwise attention gate, the pyramid MoE ViT and the `dots` audio stem. Two blockers, both recorded rather than worked around: BEYOND-PIN (vLLM `main` only, #51255, still being patched) and ORACLE-MEMORY-INFEASIBLE (~576 GB bf16 / ~290 GB fp8 against a 119-122 GiB ceiling on every host we own, and no smaller checkpoint exists) | feature |

## Resolution

-
