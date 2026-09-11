ID: ISSUE-GH-490
Title: Qwen3.5/3.8 text-only checkpoints cannot load: `Qwen3_5MoeForCausalLM` unregistered and the loader hardcodes the VL `model.language_model.` prefix. Registration + one-per-checkpoint namespace resolution LANDED 2026-08-12 (sibling row `MODEL-TEXT-qwen3-5-qwen3-5-for-causal-lm`); the issue stays OPEN because the RUN GATE is owed — no text-only Qwen3.5 checkpoint fits this hardware, AND (corrected 2026-08-12) the published MoE repos ship 3-D stacked, unquantized experts that this loader does not implement, so the MoE gate does not close on a fitting checkpoint alone; that arm is owed and is refused by name
Row: MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm
State: UNKNOWN
Kind: feature
GitHub: 490
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:113`

### Frozen archive evidence

> | [#490](https://github.com/mudler/vllm.cpp/issues/490) | `MODEL-TEXT-qwen3-5-qwen3-5-moe-for-causal-lm` | Qwen3.5/3.8 text-only checkpoints cannot load: `Qwen3_5MoeForCausalLM` unregistered and the loader hardcodes the VL `model.language_model.` prefix. Registration + one-per-checkpoint namespace resolution LANDED 2026-08-12 (sibling row `MODEL-TEXT-qwen3-5-qwen3-5-for-causal-lm`); the issue stays OPEN because the RUN GATE is owed — no text-only Qwen3.5 checkpoint fits this hardware, AND (corrected 2026-08-12) the published MoE repos ship 3-D stacked, unquantized experts that this loader does not implement, so the MoE gate does not close on a fitting checkpoint alone; that arm is owed and is refused by name | feature |

## Resolution

-
