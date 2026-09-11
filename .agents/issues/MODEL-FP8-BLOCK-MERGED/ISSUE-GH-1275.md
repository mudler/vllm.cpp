ID: ISSUE-GH-1275
Title: `docs/FEATURES.md:82` still described block-wise (fine-grained 128x128) FP8 as `LOADS, cannot run (#1189 M3)` and said `Prepare` refuses by name, three milestones after `281b4bc76` (#1189 M4) wired ten projections of the Qwen3.5 dense forward through `dense_fp8_block::MatmulFp8BlockScaledD` and narrowed that refusal to a device with no block-scaled GEMM. M4 updated `docs/USAGE.md` and left `docs/FEATURES.md` alone, so the two public pages disagreed and the keyed one said the arm cannot run at all. `AGENTS.md` keys `FEATURES.md` to a quantization-surface change and that is exactly what M4 was. NO GATE CATCHES IT and the reason is structural: `scripts/check-doc-checkpoint.py` keys on a row's LIFECYCLE move inside a row table, none of #1189's milestones has a roadmap row, so the checkpoint had nothing to require of any of the five landed commits; `scripts/check-public-doc-tables.py` bounds cell size and shape rather than truth. FIXED in the #1189 M6 flow, which changes the same surface again (the arm now runs nine GEMMs rather than ten, because `gate_up` and QKV merge). The general question is carried OPEN rather than answered here: a milestone-shaped issue with no roadmap row gets no doc-checkpoint coverage at all, and keying the checkpoint on something besides a row table is a checker-semantics change needing its own spec and red-first test
Row: MODEL-FP8-BLOCK-MERGED
State: UNKNOWN
Kind: bug
GitHub: 1275
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:411`

### Frozen archive evidence

> | [#1275](https://github.com/mudler/vllm.cpp/issues/1275) | `MODEL-FP8-BLOCK-MERGED` | `docs/FEATURES.md:82` still described block-wise (fine-grained 128x128) FP8 as `LOADS, cannot run (#1189 M3)` and said `Prepare` refuses by name, three milestones after `281b4bc76` (#1189 M4) wired ten projections of the Qwen3.5 dense forward through `dense_fp8_block::MatmulFp8BlockScaledD` and narrowed that refusal to a device with no block-scaled GEMM. M4 updated `docs/USAGE.md` and left `docs/FEATURES.md` alone, so the two public pages disagreed and the keyed one said the arm cannot run at all. `AGENTS.md` keys `FEATURES.md` to a quantization-surface change and that is exactly what M4 was. NO GATE CATCHES IT and the reason is structural: `scripts/check-doc-checkpoint.py` keys on a row's LIFECYCLE move inside a row table, none of #1189's milestones has a roadmap row, so the checkpoint had nothing to require of any of the five landed commits; `scripts/check-public-doc-tables.py` bounds cell size and shape rather than truth. FIXED in the #1189 M6 flow, which changes the same surface again (the arm now runs nine GEMMs rather than ten, because `gate_up` and QKV merge). The general question is carried OPEN rather than answered here: a milestone-shaped issue with no roadmap row gets no doc-checkpoint coverage at all, and keying the checkpoint on something besides a row table is a checker-semantics change needing its own spec and red-first test | bug |

## Resolution

-
