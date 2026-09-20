ID: ISSUE-GH-3075
Title: fix(BACKEND-GATE-ROCM-SGLANG): isolate llama c4 input corruption
Row: BACKEND-GATE-ROCM-SGLANG
State: OPEN
Kind: UNKNOWN
GitHub: 3075
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-08
Updated: 2026-09-08
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-GATE-ROCM-SGLANG`
>
> Owner: current Strix campaign operator. Parent: #3053. Spec: .agents/specs/strix-llama-c4-input-race.md.
>
> The pinned llama.cpp 10bf611e533d81f739128304991c5e133c6aebd8 produces corrupt opening tokens for Qwen3-4B BF16 at concurrency four on gfx1151. The failure survives c4-first ordering and diagnostic graph disabling; c1 opening tokens remain sensible. Evidence: #3053 issuecomment-5586767582 and qualification-59a9c319-12/result.json, SHA256 1d518a475a79947d9921be92e9893ed74a4831359063284524b3c32a0db4f400.
>
> Implement a public-API first-prefill probe: independent prompts, combined batch, and synchronized equivalent microbatch partition. Preserve token IDs, positions, sequence IDs and logits flags. For prompt lengths [6,5,6,7], pinned split_equal implies [20,1,2,1] microbatches. Establish the failure below sampling before selecting a repair. Candidate upstream ownership: ggml-org/llama.cpp#28056, #25992 and PR #27311; these are leads, not a reproduced diagnosis. Local #2557 concerns a different CPU/Q4_K case.
>
> Acceptance: committed spec before code; test-first probe and mutation review; operator leased reproduction with source/binary/model hashes, complete logits evidence and failure status. A minimal tracked comparator patch requires reproduced buffer causality and a subsequent committed design; preserve the stock pin and label any patched comparator. No silent pin update, graph-disabled performance denominator, relaxed correctness, or speed claim. Keep the bug open until the fix lands; inconclusive diagnostics name the next hypothesis.
>

## Resolution

-
