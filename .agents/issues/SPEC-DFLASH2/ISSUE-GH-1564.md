ID: ISSUE-GH-1564
Title: **The two divergent draft blocks were attributed to the SELECTOR's rank contraction, and the block shape argues against it.** W6 measured 45 of 47 draft blocks byte-identical and wrote that both flips are "the lattice op is a REDUCTION over `selector_rank`". Nothing measured that: the golden records `{call, req_row, anchor, drafts}` per block and no values, no logits and no top-2 gap, so it cannot say whether either flip was a near-tie at all, let alone in which reduction. The shape points the other way -- in BOTH blocks only slot 2 changes while slots 3-6 are byte-identical, and `src/vt/cpu/cpu_ops.cpp:3219` has step l read block row `previous`, the slot step l-1 chose, so a flipped CHILD INDEX would move the predecessor row every later step reads and four identical later slots would be four coincidences per block, twice. A different candidate ID at the SAME winning slot, a rank swap in `ComputeCandidates`' top-k over the target head's logits, produces this shape with none. `SPEC-DFLASH` D6 licenses a near-tie envelope; it does not license labelling an unmeasured flip as one nor naming the op. The attribution is WITHDRAWN in the spec and the benchmark record rather than replaced. The instrument is available and cheap: `Qwen3DFlash2Model::ComputeCandidates` already returns `(ids, values)` and so does upstream's `compute_candidates`, so the next capture records the top-2 candidate margin at the flipping slot on both sides. Blocks nothing -- the gate reads 45/47 on a majority bar and both blocks emitted the same target tokens either way
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1564
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:590`

### Frozen archive evidence

> | [#1564](https://github.com/mudler/vllm.cpp/issues/1564) | `SPEC-DFLASH2` | **The two divergent draft blocks were attributed to the SELECTOR's rank contraction, and the block shape argues against it.** W6 measured 45 of 47 draft blocks byte-identical and wrote that both flips are "the lattice op is a REDUCTION over `selector_rank`". Nothing measured that: the golden records `{call, req_row, anchor, drafts}` per block and no values, no logits and no top-2 gap, so it cannot say whether either flip was a near-tie at all, let alone in which reduction. The shape points the other way -- in BOTH blocks only slot 2 changes while slots 3-6 are byte-identical, and `src/vt/cpu/cpu_ops.cpp:3219` has step l read block row `previous`, the slot step l-1 chose, so a flipped CHILD INDEX would move the predecessor row every later step reads and four identical later slots would be four coincidences per block, twice. A different candidate ID at the SAME winning slot, a rank swap in `ComputeCandidates`' top-k over the target head's logits, produces this shape with none. `SPEC-DFLASH` D6 licenses a near-tie envelope; it does not license labelling an unmeasured flip as one nor naming the op. The attribution is WITHDRAWN in the spec and the benchmark record rather than replaced. The instrument is available and cheap: `Qwen3DFlash2Model::ComputeCandidates` already returns `(ids, values)` and so does upstream's `compute_candidates`, so the next capture records the top-2 candidate margin at the flipping slot on both sides. Blocks nothing -- the gate reads 45/47 on a majority bar and both blocks emitted the same target tokens either way | bug |

## Resolution

-
