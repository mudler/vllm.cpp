ID: ISSUE-LOCAL-01M2A5T7B187YJ9GMFSWAGB20P
Title: The best-match margin branch could no-op with no JUDGED note when a profile required no identity permutation
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

Before this change dsv4v_w6_compare.py:505 read 'if margin is not None and isinstance(p, dict):' with an 'elif margin is None:' note branch. When a judged profile declared a margin bound but the report carried no 'permutation' key AND permutation_identity_complete was false, both branches were skipped: nothing was appended to bad and NO JUDGED line was printed, so a declared bound silently applied to nothing. It was not reachable from main(), which always writes report['permutation'], and no shipped profile sets permutation_identity_complete false, so this was a latent shape rather than an observed defect. Found by fresh review 2026-09-12.

## Resolution

2026-09-12: FALSIFIED BY THE TREE, in the same change that found it. The F4 repair rewrote that branch as 'if profile["best_match_margin_above_bf16_rounding"]:' with an explicit 'if not isinstance(p, dict): bad.append(...)' arm and an else-note, so every path now either judges, fails, or prints a JUDGED line naming what it did. No path can reach the end of that block having said nothing. Closed with that evidence rather than re-specced.
