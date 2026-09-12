ID: ISSUE-LOCAL-01M29SAK6KRP61VPFCH6FMA45J
Title: DeepSeek-V4 vision: FEATURES.md still says the real mmproj-BF16.gguf has never been read, which the row's own spec falsifies at the same head
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

docs/FEATURES.md line 185 reads 'The real mmproj-BF16.gguf has never been read (W6)'. At the same head .agents/specs/deepseek-v4-flash-vision.md records the opposite under W6: the 934,462,656-byte projector was loaded and run through the shipped path on thor:gpu0, and its vision tower was compared against llama.cpp b10766 at four lead_pad rungs plus the CLI's own dump. A public document therefore contradicts the row's evidence section, in the direction that understates what the tree can do. Found by fresh review 2026-09-12.

## Resolution

2026-09-12: fixed in this change. docs/FEATURES.md line 185 now records that the real mmproj-BF16.gguf HAS been read and run at W6 -- the 934,462,656-byte projector through the shipped path, matching llama.cpp b10766 with byte-exact sentinels and the identity permutation best for 100 of 100 image rows, with the residual attributed by W6's dtype test to bf16 intermediate storage rather than to a defect -- in place of 'has never been read (W6)'. The rest of that cell, which is accurate, is untouched.
