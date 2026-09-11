ID: ISSUE-LOCAL-01M27N44F915BK5QXR6HZ3NGGA
Title: check-commit-trailers rejects / in the Assisted-by model name
Row: ENG-TRAILER-SLASH-MODEL
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-11
Closed: -

## Problem

scripts/check-commit-trailers.py rejects / in the MODEL slot of Assisted-by: AGENT:MODEL [TOOL]. The MODEL character class [A-Za-z0-9][A-Za-z0-9_.+-]* has no slash. The TOOL slot of the same regex already accepts / ([A-Za-z0-9_. +:/-]). The model identifier regolo/glm5.2 uses / as the provider/model separator (same as HuggingFace meta-llama/Llama-3, OpenAI openai/gpt-4). Every commit mangles / to - to pass the gate, and two commits (0d6f068, 4f2edb4) landed with the real slash form when CI was down. GitHub mirror: #3132.

## Resolution

-
