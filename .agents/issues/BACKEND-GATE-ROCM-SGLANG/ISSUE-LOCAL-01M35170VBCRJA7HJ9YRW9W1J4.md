ID: ISSUE-LOCAL-01M35170VBCRJA7HJ9YRW9W1J4
Title: llama.cpp oracle: document -st flag requirement for b10451
Row: BACKEND-GATE-ROCM-SGLANG
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-22
Updated: 2026-09-22
Closed: 2026-09-22

## Problem

llama-cli at b10451 enters conversation mode after generating -n tokens, spamming > prompts to stdout in a tight loop. Only -st/--single-turn causes clean exit. Without it, one run on strix:gpu0 produced a 6.2 GB log file. Discovered during Strix partial token gate work. Oracle file needs this documented.

## Resolution

PR #3270 merged as 5c71c9607. Oracle file now documents the -st flag requirement.
