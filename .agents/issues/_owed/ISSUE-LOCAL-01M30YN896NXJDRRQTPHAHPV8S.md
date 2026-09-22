ID: ISSUE-LOCAL-01M30YN896NXJDRRQTPHAHPV8S
Title: Server reference contradicts prompt logprob responses
Row: -
State: CLOSED
Kind: docs
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-21
Updated: 2026-09-21
Closed: 2026-09-21

## Problem

The server reference says HTTP responses omit prompt_logprobs, although both serving handlers serialize it and existing HTTP tests cover it. The usage guide also overstates the streaming refusal and implies no generation.

## Resolution

21 September 2026: documentation corrected at c2d11528ef7aec14139e6f32ea20efbaa9544dad and independently reviewed PASS. CPU document checks, recipe JSON, local links, and scratch mutations pass. Runtime behavior is unchanged; full preflight remains unavailable in the minimal container. See .agents/specs/docs-prompt-logprobs-reference.md Outcome.
