ID: ISSUE-LOCAL-01M30YN896NXJDRRQTPHAHPV8S
Title: Server reference contradicts prompt logprob responses
Row: -
State: OPEN
Kind: docs
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-21
Updated: 2026-09-21
Closed: -

## Problem

The server reference says HTTP responses omit prompt_logprobs, although both serving handlers serialize it and existing HTTP tests cover it. The usage guide also overstates the streaming refusal and implies no generation.

## Resolution

-
