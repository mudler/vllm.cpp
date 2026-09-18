ID: ISSUE-LOCAL-01M298Q7PPM145DQA5C30GRYJA
Title: agent-pr-body.py accepts a body whose Closes #N contradicts the canonical local record
Row: GATE-PR-BODY-TRAILERS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-11
Closed: -

## Problem

The repository sets squash_merge_commit_message = PR_BODY, so a body's 'Closes #N' closes the GitHub mirror the moment the squash lands. Local files under .agents/issues are the issue authority. Nothing checked that the branch also carries ISSUE-GH-N.md reading State: CLOSED, so the two authorities diverge at the merge and the authoritative half is left reading OPEN. Observed on four pull requests in one pass: #3101 (Closes #3098 while the record read OPEN), #3095 (Closes #3092 while its own spec gate table read FAILING), #3096 (five closing keywords against four OPEN records and one issue with no local record at all), and #3097 (Closes #3093 against a spec saying the row is not ready to land). check-agent-record.py cannot catch it: it is offline and never sees a pull request body, and it silently continues on an unresolvable bare #N.

## Resolution

-
