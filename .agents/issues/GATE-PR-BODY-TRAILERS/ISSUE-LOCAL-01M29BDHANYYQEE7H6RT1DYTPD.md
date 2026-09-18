ID: ISSUE-LOCAL-01M29BDHANYYQEE7H6RT1DYTPD
Title: the closing-keyword gate refuses a missing local record, and that arm fires on ordinary work
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

The gate landed in a2532471b refuses a body whose closing keyword names an issue with no local record. Measured across the open queue after it landed, that arm fires on 22 of 24 refusals: 15 of 15 external pull requests, 6 of 9 bot pull requests, and it would have blocked three that legitimately landed on 2026-09-11 (#3141 citing #2909, #3138 citing #2908, #3128 citing #3019). Issues predating the local-first migration have no record at all, so the arm reports migration debt rather than a defect in the change being merged. AGENTS.md names this shape directly: a gate that fires on ordinary work is the defect, not the discipline. The blast radius was never measured before the gate landed; the measurement taken beforehand counted bare citations, which is a different denominator.

## Resolution

-
