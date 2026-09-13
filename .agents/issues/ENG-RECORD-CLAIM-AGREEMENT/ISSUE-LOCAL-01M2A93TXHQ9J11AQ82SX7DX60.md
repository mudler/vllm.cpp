ID: ISSUE-LOCAL-01M2A93TXHQ9J11AQ82SX7DX60
Title: Nothing in this tree compares two copies of one fact, and `check-agent-record.py` plus `check-model-checklist.py` stayed GREEN through ten instances of that failure in a single change. The DeepSeek-V4.1-Flash scoping change (`97cb6964b`) failed SEVEN consecutive fresh reviews, every one on the same shape -- a fact corrected in one place and left standing in another -- and SIX of the ten were introduced by the repair rounds themselves, the tenth landing in the commit message of the session that had spent six rounds hunting the other nine. Both record checkers verify structure, cardinality and anchor liveness; neither reads two documents and compares a shared value, while a fact here lives in up to six places (a checklist cell, a detail cell, counting prose, an oracle file, a spec section, an issue file). This row adds ONE mechanically sound rule chosen by measurement rather than by hope: a commit sha written in the strict form `sha` (YYYY-MM-DD must carry the same date at every site that writes it that way. Over `.agents/`, 609 shas merely share a line with some date and 29 of those disagree (mostly falsely, because a sha legitimately carries author, committer, pinned and measured dates); 144 bind a date to the sha and 4 disagree; the strict form covers 56 shas over 88 sites at base SHA e1097c5e4, re-derived 2026-09-12, and yields exactly ONE disagreement, which is real (an earlier draft said 49 over 81; the two looser figures are recorded in the spec as NOT reproducible from what it states)
Row: ENG-RECORD-CLAIM-AGREEMENT
State: OPEN
Kind: gap
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

A duplicated-fact record surface with no agreement check; ten instances in one change, all green.

## Resolution

-
