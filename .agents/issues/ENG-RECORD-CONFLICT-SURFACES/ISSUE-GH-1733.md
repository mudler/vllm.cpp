ID: ISSUE-GH-1733
Title: CLOSED as a duplicate of [#1731](https://github.com/mudler/vllm.cpp/issues/1731), filed seventeen minutes after it against the same red, and recorded here rather than left unindexed because its one MEASURED claim is FALSE and an append-only row cannot be corrected in place later. Both issues report that `.agents/issue-index.md` lists [#1649](https://github.com/mudler/vllm.cpp/issues/1649) twice on `main` at `038ff61e5`, at `:592` from `a7bb3130b` (the lane that FILED it) and at `:632` from `2f2a70925` (the lane that FIXED it). #1731 is the earlier filing, already carries an index row, and is the record. #1733 adds one thing #1731 does not, and it is wrong: "the repair is measured and it is NOT blocked", on the evidence that removing one of the two rows in a worktree and running both checkers gives `agent record OK: ENGINE=170 MODEL=377 ...` and `OK: issue index append-only`. The second half is an artefact of the instrument. `scripts/check-issue-index-append-only.py:50-51` diffs `merge-base(origin/main, HEAD)..HEAD`, which reads COMMITS, so an UNCOMMITTED deletion is invisible to it. Measured on `row/FIX-ISSUE-INDEX-1649-DUP` at base `038ff61e5`: deleting `:592` in the WORKING TREE alone returns `OK: issue index append-only` at rc 0 with `git diff --numstat 038ff61e5..HEAD -- .agents/issue-index.md` EMPTY, and committing the byte-identical deletion turns the same checker rc 1 with a `removed:` line naming the row. #1733's own quoted `agent record OK: ENGINE=170 MODEL=377` is the tell, because that is the working-tree reading and the committed tree cannot produce it while the duplicate stands. So the duplicate IS base-reachable, the two checkers ARE in genuine contradiction on this tree, and the repair is the argued exception #1731's row anticipated rather than the free edit #1733 reported
Row: ENG-RECORD-CONFLICT-SURFACES
State: UNKNOWN
Kind: bug
GitHub: 1733
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:650`

### Frozen archive evidence

> | [#1733](https://github.com/mudler/vllm.cpp/issues/1733) | `ENG-RECORD-CONFLICT-SURFACES` | CLOSED as a duplicate of [#1731](https://github.com/mudler/vllm.cpp/issues/1731), filed seventeen minutes after it against the same red, and recorded here rather than left unindexed because its one MEASURED claim is FALSE and an append-only row cannot be corrected in place later. Both issues report that `.agents/issue-index.md` lists [#1649](https://github.com/mudler/vllm.cpp/issues/1649) twice on `main` at `038ff61e5`, at `:592` from `a7bb3130b` (the lane that FILED it) and at `:632` from `2f2a70925` (the lane that FIXED it). #1731 is the earlier filing, already carries an index row, and is the record. #1733 adds one thing #1731 does not, and it is wrong: "the repair is measured and it is NOT blocked", on the evidence that removing one of the two rows in a worktree and running both checkers gives `agent record OK: ENGINE=170 MODEL=377 ...` and `OK: issue index append-only`. The second half is an artefact of the instrument. `scripts/check-issue-index-append-only.py:50-51` diffs `merge-base(origin/main, HEAD)..HEAD`, which reads COMMITS, so an UNCOMMITTED deletion is invisible to it. Measured on `row/FIX-ISSUE-INDEX-1649-DUP` at base `038ff61e5`: deleting `:592` in the WORKING TREE alone returns `OK: issue index append-only` at rc 0 with `git diff --numstat 038ff61e5..HEAD -- .agents/issue-index.md` EMPTY, and committing the byte-identical deletion turns the same checker rc 1 with a `removed:` line naming the row. #1733's own quoted `agent record OK: ENGINE=170 MODEL=377` is the tell, because that is the working-tree reading and the committed tree cannot produce it while the duplicate stands. So the duplicate IS base-reachable, the two checkers ARE in genuine contradiction on this tree, and the repair is the argued exception #1731's row anticipated rather than the free edit #1733 reported | bug |

## Resolution

-
