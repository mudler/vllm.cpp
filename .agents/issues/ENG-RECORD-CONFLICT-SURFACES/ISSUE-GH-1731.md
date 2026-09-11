ID: ISSUE-GH-1731
Title: `check-agent-record` is RED on `main`: `#1649` is listed twice in `.agents/issue-index.md`, at `:592` (added by `a7bb3130b`, the lane that FILED it) and `:632` (added by `2f2a70925`, the lane that FIXED it). Neither is wrong and neither could see the other -- `merge=union` combines two appends silently, so a duplicate is the ordinary outcome when filing and fixing happen on different branches, and the checker only notices once both have landed on main, where every later PR inherits the red. Not repaired in flow: the index preamble forbids editing or deleting a row, and the two bodies carry different facts (`:632` has a second red and the mutation evidence, `:592` has the attribution to `a50c57d69`), so choosing which survives is a judgement
Row: ENG-RECORD-CONFLICT-SURFACES
State: UNKNOWN
Kind: bug
GitHub: 1731
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:647`

### Frozen archive evidence

> | [#1731](https://github.com/mudler/vllm.cpp/issues/1731) | `ENG-RECORD-CONFLICT-SURFACES` | `check-agent-record` is RED on `main`: `#1649` is listed twice in `.agents/issue-index.md`, at `:592` (added by `a7bb3130b`, the lane that FILED it) and `:632` (added by `2f2a70925`, the lane that FIXED it). Neither is wrong and neither could see the other -- `merge=union` combines two appends silently, so a duplicate is the ordinary outcome when filing and fixing happen on different branches, and the checker only notices once both have landed on main, where every later PR inherits the red. Not repaired in flow: the index preamble forbids editing or deleting a row, and the two bodies carry different facts (`:632` has a second red and the mutation evidence, `:592` has the attribution to `a50c57d69`), so choosing which survives is a judgement | bug |

## Resolution

-
