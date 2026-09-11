ID: ISSUE-GH-1270
Title: `check-agent-record.py --write-baseline` returned the moment it had a number, which is before the `if errors:` gate at the end of `main`. A tree that failed any OTHER record check could therefore still write `scripts/record-anchor-baseline.json`, and the banked figure then carried the authority of a run that never passed, on the one file whose purpose is a number nobody may quietly raise. Found while re-deriving the baseline after merging 161 commits of `main` into #851, and FIXED in that flow: the write moves below the error gate, held by `RecordAnchorRatchet.test_a_baseline_is_never_banked_from_a_tree_with_record_errors`, captured red before the change and red again when the early return is restored
Row: ENG-RECORD-ANCHOR-RATCHET
State: UNKNOWN
Kind: bug
GitHub: 1270
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:407`

### Frozen archive evidence

> | [#1270](https://github.com/mudler/vllm.cpp/issues/1270) | `ENG-RECORD-ANCHOR-RATCHET` | `check-agent-record.py --write-baseline` returned the moment it had a number, which is before the `if errors:` gate at the end of `main`. A tree that failed any OTHER record check could therefore still write `scripts/record-anchor-baseline.json`, and the banked figure then carried the authority of a run that never passed, on the one file whose purpose is a number nobody may quietly raise. Found while re-deriving the baseline after merging 161 commits of `main` into #851, and FIXED in that flow: the write moves below the error gate, held by `RecordAnchorRatchet.test_a_baseline_is_never_banked_from_a_tree_with_record_errors`, captured red before the change and red again when the early return is restored | bug |

## Resolution

-
