ID: ISSUE-GH-883
Title: **GitHub reports `CONFLICTING` on `.agents/issue-index.md` while local git merges it cleanly, so the union driver #846 armed does not stop forge conflicts.** Filed 2026 from the LTX-2.5 landing campaign against PR #880 and never indexed here; its row is appended now, in the same commit as [#2317](https://github.com/mudler/vllm.cpp/issues/2317), because the issue that first observed this class was itself untracked by the surface it is about, a row-key scan for it having returned zero. It measured both directions on one case, `git merge-tree --write-tree` rc=0 with zero CONFLICT lines and `git merge --no-commit --no-ff` rc=0 with the path reported modified rather than unmerged, and its operator consequence stands unchanged: a `CONFLICTING` verdict from the forge is not evidence of a conflict, so reproduce it with a local `git merge` before acting on it. It deliberately left the mechanism unestablished ("the leading hypothesis is that GitHub computes mergeability without applying `.gitattributes` merge drivers. I did not verify that") and proposed a two-throwaway-branch experiment to settle it; #2317 settles it instead with no throwaway PRs, by running `merge-tree` and `merge-file` over the same three blobs so attribute handling is the only variable. Related #364, #595, #846, #573
Row: ENG-RECORD-CONFLICT-SURFACES
State: UNKNOWN
Kind: bug
GitHub: 883
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:899`

### Frozen archive evidence

> | [#883](https://github.com/mudler/vllm.cpp/issues/883) | `ENG-RECORD-CONFLICT-SURFACES` | **GitHub reports `CONFLICTING` on `.agents/issue-index.md` while local git merges it cleanly, so the union driver #846 armed does not stop forge conflicts.** Filed 2026 from the LTX-2.5 landing campaign against PR #880 and never indexed here; its row is appended now, in the same commit as [#2317](https://github.com/mudler/vllm.cpp/issues/2317), because the issue that first observed this class was itself untracked by the surface it is about, a row-key scan for it having returned zero. It measured both directions on one case, `git merge-tree --write-tree` rc=0 with zero CONFLICT lines and `git merge --no-commit --no-ff` rc=0 with the path reported modified rather than unmerged, and its operator consequence stands unchanged: a `CONFLICTING` verdict from the forge is not evidence of a conflict, so reproduce it with a local `git merge` before acting on it. It deliberately left the mechanism unestablished ("the leading hypothesis is that GitHub computes mergeability without applying `.gitattributes` merge drivers. I did not verify that") and proposed a two-throwaway-branch experiment to settle it; #2317 settles it instead with no throwaway PRs, by running `merge-tree` and `merge-file` over the same three blobs so attribute handling is the only variable. Related #364, #595, #846, #573 | bug |

## Resolution

-
