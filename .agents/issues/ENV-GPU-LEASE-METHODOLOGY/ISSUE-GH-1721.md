ID: ISSUE-GH-1721
Title: `git stash` is repo-global across linked worktrees, so a bare `git stash pop` in one worktree consumes whatever sits at `stash@{0}` — which on this checkout belonged to another row. `git stash` on an already-clean tree saves nothing and prints nothing, but the paired `pop` still fires: it pulled `row/ENG-PUBLIC-DOC-PROJECTIONS`'s 56-file entry into an unrelated worktree, and only survived because the conflict made git KEEP the entry. 14 entries are on the stack, some labelled recovery. Same shape as #777 and #998: a resource that looks per-worktree and is per-repository
Row: ENV-GPU-LEASE-METHODOLOGY
State: UNKNOWN
Kind: bug
GitHub: 1721
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:645`

### Frozen archive evidence

> | [#1721](https://github.com/mudler/vllm.cpp/issues/1721) | `ENV-GPU-LEASE-METHODOLOGY` | `git stash` is repo-global across linked worktrees, so a bare `git stash pop` in one worktree consumes whatever sits at `stash@{0}` — which on this checkout belonged to another row. `git stash` on an already-clean tree saves nothing and prints nothing, but the paired `pop` still fires: it pulled `row/ENG-PUBLIC-DOC-PROJECTIONS`'s 56-file entry into an unrelated worktree, and only survived because the conflict made git KEEP the entry. 14 entries are on the stack, some labelled recovery. Same shape as #777 and #998: a resource that looks per-worktree and is per-repository | bug |

## Resolution

-
