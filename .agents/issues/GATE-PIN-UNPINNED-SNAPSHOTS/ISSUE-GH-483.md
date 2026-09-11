ID: ISSUE-GH-483
Title: `check-snapshot-pins` is evaded by ordinary modern-C++ punctuation, not by the deliberate escapes: `directory_iterator{snaps}` (brace init), `root /= "snapshots"` (compound assignment), and a lambda body severed by the `[;{}]` statement split
Row: GATE-PIN-UNPINNED-SNAPSHOTS
State: UNKNOWN
Kind: bug
GitHub: 483
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:134`

### Frozen archive evidence

> | [#483](https://github.com/mudler/vllm.cpp/issues/483) | `GATE-PIN-UNPINNED-SNAPSHOTS` | `check-snapshot-pins` is evaded by ordinary modern-C++ punctuation, not by the deliberate escapes: `directory_iterator{snaps}` (brace init), `root /= "snapshots"` (compound assignment), and a lambda body severed by the `[;{}]` statement split | bug |

## Resolution

-
