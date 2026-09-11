ID: ISSUE-GH-467
Title: `agent-preflight.sh` prints "trailer suites ok / All gates green" over a range where `check-commit-trailers.py` reds on 3 rules: it runs the checker's own MUTATION SUITE, never the checker over the actual range, so git-generated merge messages are the one commit class nothing checks
Row: ENG-TRAILER-MERGE-ARTIFACTS
State: UNKNOWN
Kind: bug
GitHub: 467
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:73`

### Frozen archive evidence

> | [#467](https://github.com/mudler/vllm.cpp/issues/467) | `ENG-TRAILER-MERGE-ARTIFACTS` | `agent-preflight.sh` prints "trailer suites ok / All gates green" over a range where `check-commit-trailers.py` reds on 3 rules: it runs the checker's own MUTATION SUITE, never the checker over the actual range, so git-generated merge messages are the one commit class nothing checks | bug |

## Resolution

-
