ID: ISSUE-GH-1247
Title: MiniMax-Music3 depth-stage RECORD, four defects, no measured number affected: (a) `test_minimax_music3_ar.cpp`, spec §15.5 and PR #1238's body all say the committed goldens are "8-wide with ONE head" so a head stride is invisible — `minimax_music3_ar_goldens.inc` sets `kMusic3DepthHeads = 2` over `kMusic3DepthHidden = 8`, so they are 2 heads of 4 and dropping the head stride from the cached KEY index reds 4 cases / 32 assertions, TWO of them at the goldens' own geometry; (b) "12 timed rounds per arm" and "all 20 runs printed `f0cfeed6eee4f55d`" cannot be reconciled with the record's own table — 5 pairs x 1 round + 3 pairs x 4 rounds is 17 rounds per arm, and 8 pairs x 2 arms is 16 processes, each printing ONE fingerprint after its round loop; (c) the heading named base `origin/main` `727163997` while the body named `fc163f62b`, and only the latter was built and timed (the delta is #1231's profiler, which the driver never enters); (d) "70 rows a frame to read 16" conflates the 14 rows the OLD arm read with the AFTER arm's 16. FIXED IN FLOW, owned by `MUSIC3-DEPTH-SPEED`, spec §15.2, §15.3, §15.5, §15.6.
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 1247
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:421`

### Frozen archive evidence

> | [#1247](https://github.com/mudler/vllm.cpp/issues/1247) | `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation` | MiniMax-Music3 depth-stage RECORD, four defects, no measured number affected: (a) `test_minimax_music3_ar.cpp`, spec §15.5 and PR #1238's body all say the committed goldens are "8-wide with ONE head" so a head stride is invisible — `minimax_music3_ar_goldens.inc` sets `kMusic3DepthHeads = 2` over `kMusic3DepthHidden = 8`, so they are 2 heads of 4 and dropping the head stride from the cached KEY index reds 4 cases / 32 assertions, TWO of them at the goldens' own geometry; (b) "12 timed rounds per arm" and "all 20 runs printed `f0cfeed6eee4f55d`" cannot be reconciled with the record's own table — 5 pairs x 1 round + 3 pairs x 4 rounds is 17 rounds per arm, and 8 pairs x 2 arms is 16 processes, each printing ONE fingerprint after its round loop; (c) the heading named base `origin/main` `727163997` while the body named `fc163f62b`, and only the latter was built and timed (the delta is #1231's profiler, which the driver never enters); (d) "70 rows a frame to read 16" conflates the 14 rows the OLD arm read with the AFTER arm's 16. FIXED IN FLOW, owned by `MUSIC3-DEPTH-SPEED`, spec §15.2, §15.3, §15.5, §15.6. | bug |

## Resolution

-
