ID: ISSUE-GH-1872
Title: **`align.audio_lag` is a bare argmax with NO margin, and on real frames it fires on a correlation difference of `4.5e-05`.** Found while measuring [#1855](https://github.com/mudler/vllm.cpp/issues/1855)'s audio direction on the frames [#1612](https://github.com/mudler/vllm.cpp/issues/1612) rendered, with no GPU and no lease. `flash` against `baseline-20260820` reads `best lag -1 samples` on `r = 0.926353` there against `0.926308` at lag 0 - **one sample of 96,480 at 48 kHz, 20.8 microseconds** - and the whole run reads `READING MISALIGNED` and exits 1 on it, which §11.6 defines as the state where the pair is not comparable at all. **The asymmetry is the defect.** `align.frames` was written WITH a bound - `margin > 1`, computed by `frame_correspondence` and printed in the report - precisely because "nearest" without "by how much" is not a correspondence. `audio_correspondence` computes the same shape of quantity, returns `best_lag` plus `r` at the best lag AND at 0, and the check then compares only the integer, over a `+/-2000` sweep of 4001 float candidates with no tie handling. **NOT a case for widening a threshold** (#1668 and §9 forbid that): the repair ADDS a bound that does not exist rather than moving one that does. **NOT REPAIRED IN FLOW**, because it changes checker semantics and therefore owes its own row, spec section, a red-before test on both a genuinely shifted track and a hairline tie, and a fresh review. **No published verdict moves**: the same-binary `flash` vs `naive` pair of §10.7 reads `best lag 0` and passes, so #1743 and #1855 stand. Owned by row `LTX25-DIT-ATTN-FLASH` and listed under `## Owed`
Row: LTX25-DIT-ATTN-FLASH
State: UNKNOWN
Kind: bug
GitHub: 1872
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:713`

### Frozen archive evidence

> | [#1872](https://github.com/mudler/vllm.cpp/issues/1872) | `LTX25-DIT-ATTN-FLASH` | **`align.audio_lag` is a bare argmax with NO margin, and on real frames it fires on a correlation difference of `4.5e-05`.** Found while measuring [#1855](https://github.com/mudler/vllm.cpp/issues/1855)'s audio direction on the frames [#1612](https://github.com/mudler/vllm.cpp/issues/1612) rendered, with no GPU and no lease. `flash` against `baseline-20260820` reads `best lag -1 samples` on `r = 0.926353` there against `0.926308` at lag 0 - **one sample of 96,480 at 48 kHz, 20.8 microseconds** - and the whole run reads `READING MISALIGNED` and exits 1 on it, which §11.6 defines as the state where the pair is not comparable at all. **The asymmetry is the defect.** `align.frames` was written WITH a bound - `margin > 1`, computed by `frame_correspondence` and printed in the report - precisely because "nearest" without "by how much" is not a correspondence. `audio_correspondence` computes the same shape of quantity, returns `best_lag` plus `r` at the best lag AND at 0, and the check then compares only the integer, over a `+/-2000` sweep of 4001 float candidates with no tie handling. **NOT a case for widening a threshold** (#1668 and §9 forbid that): the repair ADDS a bound that does not exist rather than moving one that does. **NOT REPAIRED IN FLOW**, because it changes checker semantics and therefore owes its own row, spec section, a red-before test on both a genuinely shifted track and a hairline tie, and a fresh review. **No published verdict moves**: the same-binary `flash` vs `naive` pair of §10.7 reads `best lag 0` and passes, so #1743 and #1855 stand. Owned by row `LTX25-DIT-ATTN-FLASH` and listed under `## Owed` | bug |

## Resolution

-
