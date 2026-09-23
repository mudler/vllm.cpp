ID: ISSUE-LOCAL-01M36QDYMKYFM1XX245X0ZZ6YG
Title: clang-17 HIP: pragma unroll fails on >>= step in PagedAttnDecodeSplitKvStage1
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

clang-17's HIP optimizer cannot satisfy #pragma unroll on the warp reduction loop in PagedAttnDecodeSplitKvStage1 because it uses offset >>= 1 instead of offset /= 2. The /= 2 pattern is used successfully elsewhere in the same file (lines 877, 2191). With -Werror this becomes a hard build error, blocking the Strix gfx1151 build. Fix: change >>= 1 to /= 2 to match the working pattern.

## Resolution

Fixed by changing offset >>= 1 to offset /= 2 in PagedAttnDecodeSplitKvStage1 warp reduction loop, matching the pattern used at lines 877 and 2191 in the same file.
