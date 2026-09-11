ID: ISSUE-GH-598
Title: `test_minimax_h3` is nibble-BLIND: 44 live `DequantNvfp4ToBf16` calls and 57,395 assertions all stay green with the fp4 nibble order flipped, so H3's load-time swap and the new shared `kHighFirst` parameter can compose into a double flip undetected (spec `specs/nvfp4-nibble-order.md` §5.5)
Row: ROAD-V1-H3
State: UNKNOWN
Kind: bug
GitHub: 598
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:130`

### Frozen archive evidence

> | [#598](https://github.com/mudler/vllm.cpp/issues/598) | `ROAD-V1-H3` | `test_minimax_h3` is nibble-BLIND: 44 live `DequantNvfp4ToBf16` calls and 57,395 assertions all stay green with the fp4 nibble order flipped, so H3's load-time swap and the new shared `kHighFirst` parameter can compose into a double flip undetected (spec `specs/nvfp4-nibble-order.md` §5.5) | bug |

## Resolution

-
