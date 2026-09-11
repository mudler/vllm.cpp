ID: ISSUE-GH-890
Title: `ltx2_video.cpp`'s temporal-upsampler guard tested `temporal_upsample` alone, which every BOTH-flags config also satisfies, so a genuine spatiotemporal checkpoint was told it is the temporal x2 arm and the ledger refusal naming the spatiotemporal one was unreachable from a request. Narrowed to `temporal_upsample && !spatial_upsample`, with a both-flags fixture driven through `LoadVideoEngine`
Row: LTX25-RETIRE-DEAD-ARMS
State: UNKNOWN
Kind: bug
GitHub: 890
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:234`

### Frozen archive evidence

> | [#890](https://github.com/mudler/vllm.cpp/issues/890) | `LTX25-RETIRE-DEAD-ARMS` | `ltx2_video.cpp`'s temporal-upsampler guard tested `temporal_upsample` alone, which every BOTH-flags config also satisfies, so a genuine spatiotemporal checkpoint was told it is the temporal x2 arm and the ledger refusal naming the spatiotemporal one was unreachable from a request. Narrowed to `temporal_upsample && !spatial_upsample`, with a both-flags fixture driven through `LoadVideoEngine` | bug |

## Resolution

-
