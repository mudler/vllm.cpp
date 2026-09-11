ID: ISSUE-GH-660
Title: `check-device-leakage`'s `kcuda` bucket is the token grep `\bkCUDA\b`, so `minimax_h3_video.cpp:221-226`'s `static_cast<vt::DeviceType>(device)` hardcodes CUDA as enum value 1 and counts as 0. Gate strength plus an enum-ordering hazard; the H3 video lane should ask the same seam `ltx2_video.cpp` now does (found while reviewing #553)
Row: LTX25-DEVICE-SEAM-SIBLING
State: UNKNOWN
Kind: bug
GitHub: 660
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:189`

### Frozen archive evidence

> | [#660](https://github.com/mudler/vllm.cpp/issues/660) | — | `check-device-leakage`'s `kcuda` bucket is the token grep `\bkCUDA\b`, so `minimax_h3_video.cpp:221-226`'s `static_cast<vt::DeviceType>(device)` hardcodes CUDA as enum value 1 and counts as 0. Gate strength plus an enum-ordering hazard; the H3 video lane should ask the same seam `ltx2_video.cpp` now does (found while reviewing #553) | bug |

## Resolution

-
