ID: ISSUE-GH-2050
Title: Laguna's FFN is host-orchestrated token-at-a-time — per-token host rows, the router on the host through `MatmulNK`, and a host scalar combine loop — so a device-shaped MoE entry wrapping those loops would put it in the placement seam's wired list while moving nothing and adding a round trip: supported to read, a regression to measure. The real repair is a device-resident batched FFN, which is a model rework with a performance gate
Row: ENG-HYBRID-PLACEMENT
State: UNKNOWN
Kind: gap
GitHub: 2050
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:781`

### Frozen archive evidence

> | [#2050](https://github.com/mudler/vllm.cpp/issues/2050) | `ENG-HYBRID-PLACEMENT` | Laguna's FFN is host-orchestrated token-at-a-time — per-token host rows, the router on the host through `MatmulNK`, and a host scalar combine loop — so a device-shaped MoE entry wrapping those loops would put it in the placement seam's wired list while moving nothing and adding a round trip: supported to read, a regression to measure. The real repair is a device-resident batched FFN, which is a model rework with a performance gate | gap |

## Resolution

-
