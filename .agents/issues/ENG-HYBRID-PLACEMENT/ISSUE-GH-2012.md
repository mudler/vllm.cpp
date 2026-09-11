ID: ISSUE-GH-2012
Title: `docs/FEATURES.md` compares eleven memory axes against llama.cpp and carries no row for hybrid CPU/GPU expert placement, so `-cmoe`/`-ncmoe` — a capability this engine does not have, owned by the `READY` row `ENG-HYBRID-PLACEMENT` and requested in #149 — is invisible in the comparison. The nearest row, routed-expert streaming from disk, records llama.cpp as `mmap only`, which is correct for that row and is why the gap hides: streaming moves weights toward the compute, placement moves compute toward the weights
Row: ENG-HYBRID-PLACEMENT
State: UNKNOWN
Kind: record
GitHub: 2012
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:767`

### Frozen archive evidence

> | [#2012](https://github.com/mudler/vllm.cpp/issues/2012) | `ENG-HYBRID-PLACEMENT` | `docs/FEATURES.md` compares eleven memory axes against llama.cpp and carries no row for hybrid CPU/GPU expert placement, so `-cmoe`/`-ncmoe` — a capability this engine does not have, owned by the `READY` row `ENG-HYBRID-PLACEMENT` and requested in #149 — is invisible in the comparison. The nearest row, routed-expert streaming from disk, records llama.cpp as `mmap only`, which is correct for that row and is why the gap hides: streaming moves weights toward the compute, placement moves compute toward the weights | record |

## Resolution

-
