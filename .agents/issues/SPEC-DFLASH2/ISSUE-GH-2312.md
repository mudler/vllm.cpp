ID: ISSUE-GH-2312
Title: **`check-env-doc` was RED on `main`: `21ef6f053` (#2274 / #2304) landed `VT_DFLASH_BOUNDS_DEVICE` documented in its index row and its code comment but NOT in `docs/ENVIRONMENT.md`.** A BASE failure rather than a branch one — every branch cut after that commit inherits a red `scripts/agent-preflight.sh`, cannot reach a green gate before push, and the red is charged to whichever unrelated change runs the gate next; found exactly that way while gating [#2309](https://github.com/mudler/vllm.cpp/issues/2309). Documented beside the other `VT_DFLASH_*` entries as user-facing rather than allowlisted as kernel-internal, because the readback is a `Download` that SYNCHRONIZES on a path deliberately kept sync-free, so it changes timing as well as checking. Fixed in the same flow, as the in-flow rule requires
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 2312
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:896`

### Frozen archive evidence

> | [#2312](https://github.com/mudler/vllm.cpp/issues/2312) | `SPEC-DFLASH2` | **`check-env-doc` was RED on `main`: `21ef6f053` (#2274 / #2304) landed `VT_DFLASH_BOUNDS_DEVICE` documented in its index row and its code comment but NOT in `docs/ENVIRONMENT.md`.** A BASE failure rather than a branch one — every branch cut after that commit inherits a red `scripts/agent-preflight.sh`, cannot reach a green gate before push, and the red is charged to whichever unrelated change runs the gate next; found exactly that way while gating [#2309](https://github.com/mudler/vllm.cpp/issues/2309). Documented beside the other `VT_DFLASH_*` entries as user-facing rather than allowlisted as kernel-internal, because the readback is a `Download` that SYNCHRONIZES on a path deliberately kept sync-free, so it changes timing as well as checking. Fixed in the same flow, as the in-flow rule requires | bug |

## Resolution

-
