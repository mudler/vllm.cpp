ID: ISSUE-GH-1690
Title: **`check-doc-checkpoint.py`'s `LANDING_SOURCE_FILES` omitted `include/vllm.h`, so a commit that bumps `VLLM_ABI_VERSION` could never repair the README claim it invalidated.** Found by the review of [PR #1655](https://github.com/mudler/vllm.cpp/pull/1655). The README `## Use it as a library (C API)` block quotes the ABI version out of the header; the header was in `USER_USAGE_FILES` (so an ABI change owes `docs/USAGE.md`) but not in `LANDING_SOURCE_FILES`, and the README rule refuses a claim change unaccompanied by a landing source. The claim could therefore be invalidated but not repaired by the same edit, which is how the README reached `VLLM_ABI_VERSION 21` against a header reading `23`, alongside a stale "46 exported functions" for a header declaring 47. That second half is stale by one rather than by six -- an earlier review figure of 51/52 swept in the `#define VLLM_API` visibility block and counted `vllm_*` identifiers that are typedefs and struct fields rather than exported functions -- so the case for deleting the count is that a live count of one file stored in another goes stale on any ABI addition, not that it is badly wrong. The set's own criterion already admitted it -- the checker's comment says every member is "something the README QUOTES" -- and the header was the only such source missing. FIXED IN FLOW: `include/vllm.h` is added to the set, red-before/green-after pinned by `test_the_c_abi_header_is_a_landing_source`, with `test_the_c_abi_header_permits_but_does_not_demand_readme` proving no new README obligation and the pre-existing no-class tests still green
Row: ENG-DOCS-SITE
State: UNKNOWN
Kind: bug
GitHub: 1690
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:631`

### Frozen archive evidence

> | [#1690](https://github.com/mudler/vllm.cpp/issues/1690) | `DOC-README-ABI-LANDING-SOURCE` | **`check-doc-checkpoint.py`'s `LANDING_SOURCE_FILES` omitted `include/vllm.h`, so a commit that bumps `VLLM_ABI_VERSION` could never repair the README claim it invalidated.** Found by the review of [PR #1655](https://github.com/mudler/vllm.cpp/pull/1655). The README `## Use it as a library (C API)` block quotes the ABI version out of the header; the header was in `USER_USAGE_FILES` (so an ABI change owes `docs/USAGE.md`) but not in `LANDING_SOURCE_FILES`, and the README rule refuses a claim change unaccompanied by a landing source. The claim could therefore be invalidated but not repaired by the same edit, which is how the README reached `VLLM_ABI_VERSION 21` against a header reading `23`, alongside a stale "46 exported functions" for a header declaring 47. That second half is stale by one rather than by six -- an earlier review figure of 51/52 swept in the `#define VLLM_API` visibility block and counted `vllm_*` identifiers that are typedefs and struct fields rather than exported functions -- so the case for deleting the count is that a live count of one file stored in another goes stale on any ABI addition, not that it is badly wrong. The set's own criterion already admitted it -- the checker's comment says every member is "something the README QUOTES" -- and the header was the only such source missing. FIXED IN FLOW: `include/vllm.h` is added to the set, red-before/green-after pinned by `test_the_c_abi_header_is_a_landing_source`, with `test_the_c_abi_header_permits_but_does_not_demand_readme` proving no new README obligation and the pre-existing no-class tests still green | bug |

## Resolution

-
