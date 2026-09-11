ID: ISSUE-GH-1479
Title: DFlash2 W3 tests use POSIX unistd APIs and cannot compile on native Windows
Row: ENG-RELEASE-WINDOWS
State: OPEN
Kind: UNKNOWN
GitHub: 1479
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-20
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> SPEC-DFLASH2 W3 adds two binding suites that unconditionally include POSIX-only unistd.h: 	ests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp:54 and 	ests/vllm/models/test_qwen3_dflash2_draft.cpp:46. MSVC fails with C1083 before either suite can run. This blocks the mandatory mutation that deletes the production selector call site and proves the reachability gate turns red.\n\nOwner: SPEC-DFLASH2 W3 repair flow. Replace the POSIX temporary-file handling with the repository's portable fixture pattern or a correct platform-specific adapter. Port both suites, prove the unchanged tests green on native Windows, and run the production-call-site deletion mutation.

## Resolution

-
