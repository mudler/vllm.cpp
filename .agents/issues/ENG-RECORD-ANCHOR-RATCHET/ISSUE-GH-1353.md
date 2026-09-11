ID: ISSUE-GH-1353
Title: A full disk makes `scripts/agent-preflight.sh` report record and policy defects that do not exist. Measured at `63ff58272` with 896M free of 447G: ten suites go red together — `test_check_release_binary_contract`, `test_release_postpublish_audit`, `test_check_container_matrix`, `test_release_index`, `test_release_metadata`, `test_release_accelerator_metadata`, `test_release_macos_metadata`, `test_release_windows_metadata`, `test_agent_role`, `test_agent_onboard` — and the trailing summary names all ten without mentioning the disk. The cause appears only inside one suite's captured output several screens up (`error: copy-fd: write returned: No space left on device`), interleaved with fixture failures that read exactly like findings (`ERROR: x: see (#157) and #174`, `ERROR: x: landed without a row PR`). All ten build a scratch git repository in a temporary directory, so all ten fail together and all ten fail toward a verdict about RECORDS rather than about the environment — the `.agents/verification.md` broken-instrument shape, and an agent reading it has every reason to start repairing records that were never broken. Reclaiming 2.7G of one row's own build tree and re-running the identical command turned all ten green with no tree change (`86 gates, 86 ok, 0 FAIL, 0 SKIP`, exit 0). The fix is a precondition, not a suppression: check free space before the suites that need a scratch repository and refuse naming the disk, exactly as `test_cpu_x86_llamacpp_floor` refuses to measure under contention ([#618](https://github.com/mudler/vllm.cpp/issues/618)). A gate that cannot run must say so rather than return a verdict. Found while landing [#1332](https://github.com/mudler/vllm.cpp/issues/1332) M0+M1 and NOT fixed in that flow, because it changes preflight semantics and adds a refusal path, which `AGENTS.md` routes to the normal row, spec and fresh-review path
Row: ENG-RECORD-ANCHOR-RATCHET
State: UNKNOWN
Kind: bug
GitHub: 1353
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:461`

### Frozen archive evidence

> | [#1353](https://github.com/mudler/vllm.cpp/issues/1353) | `ENG-RECORD-ANCHOR-RATCHET` | A full disk makes `scripts/agent-preflight.sh` report record and policy defects that do not exist. Measured at `63ff58272` with 896M free of 447G: ten suites go red together — `test_check_release_binary_contract`, `test_release_postpublish_audit`, `test_check_container_matrix`, `test_release_index`, `test_release_metadata`, `test_release_accelerator_metadata`, `test_release_macos_metadata`, `test_release_windows_metadata`, `test_agent_role`, `test_agent_onboard` — and the trailing summary names all ten without mentioning the disk. The cause appears only inside one suite's captured output several screens up (`error: copy-fd: write returned: No space left on device`), interleaved with fixture failures that read exactly like findings (`ERROR: x: see (#157) and #174`, `ERROR: x: landed without a row PR`). All ten build a scratch git repository in a temporary directory, so all ten fail together and all ten fail toward a verdict about RECORDS rather than about the environment — the `.agents/verification.md` broken-instrument shape, and an agent reading it has every reason to start repairing records that were never broken. Reclaiming 2.7G of one row's own build tree and re-running the identical command turned all ten green with no tree change (`86 gates, 86 ok, 0 FAIL, 0 SKIP`, exit 0). The fix is a precondition, not a suppression: check free space before the suites that need a scratch repository and refuse naming the disk, exactly as `test_cpu_x86_llamacpp_floor` refuses to measure under contention ([#618](https://github.com/mudler/vllm.cpp/issues/618)). A gate that cannot run must say so rather than return a verdict. Found while landing [#1332](https://github.com/mudler/vllm.cpp/issues/1332) M0+M1 and NOT fixed in that flow, because it changes preflight semantics and adds a refusal path, which `AGENTS.md` routes to the normal row, spec and fresh-review path | bug |

## Resolution

-
