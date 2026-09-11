ID: ISSUE-GH-2429
Title: test_check_gate_commands is RED on main: ENG-PREFLIGHT-COMPILES became runnable without re-pinning RUNNABLE_BASELINE
Row: ENG-PREFLIGHT-COMPILES
State: CLOSED
Kind: UNKNOWN
GitHub: 2429
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> `tests/scripts/test_check_gate_commands.py` fails **12 of 59** on a clean `origin/main` (`a87f2cbe4`), verified in a throwaway worktree with no local changes:
>
> ```
> FAIL: test_adding_a_row_that_is_not_runnable_also_breaks_the_pin
>   AssertionError: Items in the first set but not the second: 'ENG-PREFLIGHT-COMPILES'
> FAIL: test_check_mode_passes_on_the_shipped_record
> FAIL: test_check_mode_is_never_silently_swallowed_by_json
> FAIL: test_dropping_cudagraph_break_from_the_pin_breaks_it
> ... and 8 more
> ```
>
> `ENG-PREFLIGHT-COMPILES` now parses as **runnable** — it carries a Gates command that can fail — but it is not in `RUNNABLE_BASELINE` (`scripts/check-gate-commands.py:465`).
>
> ## Why that is a defect and not just a stale number
>
> The checker's own comment states the contract: *"ANY movement, up or down, re-pins `RUNNABLE_BASELINE` in the SAME change."* Adding the row and re-pinning the set are one semantic change; landing the first without the second leaves the suite red for everyone else.
>
> Introduced around `45b808a5c` ("test(ENG-PREFLIGHT-COMPILES): parse first Gates section"), which is precisely the edit that moves a row into the runnable population — the hazard this repository has already recorded, where a Gates-section edit silently changes what the ratchet counts.
>
> ## The mechanism, which is now the fifth instance today
>
> `scripts/agent-preflight.sh` runs this suite, and a branch runs preflight before it pushes. Nothing holds `main` to it. So the breakage lands green and is charged to whichever unrelated branch runs the gate next — this one was found by a `--fit` branch that touches none of it.
>
> Same shape as [#2312](https://github.com/mudler/vllm.cpp/issues/2312), [#2356](https://github.com/mudler/vllm.cpp/issues/2356) (`check-env-doc`, twice), [#2404](https://github.com/mudler/vllm.cpp/issues/2404) (`windows-msvc` compile break) and the `api_server` crash behind it. Root cause tracked as [#2389](https://github.com/mudler/vllm.cpp/issues/2389); this is more evidence that a gate which does not run on `main` converts a landed defect into a red on innocent branches.
>
> ## Fix
>
> Re-pin `RUNNABLE_BASELINE` to include `ENG-PREFLIGHT-COMPILES`, as the same change should have. NOT done here: the owning row should re-pin it, and re-pinning a ratchet from an unrelated branch is how a baseline drifts without anyone reviewing what moved.
>
> Row: `ENG-PREFLIGHT-COMPILES`

## Resolution

Commit `a6d2e25de662c3e327407e94665316f9da2d7dd1` already includes `ENG-PREFLIGHT-COMPILES` in the runnable baseline; the current tree falsifies the reported missing pin, and GitHub closed the issue on 2026-08-31.
