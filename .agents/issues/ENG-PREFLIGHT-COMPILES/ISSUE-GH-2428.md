ID: ISSUE-GH-2428
Title: origin/main fails its own gate ratchet at its own tip: no session can reach a green preflight
Row: ENG-PREFLIGHT-COMPILES
State: OPEN
Kind: UNKNOWN
GitHub: 2428
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `ENG-PREFLIGHT-COMPILES`
>
> `origin/main` fails its own gate ratchet at its own tip. This is not a branch-local
> problem and it is reproducible from a pristine export:
>
> ```sh
> T=$(mktemp -d); git archive origin/main | tar -x -C "$T"
> cd "$T" && python3 scripts/check-gate-commands.py --check; echo "rc=$?"
> ```
>
> ```text
> ERROR: these baseline rows left the gated population: ENG-PREFLIGHT-COMPILES.
> If that is a legitimate record edit, re-pin RUNNABLE_BASELINE in the SAME change,
> naming each row and the reason.
> rc=1
> ```
>
> `scripts/agent-preflight.sh` runs exactly this invocation (`agent-preflight.sh:371`,
> the `claim-view|check-gate-commands` arm passes `--check`), so **every session that
> runs preflight on any branch inherits this red and cannot reach a green preflight**,
> whatever its own change does. `tests/scripts/test_check_gate_commands.py` fails for
> the same underlying reason, because its ratchet test reads the live tree:
>
> ```text
> FAIL: test_adding_a_row_that_is_not_runnable_also_breaks_the_pin
> AssertionError: Items in the first set but not the second: 'ENG-PREFLIGHT-COMPILES'
> ```
>
> `ENG-PREFLIGHT-COMPILES` is present in `RUNNABLE_BASELINE`
> (`scripts/check-gate-commands.py:466-473`, added 2026-08-31 with a "GROWTH, and
> earned" note), so the pin was written; what the checker reports is that the row no
> longer RESOLVES into the gated population it was pinned against. One of the two
> moved after the other was written.
>
> **Not fixed here on purpose.** I found this while landing `VT-CPU-ELEM-SURVEY`
> (#2416), whose branch reproduces it only because it merged `origin/main`. Re-pinning
> a baseline I do not own, to make my own preflight green, is precisely the "never
> weaken a checker to make a transition pass" failure the same output warns about, and
> the checker's own message asks for the re-pin "in the SAME change" — which is the
> change that moved the row, not mine. The row that owns the pin owns the repair, and
> it should say which side moved rather than re-pinning to whatever the tree now says.
>
> Found while working #2416; that branch records the same finding in its merge commit
> message rather than acting on it.
>

## Resolution

-
