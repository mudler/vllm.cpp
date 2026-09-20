ID: ISSUE-GH-3110
Title: test(BACKEND-GATE-ROCM-SGLANG): publish lifecycle diagnostic aggregate once
Row: BACKEND-GATE-ROCM-SGLANG
State: OPEN
Kind: UNKNOWN
GitHub: 3110
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-GATE-ROCM-SGLANG`
>
> Owner: current Strix campaign operator. Parent: #3108.
>
> Strix diagnostic e6dba009-a81b-4e0f-a74e-b6f39cb3b43d completed vLLM startup, generation and normal owned-child teardown. Engine evidence is PASS, but the scratch validate-hardware-194a.py driver writes run/result.json in its per-engine finally block and then publishes the terminal aggregate to the same immutable path. audit.publish_result correctly refuses replacement with EEXIST, leaving the initial aggregate status FAIL. The job is not an overall PASS.
>
> Repair the diagnostic driver, not the immutable publication contract: publish per-engine records once and the terminal aggregate once; retain failure evidence and stop after a failed engine. Add a focused regression that executes orchestration with fake engine operations and real no-replace publication, covering success and failure. Do not weaken engine binding checks, teardown assertions, pins, or acceptance rules. Re-run bounded Strix validation after independent review. The source lifecycle change and compliant replay are owned by #3108; this issue owns the diagnostic publication repair.

## Resolution

-
