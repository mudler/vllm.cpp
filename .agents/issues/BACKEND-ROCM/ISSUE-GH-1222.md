ID: ISSUE-GH-1222
Title: Qwen3.5 M4 gate reports success when required goldens are missing
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: 1222
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-18
Updated: 2026-08-18
Closed: 2026-08-18

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Bug
>
> `tests/parity/test_qwen35_paged_engine.cpp` promises that an unavailable gate exits 77 rather than reporting success, and CTest configures `SKIP_RETURN_CODE 77`. Three missing-artifact paths violate that contract:
>
> - missing `greedy_ids.npy` returns normally;
> - missing `our_ids.npy` returns normally;
> - missing `neartie_gap_mnats.npy` returns normally.
>
> A normal return lets doctest and CTest report success, producing a false-green correctness gate. The immutable PR #559 head contains every artifact and the real ROCm gate passes, so this appears only when a committed golden is absent or a checkout is incomplete.
>
> ## Required repair
>
> - Route every unavailable prerequisite through the existing exit-77 skip behavior.
> - Add a deterministic test that removes or redirects each required artifact and proves process exit 77, without requiring GPU hardware.
> - Preserve hard failure for malformed or inconsistent artifacts; only unavailable prerequisites skip.
> - Record RED, GREEN, and mutation evidence in the owning M4 spec.
>
> ## Ownership
>
> Owned and fixed in-flow by `BACKEND-ROCM`, PR #559, during the final current-main review.

## Resolution

GitHub records closing pull request #559 (https://github.com/mudler/vllm.cpp/pull/559) merged on 2026-08-18 as commit `7b89cf30775df79155d4bcc4f3cabf57637f74fc`. GitHub closed issue #1222 on 2026-08-18.
