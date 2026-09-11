ID: ISSUE-GH-1756
Title: docs: repair benchmark links after the public-doc split
Row: ENG-PUBLIC-DOC-PROJECTIONS
State: CLOSED
Kind: bug
GitHub: 1756
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-23
Updated: 2026-08-28
Closed: 2026-08-28

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> The public-document split in #1714 retired `docs/STATUS.md` and moved benchmark details into `docs/benchmarks/`. Three user-facing references still use the old shape:
>
> - `README.md` links benchmark claims to the internal `.agents/benchmark-record.md` instead of the public detail pages.
> - `docs/SPECULATIVE-DECODING.md` calls the compact benchmark index a benchmark record instead of linking its detail page.
> - `docs/benchmarks/llama-cpp-cpu.md` names the deleted `STATUS.md`.
>
> Repair these references without changing any benchmark result, support claim, or README news headline. Verify with the README, benchmark-index, supported-model, and documentation-site checks. No GPU work applies.

## Resolution

GitHub records closing pull request #1757 (https://github.com/mudler/vllm.cpp/pull/1757) merged on 2026-08-28 as commit `a43c8c306ef8b81c0e753b08c99aa8a906df2162`. GitHub closed issue #1756 on 2026-08-28.
