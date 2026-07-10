# Test-porting protocol — port vLLM's TESTS with its code (user-directed 2026-07-10)

**Principle: upstream tests are the executable spec.** vLLM's `tests/` tree
(`/home/mudler/_git/vllm/tests/` @ pin `e24d1b24`) encodes what every feature
is REQUIRED to do — edge cases included — better than any prose. Porting code
without its tests loses the ground truth and leaves regressions undetectable.
So test porting is part of the mirror obligation, not an optional extra.

## Rules

1. **Every feature/subsystem port carries its upstream tests in the SAME
   change.** Find the matching module(s) under upstream `tests/` (e.g.
   `tests/v1/core/test_scheduler.py`, `tests/entrypoints/openai/…`,
   `tests/kernels/…`) and port the assertions to our suite. The C++ test file
   header cites the upstream test file(s) ported FROM (file:line for
   non-obvious cases) — same citation discipline as
   [ground-every-impl-in-upstream] for kernels.

2. **Three tiers, all mirrored from upstream where they exist:**
   - **T-unit** — behavioral unit tests: upstream assert semantics re-expressed
     in doctest under `tests/vllm/…` (this is what M0–M3 already did for
     `tests/v1/core/` scheduler/KV semantics — that practice is now protocol).
   - **T-parity** — golden/oracle tests: token-exact vs the pinned vLLM oracle,
     kernel-vs-golden dumps (`tools/parity/`). Required for every kernel and
     numeric path (gates.md).
   - **T-e2e** — server/API conformance (`tests/vllm/entrypoints/…`,
     `test_conformance.cpp`), mirroring upstream `tests/entrypoints/`.
3. **Traceability is 1:1 and named**: keep our test case names derived from
   upstream ones (`test_schedule_spec_decode` → `TEST_CASE("schedule_spec_decode …")`)
   and record the upstream→ours test-file mapping in
   [porting-inventory.md](porting-inventory.md) alongside the code mapping, so
   the upstream sync cycle can diff TEST deltas mechanically.
4. **Upstream sync ports test deltas too** ([upstream-sync.md](upstream-sync.md)):
   when a synced vLLM PR touches `tests/`, the port includes the test delta.
   A PR that only changes tests still gets synced — it's a spec change.
5. **Specs must inventory their tests**: every `.agents/specs/<slug>.md`
   (roadmap_v1.md §E convention) includes a **"Tests to port"** section listing
   the upstream test files/cases that cover the feature and the tier each maps
   to. A spec without it is incomplete; the implementing agent's DoD includes
   those tests green.
6. **Can't-pass-yet ≠ delete**: a ported test blocked on an unimplemented
   dependency is checked in SKIPPED with a one-line tracked reason (and shows
   up as debt), never silently dropped or watered down.
7. **Regression floor**: all ported tests run in CI (CPU tiers) or the
   `SERVE-E2E-NIGHTLY` DGX suite (GPU/model tiers). If vLLM asserts a behavior, our
   port is not DONE until an equivalent assertion runs green in our suite —
   the test-side twin of the mirror policy's "match on every axis".

## Why (recorded reasoning)

- Ground-ruling: our own tests can only assert what WE believed the behavior
  was; upstream's tests assert what vLLM actually guarantees, catching the
  believed-vs-actual gap at port time instead of in production.
- Regressions: the MVP perf campaign showed correctness held only because
  token-exact gates ran on every change; feature breadth (tools, MM, TP, spec
  decode) needs the same discipline, and upstream already wrote those suites.
