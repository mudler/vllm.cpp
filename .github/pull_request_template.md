<!-- Helper PRs: branch is `row/<ROW-ID>`, opened as a DRAFT at the START.
     The draft PR IS the claim. See .agents/specs/operator-helper-protocol.md -->

## Row

`ROW-ID` — one row per PR.

## Before starting

<!-- Record what you found before implementation. Every change needs an open
     issue. -->

- Issue/PR search and existing claim:
- Pull request shape selected at row claim:
- Roadmap or matrix row, plus `scripts/ready-for-helper.py` result when applicable:
- Exact current-code and test/evidence anchors inspected:

## What changed

<!-- One paragraph. What it does, not how you felt about it. -->

## Evidence

<!-- What actually RAN, and what it proves. Paste the command and the result.
     "Gates green" without output is not evidence. -->

- [ ] `scripts/agent-preflight.sh` passes
- [ ] tests that cover this change (name them):
- [ ] same-change doc obligations (`docs/STATUS.md`, `docs/BENCHMARKS.md`, and
      `docs/FEATURES.md` if a feature/model/backend/quant surface moved)

## Speed claims

- [ ] This PR makes NO speed claim (helpers may not: the GPU is
      operator-serialized and a contended benchmark is void), **or**
- [ ] the operator ran the numbers under `${GPU_LOCK}` and they are recorded in
      `docs/BENCHMARKS.md` with the repro recipe.

## Honest gaps

<!-- What you did NOT do, what you could not verify, what you guessed.
     This section being empty is itself a claim. -->
