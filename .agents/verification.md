# Task guide — gates, evidence, and review

How to prove something, and how to review someone else's proof. The rules are in
[`AGENTS.md`](../AGENTS.md); this is the method.

## Running a gate

Start with the smallest deterministic test that can falsify the spec. Preserve
the red result — a test that was never seen failing has proven nothing. Make it
green, run the declared focused gate, then run the full preflight.

A gate report records the immutable SHA, the exact command, the environment,
the exit status, and the evidence path. Not a summary of them.

Two traps that have produced false greens here:

- **Incremental builds mask `-Werror`.** Clean-rebuild after any header change;
  an incremental green is not a clean green.
- **A copied build directory rebuilds the original sources.** CMake caches
  absolute source paths, so `cp -a` of a build tree can produce a
  byte-identical binary from unmodified code. Build in place and verify source
  and binary checksums.

Release builds define `NDEBUG`, so `assert` is compiled out. A green Release
gate over an assert-firing bug is a latent failure, not a pass — check the build
type before believing a surprising green.

Tests that starve under `ctest -j` are re-run serially before being called a
regression.

## Is `main` green? — the baseline lane

Before spending a build cycle proving a red check is not yours, ask:

```sh
scripts/main-baseline.py            # last fully green SHA, and what is failing now
scripts/main-baseline.py --json
```

It reads the `schedule`/`workflow_dispatch` runs of `.github/workflows/ci.yml` —
the only lane whose long jobs are not cancelled by the next push — and derives
the verdict at read time. Nothing is stored, so there is no file to conflict on
and no file that can be stale relative to the runs.

Three things to know before you trust or dismiss a red check.

- **A `push` run on `main` proves almost nothing.** Its expensive jobs share a
  ref-keyed concurrency group, so the next push cancels them. Of 40 consecutive
  runs measured for [#274](https://github.com/mudler/vllm.cpp/issues/274), 26
  were `cancelled` and exactly one completed.
- **A run's own conclusion is not the verdict.** `sanitize-cpu` is
  `continue-on-error`, so a run reports `success` with the sanitizers red — run
  `31448896841` at `5812b8b6` is exactly that. The tool reads per-job
  conclusions and so should you.
- **Staleness is visible, not silent.** Every line carries the run's date. If the
  newest baseline is old, say so; never read an absent run as a pass, and never
  read `REMOTE_UNVERIFIED` as one either.

To pin a baseline on a SHA you care about right now, rather than waiting for the
4-hourly cron: `gh workflow run ci.yml --ref main`.

## Reviewing

Review happens only after the implementation's own gates pass, and only on an
immutable head, and never by the agent that wrote the code.

**Static pass:** the spec, the diff, the tests, the error paths, ownership
boundaries, and whether the claims are actually supported.

**Mutation pass:** for each critical guard, temporarily remove or corrupt it in
a *scratch copy* and prove the focused test fails. Mutate, don't just read — a
test that passes with the guard deleted was testing nothing. Restore the tree
byte-for-byte after every mutation, and never mutate the reviewed worktree.

Report `PASS` only after both passes on the same head. Every finding carries
severity, the violated requirement, a reproduction, and the expected behavior.

Do not take another agent's report at face value; the operator reruns the gate
regardless of how confident the report sounded.

## Evidence

Separate what you observed from what you inferred. Name source roots, versions,
`file:line` anchors, commands, artifacts, and limitations.

A negative result is a result: record refuted hypotheses and failed attempts,
including the regime they were measured in. "Not established" usually means
"not resolvable against the current noise floor" — the same code can read
differently once the bottleneck moves, so a discarded lever is worth re-testing
after the surrounding performance picture changes.

Public documents carry only the keyed current projection. Forensic detail stays
in the row's spec and the append-only records.
