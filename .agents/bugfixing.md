# Task guide — fixing a bug

The rules are in [`AGENTS.md`](../AGENTS.md); this is the method.

## Open the issue first

Every fix has a GitHub issue, appended to the issue index and linked from the
PR. The index row names an owning row ID, or names a spec that lists the issue
under `## Owed`.
If you found the bug while doing something else, open its own issue rather than
folding a silent fix into an unrelated change.

## Ground the premise before fixing

Verify the claimed cause in the source or in a measurement before writing any
fix. A plausible story about what is wrong is not a diagnosis, and a fix built
on a wrong premise passes review by looking reasonable.

Check the row's spec and `git log -S'<symbol>'` first. Closed root-causes get
re-derived here more often than they get re-found.

Before believing "this isn't my change", reproduce the failure on a clean build
of current `main`. A degraded or partial build drifts gates in ways that look
exactly like someone else's bug.

When a process dies without an obvious cause, check whether something else
killed it before assuming a leak — an exit code with no matching OOM record
usually means an external signal.

## Red test first

Write the smallest test that fails for the intended reason and capture the red
output. Then make the minimum complete change. A fix without a test that failed
before it is not a fix, it is a hope.

Port the upstream test if vLLM covers this behavior.

## Fix the cause, not the gate

You may never turn a red gate green by deleting an assertion, widening a scope,
or adding a path exemption. If the gate is genuinely wrong, that is its own
change with its own spec, red-before evidence, and green-after evidence.

Anchor edits by uniqueness: assert exactly one match before replacing, because
near-duplicate functions are common here and a positional replace will silently
hit the wrong one.

## Before you call it fixed

Run the focused gate, then the full gate. Hand the immutable head to a fresh
reviewer for static and mutation review. The operator reruns the gate.

Record what the bug actually was in the row's spec `## Outcome` — especially if
the obvious explanation turned out to be wrong. That is the part neither the
code nor git will tell the next person.
