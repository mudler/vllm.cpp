# GATE-PREPUSH-FAIL-LOUD — the pre-push hook refuses a checker it cannot find

Issue: [#1779](https://github.com/mudler/vllm.cpp/issues/1779).

## Scope

`.githooks/pre-push` names its checkers in a `CHECKERS` array and guards each
one with `[ -f "$work/scripts/$checker" ] || continue`. Three of the six names it
carried had no file:

| Name | Deleted by |
|---|---|
| `check-policy.py` | `0f3e44eee`, with `policy.csv` and `policy_contract.py` |
| `check-state-record.py` | `0f3e44eee`, with `state.md` and `state_record.py` |
| `check-public-doc-tables.py` | `1db7e59cf` (#1714), with the shared status ledger |

Each of the three is **deleted, not renamed**. `0f3e44eee` retired the CSV
policy registry and the state log outright, and `1db7e59cf` retired the
documentation projection gates; the `check-benchmark-index.py` it added in the
same commit is a new 73-line index-shape checker, not the 582-line table gate,
and Git pairs the two as an add and a delete rather than a rename at any
similarity threshold. So pruning the names records reality; it does not drop a
live gate.

The guard is what makes this invisible. A named checker with no file was skipped
without a word and the hook still exited 0, so it presented as six gates while
running three. `core.hooksPath` is set to `.githooks` in this checkout, so the
hook is installed and runs on every push.

The harm is not the three deliberate removals. It is that the hook could not
tell a deliberate removal from an accidental one, and would swallow the next
checker to vanish exactly the same way. AGENTS.md holds that hooks are
bypassable convenience and not evidence, so CI remains the gate of record; what
this row fixes is an instrument that reports green over a measurement it did not
take.

## Design

Both halves land together, because either one alone re-creates the defect:

1. **Fail loud.** The `|| continue` becomes an `if [ ! -f ... ]` arm that prints
   the missing name and sets `failed=1`. Pruning alone would restore the silence
   at the next deletion.
2. **Prune the three dead names.** Failing alone would red every push in the
   tree as it stands, and a gate that fires on ordinary work gets bypassed
   rather than obeyed.

The `case` arm supplying `--base` to `check-state-record.py` goes with the name;
leaving it would be a branch no value can reach.

`.githooks/README.md` listed `check-public-doc-tables.py` and described budgets
on a `docs/STATUS.md` that no longer exists. It is the same claim in prose and
is corrected in the same change, as is the hook's own header paragraph.

## Tests

`tests/scripts/test_prepush_checker_names.py`, registered in
`scripts/agent-preflight.sh`'s `SUITES` array and in the record lane of
`.github/workflows/ci.yml`.

It **executes** the hook against a scratch git repository it builds, rather than
matching its text: what is under test is whether the loop body reaches
`failed=1`, and a text match is satisfied by a `continue` that merely moved. The
scratch repository carries real commits because the hook materialises the pushed
commit with `git worktree add` and looks for the checker inside that tree.

Four execution cases and three tree cases:

- a named checker with no file fails the push, and the message names it;
- two missing names are both reported, not only counted;
- a wholly present, wholly passing list still pushes (the other direction: a
  gate that reds every push gets turned off);
- a present checker that exits non-zero still fails the push, so the behaviour
  that already worked is not traded for the new one;
- every name in the shipped array exists in `scripts/`;
- the shipped array is a subset of `scripts/agent-preflight.sh`'s, which is what
  the hook's own comment instructs and what nothing previously read;
- every `scripts/...` path named in `.githooks/README.md` exists.

## Risks

A commit that deletes a checker without pruning the array now fails its own
push. That is the intended cost and the reason the two halves ship together.
The hook reads the array from the **pushed commit's** tree, so a push whose tip
prunes the name is consistent; only a push of a tip that deletes a checker and
leaves the name behind reds, which is the case this row exists to surface.

## Gates

- `python3 tests/scripts/test_prepush_checker_names.py`
- `scripts/agent-preflight.sh`
- `bash -n .githooks/pre-push`

## Evidence

Red first, against the unmodified hook: 7 of 10 cases failed, with
`test_a_named_checker_with_no_file_fails_the_push` reporting
`0 == 0 : the hook accepted a push while a checker it names does not exist:
stdout='' stderr=''` — rc 0 and not one byte of output.

Green after the change: 10 of 10.

Mutation: restoring `[ -f "$work/scripts/$checker" ] || continue` in the fixed
tree reds the two execution cases again and leaves the tree-shape cases green,
which is the correct split — the pruned array satisfies those cases whatever the
loop does, and only an executed hook can distinguish the loop bodies. The tree
was restored byte-for-byte, verified by `sha256sum` and by re-running the suite.

## Owed

- [#1779](https://github.com/mudler/vllm.cpp/issues/1779) **part 2 is not fixed
  here**: 65 specs under `.agents/specs/` still instruct an agent to run
  `check-doc-checkpoint.py` or `check-public-doc-tables.py`, 16 of them inside a
  `## Tests and gates` or `## Gates` section, and `scripts/check-pr-size.py`,
  `scripts/check-conflict-markers.py` and `scripts/agent-preflight.sh` each
  carry a stale reference to one of the two. That is roughly 99 references.
  AGENTS.md routes a change of that size and surprise through its own row, spec
  and fresh review rather than an in-flow sweep, so it needs a row ID before it
  is claimed. The issue stays open when this row lands.

## Stop conditions

Stop and report rather than widening scope if the fail-loud arm reds a push for
a name that turns out to have been renamed rather than deleted: the repair is
then to update the name, not to drop the gate.

## Now

Landed: the hook fails loudly on a named-but-absent checker, the three dead
names are pruned, and the suite that pins both is registered on the preflight
and CI lanes. Part 2 of #1779 remains owed above and unclaimed.
