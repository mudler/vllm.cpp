# Task guide — proving a change is reached

How to show that something reaches the code you landed. The rule is
[`AGENTS.md`](../AGENTS.md) `## Nothing lands dead`; this is the method. Nothing
here is binding on its own, and nothing here may weaken the rule there.

## The case this exists for

Tensor parallelism landed with a rank-layout table, a green focused gate, and a
`tp` handle that reached nothing. The audit states it precisely
([`specs/tensor-parallelism-spike.md:96-101`](specs/tensor-parallelism-spike.md)):
the handle threaded from `LayerForward` down through the Qwen3-dense forward at
`qwen3.cpp:92,114,126,136` and **dead-ended there**. No caller above the layer
passed anything but the default `nullptr`. No production loader passed `tp` into
`LoadMergedBf16RawNK` — every call site omits it. "The ONLY tp>1 driver in the
tree is the toy MLP test."

Read what that cost. The feature was real, the test was real, the gate was
honest, and no user could arrive at any of it. Nothing in the protocol asked the
question, so nobody answered it.

The seam checkers do not catch this and were never meant to.
`check-fusion-consistency.py`, `check-runner-routing-consistency.py`, and
`check-surface-coverage.py` all ask **where** a capability routes. A capability
that routes through the correct seam and that nothing calls passes all three.

## The shapes

Dead is one defect wearing different clothes. Each of these has landed here or
in a tree like it:

| Shape | What it looks like |
|---|---|
| The unpassed parameter | A function grows an argument. Every call site takes the default. |
| The unselected branch | A code path keyed on a config field that no released checkpoint sets. |
| The flag with no default path | A capability behind an option that nothing turns on. |
| The private surface | A capability reachable only by an example that includes an internal header. `check-surface-coverage.py` polices this one; it is the same defect. |
| The test-only driver | The only thing that ever calls the new code is the test written for it. |

The last one is the hardest to see, because the gate is green and the coverage
is real. Coverage measures whether a test reached the code. It never measures
whether anything else did.

## Proving it

Answer both questions. They are not the same question.

**Does a production entry point reach this?** Name the entry point and the call
site. A production entry point is one a user arrives through: `include/vllm.h`,
the loader, `ModelRegistry::Forward`, or a registered server or command-line path
on its default configuration. Follow the chain by hand, from the entry point down
to the change. An intermediate hop that is itself unreached does not carry.

**Does a test enter through it?** The smallest test that fails for the intended
reason starts at that entry point rather than at the type. Keep the unit test as
well — it localizes a failure, and that is worth having. It is not the proof.

`git grep` for the symbol answers neither question on its own. A call site inside
a test, an example, or another unreached function is not reach. Read what the
matches are.

## The reachability mutation

The mutation pass already breaks each claimed guarantee. Add one more mutation,
and run it the same way: in a scratch copy, never in the reviewed worktree.

1. Delete the production call site — the line where the entry-point chain
   reaches the change. Not the implementation, the call.
2. Rerun the focused gate.
3. A red gate proves the test enters through the production path. Record it as
   the reachability evidence.
4. A green gate is the finding. The change is unreached, and its gate measures a
   class rather than a capability. Report it with the severity that any
   unreached change carries.
5. Restore the tree byte-for-byte, as with every other mutation.

A change that has no production call site to delete has already answered the
question.

## Landing a slice that is not reached yet

Staged work is legitimate. Landing a layer before the wave that wires it is a
normal shape here, and the rule does not forbid it. It forbids doing it
silently.

Put the argument in the commit body and the pull request body. Name three
things:

- what is not reached yet, in the specific — the parameter, the branch, the
  entry point that does not exist yet;
- the row ID that owns the wiring;
- the issue that tracks it.

Then list it under `## Owed` in the row's spec. This is the ordinary exception
mechanism of this protocol, and it works the way every other one does: the reason
stays with the diff, the author, and the date, and `git log --grep` finds it.
There is no registry, because a registry is a state log and this protocol has
none.

A wave that lands its layer without naming the wave that wires it has not made a
staging decision. It has left dead code, and the next agent has no way to tell
the two apart.

## Why no checker enforces this

A checker that could separate a live call site from a dead one would have to
resolve overloads, templates, registration tables, and configuration-keyed
dispatch across the tree. That is a compiler front end, and a wrong answer from
it is worse than no answer, because a green check reads as proof.

A floor coarse enough to be writable — "the symbol appears outside its own test"
— passes exactly the cases that hurt. The `tp` handle appeared in four
production files. It was still dead.

So reachability is a review decision, like change size. This is recorded here so
that a later reader treats it as a decision with a reason rather than as a gap
somebody forgot to close.
